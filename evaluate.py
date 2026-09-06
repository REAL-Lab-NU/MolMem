"""Evaluate MolMem under the paper's fixed-budget multi-generation search."""

import json
import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path

from ragen.env.molecule_opt.evaluation import (
    LeadSearchState,
    aggregate_metrics,
    canonicalize_smiles,
)
from ragen.utils import register_resolvers


register_resolvers()


def _normalize_memory_path(config) -> None:
    configured_path = config.evolving_skill_memory.get("save_path", "")
    if configured_path:
        config.evolving_skill_memory.save_path = to_absolute_path(configured_path)


def _load_leads(state_manager, count: int) -> list[str]:
    leads = [canonicalize_smiles(smiles) for smiles in state_manager.molecule_data_map.values()]
    if len(leads) < count:
        raise ValueError(f"Evaluation requested {count} leads, but the dataset contains {len(leads)}")
    return leads[:count]


def _initial_search_states(state_manager, leads: list[str], task: str):
    task_properties = task.split("+")
    group_size = state_manager.group_size
    search_states = {}
    budget_states = {}

    for lead_id, lead_smiles in enumerate(leads):
        env = state_manager.envs[lead_id * group_size]["env"]
        properties = env.oracle.evaluate_specific_properties(lead_smiles, task_properties)
        properties = {prop: float(properties[prop]) for prop in task_properties}
        search_states[lead_id] = LeadSearchState(
            lead_id=lead_id,
            lead_smiles=lead_smiles,
            task_properties=task_properties,
            initial_properties=properties,
        )
        budget_states[lead_id] = {
            "limit": 0,
            "oracle_calls": 0,
            "candidate_evaluations": 0,
            "events": [],
        }

    return search_states, budget_states


def _enable_strict_oracles(state_manager) -> None:
    for entry in state_manager.envs:
        oracle = getattr(entry["env"], "oracle", None)
        if oracle is not None:
            oracle.raise_on_error = True


def _evaluation_inputs(search_states, budget_states, generation: int, budget: int):
    inputs = {}
    for lead_id, search_state in search_states.items():
        budget_state = budget_states[lead_id]
        budget_state["limit"] = budget
        inputs[lead_id] = {
            "lead_smiles": search_state.lead_smiles,
            "start_smiles": search_state.best_smiles,
            "lead_properties": search_state.initial_properties,
            "start_properties": search_state.best_properties,
            "budget_state": budget_state,
            "generation": generation,
        }
    return inputs


def _consume_new_events(search_states, budget_states, offsets) -> None:
    for lead_id, search_state in search_states.items():
        budget_state = budget_states[lead_id]
        events = budget_state["events"]
        for event in events[offsets[lead_id]:]:
            if int(event["oracle_call"]) > int(budget_state["limit"]):
                raise RuntimeError(f"Oracle budget exceeded for lead {lead_id}")
            search_state.consider(event, int(event["generation"]))
        offsets[lead_id] = len(events)


def _current_results(search_states, budget_states) -> list[dict]:
    return [
        search_states[lead_id].result(budget_states[lead_id]["oracle_calls"])
        for lead_id in sorted(search_states)
    ]


@hydra.main(version_base=None, config_path="config", config_name="molecule_opt_inference")
def main(config):
    _normalize_memory_path(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoTokenizer
    from verl import DataProto

    from ragen.llm_agent.agent_proxy import VllmWrapperWg
    from ragen.llm_agent.ctx_manager import ContextManager
    from ragen.llm_agent.es_manager import EnvStateManager

    evaluation = config.evaluation
    num_leads = int(config.test_size)
    generations = int(evaluation.generations)
    oracle_budget = int(evaluation.oracle_budget)
    base_temperature = float(evaluation.base_temperature)
    temperature_increment = float(evaluation.temperature_increment)
    max_temperature = float(evaluation.max_temperature)
    base_seed = int(evaluation.seed)

    if min(num_leads, generations, oracle_budget) <= 0:
        raise ValueError("test_size, generations, and oracle_budget must be positive")
    if int(evaluation.rollouts_per_generation) <= 0:
        raise ValueError("rollouts_per_generation must be positive")
    if int(config.agent_proxy.max_turn) <= 0:
        raise ValueError("agent_proxy.max_turn must be positive")
    if base_temperature <= 0 or temperature_increment < 0:
        raise ValueError("Temperatures must be positive and non-decreasing")
    if max_temperature < base_temperature:
        raise ValueError("max_temperature must be at least base_temperature")

    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor = VllmWrapperWg(config, tokenizer)
    context_manager = ContextManager(config, tokenizer, mode="val")
    state_manager = EnvStateManager(config, mode="val")

    try:
        _enable_strict_oracles(state_manager)
        if state_manager.env_groups != num_leads:
            raise ValueError("es_manager.val.env_groups must equal test_size")

        leads = _load_leads(state_manager, num_leads)
        search_states, budget_states = _initial_search_states(
            state_manager, leads, str(config.molecule_opt_task)
        )
        event_offsets = {lead_id: 0 for lead_id in search_states}
        generation_metrics = []

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        for generation in range(1, generations + 1):
            active_leads = sum(
                state["oracle_calls"] < oracle_budget for state in budget_states.values()
            )
            if active_leads == 0:
                break

            temperature = min(
                base_temperature + (generation - 1) * temperature_increment,
                max_temperature,
            )
            actor.sampling_params.temperature = temperature
            generation_seed = base_seed + generation * 1000
            meta_info = {
                "seed": generation_seed,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
                "generation": generation,
                "temperature": temperature,
            }

            env_outputs = state_manager.reset(
                seed=generation_seed,
                global_step=generation,
                total_training_steps=generations,
                evaluation_inputs=_evaluation_inputs(
                    search_states, budget_states, generation, oracle_budget
                ),
                reload_skill_memory=generation == 1,
            )

            for _ in range(int(config.agent_proxy.max_turn)):
                if not env_outputs:
                    break
                lm_inputs = context_manager.get_lm_inputs(
                    env_outputs, prepare_for_update=False
                )
                lm_inputs.meta_info = meta_info
                lm_outputs: DataProto = actor.generate_sequences(lm_inputs)
                env_inputs = context_manager.get_env_inputs(lm_outputs)
                env_outputs = state_manager.step(env_inputs)

            _consume_new_events(search_states, budget_states, event_offsets)
            current_results = _current_results(search_states, budget_states)
            metrics = aggregate_metrics(current_results)
            mean_calls = sum(
                state["oracle_calls"] for state in budget_states.values()
            ) / num_leads
            generation_metrics.append({
                "generation": generation,
                "temperature": temperature,
                "active_leads": active_leads,
                "mean_oracle_calls": mean_calls,
                **metrics,
            })
            print(
                f"Generation {generation}/{generations} | temperature={temperature:.2f} "
                f"| oracle_calls={mean_calls:.1f}/{oracle_budget} "
                f"| SR={metrics['success_rate']:.3f} "
                f"| RI={metrics['relative_improvement']:.3f} "
                f"| Sim={metrics['similarity']:.3f}"
            )

        results = _current_results(search_states, budget_states)
        metrics = aggregate_metrics(results)
        output = {
            "task": str(config.molecule_opt_task),
            "model": str(config.actor_rollout_ref.model.path),
            "settings": {
                "num_leads": num_leads,
                "oracle_budget_per_lead": oracle_budget,
                "similarity_threshold": float(config.molecule_opt_similarity_threshold),
                "generations": generations,
                "rollouts_per_generation": int(state_manager.group_size),
                "max_turns_per_rollout": int(config.agent_proxy.max_turn),
                "base_temperature": base_temperature,
                "temperature_increment": temperature_increment,
                "max_temperature": max_temperature,
            },
            "metrics": metrics,
            "generation_metrics": generation_metrics,
            "results": results,
        }

        output_path = Path(to_absolute_path(config.inference_output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
        print(f"Saved inference results to {output_path}")
    finally:
        state_manager.close()


if __name__ == "__main__":
    main()
