"""
Environment State Manager for the LLM agent.

This module manages multiple (kinds of) environments and handles:
- Environment initialization and reset
- Rollout cache management
- Evolving Skill Memory for experience-based learning
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import PIL.Image
import logging
import random
import numpy as np

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
from ragen.llm_agent.evolving_skill_memory import (
    EvolvingSkillMemory,
    classify_multi_objective_transition,
    extract_experience_from_trajectory,
)
register_resolvers()

logger = logging.getLogger(__name__)

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(
        self,
        config,
        mode: str = "train",
        reload_validation_skill_memory: bool = True,
    ):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        self._init_envs()
        self.rollout_cache = None
        self.global_best_results = {} # Initialize tracker for best results per initial molecule
        self.oracle_call_budget_tracker = {}       # Initialize tracker for cumulative real oracle calls per molecule (no cache)
        self.total_oracle_call_budget_tracker = {} # Initialize tracker for cumulative total oracle calls per molecule (including cache)
        
        skill_config = self.sys_config.evolving_skill_memory
        extraction_config = skill_config.get('extraction', {})
        self._skill_memory_init_kwargs = {
            'max_size': skill_config.get('max_size', 1000),
            'min_score_delta': extraction_config.get(
                'min_score_delta', skill_config.get('min_score_delta', 0.1)
            ),
        }
        configured_save_path = skill_config.get(
            'save_path', f'skill_memory_{mode}.json'
        )
        self._skill_memory_path = configured_save_path
        if mode == 'val':
            # Keep the requested path independently from the currently attached
            # memory. A missing snapshot may appear after the training manager
            # saves it, so every reset must be able to retry the same path.
            self._skill_memory_load_path = configured_save_path
            self._last_skill_memory_warning_key = None
            self._validation_skill_memory_reload_enabled = bool(
                reload_validation_skill_memory
            )
            self.skill_memory = self._new_empty_validation_skill_memory()
        else:
            self.skill_memory = EvolvingSkillMemory(
                save_path=None,
                read_only=False,
                **self._skill_memory_init_kwargs,
            )
            self.skill_memory.save_path = configured_save_path
        self.current_epoch = 0
        # Raw transformations live only until this training step is summarized.
        self._skill_candidates = []
        self._set_skill_memory_on_envs()

        if mode == 'val' and self._validation_skill_memory_reload_enabled:
            logger.info(
                "Evolving Skill Memory: validation snapshot will be loaded "
                "atomically on reset"
            )

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

        self.molecule_data_map = {}  # Use dictionary to map seed to molecules
        
        if any("MoleculeOptimization" in tag for tag in self.config.env_configs.tags):
            data_path = self.sys_config.data.train_file if self.mode == "train" else self.sys_config.data.val_file
            try:
                import pandas as pd
                df = pd.read_parquet(data_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load molecule dataset from {data_path}"
                ) from exc

            smiles_column = str(self.sys_config.data.get('smiles_column', 'smiles'))
            if smiles_column not in df.columns:
                raise ValueError(
                    f"Molecule dataset {data_path} has no {smiles_column!r} column"
                )
            if df.empty:
                raise ValueError(f"Molecule dataset is empty: {data_path}")

            columns = set(df.columns)
            for idx, row in df.iterrows():
                smiles = row[smiles_column]
                if not isinstance(smiles, str) or not smiles.strip():
                    raise ValueError(
                        f"Invalid SMILES value in {data_path} at row {idx}: {smiles!r}"
                    )

                mapped_seed = idx
                if 'seed' in columns and pd.notna(row['seed']):
                    mapped_seed = row['seed']
                elif 'extra_info' in columns:
                    extra_info = row['extra_info']
                    if isinstance(extra_info, dict) and 'seed' in extra_info:
                        mapped_seed = extra_info['seed']

                if mapped_seed in self.molecule_data_map:
                    raise ValueError(
                        f"Duplicate molecule seed {mapped_seed!r} in {data_path}"
                    )
                self.molecule_data_map[mapped_seed] = smiles.strip()

            print(
                f"Loaded {len(self.molecule_data_map)} molecules with seed mapping "
                f"from {data_path}"
            )

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                
                if env_class == "molecule_opt":
                    extra_env_config = {}
                    if hasattr(cfg_template, 'molecule_opt_task'):
                        extra_env_config['molecule_opt_task'] = cfg_template.molecule_opt_task
                    if hasattr(cfg_template, 'molecule_opt_similarity_threshold'):
                        extra_env_config['molecule_opt_similarity_threshold'] = cfg_template.molecule_opt_similarity_threshold
                        similarity_threshold = float(cfg_template.molecule_opt_similarity_threshold)
                        if env_config.final_similarity_threshold is None:
                            env_config.final_similarity_threshold = similarity_threshold

                    static_config = self.sys_config.static_exemplar_memory
                    from ragen.env.molecule_opt.config import RetrieverConfig

                    env_config.retriever = RetrieverConfig(
                        url=static_config.get('url', 'http://127.0.0.1:8000/retrieve'),
                        topk=static_config.get('topk', 5),
                        hint_topk=static_config.get('hint_topk', 3),
                        timeout=static_config.get('timeout', 5.0),
                        trigger_mode=static_config.get('trigger_mode', 'on_stuck'),
                        no_improvement_threshold=static_config.get('no_improvement_threshold', 2),
                        rollback_threshold=static_config.get('rollback_threshold', 1),
                        max_triggers=static_config.get('max_triggers', 2),
                        min_improvement_margin=static_config.get('min_improvement_margin', 0.0),
                    )
                    env_obj = REGISTERED_ENVS[env_class](env_config, extra_env_config)
                else:
                    env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def _set_skill_memory_on_envs(self):
        """Inject the shared Evolving Skill Memory into every molecule env."""
        config = self.sys_config.evolving_skill_memory
        hints_config = config.get('hints', {})
        # Validation/evaluation does not advance the training global step, so it
        # must not inherit the training-only hint warm-up.
        min_step_for_hints = (
            0 if self.mode == 'val'
            else int(config.get('min_step_for_hints', 5))
        )
        if min_step_for_hints < 0:
            raise ValueError("min_step_for_hints must be non-negative")
        trigger_mode = hints_config.get(
            'trigger_mode_val' if self.mode == 'val' else 'trigger_mode',
            'always' if self.mode == 'val' else 'on_stuck',
        )
        retrieval_config = dict(config.get('retrieval', {}))

        for entry in self.envs:
            env = entry['env']
            if hasattr(env, 'set_skill_memory'):
                env.set_skill_memory(
                    self.skill_memory,
                    min_step=min_step_for_hints,
                    trigger_mode=trigger_mode,
                    no_improvement_threshold=hints_config.get('no_improvement_threshold', 2),
                    rollback_threshold=hints_config.get('rollback_threshold', 1),
                    retrieval_config=retrieval_config,
                )

    def _new_empty_validation_skill_memory(self) -> EvolvingSkillMemory:
        """Return an unattached-path, read-only memory for safe validation fallback."""
        return EvolvingSkillMemory(
            save_path=None,
            read_only=True,
            **self._skill_memory_init_kwargs,
        )

    def load_training_skill_memory(self) -> bool:
        """Load the training skill bank only when the trainer resumes a checkpoint."""
        if self.mode != 'train':
            raise RuntimeError("Training skill memory can only be loaded in train mode")

        load_path = self._skill_memory_path
        try:
            candidate = EvolvingSkillMemory(
                save_path=None,
                read_only=False,
                **self._skill_memory_init_kwargs,
            )
            candidate.load(str(load_path))
            candidate.save_path = load_path
            required_tasks = self._skill_memory_tasks()
            missing_tasks = [
                task for task in required_tasks if not candidate.has_task(task)
            ]
            if missing_tasks:
                task_list = ', '.join(repr(task) for task in missing_tasks)
                raise ValueError(
                    f"snapshot contains no evidence-validated summaries for "
                    f"task(s) {task_list}"
                )
        except Exception as exc:
            _, pkl_path = EvolvingSkillMemory._storage_paths(str(load_path))
            logger.warning(
                "Could not restore Evolving Skill Memory from %s: %s: %s. "
                "Checkpoint training will continue with an empty skill bank.",
                pkl_path,
                type(exc).__name__,
                exc,
            )
            return False

        self.skill_memory = candidate
        self._set_skill_memory_on_envs()
        logger.info("Evolving Skill Memory restored with the training checkpoint")
        return True

    def enable_validation_skill_memory_reload(self) -> None:
        """Allow validation to consume a snapshot produced by this run."""
        if self.mode != 'val':
            raise RuntimeError("Only validation managers reload read-only memory")
        self._validation_skill_memory_reload_enabled = True

    def _skill_memory_tasks(self) -> List[str]:
        """Return molecule tasks that need retrievable validated summaries."""
        return sorted({
            str(entry['env'].molecule_opt_task)
            for entry in self.envs
            if hasattr(entry['env'], 'molecule_opt_task')
        })

    def _use_empty_validation_skill_memory(
        self,
        warning_key,
        reason: str,
    ) -> bool:
        """Replace the validation memory with an empty instance and warn once."""
        empty_memory = self._new_empty_validation_skill_memory()
        self.skill_memory = empty_memory
        self._set_skill_memory_on_envs()

        if warning_key != self._last_skill_memory_warning_key:
            logger.warning(
                "Evolving Skill Memory will not be used for this validation run: "
                "%s. Continuing without evolving-memory hints; Static Exemplar "
                "Memory remains available according to its configuration.",
                reason,
            )
        self._last_skill_memory_warning_key = warning_key
        return False

    def _reload_validation_skill_memory(self) -> bool:
        """Atomically replace validation memory with a checked snapshot or an empty instance."""
        load_path = self._skill_memory_load_path
        if not load_path:
            return self._use_empty_validation_skill_memory(
                ('missing_path',),
                'skill_memory_path is empty',
            )

        _, pkl_path = EvolvingSkillMemory._storage_paths(str(load_path))
        try:
            # Loading into a fresh candidate prevents a failed or partially
            # completed load from contaminating the attached memory.
            candidate = self._new_empty_validation_skill_memory()
            candidate.load(str(load_path))
            candidate.save_path = load_path
        except FileNotFoundError:
            return self._use_empty_validation_skill_memory(
                ('missing_pkl', pkl_path),
                f'PKL snapshot does not exist at {pkl_path}',
            )
        except Exception as exc:
            return self._use_empty_validation_skill_memory(
                ('load_error', pkl_path, type(exc).__name__, str(exc)),
                f'failed to load PKL snapshot {pkl_path}: '
                f'{type(exc).__name__}: {exc}',
            )

        try:
            required_tasks = self._skill_memory_tasks()
            missing_tasks = [
                task for task in required_tasks if not candidate.has_task(task)
            ]
        except Exception as exc:
            return self._use_empty_validation_skill_memory(
                ('validation_error', pkl_path, type(exc).__name__, str(exc)),
                f'failed to validate loaded PKL snapshot {pkl_path}: '
                f'{type(exc).__name__}: {exc}',
            )
        if missing_tasks:
            task_list = ', '.join(repr(task) for task in missing_tasks)
            return self._use_empty_validation_skill_memory(
                ('missing_validated_tasks', pkl_path, tuple(missing_tasks)),
                f'PKL snapshot {pkl_path} contains no evidence-validated '
                f'summaries for task(s) {task_list}',
            )

        # Publish only after loading and all task checks have completed.
        self.skill_memory = candidate
        self._set_skill_memory_on_envs()
        self._last_skill_memory_warning_key = None
        logger.info("Evolving Skill Memory snapshot loaded from %s", pkl_path)
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        global_step: int = 0,
        total_training_steps: int = 1,
        evaluation_inputs: Optional[Dict[int, Dict[str, Any]]] = None,
        reload_skill_memory: bool = True,
    ):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        if (
            self.mode == 'val'
            and reload_skill_memory
            and self._validation_skill_memory_reload_enabled
        ):
            self._reload_validation_skill_memory()

        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed # get a random seed
        else:
            seed = 123 if seed is None else seed
        expanded_seeds = _expand_seed(seed)

        for env_id, (current_seed, env_entry) in enumerate(zip(expanded_seeds, envs)):
            options = { # Prepare options dictionary
                "global_step": global_step,
                "total_training_steps": total_training_steps,
            }

            if evaluation_inputs is not None:
                group_id = env_entry['group_id']
                if group_id not in evaluation_inputs:
                    raise ValueError(f"Missing evaluation input for group {group_id}")
                evaluation_input = evaluation_inputs[group_id]
                options.update({
                    "initial_smiles": evaluation_input["lead_smiles"],
                    "starting_smiles": evaluation_input["start_smiles"],
                    "initial_properties": evaluation_input["lead_properties"],
                    "starting_properties": evaluation_input["start_properties"],
                })
                env_entry['_evaluation_budget_state'] = evaluation_input['budget_state']
                env_entry['_evaluation_generation'] = evaluation_input['generation']
            else:
                env_entry.pop('_evaluation_budget_state', None)
                env_entry.pop('_evaluation_generation', None)
            if (
                evaluation_inputs is None
                and hasattr(self, 'molecule_data_map')
                and len(self.molecule_data_map) > 0
            ):
                if current_seed in self.molecule_data_map:
                    options["initial_smiles"] = self.molecule_data_map[current_seed]
                else:
                    all_seeds = list(self.molecule_data_map.keys())
                    seed_idx = current_seed % len(all_seeds)
                    mapped_seed = all_seeds[seed_idx]
                    options["initial_smiles"] = self.molecule_data_map[mapped_seed]

            try:
                env_entry['env'].reset(seed=current_seed, options=options)
                env_entry['status'] = EnvStatus(seed=current_seed)

                if "initial_smiles" in options:
                    if not hasattr(self, 'env_smiles_map'):
                        self.env_smiles_map = {}
                    self.env_smiles_map[env_id] = options["initial_smiles"]

            except Exception as exc:
                raise RuntimeError(
                    f"Error resetting environment {env_id} with seed {current_seed}"
                ) from exc

        active_rollouts = []
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            budget_state = env.get('_evaluation_budget_state')
            if budget_state is not None and budget_state['oracle_calls'] >= budget_state['limit']:
                env['status'].terminated = True
                env['status'].truncated = True
            else:
                active_rollouts.append(cache)
            
        self.rollout_cache = rollout_cache
        return active_rollouts

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(entry, actions):
            env = entry['env']
            budget_state = entry.get('_evaluation_budget_state')
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            skill_transitions = []
            for action in actions:
                if budget_state is not None and budget_state['oracle_calls'] >= budget_state['limit']:
                    turn_info['oracle_budget_exhausted'] = True
                    turn_done = True
                    break

                _, reward, done, info = env.step(action)
                acc_reward += reward

                total_evaluations = int(info.get('total_oracle_calls_this_step', 0))
                if budget_state is not None and total_evaluations:
                    real_oracle_calls = int(info.get('oracle_calls_this_step', 0))
                    budget_state['oracle_calls'] += real_oracle_calls
                    budget_state['candidate_evaluations'] += total_evaluations

                    multi_info = info.get('multi_objective_info', {})
                    properties = multi_info.get('new_properties') or {
                        env.target_property: info.get('score_after', 0.0)
                    }
                    event = {
                        'oracle_call': budget_state['oracle_calls'],
                        'candidate_evaluation': budget_state['candidate_evaluations'],
                        'cached': real_oracle_calls == 0,
                        'generation': entry.get('_evaluation_generation', 0),
                        'env_id': entry['env_id'],
                        'smiles': info.get('final_smiles', action),
                        'properties': dict(properties),
                        'similarity': float(info.get('similarity', 0.0)),
                    }
                    budget_state['events'].append(event)
                    info['evaluation_oracle_call'] = budget_state['oracle_calls']

                # Preserve action-level transitions so scores stay paired with molecules.
                turn_info.update(info)
                skill_transitions.append(dict(info))
                executed_actions.append(action)
                if budget_state is not None and budget_state['oracle_calls'] >= budget_state['limit']:
                    turn_info['oracle_budget_exhausted'] = True
                    turn_done = True
                    break
                if done:
                    turn_done = True
                    break

            if skill_transitions:
                turn_info['_skill_transitions'] = skill_transitions
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            return status, history

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id = entry['env_id']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(entry, valid_actions[:actions_left_before])
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
                turn_done = True
            self.rollout_cache[env_id]['history'] = history
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        # Evaluation rollouts in the same lead group share one oracle budget.
        # A later sibling can exhaust it after an earlier sibling was queued for
        # another model turn, so remove every member of that group here.
        return [
            output
            for output in env_outputs
            if (
                envs[output['env_id']].get('_evaluation_budget_state') is None
                or envs[output['env_id']]['_evaluation_budget_state']['oracle_calls']
                < envs[output['env_id']]['_evaluation_budget_state']['limit']
            )
        ]

    def get_rollout_states(self):
        """Get final rollout states with molecule-level metrics."""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            
            if hasattr(entry['env'], '_original_base_molecule_smiles'):
                initial_smiles_key = entry['env']._original_base_molecule_smiles # Use original SMILES as key
                episode_best_score = entry['env']._best_score
                episode_best_smiles = entry['env']._best_molecule_smiles
                initial_score = entry['env']._initial_score # Get initial score
                similarity_at_best = entry['env']._similarity_at_best_score # Get similarity at best score
                episode_oracle_calls = entry['env']._oracle_calls_this_episode # Get real oracle calls from this episode (no cache)
                episode_total_oracle_calls = entry['env']._total_oracle_calls_this_episode # Get total oracle calls from this episode (including cache)

                if initial_smiles_key is not None:
                     current_calls = self.oracle_call_budget_tracker.get(initial_smiles_key, 0)
                     self.oracle_call_budget_tracker[initial_smiles_key] = current_calls + episode_oracle_calls
                     
                     current_total_calls = self.total_oracle_call_budget_tracker.get(initial_smiles_key, 0)
                     self.total_oracle_call_budget_tracker[initial_smiles_key] = current_total_calls + episode_total_oracle_calls

                if initial_smiles_key is not None: # Ensure we have a key
                    current_global_best = self.global_best_results.get(initial_smiles_key, None)
                    
                    env = entry['env']
                    current_episode_best_score = episode_best_score
                    
                    is_multi_objective = hasattr(env, 'molecule_opt_task') and '+' in env.molecule_opt_task
                    
                    is_better = False
                    if current_global_best is None:
                        is_better = True
                    else:
                        if is_multi_objective:
                            current_weighted_score = getattr(env, '_best_weighted_score', float('-inf'))
                            stored_weighted_score = current_global_best.get('weighted_score', float('-inf'))
                            is_better = current_weighted_score > stored_weighted_score
                        else:
                            is_sa_task = hasattr(env, 'minimize_target') and env.minimize_target
                            if is_sa_task:
                                is_better = current_episode_best_score < current_global_best['best_score']
                            else:
                                is_better = current_episode_best_score > current_global_best['best_score']
                    
                    if is_better:
                        result_data = {
                            'initial_smiles': initial_smiles_key, # Store explicitly for JSON clarity
                            'initial_score': initial_score,
                            'best_score': episode_best_score,
                            'best_smiles': episode_best_smiles,
                            'similarity_at_best': similarity_at_best, # Store the similarity
                            'total_oracle_calls': self.oracle_call_budget_tracker.get(initial_smiles_key, 0),
                            'success': True  # Mark as successful since we found a better score
                        }
                        
                        env = entry['env']
                        if is_multi_objective:
                            if hasattr(env, '_best_weighted_score'):
                                result_data['weighted_score'] = env._best_weighted_score
                            
                            if hasattr(env, '_best_individual_properties') and env._best_individual_properties:
                                result_data['individual_new_properties'] = env._best_individual_properties.copy()
                            
                            if hasattr(env, '_best_global_improvements') and env._best_global_improvements:
                                result_data['individual_improvements'] = env._best_global_improvements.copy()
                            
                            if hasattr(env, '_original_base_molecule_smiles'):
                                task_properties = env.molecule_opt_task.split('+')
                                try:
                                    original_properties = env.oracle.evaluate_specific_properties(
                                        env._original_base_molecule_smiles, task_properties)
                                    result_data['individual_old_properties'] = original_properties
                                    result_data['task_properties'] = task_properties
                                except Exception as e:
                                    print(f"Warning: Failed to get original properties for {initial_smiles_key}: {e}")
                            
                            if 'weighted_score' in result_data:
                                result_data['best_score'] = result_data['weighted_score']
                        
                        self.global_best_results[initial_smiles_key] = result_data
                    elif current_global_best is not None:
                         self.global_best_results[initial_smiles_key]['total_oracle_calls'] = self.oracle_call_budget_tracker.get(initial_smiles_key, 0)
        
        molecule_metrics = self._calculate_molecule_level_metrics(envs, rollout_cache)
        
        multi_objective_metrics = self._calculate_multi_objective_metrics(molecule_metrics)
        
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            
            initial_smiles = None
            if hasattr(entry['env'], '_original_base_molecule_smiles'):
                initial_smiles = entry['env']._original_base_molecule_smiles
            
            if initial_smiles and initial_smiles in molecule_metrics:
                mol_metrics = molecule_metrics[initial_smiles]
                env_metric = {
                    'success': float(mol_metrics['success']),
                    'num_actions': mol_metrics['best_num_actions'],
                    'current_global_best_score_for_mol': mol_metrics['best_score'],
                    'similarity': mol_metrics['best_similarity'],
                    'score_before': mol_metrics['initial_score'],
                    'score_after': mol_metrics['best_score'],
                    'improvement': mol_metrics['improvement']
                }
            else:
                env_metric = {
                    'success': float(status.terminated and (not status.truncated)),
                    'num_actions': status.num_actions,
                }
                
                if hasattr(entry['env'], '_original_base_molecule_smiles'):
                    initial_smiles_key = entry['env']._original_base_molecule_smiles
                    if initial_smiles_key in self.global_best_results:
                        env_metric['current_global_best_score_for_mol'] = self.global_best_results[initial_smiles_key]['best_score']

            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if (k == 'success' or 
                        k in ['final_smiles', 'action_received', 'initial_smiles', 'current_best_smiles'] or
                        k.startswith('_')):  # Skip fields starting with underscore (debug/internal fields)
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    try:
                        custom_metric[k].append(float(v))
                    except (ValueError, TypeError):
                        pass
            
            if initial_smiles and initial_smiles in molecule_metrics:
                mol_metrics = molecule_metrics[initial_smiles]
                for k, v in custom_metric.items():
                    if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                        if k in ['score_after', 'current_best_score', 'similarity', 'qed_score']:
                            env_metric[k] = mol_metrics.get(f'best_{k}', max(v) if v else 0.0)
                        elif k in ['score_before', 'initial_score']:
                            env_metric[k] = mol_metrics.get(f'initial_{k}', v[0] if v else 0.0)
                        elif k in ['oracle_calls_this_episode', 'oracle_calls_this_step']:
                            env_metric[k] = mol_metrics.get(f'total_{k}', sum(v) if v else 0.0)
                        else:
                            env_metric[k] = np.mean(v) if v else 0.0  # Average for other metrics
                    else:
                        env_metric[k] = np.sum(v)
            else:
                for k, v in custom_metric.items():
                    if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                        if k in ['score_after', 'current_best_score', 'similarity', 'qed_score']:
                            env_metric[k] = max(v) if v else 0.0  # Best value across all turns
                        elif k in ['score_before', 'initial_score']:
                            env_metric[k] = v[0] if v else 0.0  # Initial value
                        elif k in ['oracle_calls_this_episode', 'oracle_calls_this_step']:
                            env_metric[k] = sum(v) if v else 0.0  # Total calls
                        else:
                            env_metric[k] = np.mean(v) if v else 0.0  # Average for other metrics
                    else:
                        env_metric[k] = np.sum(v)

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer
        
        if rollout_cache and multi_objective_metrics:
            for cache in rollout_cache:
                if 'metrics' in cache and any('MoleculeOptimization' in key for key in cache['metrics'].keys()):
                    for metric_key, metric_value in multi_objective_metrics.items():
                        prefixed_key = f"MoleculeOptimization/{metric_key}"
                        if prefixed_key not in cache['metrics']:
                            cache['metrics'][prefixed_key] = metric_value
                    break

        if self.mode == 'train':
            self._extract_skills_from_rollouts(envs, rollout_cache)

        return rollout_cache

    def _extract_skills_from_rollouts(self, envs, rollout_cache):
        """Distill adjacent molecular transitions into skill-memory entries.

        Only successful transitions are retained. For multi-objective tasks,
        every direction-normalized objective must improve by ``min_delta``;
        trade-offs and unchanged objectives are discarded.
        """
        self._skill_candidates = []
        config = self.sys_config.get('evolving_skill_memory', {})
        min_step = int(config.get('min_step_for_experience', 5))
        if min_step < 0:
            raise ValueError("min_step_for_experience must be non-negative")
        if getattr(self, 'current_epoch', 0) < min_step:
            return

        extraction_config = config.get('extraction', {})
        min_delta = extraction_config.get(
            'min_score_delta', config.get('min_score_delta', 0.1)
        )
        for entry, cache in zip(envs, rollout_cache):
            env = entry['env']
            if not hasattr(env, '_original_base_molecule_smiles'):
                continue

            task = getattr(env, 'molecule_opt_task', 'unknown')
            task_properties = task.split('+')
            original_smiles = env._original_base_molecule_smiles
            history = cache.get('history', [])

            for index, turn in enumerate(history):
                info = turn.get('info', {})
                if index == 0:
                    smiles_before = original_smiles
                else:
                    smiles_before = history[index - 1].get('info', {}).get('final_smiles')

                transition_infos = info.get('_skill_transitions')
                if not transition_infos:
                    continue
                if not isinstance(transition_infos, list):
                    raise ValueError("_skill_transitions must be a list")
                for transition_info in transition_infos:
                    smiles_after = transition_info.get('final_smiles')
                    score_before = transition_info.get('score_before')
                    score_after = transition_info.get('score_after')

                    if score_before is None or score_after is None:
                        if smiles_after:
                            smiles_before = smiles_after
                        continue
                    if (
                        not smiles_before
                        or not smiles_after
                        or smiles_before == smiles_after
                    ):
                        if smiles_after:
                            smiles_before = smiles_after
                        continue

                    objective_deltas = {}
                    memory_score_before = float(score_before)
                    memory_score_after = float(score_after)

                    if len(task_properties) > 1:
                        multi_info = transition_info.get('multi_objective_info', {})
                        step_deltas = multi_info.get('step_improvements', {})
                        if any(prop not in step_deltas for prop in task_properties):
                            smiles_before = smiles_after
                            continue

                        directions = getattr(env, 'optimization_directions', {})
                        accepted, normalized_delta, objective_deltas = (
                            classify_multi_objective_transition(
                                {
                                    prop: step_deltas[prop]
                                    for prop in task_properties
                                },
                                directions,
                                min_delta,
                            )
                        )

                        if not accepted:
                            smiles_before = smiles_after
                            continue

                        memory_score_before = 0.0
                        memory_score_after = normalized_delta
                    else:
                        raw_delta = float(score_after) - float(score_before)
                        minimize = getattr(env, 'minimize_target', False)
                        normalized_delta = -raw_delta if minimize else raw_delta
                        objective_deltas[task] = normalized_delta
                        if normalized_delta < min_delta:
                            smiles_before = smiles_after
                            continue

                    try:
                        skill = extract_experience_from_trajectory(
                            before_smiles=smiles_before,
                            after_smiles=smiles_after,
                            trajectory=[smiles_before, smiles_after],
                            score_before=memory_score_before,
                            score_after=memory_score_after,
                            task=task,
                            epoch=getattr(self, 'current_epoch', 0),
                            objective_deltas=objective_deltas,
                        )
                        self._skill_candidates.append(skill)
                    except Exception as exc:
                        logger.debug(
                            "Skipping invalid skill-memory transition: %s", exc
                        )
                    finally:
                        smiles_before = smiles_after

    def _calculate_molecule_level_metrics(self, envs, rollout_cache):
        """Calculate molecule-level metrics by aggregating across all paths for each molecule"""
        from collections import defaultdict
        
        molecule_data = defaultdict(lambda: {
            'paths': [],
            'initial_score': None,
            'best_score': 0.0,
            'best_similarity': 0.0,
            'best_num_actions': 0,
            'success': False,
            'improvement': 0.0,
            'custom_metrics': defaultdict(list)
        })
        
        for entry, cache in zip(envs, rollout_cache):
            if not hasattr(entry['env'], '_original_base_molecule_smiles'):
                continue
                
            initial_smiles = entry['env']._original_base_molecule_smiles
            status = entry['status']
            
            path_success = bool(status.terminated and (not status.truncated))
            path_score = getattr(entry['env'], '_best_score', 0.0)
            path_similarity = getattr(entry['env'], '_similarity_at_best_score', 0.0)
            path_num_actions = status.num_actions
            initial_score = getattr(entry['env'], '_initial_score', 0.0)
            
            path_custom_metrics = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if (k == 'success' or 
                        k in ['final_smiles', 'action_received', 'initial_smiles', 'current_best_smiles'] or
                        k.startswith('_')):  # Skip fields starting with underscore (debug/internal fields)
                        continue
                    if k not in path_custom_metrics:
                        path_custom_metrics[k] = []
                    try:
                        path_custom_metrics[k].append(float(v))
                    except (ValueError, TypeError):
                        continue
            
            mol_data = molecule_data[initial_smiles]
            mol_data['paths'].append({
                'success': path_success,
                'score': path_score,
                'similarity': path_similarity,
                'num_actions': path_num_actions,
                'custom_metrics': path_custom_metrics
            })
            
            if mol_data['initial_score'] is None:
                mol_data['initial_score'] = initial_score
            
            if path_score > mol_data['best_score']:
                mol_data['best_score'] = path_score
                mol_data['best_similarity'] = path_similarity
                mol_data['best_num_actions'] = path_num_actions
            
            if path_success:
                mol_data['success'] = True
            
            mol_data['improvement'] = mol_data['best_score'] - mol_data['initial_score']
            
            for k, v_list in path_custom_metrics.items():
                mol_data['custom_metrics'][k].extend(v_list)
        
        for smiles, mol_data in molecule_data.items():
            for k, v_list in mol_data['custom_metrics'].items():
                if k in ['score_after', 'current_best_score', 'similarity', 'qed_score']:
                    mol_data[f'best_{k}'] = max(v_list) if v_list else 0.0
                elif k in ['score_before', 'initial_score']:
                    mol_data[f'initial_{k}'] = v_list[0] if v_list else 0.0
                elif k in ['oracle_calls_this_episode', 'oracle_calls_this_step']:
                    mol_data[f'total_{k}'] = sum(v_list) if v_list else 0.0
                else:
                    mol_data[f'avg_{k}'] = np.mean(v_list) if v_list else 0.0
        
        return dict(molecule_data)

    def _calculate_multi_objective_metrics(self, molecule_data):
        """
        Calculate multi-objective optimization specific metrics
        
        Args:
            molecule_data: molecule-level data dictionary
            
        Returns:
            dict: multi-objective metrics statistics
        """
        from ragen.env.molecule_opt.property_utils import PROPERTY_SUCCESS_THRESHOLDS, get_supported_properties
        
        all_properties = get_supported_properties()
        
        property_stats = {prop: {
            'improvements': [],
            'initial_scores': [],
            'best_scores': [],
            'success_count': 0,
            'total_count': 0,
            'above_threshold_count': 0
        } for prop in all_properties}
        
        qed_above_09_count = 0
        total_molecules = len(molecule_data)
        all_targets_success_count = 0
        
        for smiles, mol_data in molecule_data.items():
            # Check if there is multi-objective information
            multi_obj_info = None
            reward_histories = []
            
            # Extract multi-objective information and reward history from paths
            for path_data in mol_data['paths']:
                # Extract reward history
                if 'rewards' in path_data:
                    reward_histories.extend(path_data['rewards'])
                
                # Find multi-objective information from custom metrics
                if 'custom_metrics' in path_data:
                    path_custom = path_data['custom_metrics']
                    if any('multi_objective_info' in str(k) for k in path_custom.keys()):
                        # Build multi-objective information
                        multi_obj_info = {
                            'task_properties': [],
                            'old_properties': {},
                            'new_properties': {},
                            'success_status': {}
                        }
                        
                        # Rebuild multi-objective information from environment info
                        # This needs to be obtained from actual step info, skip this complex logic for now
                        break
                
                # Find direct multi-objective information
                if 'multi_objective_info' in path_data:
                    multi_obj_info = path_data['multi_objective_info']
                    break
            
            # If no multi-objective information is found, try to infer from custom metrics
            if not multi_obj_info:
                # Check if there are property-related metrics to infer this is a multi-objective task
                property_keys = ['qed', 'logp', 'sa', 'jnk3', 'drd2']
                found_properties = []
                
                for path_data in mol_data['paths']:
                    path_custom = path_data.get('custom_metrics', {})
                    for prop in property_keys:
                        if any(prop in str(k).lower() for k in path_custom.keys()):
                            if prop not in found_properties:
                                found_properties.append(prop)
                
                if len(found_properties) > 1:
                    # This might be a multi-objective task, but information is incomplete, skip for now
                    continue
                else:
                    # Single-objective task processing
                    continue
                
            # Get task properties
            task_properties = multi_obj_info.get('task_properties', [])
            old_properties = multi_obj_info.get('old_properties', {})
            new_properties = multi_obj_info.get('new_properties', {})
            success_status = multi_obj_info.get('success_status', {})
            
            all_success = True
            for prop in task_properties:
                if prop in old_properties and prop in new_properties:
                    initial_score = old_properties[prop]
                    final_score = new_properties[prop]
                    
                    # For SA, improvement should be negative (decreasing is good)
                    if prop == 'sa':
                        improvement = initial_score - final_score  # SA decreasing is good, so use initial-final
                        best_score = min(initial_score, final_score)  # SA smaller is better
                    else:
                        improvement = final_score - initial_score  # Other properties increasing is good
                        best_score = max(initial_score, final_score)  # Other properties larger is better
                    
                    property_stats[prop]['improvements'].append(improvement)
                    property_stats[prop]['initial_scores'].append(initial_score)
                    property_stats[prop]['best_scores'].append(best_score)
                    property_stats[prop]['total_count'] += 1
                    
                    from ragen.env.molecule_opt.property_utils import PROPERTY_SUCCESS_THRESHOLDS
                    
                    prop_success_status = success_status.get(prop, {})
                    
                    is_success = False
                    if prop in PROPERTY_SUCCESS_THRESHOLDS:
                        threshold = PROPERTY_SUCCESS_THRESHOLDS[prop]["threshold"]
                        direction = PROPERTY_SUCCESS_THRESHOLDS[prop]["direction"]
                        
                        if direction == "increase":
                            is_success = improvement >= threshold
                        else:  # direction == "decrease" (SA)
                            is_success = improvement <= -threshold
                    else:
                        is_success = improvement > 0
                    
                    # Compatible with original success_status (if available)
                    if prop_success_status.get('success', False) or is_success:
                        property_stats[prop]['success_count'] += 1
                    else:
                        all_success = False

                    if prop == 'qed' and best_score > 0.9:
                        property_stats[prop]['above_threshold_count'] += 1
            
    
            if len(task_properties) > 1 and all_success:
                all_targets_success_count += 1
            if 'qed' in task_properties:
                qed_best_score = property_stats['qed']['best_scores'][-1] if property_stats['qed']['best_scores'] else 0
                if qed_best_score > 0.9:
                    qed_above_09_count += 1
            elif 'qed' in new_properties and new_properties['qed'] > 0.9:
                qed_above_09_count += 1
        
        multi_objective_metrics = {}
        
        for prop in all_properties:
            stats = property_stats[prop]
            if stats['total_count'] > 0:
                multi_objective_metrics[f'{prop}/avg_improvement'] = np.mean(stats['improvements'])
                multi_objective_metrics[f'{prop}/success_rate'] = stats['success_count'] / stats['total_count']

        
        if total_molecules > 0:
            multi_objective_metrics['all_targets_success_rate'] = all_targets_success_count / total_molecules
            multi_objective_metrics['qed_above_0.9_rate_overall'] = qed_above_09_count / total_molecules
        
        if reward_histories:
            multi_objective_metrics['reward/mean'] = np.mean(reward_histories)
            multi_objective_metrics['reward/positive_rate'] = np.sum(np.array(reward_histories) > 0) / len(reward_histories)
        
        return multi_objective_metrics

    def get_average_oracle_calls_per_molecule(self):
        """Calculate the average of real oracle calls per molecule (accumulate call counts from all paths, excluding cache and initial scoring)."""
        if not self.oracle_call_budget_tracker:
             return 0.0
             
        total_calls = sum(self.oracle_call_budget_tracker.values())
        num_molecules = len(self.oracle_call_budget_tracker)
        return total_calls / num_molecules if num_molecules > 0 else 0.0
    
    def get_average_total_oracle_calls_per_molecule(self):
        """Calculate the average of total oracle calls per molecule (accumulate call counts from all paths, including cache hits)."""
        if not self.total_oracle_call_budget_tracker:
             return 0.0
             
        total_calls = sum(self.total_oracle_call_budget_tracker.values())
        num_molecules = len(self.total_oracle_call_budget_tracker)
        return total_calls / num_molecules if num_molecules > 0 else 0.0
    
    def get_average_similarity_at_best(self):
        """Calculate the average similarity at best score across all tracked molecules."""
        if not self.global_best_results:
            return 0.0
        
        similarities = []
        for data in self.global_best_results.values():
            if 'similarity_at_best' in data:
                similarities.append(data['similarity_at_best'])
        
        if not similarities:
            return 0.0
            
        return sum(similarities) / len(similarities)
    
    def save_skill_memory(self):
        """Save the Evolving Skill Memory to disk"""
        self.skill_memory.save()

    def set_current_epoch(self, epoch: int):
        """Set the creation epoch attached to subsequently extracted skills."""
        self.current_epoch = epoch

    def trigger_gpt_summarization(self) -> int:
        """Summarize this step's candidates once and store validated skills."""
        config = self.sys_config.get('evolving_skill_memory', {}).get('gpt_summary', {})
        candidates = self._skill_candidates
        self._skill_candidates = []
        if not candidates:
            return 0

        summaries = self.skill_memory.generate_gpt_summaries(
            experiences=candidates,
            total_timeout=config.get('total_timeout', 600),
            max_workers=config.get('max_workers', 8),
            model=config.get('model'),
            credential_env_prefix=config.get(
                'credential_env_prefix', 'AZURE_OPENAI'
            ),
            max_completion_tokens=config.get('max_completion_tokens', 150),
            reasoning_effort=config.get('reasoning_effort'),
        )
        added = 0
        for candidate in candidates:
            if candidate.exp_id in summaries:
                added += int(self.skill_memory.add_experience(candidate))
        return added

    def get_current_batch_success_rate(self):
        """Calculate success rate for the current batch of trajectories.
        For multi-objective tasks, returns overall success rate (all properties improved).
        For single-objective tasks, returns standard success rate."""
        if not self.rollout_cache:
            return 0.0
        
        is_multi_objective = False
        for entry in self.envs:
            if hasattr(entry['env'], 'molecule_opt_task') and '+' in entry['env'].molecule_opt_task:
                is_multi_objective = True
                break
        
        if is_multi_objective:
            success_count = 0
            total_count = 0
            
            for cache in self.rollout_cache:
                if 'history' in cache:
                    all_success = False
                    for turn in reversed(cache['history']):
                        info = turn.get('info', {})
                        multi_obj_info = info.get('multi_objective_info', {})
                        if multi_obj_info and 'all_success' in multi_obj_info:
                            all_success = multi_obj_info['all_success']
                            break
                    
                    if all_success:
                        success_count += 1
                    total_count += 1
            
            if total_count == 0:
                return 0.0
            
            return success_count / total_count
        else:
            success_count = 0
            total_count = 0
            
            for cache in self.rollout_cache:
                if 'metrics' in cache:
                    success = cache['metrics'].get(f"{cache.get('tag', 'Unknown')}/success", 0.0)
                    success_count += success
                    total_count += 1
            
            if total_count == 0:
                return 0.0
            
            return success_count / total_count
    
    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif state is None: # handle None state from molecule env
            return "No state available"
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        if isinstance(state, list) and len(state) > 0 and isinstance(state[0], np.ndarray):
            results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
            return results
        else:
            return str(state)
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()
