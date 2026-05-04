"""
FSDP PPO Trainer with Ray-based single controller.
Adapted from the excellently written verl implementation.
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from ragen.trainer import core_algos
from ragen.trainer.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, compute_response_mask, _timer, apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

import torch
from verl.utils.torch_functional import masked_mean

from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.utils import GenerationsLogger


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, bi_level_gae=False, high_level_gamma=1.0, config=None):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        if bi_level_gae:
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                loss_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
            )
        else:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                gamma=gamma,
                lam=lam,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "rank_based_grpo" or (hasattr(AdvantageEstimator, 'RANK_BASED_GRPO') and adv_estimator == AdvantageEstimator.RANK_BASED_GRPO):
        # Rank-based GRPO using both cumulative rewards and best molecule scores
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        
        # Extract best molecule scores from non_tensor_batch if available
        best_molecule_scores = data.non_tensor_batch.get("best_molecule_scores", None)
        if best_molecule_scores is None:
            # Fallback: use cumulative rewards as best scores
            best_molecule_scores = (data.batch["token_level_rewards"] * grpo_calculation_mask).sum(dim=-1)
            print("Warning: best_molecule_scores not found in batch, using cumulative rewards as fallback")
        
        # Get rank-based GRPO parameters from config
        if config is not None:
            cumulative_weight = getattr(config.algorithm, 'cumulative_weight', 0.5)
            rank_temperature = getattr(config.algorithm, 'rank_temperature', 1.0)
        else:
            cumulative_weight = 0.5
            rank_temperature = 1.0
        
        advantages, returns = core_algos.compute_rank_based_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            best_molecule_scores=best_molecule_scores,
            index=data.non_tensor_batch["uid"],
            cumulative_weight=cumulative_weight,
            rank_temperature=rank_temperature,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.HYBRID_PPO:
        # === Hybrid PPO: Combine Turn-level and Final-level Advantages ===
        # Get hybrid configuration parameters
        if config is not None:
            turn_weight = getattr(config.algorithm, 'turn_weight', 0.6)
            final_weight = getattr(config.algorithm, 'final_weight', 0.4)
            turn_gamma = getattr(config.algorithm, 'turn_gamma', 0.95)
            final_gamma = getattr(config.algorithm, 'final_gamma', 0.99)
            base_estimator = getattr(config.algorithm, 'base_estimator', 'gae')  # Base algorithm: gae or grpo
        else:
            turn_weight = 0.6
            final_weight = 0.4
            turn_gamma = 0.95
            final_gamma = 0.99
            base_estimator = 'gae'
        
        # Ensure weights are normalized
        total_weight = turn_weight + final_weight
        turn_weight = turn_weight / total_weight
        final_weight = final_weight / total_weight
        
        # Prepare computation masks
        response_mask = data.batch["response_mask"]
        grpo_calculation_mask = response_mask
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        
        # ===============================
        # 1. Calculate Turn-level Advantages
        # ===============================
        turn_data = data.batch.copy()
        if "turn_level_rm_scores" in data.batch and data.batch["turn_level_rm_scores"] is not None:
            turn_data["token_level_rewards"] = data.batch["turn_level_rm_scores"]
        else:
            # If no turn_level_rm_scores, try using existing token_level_rewards
            turn_data["token_level_rewards"] = data.batch["token_level_rewards"]
            print("Warning: turn_level_rm_scores not found, using token_level_rewards for turn advantages")
        
        if base_estimator.lower() == 'gae':
            if "values" in data.batch:
                turn_advantages, turn_returns = core_algos.compute_gae_advantage_return(
                    token_level_rewards=turn_data["token_level_rewards"],
                    values=data.batch["values"],
                    response_mask=response_mask,
                    gamma=turn_gamma,
                    lam=lam,
                )
            else:
                # Fallback to outcome-based if no values
                turn_advantages, turn_returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                    token_level_rewards=turn_data["token_level_rewards"],
                    response_mask=response_mask,
                    gamma=turn_gamma,
                )
        elif base_estimator.lower() == 'grpo':
            turn_advantages, turn_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=turn_data["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
            raise ValueError(f"Unsupported base_estimator: {base_estimator}")
        
        # ===============================
        # 2. 计算Final-level Advantages  
        # ===============================
        final_data = data.batch.copy()
        if "rm_scores" in data.batch and data.batch["rm_scores"] is not None:
            # 使用final rewards，需要扩展到token level
            final_rewards = data.batch["rm_scores"]  # shape: (batch_size,)
            if final_rewards.dim() == 1:
                # 扩展final reward到所有response tokens的最后一个位置
                final_token_rewards = torch.zeros_like(data.batch["token_level_rewards"])
                batch_size, seq_len = final_token_rewards.shape
                
                # 找到每个序列response的最后一个token位置
                for i in range(batch_size):
                    response_positions = (response_mask[i] > 0).nonzero(as_tuple=True)[0]
                    if len(response_positions) > 0:
                        last_pos = response_positions[-1].item()
                        final_token_rewards[i, last_pos] = final_rewards[i]
                
                final_data["token_level_rewards"] = final_token_rewards
            else:
                final_data["token_level_rewards"] = final_rewards
        else:
            # 如果没有rm_scores，使用现有的token_level_rewards作为final
            final_data["token_level_rewards"] = data.batch["token_level_rewards"]
            print("Warning: rm_scores not found, using token_level_rewards for final advantages")
        
        if base_estimator.lower() == 'gae':
            if "values" in data.batch:
                final_advantages, final_returns = core_algos.compute_gae_advantage_return(
                    token_level_rewards=final_data["token_level_rewards"],
                    values=data.batch["values"],
                    response_mask=response_mask,
                    gamma=final_gamma,
                    lam=lam,
                )
            else:
                # Fallback to outcome-based if no values
                final_advantages, final_returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                    token_level_rewards=final_data["token_level_rewards"],
                    response_mask=response_mask,
                    gamma=final_gamma,
                )
        elif base_estimator.lower() == 'grpo':
            final_advantages, final_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=final_data["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        
        # ===============================
        # 3. 加权组合Advantages
        # ===============================
        combined_advantages = turn_weight * turn_advantages + final_weight * final_advantages
        combined_returns = turn_weight * turn_returns + final_weight * final_returns
        
        # 设置组合后的advantages和returns
        data.batch["advantages"] = combined_advantages
        data.batch["returns"] = combined_returns
        
        # 保存原始advantages用于分析（可选）
        data.batch["turn_advantages"] = turn_advantages
        data.batch["final_advantages"] = final_advantages
        data.batch["turn_returns"] = turn_returns
        data.batch["final_returns"] = final_returns
        
        print(f"Hybrid PPO: Combined advantages with weights - Turn: {turn_weight:.3f}, Final: {final_weight:.3f}")
        print(f"Turn advantages mean: {turn_advantages.mean().item():.6f}, std: {turn_advantages.std().item():.6f}")
        print(f"Final advantages mean: {final_advantages.mean().item():.6f}, std: {final_advantages.std().item():.6f}")
        print(f"Combined advantages mean: {combined_advantages.mean().item():.6f}, std: {combined_advantages.std().item():.6f}")
    else:
        raise NotImplementedError
    return data


class RayAgentTrainer(VerlRayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn)
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        # do not use the original val logger, but use this here
        self.generations_logger = GenerationsLogger()

        
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        assert self.config.trainer.total_training_steps is not None, "must determine total training steps"
        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
        # val_start = 100000
        # self.train_seeds = [seed for seed in range(0, self.config.trainer.total_training_steps * 1000, 1000)]
        # self.val_seeds = [seed for seed in range(val_start, val_start + self.config.trainer.validation_steps)]

    def init_agent_proxy(self):
        self.agent_proxy = LLMAgentProxy(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer
        )
    def _maybe_log_generations(self, inputs, outputs, scores, _type="val"):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.generations_to_log_to_wandb[_type]

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.generations_logger.log(self.config.trainer.logger, samples, self.global_steps, _type)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        env_metric_dict = {}
        for step in range(self.config.trainer.validation_steps):
            
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            import time
            start_time = time.time()
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            end_time = time.time()
            print(f"validation generation time: {end_time - start_time} seconds")
            for key, value in test_batch.meta_info["metrics"].items():
                # Keep essential molecule optimization metrics for validation, including multi-objective metrics
                essential_single = ["current_best_score", "average_improvement", "oracle_calls"]
                
                # Multi-objective property-specific metrics (简化版本)
                multi_objective_patterns = [
                    "avg_improvement", "success_rate", "above_0.9_rate"
                ]
                
                # Overall success rate metrics
                success_patterns = [
                    "all_targets_success_rate", "single_target_success_rate", 
                    "qed_above_0.9_rate_overall"
                ]
                
                # Reward distribution metrics
                reward_patterns = ["reward/mean", "reward/positive_rate"]
                
                # Check if this key should be included
                should_include = (
                    any(essential in key for essential in essential_single) or
                    any(pattern in key for pattern in multi_objective_patterns) or
                    any(pattern in key for pattern in success_patterns) or
                    any(pattern in key for pattern in reward_patterns)
                )
                
                # Skip old molecule_opt metrics - using new clean validation metrics instead

            # Store inputs and outputs
            if "prompts" in test_batch.batch:
                input_ids = test_batch.batch["prompts"]
            elif "input_ids" in test_batch.batch:
                input_ids = test_batch.batch["input_ids"]
            else:
                # Fallback: empty inputs
                input_ids = [[]] * len(test_batch.batch["responses"])
            
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            # Store generated outputs
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result and result["reward_extra_info"]:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
            # If no reward_extra_info, initialize with empty lists for expected keys
            elif "reward_extra_info" not in result:
                pass  # Just use rewards, no extra info needed

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, _type="val")

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # --- Add Validation Metrics (matching training metrics) ---
        if hasattr(self.agent_proxy, 'val_es_manager') and self.agent_proxy.val_es_manager is not None:
            # Oracle calls (cumulative)
            val_avg_oracle_calls = self.agent_proxy.val_es_manager.get_average_oracle_calls_per_molecule()
            metric_dict["val/oracle_calls"] = val_avg_oracle_calls
            
            # Average similarity (current step's best molecules)  
            val_avg_similarity = self.agent_proxy.val_es_manager.get_average_similarity_at_best()
            metric_dict["val/avg_similarity"] = val_avg_similarity
            
            # Calculate property-specific validation metrics using the new system
            if hasattr(self.config, 'custom_envs') and 'MoleculeOptimization' in self.config.custom_envs:
                current_task = getattr(self.config.custom_envs.MoleculeOptimization, 'molecule_opt_task', 'qed')
                best_results_dict = self.agent_proxy.val_es_manager.global_best_results
                
                if best_results_dict:
                    # Import calculation functions
                    from ragen.env.molecule_opt.property_utils import calculate_success_rates, calculate_property_averages
                    
                    # Calculate improvement-based success rates
                    val_success_rates = calculate_success_rates(current_task, best_results_dict, improvement_based=True)
                    for metric_name, value in val_success_rates.items():
                        metric_dict[f"val/{metric_name}"] = value
                    
                    # Calculate average metrics (improvement and best scores)
                    val_averages = calculate_property_averages(current_task, best_results_dict)
                    for metric_name, value in val_averages.items():
                        metric_dict[f"val/{metric_name}"] = value
                        
                    # For single QED task, also calculate the traditional 0.9 threshold success rate
                    if current_task == "qed":
                        qed_above_09_rates = calculate_success_rates(current_task, best_results_dict, improvement_based=False)
                        for metric_name, value in qed_above_09_rates.items():
                            metric_dict[f"val/{metric_name}"] = value
        
        # --- Add Validation Reward Metric ---
        if sample_scores:
            avg_val_reward = sum(sample_scores) / len(sample_scores)
            metric_dict["val/reward"] = avg_val_reward
        
        # --- Add Generations Summary for Wandb ---
        if sample_inputs and sample_outputs and sample_scores:
            # Initialize generation tracking table if first time
            if not hasattr(self, '_generations_table'):
                # Create table structure: each row is a training step, columns are molecules
                self._generations_table = []
            
            # Create current step data (no sorting, preserve original order)
            current_step_data = {"step": self.global_steps}
            
            # Add input, output, score for each molecule (up to 20 molecules)
            num_molecules_to_track = min(20, len(sample_inputs))
            for i in range(num_molecules_to_track):
                mol_idx = i + 1  # 1-indexed for readability
                current_step_data[f"input_{mol_idx}"] = sample_inputs[i]
                current_step_data[f"output_{mol_idx}"] = sample_outputs[i]
                current_step_data[f"score_{mol_idx}"] = round(sample_scores[i], 4)
            
            # Add current step to the table
            self._generations_table.append(current_step_data)
            
            # Store the table for wandb logging
            metric_dict["val/generations"] = self._generations_table

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
 
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and not self.ref_in_actor:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )


    def _save_checkpoint(self):
        """ 
        Different from VerlRayPPOTrainer, we have no dataloader so we won"t save it. Other logic is the same.
        """
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _save_best_results_to_json(self):
        """Saves the tracked global best results to a JSON file and calculates property metrics."""
        property_metrics = {}

        if hasattr(self.agent_proxy, 'train_es_manager') and self.agent_proxy.train_es_manager is not None:
            best_results_dict = self.agent_proxy.train_es_manager.global_best_results
            
            # Save JSON
            if best_results_dict: # Only save if there's something to save
                save_dir = self.config.trainer.default_local_dir
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"best_results_step_{self.global_steps}.json")
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(best_results_dict, f, indent=4)
                    print(f"Successfully saved best results tracking to {filename}")
                except Exception as e:
                    print(f"Error saving best results to JSON: {e}")
            else:
                print("No best results tracked yet, skipping JSON save.")

            # Calculate success rates using new unified system
            if hasattr(self.config, 'custom_envs') and 'MoleculeOptimization' in self.config.custom_envs and best_results_dict:
                # Get current task from config
                current_task = getattr(self.config.custom_envs.MoleculeOptimization, 'molecule_opt_task', 'qed')
                
                # Import calculation functions
                from ragen.env.molecule_opt.property_utils import calculate_success_rates, calculate_property_averages
                
                # Calculate improvement-based success rates
                improvement_success_rates = calculate_success_rates(current_task, best_results_dict, improvement_based=True)
                property_metrics.update(improvement_success_rates)
                
                # Calculate average metrics (improvement and best scores)
                property_averages = calculate_property_averages(current_task, best_results_dict)
                property_metrics.update(property_averages)
                
                # For single QED task, also calculate the traditional 0.9 threshold success rate
                if current_task == "qed":
                    qed_above_09_rates = calculate_success_rates(current_task, best_results_dict, improvement_based=False)
                    property_metrics.update(qed_above_09_rates)
            
        else:
            print("Warning: Could not access train_es_manager to save best results or calculate success rates.")
        
        return property_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
         to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()
            return inputs, outputs, scores
        
        def _filter_by_sequence_length(batch):
            """过滤过长的序列，避免训练时OOM"""
            enable_length_filtering = getattr(self.config.actor_rollout_ref.rollout, 'enable_length_filtering', True)
            
            if not enable_length_filtering:
                return batch, {"rollout/length_filtered_count": 0}
                
            max_allowed_length = self.config.actor_rollout_ref.rollout.max_model_len
            input_ids = batch.batch["input_ids"]
            attention_mask = batch.batch["attention_mask"]
            
            # 计算每个序列的实际长度
            seq_lengths = attention_mask.sum(dim=-1)
            valid_mask = seq_lengths <= max_allowed_length
            
            num_filtered = (~valid_mask).sum().item()
            if num_filtered > 0:
                print(f"[长度过滤] 过滤掉 {num_filtered} 个超长序列 (长度 > {max_allowed_length})")
                
                # 应用过滤mask
                batch.batch = batch.batch[valid_mask]
                for key, value in batch.non_tensor_batch.items():
                    if isinstance(value, np.ndarray):
                        batch.non_tensor_batch[key] = value[valid_mask]
                    else:
                        batch.non_tensor_batch[key] = [v for v, m in zip(value, valid_mask) if m]
                
                return batch, {"rollout/length_filtered_count": num_filtered}
            
            return batch, {"rollout/length_filtered_count": 0}

        def _filter_rollout_original(batch):
            """Original filter: filter rollout based on in-group std. We want those groups to have high-quality rollouts that deviates significantly from the mean"""
            rollout_filter_ratio = self.config.actor_rollout_ref.rollout.rollout_filter_ratio
            num_groups, group_size = self.config.es_manager.train.env_groups, self.config.es_manager.train.group_size

            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
            in_group_std = rm_scores.std(dim=-1)
            in_group_max = rm_scores.max(dim=-1).values
            in_group_mean = rm_scores.mean(dim=-1)
            if rollout_filter_ratio == 1:
                return batch, {"rollout/group_variance": in_group_std.mean()}

            if self.config.actor_rollout_ref.rollout.rollout_filter_type == "std_rev":
                top_groups = (-in_group_std).topk(int(rollout_filter_ratio * num_groups)).indices
            elif self.config.actor_rollout_ref.rollout.rollout_filter_type == "std":
                top_groups = in_group_std.topk(int(rollout_filter_ratio * num_groups)).indices
            else:
                raise ValueError(f"Invalid rollout filter type: {self.config.actor_rollout_ref.rollout.rollout_filter_type}")

            mask = torch.zeros(num_groups, dtype=torch.bool)
            mask[top_groups] = True
            mask = mask.unsqueeze(1).expand(-1, group_size).flatten()

            batch.batch = batch.batch[mask]

            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, mask) if m]

            metrics = {
                "rollout/group_variance": in_group_std[top_groups].mean()
            }
            return batch, metrics

        def _filter_rollout_two_stage(batch):
            """Two-stage filter: 
            Stage 1: Filter groups by variance (remove low-variance groups)
            Stage 2: Within remaining groups, keep top percentile by cumulative score
            """
            # Get filter parameters from config
            variance_filter_ratio = self.config.actor_rollout_ref.rollout.get("variance_filter_ratio", 0.75)
            score_filter_ratio = self.config.actor_rollout_ref.rollout.get("score_filter_ratio", 0.5)
            variance_filter_type = self.config.actor_rollout_ref.rollout.get("variance_filter_type", "std")
            
            num_groups, group_size = self.config.es_manager.train.env_groups, self.config.es_manager.train.group_size
            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
            
            # === Stage 1: Variance-based group filtering ===
            in_group_std = rm_scores.std(dim=-1)
            in_group_max = rm_scores.max(dim=-1).values
            in_group_mean = rm_scores.mean(dim=-1)
            
            # Select groups based on variance
            if variance_filter_ratio >= 1.0:
                # Keep all groups
                stage1_selected_groups = torch.arange(num_groups)
            else:
                num_groups_to_keep_stage1 = max(1, int(variance_filter_ratio * num_groups))
                # Handle edge case where all groups have same variance (std=0)
                if in_group_std.max() == 0:
                    # If all groups have zero variance, randomly select groups to keep
                    stage1_selected_groups = torch.randperm(num_groups)[:num_groups_to_keep_stage1]
                elif variance_filter_type == "std_rev":
                    # Keep groups with low variance (more consistent performance)
                    stage1_selected_groups = (-in_group_std).topk(num_groups_to_keep_stage1).indices
                elif variance_filter_type == "std":
                    # Keep groups with high variance (more diverse performance)
                    stage1_selected_groups = in_group_std.topk(num_groups_to_keep_stage1).indices
                else:
                    raise ValueError(f"Invalid variance_filter_type: {variance_filter_type}")

            # Create stage 1 mask
            stage1_mask = torch.zeros(num_groups, dtype=torch.bool)
            stage1_mask[stage1_selected_groups] = True
            stage1_full_mask = stage1_mask.unsqueeze(1).expand(-1, group_size).flatten()

            # === Stage 2: Score-based individual filtering within selected groups ===
            if score_filter_ratio >= 1.0:
                # Keep all samples from selected groups
                final_mask = stage1_full_mask
                stage2_kept_per_group = group_size
            else:
                samples_per_group_to_keep = max(1, int(score_filter_ratio * group_size))
                final_mask = torch.zeros_like(stage1_full_mask, dtype=torch.bool)
                
                stage2_kept_counts = []
                for group_idx in stage1_selected_groups:
                    group_start = group_idx * group_size
                    group_end = group_start + group_size
                    group_scores = rm_scores[group_idx]  # Scores for this group
                    
                    # Handle edge case where group has fewer samples than we want to keep
                    actual_samples_to_keep = min(samples_per_group_to_keep, group_size)
                    
                    # Get top samples within this group
                    if actual_samples_to_keep >= group_size:
                        # Keep all samples in this group
                        top_indices_in_group = torch.arange(group_size)
                    else:
                        # Get top k samples
                        top_indices_in_group = group_scores.topk(actual_samples_to_keep).indices
                    
                    # Update final mask
                    for idx in top_indices_in_group:
                        final_mask[group_start + idx] = True
                    
                    stage2_kept_counts.append(len(top_indices_in_group))
                
                stage2_kept_per_group = sum(stage2_kept_counts) / len(stage2_kept_counts) if stage2_kept_counts else 0

            # Apply the final mask to batch
            batch.batch = batch.batch[final_mask]
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[final_mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, final_mask) if m]
            
            # === Reorder by group_id to maintain group structure for preference learning ===
            if "group_ids" in batch.non_tensor_batch:
                group_ids = batch.non_tensor_batch["group_ids"]
                unique_groups = np.unique(group_ids)
                reorder_indices = []
                
                # Collect indices for each group in sorted order
                for group_id in sorted(unique_groups):
                    group_indices = np.where(group_ids == group_id)[0]
                    reorder_indices.extend(group_indices.tolist())
                
                reorder_indices = np.array(reorder_indices)
                
                # Reorder batch tensors
                if hasattr(batch.batch, '__getitem__'):
                    batch.batch = batch.batch[reorder_indices]
                
                # Reorder non_tensor_batch
                for key, value in batch.non_tensor_batch.items():
                    if isinstance(value, np.ndarray):
                        batch.non_tensor_batch[key] = value[reorder_indices]
                    elif isinstance(value, list):
                        batch.non_tensor_batch[key] = [value[i] for i in reorder_indices]
                
                # Debug: Print group structure after reordering
                if self.global_steps % 10 == 0:  # Only print every 10 steps
                    reordered_groups = batch.non_tensor_batch["group_ids"]
                    group_counts = {}
                    for gid in reordered_groups:
                        group_counts[gid] = group_counts.get(gid, 0) + 1
                    print(f"[Group reorder] Step {self.global_steps}: "
                          f"Groups after filter+reorder: {dict(sorted(group_counts.items()))}")
            # ================================================================================

            # Compute metrics
            selected_group_indices = stage1_selected_groups.cpu().numpy()
            original_batch_size = len(stage1_full_mask)
            final_batch_size = final_mask.sum().item()
            
            metrics = {
                "rollout/group_variance": in_group_std[stage1_selected_groups].mean() if len(stage1_selected_groups) > 0 else 0.0,
                "rollout/filter_ratio": final_batch_size / original_batch_size,
            }
            
            # Optional debug print (can be removed in production)
            if self.global_steps % 10 == 0:  # Only print every 10 steps to avoid spam
                print(f"[Two-stage filter] Step {self.global_steps}: "
                      f"{original_batch_size}→{final_batch_size} samples "
                      f"(groups: {num_groups}→{len(stage1_selected_groups)}, "
                      f"samples/group: {group_size}→{stage2_kept_per_group:.1f})")
            
            return batch, metrics

        def _filter_rollout(batch):
            """Choose between original and two-stage filtering based on config"""
            # 首先进行长度过滤，防止超长序列进入训练
            batch, length_metrics = _filter_by_sequence_length(batch)
            
            filter_mode = self.config.actor_rollout_ref.rollout.get("filter_mode", "original")
            
            if filter_mode == "two_stage":
                batch, filter_metrics = _filter_rollout_two_stage(batch)
            elif filter_mode == "original":
                batch, filter_metrics = _filter_rollout_original(batch)
            else:
                raise ValueError(f"Invalid filter_mode: {filter_mode}. Must be 'original' or 'two_stage'")
            
            # 合并metrics
            filter_metrics.update(length_metrics)
            return batch, filter_metrics

        import time
        self.start_time = time.time()
        for step in range(self.total_training_steps):
            # metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # generate a batch
                with _timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False, global_step=self.global_steps, total_training_steps=self.total_training_steps)
                    # 记录filter前的batch大小
                    self._last_batch_size_before_filter = len(batch.batch["input_ids"]) if "input_ids" in batch.batch else 0
                    batch, metrics = _filter_rollout(batch)
                    # Keep essential molecule optimization metrics for training, including multi-objective metrics
                    for key, value in batch.meta_info["metrics"].items():
                        # Essential single-objective metrics
                        essential_single = ["current_best_score", "oracle_calls", "similarity"]
                        
                        # Multi-objective property-specific metrics (简化版本)
                        multi_objective_patterns = [
                            "avg_improvement", "success_rate", "above_0.9_rate"
                        ]
                        
                        # Overall success rate metrics
                        success_patterns = [
                            "all_targets_success_rate", "single_target_success_rate", 
                            "qed_above_0.9_rate_overall"
                        ]
                        
                        # Reward distribution metrics
                        reward_patterns = ["reward/mean", "reward/positive_rate"]
                        
                        # Check if this key should be included
                        should_include = False
                        
                        # Check essential single-objective metrics
                        if any(essential in key for essential in essential_single):
                            should_include = True
                        
                        # Check multi-objective property metrics (qed/avg_improvement, logp/success_rate, etc.)
                        elif any(pattern in key for pattern in multi_objective_patterns):
                            should_include = True
                        
                        # Check overall success rate metrics
                        elif any(pattern in key for pattern in success_patterns):
                            should_include = True
                        
                        # Check reward metrics
                        elif any(pattern in key for pattern in reward_patterns):
                            should_include = True
                        
                        # Skip old molecule_opt metrics - using new clean metrics instead

                    inputs, outputs, scores = _process_batch_for_logging(batch)
                    # self._maybe_log_generations(inputs=inputs, outputs=outputs, scores=scores, _type="train")





                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # TODO: check if this is correct. Not tested yer
                    logger.log("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()
                    with _timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.agent_proxy.rollout(gen_baseline_batch, val=False)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            # dtype=object)
                # repeat to align with repeated responses in rollout
                # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # batch = batch.union(gen_batch_output)

                # NOTE reward normalization already done in ctx_manager, so set group size = 1 here
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            dtype=object)
                # batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]

                # batch.batch["response_mask"] = compute_response_mask(batch)
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if self.use_rm:
                    with _timer("reward", timing_raw):
                    # compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                # Initialize reward_extra_infos_dict to ensure it's always defined
                reward_extra_infos_dict: dict[str, list] = {}
                
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs
                with _timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)
                    avg_old_log_prob = masked_mean(old_log_prob.batch["old_log_probs"], batch.batch["response_mask"])
                    metrics.update({"rollout/old_log_prob": avg_old_log_prob})

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer("adv", timing_raw):
                    # we combine with rule-based rm
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    # reward_extra_infos_dict已经在第786行非异步分支中被赋值
                    batch.batch["token_level_scores"] = reward_tensor

                    # Only print and update if we have valid reward_extra_infos_dict
                    if reward_extra_infos_dict:
                        print(f"Reward extra info keys: {list(reward_extra_infos_dict.keys())}")
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                    else:
                        print("No reward extra info available")
                    

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]


                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=True,
                        high_level_gamma=self.config.algorithm.high_level_gamma,
                        bi_level_gae=self.config.algorithm.bi_level_gae,
                        config=self.config,
                    )

                ##### A very different setting, just here for testing: Can I normalize the advantages to have a mean of 0?
                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO and self.config.grpo_advantage_length_weight:
                    response_mask = batch.batch["response_mask"]
                    advantages = batch.batch["advantages"]
                    response_relative_lengths = (torch.sum(response_mask, dim=-1) + 1e-6) / torch.sum(response_mask, dim=-1).float().mean()
                    advantages = advantages / response_relative_lengths.unsqueeze(-1) 
                    batch.batch["advantages"] = advantages

                # update critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with _timer("dump_rollout_generations", timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                        # Also save elite pool if enabled
                        if hasattr(self.agent_proxy, 'train_es_manager') and self.agent_proxy.train_es_manager is not None:
                            self.agent_proxy.train_es_manager.save_elite_pool()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # --- Save Best Results JSON and Calculate Metrics ---
            property_metrics = self._save_best_results_to_json()
            if property_metrics:
                # Add all property metrics with clean naming (no molecule_opt prefix)
                for metric_name, value in property_metrics.items():
                    metrics.update({f"train/{metric_name}": value})

            # --- Log Training Metrics with Clean Naming ---
            if hasattr(self.agent_proxy, 'train_es_manager') and self.agent_proxy.train_es_manager is not None:
                # Oracle calls (cumulative)
                avg_oracle_calls = self.agent_proxy.train_es_manager.get_average_oracle_calls_per_molecule()
                metrics.update({"train/oracle_calls": avg_oracle_calls})
                
                # Total Oracle calls including cache hits (cumulative)
                avg_total_oracle_calls = self.agent_proxy.train_es_manager.get_average_total_oracle_calls_per_molecule()
                metrics.update({"train/total_oracle_calls": avg_total_oracle_calls})
                
                # Average similarity (current step's best molecules)
                avg_similarity = self.agent_proxy.train_es_manager.get_average_similarity_at_best()
                metrics.update({"train/avg_similarity": avg_similarity})
                
                # Current batch success rate
                batch_success_rate = self.agent_proxy.train_es_manager.get_current_batch_success_rate()
                metrics.update({"train/batch_success_rate": batch_success_rate})
                
            # --- Add Reward Statistics ---
            if "token_level_scores" in batch.batch:
                # token_level_scores is the raw reward from environment
                reward_tensor = batch.batch["token_level_scores"]
                # Sum across sequence length to get per-sample rewards
                rewards = reward_tensor.sum(-1).cpu().numpy()
                avg_reward = float(rewards.mean())
                metrics.update({"train/reward": avg_reward})
            elif "rm_scores" in batch.batch:
                # Fallback to rm_scores if available
                reward_tensor = batch.batch["rm_scores"]
                rewards = reward_tensor.sum(-1).cpu().numpy()
                avg_reward = float(rewards.mean())
                metrics.update({"train/reward": avg_reward})
            else:
                 print("Warning: Could not access train_es_manager to log metrics.")

            # add another timing metric: total time
            metrics.update({"timing_s/total": time.time() - self.start_time})
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
