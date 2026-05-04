# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import numpy as np
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
from ragen.trainer.core_algos import build_preference_pairs, compute_lipo_lambda_loss
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from peft import PeftModel


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, no_lora=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        is_peft_model = not no_lora and isinstance(self.actor_module._fsdp_wrapped_module, PeftModel)
        if is_peft_model:
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.merge_adapter()

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if is_peft_model:
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.unmerge_adapter()
        

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask", "rm_scores", "turn_level_rm_scores"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        # Check if preference loss data exists
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if "best_molecule_scores" in data.non_tensor_batch:
            non_tensor_select_keys.append("best_molecule_scores")
        if "group_ids" in data.non_tensor_batch:
            non_tensor_select_keys.append("group_ids")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs or len(non_tensor_select_keys) > 0:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs or len(non_tensor_select_keys) > 0:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = data["response_mask"]
                    # response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # === Hybrid PPO metrics logging ===
                    # Check if hybrid_ppo algorithm is used
                    if hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'adv_estimator'):
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.HYBRID_PPO:
                            # Log statistics of turn and final advantages
                            if "turn_advantages" in data and "final_advantages" in data:
                                turn_adv = data["turn_advantages"]
                                final_adv = data["final_advantages"]
                                combined_adv = data["advantages"]
                                
                                # Calculate masked statistics (only consider response part)
                                turn_masked = turn_adv * response_mask
                                final_masked = final_adv * response_mask
                                combined_masked = combined_adv * response_mask
                                
                                # Calculate number of valid tokens
                                valid_tokens = response_mask.sum()
                                
                                if valid_tokens > 0:
                                    # Log mean and standard deviation
                                    metrics["actor/hybrid_turn_adv_mean"] = (turn_masked.sum() / valid_tokens).detach().item()
                                    metrics["actor/hybrid_final_adv_mean"] = (final_masked.sum() / valid_tokens).detach().item()
                                    metrics["actor/hybrid_combined_adv_mean"] = (combined_masked.sum() / valid_tokens).detach().item()
                                    
                                    # Calculate standard deviation (need to calculate variance first)
                                    turn_var = ((turn_masked - metrics["actor/hybrid_turn_adv_mean"]) ** 2 * response_mask).sum() / valid_tokens
                                    final_var = ((final_masked - metrics["actor/hybrid_final_adv_mean"]) ** 2 * response_mask).sum() / valid_tokens
                                    combined_var = ((combined_masked - metrics["actor/hybrid_combined_adv_mean"]) ** 2 * response_mask).sum() / valid_tokens
                                    
                                    metrics["actor/hybrid_turn_adv_std"] = torch.sqrt(turn_var).detach().item()
                                    metrics["actor/hybrid_final_adv_std"] = torch.sqrt(final_var).detach().item()
                                    metrics["actor/hybrid_combined_adv_std"] = torch.sqrt(combined_var).detach().item()
                                    
                                    # Log weight information
                                    if hasattr(self.config.algorithm, 'turn_weight'):
                                        metrics["actor/hybrid_turn_weight"] = self.config.algorithm.turn_weight
                                        metrics["actor/hybrid_final_weight"] = self.config.algorithm.final_weight
                                    
                                    # Log discount factor
                                    if hasattr(self.config.algorithm, 'turn_gamma'):
                                        metrics["actor/hybrid_turn_gamma"] = self.config.algorithm.turn_gamma
                                        metrics["actor/hybrid_final_gamma"] = self.config.algorithm.final_gamma

                    # Check preference loss configuration
                    pref_enabled = False
                    pref_config = None
                    
                    # Check both direct config and algorithm config
                    if hasattr(self.config, 'preference_loss'):
                        pref_enabled = self.config.preference_loss.get('enabled', False)
                        pref_config = self.config.preference_loss
                    elif hasattr(self.config, 'algorithm') and hasattr(self.config.algorithm, 'preference_loss'):
                        pref_enabled = self.config.algorithm.preference_loss.get('enabled', False)
                        pref_config = self.config.algorithm.preference_loss

                    # Compute preference loss if enabled
                    if pref_enabled:
                        try:
                            # Extract data needed for preference loss
                            # Data here is a dict merged from batch and non_tensor_batch
                            best_molecule_scores = data.get("best_molecule_scores", None)
                            group_ids = data.get("group_ids", None)
                            
                            if best_molecule_scores is not None and group_ids is not None:
                                # Get token_level_rewards - prioritize turn-level scores for preference loss
                                if "turn_level_rm_scores" in data and data["turn_level_rm_scores"] is not None:
                                    # Check if turn_level_rm_scores contains multi-turn information
                                    turn_level_rewards = data["turn_level_rm_scores"]
                                    non_zero_count = (turn_level_rewards != 0).sum().item()
                                    if non_zero_count > len(turn_level_rewards):  # More than batch_size non-zero values, indicating multiple turns
                                        token_level_rewards = turn_level_rewards
                                    else:
                                        token_level_rewards = data.get("rm_scores", torch.zeros_like(response_mask, dtype=torch.float32))
                                else:
                                    token_level_rewards = data.get("rm_scores", torch.zeros_like(response_mask, dtype=torch.float32))
                                
                                # Additional debug: check if rewards are at turn endings
                                if (token_level_rewards != 0).any():
                                    non_zero_indices = (token_level_rewards != 0).nonzero(as_tuple=True)
                                    # Print first few positions as examples
                                    if len(non_zero_indices[0]) > 0:
                                        sample_positions = list(zip(non_zero_indices[0][:5].tolist(), non_zero_indices[1][:5].tolist()))
                                
                                # Convert numpy arrays to tensors
                                if isinstance(best_molecule_scores, np.ndarray):
                                    best_molecule_scores = torch.from_numpy(best_molecule_scores).to(torch.cuda.current_device())
                                if isinstance(group_ids, np.ndarray):
                                    group_ids = group_ids  # Keep as numpy for now, converted in build_preference_pairs
                                
                                # Build preference pairs
                                preference_data = build_preference_pairs(
                                    token_level_rewards=token_level_rewards,
                                    response_mask=response_mask,
                                    best_molecule_scores=best_molecule_scores,
                                    group_ids=group_ids,
                                    mode=pref_config.get('mode', 'both'),
                                    inter_weight_cumulative=pref_config.get('inter_weight_cumulative', 0.5),
                                    max_intra_pairs_per_trajectory=pref_config.get('max_intra_pairs_per_trajectory', 6),
                                    max_inter_pairs_per_molecule=pref_config.get('max_inter_pairs_per_molecule', 20)
                                )
                                
                                # Count pairs for metrics
                                total_pairs = len(preference_data["pairs"])
                                intra_pairs = sum(1 for pair in preference_data["pairs"] if pair[2] != 0 or pair[3] != 0)
                                inter_pairs = total_pairs - intra_pairs
                                
                                if total_pairs > 0:
                                    # Compute preference loss
                                    pref_loss, pref_metrics = compute_lipo_lambda_loss(
                                        log_probs=log_prob,
                                        old_log_probs=old_log_prob,
                                        ref_log_probs=ref_log_prob,
                                        response_mask=response_mask,
                                        preference_data=preference_data,
                                        beta=pref_config.get('beta', 0.05),
                                        temperature=pref_config.get('temperature', 1.0),
                                        intra_weight=pref_config.get('intra_weight', 1.0),
                                        inter_weight=pref_config.get('inter_weight', 1.0),
                                        use_lambda_weights=pref_config.get('use_lambda_weights', False),
                                        intra_comparison_mode=pref_config.get('intra_comparison_mode', 'cumulative'),
                                        best_molecule_scores=best_molecule_scores
                                    )
                                    
                                    # Add to policy loss
                                    pref_weight = pref_config.get('weight', 0.1)
                                    policy_loss = policy_loss + pref_loss * pref_weight
                                    
                                    # Add metrics
                                    metrics.update(pref_metrics)
                            
                        except Exception as e:
                            print(f"Preference loss computation failed: {e}")

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if entropy_coeff != 0:
                        data["actor/entropy_loss"] = entropy_loss.detach().item()
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
