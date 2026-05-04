"""
Context Manager for the LLM agent.

This module handles context construction, prompt formatting, and
trajectory management for the reinforcement learning agent.
"""
from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    # Check if it's a Qwen series model (including fine-tuned versions)
    if ("qwen" in tokenizer.name_or_path.lower() or 
        "QMO" in tokenizer.name_or_path or 
        hasattr(tokenizer, 'im_start_id') or
        "<|im_start|>" in tokenizer.get_vocab()):
        try:
            special_token = tokenizer.encode("<|im_start|>")[0]
            reward_token = tokenizer.encode("<|im_end|>")[0]
        except Exception:
            # Fallback: get directly from vocab
            vocab = tokenizer.get_vocab()
            special_token = vocab.get("<|im_start|>", 151644)  # Qwen default value
            reward_token = vocab.get("<|im_end|>", 151645)    # Qwen default value
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False, need_turn_level_for_preference: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    
    Args:
        need_turn_level_for_preference: If True, also return turn-level scores for preference loss even when use_turn_scores=False
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    turn_level_score_tensor = None  # New: turn-level scores specifically for preference loss
    
    if use_turn_scores:
        # Original logic: each turn has separate rewards
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
        turn_level_score_tensor = score_tensor.clone()  # turn-level same as main tensor
    else:
        # Original logic: total score at the end
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
        
        # New: provide turn-level information for preference loss if needed
        if need_turn_level_for_preference and all_scores and len(all_scores[0]) > 1:
            turn_level_score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
            for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
                scores = torch.tensor(scores, dtype=torch.float32)
                turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
                reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
                # Set the last token of the rows where all positions are False to True
                reward_position[~reward_position.any(dim=-1), -1] = True
                turn_level_score_tensor[reward_position] = scores
            if "qwen" in tokenizer.name_or_path.lower():
                turn_level_score_tensor = turn_level_score_tensor.roll(shifts=1, dims=-1)
    
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token
    
    if turn_level_score_tensor is not None:
        turn_level_score_tensor = turn_level_score_tensor[:, 1:]  # remove the first token
    
    return score_tensor, loss_mask, response_mask, turn_level_score_tensor



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            
            # Process all environments normally
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            
            # Special handling for molecule optimization environment dynamic instruction generation
            if env_config.env_type == "molecule_opt":
                from ragen.env.molecule_opt.property_utils import parse_task_config
                
                # Dynamically set similarity threshold
                similarity_threshold = env_config_new.get("molecule_opt_similarity_threshold", 0.4)
                if env_config_new.get("env_config"):
                    env_config_new["env_config"]["initial_similarity_threshold"] = similarity_threshold  
                    env_config_new["env_config"]["final_similarity_threshold"] = similarity_threshold
                
                try:
                    task_description, similarity_threshold_str = parse_task_config(env_config_new)
                    
                    # Check if OOD experiment is enabled
                    ood_experiment = env_config_new.get("ood_experiment", False)
                    ood_style = env_config_new.get("ood_style", "original")
                    
                    if ood_experiment and ood_style != "original":
                        env_instruction = self._generate_ood_instruction(
                            ood_style, task_description, similarity_threshold_str
                        )
                    else:
                        # Original instruction
                        env_instruction = (
                            f"You are an expert medicinal chemist specializing in molecular optimization. "
                            f"Your task is to modify the given molecule to {task_description} while keeping "
                            f"structural changes as minimal as possible. The modified molecule should maintain "
                            f"a structural similarity of at least {similarity_threshold_str} with the original molecule."
                        )
                except Exception as e:
                    print(f"⚠️ Failed to parse molecule optimization task: {e}")
                    # Fall back to default instruction
                    env_instruction = (
                        "You are an expert medicinal chemist specializing in molecular optimization. "
                        "Your task is to modify the given molecule to optimize the target properties while "
                        "keeping structural changes as minimal as possible. The modified molecule should maintain "
                        "appropriate structural similarity with the original molecule."
                    )
            
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_values = [str(v) for k, v in env_config_new["action_lookup"].items()]
                action_lookup_str = "\nYour available actions are:\n" + ", ".join(action_values)
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \"{self.action_sep}\"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _generate_ood_instruction(self, ood_style: str, task_description: str, similarity_threshold_str: str) -> str:
        """
        Generate OOD instruction templates for testing model robustness
        
        Args:
            ood_style: Style of OOD instruction ('casual', 'technical', 'creative', 'minimal', 'verbose')
            task_description: Task description (e.g., "increase QED")
            similarity_threshold_str: Similarity threshold as string
            
        Returns:
            OOD instruction string
        """
        if ood_style == "casual":
            return (
                f"Hey! You're a chemist working on molecules. "
                f"Take this molecule and make it better for {task_description}. "
                f"Don't change it too much though - keep similarity above {similarity_threshold_str}. "
                f"Just give me the modified SMILES string."
            )
        elif ood_style == "technical":
            return (
                f"OBJECTIVE: Perform molecular structure optimization targeting {task_description}. "
                f"CONSTRAINTS: Maintain Tanimoto similarity coefficient ≥ {similarity_threshold_str}. "
                f"METHOD: Apply systematic structural modifications using computational chemistry principles. "
                f"OUTPUT: Optimized molecular structure in SMILES notation."
            )
        elif ood_style == "creative":
            return (
                f"Imagine you're designing the perfect molecule. "
                f"Transform the given structure to {task_description} while keeping its molecular identity. "
                f"Think creatively but stay within similarity bounds of {similarity_threshold_str}. "
                f"What would your ideal molecule look like?"
            )
        elif ood_style == "minimal":
            return (
                f"Modify molecule: {task_description}. "
                f"Similarity ≥ {similarity_threshold_str}."
            )
        elif ood_style == "verbose":
            return (
                f"You are an expert medicinal chemist with extensive experience in pharmaceutical research "
                f"and computational drug design. Your current assignment involves the systematic optimization "
                f"of molecular structures to {task_description}. Please carefully analyze the provided "
                f"molecular structure and propose strategic modifications that will enhance the desired "
                f"properties while maintaining structural integrity. It is absolutely critical that "
                f"any proposed modifications preserve a molecular similarity of at least {similarity_threshold_str} "
                f"when compared to the original structure using standard cheminformatics similarity metrics. "
                f"Please consider the impact of your modifications on drug-like properties, synthetic "
                f"accessibility, and potential off-target effects."
            )
        elif ood_style == "question_format":
            return (
                f"Given the following molecule, how would you modify it to {task_description}? "
                f"Please ensure your modification maintains similarity ≥ {similarity_threshold_str}. "
                f"What changes would you make?"
            )
        elif ood_style == "step_by_step":
            return (
                f"Step 1: Analyze the given molecule structure. "
                f"Step 2: Identify modification sites to {task_description}. "
                f"Step 3: Apply changes while ensuring similarity ≥ {similarity_threshold_str}. "
                f"Step 4: Provide the optimized SMILES structure."
            )
        else:
            # Fallback to original style
            return (
                f"You are an expert medicinal chemist specializing in molecular optimization. "
                f"Your task is to modify the given molecule to {task_description} while keeping "
                f"structural changes as minimal as possible. The modified molecule should maintain "
                f"a structural similarity of at least {similarity_threshold_str} with the original molecule."
            )

    def _smart_truncate_conversation(self, conversation_text: str, max_length: int) -> str:
        """
        Smart truncation of conversation history, keeping system prompt and recent conversation turns
        """
        # Try to split conversation by <|im_start|>
        parts = conversation_text.split('<|im_start|>')
        if len(parts) < 3:  # Need at least system, user, assistant
            # If split fails, use simple truncation
            tokens = self.tokenizer.encode(conversation_text)
            if len(tokens) > max_length:
                truncated_tokens = tokens[-max_length:]
                return self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            return conversation_text
        
        # Keep system prompt (usually the first part)
        system_part = '<|im_start|>' + parts[1] if len(parts) > 1 else ''
        
        # Keep conversation turns from back to front
        remaining_parts = parts[2:]  # Skip system part
        kept_parts = [system_part]
        
        # Keep from newest conversation backwards
        for i in range(len(remaining_parts) - 1, -1, -1):
            test_text = ''.join(kept_parts) + '<|im_start|>' + remaining_parts[i]
            test_tokens = self.tokenizer.encode(test_text)
            
            if len(test_tokens) <= max_length:
                kept_parts.insert(-1, '<|im_start|>' + remaining_parts[i])  # Insert after system
            else:
                break
        
        result = ''.join(kept_parts)
        
        # Ensure result is not too long
        final_tokens = self.tokenizer.encode(result)
        if len(final_tokens) > max_length:
            truncated_tokens = final_tokens[:max_length]
            result = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        
        return result

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            messages = [
                {"role": "system", "content": f"You are a molecular designer. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [Modified smiles string] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    

            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            if not prepare_for_update:
                if self.config.agent_proxy.enable_think:
                    text += "<think>" # force the LLM to think before answering
                else:
                    text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        # === Smart sequence length handling ===
        enable_smart_truncation = getattr(self.config.actor_rollout_ref.rollout, 'enable_smart_truncation', True)
        max_allowed_length = self.config.actor_rollout_ref.rollout.max_model_len
        max_prompt_length = getattr(self.config.data, 'max_prompt_length', max_allowed_length - 512)  # Reserve space for response
        
        # Check if any sequence exceeds length limit
        seq_lengths = attention_mask.sum(dim=-1)
        max_actual_length = seq_lengths.max().item()
        
        if max_actual_length > max_allowed_length and enable_smart_truncation:
            print(f"[Sequence Length Warning] Detected over-long sequence: {max_actual_length} > {max_allowed_length}, performing smart truncation")
            
            # Smart truncation - keep system prompt and recent conversation turns
            truncated_texts = []
            for i, text in enumerate(llm_input_texts):
                if seq_lengths[i] > max_prompt_length:
                    # Parse conversation structure, keep system prompt and recent conversations
                    truncated_text = self._smart_truncate_conversation(text, max_prompt_length)
                    truncated_texts.append(truncated_text)
                else:
                    truncated_texts.append(text)
            
            # Re-tokenize truncated text
            inputs = self.tokenizer(truncated_texts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=max_prompt_length)
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            print(f"[Sequence Length Processing] Max length after truncation: {attention_mask.sum(dim=-1).max().item()}")
        elif max_actual_length > max_allowed_length:
            print(f"[Sequence Length Warning] Detected over-long sequence: {max_actual_length} > {max_allowed_length}, but smart truncation is disabled")
            # If smart truncation is disabled, use simple truncation to avoid crashes
            inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=max_allowed_length)
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
            
            # Check if turn-level scores are needed for preference loss
            need_turn_level_for_preference = False
            if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'actor'):
                if hasattr(self.config.actor_rollout_ref.actor, 'preference_loss'):
                    pref_config = self.config.actor_rollout_ref.actor.preference_loss
                    if pref_config.get('enabled', False) and pref_config.get('mode', 'both') in ['intra', 'both']:
                        need_turn_level_for_preference = True
            
            score_tensor, loss_mask, response_mask, turn_level_score_tensor = get_masks_and_scores(
                input_ids, 
                self.tokenizer, 
                scores, 
                use_turn_scores=self.config.agent_proxy.use_turn_scores, 
                enable_response_mask=self.config.enable_response_mask,
                need_turn_level_for_preference=need_turn_level_for_preference
            )

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
            
            # Add turn-level scores for preference loss
            if turn_level_score_tensor is not None:
                llm_inputs.batch["turn_level_rm_scores"] = turn_level_score_tensor
            else:
                llm_inputs.batch["turn_level_rm_scores"] = normalized_score_tensor  # Use original as fallback
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        # Add best molecule scores for rank-based GRPO if available
        if prepare_for_update:
            best_molecule_scores = []
            for env_output in env_outputs:
                # Try to get best molecule score from the last turn's info
                best_score = None
                for turn in reversed(env_output.get("history", [])):
                    turn_info = turn.get("info", {})
                    if "current_best_score" in turn_info:
                        best_score = turn_info["current_best_score"]
                        break
                
                # Fallback: use cumulative reward if no best score found
                if best_score is None:
                    turn_rewards = [turn.get("reward", 0) for turn in env_output.get("history", []) if "reward" in turn]
                    best_score = sum(turn_rewards) if turn_rewards else 0.0
                
                best_molecule_scores.append(best_score)
            
            llm_inputs.non_tensor_batch["best_molecule_scores"] = np.array(best_molecule_scores, dtype=np.float32)
            
            # Collect preference data for LiPO-λ loss
            turn_level_rewards = []
            for env_output in env_outputs:
                # Extract turn-level rewards from history
                turn_rewards = []
                for turn in env_output.get("history", []):
                    if "reward" in turn:
                        turn_rewards.append(turn["reward"])
                
                # Pad to consistent length (or use a different strategy)
                max_turns = 10  # Configurable maximum turns
                while len(turn_rewards) < max_turns:
                    turn_rewards.append(0.0)
                turn_rewards = turn_rewards[:max_turns]  # Truncate if necessary
                
                turn_level_rewards.append(turn_rewards)
            
            llm_inputs.non_tensor_batch["turn_level_rewards"] = np.array(turn_level_rewards, dtype=np.float32)

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            # Skip non-zero metrics generation to reduce clutter
            # Only keep essential metrics without non-zero variants
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
