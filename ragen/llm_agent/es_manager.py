"""
Environment State Manager for the LLM agent.

This module manages multiple (kinds of) environments and handles:
- Environment initialization and reset
- Rollout cache management
- Evolving Skill Memory for experience-based learning
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
from ragen.llm_agent.evolving_skill_memory import EvolvingSkillMemory
register_resolvers()

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
    def __init__(self, config, mode: str = "train"):
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
        
        # Initialize Evolving Skill Memory for experience-based learning
        self.skill_memory = None
        if self.sys_config.get('evolving_skill_memory', {}).get('enabled', False):
            skill_config = self.sys_config.evolving_skill_memory
            self.skill_memory = EvolvingSkillMemory(
                max_size=skill_config.get('max_size', 10000),
                save_path=skill_config.get('save_path', f'skill_memory_{mode}.pkl'),
                min_score_delta=skill_config.get('min_score_delta', 0.01)
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

        # Add molecule data loading
        self.molecule_data_map = {}  # Use dictionary to map seed to molecules
        
        if any("MoleculeOptimization" in tag for tag in self.config.env_configs.tags):
            # Load training/validation dataset
            data_path = self.sys_config.data.train_file if self.mode == "train" else self.sys_config.data.val_file
            try:
                import pandas as pd
                df = pd.read_parquet(data_path)
                
                # Build seed to molecule mapping
                for idx, row in df.iterrows():
                    if "seed" in row and "smiles" in row:
                        # Original format: directly use seed field
                        self.molecule_data_map[row["seed"]] = row["smiles"]
                    elif "extra_info" in row and "seed" in row["extra_info"] and "smiles" in row:
                        # Extracted format: use seed from extra_info
                        self.molecule_data_map[row["extra_info"]["seed"]] = row["smiles"]
                    elif "smiles" in row:
                        # Zinc format: use row index as seed
                        self.molecule_data_map[idx] = row["smiles"]
                
                print(f"Loaded {len(self.molecule_data_map)} molecules with seed mapping from {data_path}")
            except Exception as e:
                print(f"Error loading molecule data: {e}")

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
                
                # Special handling for molecule optimization environment, pass complete configuration
                if env_class == "molecule_opt":
                    # Pass additional configuration parameters for molecule optimization environment
                    extra_env_config = {}
                    if hasattr(cfg_template, 'molecule_opt_task'):
                        extra_env_config['molecule_opt_task'] = cfg_template.molecule_opt_task
                    if hasattr(cfg_template, 'molecule_opt_similarity_threshold'):
                        extra_env_config['molecule_opt_similarity_threshold'] = cfg_template.molecule_opt_similarity_threshold
                    env_obj = REGISTERED_ENVS[env_class](env_config, extra_env_config)
                else:
                    env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def reset(self, seed: Optional[int] = None, global_step: int = 0, total_training_steps: int = 1):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed # get a random seed
        else:
            seed = 123
        expanded_seeds = _expand_seed(seed)

        for env_id, (current_seed, env_entry) in enumerate(zip(expanded_seeds, envs)):
            options = { # Prepare options dictionary
                "global_step": global_step,
                "total_training_steps": total_training_steps
            }

            # Add corresponding initial molecules for molecule optimization environment
            if hasattr(self, 'molecule_data_map') and len(self.molecule_data_map) > 0:
                if current_seed in self.molecule_data_map:
                    options["initial_smiles"] = self.molecule_data_map[current_seed]
                else:
                    # Use modulo operation to map random seed to available molecule index
                    all_seeds = list(self.molecule_data_map.keys())
                    seed_idx = current_seed % len(all_seeds)
                    mapped_seed = all_seeds[seed_idx]
                    options["initial_smiles"] = self.molecule_data_map[mapped_seed]

            try:
                # Call environment reset, pass seed and options containing global step information
                env_entry['env'].reset(seed=current_seed, options=options)
                env_entry['status'] = EnvStatus(seed=current_seed)

                if "initial_smiles" in options:
                    if not hasattr(self, 'env_smiles_map'):
                        self.env_smiles_map = {}
                    self.env_smiles_map[env_id] = options["initial_smiles"]

            except Exception as e:
                print(f"Error resetting environment {env_id} with seed {current_seed} and options {options}: {e}")

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        return rollout_cache

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
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
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
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
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

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment - Modified for molecule-level metrics"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # First pass: Update global best results for molecule optimization
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            
            # --- Update Global Best Results for Molecule Optimization --- 
            if hasattr(entry['env'], '_original_base_molecule_smiles'):
                initial_smiles_key = entry['env']._original_base_molecule_smiles # Use original SMILES as key
                episode_best_score = entry['env']._best_score
                episode_best_smiles = entry['env']._best_molecule_smiles
                initial_score = entry['env']._initial_score # Get initial score
                similarity_at_best = entry['env']._similarity_at_best_score # Get similarity at best score
                episode_oracle_calls = entry['env']._oracle_calls_this_episode # Get real oracle calls from this episode (no cache)
                episode_total_oracle_calls = entry['env']._total_oracle_calls_this_episode # Get total oracle calls from this episode (including cache)

                # --- Update Cumulative Oracle Call Trackers (accumulate oracle calls from all paths) --- 
                if initial_smiles_key is not None:
                     # Real oracle calls (no cache hits)
                     current_calls = self.oracle_call_budget_tracker.get(initial_smiles_key, 0)
                     self.oracle_call_budget_tracker[initial_smiles_key] = current_calls + episode_oracle_calls
                     
                     # Total oracle calls (including cache hits)
                     current_total_calls = self.total_oracle_call_budget_tracker.get(initial_smiles_key, 0)
                     self.total_oracle_call_budget_tracker[initial_smiles_key] = current_total_calls + episode_total_oracle_calls
                # -------------------------------------------

                if initial_smiles_key is not None: # Ensure we have a key
                    current_global_best = self.global_best_results.get(initial_smiles_key, None)
                    
                    # Get current episode's best score (for comparison)
                    env = entry['env']
                    current_episode_best_score = episode_best_score
                    
                    # Check if this is a multi-objective task
                    is_multi_objective = hasattr(env, 'molecule_opt_task') and '+' in env.molecule_opt_task
                    
                    # Determine comparison criteria
                    is_better = False
                    if current_global_best is None:
                        is_better = True
                    else:
                        if is_multi_objective:
                            # Multi-objective task: compare weighted_score
                            current_weighted_score = getattr(env, '_best_weighted_score', float('-inf'))
                            stored_weighted_score = current_global_best.get('weighted_score', float('-inf'))
                            is_better = current_weighted_score > stored_weighted_score
                        else:
                            # Single-objective task: compare based on optimization direction
                            is_sa_task = hasattr(env, 'minimize_target') and env.minimize_target
                            if is_sa_task:
                                # For SA: smaller scores are better
                                is_better = current_episode_best_score < current_global_best['best_score']
                            else:
                                # For other properties: larger scores are better
                                is_better = current_episode_best_score > current_global_best['best_score']
                    
                    # Update if new or if the new episode best score is better than the stored global best
                    if is_better:
                        # print(f"[DEBUG] Updating global best for {initial_smiles_key}: {episode_best_score:.4f} (was {current_global_best['best_score'] if current_global_best else 'None'})")
                        
                        # Build base result data
                        result_data = {
                            'initial_smiles': initial_smiles_key, # Store explicitly for JSON clarity
                            'initial_score': initial_score,
                            'best_score': episode_best_score,
                            'best_smiles': episode_best_smiles,
                            'similarity_at_best': similarity_at_best, # Store the similarity
                            # Store the CUMULATIVE calls up to the point this new best was found
                            'total_oracle_calls': self.oracle_call_budget_tracker.get(initial_smiles_key, 0),
                            'success': True  # Mark as successful since we found a better score
                        }
                        
                        # For multi-objective tasks, get data directly from environment
                        env = entry['env']
                        if is_multi_objective:
                            # Get best data directly from environment, not rebuild from history
                            if hasattr(env, '_best_weighted_score'):
                                result_data['weighted_score'] = env._best_weighted_score
                            
                            if hasattr(env, '_best_individual_properties') and env._best_individual_properties:
                                result_data['individual_new_properties'] = env._best_individual_properties.copy()
                            
                            if hasattr(env, '_best_global_improvements') and env._best_global_improvements:
                                result_data['individual_improvements'] = env._best_global_improvements.copy()
                            
                            # Get original property values
                            if hasattr(env, '_original_base_molecule_smiles'):
                                task_properties = env.molecule_opt_task.split('+')
                                try:
                                    original_properties = env.oracle.evaluate_specific_properties(
                                        env._original_base_molecule_smiles, task_properties)
                                    result_data['individual_old_properties'] = original_properties
                                    result_data['task_properties'] = task_properties
                                except Exception as e:
                                    print(f"Warning: Failed to get original properties for {initial_smiles_key}: {e}")
                            
                            # For multi-objective tasks, use weighted_score as best_score for comparison display
                            if 'weighted_score' in result_data:
                                result_data['best_score'] = result_data['weighted_score']
                        
                        self.global_best_results[initial_smiles_key] = result_data
                    # --- Ensure existing entries also get updated total_oracle_calls for JSON --- 
                    elif current_global_best is not None:
                         # If not updating best score, still update the total call count in the record for the JSON snapshot
                         self.global_best_results[initial_smiles_key]['total_oracle_calls'] = self.oracle_call_budget_tracker.get(initial_smiles_key, 0)
        
        # Second pass: Calculate molecule-level metrics
        molecule_metrics = self._calculate_molecule_level_metrics(envs, rollout_cache)
        
        # Calculate multi-objective metrics
        multi_objective_metrics = self._calculate_multi_objective_metrics(molecule_metrics)
        
        # Third pass: Assign molecule-level metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            
            # Get molecule ID and related info
            molecule_id = entry['env_id'] // self.config.group_size if hasattr(self.config, 'group_size') else entry['env_id']
            initial_smiles = None
            if hasattr(entry['env'], '_original_base_molecule_smiles'):
                initial_smiles = entry['env']._original_base_molecule_smiles
            
            # Use molecule-level metrics instead of path-level
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
                # Fallback to path-level metrics if molecule-level not available
                env_metric = {
                    'success': float(status.terminated and (not status.truncated)),
                    'num_actions': status.num_actions,
                }
                
                # Add molecule-specific metrics if available
                if hasattr(entry['env'], '_original_base_molecule_smiles'):
                    initial_smiles_key = entry['env']._original_base_molecule_smiles
                    if initial_smiles_key in self.global_best_results:
                        env_metric['current_global_best_score_for_mol'] = self.global_best_results[initial_smiles_key]['best_score']

            # Process custom metrics (turn-level info) - collect for molecule-level aggregation
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    # Skip debug/internal fields (starting with underscore) and other non-metric fields
                    if (k == 'success' or 
                        k in ['final_smiles', 'action_received', 'initial_smiles', 'current_best_smiles'] or
                        k.startswith('_')):  # Skip fields starting with underscore (debug/internal fields)
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    try:
                        custom_metric[k].append(float(v))
                    except (ValueError, TypeError):
                        # Silently skip non-numeric values to reduce log noise
                        pass
            
            # Use molecule-level aggregated metrics when available
            if initial_smiles and initial_smiles in molecule_metrics:
                mol_metrics = molecule_metrics[initial_smiles]
                # For molecule-level: use aggregated custom metrics from all paths
                for k, v in custom_metric.items():
                    if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                        # For molecule-level: use best values across all paths of this molecule
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
                # Fallback to path-level metrics
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
        
        # Add multi-objective metrics to the first cache (they are batch-level metrics)
        if rollout_cache and multi_objective_metrics:
            # Find a MoleculeOptimization cache to add batch-level metrics
            for cache in rollout_cache:
                if 'metrics' in cache and any('MoleculeOptimization' in key for key in cache['metrics'].keys()):
                    # Add multi-objective metrics with proper prefixes
                    for metric_key, metric_value in multi_objective_metrics.items():
                        prefixed_key = f"MoleculeOptimization/{metric_key}"
                        if prefixed_key not in cache['metrics']:
                            cache['metrics'][prefixed_key] = metric_value
                    break
                    
        return rollout_cache

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
        
        # Group by molecule (initial SMILES)
        for entry, cache in zip(envs, rollout_cache):
            if not hasattr(entry['env'], '_original_base_molecule_smiles'):
                continue
                
            initial_smiles = entry['env']._original_base_molecule_smiles
            status = entry['status']
            
            # Get path-level metrics
            path_success = bool(status.terminated and (not status.truncated))
            path_score = getattr(entry['env'], '_best_score', 0.0)
            path_similarity = getattr(entry['env'], '_similarity_at_best_score', 0.0)
            path_num_actions = status.num_actions
            initial_score = getattr(entry['env'], '_initial_score', 0.0)
            
            # Collect custom metrics from this path
            path_custom_metrics = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    # Skip debug/internal fields (starting with underscore) and other non-metric fields
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
            
            # Update molecule-level aggregates
            mol_data = molecule_data[initial_smiles]
            mol_data['paths'].append({
                'success': path_success,
                'score': path_score,
                'similarity': path_similarity,
                'num_actions': path_num_actions,
                'custom_metrics': path_custom_metrics
            })
            
            # Set initial score (should be same across all paths)
            if mol_data['initial_score'] is None:
                mol_data['initial_score'] = initial_score
            
            # Update best values across all paths
            if path_score > mol_data['best_score']:
                mol_data['best_score'] = path_score
                mol_data['best_similarity'] = path_similarity
                mol_data['best_num_actions'] = path_num_actions
            
            # Molecule is successful if ANY path is successful
            if path_success:
                mol_data['success'] = True
            
            # Calculate improvement
            mol_data['improvement'] = mol_data['best_score'] - mol_data['initial_score']
            
            # Aggregate custom metrics across all paths for this molecule
            for k, v_list in path_custom_metrics.items():
                mol_data['custom_metrics'][k].extend(v_list)
        
        # Calculate molecule-level custom metrics
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
        
        # Get supported properties
        all_properties = get_supported_properties()
        
        # Initialize statistics data
        property_stats = {prop: {
            'improvements': [],
            'initial_scores': [],
            'best_scores': [],
            'success_count': 0,
            'total_count': 0,
            'above_threshold_count': 0
        } for prop in all_properties}
        
        # QED special statistics (success rate >0.9)
        qed_above_09_count = 0
        total_molecules = len(molecule_data)
        all_targets_success_count = 0
        single_target_success_count = 0
        
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
                property_keys = ['qed', 'logp', 'sa', 'jnk3', 'gsk3b', 'drd2']
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
            
            # Statistics for each property
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
                    
                    # Basic statistics
                    property_stats[prop]['improvements'].append(improvement)
                    property_stats[prop]['initial_scores'].append(initial_score)
                    property_stats[prop]['best_scores'].append(best_score)
                    property_stats[prop]['total_count'] += 1
                    
                    # Success rate statistics (using unified threshold criteria)
                    from ragen.env.molecule_opt.property_utils import PROPERTY_SUCCESS_THRESHOLDS
                    
                    prop_success_status = success_status.get(prop, {})
                    
                    # Use unified threshold to determine success
                    is_success = False
                    if prop in PROPERTY_SUCCESS_THRESHOLDS:
                        threshold = PROPERTY_SUCCESS_THRESHOLDS[prop]["threshold"]
                        direction = PROPERTY_SUCCESS_THRESHOLDS[prop]["direction"]
                        
                        if direction == "increase":
                            is_success = improvement >= threshold
                        else:  # direction == "decrease" (SA)
                            is_success = improvement <= -threshold
                    else:
                        # Properties without defined thresholds, any improvement counts as success
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
            elif len(task_properties) == 1 and all_success:
                single_target_success_count += 1
                
            if 'qed' in task_properties:
                # Use best_score calculated in the loop
                qed_best_score = property_stats['qed']['best_scores'][-1] if property_stats['qed']['best_scores'] else 0
                if qed_best_score > 0.9:
                    qed_above_09_count += 1
            elif 'qed' in new_properties and new_properties['qed'] > 0.9:
                qed_above_09_count += 1
        
        # Calculate summary metrics
        multi_objective_metrics = {}
        
        # Key statistics for each property (remove unnecessary metrics)
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

    def get_average_global_best_score(self):
        """Calculate the average of the best scores achieved across all tracked initial molecules."""
        if not self.global_best_results:
            return 0.0 # Or handle as needed, e.g., return None or float('nan')
        
        tracked_scores = [data['best_score'] for data in self.global_best_results.values()]
        if not tracked_scores:
             return 0.0
             
        return sum(tracked_scores) / len(tracked_scores)

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
    
    def get_total_unique_oracle_calls(self):
        """Get total unique oracle calls by accessing oracle cache size."""
        total_unique_calls = 0
        
        # Try to get the actual oracle call count from the environment
        for env_entry in self.envs:
            if env_entry and 'env' in env_entry:
                env = env_entry['env']
                if hasattr(env, 'oracle') and hasattr(env.oracle, 'calls'):
                    # Oracle's calls attribute records actual non-cached call counts
                    total_unique_calls = max(total_unique_calls, env.oracle.calls)
                    break  # All environments share the same oracle instance
        
        return total_unique_calls

    def get_average_improvement(self):
        """Calculate the average improvement: best_score - initial_score across all tracked molecules."""
        if not self.global_best_results:
            return 0.0
        
        improvements = []
        for data in self.global_best_results.values():
            if 'best_score' in data and 'initial_score' in data:
                improvement = data['best_score'] - data['initial_score']
                improvements.append(improvement)
        
        if not improvements:
            return 0.0
            
        return sum(improvements) / len(improvements)
    
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
    
    def get_skill_memory_stats(self):
        """Get statistics about the Evolving Skill Memory"""
        if self.skill_memory is None:
            return {}
        return self.skill_memory.get_stats()

    def save_skill_memory(self):
        """Save the Evolving Skill Memory to disk"""
        if self.skill_memory is not None:
            self.skill_memory.save()
    
    def get_current_batch_similarity_stats(self):
        """Calculate similarity statistics for the current batch of trajectories."""
        if not self.rollout_cache:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'below_threshold_ratio': 0.0}
        
        all_similarities = []
        similarity_threshold = 0.4  # Current fixed threshold used
        below_threshold_count = 0
        
        for cache in self.rollout_cache:
            if cache.get('history'):
                for turn in cache['history'][:-1]:  # Exclude last observation
                    info = turn.get('info', {})
                    if 'similarity' in info:
                        try:
                            sim = float(info['similarity'])
                            all_similarities.append(sim)
                            if sim < similarity_threshold:
                                below_threshold_count += 1
                        except (ValueError, TypeError):
                            pass
        
        if not all_similarities:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'below_threshold_ratio': 0.0}
        
        return {
            'mean': sum(all_similarities) / len(all_similarities),
            'min': min(all_similarities),
            'max': max(all_similarities),
            'count': len(all_similarities),
            'below_threshold_ratio': below_threshold_count / len(all_similarities)
        }
    
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
                # Check if this trajectory has all targets success
                if 'history' in cache:
                    # Find the last turn containing multi_objective_info
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
            # Single-objective task: use original logic
            success_count = 0
            total_count = 0
            
            for cache in self.rollout_cache:
                if 'metrics' in cache:
                    # Get success information from metrics
                    success = cache['metrics'].get(f"{cache.get('tag', 'Unknown')}/success", 0.0)
                    success_count += success
                    total_count += 1
            
            if total_count == 0:
                return 0.0
            
            return success_count / total_count
    
    def get_current_batch_molecule_success_rate(self):
        """Calculate molecule-level success rate for the current batch."""
        if not self.rollout_cache:
            return 0.0
        
        # Calculate molecule-level metrics using the existing method
        molecule_metrics = self._calculate_molecule_level_metrics(self.envs, self.rollout_cache)
        
        if not molecule_metrics:
            return 0.0
        
        successful_molecules = sum(1 for mol_data in molecule_metrics.values() if mol_data['success'])
        total_molecules = len(molecule_metrics)
        
        return successful_molecules / total_molecules if total_molecules > 0 else 0.0

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
            # Fallback for other types
            return str(state)
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
