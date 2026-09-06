import gymnasium as gym
from typing import Tuple, Any, Dict, Optional, List
import requests
import logging

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not found. Molecule modification and similarity calculation will not work.")
from ragen.env.base import BaseLanguageBasedEnv
from .config import MoleculeEnvConfig
from .oracle import MoleculeOracle
from .property_utils import (
    parse_task_config,
    validate_task,
    calculate_multi_objective_reward,
    generate_complete_multi_objective_guidance,
    get_optimization_directions,
    PROPERTY_SUCCESS_THRESHOLDS,
    PROPERTY_DIFFICULTY_WEIGHTS
)

from ragen.llm_agent.evolving_skill_memory import (
    EvolvingSkillMemory,
    detect_functional_groups,
    format_hints_for_prompt,
)

PROPERTY_MAPPING = {
    "qed": {
        "description": "drug-likeness (QED)"
    },
    "logp": {
        "description": "LogP value"
    },
    "jnk3": {
        "description": "JNK3 inhibition activity"
    },
    "drd2": {
        "description": "DRD2 inhibition activity"
    },
    "sa": {
        "description": "synthetic accessibility (SA score, lower is better)"
    },
    "penalized_logp": {
        "description": "Penalized LogP (LogP - SA score, higher is better)"
    },
    "mr": {
        "description": "molecular refractivity (MR)"
    },
    "logp_qed": {
        "description": "multi-objective LogP + QED combination (weighted average)"
    }
}

class MoleculeEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[MoleculeEnvConfig] = None, env_config: Optional[dict] = None):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for MoleculeEnv. Please install it (`pip install rdkit`).")

        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else MoleculeEnvConfig()

        self.env_config = env_config or {}
        self._parse_task_config()

        task_properties = self.molecule_opt_task.split('+')
        if len(task_properties) == 1:
            self.target_property = task_properties[0]
        else:
            self.target_property = task_properties[0]

        self.optimization_directions = get_optimization_directions(self.molecule_opt_task)
        self.minimize_target = self.optimization_directions.get(self.target_property, "maximize") == "minimize"

        if not hasattr(self.config, 'final_similarity_threshold'):
            raise ValueError("Molecule environment requires final_similarity_threshold")

        self._original_base_molecule_smiles: Optional[str] = None
        self._current_molecule_smiles: Optional[str] = None
        self._original_mol_object = None
        self._current_mol_object = None
        self._current_step: int = 0
        self._initial_score: float = 0.0
        self._global_training_step: int = 0

        self.render_mode = self.config.render_mode
        self.render_cache: Optional[str] = None

        oracle_cfg = self.config.oracle_config
        multi_objective_weights = getattr(self.config, 'multi_objective_weights', None)
        self.oracle = MoleculeOracle(
            target_property=self.target_property,
            cache_file=oracle_cfg.get("cache_file"),
            multi_objective_weights=multi_objective_weights
        )
        self.retriever_config = self.config.retriever

        self._best_molecule_smiles: Optional[str] = None
        self._best_mol_object = None
        self._best_score: float = float('inf') if self.minimize_target else float('-inf')
        self._best_step: int = 0
        self._similarity_at_best_score: float = 1.0
        self._consecutive_failures: int = 0
        self._oracle_calls_this_episode: int = 0
        self._total_oracle_calls_this_episode: int = 0
        self._best_weighted_score: float = float('-inf')
        self._best_individual_properties: Dict = {}
        self._best_global_improvements: Dict = {}
        self._candidate_property_cache: Dict[str, Dict[str, float]] = {}

        # Track consecutive outcomes that make memory guidance useful.
        self._consecutive_no_improvement: int = 0
        self._consecutive_rollbacks: int = 0
        self._memory_hint_trigger_count: int = 0
        self._pending_hint: str = ""
        self._submitted_smiles_set: set = set()

        self._skill_memory: Optional[EvolvingSkillMemory] = None
        self._skill_retrieval_config: dict = {}

    def _parse_task_config(self):
        """Parse the task description, similarity threshold, and optimization direction."""
        task = self.env_config.get('molecule_opt_task')
        if not isinstance(task, str) or not task.strip():
            raise ValueError("molecule_opt_task must be configured")
        task = task.strip()
        similarity_threshold = self.env_config.get(
            'molecule_opt_similarity_threshold', 0.4
        )
        if not validate_task(task):
            raise ValueError(
                f"Invalid molecule optimization task {task!r}. Expected a supported "
                "property or a '+'-separated combination of supported properties."
            )
        if similarity_threshold is None:
            raise ValueError("Molecule similarity threshold must be configured")

        task_description, similarity_threshold_str = parse_task_config(self.env_config)
        self.task_description = task_description
        self.similarity_threshold_str = similarity_threshold_str
        self.molecule_opt_task = task

    def replace_instruction_variables(self, instruction: str) -> str:
        """Replace variable placeholders in an instruction."""
        if instruction is None:
            return ""

        instruction = instruction.replace("${task_description}", self.task_description)
        instruction = instruction.replace("${similarity_threshold}", self.similarity_threshold_str)

        return instruction

    def _calculate_multi_objective_reward(
        self,
        old_smiles: str,
        new_smiles: str,
        old_properties: Optional[Dict[str, float]] = None,
        new_properties: Optional[Dict[str, float]] = None,
        original_properties: Optional[Dict[str, float]] = None,
    ) -> tuple[float, dict]:
        """
        Calculate the reward for a multi-objective task.

        Args:
            old_smiles: Molecular SMILES before modification.
            new_smiles: Molecular SMILES after modification.

        Returns:
            tuple: (reward, reward_info)
        """
        task_properties = self.molecule_opt_task.split('+')

        if len(task_properties) == 1:
            old_score = self.oracle(old_smiles)
            new_score = self.oracle(new_smiles)
            improvement = new_score - old_score

            if self.minimize_target:
                if improvement < 0:
                    reward = -improvement * 5
                elif improvement > 0:
                    reward = -improvement
                else:
                    reward = 0
            else:
                if improvement > 0:
                    reward = improvement * 5
                elif improvement < 0:
                    reward = improvement
                else:
                    reward = 0

            reward_info = {
                "improvement": improvement,
                "old_score": old_score,
                "new_score": new_score,
                "is_multi_objective": False
            }

            return reward, reward_info

        if old_properties is None:
            old_properties = self.oracle.evaluate_specific_properties(
                old_smiles, task_properties
            )
        if new_properties is None:
            new_properties = self.oracle.evaluate_specific_properties(
                new_smiles, task_properties
            )
        if original_properties is None and hasattr(
            self, '_original_base_molecule_smiles'
        ):
            original_properties = self.oracle.evaluate_specific_properties(
                self._original_base_molecule_smiles, task_properties
            )
        elif original_properties is None:
            logger.warning("Missing _original_base_molecule_smiles in reward calculation, using old_properties as fallback")
            original_properties = old_properties

        step_improvements = {}
        for prop in task_properties:
            old_value = old_properties.get(prop, 0.0)
            new_value = new_properties.get(prop, 0.0)
            step_improvements[prop] = new_value - old_value

        global_improvements = {}
        for prop in task_properties:
            original_value = original_properties.get(prop, 0.0)
            new_value = new_properties.get(prop, 0.0)
            global_improvements[prop] = new_value - original_value

        reward_mode = getattr(self.config, 'multi_objective_reward_mode', 'cumulative')
        if reward_mode not in ("cumulative", "stepwise"):
            logger.warning(f"Invalid multi_objective_reward_mode '{reward_mode}', defaulting to 'cumulative'")
            reward_mode = "cumulative"

        if reward_mode == "cumulative":
            improvements_for_reward = global_improvements
        else:
            improvements_for_reward = step_improvements

        total_reward, reward_info = calculate_multi_objective_reward(
            improvements_for_reward, task_properties, self.optimization_directions)

        weighted_score = 0.0
        from ragen.env.molecule_opt.property_utils import PROPERTY_DIFFICULTY_WEIGHTS
        for prop in task_properties:
            weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
            improvement = global_improvements.get(prop, 0.0)
            if self.optimization_directions.get(prop, "maximize") == "minimize":
                weighted_score += (-improvement) * weight
            else:
                weighted_score += improvement * weight

        reward_info.update({
            "old_properties": old_properties,
            "new_properties": new_properties,
            "original_properties": original_properties,
            "step_improvements": step_improvements,
            "global_improvements": global_improvements,
            "improvements": global_improvements,
            "weighted_score": weighted_score,
            "is_multi_objective": True,
            "task_properties": task_properties,
            "reward_mode": reward_mode
        })

        return total_reward, reward_info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict]:
        """
        Resets the environment using an initial SMILES string provided via 'options'.
        Also receives global training step information via 'options'.
        Returns a *minimal* initial observation.
        """
        # BaseLanguageBasedEnv.reset shadows Gym's seeding method in the MRO.
        gym.Env.reset(self, seed=seed)
        options = options or {}

        self._global_training_step = options.get("global_step", 0)

        initial_smiles_from_data = options.get("initial_smiles")
        if initial_smiles_from_data is None:
            raise ValueError(
                "MoleculeEnv.reset() requires an initial_smiles value from the dataset"
            )

        try:
            mol = Chem.MolFromSmiles(initial_smiles_from_data)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse initial SMILES {initial_smiles_from_data!r}"
            ) from exc
        if mol is None:
            raise ValueError(f"Invalid initial SMILES: {initial_smiles_from_data!r}")
        self._original_base_molecule_smiles = Chem.MolToSmiles(mol)
        self._original_mol_object = Chem.MolFromSmiles(
            self._original_base_molecule_smiles
        )

        starting_smiles = options.get("starting_smiles", self._original_base_molecule_smiles)
        starting_mol = Chem.MolFromSmiles(starting_smiles)
        if starting_mol is None:
            raise ValueError(f"Invalid starting SMILES provided: {starting_smiles}")
        self._current_molecule_smiles = Chem.MolToSmiles(starting_mol, canonical=True)
        self._current_mol_object = Chem.MolFromSmiles(self._current_molecule_smiles)

        starting_similarity = self._calculate_similarity(
            self._current_mol_object, self._original_mol_object
        )
        if starting_similarity < self.config.final_similarity_threshold:
            raise ValueError(
                "Starting molecule does not satisfy the lead similarity constraint: "
                f"{starting_similarity:.3f} < {self.config.final_similarity_threshold:.3f}"
            )

        self._current_step = 0
        task_properties = self.molecule_opt_task.split('+')
        original_all_scores = dict(options.get("initial_properties") or {})
        if any(prop not in original_all_scores for prop in task_properties):
            original_all_scores = self.oracle.evaluate_specific_properties(
                self._original_base_molecule_smiles, task_properties
            )
        starting_all_scores = dict(options.get("starting_properties") or {})
        if any(prop not in starting_all_scores for prop in task_properties):
            starting_all_scores = self.oracle.evaluate_specific_properties(
                self._current_molecule_smiles, task_properties
            )
        self._original_properties = {
            prop: float(original_all_scores[prop]) for prop in task_properties
        }
        self._current_properties = {
            prop: float(starting_all_scores[prop]) for prop in task_properties
        }

        self._initial_score = float(original_all_scores[self.target_property])
        starting_score = float(starting_all_scores[self.target_property])
        self._oracle_calls_this_episode = 0
        self._total_oracle_calls_this_episode = 0

        self._best_molecule_smiles = self._current_molecule_smiles
        self._best_mol_object = self._current_mol_object
        self._best_score = starting_score
        self._best_step = 0
        self._similarity_at_best_score = starting_similarity
        self._best_individual_properties = dict(starting_all_scores)
        self._best_global_improvements = {
            prop: starting_all_scores[prop] - original_all_scores[prop]
            for prop in task_properties
        }
        optimization_directions = getattr(
            self,
            'optimization_directions',
            get_optimization_directions(self.molecule_opt_task),
        )
        self._best_weighted_score = sum(
            PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
            * (
                -self._best_global_improvements[prop]
                if optimization_directions.get(prop, "maximize") == "minimize"
                else self._best_global_improvements[prop]
            )
            for prop in task_properties
        )
        if len(task_properties) > 1:
            self._best_score = self._best_weighted_score
        self._consecutive_failures = 0

        self._consecutive_no_improvement = 0
        self._consecutive_rollbacks = 0
        self._memory_hint_trigger_count = 0
        self._pending_hint = ""
        self._submitted_smiles_set = set()

        retrieval_results = []
        is_multi_objective = len(task_properties) > 1
        memory_budget_available = (
            self._memory_hint_trigger_count < self.retriever_config.max_triggers
        )
        static_should_inject = (
            memory_budget_available
            and self.retriever_config.trigger_mode == "always"
        )

        skill_trigger_mode = getattr(self, '_skill_hints_trigger_mode', 'on_stuck')
        min_step_for_hints = getattr(self, '_min_step_for_hints', 0)
        skill_memory = getattr(self, '_skill_memory', None)
        skill_should_inject = (
            memory_budget_available
            and skill_memory is not None
            and self._global_training_step >= min_step_for_hints
            and skill_trigger_mode == "always"
            and skill_memory.has_task(self.molecule_opt_task)
        )

        if static_should_inject and skill_should_inject:
            selected_source = self.np_random.choice(['static', 'skill'])
            static_should_inject = selected_source == 'static'
            skill_should_inject = selected_source == 'skill'

        if static_should_inject:
            # Selecting a source consumes one shared attempt, including empty
            # results or request failures.
            self._memory_hint_trigger_count += 1
            if is_multi_objective:
                retrieval_results = self._retrieve_similar_molecules_multi_objective(
                    query_smiles=self._current_molecule_smiles,
                    original_smiles=self._original_base_molecule_smiles,
                    task_properties=task_properties,
                    original_properties=original_all_scores,
                    similarity_threshold=self.config.final_similarity_threshold,
                    hint_topk=getattr(self.retriever_config, 'hint_topk', 3),
                    query_properties=starting_all_scores
                )
            else:
                retrieval_results = self._retrieve_similar_molecules(
                    query_smiles=self._current_molecule_smiles,
                    original_smiles=self._original_base_molecule_smiles,
                    similarity_threshold=self.config.final_similarity_threshold,
                    sort_by=self.target_property,
                    original_property_value=starting_score
                )

        initial_guidance = generate_complete_multi_objective_guidance(
            original_scores={},
            new_scores=original_all_scores,
            task_properties=task_properties,
            reward_info={},
            is_initial=True,
        )
        observation = f"Original Molecule: {self._original_base_molecule_smiles}\n{initial_guidance}"
        if self._current_molecule_smiles != self._original_base_molecule_smiles:
            current_scores = ", ".join(
                f"{prop}: {starting_all_scores[prop]:.3f}" for prop in task_properties
            )
            observation += (
                f"\nCurrent Molecule: {self._current_molecule_smiles}"
                f"\nCurrent molecule properties: {current_scores}"
            )

        if retrieval_results:
            if is_multi_objective:
                retrieval_text = self._format_retrieval_as_hint_multi_objective(
                    retrieval_results, task_properties, original_all_scores
                )
            else:
                retrieval_text = self._format_retrieval_results(retrieval_results)
            observation += retrieval_text

        if skill_should_inject:
            # Empty attempts consume the selected source's budget.
            self._memory_hint_trigger_count += 1
            skill_hints = self.get_skill_hints(self._current_molecule_smiles)

            if skill_hints:
                observation += f"\n\n{skill_hints}"

        self.render_cache = observation

        info = {
            "initial_smiles": self._original_base_molecule_smiles,
            "initial_score": self._initial_score,
            "starting_smiles": self._current_molecule_smiles,
            "starting_score": starting_score,
        }
        return observation, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        """Apply a molecular modification."""
        terminated = False
        truncated = False
        reward = 0.0
        is_step_successful = False

        current_properties = getattr(self, '_current_properties', None)
        if current_properties is None or self.target_property not in current_properties:
            current_properties = {
                self.target_property: self.oracle(self._current_molecule_smiles)
            }
            self._current_properties = current_properties
        score_before_modification = current_properties[self.target_property]

        current_similarity_threshold = self.config.final_similarity_threshold

        info = {
            "action_received": action,
            "similarity": 1.0,
            "score_before": score_before_modification,
            "score_after": score_before_modification,
            "is_valid_modification": False,
            "is_similar_enough": True,
            "final_smiles": self._current_molecule_smiles,
            "current_similarity_threshold": current_similarity_threshold,
            "oracle_calls_this_step": 0,
            "total_oracle_calls_this_step": 0,
            "oracle_calls_this_episode": self._oracle_calls_this_episode,
            "total_oracle_calls_this_episode": self._total_oracle_calls_this_episode,
            "current_best_score": self._best_score,
            "current_best_smiles": self._best_molecule_smiles
        }

        try:
            next_mol_object = Chem.MolFromSmiles(action)
            if next_mol_object is None:
                is_step_successful = False
                reward = -1
                info["is_valid_modification"] = False
            else:
                next_molecule_smiles = Chem.MolToSmiles(next_mol_object, canonical=True)
                current_canonical_smiles = Chem.MolToSmiles(self._current_mol_object, canonical=True)

                if (getattr(self.config, 'duplicate_detection_enabled', True) and
                    next_molecule_smiles in self._submitted_smiles_set):
                    is_step_successful = False
                    duplicate_penalty = getattr(self.config, 'duplicate_penalty', -0.5)
                    reward = duplicate_penalty
                    info["is_valid_modification"] = False
                    info["duplicate_submission"] = True
                    info["duplicate_penalty_applied"] = duplicate_penalty
                elif next_molecule_smiles == current_canonical_smiles:
                    is_step_successful = False
                    reward = -0.3
                    info["is_valid_modification"] = True
                    info["no_change_detected_in_step"] = True
                else:
                    similarity = self._calculate_similarity(next_mol_object, self._original_mol_object)
                    info["similarity"] = similarity
                    task_properties = self.molecule_opt_task.split('+')

                    if similarity < current_similarity_threshold:
                        is_step_successful = False
                        info["is_similar_enough"] = False
                        reward = -2 * (current_similarity_threshold - similarity)
                        info["is_valid_modification"] = True
                    else:
                        constraint_valid, constraint_message = self._check_structural_constraints(next_mol_object)
                        if not constraint_valid:
                            is_step_successful = False
                            reward = -1.0
                            info.update({
                                "is_valid_modification": True,
                                "is_similar_enough": True,
                                "violates_structural_constraints": True,
                                "_constraint_violation_message": constraint_message
                            })
                        else:
                            if not hasattr(self, '_candidate_property_cache'):
                                self._candidate_property_cache = {}
                            cached_properties = self._candidate_property_cache.get(
                                next_molecule_smiles
                            )
                            if cached_properties is not None and all(
                                prop in cached_properties for prop in task_properties
                            ):
                                next_properties = dict(cached_properties)
                                next_score = next_properties[self.target_property]
                                was_cached = True
                            else:
                                next_score, was_cached = self.oracle.evaluate_with_cache_info(
                                    next_molecule_smiles
                                )
                                next_properties = {self.target_property: next_score}
                                remaining_properties = [
                                    prop for prop in task_properties
                                    if prop != self.target_property
                                ]
                                if remaining_properties:
                                    next_properties.update(
                                        self.oracle.evaluate_specific_properties(
                                            next_molecule_smiles,
                                            remaining_properties,
                                        )
                                    )
                                self._candidate_property_cache[next_molecule_smiles] = dict(
                                    next_properties
                                )

                            self._total_oracle_calls_this_episode += 1
                            info["total_oracle_calls_this_step"] = 1

                            if not was_cached:
                                self._oracle_calls_this_episode += 1
                                info["oracle_calls_this_step"] = 1
                            else:
                                info["oracle_calls_this_step"] = 0

                            improvement = next_score - score_before_modification
                            info.update({
                                "score_after": next_score,
                                "is_valid_modification": True,
                                "is_similar_enough": True,
                                "violates_structural_constraints": False,
                                "oracle_calls_this_episode": self._oracle_calls_this_episode,
                                "total_oracle_calls_this_episode": self._total_oracle_calls_this_episode
                            })

                        is_step_successful = constraint_valid

                        if not constraint_valid:
                            # Preserve the structural-constraint penalty and do
                            pass
                        elif len(task_properties) > 1:
                            multi_reward, multi_reward_info = self._calculate_multi_objective_reward(
                                self._current_molecule_smiles,
                                next_molecule_smiles,
                                old_properties=self._current_properties,
                                new_properties=next_properties,
                                original_properties=self._original_properties,
                            )
                            reward = multi_reward

                            weighted_score = multi_reward_info.get('weighted_score', 0.0)
                            current_best_weighted_score = getattr(self, '_best_weighted_score', float('-inf'))

                            if weighted_score > current_best_weighted_score:
                                self._best_molecule_smiles = next_molecule_smiles
                                self._best_mol_object = next_mol_object
                                self._best_weighted_score = weighted_score
                                self._best_individual_properties = multi_reward_info.get('new_properties', {})
                                self._best_global_improvements = multi_reward_info.get('global_improvements', {})
                                self._best_score = weighted_score
                                self._best_step = self._current_step
                                self._similarity_at_best_score = similarity
                                self._best_multi_reward = multi_reward

                                info["is_new_best"] = True
                                info["current_best_score"] = self._best_weighted_score
                                info["current_best_smiles"] = self._best_molecule_smiles
                                info["current_best_properties"] = self._best_individual_properties
                                info["current_best_improvements"] = self._best_global_improvements
                            else:
                                info["is_new_best"] = False

                            if multi_reward < 0:
                                is_step_successful = False

                            info["multi_objective_info"] = multi_reward_info
                            if multi_reward_info.get("all_success", False):
                                terminated = True
                                truncated = False
                                info["success"] = True
                                is_step_successful = True

                        elif self.minimize_target:
                            if improvement < 0:
                                reward = -improvement * 5
                                if next_score < self._best_score:
                                    self._best_molecule_smiles = next_molecule_smiles
                                    self._best_mol_object = next_mol_object
                                    self._best_score = next_score
                                    self._best_individual_properties = dict(next_properties)
                                    self._best_step = self._current_step
                                    self._similarity_at_best_score = similarity
                                    info["is_new_best"] = True
                                    info["current_best_score"] = self._best_score
                                    info["current_best_smiles"] = self._best_molecule_smiles
                                else:
                                    info["is_new_best"] = False
                            elif improvement > 0:
                                is_step_successful = False
                                reward = -improvement
                                info["is_new_best"] = False
                            else:
                                reward = getattr(self.config, 'no_improvement_penalty', -0.1)
                                info["is_new_best"] = False
                                info["no_improvement"] = True
                        else:
                            if improvement > 0:
                                reward = improvement * 5
                                if next_score > self._best_score:
                                    self._best_molecule_smiles = next_molecule_smiles
                                    self._best_mol_object = next_mol_object
                                    self._best_score = next_score
                                    self._best_individual_properties = dict(next_properties)
                                    self._best_step = self._current_step
                                    self._similarity_at_best_score = similarity
                                    info["is_new_best"] = True
                                    info["current_best_score"] = self._best_score
                                    info["current_best_smiles"] = self._best_molecule_smiles
                                else:
                                    info["is_new_best"] = False
                            elif improvement < 0:
                                is_step_successful = False
                                reward = improvement
                                info["is_new_best"] = False
                            else:
                                reward = getattr(self.config, 'no_improvement_penalty', -0.1)
                                info["is_new_best"] = False
                                info["no_improvement"] = True

                        is_single_objective = (len(task_properties) == 1)

                        if constraint_valid and is_single_objective:
                            from ragen.env.molecule_opt.property_utils import PROPERTY_ABSOLUTE_THRESHOLDS

                            prop_config = PROPERTY_ABSOLUTE_THRESHOLDS.get(self.target_property)

                            if prop_config is not None:
                                base_threshold = prop_config["threshold"]
                                direction = prop_config["direction"]

                                if direction == "increase":
                                    success = next_score >= base_threshold
                                else:
                                    success = next_score <= base_threshold

                                if success:
                                    terminated = True
                                    truncated = False
                                    info["success"] = True
                                    is_step_successful = True
                                    reward = 5.0

                        if (info["is_valid_modification"] and
                            info["is_similar_enough"] and
                            not info.get("violates_structural_constraints", False) and
                            not info.get("no_change_detected_in_step", False)):
                            self._current_molecule_smiles = next_molecule_smiles
                            self._current_mol_object = next_mol_object
                            self._current_properties = dict(next_properties)
                            info["final_smiles"] = self._current_molecule_smiles
                            self._submitted_smiles_set.add(next_molecule_smiles)

        except Exception as e:
            if getattr(self.oracle, 'raise_on_error', False):
                raise
            is_step_successful = False
            reward = -1
            info["is_valid_modification"] = False
            info["_error"] = str(e)

        if not is_step_successful:
            if (not info["is_valid_modification"] or
                not info["is_similar_enough"] or
                info.get("no_change_detected_in_step", False) or
                info.get("violates_structural_constraints", False)):
                self._consecutive_failures += 1
                if self._consecutive_failures >= 1 and self._best_molecule_smiles is not None and self._best_molecule_smiles != self._current_molecule_smiles:
                    self._current_molecule_smiles = self._best_molecule_smiles
                    self._current_mol_object = self._best_mol_object
                    best_properties = getattr(self, '_best_individual_properties', None)
                    if not best_properties:
                        best_properties = {self.target_property: self._best_score}
                    self._current_properties = dict(best_properties)
                    info["rollback_to_best"] = True
                    info["rollback_triggered_by_consecutive_failures"] = True
                    info["final_smiles"] = self._current_molecule_smiles
                    self._consecutive_failures = 0
            else:
                self._consecutive_failures = 0
        else:
            self._consecutive_failures = 0

        multi_info = info.get("multi_objective_info", {})
        multi_step_stalled = False
        if multi_info.get("is_multi_objective", False):
            step_improvements = multi_info.get("step_improvements", {})
            multi_step_stalled = any(
                (
                    -float(step_improvements.get(prop, 0.0))
                    if self.optimization_directions.get(prop, "maximize") == "minimize"
                    else float(step_improvements.get(prop, 0.0))
                ) <= 0.0
                for prop in self.molecule_opt_task.split('+')
            )

        # Classify duplicates before invalid SMILES because both set valid=False.
        if (info.get("duplicate_submission", False) or
                info.get("no_change_detected_in_step", False)):
            hint_routing_outcome = "unchanged"
        elif not info["is_valid_modification"]:
            hint_routing_outcome = "invalid_smiles"
        elif not info["is_similar_enough"]:
            hint_routing_outcome = "low_similarity"
        elif info.get("violates_structural_constraints", False):
            hint_routing_outcome = "constraint_violation"
        elif multi_step_stalled:
            hint_routing_outcome = "no_improvement"
        elif reward <= 0:
            hint_routing_outcome = "no_improvement"
        else:
            hint_routing_outcome = "improvement"

        if hint_routing_outcome == "no_improvement":
            self._consecutive_no_improvement += 1
        else:
            self._consecutive_no_improvement = 0

        if hint_routing_outcome in {"low_similarity", "unchanged"}:
            self._consecutive_rollbacks += 1
        else:
            self._consecutive_rollbacks = 0

        retrieval_should_trigger = False
        retrieval_trigger_reason = ""
        skill_should_trigger = False
        has_next_turn = (
            not terminated and self._current_step < self.config.max_steps - 1
        )
        memory_hint_trigger_count = getattr(
            self, '_memory_hint_trigger_count', 0
        )
        memory_budget_available = (
            memory_hint_trigger_count < self.retriever_config.max_triggers
        )

        if (
            has_next_turn
            and memory_budget_available
            and self.retriever_config.trigger_mode == "on_stuck"
        ):
            if self._consecutive_rollbacks >= self.retriever_config.rollback_threshold:
                retrieval_should_trigger = True
                retrieval_trigger_reason = hint_routing_outcome
            elif self._consecutive_no_improvement >= self.retriever_config.no_improvement_threshold:
                retrieval_should_trigger = True
                retrieval_trigger_reason = "no_improvement"

        min_step_for_hints = getattr(self, '_min_step_for_hints', 0)
        step_ready = self._global_training_step >= min_step_for_hints
        skill_memory = getattr(self, '_skill_memory', None)
        if (
            has_next_turn
            and memory_budget_available
            and step_ready
            and skill_memory is not None
        ):
            hints_trigger_mode = getattr(self, '_skill_hints_trigger_mode', 'on_stuck')
            if hints_trigger_mode == "on_stuck":
                if skill_memory.has_task(self.molecule_opt_task):
                    exp_no_imp_threshold = getattr(self, '_skill_hints_no_improvement_threshold', 2)
                    exp_rollback_threshold = getattr(self, '_skill_hints_rollback_threshold', 1)

                    if self._consecutive_rollbacks >= exp_rollback_threshold:
                        skill_should_trigger = True
                    elif self._consecutive_no_improvement >= exp_no_imp_threshold:
                        skill_should_trigger = True

        # The chosen source owns the attempt; an empty result has no fallback.
        if retrieval_should_trigger and skill_should_trigger:
            chosen = self.np_random.choice(['retrieval', 'skill'])
            if chosen == 'retrieval':
                skill_should_trigger = False
            else:
                retrieval_should_trigger = False

        if retrieval_should_trigger:
            # Selecting a source consumes one shared attempt, including empty
            # results or request failures.
            self._memory_hint_trigger_count = memory_hint_trigger_count + 1
            step_task_properties = self.molecule_opt_task.split('+')
            step_is_multi_objective = len(step_task_properties) > 1

            if step_is_multi_objective:
                step_original_properties = dict(self._original_properties)
                step_query_properties = dict(self._current_properties)
                retrieval_results = self._retrieve_similar_molecules_multi_objective(
                    query_smiles=self._current_molecule_smiles,
                    original_smiles=self._original_base_molecule_smiles,
                    task_properties=step_task_properties,
                    original_properties=step_original_properties,
                    similarity_threshold=self.config.final_similarity_threshold,
                    hint_topk=getattr(self.retriever_config, 'hint_topk', 3),
                    query_properties=step_query_properties
                )
            else:
                retrieval_results = self._retrieve_similar_molecules(
                    query_smiles=self._current_molecule_smiles,
                    original_smiles=self._original_base_molecule_smiles,
                    similarity_threshold=self.config.final_similarity_threshold,
                    sort_by=self.target_property,
                    original_property_value=self._best_score
                )

            self._consecutive_no_improvement = 0
            self._consecutive_rollbacks = 0

            if retrieval_results:
                if step_is_multi_objective:
                    self._pending_hint = self._format_retrieval_as_hint_multi_objective(
                        retrieval_results, step_task_properties, step_original_properties
                    )
                else:
                    self._pending_hint = self._format_retrieval_as_hint(
                        retrieval_results,
                        trigger_reason=retrieval_trigger_reason
                    )

        if skill_should_trigger:
            self._consecutive_no_improvement = 0
            self._consecutive_rollbacks = 0

            # Empty attempts consume the selected source's budget.
            self._memory_hint_trigger_count = memory_hint_trigger_count + 1
            skill_hints = self.get_skill_hints(self._current_molecule_smiles)
            if skill_hints:
                if self._pending_hint:
                    self._pending_hint += f"\n\n{skill_hints}"
                else:
                    self._pending_hint = skill_hints

        self._current_step += 1
        if self._current_step >= self.config.max_steps:
            truncated = True

        observation = self._get_step_observation(reward, info)

        if self._pending_hint:
            observation += self._pending_hint
            self._pending_hint = ""

        self.render_cache = observation
        done = terminated or truncated

        return observation, reward, done, info

    def render(self) -> Optional[str]:
        """Returns the cached textual representation of the current state."""
        if self.render_mode == "text":
            return self.render_cache
        else:
             raise NotImplementedError(f"Render mode '{self.render_mode}' not supported.")

    def close(self):
        """Clean up any resources."""
        if hasattr(self.oracle, 'save_cache') and self.config.oracle_config.get('cache_file'):
             try:
                 self.oracle.save_cache(self.config.oracle_config['cache_file'])
                 print(f"Oracle cache saved to {self.config.oracle_config['cache_file']}")
             except Exception as e:
                 print(f"Error saving oracle cache: {e}")
        self.render_cache = None


    def _get_step_observation(self, last_reward: float, last_info: Dict) -> str:
        step_info = f"Step {self._current_step} of {self.config.max_steps}\\n"
        target_prop_name = PROPERTY_MAPPING.get(self.target_property, {}).get('description', self.target_property)
        direction_hint = "(lower is better)" if self.minimize_target else "(higher is better)"

        def get_detailed_score_display(smiles, score, label="Score"):
            if self.target_property == 'logp_qed':
                components = self.oracle.evaluate_with_components(smiles)
                return (f"{label} (LogP: {components['logp']:.3f}, QED: {components['qed']:.3f}, "
                       f"Combined: {components['combined']:.3f})")
            else:
                return f"{label}: {score:.3f}"

        rollback_message_segment = ""
        rollback_notice_segment = ""
        if last_info.get("rollback_triggered_by_consecutive_failures", False):
            failed_attempt_smiles = last_info.get('action_received', 'N/A')
            current_properties = getattr(self, '_current_properties', {})
            current_score = current_properties.get(self.target_property)
            if current_score is None:
                current_score = self.oracle(self._current_molecule_smiles)
            score_of_failed_attempt = last_info.get('score_after', current_score)
            score_before_failed_attempt = last_info.get('score_before', current_score)
            failed_improvement = score_of_failed_attempt - score_before_failed_attempt

            failed_attempt_details = (
                f"Your last proposed modification was to '{failed_attempt_smiles}'.\\n"
                f"This resulted in a score of {score_of_failed_attempt:.3f} {direction_hint} (change from {score_before_failed_attempt:.3f}: {failed_improvement:+.3f}).\\n"
            )
            rollback_message_segment = (
                 f"Due to the unsuccessful modification detailed above, the environment has reverted to your previously best molecule (from step {self._best_step + 1}).\\n"
                 f"{failed_attempt_details}"
                 f"The best score achieved so far is {self._best_score:.3f} {direction_hint} with SMILES: {self._best_molecule_smiles}.\\n"
                 f"Your current molecule for the next attempt is now: {self._current_molecule_smiles} (Score: {current_score:.3f} {direction_hint}).\\n"
                 f"Please consider different modification strategies.\\n"
            )
            rollback_notice_segment = (
                "The environment reverted to the best valid molecule before "
                "this next attempt.\\n"
            )

        current_required_threshold = last_info.get("current_similarity_threshold", self.config.final_similarity_threshold)
        current_mol_for_next_turn_display = self._current_molecule_smiles
        current_properties = getattr(self, '_current_properties', {})
        score_of_current_mol_for_next_turn = current_properties.get(self.target_property)
        if score_of_current_mol_for_next_turn is None:
            score_of_current_mol_for_next_turn = self.oracle(
                current_mol_for_next_turn_display
            )


        # Prefer failure-specific feedback over generic rollback text.
        if last_info.get("duplicate_submission", False):
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                f"{rollback_notice_segment}"
                f"Repeated submission: '{last_info.get('action_received', 'N/A')}' was already tried.\\n"
                f"Current molecule for the next attempt:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Current required similarity >= {current_required_threshold:.3f}\\n"
                f"Please propose a different modification.\\n"
                f"Reward: {last_reward:.3f}\\n"
            )

        if not last_info["is_valid_modification"]:
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                f"{rollback_notice_segment}"
                f"Invalid SMILES format provided: '{last_info.get('action_received', 'N/A')}'. Ensure the SMILES is correct.\\n"
                f"Current molecule for the next attempt:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Current required similarity >= {current_required_threshold:.3f}\\n"
                f"Reward: {last_reward:.3f}\\n"
            )

        if last_info.get("no_change_detected_in_step", False):
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                f"{rollback_notice_segment}"
                f"No modification detected. Your proposed SMILES '{last_info.get('action_received')}' matches the current molecule.\\n"
                f"Current molecule for the next attempt:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Current required similarity >= {current_required_threshold:.3f}\\n"
                f"Please propose a different modification.\\n"
                f"Reward: {last_reward:.3f}\\n"

            )

        if not last_info["is_similar_enough"]:
            similarity_achieved = last_info.get('similarity', 0.0)
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                f"{rollback_notice_segment}"
                f"Similarity too low: {similarity_achieved:.3f} < required {current_required_threshold:.3f} for proposed SMILES '{last_info.get('action_received', 'N/A')}'.\\n"
                f"Current molecule for the next attempt:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Reward: {last_reward:.3f}\\n"
                f"Consider smaller, more conservative changes to the current molecule.\\n"
            )

        if last_info.get("violates_structural_constraints", False):
            constraint_message = last_info.get("_constraint_violation_message", "Unknown constraint violation")
            current_score_display = get_detailed_score_display(current_mol_for_next_turn_display, score_of_current_mol_for_next_turn, "Current score").replace("Current score:", "")
            return (
                f"{step_info}"
                f"{rollback_notice_segment}"
                f"Structural constraint violation: {constraint_message}\\n"
                f"Proposed SMILES: '{last_info.get('action_received', 'N/A')}'\\n"
                f"Current molecule for the next attempt:\\n{current_mol_for_next_turn_display}\\n"
                f"Current score {direction_hint}: {current_score_display}\\n"
                f"Correct the stated constraint violation before attempting further optimization.\\n"
                f"Reward: {last_reward:.3f}\\n"
            )

        if rollback_message_segment:
            return (
                f"{step_info}"
                f"{rollback_message_segment}"
                f"Reward for the last attempted action (that triggered rollback): {last_reward:.3f}\\n"
            )

        actual_modified_molecule = last_info.get("final_smiles", current_mol_for_next_turn_display)
        score_of_modified_molecule = last_info['score_after']
        improvement = score_of_modified_molecule - last_info["score_before"]
        similarity_achieved = last_info.get('similarity', 0.0)
        task_properties = self.molecule_opt_task.split('+')
        multi_obj_info = last_info.get("multi_objective_info", {})

        if multi_obj_info.get("is_multi_objective", False) or len(task_properties) > 1:
            old_properties = multi_obj_info.get("old_properties", {})
            new_properties = multi_obj_info.get("new_properties", {})

            guidance = generate_complete_multi_objective_guidance(
                original_scores=old_properties,
                new_scores=new_properties,
                task_properties=task_properties,
                reward_info=multi_obj_info,
                is_initial=False
            )
        else:
            guidance = ""
            if self.minimize_target:
                if last_info.get("is_new_best", False):
                    guidance = f"Excellent! New best score achieved {direction_hint}! Keep improving."
                elif improvement < 0:
                    guidance = f"Good improvement! Score decreased {direction_hint}. Continue reducing the score."
                elif improvement == 0:
                    guidance = "No score change. Valid modification, but property didn't change. Try alternative strategies."
                else:
                    guidance = f"Score increased {direction_hint}. This worsened the property. Try different modifications to reduce the score."
            else:
                if last_info.get("is_new_best", False):
                    guidance = f"Great job! New best score achieved {direction_hint}! Keep refining."
                elif improvement > 0:
                    guidance = f"Good improvement! Score increased {direction_hint}. Keep refining."
                elif improvement == 0:
                    guidance = "No score change. Valid modification, but property didn't change. Try alternative strategies."
                else:
                    guidance = f"Score decreased. Valid modification, but negatively impacted property {direction_hint}. Consider alternative strategies."

        before_smiles = last_info.get('action_received', current_mol_for_next_turn_display)
        before_score_display = get_detailed_score_display(
            self._current_molecule_smiles if hasattr(self, '_current_molecule_smiles') else before_smiles,
            last_info['score_before'],
            "Previous Score"
        )

        after_score_display = get_detailed_score_display(
            actual_modified_molecule,
            score_of_modified_molecule,
            f"New Score ({target_prop_name} {direction_hint})"
        )

        return (
            f"{step_info}"
            f"Action: Proposed '{last_info.get('action_received', 'N/A')}'\\n"
            f"{before_score_display}\\n"
            f"Resulting Molecule: {actual_modified_molecule}\\n"
            f"Similarity to Original: {similarity_achieved:.3f} (required >= {current_required_threshold:.3f})\\n"
            f"{after_score_display} (change: {improvement:+.3f})\\n"
            f"{guidance}\\n"
            f"Reward: {last_reward:.3f}\\n"
        )

    def _calculate_similarity(self, mol1: Optional[Chem.Mol], mol2: Optional[Chem.Mol]) -> float:
        """Calculates Tanimoto similarity between two RDKit Mol objects."""
        if mol1 is None or mol2 is None:
            return 0.0
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            print(f"Warning: Similarity calculation failed - {e}")
            return 0.0

    def _get_heavy_atom_count(self, mol: Optional[Chem.Mol]) -> int:
        """Calculate the number of heavy atoms (non-hydrogen atoms) in the molecule"""
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()

    def _get_longest_carbon_chain(self, mol: Optional[Chem.Mol]) -> int:
        """Calculate the longest linear carbon chain length in the molecule"""
        if mol is None:
            return 0

        try:
            carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']

            if not carbon_atoms:
                return 0

            max_chain_length = 0

            for start_carbon in carbon_atoms:
                visited = set()
                chain_length = self._dfs_carbon_chain(mol, start_carbon, visited)
                max_chain_length = max(max_chain_length, chain_length)

            return max_chain_length
        except Exception as e:
            print(f"Warning: Carbon chain calculation failed - {e}")
            return 0

    def _dfs_carbon_chain(self, mol: Chem.Mol, atom_idx: int, visited: set) -> int:
        """Depth-first search to calculate the longest carbon chain starting from a specified carbon atom"""
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)

        if atom.GetSymbol() != 'C':
            return 0

        max_length = 1

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor.GetSymbol() == 'C' and neighbor_idx not in visited:
                chain_length = 1 + self._dfs_carbon_chain(mol, neighbor_idx, visited.copy())
                max_length = max(max_length, chain_length)

        return max_length

    def _check_structural_constraints(self, mol: Optional[Chem.Mol]) -> tuple[bool, str]:
        """
        Check if molecule violates structural constraints

        Returns:
            (is_valid, violation_message)
        """
        if mol is None:
            return False, "Invalid molecule structure"

        heavy_atom_count = self._get_heavy_atom_count(mol)
        max_heavy_atoms = self.config.max_heavy_atoms
        if heavy_atom_count > max_heavy_atoms:
            return False, f"Molecule too large: contains {heavy_atom_count} heavy atoms (limit ≤{max_heavy_atoms}). Large molecules typically have poor synthesizability and drug-likeness."

        longest_carbon_chain = self._get_longest_carbon_chain(mol)
        max_carbon_chain = self.config.max_carbon_chain_length
        if longest_carbon_chain > max_carbon_chain:
            return False, f"Carbon chain too long: longest linear carbon chain is {longest_carbon_chain} atoms (limit ≤{max_carbon_chain}). Long carbon chains reduce synthesizability and drug-likeness. Cannot simply add carbon atoms to optimize properties."

        return True, ""

    def _retrieve_similar_molecules(
        self,
        query_smiles: str,
        original_smiles: str,
        similarity_threshold: float,
        sort_by: Optional[str] = None,
        original_property_value: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve similar molecules from the retrieval server with dual-molecule filtering.

        Args:
            query_smiles: Query molecule SMILES (used for FAISS search)
            original_smiles: Original molecule SMILES (used for similarity filtering)
            similarity_threshold: Minimum similarity to original molecule
            sort_by: Property to sort by (e.g., 'qed', 'logp', 'sa')
            original_property_value: Original molecule's property value for strict filtering

        Returns:
            List of similar molecules with their properties (filtered by original similarity
            and optionally by property > original)
        """
        try:
            request_data = {
                "queries": [query_smiles],
                "original_smiles": [original_smiles],
                "similarity_threshold": similarity_threshold,
                "topk": self.retriever_config.topk,
                "return_scores": True,
                "sort_by": sort_by or self.target_property,
                "descending": not self.minimize_target,
                "filter_better_only": True,
                "min_improvement_margin": getattr(self.retriever_config, 'min_improvement_margin', 0.0)
            }
            if original_property_value is not None:
                request_data["original_property_value"] = original_property_value

            response = requests.post(
                self.retriever_config.url,
                json=request_data,
                timeout=self.retriever_config.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Static Exemplar Memory returned HTTP {response.status_code}: "
                    f"{response.text}"
                )
            data = response.json()
            result_batches = data.get("result")
            if (
                not isinstance(result_batches, list)
                or len(result_batches) != 1
                or not isinstance(result_batches[0], list)
            ):
                raise RuntimeError("Static Exemplar Memory returned a malformed response")
            hint_topk = int(getattr(self.retriever_config, 'hint_topk', 3))
            return result_batches[0][:hint_topk]

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Static Exemplar Memory timed out after {self.retriever_config.timeout}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Static Exemplar Memory is unavailable; start the retrieval server"
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Static Exemplar Memory query failed") from exc

    def _format_retrieval_as_hint(self, results: List[Dict], trigger_reason: str = "") -> str:
        """
        Format retrieval results as a hint for the LLM (professional, no emojis).

        Args:
            results: Filtered retrieval results with similarity_to_original
            trigger_reason: Why the hint was triggered ("low_similarity",
                "unchanged", or "no_improvement")

        Returns:
            Formatted hint string
        """
        if not results:
            return ""

        prop_name = PROPERTY_MAPPING.get(self.target_property, {}).get('description', self.target_property)
        direction = "lower is better" if self.minimize_target else "higher is better"

        text = "\n\n=== SIMILAR HIGH-SCORING MOLECULES FOR REFERENCE ===\n"
        text += f"Here are {len(results)} similar molecules with high {prop_name} scores ({direction}):\n\n"

        for i, item in enumerate(results, 1):
            sim_to_original = item.get('similarity_to_original', 0)
            properties = item.get('properties', {})

            doc = item.get('document', {})
            contents = doc.get('contents', '')
            smiles = ""
            for line in contents.split('\n'):
                if line.strip().startswith('SMILES:'):
                    smiles = line.split(':', 1)[1].strip()
                    break

            prop_value = properties.get(self.target_property, None)

            text += f"{i}. SMILES: {smiles}\n"
            if prop_value is not None:
                text += f"   {prop_name}: {prop_value:.3f}\n"
            text += f"   Similarity to original: {sim_to_original:.3f}\n\n"

        return text

    def _retrieve_similar_molecules_multi_objective(
        self,
        query_smiles: str,
        original_smiles: str,
        task_properties: List[str],
        original_properties: Dict[str, float],
        similarity_threshold: float,
        hint_topk: int = 3,
        query_properties: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Retrieve similar molecules for multi-objective optimization.

        Filtering logic:
        1. Filter candidates with weighted_score > 0 (better than query molecule)
        2. Keep only molecules for which every objective meets its success threshold
        3. Sort by weighted_score and take the top ``hint_topk``

        Args:
            query_smiles: Current molecule SMILES
            original_smiles: Original molecule SMILES
            task_properties: List of target properties (e.g., ['qed', 'sa'])
            original_properties: Original molecule's property values (for success judgment)
            similarity_threshold: Minimum similarity to original
            hint_topk: Number of molecules to return for prompt
            query_properties: Current molecule's property values (for weighted_score filtering)

        Returns:
            List of molecules with computed improvements and success status
        """
        try:
            filter_properties = (
                query_properties
                if query_properties is not None
                else original_properties
            )
            objectives = []
            for prop in task_properties:
                if prop not in original_properties or prop not in filter_properties:
                    raise ValueError(f"Missing retrieval baseline for objective {prop}")
                threshold_config = PROPERTY_SUCCESS_THRESHOLDS.get(prop)
                if threshold_config is None:
                    raise ValueError(f"Missing success threshold for objective {prop}")
                objectives.append({
                    "name": prop,
                    "direction": self.optimization_directions.get(prop, "maximize"),
                    "weight": PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0),
                    "original_value": float(original_properties[prop]),
                    "query_value": float(filter_properties[prop]),
                    "success_margin": float(threshold_config["threshold"]),
                })

            request_data = {
                "queries": [query_smiles],
                "original_smiles": [original_smiles],
                "similarity_threshold": similarity_threshold,
                "topk": self.retriever_config.topk,
                "return_scores": True,
                "filter_better_only": False,
                "objectives": objectives,
            }

            response = requests.post(
                self.retriever_config.url,
                json=request_data,
                timeout=self.retriever_config.timeout
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Static Exemplar Memory returned HTTP {response.status_code}: "
                    f"{response.text}"
                )

            data = response.json()
            result_batches = data.get("result")
            if (
                not isinstance(result_batches, list)
                or len(result_batches) != 1
                or not isinstance(result_batches[0], list)
            ):
                raise RuntimeError("Static Exemplar Memory returned a malformed response")
            raw_results = result_batches[0]

            if not raw_results:
                return []

            processed = []
            for item in raw_results:
                sim_to_original = item.get('similarity_to_original', 0)
                properties = item.get('properties', {})

                doc = item.get('document', {})
                contents = doc.get('contents', '')
                smiles = ""
                for line in contents.split('\n'):
                    if line.strip().startswith('SMILES:'):
                        smiles = line.split(':', 1)[1].strip()
                        break

                if not smiles:
                    continue

                if any(
                    prop not in properties
                    or prop not in original_properties
                    or prop not in filter_properties
                    for prop in task_properties
                ):
                    continue

                improvements = {}
                all_success = True

                for prop in task_properties:
                    candidate_val = properties[prop]
                    original_val = original_properties[prop]
                    improvement = candidate_val - original_val
                    improvements[prop] = improvement

                    config = PROPERTY_SUCCESS_THRESHOLDS.get(prop)
                    if config:
                        threshold = config["threshold"]
                        direction = config["direction"]
                        if direction == "increase":
                            if improvement < threshold:
                                all_success = False
                        else:
                            if improvement > -threshold:
                                all_success = False
                    else:
                        all_success = False

                weighted_score = 0.0
                for prop in task_properties:
                    weight = PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
                    candidate_val = properties.get(prop, 0.0)
                    query_val = filter_properties.get(prop, 0.0)
                    query_improvement = candidate_val - query_val
                    direction = self.optimization_directions.get(prop, "maximize")
                    if direction == "minimize":
                        weighted_score += (-query_improvement) * weight
                    else:
                        weighted_score += query_improvement * weight

                if weighted_score <= 0:
                    continue

                processed.append({
                    'smiles': smiles,
                    'properties': properties,
                    'improvements': improvements,
                    'similarity_to_original': sim_to_original,
                    'weighted_score': weighted_score,
                    'is_success': all_success
                })

            # Multi-objective exemplars must satisfy every objective.
            successful = [m for m in processed if m['is_success']]
            successful.sort(key=lambda x: x['weighted_score'], reverse=True)
            return successful[:hint_topk]

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Static Exemplar Memory timed out after {self.retriever_config.timeout}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Static Exemplar Memory is unavailable; start the retrieval server"
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Static Exemplar Memory query failed") from exc

    def _format_retrieval_as_hint_multi_objective(
        self,
        results: List[Dict],
        task_properties: List[str],
        original_properties: Dict[str, float]
    ) -> str:
        """
        Format multi-objective retrieval results as a hint for the LLM (simplified format).

        Args:
            results: Processed retrieval results with improvements and success status
            task_properties: List of target properties
            original_properties: Original molecule's property values

        Returns:
            Formatted hint string
        """
        if not results:
            return ""

        prop_names = [prop.upper() for prop in task_properties]

        text = "\n\n=== SIMILAR HIGH-SCORING MOLECULES FOR REFERENCE ===\n"
        text += f"Here are {len(results)} similar molecules with good {' and '.join(prop_names)} scores:\n\n"

        for i, item in enumerate(results, 1):
            smiles = item.get('smiles', '')
            properties = item.get('properties', {})
            improvements = item.get('improvements', {})
            sim_to_original = item.get('similarity_to_original', 0)

            text += f"{i}. SMILES: {smiles}\n"

            for prop in task_properties:
                prop_val = properties.get(prop, 0.0)
                improvement = improvements.get(prop, 0.0)
                sign = "+" if improvement >= 0 else ""
                text += f"   {prop.upper()}: {prop_val:.3f} ({sign}{improvement:.2f})\n"

            text += f"   Similarity: {sim_to_original:.3f}\n\n"

        return text

    def _format_retrieval_results(self, results: List[Dict]) -> str:
        """
        Format retrieval results for LLM observation.

        Args:
            results: List of retrieval results

        Returns:
            Formatted string for observation
        """
        if not results:
            return ""

        prop_name = PROPERTY_MAPPING.get(self.target_property, {}).get('description', self.target_property)
        direction = "lower is better" if self.minimize_target else "higher is better"

        text = "\n\n=== SIMILAR HIGH-SCORING MOLECULES FOR REFERENCE ===\n"
        text += f"Here are {len(results)} similar molecules with high {prop_name} scores ({direction}):\n\n"

        target_prop_upper = self.target_property.upper()
        pattern_parens = f'({target_prop_upper})'
        pattern_colon = f'{target_prop_upper}:'
        pattern_activity = f'{target_prop_upper} ACTIVITY'

        for i, item in enumerate(results, 1):
            score = item.get('similarity_to_original', item.get('score', 0))
            doc = item.get('document', {})
            contents = doc.get('contents', '')

            smiles = ""
            property_value = item.get('properties', {}).get(self.target_property)
            for line in contents.split('\n'):
                if not line.strip():
                    continue

                if line.startswith('SMILES:'):
                    smiles = line.split(':', 1)[1].strip()
                else:
                    line_upper = line.upper()
                    if (pattern_parens in line_upper or
                        line_upper.startswith(pattern_colon) or
                        pattern_activity in line_upper):
                        property_value = line.split(':', 1)[1].strip()

                if smiles and property_value not in (None, ""):
                    break

            text += f"{i}. SMILES: {smiles}\n"
            text += f"   Tanimoto Similarity: {score:.3f}\n"
            if property_value not in (None, ""):
                if isinstance(property_value, (int, float)):
                    property_value = f"{property_value:.3f}"
                text += f"   {prop_name}: {property_value}\n"
            text += "\n"

        return text

    def set_skill_memory(self, memory: EvolvingSkillMemory, min_step: int = 0,
                         trigger_mode: str = "on_stuck",
                         no_improvement_threshold: int = 2,
                         rollback_threshold: int = 1,
                         retrieval_config: dict = None):
        """Set the Evolving Skill Memory used for dynamic hints.

        Args:
            memory: EvolvingSkillMemory instance
            min_step: Minimum training global step before injecting hints
            trigger_mode: "always" or "on_stuck"
            no_improvement_threshold: Steps without improvement to trigger (on_stuck mode)
            rollback_threshold: Consecutive low-similarity or unchanged outputs
                required to trigger a hint
            retrieval_config: Config for retrieval parameters (top_k, similarity_threshold, etc.)
        """
        if trigger_mode not in {"always", "on_stuck"}:
            raise ValueError(
                "Evolving Skill Memory trigger_mode must be 'always' or 'on_stuck'"
            )
        if memory is None:
            raise ValueError("Evolving Skill Memory must not be None")
        if min_step < 0:
            raise ValueError("Evolving Skill Memory min_step must be non-negative")
        if no_improvement_threshold <= 0 or rollback_threshold <= 0:
            raise ValueError("Evolving Skill Memory trigger thresholds must be positive")
        retrieval_config = retrieval_config or {}
        for field_name in ('top_k_by_fg', 'top_k_by_similarity'):
            if retrieval_config.get(field_name, 1) <= 0:
                raise ValueError(f"Evolving Skill Memory {field_name} must be positive")
        retrieval_defaults = {
            'similarity_threshold': 0.5,
            'min_jaccard': 0.8,
        }
        for field_name, default_value in retrieval_defaults.items():
            value = retrieval_config.get(field_name, default_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Evolving Skill Memory {field_name} must be in [0, 1]")

        self._skill_memory = memory
        self._min_step_for_hints = int(min_step)
        self._skill_hints_trigger_mode = trigger_mode
        self._skill_hints_no_improvement_threshold = no_improvement_threshold
        self._skill_hints_rollback_threshold = rollback_threshold
        self._skill_retrieval_config = retrieval_config

    def get_skill_hints(self, smiles: str = None) -> str:
        """
        Query Evolving Skill Memory and generate hints for the current molecule.

        Args:
            smiles: SMILES to query for (defaults to current molecule)

        Returns:
            Formatted hints string to inject into observation
        """
        if self._skill_memory is None:
            return ""

        query_smiles = smiles or self._current_molecule_smiles
        if not query_smiles:
            return ""

        try:
            retrieval_cfg = getattr(self, '_skill_retrieval_config', {})
            top_k_by_fg = retrieval_cfg.get('top_k_by_fg', 3)
            top_k_by_similarity = retrieval_cfg.get('top_k_by_similarity', 2)
            similarity_threshold = retrieval_cfg.get('similarity_threshold', 0.5)
            min_jaccard = retrieval_cfg.get('min_jaccard', 0.8)
            current_fgs = detect_functional_groups(query_smiles)

            hints = []

            fg_hints = self._skill_memory.query_by_functional_group(
                current_fgs=current_fgs,
                task=self.molecule_opt_task,
                top_k=top_k_by_fg,
                min_jaccard=min_jaccard,
                smiles=query_smiles,
            )
            hints.extend(fg_hints)

            sim_hints = self._skill_memory.query_by_similarity(
                smiles=query_smiles,
                task=self.molecule_opt_task,
                top_k=top_k_by_similarity,
                min_similarity=similarity_threshold,
            )
            hints.extend(sim_hints)

            seen_ids = set()
            unique_hints = []
            for h in hints:
                if h.exp_id not in seen_ids:
                    seen_ids.add(h.exp_id)
                    unique_hints.append(h)

            # Fingerprint and functional-group retrieval are independent
            # relevance signals. Keep their union, as in Eq. (5), and use the
            # fingerprint similarity only to break equal-improvement ties.
            query_mol = Chem.MolFromSmiles(query_smiles) if RDKIT_AVAILABLE else None
            relevant_hints = []
            for hint in unique_hints:
                before_mol = (
                    Chem.MolFromSmiles(hint.before_smiles)
                    if RDKIT_AVAILABLE else None
                )
                similarity = self._calculate_similarity(query_mol, before_mol)
                relevant_hints.append((
                    hint,
                    similarity,
                    hint.improvement_delta,
                ))

            # Both branches already applied their own relevance threshold.
            # Rank the independent union by observed improvement.
            relevant_hints.sort(
                key=lambda item: (item[2], item[1]), reverse=True
            )
            ranked_hints = [item[0] for item in relevant_hints]
            final_hints = self._skill_memory.select_distinct_actions(
                ranked_hints,
                top_k=top_k_by_fg + top_k_by_similarity,
            )

            if not final_hints:
                return ""

            formatted = format_hints_for_prompt(
                experiences=final_hints,
                task=self.molecule_opt_task,
            )
            if not formatted:
                return ""

            return formatted

        except Exception as exc:
            raise RuntimeError("Evolving Skill Memory query failed") from exc
