import os
import yaml
from rdkit import Chem
import tdc
import math
import shutil
import uuid
import time
import warnings
import logging

logging.getLogger('tdc').setLevel(logging.WARNING)

ORACLE_SILENT_MODE = os.environ.get('ORACLE_SILENT_MODE', 'false').lower() == 'true'

_GLOBAL_TDC_ORACLES = {}

def get_or_create_tdc_oracle(oracle_name: str):
    """
    Return a shared TDC oracle instance.

    Args:
        oracle_name: TDC oracle name, such as ``QED``, ``LogP``, or ``JNK3``.

    Returns:
        The shared oracle instance.
    """
    if oracle_name not in _GLOBAL_TDC_ORACLES:
        try:
            if not ORACLE_SILENT_MODE:
                print(f"Initializing {oracle_name} oracle...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _GLOBAL_TDC_ORACLES[oracle_name] = tdc.Oracle(name=oracle_name)
            if not ORACLE_SILENT_MODE:
                print(f"Initialized {oracle_name} oracle")
        except Exception as e:
            print(f"Failed to initialize {oracle_name} oracle: {e}")
            raise e
    return _GLOBAL_TDC_ORACLES[oracle_name]

def clear_global_tdc_oracles():
    """Clear the shared TDC oracle cache."""
    global _GLOBAL_TDC_ORACLES
    _GLOBAL_TDC_ORACLES.clear()
    if not ORACLE_SILENT_MODE:
        print("Cleared the shared TDC oracle cache")

def get_global_tdc_oracle_stats():
    """Return shared TDC oracle cache statistics."""
    return {
        'cached_oracles': list(_GLOBAL_TDC_ORACLES.keys()),
        'total_count': len(_GLOBAL_TDC_ORACLES)
    }

class MoleculeOracle:
    """Molecular property evaluator."""

    def __init__(self, target_property='qed', cache_file=None, multi_objective_weights=None):
        self.target_property = target_property
        self.calls = 0
        self.total_calls = 0
        self.cache = {}
        self.use_fallback = False
        self.raise_on_error = False

        self.multi_objective_weights = multi_objective_weights or {'logp': 0.5, 'qed': 0.5}

        self.instance_id = f"{os.getpid()}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        self.property_evaluators = {}

        self._preload_common_evaluators()

        try:
            property_to_oracle = {
                'qed': 'QED',
                'logp': 'LogP',
                'jnk3': 'JNK3',
                'drd2': 'DRD2',
                'sa': 'SA'
            }

            if target_property in property_to_oracle:
                oracle_name = property_to_oracle[target_property]
                try:
                    self.evaluator = get_or_create_tdc_oracle(oracle_name)
                except Exception as e:
                    if target_property == 'jnk3':
                        print(f"Initial JNK3 model load failed: {e}")
                        if not ORACLE_SILENT_MODE:
                            print("Clearing the TDC cache and retrying...")
                        self._clear_tdc_cache()
                        self.evaluator = get_or_create_tdc_oracle(oracle_name)
                        if not ORACLE_SILENT_MODE:
                            print("Reloaded the JNK3 model")
                    else:
                        raise e
            elif target_property == 'mr':
                self.use_fallback = True
                self.evaluator = None
            else:
                raise ValueError(f"Unsupported property: {target_property}")
        except Exception as e:
            if target_property in ['jnk3', 'drd2']:
                print(f"Failed to load the {target_property.upper()} model: {e}")
                raise RuntimeError(
                    f"Failed to load the {target_property.upper()} model: {e}. "
                    "Check network access or clear the TDC cache and retry."
                )
            else:
                raise

        if cache_file and os.path.exists(cache_file):
            self.load_cache(cache_file)

    def _clear_tdc_cache(self):
        """Clear the TDC cache directory."""
        try:
            tdc_home = os.path.expanduser("~/.tdc")
            if os.path.exists(tdc_home):
                if not ORACLE_SILENT_MODE:
                    print(f"Clearing TDC cache: {tdc_home}")
                shutil.rmtree(tdc_home)
                if not ORACLE_SILENT_MODE:
                    print("Cleared the TDC cache")
            else:
                if not ORACLE_SILENT_MODE:
                    print("No TDC cache directory found")
        except Exception as e:
            print(f"Failed to clear the TDC cache: {e}")

    def _preload_common_evaluators(self):
        """Preload shared property evaluators."""
        try:
            common_properties = {
                'qed': 'QED',
                'logp': 'LogP',
                'sa': 'SA',
                'jnk3': 'JNK3',
                'drd2': 'DRD2'
            }
            if not ORACLE_SILENT_MODE:
                print("Preloading molecular property evaluators...")

            for prop, oracle_name in common_properties.items():
                try:
                    self.property_evaluators[prop] = get_or_create_tdc_oracle(oracle_name)
                except Exception as e:
                    if not ORACLE_SILENT_MODE:
                        print(f"Failed to preload the {prop} evaluator: {e}")
                    continue

            if not ORACLE_SILENT_MODE:
                print(f"Preloaded {len(self.property_evaluators)} evaluators")
        except Exception as e:
            if not ORACLE_SILENT_MODE:
                print(f"Failed to preload property evaluators: {e}")
            self.property_evaluators = {}

    def __call__(self, smiles):
        """Evaluate one molecule or a list of molecules."""
        if isinstance(smiles, list):
            return [self.evaluate(smi) for smi in smiles]
        else:
            return self.evaluate(smiles)

    def evaluate(self, smiles):
        """Evaluate one molecule."""
        if not smiles:
            return 0.0

        self.total_calls += 1

        was_cached = smiles in self.cache
        if was_cached:
            return self.cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            if self.use_fallback:
                score = self._fallback_evaluate(smiles, mol)
            else:
                score = float(self.evaluator(smiles))

            if not math.isfinite(score):
                if self.raise_on_error:
                    raise RuntimeError(
                        f"{self.target_property} oracle returned a non-finite value"
                    )
                score = 0.0

            self.cache[smiles] = score
            self.calls += 1

            return score
        except Exception as e:
            if self.raise_on_error:
                raise RuntimeError(
                    f"{self.target_property} oracle failed for {smiles}: {e}"
                ) from e
            print(f"Property evaluation failed: {e}")
            return 0.0

    def evaluate_with_cache_info(self, smiles):
        """Evaluate one molecule and report whether the result was cached."""
        if not smiles:
            return 0.0, True

        self.total_calls += 1

        was_cached = smiles in self.cache
        if was_cached:
            return self.cache[smiles], True

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0, False

            if self.use_fallback:
                score = self._fallback_evaluate(smiles, mol)
            else:
                score = float(self.evaluator(smiles))

            if not math.isfinite(score):
                if self.raise_on_error:
                    raise RuntimeError(
                        f"{self.target_property} oracle returned a non-finite value"
                    )
                score = 0.0

            self.cache[smiles] = score
            self.calls += 1

            return score, False
        except Exception as e:
            if self.raise_on_error:
                raise RuntimeError(
                    f"{self.target_property} oracle failed for {smiles}: {e}"
                ) from e
            print(f"Property evaluation failed: {e}")
            return 0.0, False

    def evaluate_with_components(self, smiles):
        """
        Evaluate a molecule and return multi-objective components.

        Returns:
            Component scores and their combined value.
        """
        if self.target_property == 'logp_qed':
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return {'logp': 0.0, 'qed': 0.0, 'combined': 0.0}

                logp_value = float(self.logp_evaluator(smiles))
                qed_value = float(self.qed_evaluator(smiles))

                import math
                normalized_logp = 1 / (1 + math.exp(-0.5 * logp_value))
                normalized_qed = qed_value

                logp_weight = self.multi_objective_weights.get('logp', 0.5)
                qed_weight = self.multi_objective_weights.get('qed', 0.5)

                total_weight = logp_weight + qed_weight
                if total_weight > 0:
                    logp_weight /= total_weight
                    qed_weight /= total_weight

                combined_score = logp_weight * normalized_logp + qed_weight * normalized_qed

                return {
                    'logp': logp_value,
                    'qed': qed_value,
                    'combined': combined_score
                }
            except Exception as e:
                print(f"Warning: Failed to get components for {smiles}: {e}")
                return {'logp': 0.0, 'qed': 0.0, 'combined': 0.0}
        else:
            combined_score = self.evaluate(smiles)
            return {'combined': combined_score}

    def reset_cache_for_experiment(self):
        """Reset the evaluation cache and counters."""
        self.cache.clear()
        self.calls = 0
        self.total_calls = 0
        if not ORACLE_SILENT_MODE:
            print(f"Reset oracle cache for instance {self.instance_id}")

    def evaluate_all_properties(self, smiles: str) -> dict:
        """
        Evaluate all supported properties.

        Args:
            smiles: Molecule SMILES.

        Returns:
            Property-to-score mapping.
        """
        if not smiles:
            return {prop: 0.0 for prop in ['qed', 'logp', 'sa', 'jnk3', 'drd2']}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {prop: 0.0 for prop in ['qed', 'logp', 'sa', 'jnk3', 'drd2']}

        all_properties = ['qed', 'logp', 'sa', 'jnk3', 'drd2']
        return self.evaluate_specific_properties(smiles, all_properties)

    def evaluate_specific_properties(self, smiles: str, properties: list) -> dict:
        """
        Evaluate selected molecular properties.

        Args:
            smiles: Molecule SMILES.
            properties: Property names to evaluate.

        Returns:
            Property-to-score mapping.
        """
        if not smiles:
            return {prop: 0.0 for prop in properties}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {prop: 0.0 for prop in properties}

        results = {}

        for prop in properties:
            try:
                if prop not in self.property_evaluators:
                    property_to_oracle = {
                        'qed': 'QED',
                        'logp': 'LogP',
                        'sa': 'SA',
                        'jnk3': 'JNK3',
                        'drd2': 'DRD2'
                    }

                    if prop in property_to_oracle:
                        oracle_name = property_to_oracle[prop]
                        self.property_evaluators[prop] = get_or_create_tdc_oracle(oracle_name)
                    else:
                        print(f"Warning: Unknown property {prop}, setting to 0.0")
                        results[prop] = 0.0
                        continue

                evaluator = self.property_evaluators[prop]
                score = float(evaluator(smiles))
                if not math.isfinite(score):
                    if self.raise_on_error:
                        raise RuntimeError(f"{prop} oracle returned a non-finite value")
                    score = 0.0
                results[prop] = score

            except Exception as e:
                if self.raise_on_error:
                    raise RuntimeError(
                        f"{prop} oracle failed for {smiles}: {e}"
                    ) from e
                print(f"{prop.upper()} evaluation failed: {e}")
                if prop == 'sa':
                    results[prop] = 5.0
                else:
                    results[prop] = 0.0

        return results

    def get_cache_stats(self):
        """Return cache statistics."""
        cache_hits = self.total_calls - self.calls
        cache_hit_ratio = cache_hits / self.total_calls if self.total_calls > 0 else 0

        return {
            'instance_id': self.instance_id,
            'real_oracle_calls': self.calls,
            'total_calls_including_cache': self.total_calls,
            'cache_hits': cache_hits,
            'cache_size': len(self.cache),
            'cache_hit_ratio': cache_hit_ratio
        }

    def _fallback_evaluate(self, smiles, mol):
        """Evaluate locally implemented properties."""
        if self.target_property == 'mr':
            from rdkit.Chem import Descriptors
            try:
                mr_value = Descriptors.MolMR(mol)
                return mr_value
            except Exception as e:
                print(f"Warning: Failed to calculate MR for {smiles}: {e}")
                return 0.0
        elif self.target_property == 'logp_qed':
            try:
                logp_value = float(self.logp_evaluator(smiles))
                qed_value = float(self.qed_evaluator(smiles))

                import math
                normalized_logp = 1 / (1 + math.exp(-0.5 * logp_value))

                normalized_qed = qed_value

                logp_weight = self.multi_objective_weights.get('logp', 0.5)
                qed_weight = self.multi_objective_weights.get('qed', 0.5)

                total_weight = logp_weight + qed_weight
                if total_weight > 0:
                    logp_weight /= total_weight
                    qed_weight /= total_weight

                combined_score = logp_weight * normalized_logp + qed_weight * normalized_qed

                return combined_score
            except Exception as e:
                print(f"Warning: Failed to calculate LogP+QED for {smiles}: {e}")
                return 0.0
        else:
            return 0.0

    def save_cache(self, file_path):
        """Save the cache to disk."""
        with open(file_path, 'w') as f:
            yaml.dump(self.cache, f)

    def load_cache(self, file_path):
        """Load the cache from disk."""
        with open(file_path, 'r') as f:
            self.cache = yaml.safe_load(f) or {}
