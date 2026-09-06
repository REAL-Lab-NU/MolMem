from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .property_utils import (
    PROPERTY_DIFFICULTY_WEIGHTS,
    PROPERTY_OPTIMIZATION_DIRECTIONS,
    PROPERTY_SUCCESS_THRESHOLDS,
)


SINGLE_PROPERTY_SUCCESS = {
    "qed": ("maximize", 0.9),
    "logp": ("maximize", 2.0),
    "jnk3": ("maximize", 0.1),
    "drd2": ("maximize", 0.8),
    "sa": ("minimize", 2.5),
}

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_COMPARISON_TOLERANCE = 1e-12


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def tanimoto_similarity(first: str, second: str) -> float:
    first_mol = Chem.MolFromSmiles(first)
    second_mol = Chem.MolFromSmiles(second)
    if first_mol is None or second_mol is None:
        return 0.0
    first_fp = _MORGAN_GENERATOR.GetFingerprint(first_mol)
    second_fp = _MORGAN_GENERATOR.GetFingerprint(second_mol)
    return float(DataStructs.TanimotoSimilarity(first_fp, second_fp))


def directional_delta(property_name: str, before: float, after: float) -> float:
    delta = float(after) - float(before)
    direction = PROPERTY_OPTIMIZATION_DIRECTIONS.get(property_name, "maximize")
    return -delta if direction == "minimize" else delta


def candidate_utility(
    task_properties: Iterable[str],
    initial_properties: Dict[str, float],
    candidate_properties: Dict[str, float],
) -> float:
    return sum(
        PROPERTY_DIFFICULTY_WEIGHTS.get(prop, 1.0)
        * directional_delta(prop, initial_properties[prop], candidate_properties[prop])
        for prop in task_properties
    )


def meets_task_success(
    task_properties: list[str],
    initial_properties: Dict[str, float],
    candidate_properties: Dict[str, float],
) -> bool:
    if any(
        prop not in initial_properties
        or prop not in candidate_properties
        or not math.isfinite(float(initial_properties[prop]))
        or not math.isfinite(float(candidate_properties[prop]))
        for prop in task_properties
    ):
        return False

    if len(task_properties) == 1:
        prop = task_properties[0]
        direction, threshold = SINGLE_PROPERTY_SUCCESS[prop]
        value = candidate_properties[prop]
        if direction == "minimize":
            return value <= threshold + _COMPARISON_TOLERANCE
        return value + _COMPARISON_TOLERANCE >= threshold

    for prop in task_properties:
        threshold_config = PROPERTY_SUCCESS_THRESHOLDS[prop]
        raw_delta = candidate_properties[prop] - initial_properties[prop]
        if threshold_config["direction"] == "decrease":
            if raw_delta > -threshold_config["threshold"] + _COMPARISON_TOLERANCE:
                return False
        elif raw_delta + _COMPARISON_TOLERANCE < threshold_config["threshold"]:
            return False
    return True


def relative_improvement(
    task_properties: Iterable[str],
    initial_properties: Dict[str, float],
    candidate_properties: Dict[str, float],
) -> float:
    values = []
    for prop in task_properties:
        initial = float(initial_properties[prop])
        candidate = float(candidate_properties[prop])
        if not math.isfinite(initial) or not math.isfinite(candidate):
            raise ValueError(f"Non-finite {prop} value in relative-improvement input")
        improvement = directional_delta(prop, initial, candidate)
        if abs(initial) <= _COMPARISON_TOLERANCE:
            values.append(
                0.0
                if abs(improvement) <= _COMPARISON_TOLERANCE
                else math.copysign(min(abs(improvement), 10.0), improvement)
            )
        else:
            values.append(improvement / abs(initial))
    return sum(values) / len(values) if values else 0.0


@dataclass
class LeadSearchState:
    lead_id: int
    lead_smiles: str
    task_properties: list[str]
    initial_properties: Dict[str, float]
    best_smiles: str = field(init=False)
    best_properties: Dict[str, float] = field(init=False)
    best_utility: float = 0.0
    best_similarity: float = 1.0
    best_generation: int = 0
    success: bool = False
    first_success_call: Optional[int] = None

    def __post_init__(self):
        self.best_smiles = self.lead_smiles
        self.best_properties = dict(self.initial_properties)
        if meets_task_success(
            self.task_properties, self.initial_properties, self.initial_properties
        ):
            self.success = True
            self.first_success_call = 0

    def consider(self, event: Dict, generation: int) -> None:
        properties = event.get("properties", {})
        if any(prop not in properties for prop in self.task_properties):
            return
        if any(not math.isfinite(float(properties[prop])) for prop in self.task_properties):
            return

        similarity = float(event.get("similarity", 0.0))
        if not math.isfinite(similarity):
            return

        candidate_is_successful = meets_task_success(
            self.task_properties, self.initial_properties, properties
        )
        if candidate_is_successful:
            if self.first_success_call is None:
                self.first_success_call = int(event["oracle_call"])

        utility = candidate_utility(
            self.task_properties, self.initial_properties, properties
        )
        if candidate_is_successful and len(self.task_properties) > 1:
            utility += 5.0 + 2.0 * len(self.task_properties)
        better_than_incumbent = (
            utility > self.best_utility + _COMPARISON_TOLERANCE
            or (
                abs(utility - self.best_utility) <= _COMPARISON_TOLERANCE
                and similarity > self.best_similarity
            )
        )
        if better_than_incumbent:
            self.best_smiles = event["smiles"]
            self.best_properties = {prop: float(properties[prop]) for prop in self.task_properties}
            self.best_utility = float(utility)
            self.best_similarity = similarity
            self.best_generation = generation
        if candidate_is_successful:
            self.success = True

    def result(self, oracle_calls: int) -> Dict:
        return {
            "lead_id": self.lead_id,
            "lead_smiles": self.lead_smiles,
            "initial_properties": self.initial_properties,
            "best_smiles": self.best_smiles,
            "best_properties": self.best_properties,
            "success": self.success,
            "first_success_call": self.first_success_call,
            "relative_improvement": relative_improvement(
                self.task_properties,
                self.initial_properties,
                self.best_properties,
            ),
            "similarity": self.best_similarity,
            "oracle_calls": oracle_calls,
            "best_generation": self.best_generation,
        }


def aggregate_metrics(results: Iterable[Dict]) -> Dict[str, float]:
    results = list(results)
    if not results:
        return {
            "success_rate": 0.0,
            "relative_improvement": 0.0,
            "similarity": 0.0,
        }
    count = len(results)
    return {
        "success_rate": sum(bool(item["success"]) for item in results) / count,
        "relative_improvement": sum(item["relative_improvement"] for item in results) / count,
        "similarity": sum(item["similarity"] for item in results) / count,
    }
