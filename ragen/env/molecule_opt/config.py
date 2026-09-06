from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MoleculeEnvConfig:
    """Configuration for the Molecule Optimization Environment."""

    final_similarity_threshold: float = 0.4
    max_steps: int = 5
    minimize_property: bool = False
    max_heavy_atoms: int = 50
    max_carbon_chain_length: int = 7
    no_improvement_penalty: float = -0.1
    duplicate_detection_enabled: bool = True
    duplicate_penalty: float = -0.5
    multi_objective_weights: Dict = field(default_factory=lambda: {
        'logp': 0.5,
        'qed': 0.5,
    })
    oracle_config: Dict = field(default_factory=lambda: {
        "target_property": "qed",
        "cache_file": None,
    })
    render_mode: str = "text"
    target_property: str = "qed"
    retriever: "RetrieverConfig" = field(default_factory=lambda: RetrieverConfig())

    def __post_init__(self):
        self.supported_properties = [
            "qed", "logp", "sa", "jnk3", "drd2",
            "penalized_logp", "mr", "logp_qed",
        ]
        if self.target_property not in self.supported_properties:
            raise ValueError(f"Unsupported target_property: {self.target_property}. Supported: {self.supported_properties}")


@dataclass
class RetrieverConfig:
    """Configuration for molecular similarity retrieval."""

    url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 5
    hint_topk: int = 3
    timeout: float = 5.0
    trigger_mode: str = "on_stuck"
    no_improvement_threshold: int = 2
    rollback_threshold: int = 1
    max_triggers: int = 2  # Shared limit across static and evolving memory
    min_improvement_margin: float = 0.0

    def __post_init__(self):
        if self.trigger_mode not in {"always", "on_stuck"}:
            raise ValueError(
                "Static Exemplar Memory trigger_mode must be 'always' or 'on_stuck'"
            )
        if not self.url:
            raise ValueError("Static Exemplar Memory url must not be empty")
        for field_name in (
            'topk',
            'hint_topk',
            'no_improvement_threshold',
            'rollback_threshold',
            'max_triggers',
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"Memory retrieval {field_name} must be positive")
        if self.timeout <= 0:
            raise ValueError("Static Exemplar Memory timeout must be positive")
