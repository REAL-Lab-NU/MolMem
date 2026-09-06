"""Environment registry for the MolMem molecular optimization release."""

from .molecule_opt import MoleculeEnv, MoleculeEnvConfig


REGISTERED_ENVS = {
    "molecule_opt": MoleculeEnv,
}

REGISTERED_ENV_CONFIGS = {
    "molecule_opt": MoleculeEnvConfig,
}

__all__ = [
    "MoleculeEnv",
    "MoleculeEnvConfig",
    "REGISTERED_ENVS",
    "REGISTERED_ENV_CONFIGS",
]
