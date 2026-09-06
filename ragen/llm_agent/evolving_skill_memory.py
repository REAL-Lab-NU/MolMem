"""
Evolving Skill Memory for MolMem

This module implements the Evolving Skill Memory component of the MolMem
(Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization) framework.

Evolving Skill Memory distills successful optimization trajectories into reusable
strategies that can be applied to future optimization tasks.

This module provides:
1. Skill extraction from optimization trajectories
2. Functional group detection and change analysis
3. Skill storage with multiple retrieval methods (by functional group, similarity, task)
4. GPT-based strategy summarization
"""

import json
import hashlib
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import (
    Fragments,
    DataStructs,
    Descriptors,
    rdFingerprintGenerator,
    rdMolDescriptors,
    rdFMCS,
)
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

logger = logging.getLogger(__name__)


_ACTION_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    includeChirality=True,
)

MIN_SKILL_TRANSITION_SIMILARITY = 0.4


def _evidence_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "item"


def skill_transition_rejection_reason(
    before_smiles: str,
    after_smiles: str,
    score_before: float,
    score_after: float,
    score_delta: Optional[float] = None,
    objective_deltas: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Return why a transition is unsuitable for skill memory, if any."""
    molecules = {}
    for label, smiles in (("before", before_smiles), ("after", after_smiles)):
        if not isinstance(smiles, str) or not smiles.strip():
            return f"missing {label} SMILES"
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return f"invalid {label} SMILES"
            Chem.SanitizeMol(mol)
        except Exception:
            return f"unsanitizable {label} SMILES"
        if len(Chem.GetMolFrags(mol)) != 1:
            return f"disconnected {label} molecule"
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return f"dummy atom in {label} molecule"
        if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
            return f"radical in {label} molecule"
        molecules[label] = mol

    before_canonical = Chem.MolToSmiles(
        molecules["before"], canonical=True, isomericSmiles=True
    )
    after_canonical = Chem.MolToSmiles(
        molecules["after"], canonical=True, isomericSmiles=True
    )
    if before_canonical == after_canonical:
        return "unchanged canonical molecule"

    numeric_values = [score_before, score_after]
    if score_delta is not None:
        numeric_values.append(score_delta)
    if objective_deltas is not None:
        if not isinstance(objective_deltas, dict):
            return "invalid objective deltas"
        numeric_values.extend(objective_deltas.values())
    try:
        if not all(math.isfinite(float(value)) for value in numeric_values):
            return "non-finite score or objective delta"
    except (TypeError, ValueError):
        return "invalid score or objective delta"

    before_fp = GetMorganFingerprintAsBitVect(
        molecules["before"], 2, nBits=2048
    )
    after_fp = GetMorganFingerprintAsBitVect(
        molecules["after"], 2, nBits=2048
    )
    similarity = DataStructs.TanimotoSimilarity(before_fp, after_fp)
    if similarity < MIN_SKILL_TRANSITION_SIMILARITY:
        return (
            f"before/after similarity {similarity:.3f} is below "
            f"{MIN_SKILL_TRANSITION_SIMILARITY:.3f}"
        )
    return None


@lru_cache(maxsize=32768)
def structural_action_signature(
    before_smiles: str,
    after_smiles: str,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Return a graph-only signature for soft grouping of structural edits.

    The signed sparse ECFP4 count difference captures the local environments
    removed from and added to the molecule. Objective/descriptor deltas, GPT
    output and objective values are deliberately excluded. This signature is
    only a retrieval-diversity key; storage identity remains the exact
    canonical before/after pair.
    """
    before = Chem.MolFromSmiles(before_smiles)
    after = Chem.MolFromSmiles(after_smiles)
    if before is None or after is None:
        return None

    try:
        before_counts = _ACTION_FINGERPRINT_GENERATOR.GetSparseCountFingerprint(
            before
        ).GetNonzeroElements()
        after_counts = _ACTION_FINGERPRINT_GENERATOR.GetSparseCountFingerprint(
            after
        ).GetNonzeroElements()
    except Exception:
        return None

    feature_ids = set(before_counts) | set(after_counts)
    signature = tuple(sorted(
        (
            int(feature_id),
            int(after_counts.get(feature_id, 0) - before_counts.get(feature_id, 0)),
        )
        for feature_id in feature_ids
        if after_counts.get(feature_id, 0) != before_counts.get(feature_id, 0)
    ))
    return signature or None


# ============================================================
# RDKit Name Mapping (for GPT prompt clarity)
# ============================================================

RDKIT_NAME_MAP = {
    "para_hydroxylation": "Para-Hydroxylation Site",
    "Al_COO": "Aliphatic Carboxylic Acid",
    "Ar_N": "Aromatic Nitrogen",
    "C_O_noCOO": "Ketone/Aldehyde Carbonyl",
    "allylic_oxid": "Allylic Oxidation Pattern",
    "NH0": "Tertiary Amine",
    "NH1": "Secondary Amine",
    "NH2": "Primary Amine",
    "aryl_methyl": "Aryl Methyl Group",
    "Ar_NH": "Aromatic NH",
    "Ar_OH": "Aromatic Hydroxyl",
    "Al_OH": "Aliphatic Hydroxyl",
    "COO": "Carboxylic Acid/Ester",
    "COO2": "Carboxylic Acid",
    "C_O": "Carbonyl",
    "Imine": "Imine",
    "amide": "Amide",
    "aniline": "Aniline",
    "benzene": "Benzene Ring",
    "furan": "Furan Ring",
    "thiophene": "Thiophene Ring",
    "pyridine": "Pyridine Ring",
    "phenol": "Phenol",
}


def map_fg_name(name: str) -> str:
    """Map RDKit fragment name to human-readable description."""
    return RDKIT_NAME_MAP.get(name, name)


# ============================================================
# Functional Group Pruner (Hierarchy-based Cleanup)
# ============================================================

class FunctionalGroupPruner:
    """
    Removes redundant RDKit functional group tags based on chemical hierarchy.
    Example: If 'aniline' is present, remove 'benzene', 'NH0/1/2', 'Ar_N'.
    """

    # Higher-level structures and their redundant sub-components
    HIERARCHY_MAP = {
        # Aromatic + Nitrogen
        "aniline": ["benzene", "Ar_N", "Ar_NH", "NH0", "NH1", "NH2", "N_O"],
        "phenol": ["benzene", "Ar_OH", "OH"],
        "pyridine": ["Ar_N", "NH0"],
        "thiophene": ["sulfide"],
        "furan": ["ether"],

        # Carbonyl derivatives
        "amide": ["C_O", "C_O_noCOO", "NH0", "NH1", "NH2", "priamide"],
        "urea": ["amide", "C_O", "C_O_noCOO", "NH0", "NH1", "NH2"],
        "carbamate": ["ester", "amide", "C_O", "C_O_noCOO", "NH0", "NH1"],
        "alkyl_carbamate": ["ester", "amide", "C_O", "C_O_noCOO", "NH0", "NH1"],
        "ester": ["C_O", "C_O_noCOO", "ether"],
        "carboxylic_acid": ["C_O", "C_O_noCOO", "OH"],
        "ketone": ["C_O", "C_O_noCOO"],
        "aldehyde": ["C_O", "C_O_noCOO"],

        # Nitrogen structures
        "nitro": ["N_O"],
        "hydrazone": ["Imine"],

        # Ring systems
        "benzene": ["bicyclic"],
    }

    # Generic redundancy rules
    GENERIC_REDUNDANCY = {
        "COO": ["C_O", "C_O_noCOO"],
        "COO2": ["C_O", "C_O_noCOO", "COO"],
        "Al_COO": ["COO", "C_O"],
        "Ar_COO": ["COO", "C_O"],
    }

    @classmethod
    def prune(cls, tags: List[str]) -> List[str]:
        """
        Remove redundant tags based on chemical hierarchy.

        Args:
            tags: Raw list of functional group tags

        Returns:
            Cleaned list with redundancies removed
        """
        tag_set = set(tags)
        to_remove = set()

        for tag in tag_set:
            if tag in cls.HIERARCHY_MAP:
                children = cls.HIERARCHY_MAP[tag]
                to_remove.update(child for child in children if child in tag_set)

            if tag in cls.GENERIC_REDUNDANCY:
                children = cls.GENERIC_REDUNDANCY[tag]
                to_remove.update(child for child in children if child in tag_set)

        return sorted(list(tag_set - to_remove))


# ============================================================
# Functional Group Detection
# ============================================================

# Use ALL 85 RDKit built-in fragment functions for comprehensive coverage
# These are auto-generated from rdkit.Chem.Fragments module
RDKIT_FRAGMENT_FUNCS = [
    ('Al_COO', Fragments.fr_Al_COO),           # Aliphatic carboxylic acid
    ('Al_OH', Fragments.fr_Al_OH),             # Aliphatic hydroxyl
    ('Al_OH_noTert', Fragments.fr_Al_OH_noTert),  # Aliphatic hydroxyl (no tertiary)
    ('ArN', Fragments.fr_ArN),                 # Aromatic nitrogen
    ('Ar_COO', Fragments.fr_Ar_COO),           # Aromatic carboxylic acid
    ('Ar_N', Fragments.fr_Ar_N),               # N attached to aromatic
    ('Ar_NH', Fragments.fr_Ar_NH),             # Aromatic NH
    ('Ar_OH', Fragments.fr_Ar_OH),             # Aromatic hydroxyl
    ('COO', Fragments.fr_COO),                 # Carboxylic acid/ester
    ('COO2', Fragments.fr_COO2),               # Carboxylic acid
    ('C_O', Fragments.fr_C_O),                 # Carbonyl O
    ('C_O_noCOO', Fragments.fr_C_O_noCOO),     # Carbonyl O (not COOH)
    ('C_S', Fragments.fr_C_S),                 # Thiocarbonyl
    ('HOCCN', Fragments.fr_HOCCN),             # C(OH)CCN
    ('Imine', Fragments.fr_Imine),             # Imine
    ('NH0', Fragments.fr_NH0),                 # Tertiary amine
    ('NH1', Fragments.fr_NH1),                 # Secondary amine
    ('NH2', Fragments.fr_NH2),                 # Primary amine
    ('N_O', Fragments.fr_N_O),                 # N-O
    ('Ndealkylation1', Fragments.fr_Ndealkylation1),
    ('Ndealkylation2', Fragments.fr_Ndealkylation2),
    ('Nhpyrrole', Fragments.fr_Nhpyrrole),     # NH pyrrole
    ('SH', Fragments.fr_SH),                   # Thiol
    ('aldehyde', Fragments.fr_aldehyde),       # Aldehyde
    ('alkyl_carbamate', Fragments.fr_alkyl_carbamate),
    ('alkyl_halide', Fragments.fr_alkyl_halide),
    ('allylic_oxid', Fragments.fr_allylic_oxid),
    ('amide', Fragments.fr_amide),             # Amide
    ('amidine', Fragments.fr_amidine),         # Amidine
    ('aniline', Fragments.fr_aniline),         # Aniline
    ('aryl_methyl', Fragments.fr_aryl_methyl), # Aryl methyl
    ('azide', Fragments.fr_azide),             # Azide
    ('azo', Fragments.fr_azo),                 # Azo
    ('barbitur', Fragments.fr_barbitur),       # Barbiturate
    ('benzene', Fragments.fr_benzene),         # Benzene
    ('benzodiazepine', Fragments.fr_benzodiazepine),
    ('bicyclic', Fragments.fr_bicyclic),       # Bicyclic
    ('diazo', Fragments.fr_diazo),             # Diazo
    ('dihydropyridine', Fragments.fr_dihydropyridine),
    ('epoxide', Fragments.fr_epoxide),         # Epoxide
    ('ester', Fragments.fr_ester),             # Ester
    ('ether', Fragments.fr_ether),             # Ether
    ('furan', Fragments.fr_furan),             # Furan
    ('guanido', Fragments.fr_guanido),         # Guanidine
    ('halogen', Fragments.fr_halogen),         # Halogen
    ('hdrzine', Fragments.fr_hdrzine),         # Hydrazine
    ('hdrzone', Fragments.fr_hdrzone),         # Hydrazone
    ('imidazole', Fragments.fr_imidazole),     # Imidazole
    ('imide', Fragments.fr_imide),             # Imide
    ('isocyan', Fragments.fr_isocyan),         # Isocyanate
    ('isothiocyan', Fragments.fr_isothiocyan), # Isothiocyanate
    ('ketone', Fragments.fr_ketone),           # Ketone
    ('ketone_Topliss', Fragments.fr_ketone_Topliss),
    ('lactam', Fragments.fr_lactam),           # Lactam
    ('lactone', Fragments.fr_lactone),         # Lactone
    ('methoxy', Fragments.fr_methoxy),         # Methoxy
    ('morpholine', Fragments.fr_morpholine),   # Morpholine
    ('nitrile', Fragments.fr_nitrile),         # Nitrile
    ('nitro', Fragments.fr_nitro),             # Nitro
    ('nitro_arom', Fragments.fr_nitro_arom),   # Aromatic nitro
    ('nitro_arom_nonortho', Fragments.fr_nitro_arom_nonortho),
    ('nitroso', Fragments.fr_nitroso),         # Nitroso
    ('oxazole', Fragments.fr_oxazole),         # Oxazole
    ('oxime', Fragments.fr_oxime),             # Oxime
    ('para_hydroxylation', Fragments.fr_para_hydroxylation),
    ('phenol', Fragments.fr_phenol),           # Phenol
    ('phenol_noOrthoHbond', Fragments.fr_phenol_noOrthoHbond),
    ('phos_acid', Fragments.fr_phos_acid),     # Phosphoric acid
    ('phos_ester', Fragments.fr_phos_ester),   # Phosphoric ester
    ('piperdine', Fragments.fr_piperdine),     # Piperidine
    ('piperzine', Fragments.fr_piperzine),     # Piperazine
    ('priamide', Fragments.fr_priamide),       # Primary amide
    ('prisulfonamd', Fragments.fr_prisulfonamd),  # Primary sulfonamide
    ('pyridine', Fragments.fr_pyridine),       # Pyridine
    ('quatN', Fragments.fr_quatN),             # Quaternary N
    ('sulfide', Fragments.fr_sulfide),         # Sulfide
    ('sulfonamd', Fragments.fr_sulfonamd),     # Sulfonamide
    ('sulfone', Fragments.fr_sulfone),         # Sulfone
    ('term_acetylene', Fragments.fr_term_acetylene),  # Terminal acetylene
    ('tetrazole', Fragments.fr_tetrazole),     # Tetrazole
    ('thiazole', Fragments.fr_thiazole),       # Thiazole
    ('thiocyan', Fragments.fr_thiocyan),       # Thiocyanate
    ('thiophene', Fragments.fr_thiophene),     # Thiophene
    ('unbrch_alkane', Fragments.fr_unbrch_alkane),  # Unbranched alkane
    ('urea', Fragments.fr_urea),               # Urea
]

# Custom SMARTS only for structures NOT covered by RDKit Fragments
# (indole, quinoline, naphthalene, cyclohexyl, phosphate ester)
CUSTOM_SMARTS_PATTERNS = {
    'indole': 'c1ccc2[nH]ccc2c1',
    'quinoline': 'c1ccc2ncccc2c1',
    'naphthalene': 'c1ccc2ccccc2c1',
    'cyclohexyl': 'C1CCCCC1',
    'phosphate_ester': '[PX4](=[OX1])([OX2])([OX2])[OX2]',
}


def detect_functional_groups(smiles: str) -> Dict[str, int]:
    """
    Detect functional groups in a molecule using RDKit Fragments + custom SMARTS.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        Dict mapping functional group names to their counts
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    detected = {}

    for name, func in RDKIT_FRAGMENT_FUNCS:
        try:
            count = func(mol)
            if count > 0:
                detected[name] = count
        except Exception:
            continue

    for name, smarts in CUSTOM_SMARTS_PATTERNS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                detected[name] = len(matches)
        except Exception:
            continue

    return detected


def compute_fg_diff(before_smiles: str, after_smiles: str) -> Dict[str, int]:
    """
    Compute functional group changes between two molecules.

    Args:
        before_smiles: SMILES of the starting molecule
        after_smiles: SMILES of the result molecule

    Returns:
        Dict mapping functional group names to change counts
        (positive = added, negative = removed)
    """
    fg_before = detect_functional_groups(before_smiles)
    fg_after = detect_functional_groups(after_smiles)

    all_fgs = set(fg_before.keys()) | set(fg_after.keys())
    diff = {}

    for fg in all_fgs:
        change = fg_after.get(fg, 0) - fg_before.get(fg, 0)
        if change != 0:
            diff[fg] = change

    return diff


# ============================================================
# MCS and Scaffold Analysis
# ============================================================

def extract_non_mcs_fragment(mol, match_indices: tuple) -> str:
    """
    Extract atoms not in MCS as a fragment SMILES.

    Args:
        mol: RDKit molecule
        match_indices: Tuple of atom indices that are part of MCS

    Returns:
        SMILES string of non-MCS atoms, or atom symbols if extraction fails
    """
    if not match_indices or mol is None:
        return ''

    all_atoms = set(range(mol.GetNumAtoms()))
    mcs_atoms = set(match_indices)
    non_mcs_atoms = all_atoms - mcs_atoms

    if not non_mcs_atoms:
        return ''

    try:
        emol = Chem.RWMol(mol)
        atoms_to_remove = sorted(mcs_atoms, reverse=True)
        for atom_idx in atoms_to_remove:
            emol.RemoveAtom(atom_idx)

        fragment = emol.GetMol()
        if fragment and fragment.GetNumAtoms() > 0:
            try:
                Chem.SanitizeMol(fragment)
                return Chem.MolToSmiles(fragment)
            except Exception:
                pass
    except Exception:
        pass

    atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in sorted(non_mcs_atoms)]
    return ','.join(atoms)


def analyze_mcs_changes(before_smiles: str, after_smiles: str) -> Dict:
    """
    Analyze molecular changes using Maximum Common Substructure (MCS).

    Args:
        before_smiles: SMILES of starting molecule
        after_smiles: SMILES of result molecule

    Returns:
        Dict containing:
        - mcs_smarts: SMARTS string of MCS
        - mcs_num_atoms: Number of atoms in MCS
        - removed_fragment: Fragment removed from before molecule
        - added_fragment: Fragment added to after molecule
        - modification_type: 'addition', 'removal', 'replacement', or 'no_change'
    """
    mol_before = Chem.MolFromSmiles(before_smiles)
    mol_after = Chem.MolFromSmiles(after_smiles)

    if mol_before is None or mol_after is None:
        return {
            'mcs_smarts': '',
            'mcs_num_atoms': 0,
            'removed_fragment': '',
            'added_fragment': '',
            'modification_type': 'invalid_smiles'
        }

    # Compute MCS with timeout (2 seconds to avoid blocking)
    try:
        mcs_result = rdFMCS.FindMCS(
            [mol_before, mol_after],
            timeout=2,
            matchValences=True,
            ringMatchesRingOnly=True
        )
    except Exception as e:
        logger.warning(f"MCS computation failed: {type(e).__name__}: {e}")
        return {
            'mcs_smarts': '',
            'mcs_num_atoms': 0,
            'removed_fragment': '',
            'added_fragment': '',
            'modification_type': 'mcs_failed'
        }

    if mcs_result.numAtoms == 0:
        return {
            'mcs_smarts': '',
            'mcs_num_atoms': 0,
            'removed_fragment': before_smiles,
            'added_fragment': after_smiles,
            'modification_type': 'complete_replacement'
        }

    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    match_before = mol_before.GetSubstructMatch(mcs_mol) if mcs_mol else ()
    match_after = mol_after.GetSubstructMatch(mcs_mol) if mcs_mol else ()

    non_mcs_before = mol_before.GetNumAtoms() - len(match_before)
    non_mcs_after = mol_after.GetNumAtoms() - len(match_after)

    removed_fragment = extract_non_mcs_fragment(mol_before, match_before)
    added_fragment = extract_non_mcs_fragment(mol_after, match_after)

    if non_mcs_before == 0 and non_mcs_after > 0:
        mod_type = 'addition'
    elif non_mcs_before > 0 and non_mcs_after == 0:
        mod_type = 'removal'
    elif non_mcs_before > 0 and non_mcs_after > 0:
        mod_type = 'replacement'
    else:
        mod_type = 'no_change'

    return {
        'mcs_smarts': mcs_result.smartsString,
        'mcs_num_atoms': mcs_result.numAtoms,
        'removed_fragment': removed_fragment,
        'added_fragment': added_fragment,
        'modification_type': mod_type
    }


def analyze_scaffold_changes(before_smiles: str, after_smiles: str) -> Dict:
    """
    Analyze scaffold changes using Murcko decomposition.

    Args:
        before_smiles: SMILES of starting molecule
        after_smiles: SMILES of result molecule

    Returns:
        Dict containing:
        - before_scaffold: Murcko scaffold of before molecule
        - after_scaffold: Murcko scaffold of after molecule
        - scaffold_changed: Whether scaffold changed
        - is_scaffold_hop: True if scaffold changed but generic scaffold same (bioisosteric)
        - scaffold_type: 'side_chain_mod', 'scaffold_hop', 'ring_fusion', 'ring_removal', 'scaffold_replacement'
    """
    mol_before = Chem.MolFromSmiles(before_smiles)
    mol_after = Chem.MolFromSmiles(after_smiles)

    if mol_before is None or mol_after is None:
        return {
            'before_scaffold': '',
            'after_scaffold': '',
            'scaffold_changed': False,
            'is_scaffold_hop': False,
            'scaffold_type': 'invalid_smiles'
        }

    try:
        scaffold_before = GetScaffoldForMol(mol_before)
        scaffold_after = GetScaffoldForMol(mol_after)
    except Exception as e:
        logger.warning(f"Scaffold extraction failed: {type(e).__name__}: {e}")
        return {
            'before_scaffold': '',
            'after_scaffold': '',
            'scaffold_changed': False,
            'is_scaffold_hop': False,
            'scaffold_type': 'scaffold_failed'
        }

    scaffold_before_smiles = Chem.MolToSmiles(scaffold_before) if scaffold_before and scaffold_before.GetNumAtoms() > 0 else ''
    scaffold_after_smiles = Chem.MolToSmiles(scaffold_after) if scaffold_after and scaffold_after.GetNumAtoms() > 0 else ''

    scaffold_changed = scaffold_before_smiles != scaffold_after_smiles

    generic_same = False
    if scaffold_before and scaffold_after and scaffold_before.GetNumAtoms() > 0 and scaffold_after.GetNumAtoms() > 0:
        try:
            generic_before = MakeScaffoldGeneric(scaffold_before)
            generic_after = MakeScaffoldGeneric(scaffold_after)
            generic_same = Chem.MolToSmiles(generic_before) == Chem.MolToSmiles(generic_after)
        except Exception:
            generic_same = False

    if not scaffold_changed:
        scaffold_type = 'side_chain_mod'
    elif generic_same:
        scaffold_type = 'scaffold_hop'  # e.g., Benzene → Pyridine
    else:
        rings_before = mol_before.GetRingInfo().NumRings()
        rings_after = mol_after.GetRingInfo().NumRings()
        if rings_after > rings_before:
            scaffold_type = 'ring_fusion'
        elif rings_after < rings_before:
            scaffold_type = 'ring_removal'
        else:
            scaffold_type = 'scaffold_replacement'

    return {
        'before_scaffold': scaffold_before_smiles,
        'after_scaffold': scaffold_after_smiles,
        'scaffold_changed': scaffold_changed,
        'is_scaffold_hop': scaffold_type == 'scaffold_hop',
        'scaffold_type': scaffold_type
    }


def analyze_transformation(before_smiles: str, after_smiles: str) -> Dict:
    """
    Analyze structural transformation between two molecules.

    Returns:
        Dict containing atom changes, ring changes, MW change, and FG changes
    """
    mol_before = Chem.MolFromSmiles(before_smiles)
    mol_after = Chem.MolFromSmiles(after_smiles)

    if mol_before is None or mol_after is None:
        return {}

    atom_counts_before = defaultdict(int)
    atom_counts_after = defaultdict(int)

    for atom in mol_before.GetAtoms():
        atom_counts_before[atom.GetSymbol()] += 1
    for atom in mol_after.GetAtoms():
        atom_counts_after[atom.GetSymbol()] += 1

    all_atoms = set(atom_counts_before.keys()) | set(atom_counts_after.keys())
    atom_changes = {}
    for atom in all_atoms:
        change = atom_counts_after.get(atom, 0) - atom_counts_before.get(atom, 0)
        if change != 0:
            atom_changes[atom] = change

    ring_before = mol_before.GetRingInfo().NumRings()
    ring_after = mol_after.GetRingInfo().NumRings()
    ring_changes = ring_after - ring_before

    mw_before = Descriptors.MolWt(mol_before)
    mw_after = Descriptors.MolWt(mol_after)
    mw_change = mw_after - mw_before

    try:
        psa_change = Descriptors.TPSA(mol_after) - Descriptors.TPSA(mol_before)
    except Exception:
        psa_change = 0.0

    try:
        hbd_change = rdMolDescriptors.CalcNumHBD(mol_after) - rdMolDescriptors.CalcNumHBD(mol_before)
    except Exception:
        hbd_change = 0

    try:
        hba_change = rdMolDescriptors.CalcNumHBA(mol_after) - rdMolDescriptors.CalcNumHBA(mol_before)
    except Exception:
        hba_change = 0

    try:
        rotatable_bonds_change = rdMolDescriptors.CalcNumRotatableBonds(mol_after) - rdMolDescriptors.CalcNumRotatableBonds(mol_before)
    except Exception:
        rotatable_bonds_change = 0

    try:
        stereo_before = len(Chem.FindMolChiralCenters(mol_before, includeUnassigned=True))
        stereo_after = len(Chem.FindMolChiralCenters(mol_after, includeUnassigned=True))
        stereo_centers_change = stereo_after - stereo_before
    except Exception:
        stereo_centers_change = 0

    fg_diff = compute_fg_diff(before_smiles, after_smiles)
    fg_removed = [fg for fg, change in fg_diff.items() if change < 0]
    fg_added = [fg for fg, change in fg_diff.items() if change > 0]

    mcs_analysis = analyze_mcs_changes(before_smiles, after_smiles)

    scaffold_analysis = analyze_scaffold_changes(before_smiles, after_smiles)

    return {
        'atom_changes': atom_changes,
        'ring_changes': ring_changes,
        'mw_change': mw_change,
        'psa_change': psa_change,
        'hbd_change': hbd_change,
        'hba_change': hba_change,
        'rotatable_bonds_change': rotatable_bonds_change,
        'stereo_centers_change': stereo_centers_change,
        'fg_removed': fg_removed,
        'fg_added': fg_added,
        'fg_diff': fg_diff,
        'mcs_smarts': mcs_analysis.get('mcs_smarts', ''),
        'removed_fragment': mcs_analysis.get('removed_fragment', ''),
        'added_fragment': mcs_analysis.get('added_fragment', ''),
        'modification_type': mcs_analysis.get('modification_type', ''),
        'before_scaffold': scaffold_analysis.get('before_scaffold', ''),
        'after_scaffold': scaffold_analysis.get('after_scaffold', ''),
        'scaffold_changed': scaffold_analysis.get('scaffold_changed', False),
        'is_scaffold_hop': scaffold_analysis.get('is_scaffold_hop', False),
        'scaffold_type': scaffold_analysis.get('scaffold_type', ''),
    }


# ============================================================
# Data Structures
# ============================================================

def classify_multi_objective_transition(
    step_deltas: Dict[str, float],
    directions: Dict[str, str],
    min_delta: float = 0.1,
) -> Tuple[bool, float, Dict[str, float]]:
    """Accept a multi-objective step only when every objective improves.

    Returned per-objective deltas are direction-normalized, so positive always
    means improvement. Every objective must reach ``min_delta``; trade-offs and
    unchanged objectives are discarded rather than stored as skills.
    """
    objective_deltas = {
        name: (-float(value) if directions.get(name, 'maximize') == 'minimize' else float(value))
        for name, value in step_deltas.items()
    }
    if not objective_deltas:
        return False, 0.0, {}
    if any(name not in objective_deltas for name in directions):
        return False, 0.0, objective_deltas

    accepted = all(value >= min_delta for value in objective_deltas.values())
    aggregate_delta = sum(objective_deltas.values()) if accepted else 0.0
    return accepted, aggregate_delta, objective_deltas


@dataclass
class SkillEntry:
    """Single skill entry for molecular optimization (Evolving Skill Memory)"""
    # Basic info
    exp_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""  # e.g., "sa", "qed", "jnk3", "qed+logp"
    created_epoch: int = 0

    # Molecular transformation
    before_smiles: str = ""
    after_smiles: str = ""
    trajectory: List[str] = field(default_factory=list)  # Full trajectory [s0, s1, s2, ...]

    # Score changes
    score_before: float = 0.0
    score_after: float = 0.0
    score_delta: float = 0.0  # after - before
    # Positive deltas are direction-normalized improvements.
    objective_deltas: Dict[str, float] = field(default_factory=dict)

    @property
    def improvement_delta(self) -> float:
        """Observed improvement used for ranking and capacity control."""
        if self.objective_deltas:
            return sum(float(value) for value in self.objective_deltas.values())
        return -float(self.score_delta) if self.task == 'sa' else float(self.score_delta)

    # Structural analysis (for GPT context)
    atom_changes: Dict[str, int] = field(default_factory=dict)  # {"C": -3, "N": +1}
    ring_changes: int = 0
    mw_change: float = 0.0
    psa_change: float = 0.0           # Polar surface area change
    hbd_change: int = 0               # H-bond donor change
    hba_change: int = 0               # H-bond acceptor change
    rotatable_bonds_change: int = 0   # Rotatable bonds change
    stereo_centers_change: int = 0    # Stereo centers change
    functional_groups_removed: List[str] = field(default_factory=list)
    functional_groups_added: List[str] = field(default_factory=list)

    # MCS analysis (Maximum Common Substructure)
    mcs_smarts: str = ""              # SMARTS of common structure
    removed_fragment: str = ""        # Fragment removed from before molecule
    added_fragment: str = ""          # Fragment added to after molecule
    modification_type: str = ""       # 'addition', 'removal', 'replacement', 'no_change'

    # Scaffold analysis (Murcko decomposition)
    before_scaffold: str = ""         # Murcko scaffold of before molecule
    after_scaffold: str = ""          # Murcko scaffold of after molecule
    scaffold_changed: bool = False    # Whether scaffold changed
    is_scaffold_hop: bool = False     # True if bioisosteric replacement (e.g., Benzene→Pyridine)
    scaffold_type: str = ""           # 'side_chain_mod', 'scaffold_hop', 'ring_fusion', etc.

    # GPT-generated summary and its locally verified evidence binding.
    gpt_summary: Optional[str] = None
    gpt_evidence_ids: List[str] = field(default_factory=list)
    gpt_evidence_hash: str = ""
    gpt_summary_plan: Dict[str, Any] = field(default_factory=dict)

    # Merge tracking fields
    occurrence_count: int = 1  # Number of times this experience pattern was observed
    merged_from: List[str] = field(default_factory=list)  # List of exp_ids merged into this

    # Retrieval indices (computed on demand)
    before_fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    functional_group_tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        if d['before_fingerprint'] is not None:
            d['before_fingerprint'] = d['before_fingerprint'].tolist()
        d['functional_group_tags'] = list(d['functional_group_tags'])
        return d

    def to_dict_for_json(self) -> Dict:
        """Convert to JSON-serializable dict without fingerprint (for human readability)"""
        d = self.to_dict()
        d.pop('before_fingerprint', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillEntry':
        """Create a validated skill entry from serialized data."""
        if not isinstance(data, dict):
            raise ValueError("SkillEntry payload must be a dictionary")
        data = dict(data)
        unknown_fields = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown_fields:
            raise ValueError(f"Unknown SkillEntry fields: {unknown_fields}")

        required_fields = ['before_smiles', 'after_smiles', 'task']
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        if data.get('before_fingerprint') is not None:
            data['before_fingerprint'] = np.asarray(data['before_fingerprint'])
        data['functional_group_tags'] = set(
            data.get('functional_group_tags') or []
        )

        data.setdefault('occurrence_count', 1)
        data.setdefault('merged_from', [])
        data.setdefault('exp_id', str(uuid.uuid4()))
        data.setdefault('created_epoch', 0)
        data.setdefault('trajectory', [])
        data.setdefault('score_before', 0.0)
        data.setdefault('score_after', 0.0)
        data.setdefault('score_delta', 0.0)
        data.setdefault('objective_deltas', {})
        data.setdefault('atom_changes', {})
        data.setdefault('ring_changes', 0)
        data.setdefault('mw_change', 0.0)
        data.setdefault('psa_change', 0.0)
        data.setdefault('hbd_change', 0)
        data.setdefault('hba_change', 0)
        data.setdefault('rotatable_bonds_change', 0)
        data.setdefault('stereo_centers_change', 0)

        data.setdefault('mcs_smarts', '')
        data.setdefault('removed_fragment', '')
        data.setdefault('added_fragment', '')
        data.setdefault('modification_type', '')

        data.setdefault('before_scaffold', '')
        data.setdefault('after_scaffold', '')
        data.setdefault('scaffold_changed', False)
        data.setdefault('is_scaffold_hop', False)
        data.setdefault('scaffold_type', '')

        try:
            data['psa_change'] = float(data['psa_change'])
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid numeric field: psa_change") from exc
        for field_name in ['hbd_change', 'hba_change', 'rotatable_bonds_change', 'stereo_centers_change']:
            try:
                data[field_name] = int(data[field_name])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric field: {field_name}") from exc
        data.setdefault('functional_groups_removed', [])
        data.setdefault('functional_groups_added', [])
        data.setdefault('gpt_summary', None)
        data.setdefault('gpt_evidence_ids', [])
        data.setdefault('gpt_evidence_hash', '')
        data.setdefault('gpt_summary_plan', {})
        data.setdefault('before_fingerprint', None)
        data.setdefault('functional_group_tags', set())

        try:
            return cls(**data)
        except TypeError as e:
            logger.error(f"Invalid data format for SkillEntry: {e}")
            raise ValueError(f"Invalid data format: {e}")

    def compute_indices(self):
        """Compute fingerprint and FG tags for retrieval"""
        self.before_fingerprint = None
        self.functional_group_tags = set()
        mol = Chem.MolFromSmiles(self.before_smiles)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            self.before_fingerprint = np.zeros(2048, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, self.before_fingerprint)

        fg_before = detect_functional_groups(self.before_smiles)
        self.functional_group_tags = set(
            FunctionalGroupPruner.prune(list(fg_before.keys()))
        )


@lru_cache(maxsize=32768)
def _source_action_prerequisites(
    change_ids: Tuple[str, ...],
    atom_changes: Tuple[Tuple[str, int], ...],
    ring_changes: int,
) -> Tuple[frozenset, frozenset, bool]:
    change_id_set = set(change_ids)
    required_fg_ids = frozenset(
        evidence_id.removeprefix("fg.removed.")
        for evidence_id in change_id_set
        if evidence_id.startswith("fg.removed.")
    )
    required_atoms = frozenset(
        symbol
        for symbol, delta in atom_changes
        if delta < 0 and f"atoms.{_evidence_slug(symbol)}" in change_id_set
    )
    requires_ring = "rings.delta" in change_id_set and ring_changes < 0
    return required_fg_ids, required_atoms, requires_ring


def _query_action_profile(
    query_functional_groups: Optional[Dict[str, int]],
    query_mol: Optional[Chem.Mol],
) -> Tuple[Optional[frozenset], Optional[frozenset], Optional[bool]]:
    query_fg_ids = None
    if query_functional_groups is not None:
        query_fg_ids = frozenset(
            _evidence_slug(name)
            for name, count in query_functional_groups.items()
            if count > 0
        )
    if query_mol is None:
        return query_fg_ids, None, None
    query_atoms = frozenset(atom.GetSymbol() for atom in query_mol.GetAtoms())
    has_ring = query_mol.GetRingInfo().NumRings() > 0
    return query_fg_ids, query_atoms, has_ring


def _source_action_matches_profile(
    exp: SkillEntry,
    query_profile: Tuple[Optional[frozenset], Optional[frozenset], Optional[bool]],
) -> bool:
    change_ids = tuple(sorted(
        (exp.gpt_summary_plan or {}).get("change_evidence_ids") or []
    ))
    atom_changes = []
    for symbol, delta in (exp.atom_changes or {}).items():
        try:
            atom_changes.append((str(symbol), int(delta)))
        except (TypeError, ValueError):
            continue
    try:
        ring_changes = int(exp.ring_changes)
    except (TypeError, ValueError):
        ring_changes = 0
    required_fg_ids, required_atoms, requires_ring = (
        _source_action_prerequisites(
            change_ids,
            tuple(sorted(atom_changes)),
            ring_changes,
        )
    )
    query_fg_ids, query_atoms, has_ring = query_profile
    return (
        (query_fg_ids is None or required_fg_ids.issubset(query_fg_ids))
        and (query_atoms is None or required_atoms.issubset(query_atoms))
        and (has_ring is None or not requires_ring or has_ring)
    )


# ============================================================
# Evolving Skill Memory
# ============================================================

class EvolvingSkillMemory:
    """
    Evolving Skill Memory for storing and retrieving molecular optimization skills.

    This component distills successful optimization trajectories into reusable
    strategies that can be applied to future optimization tasks.

    Supports retrieval by functional group and molecular similarity.
    Thread-safe for concurrent access.
    """

    def __init__(self,
                 max_size: int = 1000,
                 save_path: Optional[str] = None,
                 min_score_delta: float = 0.1,
                 read_only: bool = False):
        """
        Args:
            max_size: Maximum number of experiences to store
            save_path: Path to save/load the memory
            min_score_delta: Minimum score change to record
            read_only: If True, memory only loads from file and cannot be updated
        """
        if max_size <= 0:
            raise ValueError("Evolving Skill Memory max_size must be positive")
        if min_score_delta <= 0:
            raise ValueError("Evolving Skill Memory min_score_delta must be positive")
        self.max_size = max_size
        self.save_path = save_path
        self.min_score_delta = min_score_delta
        self.read_only = read_only

        self._lock = threading.RLock()
        # Serialize full saves so an older snapshot cannot overwrite a newer one.
        self._save_lock = threading.Lock()
        self._summary_generation_lock = threading.Lock()

        self.experiences: Dict[str, List[SkillEntry]] = defaultdict(list)

        # Exact-pair deduplication is the only supported merge operation.
        self._canonical_pair_index: Dict[Tuple[Tuple[str, ...], str, str], SkillEntry] = {}

        _, pkl_path = self._storage_paths(save_path) if save_path else (None, None)
        if pkl_path and os.path.exists(pkl_path):
            self.load()
        elif self.read_only and save_path:
            logger.warning(f"Evolving Skill Memory file not found: {save_path}")

    @staticmethod
    def _canonical_smiles(smiles: str) -> Optional[str]:
        """Return stereochemistry-preserving canonical SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _get_smiles_key(
        self, exp: SkillEntry
    ) -> Optional[Tuple[Tuple[str, ...], str, str]]:
        """Return an exact task + canonical before/after transformation key."""
        before = self._canonical_smiles(exp.before_smiles)
        after = self._canonical_smiles(exp.after_smiles)
        if before is None or after is None:
            return None
        return self._task_signature(exp.task), before, after

    @staticmethod
    def _task_signature(task: str) -> Tuple[str, ...]:
        """Return an order-independent signature for a task expression."""
        return tuple(sorted(part.strip() for part in task.split('+') if part.strip()))

    @staticmethod
    def select_distinct_actions(
        experiences: List[SkillEntry],
        top_k: int,
    ) -> List[SkillEntry]:
        """Keep the first ranked representative of each structural action.

        An unavailable signature fails open by using the experience ID, so an
        analysis failure can reduce diversity but can never collapse unrelated
        experiences.
        """
        if top_k <= 0:
            return []

        selected = []
        seen_actions = set()
        for exp in experiences:
            signature = structural_action_signature(
                exp.before_smiles,
                exp.after_smiles,
            )
            action_key = (
                ("action", signature)
                if signature is not None
                else ("experience", exp.exp_id)
            )
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            selected.append(exp)
            if len(selected) == top_k:
                break
        return selected

    def _experiences_for_task(self, task: str) -> List[SkillEntry]:
        signature = self._task_signature(task)
        return [
            exp
            for stored_task, experiences in self.experiences.items()
            if self._task_signature(stored_task) == signature
            for exp in experiences
        ]

    def has_task(self, task: str) -> bool:
        """Return whether validated retrievable memory exists for the task."""
        return any(
            self._is_retrievable(exp)
            for exp in self._experiences_for_task(task)
        )

    def _merge_experience(self, existing: SkillEntry, new: SkillEntry) -> bool:
        """
        Merge an exact canonical pair, keeping the strongest normalized gain.

        Args:
            existing: The skill already in memory
            new: The new experience to merge

        Returns:
            True (always succeeds)
        """
        existing.occurrence_count += max(1, int(new.occurrence_count))
        existing.merged_from.extend([new.exp_id, *new.merged_from])

        if len(existing.merged_from) > 100:
            existing.merged_from = existing.merged_from[-100:]

        is_better = new.improvement_delta > existing.improvement_delta

        if is_better:
            # Move all transformation-dependent state with the new representative.
            replacement_fields = (
                'created_epoch', 'before_smiles', 'after_smiles', 'trajectory',
                'score_before', 'score_after', 'score_delta',
                'objective_deltas', 'atom_changes', 'ring_changes', 'mw_change',
                'psa_change', 'hbd_change', 'hba_change',
                'rotatable_bonds_change', 'stereo_centers_change',
                'functional_groups_removed', 'functional_groups_added',
                'mcs_smarts', 'removed_fragment', 'added_fragment',
                'modification_type', 'before_scaffold', 'after_scaffold',
                'scaffold_changed', 'is_scaffold_hop', 'scaffold_type',
                'before_fingerprint', 'functional_group_tags',
                'gpt_summary', 'gpt_evidence_ids',
                'gpt_evidence_hash', 'gpt_summary_plan',
            )
            for field_name in replacement_fields:
                setattr(existing, field_name, getattr(new, field_name))
        logger.debug(f"Merged experience {new.exp_id} into {existing.exp_id}, count={existing.occurrence_count}")
        return True

    @staticmethod
    def _is_retrievable(exp: SkillEntry) -> bool:
        return (
            bool(exp.gpt_summary and exp.gpt_summary.strip())
            and bool(exp.gpt_evidence_ids)
            and bool(exp.gpt_evidence_hash)
            and bool(exp.gpt_summary_plan)
        )

    def _rebuild_canonical_pair_index_locked(self) -> None:
        self._canonical_pair_index = {}
        for experiences in self.experiences.values():
            for exp in experiences:
                key = self._get_smiles_key(exp)
                if key is not None and key not in self._canonical_pair_index:
                    self._canonical_pair_index[key] = exp

    def add_experience(self, exp: SkillEntry) -> bool:
        """
        Add a new skill to memory (thread-safe).

        Args:
            exp: SkillEntry to add

        Returns:
            True if added, False if filtered out
        """
        if self.read_only:
            return False

        if skill_transition_rejection_reason(
            exp.before_smiles,
            exp.after_smiles,
            exp.score_before,
            exp.score_after,
            exp.score_delta,
            exp.objective_deltas,
        ) is not None:
            return False

        if not self._is_retrievable(exp):
            return False
        record, _ = validate_gpt_summary_payload(exp, exp.gpt_summary_plan)
        if record is None or record['summary'] != exp.gpt_summary:
            return False
        exp.gpt_summary = str(record['summary'])
        exp.gpt_evidence_ids = list(record['evidence_ids'])
        exp.gpt_evidence_hash = str(record['evidence_hash'])
        exp.gpt_summary_plan = dict(record['summary_plan'])

        # Every direction-normalized objective must meet the success margin.
        task_properties = exp.task.split('+')
        if exp.objective_deltas:
            if set(exp.objective_deltas) != set(task_properties):
                return False
            accepted = all(
                exp.objective_deltas[prop] >= self.min_score_delta
                for prop in task_properties
            )
        elif len(task_properties) > 1:
            return False
        elif exp.task == 'sa':
            accepted = -exp.score_delta >= self.min_score_delta
        else:
            accepted = exp.score_delta >= self.min_score_delta
        if not accepted:
            return False

        # Retrieval indices are always derived from the source molecule.
        try:
            exp.compute_indices()
        except Exception as e:
            logger.warning(f"Failed to compute indices for experience: {e}")
            return False
        if exp.before_fingerprint is None:
            return False

        with self._lock:
            task_exps = self.experiences[exp.task]

            smiles_key = self._get_smiles_key(exp)
            existing = (
                self._canonical_pair_index.get(smiles_key)
                if smiles_key is not None
                else None
            )
            if existing is not None:
                return self._merge_experience(existing, exp)

            task_exps.append(exp)
            if smiles_key is not None:
                self._canonical_pair_index[smiles_key] = exp

            if len(task_exps) > self.max_size:
                self._trim_task_locked(exp.task)
                self._rebuild_canonical_pair_index_locked()

        return True

    def _trim_task_locked(self, task: str) -> None:
        """Keep the highest-improvement skills for one task."""
        ranked = sorted(
            self.experiences[task],
            key=lambda exp: exp.improvement_delta,
            reverse=True,
        )
        self.experiences[task] = ranked[:self.max_size]

    def query_by_functional_group(self,
                                   current_fgs: Dict[str, int],
                                   task: str,
                                   top_k: int = 3,
                                   min_jaccard: float = 0.8,
                                   smiles: Optional[str] = None) -> List[SkillEntry]:
        """
        Query experiences by matching functional groups.

        Uses Jaccard similarity as a relevance filter, then ranks by the
        observed improvement of the source edit.

        Args:
            current_fgs: Detected FGs in current molecule {fg_name: count}
            task: Task type to filter by
            top_k: Number of results to return
            min_jaccard: Minimum Jaccard similarity threshold
            smiles: Current molecule used to validate source-side action
                requirements
        Returns:
            List of relevant experiences sorted by improvement delta
        """
        task_experiences = self._experiences_for_task(task)
        if not task_experiences:
            return []

        query_mol = Chem.MolFromSmiles(smiles) if smiles else None
        if smiles and query_mol is None:
            return []
        query_profile = _query_action_profile(current_fgs, query_mol)
        current_fg_set = set(FunctionalGroupPruner.prune(list(current_fgs.keys())))
        results = []

        for exp in task_experiences:
            if not self._is_retrievable(exp):
                continue
            if not _source_action_matches_profile(exp, query_profile):
                continue
            intersection = len(current_fg_set & exp.functional_group_tags)
            union = len(current_fg_set | exp.functional_group_tags)
            if intersection > 0 and union > 0:
                jaccard_sim = intersection / union

                if jaccard_sim < min_jaccard:
                    continue

                results.append((exp, jaccard_sim))

        results.sort(
            key=lambda item: (item[0].improvement_delta, item[1]),
            reverse=True,
        )
        ranked = [exp for exp, _ in results]
        return self.select_distinct_actions(ranked, top_k)

    def query_by_similarity(self,
                            smiles: str,
                            task: str,
                            top_k: int = 2,
                            min_similarity: float = 0.5) -> List[SkillEntry]:
        """
        Query experiences by molecular similarity.

        Uses Tanimoto similarity as a relevance filter, then ranks by the
        observed improvement of the source edit.

        Args:
            smiles: SMILES of current molecule
            task: Task type
            top_k: Number of results
            min_similarity: Minimum Tanimoto similarity threshold
        Returns:
            List of similar experiences sorted by improvement delta
        """
        task_experiences = self._experiences_for_task(task)
        if not task_experiences:
            return []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        query_fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        query_arr = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(query_fp, query_arr)
        current_fgs = detect_functional_groups(smiles)
        query_profile = _query_action_profile(current_fgs, mol)

        results = []
        for exp in task_experiences:
            if not self._is_retrievable(exp):
                continue
            if not _source_action_matches_profile(exp, query_profile):
                continue
            if exp.before_fingerprint is None:
                continue

            intersection = np.sum(query_arr & exp.before_fingerprint)
            union = np.sum(query_arr | exp.before_fingerprint)
            similarity = intersection / union if union > 0 else 0

            if similarity >= min_similarity:
                results.append((exp, similarity))

        results.sort(
            key=lambda item: (item[0].improvement_delta, item[1]),
            reverse=True,
        )
        ranked = [exp for exp, _ in results]
        return self.select_distinct_actions(ranked, top_k)

    def generate_gpt_summaries(
        self,
        experiences: List[SkillEntry],
        total_timeout: int = 600,
        max_workers: int = 8,
        model: Optional[str] = None,
        credential_env_prefix: str = "AZURE_OPENAI",
        max_completion_tokens: int = 150,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Summarize a detached candidate batch exactly once."""
        with self._summary_generation_lock:
            return self._generate_gpt_summaries_once(
                experiences=experiences,
                total_timeout=total_timeout,
                max_workers=max_workers,
                model=model,
                credential_env_prefix=credential_env_prefix,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )

    def _generate_gpt_summaries_once(
        self,
        experiences: List[SkillEntry],
        total_timeout: int = 600,
        max_workers: int = 8,
        model: Optional[str] = None,
        credential_env_prefix: str = "AZURE_OPENAI",
        max_completion_tokens: int = 150,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate evidence-validated summaries for every candidate in one pass.

        Args:
            experiences: Detached transformations collected in the current step.
            total_timeout: Maximum total time in seconds (default 10 minutes)
            max_workers: Number of parallel requests (default 8)
            model: Azure deployment name. Defaults to the deployment variable
                under ``credential_env_prefix`` and finally ``gpt-4o``.
            credential_env_prefix: Prefix for the Azure API key, endpoint,
                API version, and deployment environment variables.
            max_completion_tokens: Completion budget for each summary.
            reasoning_effort: Optional reasoning effort for GPT-5 deployments.

        Returns:
            Dict mapping candidate exp_id to validated summary record. Each
            returned record has also been attached to the detached entry.
        """
        import time
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        if total_timeout <= 0:
            raise ValueError("total_timeout must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        exp_ids = [exp.exp_id for exp in experiences]
        if len(exp_ids) != len(set(exp_ids)):
            raise ValueError("Candidate experience IDs must be unique")

        experiences = [
            exp for exp in experiences
            if skill_transition_rejection_reason(
                exp.before_smiles,
                exp.after_smiles,
                exp.score_before,
                exp.score_after,
                exp.score_delta,
                exp.objective_deltas,
            ) is None
        ]

        # Collapse exact pairs before incurring API cost, retaining the edit
        # with the largest observed improvement. Existing stronger cards need
        # no new summary.
        best_by_key: Dict[Any, SkillEntry] = {}
        for exp in experiences:
            if self._is_retrievable(exp):
                continue
            key = self._get_smiles_key(exp)
            candidate_key = ("pair", key) if key is not None else ("id", exp.exp_id)
            existing = best_by_key.get(candidate_key)
            if existing is None or exp.improvement_delta > existing.improvement_delta:
                best_by_key[candidate_key] = exp

        with self._lock:
            active_pairs = dict(self._canonical_pair_index)
        candidates = [
            exp
            for candidate_key, exp in best_by_key.items()
            if candidate_key[0] != "pair"
            or candidate_key[1] not in active_pairs
            or exp.improvement_delta > active_pairs[candidate_key[1]].improvement_delta
        ]
        candidates.sort(key=lambda exp: exp.improvement_delta, reverse=True)

        if not candidates:
            return {}

        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                AzureOpenAI,
                BadRequestError,
                PermissionDeniedError,
                RateLimitError,
            )
        except ImportError:
            raise RuntimeError(
                "openai is required for evidence-validated Evolving Skill Memory"
            )

        env_prefix = credential_env_prefix.rstrip('_')
        api_key_name = f"{env_prefix}_API_KEY"
        endpoint_name = f"{env_prefix}_ENDPOINT"
        api_version_name = f"{env_prefix}_API_VERSION"
        deployment_name = f"{env_prefix}_DEPLOYMENT"

        api_key = os.environ.get(api_key_name)
        endpoint = os.environ.get(endpoint_name)
        if not api_key or not endpoint:
            raise RuntimeError(
                f"GPT summarization requires both {api_key_name} and {endpoint_name}"
            )

        deployment = model or os.environ.get(deployment_name) or "gpt-4o"

        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=os.environ.get(api_version_name, "2024-02-15-preview"),
                azure_endpoint=endpoint,
                timeout=30.0,
                max_retries=0,
            )
        except Exception as e:
            raise RuntimeError("Failed to create Azure OpenAI client") from e

        deadline = time.monotonic() + float(total_timeout)
        stop_event = threading.Event()

        def process_single_exp(exp: SkillEntry) -> tuple:
            """Return exp_id, status, validated record/reason."""
            catalog = build_summary_evidence(exp)
            kinds = {item["kind"] for item in catalog.values()}
            if "objective" not in kinds or len(kinds - {"objective"}) == 0:
                return exp.exp_id, "rejected", "insufficient_rdkit_evidence"

            prompt = format_gpt_prompt(exp)
            max_retries = 2

            for attempt in range(max_retries):
                remaining = deadline - time.monotonic()
                if stop_event.is_set() or remaining <= 0:
                    return exp.exp_id, "retryable", "batch_timeout"
                try:
                    request = {
                        "model": deployment,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": _structured_summary_response_format(catalog),
                        "timeout": min(30.0, max(0.1, remaining)),
                    }
                    if deployment.lower().startswith("gpt-5"):
                        request["max_completion_tokens"] = int(max_completion_tokens)
                        if reasoning_effort:
                            request["reasoning_effort"] = reasoning_effort
                    else:
                        request["max_tokens"] = int(max_completion_tokens)
                        request["temperature"] = 0.3

                    response = client.chat.completions.create(**request)
                    message = response.choices[0].message
                    if getattr(message, "refusal", None):
                        return exp.exp_id, "rejected", "model_refusal"
                    content = (message.content or "").strip()
                    try:
                        payload = json.loads(content)
                    except (TypeError, json.JSONDecodeError):
                        return exp.exp_id, "rejected", "invalid_json"
                    record, error = validate_gpt_summary_payload(exp, payload)
                    if record is None:
                        return exp.exp_id, "rejected", error
                    return exp.exp_id, "validated", record

                except RateLimitError:
                    if attempt < max_retries - 1:
                        stop_event.wait(min(2.0, max(0.0, deadline - time.monotonic())))
                    continue
                except (APITimeoutError, APIConnectionError):
                    if attempt < max_retries - 1:
                        stop_event.wait(min(1.0, max(0.0, deadline - time.monotonic())))
                    continue
                except AuthenticationError as e:
                    logger.error(f"Authentication failed: {e}")
                    return exp.exp_id, "fatal", "authentication_error"
                except PermissionDeniedError as e:
                    logger.error(f"GPT permission denied: {e}")
                    return exp.exp_id, "fatal", "permission_denied"
                except BadRequestError as e:
                    logger.debug(f"GPT structured-output request rejected: {e}")
                    return exp.exp_id, "rejected", "bad_request"
                except Exception as e:
                    logger.debug(f"GPT failed for {exp.exp_id}: {e}")
                    return exp.exp_id, "retryable", type(e).__name__

            return exp.exp_id, "retryable", "transient_api_error"

        summaries: Dict[str, Dict[str, Any]] = {}
        fatal_reason = ""
        timed_out = False
        request_rejected = False
        transient_failure = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
        candidate_iter = iter(candidates)
        in_flight = {}

        def fill_worker_slots() -> None:
            while len(in_flight) < max_workers and time.monotonic() < deadline:
                try:
                    exp = next(candidate_iter)
                except StopIteration:
                    return
                in_flight[executor.submit(process_single_exp, exp)] = exp

        try:
            fill_worker_slots()
            while in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                done, _ = wait(
                    tuple(in_flight),
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    timed_out = True
                    break
                for future in done:
                    exp = in_flight.pop(future)
                    try:
                        exp_id, status, result = future.result()
                    except Exception as exc:
                        exp_id = exp.exp_id
                        status, result = "failed", type(exc).__name__
                    if status == "validated":
                        summaries[exp_id] = result
                    elif status == "fatal":
                        fatal_reason = str(result)
                        stop_event.set()
                        break
                    elif status == "rejected" and result == "bad_request":
                        request_rejected = True
                    elif status in {"retryable", "failed"}:
                        transient_failure = True
                if fatal_reason:
                    break
                fill_worker_slots()
        finally:
            if timed_out or fatal_reason:
                stop_event.set()
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        if timed_out:
            logger.warning("GPT summarization reached the %ss deadline", total_timeout)
        if request_rejected:
            logger.warning(
                "GPT rejected at least one summary request; affected skills were discarded"
            )
        if transient_failure and not timed_out:
            logger.warning(
                "At least one GPT summary request failed; affected skills were discarded"
            )

        if fatal_reason:
            raise RuntimeError(
                f"GPT summarization cannot continue: {fatal_reason}"
            )

        by_id = {exp.exp_id: exp for exp in candidates}
        attached: Dict[str, Dict[str, Any]] = {}
        for exp_id, record in summaries.items():
            exp = by_id[exp_id]
            checked, _ = validate_gpt_summary_payload(
                exp, record.get("summary_plan", record)
            )
            if checked is None:
                continue
            exp.gpt_summary = str(checked["summary"])
            exp.gpt_evidence_ids = list(checked["evidence_ids"])
            exp.gpt_evidence_hash = str(checked["evidence_hash"])
            exp.gpt_summary_plan = dict(checked["summary_plan"])
            attached[exp_id] = checked

        return attached

    def __len__(self) -> int:
        """Return the total number of stored skills."""
        return sum(len(exps) for exps in self.experiences.values())

    @staticmethod
    def _storage_paths(path: str) -> Tuple[str, str]:
        """Return distinct human-readable JSON and full PKL paths."""
        root, extension = os.path.splitext(path)
        extension = extension.lower()
        if extension == '.json':
            return path, root + '.pkl'
        if extension == '.pkl':
            return root + '.json', path
        return path + '.json', path + '.pkl'

    def save(self, path: Optional[str] = None):
        """Save one serialized memory snapshot (thread-safe)."""
        if self.read_only:
            return

        save_path = path or self.save_path
        if not save_path:
            return

        with self._save_lock:
            self._save_once(save_path)

    def _save_once(self, save_path: str) -> None:
        """Build and persist a snapshot while ``_save_lock`` is held."""

        json_path, pkl_path = self._storage_paths(save_path)

        save_dir = os.path.dirname(json_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with self._lock:
            data_full = {
                'max_size': self.max_size,
                'min_score_delta': self.min_score_delta,
                'experiences': {
                    task: [exp.to_dict() for exp in exps]
                    for task, exps in self.experiences.items()
                },
            }
            data_json = {
                'max_size': self.max_size,
                'min_score_delta': self.min_score_delta,
                'experiences': {
                    task: [exp.to_dict_for_json() for exp in exps]
                    for task, exps in self.experiences.items()
                },
            }

        tmp_json_path = None
        tmp_pkl_path = None
        try:
            with tempfile.NamedTemporaryFile('w', delete=False,
                                             dir=save_dir or '.', suffix='.json') as tmp_f:
                json.dump(data_json, tmp_f, indent=2)
                tmp_json_path = tmp_f.name
            # Serialize both snapshots before atomically replacing either file.
            with tempfile.NamedTemporaryFile('wb', delete=False,
                                             dir=save_dir or '.', suffix='.pkl') as tmp_f:
                pickle.dump(data_full, tmp_f)
                tmp_pkl_path = tmp_f.name

            # Commit authoritative PKL last.
            shutil.move(tmp_json_path, json_path)
            tmp_json_path = None
            shutil.move(tmp_pkl_path, pkl_path)
            tmp_pkl_path = None

            logger.info("Evolving Skill Memory saved to %s and %s", json_path, pkl_path)
        except Exception as e:
            logger.error(f"Failed to save Evolving Skill Memory: {e}")
            for tmp_path in [tmp_json_path, tmp_pkl_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            raise RuntimeError("Failed to persist Evolving Skill Memory") from e

    def load(self, path: Optional[str] = None):
        """Load memory from disk (PKL only)."""
        load_path = path or self.save_path
        if not load_path:
            return

        # Only load from PKL (JSON is for human readability only)
        _, pkl_path = self._storage_paths(load_path)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Evolving Skill Memory PKL file not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid Evolving Skill Memory PKL payload")

        stored_max_size = int(data.get('max_size', self.max_size))
        stored_min_score_delta = float(
            data.get('min_score_delta', self.min_score_delta)
        )
        if stored_max_size <= 0 or stored_min_score_delta <= 0:
            raise ValueError("Invalid Evolving Skill Memory limits in PKL snapshot")
        if stored_max_size != self.max_size:
            logger.info(
                "Loading memory saved with max_size=%d under runtime max_size=%d; "
                "the runtime limit remains authoritative",
                stored_max_size,
                self.max_size,
            )
        if stored_min_score_delta != self.min_score_delta:
            logger.info(
                "Loading memory saved with min_score_delta=%g under runtime "
                "min_score_delta=%g; the runtime threshold remains authoritative",
                stored_min_score_delta,
                self.min_score_delta,
            )

        def is_accepted_skill(exp: SkillEntry) -> bool:
            """Return whether a serialized entry is a successful skill."""
            objective_deltas = exp.objective_deltas or {}
            if objective_deltas:
                task_properties = exp.task.split('+')
                if set(objective_deltas) != set(task_properties):
                    return False
                return all(
                    float(objective_deltas[prop]) >= self.min_score_delta
                    for prop in task_properties
                )
            task = exp.task
            if '+' in task:
                return False
            score_delta = float(exp.score_delta)
            if task == 'sa':
                score_delta = -score_delta
            return score_delta >= self.min_score_delta

        stored_experiences = data.get('experiences', {})
        if not isinstance(stored_experiences, dict):
            raise ValueError("Invalid experiences mapping in PKL snapshot")

        parsed_experiences: Dict[str, List[SkillEntry]] = defaultdict(list)
        stored_count = 0
        retained_count = 0
        for task, serialized_entries in stored_experiences.items():
            if not isinstance(task, str) or not isinstance(serialized_entries, list):
                raise ValueError("Invalid task bucket in PKL snapshot")
            stored_count += len(serialized_entries)
            for serialized_entry in serialized_entries:
                entry = SkillEntry.from_dict(serialized_entry)
                if entry.task != task:
                    raise ValueError(
                        f"SkillEntry task {entry.task!r} does not match bucket {task!r}"
                    )
                structurally_valid = skill_transition_rejection_reason(
                    entry.before_smiles,
                    entry.after_smiles,
                    entry.score_before,
                    entry.score_after,
                    entry.score_delta,
                    entry.objective_deltas,
                ) is None
                if is_accepted_skill(entry) and structurally_valid:
                    parsed_experiences[task].append(entry)
                    retained_count += 1

        self.experiences = parsed_experiences

        all_ids = [
            exp.exp_id
            for experiences in self.experiences.values()
            for exp in experiences
        ]
        seen_ids: Set[str] = set()
        duplicate_ids: Set[str] = set()
        for exp_id in all_ids:
            if exp_id in seen_ids:
                duplicate_ids.add(exp_id)
            seen_ids.add(exp_id)
        if duplicate_ids:
            sample = ", ".join(sorted(duplicate_ids)[:3])
            raise ValueError(
                "Duplicate experience IDs in PKL snapshot; refusing ambiguous "
                f"summary relinking (examples: {sample})"
            )

        invalidated = 0
        validated_experiences: Dict[str, List[SkillEntry]] = defaultdict(list)
        for task, experiences in self.experiences.items():
            for exp in experiences:
                record, _ = validate_gpt_summary_payload(
                    exp, exp.gpt_summary_plan,
                )
                if record is None or record['summary'] != exp.gpt_summary:
                    invalidated += 1
                    continue
                exp.gpt_summary = str(record['summary'])
                exp.gpt_evidence_ids = list(record['evidence_ids'])
                exp.gpt_evidence_hash = str(record['evidence_hash'])
                exp.gpt_summary_plan = dict(record['summary_plan'])
                exp.compute_indices()
                if exp.before_fingerprint is None:
                    invalidated += 1
                    continue
                validated_experiences[task].append(exp)
        self.experiences = validated_experiences

        deduplicated: Dict[str, List[SkillEntry]] = defaultdict(list)
        pair_index = {}
        for task, experiences in self.experiences.items():
            for exp in experiences:
                key = self._get_smiles_key(exp)
                existing = pair_index.get(key) if key is not None else None
                if existing is None:
                    deduplicated[task].append(exp)
                    if key is not None:
                        pair_index[key] = exp
                else:
                    self._merge_experience(existing, exp)
        self.experiences = deduplicated

        for task in list(self.experiences):
            if len(self.experiences[task]) > self.max_size:
                self._trim_task_locked(task)
        self._rebuild_canonical_pair_index_locked()

        discarded_count = stored_count - retained_count
        if discarded_count:
            logger.info(
                "Discarded %d entries that did not satisfy the skill-memory contract",
                discarded_count,
            )
        if invalidated:
            logger.warning(
                "Discarded %d stored entries without a valid evidence-bound summary",
                invalidated,
            )
        logger.info("Evolving Skill Memory loaded from %s", pkl_path)


# ============================================================
# GPT Summary Generation
# ============================================================

def build_summary_evidence(exp: SkillEntry) -> Dict[str, Dict[str, Any]]:
    """Recompute the facts GPT may cite from the two source molecules.

    Stored MCS labels are intentionally excluded: a maximum common
    substructure can be ambiguous, whereas every fact below is recomputed from
    RDKit descriptors or an explicit RDKit/SMARTS match.
    """
    before = Chem.MolFromSmiles(exp.before_smiles)
    after = Chem.MolFromSmiles(exp.after_smiles)
    if before is None or after is None:
        return {}

    catalog: Dict[str, Dict[str, Any]] = {}

    objective_deltas = dict(exp.objective_deltas)
    if not objective_deltas:
        delta = -float(exp.score_delta) if exp.task == 'sa' else float(exp.score_delta)
        objective_deltas = {exp.task: delta}
    for objective, delta in sorted(objective_deltas.items()):
        delta = float(delta)
        if delta <= 0:
            continue
        evidence_id = f"objective.{_evidence_slug(objective)}"
        catalog[evidence_id] = {
            "kind": "objective",
            "source": "recorded_direction_normalized_delta",
            "text": (
                f"the observed direction-normalized {objective.upper()} "
                f"improvement of {delta:+.3f}"
            ),
        }

    fg_diff = compute_fg_diff(exp.before_smiles, exp.after_smiles)
    removed_names = FunctionalGroupPruner.prune(
        [name for name, delta in fg_diff.items() if delta < 0]
    )
    added_names = FunctionalGroupPruner.prune(
        [name for name, delta in fg_diff.items() if delta > 0]
    )
    for direction, names in (("removed", removed_names), ("added", added_names)):
        for name in sorted(names):
            count = abs(int(fg_diff[name]))
            action = "removing" if direction == "removed" else "adding"
            occurrence = "occurrence" if count == 1 else "occurrences"
            evidence_id = f"fg.{direction}.{_evidence_slug(name)}"
            catalog[evidence_id] = {
                "kind": "functional_group",
                "source": "rdkit_fragments_or_smarts",
                "text": f"{action} {count} {occurrence} of {map_fg_name(name)}",
            }

    ring_delta = after.GetRingInfo().NumRings() - before.GetRingInfo().NumRings()
    if ring_delta:
        action = "increasing" if ring_delta > 0 else "decreasing"
        catalog["rings.delta"] = {
            "kind": "ring_count",
            "source": "rdkit_ring_info",
            "text": f"{action} the RDKit ring count by {abs(ring_delta)}",
        }

    def add_float_delta(
        evidence_id: str,
        label: str,
        value: float,
        unit: str,
        tolerance: float = 0.05,
    ) -> None:
        if abs(value) < tolerance:
            return
        action = "increasing" if value > 0 else "decreasing"
        suffix = f" {unit}" if unit else ""
        catalog[evidence_id] = {
            "kind": "property",
            "source": "rdkit_descriptor",
            "text": f"{action} {label} by {abs(value):.1f}{suffix}",
        }

    add_float_delta(
        "property.mw",
        "molecular weight",
        Descriptors.MolWt(after) - Descriptors.MolWt(before),
        "Da",
    )
    add_float_delta(
        "property.tpsa",
        "topological polar surface area",
        Descriptors.TPSA(after) - Descriptors.TPSA(before),
        "Å²",
    )

    integer_properties = (
        ("property.hbd", "hydrogen-bond donor count", rdMolDescriptors.CalcNumHBD),
        ("property.hba", "hydrogen-bond acceptor count", rdMolDescriptors.CalcNumHBA),
        (
            "property.rotatable_bonds",
            "rotatable-bond count",
            rdMolDescriptors.CalcNumRotatableBonds,
        ),
    )
    for evidence_id, label, calculator in integer_properties:
        delta = int(calculator(after) - calculator(before))
        if not delta:
            continue
        action = "increasing" if delta > 0 else "decreasing"
        catalog[evidence_id] = {
            "kind": "property",
            "source": "rdkit_descriptor",
            "text": f"{action} {label} by {abs(delta)}",
        }

    atom_delta = defaultdict(int)
    for atom in after.GetAtoms():
        atom_delta[atom.GetSymbol()] += 1
    for atom in before.GetAtoms():
        atom_delta[atom.GetSymbol()] -= 1
    for symbol, delta in sorted(atom_delta.items()):
        if not delta:
            continue
        action = "adding" if delta > 0 else "removing"
        occurrence = "occurrence" if abs(delta) == 1 else "occurrences"
        catalog[f"atoms.{_evidence_slug(symbol)}"] = {
            "kind": "atom_count",
            "source": "rdkit_atom_count",
            "text": f"{action} {abs(delta)} {symbol} atom {occurrence}",
        }

    return catalog


def _summary_evidence_hash(
    exp: SkillEntry,
    catalog: Dict[str, Dict[str, Any]],
    plan: Dict[str, Any],
) -> str:
    before = Chem.MolToSmiles(
        Chem.MolFromSmiles(exp.before_smiles), canonical=True, isomericSmiles=True
    )
    after = Chem.MolToSmiles(
        Chem.MolFromSmiles(exp.after_smiles), canonical=True, isomericSmiles=True
    )
    source = {
        "task": sorted(part.strip() for part in exp.task.split('+') if part.strip()),
        "before": before,
        "after": after,
        "plan": plan,
        "selected_evidence": {
            evidence_id: catalog[evidence_id]
            for evidence_id in (
                plan["change_evidence_ids"] + plan["objective_evidence_ids"]
            )
        },
    }
    return hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_gpt_summary_payload(
    exp: SkillEntry, payload: Any
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Validate a GPT-authored action sentence and its evidence bindings.

    The sentence remains free natural language. The structured evidence IDs
    are checked against facts recomputed from the molecule pair so summaries
    with missing or invalid evidence never become retrievable.
    """
    if not isinstance(payload, dict):
        return None, "response_not_object"
    required_fields = {
        "usable", "summary", "change_evidence_ids", "objective_evidence_ids",
    }
    if set(payload) != required_fields:
        return None, "unexpected_response_fields"

    usable = payload.get("usable")
    if not isinstance(usable, bool):
        return None, "invalid_usable"
    if not usable:
        return None, "model_marked_not_usable"

    summary = payload.get("summary")
    change_ids = payload.get("change_evidence_ids")
    objective_ids = payload.get("objective_evidence_ids")
    if not isinstance(summary, str):
        return None, "invalid_summary"
    summary = summary.strip()
    if not 12 <= len(summary) <= 500 or "\n" in summary or "\r" in summary:
        return None, "invalid_summary"
    if summary[-1] not in ".!?":
        return None, "summary_not_one_sentence"
    if re.search(r"[.!?]\s+\S", summary[:-1]):
        return None, "summary_not_one_sentence"
    if "{{" in summary or "}}" in summary:
        return None, "invalid_summary"
    for evidence_ids in (change_ids, objective_ids):
        if (
            not isinstance(evidence_ids, list)
            or any(not isinstance(item, str) for item in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            return None, "invalid_evidence_ids"
    if (
        not 1 <= len(change_ids) <= 3
        or not 1 <= len(objective_ids) <= 3
        or len(change_ids) + len(objective_ids) > 4
    ):
        return None, "invalid_evidence_count"

    catalog = build_summary_evidence(exp)
    evidence_ids = change_ids + objective_ids
    if len(set(evidence_ids)) != len(evidence_ids):
        return None, "duplicate_evidence_id"
    if any(evidence_id not in catalog for evidence_id in evidence_ids):
        return None, "unknown_evidence_id"
    if any(catalog[item]["kind"] == "objective" for item in change_ids):
        return None, "objective_in_change_evidence"
    if any(catalog[item]["kind"] != "objective" for item in objective_ids):
        return None, "nonobjective_in_objective_evidence"

    expected_objective_ids = {
        f"objective.{_evidence_slug(objective)}"
        for objective in EvolvingSkillMemory._task_signature(exp.task)
    }
    if len(expected_objective_ids) > 3:
        return None, "unsupported_objective_count"
    if set(objective_ids) != expected_objective_ids:
        return None, "incomplete_objective_evidence"

    plan = {
        "usable": True,
        "summary": summary,
        "change_evidence_ids": list(change_ids),
        "objective_evidence_ids": list(objective_ids),
    }

    return {
        "summary": summary,
        "evidence_ids": list(evidence_ids),
        "evidence_hash": _summary_evidence_hash(exp, catalog, plan),
        "summary_plan": plan,
    }, ""


def _structured_summary_response_format(
    catalog: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    change_ids = sorted(
        evidence_id for evidence_id, item in catalog.items()
        if item["kind"] != "objective"
    )
    objective_ids = sorted(
        evidence_id for evidence_id, item in catalog.items()
        if item["kind"] == "objective"
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "molmem_skill_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "usable": {
                        "type": "boolean",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "One actionable sentence whose structural claims are "
                            "fully supported by the selected change evidence."
                        ),
                    },
                    "change_evidence_ids": {
                        "type": "array",
                        "description": "The smallest evidence set that entails the action.",
                        "items": {"type": "string", "enum": change_ids},
                    },
                    "objective_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": objective_ids},
                    },
                },
                "required": [
                    "usable", "summary", "change_evidence_ids",
                    "objective_evidence_ids",
                ],
                "additionalProperties": False,
            },
        },
    }


def format_gpt_prompt(exp: SkillEntry) -> str:
    """Ask GPT for one free action sentence plus verifiable evidence IDs."""
    catalog = build_summary_evidence(exp)
    change_evidence = {
        evidence_id: {"kind": item["kind"], "fact": item["text"]}
        for evidence_id, item in catalog.items()
        if item["kind"] != "objective"
    }
    objective_evidence = {
        evidence_id: {"kind": item["kind"], "fact": item["text"]}
        for evidence_id, item in catalog.items()
        if item["kind"] == "objective"
    }
    fg_removed = [
        map_fg_name(name) for name in (exp.functional_groups_removed or [])
    ]
    fg_added = [
        map_fg_name(name) for name in (exp.functional_groups_added or [])
    ]
    objective_deltas = {
        name: float(value)
        for name, value in sorted((exp.objective_deltas or {}).items())
    }
    return f"""Identify the ONE most important coherent source-to-target structural
action in this successful molecular transformation. Prefer a concrete, local,
transferable edit that could plausibly matter for the task over a list of every
difference. If the pair is too ambiguous, too global, or has no reliable
reusable structural action, set usable to false.

When usable is true, write summary yourself as ONE concise, self-contained,
actionable sentence. State only the structural edit; a qualitative effect is
optional and should be included only when directly supported. Do not include
scores, numerical changes, or a list of descriptors. Do not copy internal
detector labels into the sentence; translate only confidently recognized
chemistry into standard language. Do not claim a mechanism or causality, and do
not use framing such as "on a similar scaffold" or "in the source example".
Describe exactly one edit, normally as one add/remove/replace/convert operation;
never combine separate edits with "and", "while", or a second action clause. If
multiple coupled edits cannot be reduced to one faithful transferable action,
set usable to false. Do not guess an approximate chemical name when the local
structure is unclear.

Return only a JSON object with usable, summary, ordered change_evidence_ids, and
ordered objective_evidence_ids. When usable is true, select one to three change
evidence IDs that directly support the chosen action, using the smallest useful
set (normally one or two). Every structural noun and edit direction in the
summary must be entailed by those selected facts; never attach valid evidence
IDs to an unrelated claim. Do not select a descriptor merely because it changed.
Always include every objective
evidence ID for auditing, but the summary does not need to repeat the objective
or its value. When usable is false, return an empty summary and empty
change_evidence_ids. The numeric and descriptor information below is context for
choosing the action, never the action's identity.

=== Task and observed outcome (context only) ===
Task: {exp.task}
Score: {float(exp.score_before):.3f} -> {float(exp.score_after):.3f} ({float(exp.score_delta):+.3f})
Direction-normalized objective deltas: {json.dumps(objective_deltas, sort_keys=True)}

=== Molecules ===
Before SMILES: {exp.before_smiles}
After SMILES:  {exp.after_smiles}

=== MCS-derived context (may be ambiguous; do not force it) ===
Modification: {exp.modification_type or 'unknown'}
Removed fragment: {exp.removed_fragment or 'none'}
Added fragment: {exp.added_fragment or 'none'}

=== Scaffold context ===
Before scaffold: {exp.before_scaffold or 'none'}
After scaffold:  {exp.after_scaffold or 'none'}
Scaffold type: {exp.scaffold_type or 'unknown'}

=== Functional-group context ===
Removed: {json.dumps(fg_removed)}
Added: {json.dumps(fg_added)}

=== Descriptor changes (context only; do not list in summary) ===
MW: {float(exp.mw_change):+.1f} Da | Rings: {int(exp.ring_changes):+d}
TPSA: {float(exp.psa_change):+.1f} A2 | HBD: {int(exp.hbd_change):+d} | HBA: {int(exp.hba_change):+d}

=== Verifiable change evidence ===
{json.dumps(change_evidence, indent=2, sort_keys=True)}
=== Objective evidence (include all IDs for audit) ===
{json.dumps(objective_evidence, indent=2, sort_keys=True)}"""


def format_hints_for_prompt(experiences: List[SkillEntry], task: str) -> str:
    """
    Format experiences into prompt hints.

    Only locally evidence-validated GPT summaries may reach the policy prompt.

    Args:
        experiences: List of relevant experiences
        task: Current task type

    Returns:
        Formatted string to inject into prompt, or empty string if no valid experiences
    """
    validated = [
        exp for exp in experiences
        if EvolvingSkillMemory._is_retrievable(exp)
    ]
    if not validated:
        return ""

    lines = [f"\n=== Potential Useful Strategies for {task} ==="]

    for index, exp in enumerate(validated, 1):
        lines.append(f"{index}. {exp.gpt_summary}")

    lines.append("\nThese strategies may not always apply. Use them as reference when appropriate.\n")

    return "\n".join(lines)


# ============================================================
# Experience Extraction Helper
# ============================================================

def extract_experience_from_trajectory(
    before_smiles: str,
    after_smiles: str,
    trajectory: List[str],
    score_before: float,
    score_after: float,
    task: str,
    epoch: int = 0,
    objective_deltas: Optional[Dict[str, float]] = None,
) -> SkillEntry:
    """
    Create a SkillEntry from trajectory data.

    Args:
        before_smiles: Initial molecule SMILES
        after_smiles: Final molecule SMILES
        trajectory: List of intermediate SMILES
        score_before: Initial score
        score_after: Final score
        task: Task type
        epoch: Current training epoch
        objective_deltas: Direction-normalized per-objective changes

    Returns:
        SkillEntry object
    """
    score_delta = score_after - score_before
    rejection_reason = skill_transition_rejection_reason(
        before_smiles,
        after_smiles,
        score_before,
        score_after,
        score_delta,
        objective_deltas,
    )
    if rejection_reason is not None:
        raise ValueError(f"Invalid skill transition: {rejection_reason}")

    analysis = analyze_transformation(before_smiles, after_smiles)

    exp = SkillEntry(
        task=task,
        created_epoch=epoch,
        before_smiles=before_smiles,
        after_smiles=after_smiles,
        trajectory=trajectory,
        score_before=score_before,
        score_after=score_after,
        score_delta=score_delta,
        objective_deltas=objective_deltas or {},
        atom_changes=analysis.get('atom_changes', {}),
        ring_changes=analysis.get('ring_changes', 0),
        mw_change=analysis.get('mw_change', 0.0),
        psa_change=analysis.get('psa_change', 0.0),
        hbd_change=analysis.get('hbd_change', 0),
        hba_change=analysis.get('hba_change', 0),
        rotatable_bonds_change=analysis.get('rotatable_bonds_change', 0),
        stereo_centers_change=analysis.get('stereo_centers_change', 0),
        functional_groups_removed=analysis.get('fg_removed', []),
        functional_groups_added=analysis.get('fg_added', []),
        mcs_smarts=analysis.get('mcs_smarts', ''),
        removed_fragment=analysis.get('removed_fragment', ''),
        added_fragment=analysis.get('added_fragment', ''),
        modification_type=analysis.get('modification_type', ''),
        before_scaffold=analysis.get('before_scaffold', ''),
        after_scaffold=analysis.get('after_scaffold', ''),
        scaffold_changed=analysis.get('scaffold_changed', False),
        is_scaffold_hop=analysis.get('is_scaffold_hop', False),
        scaffold_type=analysis.get('scaffold_type', ''),
    )

    exp.compute_indices()

    return exp
