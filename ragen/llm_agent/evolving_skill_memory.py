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
import logging
import os
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import Fragments, DataStructs, Descriptors, rdMolDescriptors, rdFMCS
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

logger = logging.getLogger(__name__)


# ============================================================
# RDKit Name Mapping (for GPT prompt clarity)
# ============================================================

# Map RDKit fragment names to human-readable chemical descriptions
# This prevents GPT from misinterpreting technical names like "para_hydroxylation"
RDKIT_NAME_MAP = {
    "para_hydroxylation": "Para-substituted Aniline Pattern",
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


def clean_fragment_string(frag_str: str) -> str:
    """Clean up fragmented atom soup from MCS extraction failures.

    If the fragment string is just a comma-separated list of atoms without
    proper structural information, replace it with a clearer message.
    """
    if not frag_str:
        return ""
    # If comma count > 3 and no ring/bond markers, it's atom soup
    # Use > 3 instead of > 2 to avoid false positives on small valid fragments
    has_structure = any(m in frag_str for m in ['(', ')', '=', '#', '-', '[', ']'])
    if frag_str.count(',') > 3 and not has_structure:
        return "Complex Substructure (Refer to SMILES)"
    return frag_str


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
            # Check hierarchy map
            if tag in cls.HIERARCHY_MAP:
                children = cls.HIERARCHY_MAP[tag]
                to_remove.update(child for child in children if child in tag_set)

            # Check generic redundancy
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

    # 1. RDKit built-in fragment detection
    for name, func in RDKIT_FRAGMENT_FUNCS:
        try:
            count = func(mol)
            if count > 0:
                detected[name] = count
        except Exception:
            continue

    # 2. Custom SMARTS patterns
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

    # Try to extract fragment using RWMol
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

    # Fallback: return atom symbols
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

    # Get matching atom indices
    match_before = mol_before.GetSubstructMatch(mcs_mol) if mcs_mol else ()
    match_after = mol_after.GetSubstructMatch(mcs_mol) if mcs_mol else ()

    # Count non-MCS atoms
    non_mcs_before = mol_before.GetNumAtoms() - len(match_before)
    non_mcs_after = mol_after.GetNumAtoms() - len(match_after)

    # Extract fragments
    removed_fragment = extract_non_mcs_fragment(mol_before, match_before)
    added_fragment = extract_non_mcs_fragment(mol_after, match_after)

    # Determine modification type
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

    # Extract Murcko scaffolds
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

    # Check generic scaffold (ignore heteroatom types)
    generic_same = False
    if scaffold_before and scaffold_after and scaffold_before.GetNumAtoms() > 0 and scaffold_after.GetNumAtoms() > 0:
        try:
            generic_before = MakeScaffoldGeneric(scaffold_before)
            generic_after = MakeScaffoldGeneric(scaffold_after)
            generic_same = Chem.MolToSmiles(generic_before) == Chem.MolToSmiles(generic_after)
        except Exception:
            generic_same = False

    # Determine scaffold type
    if not scaffold_changed:
        scaffold_type = 'side_chain_mod'
    elif generic_same:
        scaffold_type = 'scaffold_hop'  # e.g., Benzene → Pyridine
    else:
        # Compare ring counts
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

    # Atom count changes
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

    # Ring count changes
    ring_before = mol_before.GetRingInfo().NumRings()
    ring_after = mol_after.GetRingInfo().NumRings()
    ring_changes = ring_after - ring_before

    # Molecular weight change
    mw_before = Descriptors.MolWt(mol_before)
    mw_after = Descriptors.MolWt(mol_after)
    mw_change = mw_after - mw_before

    # PSA (Polar Surface Area) change
    try:
        psa_change = Descriptors.TPSA(mol_after) - Descriptors.TPSA(mol_before)
    except Exception:
        psa_change = 0.0

    # H-bond donor/acceptor changes
    try:
        hbd_change = rdMolDescriptors.CalcNumHBD(mol_after) - rdMolDescriptors.CalcNumHBD(mol_before)
    except Exception:
        hbd_change = 0

    try:
        hba_change = rdMolDescriptors.CalcNumHBA(mol_after) - rdMolDescriptors.CalcNumHBA(mol_before)
    except Exception:
        hba_change = 0

    # Rotatable bonds change
    try:
        rotatable_bonds_change = rdMolDescriptors.CalcNumRotatableBonds(mol_after) - rdMolDescriptors.CalcNumRotatableBonds(mol_before)
    except Exception:
        rotatable_bonds_change = 0

    # Stereo centers change
    try:
        stereo_before = len(Chem.FindMolChiralCenters(mol_before, includeUnassigned=True))
        stereo_after = len(Chem.FindMolChiralCenters(mol_after, includeUnassigned=True))
        stereo_centers_change = stereo_after - stereo_before
    except Exception:
        stereo_centers_change = 0

    # Functional group changes
    fg_diff = compute_fg_diff(before_smiles, after_smiles)
    fg_removed = [fg for fg, change in fg_diff.items() if change < 0]
    fg_added = [fg for fg, change in fg_diff.items() if change > 0]

    # MCS analysis
    mcs_analysis = analyze_mcs_changes(before_smiles, after_smiles)

    # Scaffold analysis
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
        # MCS analysis fields
        'mcs_smarts': mcs_analysis.get('mcs_smarts', ''),
        'removed_fragment': mcs_analysis.get('removed_fragment', ''),
        'added_fragment': mcs_analysis.get('added_fragment', ''),
        'modification_type': mcs_analysis.get('modification_type', ''),
        # Scaffold analysis fields
        'before_scaffold': scaffold_analysis.get('before_scaffold', ''),
        'after_scaffold': scaffold_analysis.get('after_scaffold', ''),
        'scaffold_changed': scaffold_analysis.get('scaffold_changed', False),
        'is_scaffold_hop': scaffold_analysis.get('is_scaffold_hop', False),
        'scaffold_type': scaffold_analysis.get('scaffold_type', ''),
    }


def aggregate_fg_statistics(experiences: List['SkillEntry']) -> Dict:
    """
    Aggregate functional group change statistics from experiences.

    Args:
        experiences: List of SkillEntry objects

    Returns:
        Dict mapping change patterns (e.g., "-amine_secondary", "+benzene") to stats
    """
    fg_changes = defaultdict(list)  # {pattern: [score_delta_list]}

    for exp in experiences:
        diff = compute_fg_diff(exp.before_smiles, exp.after_smiles)
        for fg, change in diff.items():
            pattern = f"-{fg}" if change < 0 else f"+{fg}"
            fg_changes[pattern].append(exp.score_delta)

    # Compute statistics
    stats = {}
    for pattern, deltas in fg_changes.items():
        stats[pattern] = {
            'count': len(deltas),
            'avg_improvement': sum(deltas) / len(deltas) if deltas else 0,
            'max_improvement': max(deltas) if deltas else 0,
            'min_improvement': min(deltas) if deltas else 0,
        }

    return stats


# ============================================================
# Data Structures
# ============================================================

@dataclass
class SkillEntry:
    """Single skill entry for molecular optimization (Evolving Skill Memory)"""
    # Basic info
    exp_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
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
    success: bool = False  # Whether optimization succeeded

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

    # GPT-generated summary (filled later)
    gpt_summary: Optional[str] = None

    # Merge tracking fields
    occurrence_count: int = 1  # Number of times this experience pattern was observed
    merged_from: List[str] = field(default_factory=list)  # List of exp_ids merged into this

    # Retrieval indices (computed on demand)
    before_fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    functional_group_tags: Set[str] = field(default_factory=set)

    # Usage tracking fields (SVS system)
    retrieval_count: int = 0              # N_e: number of times retrieved
    total_usage_delta: float = 0.0        # cumulative score change after usage
    positive_usage_count: int = 0         # times with positive outcome
    negative_usage_count: int = 0         # times with negative outcome
    is_deprecated: bool = False           # soft delete flag
    deprecated_reason: str = ""           # reason for deprecation

    # Consolidation tracking fields
    is_consolidated: bool = False         # has been merged into another experience
    consolidated_into: str = ""           # exp_id of the consolidated experience
    is_from_consolidation: bool = False   # whether created from consolidation
    consolidated_from: List[str] = field(default_factory=list)  # source exp_ids

    @property
    def usage_count(self) -> int:
        """Total usage count (with recorded outcomes)"""
        return self.positive_usage_count + self.negative_usage_count

    @property
    def avg_usage_delta(self) -> float:
        """Average outcome delta per usage"""
        return self.total_usage_delta / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def usage_success_rate(self) -> float:
        """Success rate when used"""
        return self.positive_usage_count / self.usage_count if self.usage_count > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        d = asdict(self)
        # Convert numpy array to list
        if d['before_fingerprint'] is not None:
            d['before_fingerprint'] = d['before_fingerprint'].tolist()
        # Convert set to list
        d['functional_group_tags'] = list(d['functional_group_tags'])
        return d

    def to_dict_for_json(self) -> Dict:
        """Convert to JSON-serializable dict without fingerprint (for human readability)"""
        d = self.to_dict()
        # Remove fingerprint field to keep JSON readable
        d.pop('before_fingerprint', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillEntry':
        """Create from dict with validation and backward compatibility"""
        # Validate required fields
        required_fields = ['before_smiles', 'after_smiles', 'task']
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Convert list back to numpy array
        if data.get('before_fingerprint'):
            data['before_fingerprint'] = np.array(data['before_fingerprint'])
        # Convert list back to set
        if data.get('functional_group_tags'):
            data['functional_group_tags'] = set(data['functional_group_tags'])

        # Backward compatibility: set defaults for new/optional fields
        data.setdefault('occurrence_count', 1)
        data.setdefault('merged_from', [])
        data.setdefault('exp_id', str(uuid.uuid4())[:8])
        data.setdefault('created_epoch', 0)
        data.setdefault('trajectory', [])
        data.setdefault('score_before', 0.0)
        data.setdefault('score_after', 0.0)
        data.setdefault('score_delta', 0.0)
        data.setdefault('success', False)
        data.setdefault('atom_changes', {})
        data.setdefault('ring_changes', 0)
        data.setdefault('mw_change', 0.0)
        data.setdefault('psa_change', 0.0)
        data.setdefault('hbd_change', 0)
        data.setdefault('hba_change', 0)
        data.setdefault('rotatable_bonds_change', 0)
        data.setdefault('stereo_centers_change', 0)

        # MCS analysis defaults
        data.setdefault('mcs_smarts', '')
        data.setdefault('removed_fragment', '')
        data.setdefault('added_fragment', '')
        data.setdefault('modification_type', '')

        # Scaffold analysis defaults
        data.setdefault('before_scaffold', '')
        data.setdefault('after_scaffold', '')
        data.setdefault('scaffold_changed', False)
        data.setdefault('is_scaffold_hop', False)
        data.setdefault('scaffold_type', '')

        # Type validation for new fields
        try:
            data['psa_change'] = float(data['psa_change'])
        except (ValueError, TypeError):
            data['psa_change'] = 0.0
        for field in ['hbd_change', 'hba_change', 'rotatable_bonds_change', 'stereo_centers_change']:
            try:
                data[field] = int(data[field])
            except (ValueError, TypeError):
                data[field] = 0
        data.setdefault('functional_groups_removed', [])
        data.setdefault('functional_groups_added', [])
        data.setdefault('gpt_summary', None)
        data.setdefault('before_fingerprint', None)
        data.setdefault('functional_group_tags', set())

        # SVS tracking fields defaults (for backward compatibility)
        data.setdefault('retrieval_count', 0)
        data.setdefault('total_usage_delta', 0.0)
        data.setdefault('positive_usage_count', 0)
        data.setdefault('negative_usage_count', 0)
        data.setdefault('is_deprecated', False)
        data.setdefault('deprecated_reason', '')

        # Consolidation tracking fields defaults
        data.setdefault('is_consolidated', False)
        data.setdefault('consolidated_into', '')
        data.setdefault('is_from_consolidation', False)
        data.setdefault('consolidated_from', [])

        try:
            return cls(**data)
        except TypeError as e:
            logger.error(f"Invalid data format for SkillEntry: {e}")
            raise ValueError(f"Invalid data format: {e}")

    def compute_indices(self):
        """Compute fingerprint and FG tags for retrieval"""
        # Compute Morgan fingerprint
        mol = Chem.MolFromSmiles(self.before_smiles)
        if mol:
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            self.before_fingerprint = np.zeros(2048, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, self.before_fingerprint)

        # Compute FG tags
        fg_before = detect_functional_groups(self.before_smiles)
        fg_after = detect_functional_groups(self.after_smiles)

        # Tags are the FGs that were present before or added/removed
        # Apply pruning to remove redundant low-level tags
        raw_tags = set(fg_before.keys()) | set(fg_after.keys())
        self.functional_group_tags = set(FunctionalGroupPruner.prune(list(raw_tags)))


# ============================================================
# Evolving Skill Memory
# ============================================================

class EvolvingSkillMemory:
    """
    Evolving Skill Memory for storing and retrieving molecular optimization skills.

    This component distills successful optimization trajectories into reusable
    strategies that can be applied to future optimization tasks.

    Supports three retrieval methods: by functional group, by similarity, by task.
    Thread-safe for concurrent access.
    """

    def __init__(self,
                 max_size: int = 10000,
                 save_path: Optional[str] = None,
                 include_failures: bool = True,
                 min_score_delta: float = 0.01,
                 max_pending_size: int = 1000,
                 merge_enabled: bool = True,
                 merge_by_smiles: bool = True,
                 merge_by_fg_pattern: bool = True,
                 read_only: bool = False,
                 svs_config: Optional[Dict] = None):
        """
        Args:
            max_size: Maximum number of experiences to store
            save_path: Path to save/load the pool
            include_failures: Whether to store failure experiences
            min_score_delta: Minimum score change to record
            max_pending_size: Maximum pending experiences for GPT (prevents memory leak)
            merge_enabled: Whether to merge duplicate experiences
            merge_by_smiles: Merge experiences with identical SMILES pairs
            merge_by_fg_pattern: Merge experiences with same FG change pattern
            read_only: If True, pool only loads from file and doesn't save or add new experiences
            svs_config: SVS (Skill Value Score) configuration dict with keys: w1, w2, c
        """
        self.max_size = max_size
        self.save_path = save_path
        self.include_failures = include_failures
        self.min_score_delta = min_score_delta
        self.max_pending_size = max_pending_size
        self.read_only = read_only

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Task -> experiences mapping
        self.experiences: Dict[str, List[SkillEntry]] = defaultdict(list)

        # Pending experiences waiting for GPT summary
        self.pending_for_gpt: List[SkillEntry] = []

        # FG statistics cache (updated periodically)
        self.fg_statistics: Dict[str, Dict] = {}

        # Merge configuration
        self.merge_enabled = merge_enabled
        self.merge_by_smiles = merge_by_smiles
        self.merge_by_fg_pattern = merge_by_fg_pattern

        # SVS (Skill Value Score) configuration
        self.svs_config = svs_config or {
            'w1': 0.3,   # initial delta weight
            'w2': 0.7,   # usage delta weight
            'c': 0.5,    # exploration factor
        }
        self._total_retrieval_count = 0  # global retrieval counter

        # Load existing if available
        if save_path and os.path.exists(save_path):
            self.load()
        elif self.read_only and save_path:
            logger.warning(f"[Experience Pool] Read-only mode but file not found: {save_path}")

    def _get_smiles_key(self, exp: SkillEntry) -> str:
        """Generate unique key based on SMILES pair"""
        return f"{exp.before_smiles}|{exp.after_smiles}"

    def _get_fg_pattern_key(self, exp: SkillEntry) -> str:
        """Generate unique key based on functional group change pattern"""
        # Filter out empty strings and strip whitespace
        removed = sorted([fg.strip() for fg in (exp.functional_groups_removed or []) if fg and fg.strip()])
        added = sorted([fg.strip() for fg in (exp.functional_groups_added or []) if fg and fg.strip()])
        return f"removed:{','.join(removed)}|added:{','.join(added)}"

    def _merge_experience(self, existing: SkillEntry, new: SkillEntry) -> bool:
        """
        Merge two experiences, keeping the one with best score_delta.

        Args:
            existing: The experience already in the pool
            new: The new experience to merge

        Returns:
            True (always succeeds)
        """
        existing.occurrence_count += 1
        existing.merged_from.append(new.exp_id)

        # Limit merged_from list size to prevent memory growth
        if len(existing.merged_from) > 100:
            existing.merged_from = existing.merged_from[-100:]

        # Determine if new experience is better (task-aware comparison)
        # For SA task: smaller delta is better (we want to decrease score)
        # For others: larger delta is better (we want to increase score)
        is_better = False
        if existing.task == 'sa':
            # SA: more negative delta = better improvement
            is_better = new.score_delta < existing.score_delta
        else:
            # Other tasks: more positive delta = better improvement
            is_better = new.score_delta > existing.score_delta

        if is_better:
            existing.score_before = new.score_before
            existing.score_after = new.score_after
            existing.score_delta = new.score_delta
            existing.atom_changes = new.atom_changes
            existing.ring_changes = new.ring_changes
            existing.mw_change = new.mw_change
            existing.trajectory = new.trajectory

            # If new experience has GPT summary and existing doesn't, use new one
            if new.gpt_summary and not existing.gpt_summary:
                existing.gpt_summary = new.gpt_summary

        logger.debug(f"Merged experience {new.exp_id} into {existing.exp_id}, count={existing.occurrence_count}")
        return True

    def add_experience(self, exp: SkillEntry) -> bool:
        """
        Add a new experience to the pool (thread-safe).

        Args:
            exp: SkillEntry to add

        Returns:
            True if added, False if filtered out
        """
        # Read-only mode: don't add new experiences
        if self.read_only:
            return False

        # Filter by success/failure preference
        if not self.include_failures and not exp.success:
            return False

        # Filter by minimum delta
        if abs(exp.score_delta) < self.min_score_delta:
            return False

        # Compute retrieval indices if not already done
        if exp.before_fingerprint is None:
            try:
                exp.compute_indices()
            except Exception as e:
                logger.warning(f"Failed to compute indices for experience: {e}")
                return False

        with self._lock:
            task_exps = self.experiences[exp.task]

            # Check for duplicates and merge if enabled
            if self.merge_enabled:
                # 1. Check SMILES duplicates
                if self.merge_by_smiles:
                    smiles_key = self._get_smiles_key(exp)
                    for existing in task_exps:
                        if self._get_smiles_key(existing) == smiles_key:
                            return self._merge_experience(existing, exp)

                # 2. Check FG pattern duplicates
                if self.merge_by_fg_pattern:
                    fg_key = self._get_fg_pattern_key(exp)
                    # Only merge by FG pattern if there are actual FG changes
                    if fg_key != "removed:|added:":
                        for existing in task_exps:
                            if self._get_fg_pattern_key(existing) == fg_key:
                                return self._merge_experience(existing, exp)

            # No duplicates found, add as new experience
            task_exps.append(exp)

            # Add to pending for GPT (with size limit to prevent memory leak)
            self.pending_for_gpt.append(exp)
            if len(self.pending_for_gpt) > self.max_pending_size:
                # Remove oldest pending experiences
                self.pending_for_gpt = self.pending_for_gpt[-self.max_pending_size:]

            # Enforce max size by removing lowest SVS experiences
            if len(task_exps) > self.max_size:
                # Filter out deprecated experiences first
                active_exps = [e for e in task_exps if not e.is_deprecated]
                # Pre-fetch total count for thread-safe SVS computation
                total_count = self._total_retrieval_count
                # Sort by SVS score (higher = keep)
                active_exps.sort(key=lambda x: self.compute_svs(x, total_count), reverse=True)
                # Keep top max_size active experiences
                self.experiences[exp.task] = active_exps[:self.max_size]

        return True

    # ============================================================
    # SVS (Skill Value Score) Methods
    # ============================================================

    def compute_svs(self, exp: SkillEntry, total_retrieval_count: Optional[int] = None) -> float:
        """
        Compute Skill Value Score for an experience.

        SVS = (w1 × |initial_delta| + w2 × avg_usage_delta) + c × sqrt(ln(N_total+1) / (N_e+1))

        Args:
            exp: The experience to score
            total_retrieval_count: Optional pre-fetched count for thread safety

        Returns:
            SVS score (higher = better)
        """
        import math

        w1 = self.svs_config.get('w1', 0.3)
        w2 = self.svs_config.get('w2', 0.7)
        c = self.svs_config.get('c', 0.5)

        # Base value: weighted combination of initial and usage deltas
        initial_value = abs(exp.score_delta)  # use absolute value for SA compatibility
        usage_value = exp.avg_usage_delta if exp.usage_count > 0 else 0.0
        base_score = w1 * initial_value + w2 * usage_value

        # Thread-safe access to total retrieval count
        if total_retrieval_count is None:
            with self._lock:
                total_retrieval_count = self._total_retrieval_count

        # UCB exploration bonus: encourages trying less-used experiences
        if total_retrieval_count > 0:
            explore_bonus = c * math.sqrt(
                math.log(total_retrieval_count + 1) / (exp.retrieval_count + 1)
            )
        else:
            explore_bonus = c  # initial bonus when no retrievals yet

        return base_score + explore_bonus

    def _find_by_id(self, exp_id: str) -> Optional[SkillEntry]:
        """Find experience by exp_id across all tasks."""
        for task_exps in self.experiences.values():
            for exp in task_exps:
                if exp.exp_id == exp_id:
                    return exp
        return None

    def record_retrieval(self, exp_ids: List[str]) -> int:
        """
        Record that experiences were retrieved and shown to the model.

        Args:
            exp_ids: List of experience IDs that were retrieved

        Returns:
            Number of experiences successfully updated
        """
        if self.read_only:
            return 0

        updated = 0
        with self._lock:
            self._total_retrieval_count += len(exp_ids)
            for exp_id in exp_ids:
                exp = self._find_by_id(exp_id)
                if exp:
                    exp.retrieval_count += 1
                    updated += 1
        return updated

    def record_outcome(self, exp_ids: List[str], score_delta: float) -> int:
        """
        Record the outcome after using retrieved experiences.

        Args:
            exp_ids: List of experience IDs that were used
            score_delta: Score change after the step (positive = improvement)

        Returns:
            Number of experiences successfully updated
        """
        if self.read_only:
            return 0

        updated = 0
        with self._lock:
            for exp_id in exp_ids:
                exp = self._find_by_id(exp_id)
                if exp:
                    exp.total_usage_delta += score_delta
                    if score_delta > 0:
                        exp.positive_usage_count += 1
                    else:
                        exp.negative_usage_count += 1
                    updated += 1
        return updated

    def deprecate_experience(self, exp_id: str, reason: str = "manual") -> bool:
        """
        Soft-delete an experience by marking it as deprecated.

        Args:
            exp_id: Experience ID to deprecate
            reason: Reason for deprecation

        Returns:
            True if experience was found and deprecated
        """
        if self.read_only:
            return False

        with self._lock:
            exp = self._find_by_id(exp_id)
            if exp:
                exp.is_deprecated = True
                exp.deprecated_reason = reason
                return True
        return False

    def auto_prune(self,
                   min_retrieval: int = 10,
                   max_negative_rate: float = 0.7) -> int:
        """
        Automatically deprecate low-effectiveness experiences.

        Criteria: retrieval_count >= min_retrieval AND usage_success_rate < (1 - max_negative_rate)

        Args:
            min_retrieval: Minimum retrievals before considering for pruning
            max_negative_rate: Maximum allowed negative outcome rate

        Returns:
            Number of experiences deprecated
        """
        if self.read_only:
            return 0

        pruned_count = 0
        with self._lock:
            for task_exps in self.experiences.values():
                for exp in task_exps:
                    if exp.is_deprecated:
                        continue
                    # Check pruning criteria
                    if (exp.retrieval_count >= min_retrieval and
                        exp.usage_count > 0 and
                        exp.usage_success_rate < (1 - max_negative_rate)):
                        exp.is_deprecated = True
                        exp.deprecated_reason = f"auto_prune:success_rate={exp.usage_success_rate:.2f}"
                        pruned_count += 1

        if pruned_count > 0:
            logger.info(f"[Experience Pool] Auto-pruned {pruned_count} low-effectiveness experiences")

        return pruned_count

    # ============================================================
    # Consolidation Methods (GPT-based experience merging)
    # ============================================================

    def _compute_tanimoto(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute Tanimoto similarity between two fingerprints."""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0

    def find_consolidation_candidates(self,
                                       task: str,
                                       min_svs_percentile: float = 0.7,
                                       similarity_threshold: float = 0.8,
                                       min_group_size: int = 3,
                                       max_group_size: int = 5) -> List[List['SkillEntry']]:
        """
        Find groups of similar high-SVS experiences that can be consolidated.

        Strategy:
        1. Filter to top N% by SVS score
        2. Cluster by Tanimoto fingerprint similarity
        3. Return groups meeting min_group_size

        Args:
            task: Task type to consolidate
            min_svs_percentile: Only consider top N% SVS experiences (0.7 = top 30%)
            similarity_threshold: Minimum Tanimoto similarity for grouping
            min_group_size: Minimum experiences to form a consolidation group
            max_group_size: Maximum experiences per group

        Returns:
            List of experience groups suitable for consolidation
        """
        if task not in self.experiences:
            return []

        # Get active experiences with GPT summary
        active_exps = [e for e in self.experiences[task]
                       if not e.is_deprecated and not e.is_consolidated and e.gpt_summary]

        if len(active_exps) < min_group_size:
            return []

        # Score and sort by SVS
        total_count = self._total_retrieval_count
        scored = [(e, self.compute_svs(e, total_count)) for e in active_exps]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N% (at least min_group_size * 2 for clustering)
        cutoff_idx = max(int(len(scored) * (1 - min_svs_percentile)), min_group_size * 2)
        top_exps = [e for e, _ in scored[:cutoff_idx]]

        # Cluster by fingerprint similarity
        groups = []
        used = set()

        for exp in top_exps:
            if exp.exp_id in used:
                continue
            if exp.before_fingerprint is None:
                continue

            # Build group with similar experiences
            group = [exp]
            for other in top_exps:
                if other.exp_id == exp.exp_id or other.exp_id in used:
                    continue
                if other.before_fingerprint is None:
                    continue

                sim = self._compute_tanimoto(exp.before_fingerprint, other.before_fingerprint)
                if sim >= similarity_threshold:
                    group.append(other)
                    if len(group) >= max_group_size:
                        break

            if len(group) >= min_group_size:
                groups.append(group)
                used.update(e.exp_id for e in group)

        return groups

    def consolidate_experiences(self,
                                group: List['SkillEntry'],
                                gpt_client) -> Optional['SkillEntry']:
        """
        Consolidate a group of similar experiences into one using GPT.

        Args:
            group: List of experiences to merge
            gpt_client: OpenAI client for GPT calls

        Returns:
            New consolidated experience, or None if failed
        """
        if len(group) < 2 or self.read_only:
            return None

        task = group[0].task
        total_count = self._total_retrieval_count

        # Prepare GPT prompt
        strategies = "\n".join([
            f"{i+1}. {exp.gpt_summary} (SVS: {self.compute_svs(exp, total_count):.2f})"
            for i, exp in enumerate(group)
        ])

        prompt = f"""You are an expert medicinal chemist. Analyze these {len(group)} similar molecular optimization strategies and consolidate them into ONE more profound, generalizable strategy.

=== Task ===
{task} optimization

=== Strategies to Consolidate ===
{strategies}

=== Instructions ===
1. Identify the COMMON PATTERN across all strategies
2. Abstract away specific molecule details
3. Create ONE unified strategy that captures the essence

=== Output Format ===
Write a single, actionable strategy sentence (15-30 words) that:
- Describes the general chemical transformation principle
- Is more abstract than the individual strategies
- Can be applied to similar molecules

OUTPUT ONLY THE STRATEGY SENTENCE, nothing else."""

        # Call GPT
        try:
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            consolidated_summary = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Consolidation] GPT call failed: {e}")
            return None

        # Calculate SVS-weighted average statistics
        svs_scores = [self.compute_svs(e, total_count) for e in group]
        total_svs = sum(svs_scores)

        if total_svs == 0:
            total_svs = 1  # Prevent division by zero

        weighted_retrieval = sum(e.retrieval_count * s for e, s in zip(group, svs_scores)) / total_svs
        weighted_usage_delta = sum(e.total_usage_delta * s for e, s in zip(group, svs_scores)) / total_svs
        weighted_positive = sum(e.positive_usage_count * s for e, s in zip(group, svs_scores)) / total_svs
        weighted_negative = sum(e.negative_usage_count * s for e, s in zip(group, svs_scores)) / total_svs

        # Select best representative (highest SVS) as template
        best_idx = svs_scores.index(max(svs_scores))
        best_exp = group[best_idx]

        # Create new consolidated experience
        new_exp = SkillEntry(
            exp_id=str(uuid.uuid4()),
            task=task,
            before_smiles=best_exp.before_smiles,
            after_smiles=best_exp.after_smiles,
            score_before=best_exp.score_before,
            score_after=best_exp.score_after,
            score_delta=sum(e.score_delta for e in group) / len(group),
            success=best_exp.success,
            gpt_summary=consolidated_summary,
            before_fingerprint=best_exp.before_fingerprint.copy() if best_exp.before_fingerprint is not None else None,
            functional_group_tags=best_exp.functional_group_tags.copy(),
            # Inherit weighted average statistics
            retrieval_count=int(weighted_retrieval),
            total_usage_delta=weighted_usage_delta,
            positive_usage_count=int(weighted_positive),
            negative_usage_count=int(weighted_negative),
            # Mark as consolidation product
            is_from_consolidation=True,
            consolidated_from=[e.exp_id for e in group],
            occurrence_count=sum(e.occurrence_count for e in group)
        )

        # Soft-delete original experiences and add new one
        with self._lock:
            for exp in group:
                exp.is_consolidated = True
                exp.consolidated_into = new_exp.exp_id

            self.experiences[task].append(new_exp)

        logger.info(f"[Consolidation] Merged {len(group)} experiences -> {new_exp.exp_id[:8]}...")
        return new_exp

    def run_consolidation(self,
                          gpt_client,
                          tasks: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, int]:
        """
        Run consolidation on specified tasks.

        Args:
            gpt_client: OpenAI client for GPT calls
            tasks: List of tasks to consolidate (None = all tasks)
            **kwargs: Arguments passed to find_consolidation_candidates

        Returns:
            Dict mapping task -> number of consolidations performed
        """
        if self.read_only:
            return {}

        if tasks is None:
            tasks = list(self.experiences.keys())

        results = {}
        for task in tasks:
            groups = self.find_consolidation_candidates(task, **kwargs)
            count = 0
            for group in groups:
                new_exp = self.consolidate_experiences(group, gpt_client)
                if new_exp:
                    count += 1
            results[task] = count

        total = sum(results.values())
        if total > 0:
            logger.info(f"[Consolidation] Total: {total} consolidations across {len(results)} tasks")

        return results

    def query_by_functional_group(self,
                                   current_fgs: Dict[str, int],
                                   task: str,
                                   top_k: int = 3,
                                   min_jaccard: float = 0.3,
                                   success_only: bool = True) -> List[SkillEntry]:
        """
        Query experiences by matching functional groups.

        Uses two-stage filtering: first by Jaccard similarity, then ranked by SVS.

        Args:
            current_fgs: Detected FGs in current molecule {fg_name: count}
            task: Task type to filter by
            top_k: Number of results to return
            min_jaccard: Minimum Jaccard similarity threshold
            success_only: Only return successful experiences

        Returns:
            List of relevant experiences sorted by SVS score
        """
        if task not in self.experiences:
            return []

        # Apply pruning to input FGs for consistent comparison
        current_fg_set = set(FunctionalGroupPruner.prune(list(current_fgs.keys())))
        results = []

        for exp in self.experiences[task]:
            # Skip deprecated or consolidated experiences
            if exp.is_deprecated or exp.is_consolidated:
                continue
            # Only include experiences with GPT summary
            if not exp.gpt_summary:
                continue
            if success_only and not exp.success:
                continue

            # Calculate Jaccard similarity for FG overlap (normalized)
            intersection = len(current_fg_set & exp.functional_group_tags)
            union = len(current_fg_set | exp.functional_group_tags)
            if intersection > 0 and union > 0:
                # Jaccard similarity normalizes for molecule complexity
                jaccard_sim = intersection / union

                # Apply Jaccard threshold filter
                if jaccard_sim < min_jaccard:
                    continue

                # Compute SVS score for ranking
                svs = self.compute_svs(exp)
                results.append((exp, svs, jaccard_sim))

        # Sort by SVS score (higher = better)
        results.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _, _ in results[:top_k]]

    def query_by_similarity(self,
                            smiles: str,
                            task: str,
                            top_k: int = 2,
                            min_similarity: float = 0.3) -> List[SkillEntry]:
        """
        Query experiences by molecular similarity.

        Uses two-stage filtering: first by Tanimoto similarity, then ranked by SVS.

        Args:
            smiles: SMILES of current molecule
            task: Task type
            top_k: Number of results
            min_similarity: Minimum Tanimoto similarity threshold

        Returns:
            List of similar experiences sorted by SVS score
        """
        if task not in self.experiences:
            return []

        # Compute query fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        query_fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        query_arr = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(query_fp, query_arr)

        results = []
        for exp in self.experiences[task]:
            # Skip deprecated or consolidated experiences
            if exp.is_deprecated or exp.is_consolidated:
                continue
            # Only include experiences with GPT summary
            if not exp.gpt_summary:
                continue
            if exp.before_fingerprint is None:
                continue

            # Compute Tanimoto similarity
            intersection = np.sum(query_arr & exp.before_fingerprint)
            union = np.sum(query_arr | exp.before_fingerprint)
            similarity = intersection / union if union > 0 else 0

            if similarity >= min_similarity:
                # Compute SVS score for ranking
                svs = self.compute_svs(exp)
                results.append((exp, svs, similarity))

        # Sort by SVS score (higher = better)
        results.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _, _ in results[:top_k]]

    def query_by_task(self,
                      task: str,
                      top_k: int = 2,
                      success_only: bool = True) -> List[SkillEntry]:
        """
        Query best experiences for a task (general strategies).
        Only returns experiences with GPT summary.

        Args:
            task: Task type
            top_k: Number of results
            success_only: Only return successful experiences

        Returns:
            List of top experiences by improvement (with GPT summary only)
        """
        if task not in self.experiences:
            return []

        exps = self.experiences[task]
        # Only include experiences with GPT summary
        exps = [e for e in exps if e.gpt_summary]
        if success_only:
            exps = [e for e in exps if e.success]

        # Sort by improvement
        # SA task: lower score is better (negative delta = improvement), sort ascending
        # Other tasks: higher score is better (positive delta = improvement), sort descending
        if task == 'sa':
            exps = sorted(exps, key=lambda x: x.score_delta, reverse=False)
        else:
            exps = sorted(exps, key=lambda x: x.score_delta, reverse=True)
        return exps[:top_k]

    def get_pending_experiences(self, batch_size: int = 30) -> List[SkillEntry]:
        """Get experiences pending GPT summarization"""
        pending = self.pending_for_gpt[:batch_size]
        return pending

    def update_gpt_summaries(self, summaries: Dict[str, str]):
        """
        Update experiences with GPT-generated summaries.

        Args:
            summaries: Dict mapping exp_id to summary text
        """
        # Update summaries in BOTH pending_for_gpt AND experiences
        # (needed because after load(), they are different objects)
        for exp in self.pending_for_gpt:
            if exp.exp_id in summaries:
                exp.gpt_summary = summaries[exp.exp_id]

        # Also update in experiences (in case objects differ after load)
        for task_exps in self.experiences.values():
            for exp in task_exps:
                if exp.exp_id in summaries:
                    exp.gpt_summary = summaries[exp.exp_id]

        # Remove summarized exp_ids from pending
        summarized_ids = set(summaries.keys())
        self.pending_for_gpt = [
            exp for exp in self.pending_for_gpt
            if exp.exp_id not in summarized_ids
        ]

    def generate_gpt_summaries(self, batch_size: int = None, total_timeout: int = 600, max_workers: int = 8) -> Dict[str, str]:
        """
        Generate GPT summaries for ALL pending experiences using parallel requests.

        Args:
            batch_size: Ignored (processes all pending). Kept for backwards compatibility.
            total_timeout: Maximum total time in seconds (default 10 minutes)
            max_workers: Number of parallel requests (default 8)

        Returns:
            Dict mapping exp_id to generated summary
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Get ALL pending experiences
        with self._lock:
            pending = list(self.pending_for_gpt)

        if not pending:
            return {}

        try:
            from openai import AzureOpenAI, RateLimitError, AuthenticationError, APITimeoutError, APIConnectionError
        except ImportError:
            logger.warning("openai package not installed, skipping GPT summarization")
            return {}

        # Get API key from environment
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            api_key = "YOUR_API_KEY_HERE"  # Replace with your API key
            logger.warning("Set AZURE_OPENAI_API_KEY environment variable for production.")

        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT",
                                              "YOUR_AZURE_ENDPOINT"),
                timeout=30.0
            )
        except Exception as e:
            logger.warning(f"Failed to create Azure OpenAI client: {e}")
            return {}

        def process_single_exp(exp: SkillEntry) -> tuple:
            """Process a single experience and return (exp_id, summary or None)"""
            prompt = format_gpt_prompt(exp)
            max_retries = 2

            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()

                    if len(summary) < 10:
                        return (exp.exp_id, None)

                    return (exp.exp_id, summary)

                except RateLimitError:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    continue
                except (APITimeoutError, APIConnectionError):
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    continue
                except AuthenticationError as e:
                    logger.error(f"Authentication failed: {e}")
                    return (exp.exp_id, None)
                except Exception as e:
                    logger.debug(f"GPT failed for {exp.exp_id}: {e}")
                    return (exp.exp_id, None)

            return (exp.exp_id, None)

        summaries = {}
        start_time = time.time()

        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_exp, exp): exp for exp in pending}

            for future in as_completed(futures):
                # Check timeout
                if time.time() - start_time > total_timeout:
                    logger.warning(f"GPT summarization timeout after {total_timeout}s")
                    break

                try:
                    exp_id, summary = future.result(timeout=30)
                    if summary:
                        summaries[exp_id] = summary
                except Exception as e:
                    logger.debug(f"Future failed: {e}")

        # Update summaries
        if summaries:
            with self._lock:
                self.update_gpt_summaries(summaries)
            logger.info(f"[GPT Summary] Generated {len(summaries)}/{len(pending)} summaries in {time.time()-start_time:.1f}s")

        return summaries

    def get_fg_statistics(self, task: str = None) -> Dict:
        """
        Get functional group change statistics.

        Args:
            task: Optional task filter

        Returns:
            Dict of FG change statistics
        """
        if task:
            exps = self.experiences.get(task, [])
        else:
            exps = []
            for task_exps in self.experiences.values():
                exps.extend(task_exps)

        return aggregate_fg_statistics(exps)

    def get_fg_success_rates(self, task: str = None, min_count: int = 2) -> Dict[str, Dict]:
        """
        Learn functional group transformation success rates from experience pool.

        This method analyzes the experience pool to determine which functional group
        additions/removals lead to better scores, enabling dynamic learning of
        effective molecular transformations.

        Args:
            task: Optional task filter (e.g., 'qed', 'sa')
            min_count: Minimum observation count to include in results

        Returns:
            Dict with 'recommended' (positive effect) and 'avoid' (negative effect) FG changes,
            each containing transformation details sorted by effectiveness
        """
        # Thread-safe access to experiences
        with self._lock:
            if task:
                exps = list(self.experiences.get(task, []))
            else:
                exps = []
                for task_exps in self.experiences.values():
                    exps.extend(task_exps)

        if not exps:
            return {'recommended': [], 'avoid': [], 'all_stats': {}}

        # Collect FG transformation statistics
        fg_transforms = defaultdict(list)  # {transform_pattern: [(score_delta, task), ...]}

        for exp in exps:
            # Validate score_delta is valid
            if exp.score_delta is None or np.isnan(exp.score_delta):
                continue

            # Get FG changes
            for fg in (exp.functional_groups_added or []):
                if fg and fg.strip():
                    fg_transforms[f"+{fg.strip()}"].append((exp.score_delta, exp.task))
            for fg in (exp.functional_groups_removed or []):
                if fg and fg.strip():
                    fg_transforms[f"-{fg.strip()}"].append((exp.score_delta, exp.task))

        # Compute statistics for each transformation
        all_stats = {}
        for pattern, observations in fg_transforms.items():
            if len(observations) < min_count:
                continue

            deltas = [d for d, _ in observations]
            tasks = [t for _, t in observations]

            # For SA task, lower delta is better (minimize SA)
            # For others, higher delta is better (maximize QED, etc.)
            # Check if 'sa' appears in any task name (including multi-objective like 'sa+qed')
            sa_count = sum(1 for t in tasks if 'sa' in t.lower())
            is_sa_dominant = sa_count > len(tasks) / 2

            avg_delta = sum(deltas) / len(deltas)
            success_rate = sum(1 for d in deltas if d > 0) / len(deltas) if not is_sa_dominant else \
                          sum(1 for d in deltas if d < 0) / len(deltas)

            all_stats[pattern] = {
                'pattern': pattern,
                'count': len(observations),
                'avg_delta': avg_delta,
                'success_rate': success_rate,
                'max_delta': max(deltas),
                'min_delta': min(deltas),
                'std_delta': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                'is_sa_task': is_sa_dominant
            }

        # Separate into recommended and avoid based on average effect
        recommended = []
        avoid = []

        for pattern, stats in all_stats.items():
            # Determine effectiveness based on task type
            if stats['is_sa_task']:
                # For SA: negative delta = good (lowering SA)
                is_effective = stats['avg_delta'] < 0
            else:
                # For others: positive delta = good
                is_effective = stats['avg_delta'] > 0

            if is_effective:
                recommended.append(stats)
            else:
                avoid.append(stats)

        # Sort by effectiveness (absolute average delta)
        recommended.sort(key=lambda x: abs(x['avg_delta']), reverse=True)
        avoid.sort(key=lambda x: abs(x['avg_delta']), reverse=True)

        return {
            'recommended': recommended,
            'avoid': avoid,
            'all_stats': all_stats
        }

    def get_top_transformations(self, task: str, top_k: int = 5) -> List[str]:
        """
        Get top recommended functional group transformations for a task.

        Args:
            task: Task type (e.g., 'qed', 'sa')
            top_k: Number of recommendations to return

        Returns:
            List of recommended transformation patterns (e.g., ['+phenol', '-nitro'])
        """
        success_rates = self.get_fg_success_rates(task=task, min_count=2)
        recommended = success_rates.get('recommended', [])
        return [r['pattern'] for r in recommended[:top_k]]

    def __len__(self) -> int:
        """Return total number of experiences in the pool"""
        return sum(len(exps) for exps in self.experiences.values())

    def get_pool_stats(self) -> Dict:
        """Get overall pool statistics"""
        total = sum(len(exps) for exps in self.experiences.values())
        success_count = sum(
            sum(1 for e in exps if e.success)
            for exps in self.experiences.values()
        )
        with_summary = sum(
            sum(1 for e in exps if e.gpt_summary)
            for exps in self.experiences.values()
        )

        # Calculate avg_delta across all experiences
        total_delta = sum(
            exp.score_delta for task_exps in self.experiences.values() for exp in task_exps
        )
        avg_delta = total_delta / total if total > 0 else 0.0

        # SVS usage tracking stats
        all_exps = [exp for task_exps in self.experiences.values() for exp in task_exps]
        deprecated_count = sum(1 for e in all_exps if e.is_deprecated)
        high_freq_count = sum(1 for e in all_exps if e.retrieval_count >= 5)
        total_retrievals = sum(e.retrieval_count for e in all_exps)
        avg_retrieval = total_retrievals / total if total > 0 else 0.0

        # Calculate average SVS for non-deprecated experiences
        active_exps = [e for e in all_exps if not e.is_deprecated]
        if active_exps:
            avg_svs = sum(self.compute_svs(e) for e in active_exps) / len(active_exps)
        else:
            avg_svs = 0.0

        return {
            'total_experiences': total,
            'success_experiences': success_count,
            'failure_experiences': total - success_count,
            'with_gpt_summary': with_summary,
            'pending_for_gpt': len(self.pending_for_gpt),
            'avg_delta': avg_delta,
            'tasks': list(self.experiences.keys()),
            'experiences_per_task': {
                task: len(exps) for task, exps in self.experiences.items()
            },
            # SVS usage tracking
            'usage_tracking': {
                'total_retrievals': self._total_retrieval_count,
                'avg_retrieval_per_exp': avg_retrieval,
                'high_freq_count': high_freq_count,
                'deprecated_count': deprecated_count,
                'active_count': total - deprecated_count,
                'avg_svs': avg_svs,
            },
            # Consolidation stats
            'consolidation': {
                'consolidated_count': sum(1 for e in all_exps if e.is_consolidated),
                'from_consolidation_count': sum(1 for e in all_exps if e.is_from_consolidation),
            },
        }

    def save(self, path: Optional[str] = None):
        """Save pool to disk atomically (thread-safe)"""
        # Read-only mode: don't save
        if self.read_only:
            return

        save_path = path or self.save_path
        if not save_path:
            return

        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with self._lock:
            # PKL: full data with fingerprints
            data_full = {
                'max_size': self.max_size,
                'include_failures': self.include_failures,
                'min_score_delta': self.min_score_delta,
                'experiences': {
                    task: [exp.to_dict() for exp in exps]
                    for task, exps in self.experiences.items()
                },
                'pending_for_gpt': [exp.to_dict() for exp in self.pending_for_gpt],
                'fg_statistics': self.fg_statistics,
                # SVS tracking state
                'svs_config': self.svs_config,
                '_total_retrieval_count': self._total_retrieval_count,
            }
            # JSON: simplified data without fingerprints (for human readability)
            data_json = {
                'max_size': self.max_size,
                'include_failures': self.include_failures,
                'min_score_delta': self.min_score_delta,
                'experiences': {
                    task: [exp.to_dict_for_json() for exp in exps]
                    for task, exps in self.experiences.items()
                },
                'pending_for_gpt': [exp.to_dict_for_json() for exp in self.pending_for_gpt],
                'fg_statistics': self.fg_statistics,
                # SVS tracking state
                'svs_config': self.svs_config,
                '_total_retrieval_count': self._total_retrieval_count,
            }

        # Atomic save for JSON using temp file + move
        tmp_json_path = None
        tmp_pkl_path = None
        try:
            with tempfile.NamedTemporaryFile('w', delete=False,
                                             dir=save_dir or '.', suffix='.json') as tmp_f:
                json.dump(data_json, tmp_f, indent=2)
                tmp_json_path = tmp_f.name
            shutil.move(tmp_json_path, save_path)
            tmp_json_path = None  # Clear after successful move

            # Atomic save for pickle (full data)
            pkl_path = save_path.replace('.json', '.pkl')
            with tempfile.NamedTemporaryFile('wb', delete=False,
                                             dir=save_dir or '.', suffix='.pkl') as tmp_f:
                pickle.dump(data_full, tmp_f)
                tmp_pkl_path = tmp_f.name
            shutil.move(tmp_pkl_path, pkl_path)
            tmp_pkl_path = None  # Clear after successful move

            logger.info(f"Experience pool saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save experience pool: {e}")
            # Clean up any remaining temp files
            for tmp_path in [tmp_json_path, tmp_pkl_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    def load(self, path: Optional[str] = None):
        """Load pool from disk (PKL only)"""
        load_path = path or self.save_path
        if not load_path:
            return

        # Only load from PKL (JSON is for human readability only)
        pkl_path = load_path.replace('.json', '.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Experience pool PKL file not found: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # Restore state
        self.max_size = data.get('max_size', self.max_size)
        self.include_failures = data.get('include_failures', self.include_failures)
        self.min_score_delta = data.get('min_score_delta', self.min_score_delta)

        self.experiences = {
            task: [SkillEntry.from_dict(exp_data) for exp_data in exps]
            for task, exps in data.get('experiences', {}).items()
        }

        self.pending_for_gpt = [
            SkillEntry.from_dict(exp_data)
            for exp_data in data.get('pending_for_gpt', [])
        ]

        self.fg_statistics = data.get('fg_statistics', {})

        # Restore SVS tracking state (with defaults for backward compatibility)
        if 'svs_config' in data:
            self.svs_config.update(data['svs_config'])
        self._total_retrieval_count = data.get('_total_retrieval_count', 0)

        print(f"Loaded experience pool with {self.get_pool_stats()['total_experiences']} experiences")


# ============================================================
# GPT Summary Generation
# ============================================================

GPT_SUMMARY_PROMPT = """Analyze this molecular transformation for {task} optimization:

=== Molecules ===
Before: {before_smiles}
After:  {after_smiles}
Score:  {score_before:.3f} -> {score_after:.3f} ({score_delta:+.3f})

=== MCS Analysis ===
Modification: {modification_type}
- Removed: {removed_fragment}
- Added:   {added_fragment}

=== Scaffold Analysis ===
Before Scaffold: {before_scaffold}
After Scaffold:  {after_scaffold}
Scaffold Type: {scaffold_type}

=== Functional Group Changes ===
- Removed: {fg_removed}
- Added:   {fg_added}

=== Property Changes ===
- MW: {mw_change:+.1f} Da | Rings: {ring_changes:+d}
- PSA: {psa_change:+.1f} A2 | HBD: {hbd_change:+d} | HBA: {hba_change:+d}

Result: {result}

=== Task ===
Generate ONE actionable strategy sentence following this format:

CONSTRAINTS:
1. Focus on the 1-2 MOST IMPORTANT functional group changes
2. Format: "[Action] [What] [Where (if clear)] to [Effect]"
3. If scaffold_type is "scaffold_hop", mention the core change (e.g., "Replace benzene with pyridine").
4. Use the Removed/Added Fragment from MCS for precise description.
5. If location is clear from MCS, specify it (e.g., "on the aromatic ring").

EXAMPLES:
- "Replace benzene core with pyridine to improve water solubility and {task}." (scaffold_hop)
- "Add fluorine (-F) to the aromatic ring to enhance metabolic stability." (addition)
- "Remove the sulfonamide group from aromatic ring to reduce polar surface area." (removal)
- "Replace methoxy (-OCH3) with fluorine (-F) to decrease MW and improve {task}." (replacement)

Focus on: WHAT changed, WHERE (if clear from MCS), and WHY it improves {task}:"""


def format_gpt_prompt(exp: SkillEntry) -> str:
    """Format experience into GPT prompt with type-safe values.

    Applies name mapping for RDKit fragments and cleans up MCS atom soup.
    """
    # Map RDKit fragment names to human-readable descriptions
    fg_removed = [map_fg_name(fg) for fg in (exp.functional_groups_removed or [])]
    fg_added = [map_fg_name(fg) for fg in (exp.functional_groups_added or [])]

    # Clean up MCS fragment atom soup
    removed_frag = clean_fragment_string(exp.removed_fragment or 'none')
    added_frag = clean_fragment_string(exp.added_fragment or 'none')

    return GPT_SUMMARY_PROMPT.format(
        task=exp.task,
        before_smiles=exp.before_smiles,
        after_smiles=exp.after_smiles,
        score_before=float(exp.score_before) if exp.score_before else 0.0,
        score_after=float(exp.score_after) if exp.score_after else 0.0,
        score_delta=float(exp.score_delta) if exp.score_delta else 0.0,
        # MCS analysis (cleaned)
        modification_type=exp.modification_type or 'unknown',
        removed_fragment=removed_frag if removed_frag else 'none',
        added_fragment=added_frag if added_frag else 'none',
        # Scaffold analysis
        before_scaffold=exp.before_scaffold or 'none',
        after_scaffold=exp.after_scaffold or 'none',
        scaffold_type=exp.scaffold_type or 'unknown',
        # Functional groups (mapped to readable names)
        fg_removed=fg_removed,
        fg_added=fg_added,
        # Properties
        mw_change=float(exp.mw_change) if exp.mw_change else 0.0,
        ring_changes=int(exp.ring_changes) if exp.ring_changes else 0,
        psa_change=float(exp.psa_change) if exp.psa_change else 0.0,
        hbd_change=int(exp.hbd_change) if exp.hbd_change else 0,
        hba_change=int(exp.hba_change) if exp.hba_change else 0,
        result="SUCCESS" if exp.success else "FAILURE"
    )


# ============================================================
# Prompt Injection Formatting
# ============================================================

def format_hints_for_prompt(experiences: List[SkillEntry],
                            task: str,
                            include_failures: bool = False) -> str:
    """
    Format experiences into prompt hints.

    Only includes experiences that have GPT-generated summaries.
    Returns empty string if no relevant experiences with summaries are found.

    Args:
        experiences: List of relevant experiences
        task: Current task type
        include_failures: Whether to include failure strategies (default False)

    Returns:
        Formatted string to inject into prompt, or empty string if no valid experiences
    """
    if not experiences:
        return ""

    # Filter only experiences with GPT summaries (no fallback to raw FG lists)
    success_exps_with_summary = [e for e in experiences if e.success and e.gpt_summary]

    # If no experiences with GPT summaries, return empty (don't force add hints)
    if not success_exps_with_summary:
        return ""

    lines = [f"\n=== Potential Useful Strategies for {task} ==="]

    for i, exp in enumerate(success_exps_with_summary[:5], 1):
        lines.append(f"{i}. {exp.gpt_summary}")

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
    success: bool,
    epoch: int = 0
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
        success: Whether optimization succeeded
        epoch: Current training epoch

    Returns:
        SkillEntry object
    """
    # Analyze transformation
    analysis = analyze_transformation(before_smiles, after_smiles)

    exp = SkillEntry(
        task=task,
        created_epoch=epoch,
        before_smiles=before_smiles,
        after_smiles=after_smiles,
        trajectory=trajectory,
        score_before=score_before,
        score_after=score_after,
        score_delta=score_after - score_before,
        success=success,
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
        # MCS analysis
        mcs_smarts=analysis.get('mcs_smarts', ''),
        removed_fragment=analysis.get('removed_fragment', ''),
        added_fragment=analysis.get('added_fragment', ''),
        modification_type=analysis.get('modification_type', ''),
        # Scaffold analysis
        before_scaffold=analysis.get('before_scaffold', ''),
        after_scaffold=analysis.get('after_scaffold', ''),
        scaffold_changed=analysis.get('scaffold_changed', False),
        is_scaffold_hop=analysis.get('is_scaffold_hop', False),
        scaffold_type=analysis.get('scaffold_type', ''),
    )

    # Compute retrieval indices
    exp.compute_indices()

    return exp
