#!/usr/bin/env python3
"""
Static Exemplar Memory Server for MolMem

This module implements the Static Exemplar Memory component of the MolMem
(Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization) framework.

Static Exemplar Memory provides cold-start grounding by retrieving similar molecules
from a large pre-indexed chemical database using:
- Morgan fingerprints (ECFP4: radius=2, 2048-bit)
- FAISS index for fast approximate nearest neighbor search
- Tanimoto similarity filtering (threshold ≥ 0.4)

Usage:
    python static_exemplar_memory.py \
        --index_path molecular_faiss.index \
        --metadata_path molecular_metadata.pkl \
        --corpus_path molecular_corpus.jsonl \
        --topk 10 \
        --port 8000
"""

import argparse
import copy
import pickle
import json
import logging
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import lru_cache

import numpy as np
import faiss
import datasets
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _mol_from_smiles_compat(smiles: str):
    """Parse SMILES, tolerating records that fail only RDKit kekulization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    sanitize_without_kekulization = (
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    )
    result = Chem.SanitizeMol(
        mol,
        sanitizeOps=sanitize_without_kekulization,
        catchErrors=True,
    )
    return mol if result == Chem.SanitizeFlags.SANITIZE_NONE else None


def load_corpus(corpus_path: str, use_dict: bool = True):
    """
    Load corpus with optimized loading strategy.

    Args:
        corpus_path: Path to JSONL corpus file
        use_dict: If True, load into memory dict (fast, O(1) access)
                 If False, use HuggingFace datasets (slower but memory efficient)

    Returns:
        corpus: Dictionary or HF dataset
    """
    if use_dict:
        # Fast loading: Load entire corpus into memory dictionary
        logger.info(f"Loading corpus into memory dict from: {corpus_path}")
        corpus_dict = {}
        with open(corpus_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    corpus_dict[idx] = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {idx}: {e}")
                    continue
        logger.info(f"✓ Loaded {len(corpus_dict)} documents into memory")
        return corpus_dict
    else:
        # Memory-efficient loading: Use HuggingFace datasets
        logger.info(f"Loading corpus using HuggingFace datasets from: {corpus_path}")
        corpus = datasets.load_dataset(
            'json',
            data_files=corpus_path,
            split="train",
            num_proc=4
        )
        return corpus


def load_docs(corpus, doc_idxs):
    """
    Load documents by indices with support for both dict and dataset.

    Args:
        corpus: Dictionary or HF dataset
        doc_idxs: List of document indices

    Returns:
        results: List of documents
    """
    if isinstance(corpus, dict):
        # O(1) dictionary lookup
        results = [corpus[int(idx)] for idx in doc_idxs if int(idx) in corpus]
    else:
        # HuggingFace dataset access
        results = [corpus[int(idx)] for idx in doc_idxs]
    return results


class MolecularFingerprinter:
    """Convert SMILES to Morgan fingerprints with LRU cache"""

    def __init__(self, radius: int = 2, n_bits: int = 2048, cache_size: int = 50000):
        self.radius = radius
        self.n_bits = n_bits
        # LRU cache for fingerprints (about 50MB for 50000 entries)
        self._fp_cache = {}
        self._cache_order = []  # Track access order for LRU
        self._cache_size = cache_size
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self):
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._fp_cache),
            'max_cache_size': self._cache_size
        }

    def clear_cache(self):
        """Clear fingerprint cache"""
        self._fp_cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Convert single SMILES to fingerprint with LRU caching.

        Args:
            smiles: SMILES string

        Returns:
            Fingerprint as numpy array (zero vector if SMILES is invalid)
        """
        # Check cache first
        if smiles in self._fp_cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._cache_order.remove(smiles)
            self._cache_order.append(smiles)
            return self._fp_cache[smiles].copy()  # Return copy to avoid modification

        # Cache miss - compute fingerprint
        self._cache_misses += 1

        mol = _mol_from_smiles_compat(smiles)
        if mol is None:
            # Invalid SMILES: log warning and return zero vector
            logger.warning(f"Invalid SMILES string: {smiles}")
            return np.zeros(self.n_bits, dtype=np.float32)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )

        arr = np.zeros(self.n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        # Add to cache
        self._fp_cache[smiles] = arr
        self._cache_order.append(smiles)

        # Evict oldest if cache is full
        if len(self._fp_cache) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._fp_cache[oldest]

        return arr

def _compute_single_tanimoto(args):
    """
    Helper function for parallel Tanimoto computation.
    Must be at module level for multiprocessing pickling.

    Args:
        args: Tuple of (query_fp_bits, candidate_smiles, radius, n_bits)

    Returns:
        Tanimoto similarity score
    """
    query_fp_bits, candidate_smiles, radius, n_bits = args

    mol = _mol_from_smiles_compat(candidate_smiles)
    if mol is None:
        return 0.0

    # Reconstruct query fingerprint from bit positions
    query_fp = DataStructs.ExplicitBitVect(n_bits)
    for bit in query_fp_bits:
        query_fp.SetBit(bit)

    # Generate candidate fingerprint
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    return DataStructs.TanimotoSimilarity(query_fp, candidate_fp)


class StaticExemplarMemory:
    """
    Static Exemplar Memory for cold-start grounding in MolMem.

    This memory component retrieves similar molecules from a large pre-indexed
    chemical database to provide initial guidance when optimizing a new lead molecule.

    Features:
    - Fast similarity search via FAISS + Morgan fingerprints
    - Tanimoto similarity filtering (threshold ≥ 0.4)
    - Property-aware retrieval (QED, LogP, SA, JNK3, DRD2)
    Query format: SMILES strings (e.g., "CCO", "CC(C)O")
    """

    def __init__(self, config):
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path
        self.metadata_path = config.metadata_path
        self.similarity_threshold = config.similarity_threshold
        self.faiss_gpu = config.faiss_gpu
        self.radius = config.morgan_radius
        self.n_bits = config.morgan_nbits
        self.search_multiplier = config.search_multiplier
        self.tanimoto_parallel_threshold = config.tanimoto_parallel_threshold

        # Load FAISS index
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        # Set nprobe for IndexIVFFlat (if applicable)
        if hasattr(self.index, 'nprobe') and hasattr(config, 'nprobe') and config.nprobe is not None:
            self.index.nprobe = config.nprobe
            print(f"✓ Set nprobe={config.nprobe} for IndexIVFFlat")

        if self.faiss_gpu:
            gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
            if gpu_count <= 0 or not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "--faiss_gpu was requested, but this Python environment has no "
                    "usable FAISS-GPU device. Install a GPU-enabled FAISS build and "
                    "verify faiss.get_num_gpus() > 0 before starting the server."
                )
            print("Enabling GPU acceleration for FAISS...")
            # Use StandardGpuResources for explicit memory management
            self.gpu_resources = faiss.StandardGpuResources()
            # Set temp memory to 512MB to avoid fragmentation
            self.gpu_resources.setTempMemory(512 * 1024 * 1024)

            # Use single GPU transfer (more stable than all_gpus with shard)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index, co)
            print("✓ GPU acceleration enabled with explicit memory management")

        # Load metadata
        print(f"Loading metadata from: {self.metadata_path}")
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.smiles_list = metadata['smiles_list']
        self.properties = metadata['properties']
        self.n_molecules = metadata['n_molecules']

        print(f"✓ Loaded {self.n_molecules:,} molecules")

        # Load corpus (for formatted output)
        print(f"Loading corpus from: {self.corpus_path}")
        self.corpus = load_corpus(self.corpus_path)

        # Initialize fingerprinter
        self.fingerprinter = MolecularFingerprinter(
            radius=self.radius,
            n_bits=self.n_bits
        )

        # Load binary fingerprints for fast Tanimoto calculation
        binary_fp_path = Path(self.index_path).parent / "molecular_binary_fingerprints.npy"
        bit_counts_path = Path(self.index_path).parent / "molecular_bit_counts.npy"

        if binary_fp_path.exists() and bit_counts_path.exists():
            print("Loading binary fingerprints for fast Tanimoto...")
            self.binary_fingerprints = np.load(str(binary_fp_path))
            self.bit_counts = np.load(str(bit_counts_path))
            self.use_stored_fingerprints = True
            print(f"✓ Loaded {len(self.binary_fingerprints):,} binary fingerprints")
        else:
            self.binary_fingerprints = None
            self.bit_counts = None
            self.use_stored_fingerprints = False
            logger.warning("Binary fingerprints not found, using RDKit fallback")

        print("✓ Molecular retriever initialized")
        print(f"  - Fingerprint: Morgan (radius={self.radius}, bits={self.n_bits})")
        print(f"  - Tanimoto threshold: {self.similarity_threshold}")
        print(f"  - FAISS GPU: {self.faiss_gpu}")

    def _calculate_true_tanimoto(self, query_smiles: str, candidate_smiles_list: List[str]) -> List[float]:
        """
        Calculate true Tanimoto similarity for candidate molecules.

        Automatically uses parallel computation when number of candidates exceeds threshold.

        Args:
            query_smiles: Query SMILES string
            candidate_smiles_list: List of candidate SMILES

        Returns:
            List of Tanimoto similarities (0-1 range)
        """
        n_candidates = len(candidate_smiles_list)

        # Use parallel computation for large candidate sets
        if n_candidates >= self.tanimoto_parallel_threshold:
            return self._calculate_true_tanimoto_parallel(query_smiles, candidate_smiles_list)
        else:
            return self._calculate_true_tanimoto_serial(query_smiles, candidate_smiles_list)

    def _calculate_true_tanimoto_serial(self, query_smiles: str, candidate_smiles_list: List[str]) -> List[float]:
        """
        Serial version of Tanimoto calculation (for small candidate sets).

        Args:
            query_smiles: Query SMILES string
            candidate_smiles_list: List of candidate SMILES

        Returns:
            List of Tanimoto similarities (0-1 range)
        """
        query_mol = _mol_from_smiles_compat(query_smiles)
        if query_mol is None:
            return [0.0] * len(candidate_smiles_list)

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, self.radius, nBits=self.n_bits)

        tanimoto_scores = []
        for smiles in candidate_smiles_list:
            mol = _mol_from_smiles_compat(smiles)
            if mol is None:
                tanimoto_scores.append(0.0)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            tanimoto_scores.append(sim)

        return tanimoto_scores

    def _calculate_true_tanimoto_parallel(self, query_smiles: str, candidate_smiles_list: List[str]) -> List[float]:
        """
        Parallel version of Tanimoto calculation (for large candidate sets).

        Uses multiprocessing to speed up computation when many candidates need scoring.

        Args:
            query_smiles: Query SMILES string
            candidate_smiles_list: List of candidate SMILES

        Returns:
            List of Tanimoto similarities (0-1 range)
        """
        query_mol = _mol_from_smiles_compat(query_smiles)
        if query_mol is None:
            return [0.0] * len(candidate_smiles_list)

        # Get query fingerprint
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, self.radius, nBits=self.n_bits)

        # Convert fingerprint to list of bit positions (for pickling)
        query_fp_bits = list(query_fp.GetOnBits())

        # Prepare arguments for parallel computation
        args_list = [
            (query_fp_bits, smiles, self.radius, self.n_bits)
            for smiles in candidate_smiles_list
        ]

        # Use multiprocessing pool
        n_workers = min(cpu_count(), 4)  # Limit to 4 workers to avoid overhead
        with Pool(processes=n_workers) as pool:
            tanimoto_scores = pool.map(_compute_single_tanimoto, args_list)

        return tanimoto_scores

    def _calculate_tanimoto_vectorized(self, query_fp: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
        """
        Calculate exact Tanimoto similarity using stored binary fingerprints.

        Uses Hamming distance on packed binary fingerprints for fast vectorized computation.
        This is mathematically equivalent to RDKit's Tanimoto calculation.

        Args:
            query_fp: Query fingerprint (2048,) float32
            candidate_indices: Candidate molecule indices (N,)

        Returns:
            Tanimoto similarities (N,) float64
        """
        # 1. Convert query fp to packed binary format
        query_binary = (query_fp > 0).astype(np.uint8)
        query_packed = np.packbits(query_binary)  # (256,)
        query_bits = int(np.sum(query_binary))

        # 2. Lookup stored fingerprints and bit counts
        candidate_packed = self.binary_fingerprints[candidate_indices]  # (N, 256)
        candidate_bits = self.bit_counts[candidate_indices]  # (N,)

        # 3. Compute Hamming distance (XOR + popcount)
        xor_result = candidate_packed ^ query_packed  # (N, 256) broadcasting
        hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1, dtype=np.uint16)  # (N,)

        # 4. Convert Hamming to Tanimoto (exact formula)
        # Hamming(A,B) = |A| + |B| - 2*|A∩B|
        # => |A∩B| = (|A| + |B| - Hamming) / 2
        # Tanimoto = |A∩B| / (|A| + |B| - |A∩B|)
        intersection = (query_bits + candidate_bits - hamming_dist) / 2.0
        union = query_bits + candidate_bits - intersection
        tanimoto = np.where(union > 0, intersection / union, 0.0)

        return tanimoto.astype(np.float64)

    def _search(self, query: str, original_smiles: str, num: int = None,
               return_score: bool = False, sort_by: str = None, descending: bool = True,
               similarity_threshold: float = None, original_property_value: float = None,
               filter_better_only: bool = True, min_improvement_margin: float = 0.0,
               objectives: Optional[List[Dict[str, Any]]] = None):
        """
        Search for similar molecules with dual-molecule filtering.

        Uses query SMILES for FAISS search but filters by similarity to original SMILES.

        Args:
            query: Query SMILES string (used for FAISS search)
            original_smiles: Original SMILES string (used for Tanimoto filtering)
            num: Number of results to return
            return_score: Whether to return similarity scores
            sort_by: Property to sort by ("qed", "logp", "sa", etc.), None for similarity
            descending: Sort descending (True) or ascending (False)
            similarity_threshold: Override default similarity threshold
            original_property_value: Original molecule's property value for filtering
            filter_better_only: Only return molecules with better property than original
            objectives: Optional multi-objective filters and ranking specification

        Returns:
            results: List of document dictionaries
            scores: List of similarity_to_original values (if return_score=True)
            similarities_to_original: List of Tanimoto similarities to original molecule
        """
        if num is None:
            num = self.topk

        objective_specs = list(objectives or [])
        objective_names = set()
        for spec in objective_specs:
            required = {
                'name', 'direction', 'weight', 'original_value',
                'query_value', 'success_margin',
            }
            if not isinstance(spec, dict) or set(spec) != required:
                raise ValueError("Each objective must contain exactly the supported fields")
            name = str(spec['name'])
            direction = str(spec['direction'])
            if not name or name in objective_names:
                raise ValueError("Objective names must be non-empty and unique")
            if direction not in {'maximize', 'minimize'}:
                raise ValueError(f"Unsupported objective direction: {direction}")
            numeric_values = (
                float(spec['weight']),
                float(spec['original_value']),
                float(spec['query_value']),
                float(spec['success_margin']),
            )
            if not all(np.isfinite(value) for value in numeric_values):
                raise ValueError("Objective values must be finite")
            if numeric_values[0] <= 0 or numeric_values[3] < 0:
                raise ValueError("Objective weight must be positive and margin non-negative")
            objective_names.add(name)

        # Use provided threshold or default
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        # Step 1: FAISS recall candidate set (using query molecule)
        query_fp = self.fingerprinter.smiles_to_fingerprint(query)
        query_fp_for_search = query_fp.reshape(1, -1).copy()
        faiss.normalize_L2(query_fp_for_search)

        # Also compute original molecule fingerprint for filtering
        original_fp = self.fingerprinter.smiles_to_fingerprint(original_smiles)

        # Request many more candidates before exact Tanimoto filtering.
        search_k = min(num * self.search_multiplier, self.n_molecules)
        _, candidate_idxs = self.index.search(query_fp_for_search, k=search_k)
        candidate_idxs = candidate_idxs[0]

        # Step 2: Get candidate SMILES
        candidate_smiles = [self.smiles_list[idx] for idx in candidate_idxs]

        # Step 3: Calculate Tanimoto similarity to ORIGINAL molecule (for filtering)
        if self.use_stored_fingerprints:
            # Fast path: use stored binary fingerprints
            tanimoto_to_original = self._calculate_tanimoto_vectorized(
                original_fp, candidate_idxs
            ).tolist()
        else:
            # Fallback: regenerate fingerprints from SMILES
            tanimoto_to_original = self._calculate_true_tanimoto(original_smiles, candidate_smiles)

        # Step 4: Filter by Tanimoto threshold to ORIGINAL molecule AND property value
        filtered_data = []
        for idx, smiles, sim_to_original in zip(candidate_idxs, candidate_smiles, tanimoto_to_original):
            if sim_to_original >= threshold:
                props = self.properties[idx]

                # Property filtering: only keep molecules with better property than original (with margin)
                if filter_better_only and original_property_value is not None and sort_by:
                    prop_value = props.get(sort_by, None)
                    if prop_value is not None:
                        # For SA (minimize): prop_value < original_property_value - margin is better
                        # For others (maximize): prop_value > original_property_value + margin is better
                        if sort_by == 'sa':
                            # SA: lower is better - need to be at least margin smaller
                            if prop_value >= original_property_value - min_improvement_margin:
                                continue  # Skip - not enough improvement
                        else:
                            # QED, LogP, JNK3, and DRD2 use higher-is-better filtering.
                            if prop_value <= original_property_value + min_improvement_margin:
                                continue  # Skip - not enough improvement

                objective_score = None
                if objective_specs:
                    objective_score = 0.0
                    objective_valid = True
                    for spec in objective_specs:
                        name = str(spec['name'])
                        value = props.get(name)
                        if value is None or not np.isfinite(float(value)):
                            objective_valid = False
                            break
                        sign = -1.0 if spec['direction'] == 'minimize' else 1.0
                        improvement_from_lead = sign * (
                            float(value) - float(spec['original_value'])
                        )
                        if improvement_from_lead < float(spec['success_margin']):
                            objective_valid = False
                            break
                        objective_score += float(spec['weight']) * sign * (
                            float(value) - float(spec['query_value'])
                        )
                    if not objective_valid or objective_score <= 0.0:
                        continue

                filtered_data.append({
                    'idx': idx,
                    'smiles': smiles,
                    'similarity_to_original': sim_to_original,
                    'properties': props,
                    'objective_score': objective_score,
                })

        if len(filtered_data) == 0:
            logger.warning(f'No molecules found with similarity >= {threshold} and better property')
            if return_score:
                return [], [], []
            else:
                return []

        # Step 5: Sort by property or similarity to original
        if objective_specs:
            filtered_data.sort(
                key=lambda item: item['objective_score'], reverse=True
            )
        elif sort_by and sort_by in ['qed', 'logp', 'sa', 'jnk3', 'drd2', 'molecular_weight']:
            filtered_data.sort(
                key=lambda x: x['properties'].get(sort_by, -float('inf') if descending else float('inf')),
                reverse=descending
            )
        else:
            # Sort by similarity to original (descending)
            filtered_data.sort(key=lambda x: x['similarity_to_original'], reverse=True)

        # Step 6: Take top-k
        top_k_data = filtered_data[:num]

        if len(top_k_data) < num:
            logger.warning(f'Only {len(top_k_data)} molecules found (requested {num})')

        # Step 7: Load documents and build rich results
        top_k_idxs = [item['idx'] for item in top_k_data]
        docs = load_docs(self.corpus, top_k_idxs)

        # Build results with properties and similarity
        results = []
        for doc, item in zip(docs, top_k_data):
            results.append({
                'document': doc,
                'similarity_to_original': item['similarity_to_original'],
                'properties': item['properties']
            })

        return results

    def _batch_search(
        self,
        query_list: List[str],
        original_smiles_list: List[str],
        num: int = None,
        return_score: bool = False,
        sort_by: str = None,
        descending: bool = True,
        similarity_threshold: float = None,
        original_property_value: float = None,
        filter_better_only: bool = True,
        min_improvement_margin: float = 0.0,
        objectives: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Batch search for multiple query SMILES with dual-molecule filtering.

        Args:
            query_list: List of query SMILES strings (used for FAISS search)
            original_smiles_list: List of original SMILES strings (used for filtering)
            num: Number of results per query
            return_score: Whether to return similarity scores
            sort_by: Property to sort by ("qed", "logp", "sa", etc.)
            descending: Sort descending (True) or ascending (False)
            similarity_threshold: Override default similarity threshold
            original_property_value: Original molecule's property value for filtering
            filter_better_only: Only return molecules with better property than original

        Returns:
            results: List of result lists (one per query)
            scores: List of similarity_to_original lists (if return_score=True)
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        if isinstance(original_smiles_list, str):
            original_smiles_list = [original_smiles_list]

        # Validate lengths match
        if len(query_list) != len(original_smiles_list):
            raise ValueError(
                f"queries and original_smiles must have same length: "
                f"{len(query_list)} vs {len(original_smiles_list)}"
            )

        # Handle empty query list
        if not query_list:
            logger.warning("Empty query list provided to batch_search")
            if return_score:
                return [], []
            else:
                return []

        if num is None:
            num = self.topk

        # Use single search for each query
        all_results = []

        for query, original in zip(query_list, original_smiles_list):
            query_results = self._search(
                query=query,
                original_smiles=original,
                num=num,
                sort_by=sort_by,
                descending=descending,
                similarity_threshold=similarity_threshold,
                original_property_value=original_property_value,
                filter_better_only=filter_better_only,
                min_improvement_margin=min_improvement_margin,
                objectives=objectives,
            )
            all_results.append(query_results)

        return all_results


#####################################
# FastAPI server
#####################################

class Config:
    """Configuration for molecular retrieval server"""

    def __init__(
        self,
        retrieval_topk: int = 10,
        index_path: str = "molecular_faiss.index",
        metadata_path: str = "molecular_metadata.pkl",
        corpus_path: str = "molecular_corpus.jsonl",
        similarity_threshold: float = 0.4,
        nprobe: int = 800,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
        faiss_gpu: bool = False,
        search_multiplier: int = 40,
        tanimoto_parallel_threshold: int = 1000  # Use parallel computation when candidates > this threshold
    ):
        # Validate file paths
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not Path(corpus_path).exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        # Validate similarity threshold
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")

        if similarity_threshold < 0.3:
            logger.warning(
                f"Very low similarity threshold {similarity_threshold} may return too many results and slow down search"
            )

        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.corpus_path = corpus_path
        self.similarity_threshold = similarity_threshold
        self.nprobe = nprobe
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.faiss_gpu = faiss_gpu
        self.search_multiplier = search_multiplier
        self.tanimoto_parallel_threshold = tanimoto_parallel_threshold


class ObjectiveSpec(BaseModel):
    """One property in a joint multi-objective retrieval request."""

    name: str
    direction: Literal['maximize', 'minimize']
    weight: float
    original_value: float
    query_value: float
    success_margin: float


class QueryRequest(BaseModel):
    """Request model for retrieval endpoint"""
    queries: List[str]
    original_smiles: List[str]  # Required: original molecules for similarity filtering
    similarity_threshold: Optional[float] = None  # Override default threshold
    original_property_value: Optional[float] = None  # Original molecule's property value for filtering
    filter_better_only: bool = True  # Only return molecules with better property than original
    min_improvement_margin: float = 0.0  # Minimum improvement required over original property value
    topk: Optional[int] = None
    return_scores: bool = False
    sort_by: Optional[str] = None
    descending: bool = True  # True for max (QED↑), False for min (LogP↓)
    objectives: Optional[List[ObjectiveSpec]] = None


app = FastAPI()

# Global variables (initialized in main block)
config = None
retriever = None


@lru_cache(maxsize=4096)
def _cached_batch_search(
    query_list: tuple,
    original_smiles_list: tuple,
    num: int,
    sort_by: Optional[str],
    descending: bool,
    similarity_threshold: Optional[float],
    original_property_value: Optional[float],
    filter_better_only: bool,
    min_improvement_margin: float,
    objective_specs: tuple,
):
    """Cache complete static-retrieval results for identical requests."""
    if retriever is None:
        raise RuntimeError("Server not initialized. Please run as main script.")

    return retriever._batch_search(
        query_list=list(query_list),
        original_smiles_list=list(original_smiles_list),
        num=num,
        sort_by=sort_by,
        descending=descending,
        similarity_threshold=similarity_threshold,
        original_property_value=original_property_value,
        filter_better_only=filter_better_only,
        min_improvement_margin=min_improvement_margin,
        objectives=[
            {
                'name': item[0],
                'direction': item[1],
                'weight': item[2],
                'original_value': item[3],
                'query_value': item[4],
                'success_margin': item[5],
            }
            for item in objective_specs
        ],
    )


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Molecular retrieval endpoint with dual-molecule filtering.

    Uses query SMILES for FAISS search but filters by similarity to original SMILES.

    Input format:
    {
      "queries": ["CCO", "CC(C)O"],           # Query molecules (for FAISS search)
      "original_smiles": ["CCO", "CC(C)O"],   # Original molecules (for filtering)
      "similarity_threshold": 0.4,             # Optional: override default threshold
      "topk": 10,
      "return_scores": true,
      "sort_by": "qed",
      "descending": true  # True for maximize (QED↑), False for minimize (LogP↓)
    }

    Output format:
    {
      "result": [
        [{"document": {...}, "score": 0.85}, ...],  # score is similarity to original
        [{"document": {...}, "score": 0.90}, ...],
      ]
    }

    Note: Results are filtered by similarity to original >= threshold,
          then sorted by specified property or similarity.
    """
    if config is None or retriever is None:
        raise RuntimeError("Server not initialized. Please run as main script.")

    # Validate required fields
    if len(request.queries) != len(request.original_smiles):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"queries and original_smiles must have same length: "
                   f"{len(request.queries)} vs {len(request.original_smiles)}"
        )

    if not request.topk:
        request.topk = config.retrieval_topk

    objective_specs = tuple(sorted(
        (
            item.name,
            item.direction,
            float(item.weight),
            float(item.original_value),
            float(item.query_value),
            float(item.success_margin),
        )
        for item in (request.objectives or [])
    ))

    all_results = _cached_batch_search(
        query_list=tuple(request.queries),
        original_smiles_list=tuple(request.original_smiles),
        num=request.topk,
        sort_by=request.sort_by,
        descending=request.descending,
        similarity_threshold=request.similarity_threshold,
        original_property_value=request.original_property_value,
        filter_better_only=request.filter_better_only,
        min_improvement_margin=request.min_improvement_margin,
        objective_specs=objective_specs,
    )

    # Format response - each result contains document, similarity_to_original, and properties
    # Keep response serialization from mutating the cached value.
    return {"result": copy.deepcopy(all_results)}


@app.get("/cache_stats")
def cache_stats_endpoint():
    """
    Get cache statistics for fingerprint cache.

    Output format:
    {
      "fingerprint_cache": {
        "hits": 12345,
        "misses": 6789,
        "hit_rate": 0.645,
        "cache_size": 1234,
        "max_cache_size": 50000
      }
    }
    """
    if retriever is None:
        raise RuntimeError("Server not initialized. Please run as main script.")

    fp_stats = retriever.fingerprinter.get_cache_stats()

    result_cache = _cached_batch_search.cache_info()
    return {
        "fingerprint_cache": fp_stats,
        "retrieval_result_cache": {
            "hits": result_cache.hits,
            "misses": result_cache.misses,
            "hit_rate": (
                result_cache.hits / (result_cache.hits + result_cache.misses)
                if result_cache.hits + result_cache.misses > 0
                else 0.0
            ),
            "cache_size": result_cache.currsize,
            "max_cache_size": result_cache.maxsize,
        },
    }


@app.post("/clear_cache")
def clear_cache_endpoint():
    """Clear all caches (fingerprint cache)."""
    if retriever is None:
        raise RuntimeError("Server not initialized. Please run as main script.")

    retriever.fingerprinter.clear_cache()
    _cached_batch_search.cache_clear()

    return {"status": "success", "message": "All caches cleared"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch molecular retrieval server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python retrieval/server/static_exemplar_memory.py

  # Custom paths
  python retrieval/server/static_exemplar_memory.py \\
      --index_path path/to/index.index \\
      --metadata_path path/to/metadata.pkl \\
      --corpus_path path/to/corpus.jsonl

  # With GPU acceleration
  python retrieval/server/static_exemplar_memory.py --faiss_gpu

  # Custom port and threshold
  python retrieval/server/static_exemplar_memory.py --port 8001 --threshold 0.5
        """
    )

    parser.add_argument(
        "--index_path",
        type=str,
        default="molecular_faiss.index",
        help="Path to FAISS index file"
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        default="molecular_metadata.pkl",
        help="Path to metadata pickle file"
    )

    parser.add_argument(
        "--corpus_path",
        type=str,
        default="molecular_corpus.jsonl",
        help="Path to corpus JSON Lines file"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Default number of results to return"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Tanimoto similarity threshold (0-1)"
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=800,
        help="Number of clusters to search for IndexIVFFlat (default: 800)"
    )

    parser.add_argument(
        "--faiss_gpu",
        action="store_true",
        help="Use GPU acceleration for FAISS"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )

    args = parser.parse_args()

    # Build config (file paths are validated in Config.__init__)
    config = Config(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        similarity_threshold=args.threshold,
        nprobe=args.nprobe,
        morgan_radius=2,
        morgan_nbits=2048,
        faiss_gpu=args.faiss_gpu
    )

    # Initialize retriever (loaded once, reused for all requests)
    print("=" * 80)
    print("Initializing Static Exemplar Memory Server")
    print("=" * 80)
    print("✓ Query fingerprint LRU cache: 50,000 entries (~50MB)")
    print("✓ Corpus loading: Memory dict for fast O(1) access")
    if args.faiss_gpu:
        print("✓ GPU acceleration: ENABLED")
    retriever = StaticExemplarMemory(config)
    print("=" * 80)
    print(f"Server ready at http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  POST http://{args.host}:{args.port}/retrieve - Molecular search")
    print(f"  GET  http://{args.host}:{args.port}/cache_stats - Cache statistics")
    print(f"  POST http://{args.host}:{args.port}/clear_cache - Clear caches")
    print("=" * 80)

    # Launch server
    uvicorn.run(app, host=args.host, port=args.port)
