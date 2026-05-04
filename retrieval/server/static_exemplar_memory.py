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
import pickle
import json
import logging
import warnings
from typing import List, Dict, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial, lru_cache

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

        mol = Chem.MolFromSmiles(smiles)
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

    def batch_smiles_to_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """Convert batch of SMILES to fingerprints"""
        fingerprints = []
        for smiles in smiles_list:
            fp = self.smiles_to_fingerprint(smiles)
            fingerprints.append(fp)
        return np.vstack(fingerprints)


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

    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        return 0.0

    # Reconstruct query fingerprint from bit positions
    query_fp = DataStructs.ExplicitBitVect(n_bits)
    for bit in query_fp_bits:
        query_fp.SetBit(bit)

    # Generate candidate fingerprint
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    return DataStructs.TanimotoSimilarity(query_fp, candidate_fp)


class BaseRetriever:
    """Base retriever class for Search-R1 compatibility"""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool, **kwargs):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool, **kwargs):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False, **kwargs):
        return self._search(query, num, return_score, **kwargs)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, **kwargs):
        return self._batch_search(query_list, num, return_score, **kwargs)


class StaticExemplarMemory(BaseRetriever):
    """
    Static Exemplar Memory for cold-start grounding in MolMem.

    This memory component retrieves similar molecules from a large pre-indexed
    chemical database to provide initial guidance when optimizing a new lead molecule.

    Features:
    - Fast similarity search via FAISS + Morgan fingerprints
    - Tanimoto similarity filtering (threshold ≥ 0.4)
    - Property-aware retrieval (QED, LogP, SA, JNK3, DRD2, GSK3B)
    - Compatible with Search-R1 API

    Query format: SMILES strings (e.g., "CCO", "CC(C)O")
    """

    def __init__(self, config):
        super().__init__(config)

        # Configuration
        self.metadata_path = config.metadata_path
        self.similarity_threshold = config.similarity_threshold
        self.faiss_gpu = config.faiss_gpu
        self.radius = config.morgan_radius
        self.n_bits = config.morgan_nbits
        self.faiss_threshold = config.faiss_threshold  # Optional: for fast filtering
        # Performance tuning
        self.search_multiplier_with_filter = config.search_multiplier_with_filter
        self.search_multiplier_no_filter = config.search_multiplier_no_filter
        self.tanimoto_parallel_threshold = config.tanimoto_parallel_threshold

        # Load FAISS index
        print(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        # Set nprobe for IndexIVFFlat (if applicable)
        if hasattr(self.index, 'nprobe') and hasattr(config, 'nprobe') and config.nprobe is not None:
            self.index.nprobe = config.nprobe
            print(f"✓ Set nprobe={config.nprobe} for IndexIVFFlat")

        if self.faiss_gpu and faiss.get_num_gpus() > 0:
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
            print(f"Loading binary fingerprints for fast Tanimoto...")
            self.binary_fingerprints = np.load(str(binary_fp_path))
            self.bit_counts = np.load(str(bit_counts_path))
            self.use_stored_fingerprints = True
            print(f"✓ Loaded {len(self.binary_fingerprints):,} binary fingerprints")
        else:
            self.binary_fingerprints = None
            self.bit_counts = None
            self.use_stored_fingerprints = False
            logger.warning("Binary fingerprints not found, using RDKit fallback")

        print(f"✓ Molecular retriever initialized")
        print(f"  - Fingerprint: Morgan (radius={self.radius}, bits={self.n_bits})")
        print(f"  - Tanimoto threshold: {self.similarity_threshold}")
        if self.faiss_threshold:
            print(f"  - FAISS threshold: {self.faiss_threshold:.4f} (auto-calculated, ~3-5x speedup)")
        else:
            print(f"  - FAISS threshold: None (disabled, use --enable-fast-filter for speedup)")
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
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return [0.0] * len(candidate_smiles_list)

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, self.radius, nBits=self.n_bits)

        tanimoto_scores = []
        for smiles in candidate_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
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
        query_mol = Chem.MolFromSmiles(query_smiles)
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
               filter_better_only: bool = True, min_improvement_margin: float = 0.0):
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

        Returns:
            results: List of document dictionaries
            scores: List of similarity_to_original values (if return_score=True)
            similarities_to_original: List of Tanimoto similarities to original molecule
        """
        if num is None:
            num = self.topk

        # Use provided threshold or default
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        # Step 1: FAISS recall candidate set (using query molecule)
        query_fp = self.fingerprinter.smiles_to_fingerprint(query)
        query_fp_for_search = query_fp.reshape(1, -1).copy()
        faiss.normalize_L2(query_fp_for_search)

        # Also compute original molecule fingerprint for filtering
        original_fp = self.fingerprinter.smiles_to_fingerprint(original_smiles)

        # Request many more candidates for filtering
        if self.faiss_threshold is not None:
            search_k = min(num * self.search_multiplier_with_filter, self.n_molecules)
        else:
            search_k = min(num * self.search_multiplier_no_filter, self.n_molecules)

        faiss_scores, candidate_idxs = self.index.search(query_fp_for_search, k=search_k)
        faiss_scores = faiss_scores[0]
        candidate_idxs = candidate_idxs[0]

        # Step 2: Fast filtering with FAISS threshold (if available)
        if self.faiss_threshold is not None:
            mask = faiss_scores >= self.faiss_threshold
            candidate_idxs = candidate_idxs[mask]
            faiss_scores = faiss_scores[mask]

            if len(candidate_idxs) == 0:
                logger.warning(f'No molecules found with FAISS score >= {self.faiss_threshold}')
                if return_score:
                    return [], [], []
                else:
                    return []

        # Step 3: Get candidate SMILES
        candidate_smiles = [self.smiles_list[idx] for idx in candidate_idxs]

        # Step 4: Calculate Tanimoto similarity to ORIGINAL molecule (for filtering)
        if self.use_stored_fingerprints:
            # Fast path: use stored binary fingerprints
            tanimoto_to_original = self._calculate_tanimoto_vectorized(
                original_fp, candidate_idxs
            ).tolist()
        else:
            # Fallback: regenerate fingerprints from SMILES
            tanimoto_to_original = self._calculate_true_tanimoto(original_smiles, candidate_smiles)

        # Step 5: Filter by Tanimoto threshold to ORIGINAL molecule AND property value
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
                            # QED, LogP, JNK3, DRD2, GSK3B: higher is better - need to be at least margin bigger
                            if prop_value <= original_property_value + min_improvement_margin:
                                continue  # Skip - not enough improvement

                filtered_data.append({
                    'idx': idx,
                    'smiles': smiles,
                    'similarity_to_original': sim_to_original,
                    'properties': props
                })

        if len(filtered_data) == 0:
            logger.warning(f'No molecules found with similarity >= {threshold} and better property')
            if return_score:
                return [], [], []
            else:
                return []

        # Step 6: Sort by property or similarity to original
        if sort_by and sort_by in ['qed', 'logp', 'sa', 'jnk3', 'drd2', 'gsk3b', 'molecular_weight']:
            filtered_data.sort(
                key=lambda x: x['properties'].get(sort_by, -float('inf') if descending else float('inf')),
                reverse=descending
            )
        else:
            # Sort by similarity to original (descending)
            filtered_data.sort(key=lambda x: x['similarity_to_original'], reverse=True)

        # Step 7: Take top-k
        top_k_data = filtered_data[:num]

        if len(top_k_data) < num:
            logger.warning(f'Only {len(top_k_data)} molecules found (requested {num})')

        # Step 8: Load documents and build rich results
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
        min_improvement_margin: float = 0.0
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
                min_improvement_margin=min_improvement_margin
            )
            all_results.append(query_results)

        return all_results


#####################################
# FastAPI server
#####################################

@lru_cache(maxsize=8)  # Cache up to 8 different calibration files
def load_calibration_thresholds(calibration_file: str = "faiss_tanimoto_calibration_piecewise.json") -> Optional[dict]:
    """
    Load FAISS threshold calibration from file with caching.

    Results are cached to avoid re-reading the same file multiple times.
    Cache size is limited to 8 entries to support multiple calibration files.

    Args:
        calibration_file: Path to calibration JSON file

    Returns:
        Dictionary mapping Tanimoto thresholds to FAISS thresholds, or None if loading fails
    """
    if not Path(calibration_file).exists():
        logger.warning(f"Calibration file not found: {calibration_file}")
        return None

    try:
        with open(calibration_file, 'r') as f:
            calibration = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse calibration file {calibration_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading calibration file {calibration_file}: {e}")
        return None

    # Extract threshold mapping
    thresholds = calibration.get('thresholds', {})
    if not thresholds:
        logger.warning(f"No thresholds found in calibration file: {calibration_file}")
        return None

    logger.info(f"Loaded calibration thresholds for {len(thresholds)} Tanimoto values: {list(thresholds.keys())}")
    return thresholds


def get_faiss_threshold_for_tanimoto(tanimoto_threshold: float, calibration_file: str) -> Optional[float]:
    """
    Auto-calculate FAISS threshold based on Tanimoto threshold using calibration data.

    Supports both exact matches and linear interpolation for intermediate values.
    For example, if calibration has 0.4 and 0.5, requesting 0.45 will interpolate.

    Args:
        tanimoto_threshold: Target Tanimoto threshold (0-1 range)
        calibration_file: Path to calibration JSON file

    Returns:
        Corresponding FAISS threshold, or None if calibration not available

    Raises:
        ValueError: If tanimoto_threshold is outside valid range [0, 1]
    """
    # Validate input
    if not 0.0 <= tanimoto_threshold <= 1.0:
        raise ValueError(f"tanimoto_threshold must be in [0, 1], got {tanimoto_threshold}")

    thresholds = load_calibration_thresholds(calibration_file)
    if not thresholds:
        return None

    # Look up threshold (convert to string key for exact match)
    threshold_key = str(tanimoto_threshold)
    if threshold_key in thresholds:
        # Exact match: use piecewise model result (most accurate)
        faiss_threshold = thresholds[threshold_key].get('faiss_threshold_piecewise')
        logger.info(f"Exact match: Tanimoto {tanimoto_threshold} → FAISS {faiss_threshold:.4f}")
        return faiss_threshold

    # No exact match: try linear interpolation
    available_thresholds = sorted([float(k) for k in thresholds.keys()])

    if tanimoto_threshold < min(available_thresholds):
        # Below calibrated range: use nearest (lowest)
        nearest = min(available_thresholds)
        faiss_threshold = thresholds[str(nearest)].get('faiss_threshold_piecewise')
        logger.warning(
            f"Tanimoto threshold {tanimoto_threshold} is below calibrated range "
            f"[{min(available_thresholds)}, {max(available_thresholds)}]. "
            f"Using nearest threshold {nearest} → FAISS {faiss_threshold:.4f}"
        )
        return faiss_threshold

    if tanimoto_threshold > max(available_thresholds):
        # Above calibrated range: use nearest (highest)
        nearest = max(available_thresholds)
        faiss_threshold = thresholds[str(nearest)].get('faiss_threshold_piecewise')
        logger.warning(
            f"Tanimoto threshold {tanimoto_threshold} is above calibrated range "
            f"[{min(available_thresholds)}, {max(available_thresholds)}]. "
            f"Using nearest threshold {nearest} → FAISS {faiss_threshold:.4f}"
        )
        return faiss_threshold

    # Within calibrated range: linear interpolation
    for i in range(len(available_thresholds) - 1):
        lower = available_thresholds[i]
        upper = available_thresholds[i + 1]

        if lower <= tanimoto_threshold <= upper:
            lower_faiss = thresholds[str(lower)].get('faiss_threshold_piecewise')
            upper_faiss = thresholds[str(upper)].get('faiss_threshold_piecewise')

            # Linear interpolation
            weight = (tanimoto_threshold - lower) / (upper - lower)
            interpolated = lower_faiss + weight * (upper_faiss - lower_faiss)

            logger.info(
                f"Interpolated: Tanimoto {tanimoto_threshold} "
                f"(between {lower} and {upper}) → FAISS {interpolated:.4f}"
            )
            return interpolated

    # Should not reach here
    logger.error(f"Failed to find threshold for {tanimoto_threshold}")
    return None


class Config:
    """Configuration for molecular retrieval server"""

    def __init__(
        self,
        retrieval_method: str = "molecular",
        retrieval_topk: int = 10,
        index_path: str = "molecular_faiss.index",
        metadata_path: str = "molecular_metadata.pkl",
        corpus_path: str = "molecular_corpus.jsonl",
        similarity_threshold: float = 0.4,
        enable_fast_filter: bool = False,  # Enable FAISS threshold pre-filtering
        calibration_file: str = "faiss_tanimoto_calibration_piecewise.json",
        nprobe: int = 500,  # Optimized: Binary IVF nlist=16730, 73.5% success rate, 9.82ms latency
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
        faiss_gpu: bool = False,
        # Performance tuning parameters
        search_multiplier_with_filter: int = 500,  # Search k multiplier when fast filter is enabled
        search_multiplier_no_filter: int = 40,  # Search k multiplier when fast filter is disabled (max 40 for GPU 2048 limit)
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

        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.corpus_path = corpus_path
        self.similarity_threshold = similarity_threshold
        self.enable_fast_filter = enable_fast_filter
        self.calibration_file = calibration_file
        self.nprobe = nprobe
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.faiss_gpu = faiss_gpu
        # Performance tuning
        self.search_multiplier_with_filter = search_multiplier_with_filter
        self.search_multiplier_no_filter = search_multiplier_no_filter
        self.tanimoto_parallel_threshold = tanimoto_parallel_threshold

        # Auto-calculate FAISS threshold if fast filter is enabled
        if self.enable_fast_filter:
            self.faiss_threshold = get_faiss_threshold_for_tanimoto(
                similarity_threshold, calibration_file
            )
            if self.faiss_threshold is None:
                # Fast filter was explicitly requested but calibration unavailable
                # Raise error instead of silently disabling
                raise FileNotFoundError(
                    f"Fast filter enabled but calibration unavailable.\n"
                    f"Options:\n"
                    f"  1. Disable fast filter: remove --enable-fast-filter flag\n"
                    f"  2. Generate calibration: python calibrate_faiss_tanimoto_piecewise.py --load-pairs calibration_pairs_100k.pkl\n"
                    f"  3. Place calibration file at: {calibration_file}"
                )
        else:
            self.faiss_threshold = None


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
    sort_by: Optional[str] = None  # "qed", "logp", "sa", "jnk3", "drd2", "gsk3b"
    descending: bool = True  # True for max (QED↑), False for min (LogP↓)


app = FastAPI()

# Global variables (initialized in main block)
config = None
retriever = None


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
      "sort_by": "qed",  # Optional: "qed", "logp", "sa", "jnk3", "drd2", "gsk3b"
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

    # Perform batch retrieval with dual-molecule filtering
    all_results = retriever._batch_search(
        query_list=request.queries,
        original_smiles_list=request.original_smiles,
        num=request.topk,
        sort_by=request.sort_by,
        descending=request.descending,
        similarity_threshold=request.similarity_threshold,
        original_property_value=request.original_property_value,
        filter_better_only=request.filter_better_only,
        min_improvement_margin=request.min_improvement_margin
    )

    # Format response - each result contains document, similarity_to_original, and properties
    return {"result": all_results}


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

    return {
        "fingerprint_cache": fp_stats
    }


@app.post("/clear_cache")
def clear_cache_endpoint():
    """Clear all caches (fingerprint cache)."""
    if retriever is None:
        raise RuntimeError("Server not initialized. Please run as main script.")

    retriever.fingerprinter.clear_cache()

    return {"status": "success", "message": "All caches cleared"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch molecular retrieval server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python molecular_retrieval_server.py

  # Custom paths
  python molecular_retrieval_server.py \\
      --index_path path/to/index.index \\
      --metadata_path path/to/metadata.pkl \\
      --corpus_path path/to/corpus.jsonl

  # With GPU acceleration
  python molecular_retrieval_server.py --faiss_gpu

  # Custom port and threshold
  python molecular_retrieval_server.py --port 8001 --threshold 0.5
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
        "--enable-fast-filter",
        action="store_true",
        help="Enable FAISS threshold pre-filtering for 3-5x speedup (requires calibration file)"
    )

    parser.add_argument(
        "--calibration-file",
        type=str,
        default="faiss_tanimoto_calibration_piecewise.json",
        help="Calibration file for auto FAISS threshold calculation"
    )

    parser.add_argument(
        "--nprobe",
        type=int,
        default=3,
        help="Number of clusters to search for IndexIVFFlat (default: 3, optimal from testing)"
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
        retrieval_method="molecular",
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        similarity_threshold=args.threshold,
        enable_fast_filter=args.enable_fast_filter,
        calibration_file=args.calibration_file,
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
