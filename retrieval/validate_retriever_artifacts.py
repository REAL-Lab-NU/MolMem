#!/usr/bin/env python3
"""Validate alignment and search behavior of MolMem retrieval artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

try:
    from retrieval.build_retriever_index import (
        BINARY_NAME,
        BIT_COUNT_NAME,
        INDEX_NAME,
        MANIFEST_NAME,
        PAPER_N_BITS,
        PAPER_NLIST,
        PAPER_RADIUS,
        _validate_index_ids,
        fingerprint_smiles_batch,
        log,
        sha256_file,
        validate_corpus_alignment,
    )
except ModuleNotFoundError:  # Support direct execution as ``python retrieval/...py``.
    from build_retriever_index import (  # type: ignore[no-redef]
        BINARY_NAME,
        BIT_COUNT_NAME,
        INDEX_NAME,
        MANIFEST_NAME,
        PAPER_N_BITS,
        PAPER_NLIST,
        PAPER_RADIUS,
        _validate_index_ids,
        fingerprint_smiles_batch,
        log,
        sha256_file,
        validate_corpus_alignment,
    )


def load_metadata(metadata_path: Path, limit: int | None = None) -> tuple[list[str], int]:
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    required = {"smiles_list", "properties", "n_bits", "radius", "n_molecules"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Metadata is missing required keys: {sorted(missing)}")

    smiles = metadata["smiles_list"]
    properties = metadata["properties"]
    n_molecules = int(metadata["n_molecules"])
    if len(smiles) != n_molecules or len(properties) != n_molecules:
        raise ValueError("Metadata molecule, SMILES, and property counts do not match")
    if int(metadata["n_bits"]) != PAPER_N_BITS or int(metadata["radius"]) != PAPER_RADIUS:
        raise ValueError("Metadata does not use MolMem's ECFP4 radius=2, 2048-bit settings")
    if limit is not None:
        if not 0 < limit <= n_molecules:
            raise ValueError(f"Invalid validation limit {limit} for {n_molecules} molecules")
        return smiles[:limit], limit
    return smiles, n_molecules


def validate(
    data_dir: Path,
    corpus_path: Path,
    metadata_path: Path,
    expected_nlist: int,
    samples: int,
    seed: int,
    check_corpus: bool,
    check_ids: bool,
    check_hashes: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    index_path = data_dir / INDEX_NAME
    binary_path = data_dir / BINARY_NAME
    counts_path = data_dir / BIT_COUNT_NAME
    manifest_path = data_dir / MANIFEST_NAME
    required_paths = [index_path, binary_path, counts_path, metadata_path, manifest_path]
    if check_corpus:
        required_paths.append(corpus_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if check_hashes:
        log("Checking artifact sizes and SHA256 values from the build manifest")
        artifact_paths = {
            "index": index_path,
            "binary": binary_path,
            "counts": counts_path,
        }
        for name, path in artifact_paths.items():
            record = manifest.get("artifacts", {}).get(name, {})
            if int(record.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"Manifest size mismatch for {path.name}")
            if record.get("sha256") != sha256_file(path):
                raise ValueError(f"Manifest SHA256 mismatch for {path.name}")
        input_paths = {"metadata": metadata_path}
        if check_corpus:
            input_paths["corpus"] = corpus_path
        for name, path in input_paths.items():
            record = manifest.get("inputs", {}).get(name, {})
            if int(record.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"Manifest input size mismatch for {path.name}")
            if record.get("sha256") != sha256_file(path):
                raise ValueError(f"Manifest input SHA256 mismatch for {path.name}")

    log(f"Loading metadata: {metadata_path}")
    smiles, n_molecules = load_metadata(metadata_path, limit=limit)
    if int(manifest.get("n_molecules", -1)) != n_molecules:
        raise ValueError(
            f"Manifest n_molecules={manifest.get('n_molecules')} does not match {n_molecules}"
        )
    if int(manifest.get("faiss", {}).get("nlist", -1)) != expected_nlist:
        raise ValueError("Manifest nlist does not match --expected-nlist")
    if check_corpus:
        validate_corpus_alignment(
            corpus_path,
            smiles,
            require_exact_count=limit is None,
        )

    log(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))
    failures: list[str] = []
    if not isinstance(index, faiss.IndexIVFFlat):
        raise ValueError(f"index type is {type(index).__name__}, expected IndexIVFFlat")
    if not index.is_trained:
        failures.append("index is not trained")
    if index.ntotal != n_molecules:
        failures.append(f"index.ntotal={index.ntotal}, metadata={n_molecules}")
    if index.d != PAPER_N_BITS:
        failures.append(f"index.d={index.d}, expected {PAPER_N_BITS}")
    if index.nlist != expected_nlist:
        failures.append(f"index.nlist={index.nlist}, expected {expected_nlist}")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        failures.append(
            f"index.metric_type={index.metric_type}, expected METRIC_INNER_PRODUCT"
        )
    if failures:
        raise ValueError("; ".join(failures))

    binary = np.load(binary_path, mmap_mode="r")
    counts = np.load(counts_path, mmap_mode="r")
    expected_binary_shape = (n_molecules, PAPER_N_BITS // 8)
    if binary.shape != expected_binary_shape or binary.dtype != np.uint8:
        raise ValueError(
            f"Binary fingerprint array is {binary.shape}/{binary.dtype}; "
            f"expected {expected_binary_shape}/uint8"
        )
    if counts.shape != (n_molecules,) or counts.dtype != np.uint16:
        raise ValueError(
            f"Bit-count array is {counts.shape}/{counts.dtype}; "
            f"expected {(n_molecules,)}/uint16"
        )

    if check_ids:
        log("Checking every ID stored in the FAISS inverted lists")
        _validate_index_ids(index, n_molecules)

    rng = np.random.default_rng(seed)
    sample_count = min(samples, n_molecules)
    rows = rng.choice(n_molecules, size=sample_count, replace=False)
    sample_smiles = [smiles[int(row)] for row in rows]
    packed, bit_counts, invalid, fallback = fingerprint_smiles_batch(sample_smiles)
    if invalid:
        raise ValueError(f"Sample contains invalid SMILES: {invalid[:5]}")
    if not np.array_equal(packed, np.asarray(binary[rows])):
        raise ValueError("Stored binary fingerprints disagree with regenerated RDKit fingerprints")
    if not np.array_equal(bit_counts, np.asarray(counts[rows])):
        raise ValueError("Stored bit counts disagree with regenerated RDKit fingerprints")

    manifest_fallback_rows = manifest.get("fingerprint", {}).get(
        "kekulization_fallback_rows", []
    )
    for item in manifest_fallback_rows:
        metadata_row = int(item[0])
        if metadata_row >= n_molecules:
            continue
        expected_packed, expected_counts, invalid_rows, fallback_rows = (
            fingerprint_smiles_batch([smiles[metadata_row]])
        )
        if invalid_rows or not fallback_rows:
            raise ValueError(f"Manifest fallback row {metadata_row} is not reproducible")
        if not np.array_equal(expected_packed[0], binary[metadata_row]):
            raise ValueError(f"Stored fallback fingerprint mismatch at row {metadata_row}")
        if int(expected_counts[0]) != int(counts[metadata_row]):
            raise ValueError(f"Stored fallback bit count mismatch at row {metadata_row}")

    queries = np.unpackbits(packed, axis=1, count=PAPER_N_BITS).astype(np.float32)
    faiss.normalize_L2(queries)
    index.nprobe = min(800, index.nlist)
    distances, neighbors = index.search(queries, min(10, n_molecules))
    if np.any(neighbors[:, 0] < 0):
        raise ValueError("Self-search returned missing nearest neighbors")
    if np.any(distances[:, 0] < 0.9999):
        worst = float(np.min(distances[:, 0]))
        raise ValueError(f"Self-search similarity is too low; worst top score={worst:.8f}")
    for query_row, candidate_id in enumerate(neighbors[:, 0]):
        if not np.array_equal(packed[query_row], binary[int(candidate_id)]):
            raise ValueError(
                "Top self-search result does not have the query fingerprint: "
                f"query metadata row={int(rows[query_row])}, result={int(candidate_id)}"
            )

    result = {
        "status": "ok",
        "n_molecules": n_molecules,
        "index_type": type(index).__name__,
        "dimension": index.d,
        "nlist": index.nlist,
        "metric": "METRIC_INNER_PRODUCT",
        "nprobe_for_self_test": index.nprobe,
        "self_test_samples": sample_count,
        "minimum_top_similarity": float(np.min(distances[:, 0])),
        "sample_kekulization_fallback_rows": [
            int(rows[local_row]) for local_row, _ in fallback
        ],
        "corpus_alignment_checked": check_corpus,
        "all_index_ids_checked": check_ids,
        "manifest_hashes_checked": check_hashes,
        "manifest_kekulization_fallback_rows": [
            int(item[0]) for item in manifest_fallback_rows
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--expected-nlist", type=int, default=PAPER_NLIST)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, help="Validate a smoke index built from the first N rows")
    parser.add_argument("--skip-corpus-check", action="store_true")
    parser.add_argument("--skip-id-check", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_nlist <= 0 or args.samples <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--expected-nlist, --samples, and --limit (when set) must be positive")
    args.data_dir = args.data_dir.resolve()
    args.metadata = (args.metadata or args.data_dir / "molecular_metadata.pkl").resolve()
    args.corpus = (args.corpus or args.data_dir / "molecular_corpus.jsonl").resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate(
        data_dir=args.data_dir,
        corpus_path=args.corpus,
        metadata_path=args.metadata,
        expected_nlist=args.expected_nlist,
        samples=args.samples,
        seed=args.seed,
        check_corpus=not args.skip_corpus_check,
        check_ids=not args.skip_id_check,
        check_hashes=not args.skip_hash_check,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
