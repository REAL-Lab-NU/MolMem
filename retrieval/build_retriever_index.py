#!/usr/bin/env python3
"""Build the paper-version MolMem static exemplar index from metadata.

The row order in ``molecular_metadata.pkl`` is the canonical document ID order.
For every SMILES this script generates an ECFP4 bit vector, stores its packed
binary representation for exact Tanimoto scoring, L2-normalizes the float32
vector, and adds it to an inner-product FAISS IndexIVFFlat.

The paper configuration is the default: radius=2, 2048 bits, nlist=1689.
Large intermediate arrays are memory-mapped so the full build does not require
holding roughly 22 GiB of fingerprints in RAM. Fingerprint generation can be
resumed with ``--resume`` after interruption.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem


PAPER_NLIST = 1689
PAPER_N_BITS = 2048
PAPER_RADIUS = 2
DEFAULT_SEED = 1234

INDEX_NAME = "molecular_faiss.index"
BINARY_NAME = "molecular_binary_fingerprints.npy"
BIT_COUNT_NAME = "molecular_bit_counts.npy"
MANIFEST_NAME = "molecular_faiss.manifest.json"

_WORK_FLOAT_NAME = ".molecular_fingerprints.float32.npy.partial"
_WORK_BINARY_NAME = f".{BINARY_NAME}.partial"
_WORK_COUNTS_NAME = f".{BIT_COUNT_NAME}.partial"
_STATE_NAME = ".molecular_faiss.build-state.json"
_INDEX_PARTIAL_NAME = f".{INDEX_NAME}.partial"

_WORKER_SMILES: Sequence[str] | None = None
_WORKER_RADIUS = PAPER_RADIUS
_WORKER_N_BITS = PAPER_N_BITS


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def publish_hardlinked_artifact(partial: Path, final: Path) -> None:
    """Atomically publish ``partial`` while retaining it for crash recovery."""
    staging = final.with_name(f".{final.name}.publish-{os.getpid()}")
    if staging.exists():
        staging.unlink()
    os.link(partial, staging)
    os.replace(staging, final)


def check_disk_space(output_dir: Path, n_vectors: int, n_bits: int, nlist: int) -> None:
    """Fail before computation if partial and index artifacts cannot fit."""
    expected_sizes = {
        output_dir / _WORK_FLOAT_NAME: n_vectors * n_bits * np.dtype(np.float32).itemsize + 256,
        output_dir / _WORK_BINARY_NAME: n_vectors * (n_bits // 8) + 256,
        output_dir / _WORK_COUNTS_NAME: n_vectors * np.dtype(np.uint16).itemsize + 256,
        output_dir / _INDEX_PARTIAL_NAME: int(
            (n_vectors * (n_bits * np.dtype(np.float32).itemsize + 8) + nlist * n_bits * 4)
            * 1.02
        ),
    }
    remaining = 0
    for path, expected in expected_sizes.items():
        allocated = path.stat().st_blocks * 512 if path.exists() else 0
        remaining += max(expected - allocated, 0)
    reserve = 2 * 1024**3
    free = shutil.disk_usage(output_dir).free
    required = remaining + reserve
    if free < required:
        raise OSError(
            f"Insufficient disk space in {output_dir}: free={free / 1024**3:.1f} GiB, "
            f"required≈{required / 1024**3:.1f} GiB including a 2 GiB reserve"
        )
    log(
        f"Disk preflight passed: free={free / 1024**3:.1f} GiB, "
        f"remaining build allocation≈{remaining / 1024**3:.1f} GiB"
    )


def load_metadata_spec(
    metadata_path: Path,
    limit: int | None = None,
) -> tuple[list[str], int, int, int]:
    log(f"Loading metadata: {metadata_path}")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    required = {"smiles_list", "properties", "n_bits", "radius", "n_molecules"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Metadata is missing required keys: {sorted(missing)}")

    smiles = metadata["smiles_list"]
    properties = metadata["properties"]
    n_molecules = int(metadata["n_molecules"])
    n_bits = int(metadata["n_bits"])
    radius = int(metadata["radius"])

    if not isinstance(smiles, list):
        raise TypeError("metadata['smiles_list'] must be a list")
    if len(smiles) != n_molecules or len(properties) != n_molecules:
        raise ValueError(
            "Metadata lengths disagree: "
            f"n_molecules={n_molecules}, smiles={len(smiles)}, properties={len(properties)}"
        )
    if n_bits != PAPER_N_BITS or radius != PAPER_RADIUS:
        raise ValueError(
            f"Expected paper ECFP4 settings radius={PAPER_RADIUS}, n_bits={PAPER_N_BITS}; "
            f"metadata contains radius={radius}, n_bits={n_bits}"
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        if limit > n_molecules:
            raise ValueError(f"--limit={limit} exceeds metadata size {n_molecules}")
        smiles = smiles[:limit]

    # Drop the large properties list as soon as its alignment has been checked.
    del properties, metadata
    return smiles, n_bits, radius, n_molecules


def _corpus_smiles(record: dict[str, Any]) -> str:
    contents = record.get("contents")
    if not isinstance(contents, str):
        raise ValueError("Corpus record has no string 'contents' field")
    first_line = contents.split("\n", 1)[0]
    prefix = "SMILES: "
    if not first_line.startswith(prefix):
        raise ValueError("Corpus record does not start with 'SMILES: '")
    return first_line[len(prefix) :]


def validate_corpus_alignment(
    corpus_path: Path,
    smiles: Sequence[str],
    require_exact_count: bool,
) -> int:
    """Verify that corpus document IDs and SMILES match metadata row-for-row."""
    expected = len(smiles)
    log(f"Checking corpus/metadata alignment for {expected:,} rows: {corpus_path}")
    checked = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for expected_id, expected_smiles in enumerate(smiles):
            line = handle.readline()
            if not line:
                raise ValueError(
                    f"Corpus ended at row {expected_id:,}; expected at least {expected:,} rows"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid corpus JSON at row {expected_id:,}: {exc}") from exc

            try:
                document_id = int(record.get("id"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid corpus id at row {expected_id:,}") from exc
            if document_id != expected_id:
                raise ValueError(
                    f"Corpus id mismatch at row {expected_id:,}: found {document_id}"
                )

            actual_smiles = _corpus_smiles(record)
            if actual_smiles != expected_smiles:
                raise ValueError(
                    f"Corpus/metadata SMILES mismatch at row {expected_id:,}: "
                    f"metadata={expected_smiles!r}, corpus={actual_smiles!r}"
                )
            checked += 1
            if checked % 500_000 == 0:
                log(f"Corpus alignment: {checked:,}/{expected:,}")

        if require_exact_count and handle.readline():
            raise ValueError(f"Corpus contains more than the expected {expected:,} rows")

    log(f"Corpus alignment verified: {checked:,} rows")
    return checked


def molecule_from_smiles_compat(smiles: str) -> tuple[Chem.Mol | None, bool]:
    """Parse SMILES, tolerating ChEMBL records that fail only kekulization.

    ChEMBL row 1,833,339 in the release metadata is accepted when all RDKit
    sanitization steps except kekulization are applied. Keeping aromatic bond
    flags is sufficient and deterministic for Morgan fingerprint generation.
    Truly invalid records still return ``None`` and stop the build.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        return molecule, False

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        return None, False
    sanitize_without_kekulization = (
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    )
    result = Chem.SanitizeMol(
        molecule,
        sanitizeOps=sanitize_without_kekulization,
        catchErrors=True,
    )
    if result != Chem.SanitizeFlags.SANITIZE_NONE:
        return None, False
    return molecule, True


def fingerprint_smiles_batch(
    smiles: Sequence[str],
    radius: int = PAPER_RADIUS,
    n_bits: int = PAPER_N_BITS,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    """Return packed bits, bit counts, invalid rows, and fallback-parsed rows."""
    bits = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    invalid: list[tuple[int, str]] = []
    fallback: list[tuple[int, str]] = []
    for row, value in enumerate(smiles):
        molecule, used_fallback = molecule_from_smiles_compat(value)
        if molecule is None:
            invalid.append((row, value))
            continue
        if used_fallback:
            fallback.append((row, value))
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius,
            nBits=n_bits,
        )
        DataStructs.ConvertToNumpyArray(fingerprint, bits[row])

    counts = bits.sum(axis=1, dtype=np.uint16)
    packed = np.packbits(bits, axis=1)
    return packed, counts, invalid, fallback


def _init_worker(smiles: Sequence[str], radius: int, n_bits: int) -> None:
    global _WORKER_SMILES, _WORKER_RADIUS, _WORKER_N_BITS
    _WORKER_SMILES = smiles
    _WORKER_RADIUS = radius
    _WORKER_N_BITS = n_bits
    RDLogger.DisableLog("rdApp.*")


def _fingerprint_range(
    bounds: tuple[int, int],
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    start, end = bounds
    if _WORKER_SMILES is None:
        raise RuntimeError("Fingerprint worker was not initialized")
    packed, counts, invalid_local, fallback_local = fingerprint_smiles_batch(
        _WORKER_SMILES[start:end],
        radius=_WORKER_RADIUS,
        n_bits=_WORKER_N_BITS,
    )
    invalid = [(start + row, value) for row, value in invalid_local]
    fallback = [(start + row, value) for row, value in fallback_local]
    return start, packed, counts, invalid, fallback


def _ranges(start: int, stop: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for offset in range(start, stop, batch_size):
        yield offset, min(offset + batch_size, stop)


def _state_payload(
    args: argparse.Namespace,
    n_vectors: int,
    rows: int,
    fallback_rows: Sequence[tuple[int, str]] = (),
) -> dict[str, Any]:
    return {
        "metadata_path": str(args.metadata.resolve()),
        "metadata_bytes": args.metadata_bytes,
        "metadata_sha256": args.metadata_sha256,
        "n_vectors": n_vectors,
        "n_bits": PAPER_N_BITS,
        "radius": PAPER_RADIUS,
        "nlist": args.nlist,
        "seed": args.seed,
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "fingerprint_algorithm": "Morgan/ECFP4",
        "binary_packbits_bitorder": "big",
        "fingerprinted_rows": rows,
        "kekulization_fallback_rows": [list(item) for item in fallback_rows],
    }


def _validate_resume_state(
    state: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[int, list[tuple[int, str]]]:
    identity_keys = (
        "metadata_path",
        "metadata_bytes",
        "metadata_sha256",
        "n_vectors",
        "n_bits",
        "radius",
        "nlist",
        "seed",
        "rdkit_version",
        "numpy_version",
        "fingerprint_algorithm",
        "binary_packbits_bitorder",
    )
    for key in identity_keys:
        if state.get(key) != expected.get(key):
            raise ValueError(
                f"Resume state mismatch for {key}: {state.get(key)!r} != {expected.get(key)!r}"
            )
    rows = int(state.get("fingerprinted_rows", 0))
    if not 0 <= rows <= int(expected["n_vectors"]):
        raise ValueError(f"Invalid fingerprinted_rows in resume state: {rows}")
    fallback_rows = [
        (int(item[0]), str(item[1]))
        for item in state.get("kekulization_fallback_rows", [])
    ]
    if any(row < 0 or row >= rows for row, _ in fallback_rows):
        raise ValueError("Resume state contains a fallback row outside completed fingerprints")
    return rows, fallback_rows


def _open_work_arrays(
    output_dir: Path,
    n_vectors: int,
    n_bits: int,
    resume: bool,
    overwrite: bool,
    state_expected: dict[str, Any],
) -> tuple[np.memmap, np.memmap, np.memmap, int, list[tuple[int, str]]]:
    float_path = output_dir / _WORK_FLOAT_NAME
    binary_path = output_dir / _WORK_BINARY_NAME
    counts_path = output_dir / _WORK_COUNTS_NAME
    state_path = output_dir / _STATE_NAME
    partials = (float_path, binary_path, counts_path, state_path)

    if resume and state_path.exists():
        if not all(path.exists() for path in partials):
            raise FileNotFoundError("Resume state exists but one or more partial arrays are missing")
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        completed, fallback_rows = _validate_resume_state(state, state_expected)
        vectors = np.load(float_path, mmap_mode="r+")
        packed = np.load(binary_path, mmap_mode="r+")
        counts = np.load(counts_path, mmap_mode="r+")
        expected_shapes = ((n_vectors, n_bits), (n_vectors, n_bits // 8), (n_vectors,))
        if (vectors.shape, packed.shape, counts.shape) != expected_shapes:
            raise ValueError(
                "Partial array shapes do not match this build: "
                f"{vectors.shape}, {packed.shape}, {counts.shape}"
            )
        expected_dtypes = (np.dtype(np.float32), np.dtype(np.uint8), np.dtype(np.uint16))
        if (vectors.dtype, packed.dtype, counts.dtype) != expected_dtypes:
            raise ValueError(
                "Partial array dtypes do not match this build: "
                f"{vectors.dtype}, {packed.dtype}, {counts.dtype}"
            )
        log(f"Resuming fingerprint generation at row {completed:,}/{n_vectors:,}")
        return vectors, packed, counts, completed, fallback_rows

    existing = [path for path in partials if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Partial build files already exist: {names}. Use --resume or --overwrite."
        )
    for path in existing:
        path.unlink()

    vectors = np.lib.format.open_memmap(
        float_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_vectors, n_bits),
    )
    packed = np.lib.format.open_memmap(
        binary_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_vectors, n_bits // 8),
    )
    counts = np.lib.format.open_memmap(
        counts_path,
        mode="w+",
        dtype=np.uint16,
        shape=(n_vectors,),
    )
    atomic_json_dump(state_path, state_expected)
    return vectors, packed, counts, 0, []


def generate_fingerprints(
    args: argparse.Namespace,
    smiles: Sequence[str],
    n_bits: int,
) -> tuple[np.memmap, np.memmap, np.memmap, list[tuple[int, str]]]:
    output_dir = args.output_dir
    n_vectors = len(smiles)
    state_expected = _state_payload(args, n_vectors, 0)
    vectors, packed_all, counts_all, start_row, fallback_rows = _open_work_arrays(
        output_dir,
        n_vectors,
        n_bits,
        args.resume,
        args.overwrite,
        state_expected,
    )
    if start_row == n_vectors:
        log("Fingerprint work arrays are already complete")
        return vectors, packed_all, counts_all, fallback_rows

    started = time.monotonic()
    ranges = _ranges(start_row, n_vectors, args.fingerprint_batch_size)
    completed = start_row

    global _WORKER_SMILES, _WORKER_RADIUS, _WORKER_N_BITS
    _WORKER_SMILES = smiles
    _WORKER_RADIUS = args.radius
    _WORKER_N_BITS = n_bits
    RDLogger.DisableLog("rdApp.*")

    if args.workers == 1:
        results: Iterable[
            tuple[
                int,
                np.ndarray,
                np.ndarray,
                list[tuple[int, str]],
                list[tuple[int, str]],
            ]
        ] = (
            _fingerprint_range(bounds) for bounds in ranges
        )
        pool = None
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Parallel full-corpus generation currently requires the 'fork' start method")
        context = mp.get_context("fork")
        pool = context.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(smiles, args.radius, n_bits),
        )
        results = pool.imap(_fingerprint_range, ranges, chunksize=1)

    last_report = time.monotonic()
    try:
        for start, packed, counts, invalid, fallback in results:
            if invalid:
                preview = ", ".join(f"{row}:{value!r}" for row, value in invalid[:5])
                raise ValueError(
                    f"Found {len(invalid)} invalid SMILES in batch starting at {start:,}: {preview}"
                )
            if np.any(counts == 0):
                zero_rows = np.flatnonzero(counts == 0)[:5] + start
                raise ValueError(f"Found zero-bit fingerprints at rows {zero_rows.tolist()}")

            end = start + len(packed)
            float_batch = np.unpackbits(packed, axis=1, count=n_bits).astype(np.float32)
            faiss.normalize_L2(float_batch)
            vectors[start:end] = float_batch
            packed_all[start:end] = packed
            counts_all[start:end] = counts
            completed = end
            fallback_rows.extend(fallback)

            now = time.monotonic()
            if now - last_report >= args.progress_interval or completed == n_vectors:
                vectors.flush()
                packed_all.flush()
                counts_all.flush()
                atomic_json_dump(
                    output_dir / _STATE_NAME,
                    _state_payload(args, n_vectors, completed, fallback_rows),
                )
                elapsed = max(now - started, 1e-9)
                rate = (completed - start_row) / elapsed
                remaining = (n_vectors - completed) / max(rate, 1e-9)
                log(
                    f"Fingerprints: {completed:,}/{n_vectors:,} "
                    f"({100.0 * completed / n_vectors:.1f}%), {rate:,.0f} mol/s, "
                    f"ETA {remaining / 60:.1f} min"
                )
                last_report = now
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        vectors.flush()
        packed_all.flush()
        counts_all.flush()
        atomic_json_dump(
            output_dir / _STATE_NAME,
            _state_payload(args, n_vectors, completed, fallback_rows),
        )

    log(f"Fingerprint generation complete: {n_vectors:,} rows")
    if fallback_rows:
        log(
            f"Used kekulization-tolerant parsing for {len(fallback_rows)} row(s): "
            + ", ".join(str(row) for row, _ in fallback_rows)
        )
    return vectors, packed_all, counts_all, fallback_rows


def build_index(args: argparse.Namespace, vectors: np.ndarray) -> faiss.IndexIVFFlat:
    n_vectors, dimension = vectors.shape
    if n_vectors < args.nlist:
        raise ValueError(f"Cannot train nlist={args.nlist} with only {n_vectors} vectors")

    faiss.omp_set_num_threads(args.faiss_threads)
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        args.nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.cp.seed = args.seed
    index.cp.niter = args.train_iterations
    index.cp.max_points_per_centroid = 256
    index.cp.min_points_per_centroid = 39
    log(
        f"Training IndexIVFFlat: vectors={n_vectors:,}, d={dimension}, "
        f"nlist={args.nlist}, metric=inner_product, seed={args.seed}, "
        f"iterations={args.train_iterations}, threads={args.faiss_threads}"
    )
    started = time.monotonic()
    index.train(vectors)
    log(f"FAISS training complete in {(time.monotonic() - started) / 60:.1f} min")

    started = time.monotonic()
    for start, end in _ranges(0, n_vectors, args.add_batch_size):
        ids = np.arange(start, end, dtype=np.int64)
        index.add_with_ids(np.asarray(vectors[start:end]), ids)
        if end == n_vectors or end % (args.add_batch_size * 10) == 0:
            log(f"FAISS add: {end:,}/{n_vectors:,}")
    log(f"FAISS add complete in {(time.monotonic() - started) / 60:.1f} min")

    if not index.is_trained or index.ntotal != n_vectors:
        raise RuntimeError(
            f"Built index failed count check: trained={index.is_trained}, "
            f"ntotal={index.ntotal}, expected={n_vectors}"
        )
    return index


def _validate_index_ids(index: faiss.IndexIVFFlat, expected: int) -> None:
    seen = np.zeros(expected, dtype=np.bool_)
    total = 0
    inverted_lists = index.invlists
    for list_number in range(index.nlist):
        size = inverted_lists.list_size(list_number)
        if size == 0:
            continue
        pointer = inverted_lists.get_ids(list_number)
        try:
            ids = faiss.rev_swig_ptr(pointer, size).copy()
        finally:
            inverted_lists.release_ids(list_number, pointer)
        if np.any(ids < 0) or np.any(ids >= expected):
            raise RuntimeError(f"Index list {list_number} contains out-of-range IDs")
        if np.unique(ids).size != ids.size or np.any(seen[ids]):
            raise RuntimeError(f"Index list {list_number} contains duplicate IDs")
        seen[ids] = True
        total += size
    if total != expected or not np.all(seen):
        raise RuntimeError(
            f"Index ID coverage failed: found={total:,}, expected={expected:,}, "
            f"missing={int(np.count_nonzero(~seen)):,}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MolMem's paper-version static exemplar FAISS index from metadata."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nlist", type=int, default=PAPER_NLIST)
    parser.add_argument("--radius", type=int, default=PAPER_RADIUS)
    parser.add_argument("--n-bits", type=int, default=PAPER_N_BITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-iterations", type=int, default=25)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--faiss-threads", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--fingerprint-batch-size", type=int, default=2048)
    parser.add_argument("--add-batch-size", type=int, default=10_000)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Build only the first N rows (for smoke testing; never use for release artifacts).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work-vectors", action="store_true")
    parser.add_argument("--skip-corpus-check", action="store_true")
    args = parser.parse_args(argv)

    for name in (
        "nlist",
        "radius",
        "n_bits",
        "train_iterations",
        "workers",
        "faiss_threads",
        "fingerprint_batch_size",
        "add_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.radius != PAPER_RADIUS or args.n_bits != PAPER_N_BITS:
        parser.error(
            f"MolMem release artifacts require --radius={PAPER_RADIUS} and --n-bits={PAPER_N_BITS}"
        )
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if args.corpus is None and not args.skip_corpus_check:
        parser.error("--corpus is required unless --skip-corpus-check is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.metadata = args.metadata.resolve()
    args.corpus = args.corpus.resolve() if args.corpus else None
    args.output_dir = args.output_dir.resolve()

    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if args.corpus is not None and not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.metadata_bytes = args.metadata.stat().st_size
    log(f"Hashing metadata input before build: {args.metadata.name}")
    args.metadata_sha256 = sha256_file(args.metadata)
    args.corpus_bytes = args.corpus.stat().st_size if args.corpus is not None else None
    if args.corpus is not None:
        log(f"Hashing corpus input before build: {args.corpus.name}")
        args.corpus_sha256 = sha256_file(args.corpus)
    else:
        args.corpus_sha256 = None

    final_paths = {
        "index": args.output_dir / INDEX_NAME,
        "binary": args.output_dir / BINARY_NAME,
        "counts": args.output_dir / BIT_COUNT_NAME,
        "manifest": args.output_dir / MANIFEST_NAME,
    }
    existing_final = [path for path in final_paths.values() if path.exists()]
    if existing_final and not args.overwrite:
        names = ", ".join(str(path) for path in existing_final)
        raise FileExistsError(f"Output artifacts already exist: {names}. Use --overwrite to replace.")

    build_started = time.monotonic()
    smiles, n_bits, radius, metadata_total = load_metadata_spec(args.metadata, args.limit)
    if radius != args.radius or n_bits != args.n_bits:
        raise ValueError("CLI fingerprint settings do not match metadata")
    n_vectors = len(smiles)
    is_release_build = args.limit is None
    if is_release_build and args.nlist != PAPER_NLIST:
        raise ValueError(
            f"A full release build must use the paper nlist={PAPER_NLIST}; got {args.nlist}. "
            "Use --limit only when building a smaller index."
        )
    if is_release_build and (
        args.seed != DEFAULT_SEED
        or args.train_iterations != 25
        or args.skip_corpus_check
    ):
        raise ValueError(
            "A full release build requires seed=1234, train_iterations=25, "
            "and the complete corpus alignment check"
        )
    log(
        f"Build specification: rows={n_vectors:,}/{metadata_total:,}, radius={radius}, "
        f"bits={n_bits}, nlist={args.nlist}"
    )
    check_disk_space(args.output_dir, n_vectors, n_bits, args.nlist)

    if not args.skip_corpus_check:
        assert args.corpus is not None
        validate_corpus_alignment(args.corpus, smiles, require_exact_count=is_release_build)

    vectors, packed, counts, fallback_rows = generate_fingerprints(args, smiles, n_bits)
    index = build_index(args, vectors)
    _validate_index_ids(index, n_vectors)
    log("Index document-ID coverage verified")

    index_partial = args.output_dir / _INDEX_PARTIAL_NAME
    faiss.write_index(index, str(index_partial))
    del index
    gc.collect()
    reloaded = faiss.read_index(str(index_partial))
    if (
        reloaded.ntotal != n_vectors
        or reloaded.d != n_bits
        or reloaded.nlist != args.nlist
        or not reloaded.is_trained
        or reloaded.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise RuntimeError("Serialized FAISS index failed round-trip validation")
    del reloaded
    gc.collect()

    # Hash all partial artifacts before publication. The manifest is the final
    # commit marker; hard links leave partials available if publication stops.
    packed.flush()
    counts.flush()
    del packed, counts

    artifacts: dict[str, Any] = {}
    partial_paths = {
        "index": index_partial,
        "binary": args.output_dir / _WORK_BINARY_NAME,
        "counts": args.output_dir / _WORK_COUNTS_NAME,
    }
    for name, path in partial_paths.items():
        log(f"Hashing {final_paths[name].name} ({path.stat().st_size / (1024 ** 3):.2f} GiB)")
        artifacts[name] = {
            "filename": final_paths[name].name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    inputs: dict[str, Any] = {
        "metadata": {
            "path": str(args.metadata),
            "bytes": args.metadata_bytes,
            "sha256": args.metadata_sha256,
        }
    }
    if args.corpus is not None:
        inputs["corpus"] = {
            "path": str(args.corpus),
            "bytes": args.corpus_bytes,
            "sha256": args.corpus_sha256,
        }

    manifest = {
        "release_build": is_release_build,
        "n_molecules": n_vectors,
        "metadata_n_molecules": metadata_total,
        "fingerprint": {
            "algorithm": "Morgan/ECFP4",
            "radius": radius,
            "n_bits": n_bits,
            "float_dtype": "float32",
            "normalization": "L2",
            "binary_packbits_bitorder": "big",
            "kekulization_fallback_rows": [list(item) for item in fallback_rows],
        },
        "faiss": {
            "index_type": "IndexIVFFlat",
            "quantizer": "IndexFlatIP",
            "metric": "METRIC_INNER_PRODUCT",
            "nlist": args.nlist,
            "seed": args.seed,
            "train_iterations": args.train_iterations,
            "max_points_per_centroid": 256,
            "min_points_per_centroid": 39,
            "omp_threads": args.faiss_threads,
            "version": getattr(faiss, "__version__", "unknown"),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "inputs": inputs,
        "artifacts": artifacts,
        "build_seconds": time.monotonic() - build_started,
    }

    log("Rechecking source hashes before publishing artifacts")
    if sha256_file(args.metadata) != args.metadata_sha256:
        raise RuntimeError("Metadata changed during the build")
    if args.corpus is not None and sha256_file(args.corpus) != args.corpus_sha256:
        raise RuntimeError("Corpus changed during the build")

    manifest_partial = args.output_dir / f".{MANIFEST_NAME}.partial"
    atomic_json_dump(manifest_partial, manifest)
    for name, partial in partial_paths.items():
        publish_hardlinked_artifact(partial, final_paths[name])
    os.replace(manifest_partial, final_paths["manifest"])

    for partial in partial_paths.values():
        if partial.exists():
            partial.unlink()

    state_path = args.output_dir / _STATE_NAME
    if state_path.exists():
        state_path.unlink()
    if not args.keep_work_vectors:
        del vectors
        float_work = args.output_dir / _WORK_FLOAT_NAME
        if float_work.exists():
            float_work.unlink()
    log(f"Build complete: {final_paths['index']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; partial fingerprint arrays were retained for --resume")
        raise SystemExit(130)
