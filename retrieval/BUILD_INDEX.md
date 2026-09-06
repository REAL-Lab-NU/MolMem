# Static Exemplar Index Reproduction

This document reproduces the released MolMem Static Exemplar Memory index from
the two canonical release inputs. It does not regenerate oracle properties from
raw ChEMBL tables; those precomputed values and their canonical document order
are already contained in the release metadata and corpus.

## Index specification

The index follows the paper configuration:

- Source bank: ChEMBL 36, 2,854,654 molecules
- Fingerprint: Morgan/ECFP4, radius 2, 2,048 bits
- Float representation: `float32`, L2-normalized
- FAISS index: `IndexIVFFlat`
- Quantizer and ranking metric: `IndexFlatIP` / inner product
- Number of IVF lists: `nlist=1689`
- Clustering seed: 1234
- Clustering iterations: 25
- Maximum training points per centroid: 256

For normalized binary fingerprints, inner-product ranking is equivalent to
cosine ranking. The release server subsequently applies exact Tanimoto
similarity using the packed binary fingerprints generated alongside the FAISS
index.

`nprobe` is a query-time parameter and is not part of index construction.
Changing it does not require rebuilding the index.

## Canonical release inputs

Download these two files from the MolMem GitHub Release into
`retriever_data/`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `molecular_corpus.jsonl` | 838,272,986 | `ba3893cced70854836a5d769f7e98d9751d6f0e62d95893fbcd93adfffc949e8` |
| `molecular_metadata.pkl` | 467,304,684 | `465469f330523ab2ba8d3c07beaf007dfa003fc800cc7426979e1522e6e3f57b` |

Verify them before construction:

```bash
sha256sum retriever_data/molecular_corpus.jsonl \
    retriever_data/molecular_metadata.pkl
wc -l retriever_data/molecular_corpus.jsonl
```

The expected corpus line count is `2,854,654`. Only load the PKL downloaded
from the trusted MolMem release.

The paper describes construction-time metadata in SQLite. The public PKL is a
portable ordered export used by the released server: FAISS document ID `i`
maps directly to `metadata["smiles_list"][i]`,
`metadata["properties"][i]`, and corpus record `i`.

## Environment

Python 3.11 is recommended. Create a dedicated Conda environment and install
the pinned retrieval dependencies from the repository root:

```bash
conda create -n retriever python=3.11 pip
conda activate retriever
pip install -r requirements-retrieval.txt
pip install faiss-cpu==1.8.0
```

`requirements-retrieval.txt` intentionally omits FAISS because serving uses a
GPU build. The command above installs the CPU build used for index construction.

For a full CPU build, use at least 48 GiB RAM and 50 GiB free disk space;
64 GiB RAM and 60 GiB free disk are safer. Rebuilding while retaining an old
22 GiB index requires additional disk space.

## Full construction

From the MolMem repository root, run:

```bash
python retrieval/build_retriever_index.py \
    --metadata retriever_data/molecular_metadata.pkl \
    --corpus retriever_data/molecular_corpus.jsonl \
    --output-dir retriever_data \
    --nlist 1689 \
    --workers 32 \
    --faiss-threads 64 \
    --resume
```

`--workers` controls parallel RDKit fingerprint generation and does not change
document order. Adjust it and `--faiss-threads` for the available CPU and memory.

The builder performs the following checks and stages:

1. Verifies metadata shape and fixed ECFP4 settings.
2. Checks every corpus ID and SMILES against the PKL order.
3. Generates normalized float fingerprints and packed binary fingerprints.
4. Trains `IndexIVFFlat(nlist=1689)` with fixed clustering parameters.
5. Adds all vectors with explicit metadata row IDs.
6. Checks complete ID coverage, serialization, and input stability.
7. Hashes staged artifacts and publishes the manifest last as a commit marker.

Fingerprint generation is resumable. If interrupted, rerun the same command
with `--resume`. The completed fingerprint matrix is reused; an interruption
during FAISS clustering may require repeating clustering but not RDKit
fingerprint generation.

To intentionally replace already completed artifacts, use `--overwrite`.
Do not use `--limit` for release construction; it exists only for smoke tests.

## Output artifacts

A successful build creates:

| File | Purpose |
| --- | --- |
| `molecular_faiss.index` | Float IVF recall index |
| `molecular_binary_fingerprints.npy` | Packed 2,048-bit fingerprints for exact Tanimoto |
| `molecular_bit_counts.npy` | Precomputed fingerprint population counts |
| `molecular_faiss.manifest.json` | Parameters, versions, sizes, and checksums |

The input checksums must match exactly. Output index bytes may differ across
FAISS builds, CPU architectures, or thread schedules; the generated manifest
and the structural/search validator are authoritative for that build.

## Independent validation

Run the full post-build validator:

```bash
python retrieval/validate_retriever_artifacts.py \
    --data-dir retriever_data \
    --samples 1000
```

It verifies:

- manifest sizes and SHA256 values;
- all 2,854,654 corpus/metadata rows;
- index type, dimension, metric, `nlist`, and `ntotal`;
- every ID in every inverted list, including duplicates and missing IDs;
- packed fingerprints and bit counts against regenerated RDKit values;
- randomized self-searches whose top similarity must be approximately 1.0.

One canonical molecule needs RDKit's non-kekulizing compatibility path. The
builder, validator, and online fingerprinter handle it consistently and record
it in the generated manifest; no manual preprocessing is required.

For distribution, publish the canonical corpus and PKL as GitHub Release
assets. The generated FAISS index is intentionally ignored by Git and rebuilt
locally with this procedure.
