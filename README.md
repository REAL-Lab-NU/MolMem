# MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization

[![Conference](https://img.shields.io/badge/ACL%202026-Main-b31b1b)](https://arxiv.org/abs/2604.12237)
[![arXiv](https://img.shields.io/badge/arXiv-2604.12237-b31b1b)](https://arxiv.org/abs/2604.12237)

Official code for our ACL 2026 Main paper. MolMem combines a static ChEMBL
exemplar bank with an evolving skill memory for single- and multi-objective
molecular optimization.

## Installation

Python 3.11 is recommended. Create the training environment from the repository
root:

```bash
conda create -n ragen python=3.11 pip -y
conda activate ragen
python -m pip install -r requirements.txt
python -m pip install --no-deps PyTDC==1.1.15
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install --force-reinstall --no-deps scikit-learn==1.2.2
```

`PyTDC` is installed without its optional dependency bundle because that bundle
pins an older Transformers stack; MolMem uses only its molecular oracles.

Use a separate environment to build the retrieval index:

```bash
conda create -n retriever python=3.11 pip -y
conda activate retriever
python -m pip install -r requirements-retrieval.txt
python -m pip install faiss-cpu==1.8.0
```

Serving requires a GPU-enabled FAISS installation. Create another environment
and follow [retrieval/FAISS_GPU.md](retrieval/FAISS_GPU.md) for the FAISS build:

```bash
conda create -n retriever-gpu python=3.11 pip -y
conda activate retriever-gpu
python -m pip install -r requirements-retrieval.txt
```

## SFT Initialization

The RL stage starts from a molecular instruction-tuned model. We use
[AlexanderWang915/MolOptIns](https://huggingface.co/datasets/AlexanderWang915/MolOptIns)
as the SFT dataset and fine-tune Qwen2.5-1.5B with
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The resulting SFT
checkpoint is available at
[AlexanderWang915/qwen2.5-1.5b-moloptins](https://huggingface.co/AlexanderWang915/qwen2.5-1.5b-moloptins).
You can reproduce the SFT stage with MolOptIns and LLaMA-Factory or use the
released checkpoint directly; the training command below uses the released
checkpoint.

## Build the Static Exemplar Index

Download these GitHub Release assets into `retriever_data/`:

- `molecular_corpus.jsonl`
- `molecular_metadata.pkl`

Build and validate the local FAISS index in the `retriever` environment:

```bash
conda activate retriever

python retrieval/build_retriever_index.py \
    --metadata retriever_data/molecular_metadata.pkl \
    --corpus retriever_data/molecular_corpus.jsonl \
    --output-dir retriever_data \
    --workers 32 \
    --faiss-threads 64 \
    --resume

python retrieval/validate_retriever_artifacts.py \
    --data-dir retriever_data \
    --samples 1000
```

The reference index uses Morgan/ECFP4 fingerprints with radius 2 and 2,048
bits, and FAISS `IndexIVFFlat` with `nlist=1689`. See
[retrieval/BUILD_INDEX.md](retrieval/BUILD_INDEX.md) for checksums, hardware
requirements, resume behavior, and detailed build instructions.

## Configure Azure OpenAI

GPT summaries are generated during training. Export the Azure credentials
before starting MolMem:

```bash
export AZURE_OPENAI_API_KEY=your_api_key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-12-01-preview
export AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Start the Retrieval Server

Keep the server running on a dedicated GPU:

```bash
conda activate retriever-gpu
RETRIEVER_PYTHON="$CONDA_PREFIX/bin/python" \
    bash start_static_exemplar_server.sh \
    --data-dir retriever_data \
    --port 8000 \
    --gpu 0 \
    --nprobe 800
```

Verify that it is ready:

```bash
curl -sS http://127.0.0.1:8000/cache_stats
```

## Train

The example below keeps GPU 0 for retrieval and uses GPUs 1 and 2 for training:

```bash
conda activate ragen

TRAINER_GPUS=1,2 bash run_molecule_opt.sh \
    molecule_opt_task=qed \
    train_size=128 \
    model_path=AlexanderWang915/qwen2.5-1.5b-moloptins
```

Append Hydra overrides such as `trainer.total_training_steps=100`,
`trainer.save_freq=20`, or `trainer.default_local_dir=/path/to/output` to the
same command. Checkpoints and `skill_memory.json`/`skill_memory.pkl` are written
under `trainer.default_local_dir`.

## Evaluate

Training checkpoints are stored as FSDP shards. Merge the actor weights into a
Hugging Face directory before evaluation:

```bash
python merge_fsdp_checkpoint.py \
    --base-model AlexanderWang915/qwen2.5-1.5b-moloptins \
    --checkpoint-dir /path/to/run/global_step_100/actor \
    --output-dir /path/to/run/global_step_100/actor/merged_hf
```

Keep the retrieval server running, then evaluate the merged checkpoint together
with its skill-memory PKL:

```bash
conda activate ragen

python evaluate.py --config-name molecule_opt_inference \
    model_path=/path/to/run/global_step_100/actor/merged_hf \
    skill_memory_path=/path/to/skill_memory.pkl \
    molecule_opt_task=qed \
    number_of_gpus=2 \
    system.CUDA_VISIBLE_DEVICES='"1,2"' \
    inference_output=/path/to/inference_results.json
```

The evaluation defaults follow the paper: 200 leads, a 500-call oracle budget
per lead, 20 generations, 32 rollouts per generation, five turns per rollout,
and a temperature schedule from 0.9 to 2.0 in increments of 0.1.

## Tasks and Data

Supported properties are `qed`, `logp`, `sa`, `jnk3`, and `drd2`. Join property
names with `+` for multi-objective tasks, for example `qed+logp` or
`drd2+qed+sa`.

Run training and evaluation from the repository root so PyTDC can find the
included JNK3, DRD2, and SA oracle assets under `oracle/`.

Dataset files use this layout:

```text
data/
└── {task}/
    ├── train/{task}_train_{size}.parquet
    ├── val/{task}_val_32.parquet
    └── test/{task}_test_200.parquet
```

Each Parquet file must contain a `smiles` column. A `seed` column is optional;
the row index is used when it is absent.

## Citation

```bibtex
@inproceedings{wang2026molmem,
  title={Molmem: Memory-augmented agentic reinforcement learning for sample-efficient molecular optimization},
  author={Wang, Ziqing and Wen, Yibo and Pandey, Abhishek and Liu, Han and Ding, Kaize},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={43694--43712},
  year={2026}
}
```

## License

This project is released for academic research purposes.
