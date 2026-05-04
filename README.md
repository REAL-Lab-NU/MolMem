# MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization

## Overview

MolMem (**Mol**ecular optimization with **Mem**ory) is a multi-turn agentic reinforcement learning framework with a dual-memory system for sample-efficient molecular optimization. It iteratively refines a lead compound to improve molecular properties while preserving structural similarity to the original molecule under a limited oracle budget.

### Key Features

- **Dual-Memory System**: Combines Static Exemplar Memory for cold-start grounding with Evolving Skill Memory for experience-based learning
- **Sample Efficiency**: Achieves high success rates with limited oracle evaluations
- **Multi-objective Optimization**: Supports both single and multi-property optimization tasks
- **Turn-level Reward**: Uses turn-level scoring to guide optimization

## Architecture

```
MolMem/
├── config/                          # Configuration files
│   ├── base.yaml                   # Base configuration
│   └── molecule_opt.yaml           # Molecule optimization config
├── ragen/
│   ├── env/
│   │   └── molecule_opt/           # Molecule optimization environment
│   │       ├── config.py           # Environment & memory configs
│   │       ├── env.py              # Environment implementation
│   │       └── property_utils.py   # Property calculation utilities
│   └── llm_agent/
│       ├── evolving_skill_memory.py  # Evolving Skill Memory module
│       ├── es_manager.py           # Environment state manager
│       └── ctx_manager.py          # Context manager
├── retrieval/
│   └── server/
│       └── static_exemplar_memory.py  # Static Exemplar Memory server
├── start_static_exemplar_server.sh    # Server startup script
└── train.py                        # Training entry point
```

## Dual-Memory System

### Static Exemplar Memory

Provides cold-start grounding by retrieving similar molecules from a large pre-indexed chemical database.

- Uses Morgan fingerprints (ECFP4, radius=2, 2048-bit)
- FAISS index for fast approximate nearest neighbor search
- Tanimoto similarity filtering (threshold >= 0.4)
- Property-aware retrieval (QED, LogP, SA, JNK3, DRD2)

### Evolving Skill Memory

Distills successful optimization trajectories into reusable strategies.

- Extracts skills from successful optimizations
- Functional group detection and change analysis
- Multiple retrieval methods (by functional group, similarity, task)
- Optional GPT-based strategy summarization

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers rdkit-pypi
pip install hydra-core omegaconf
pip install pandas numpy

# For Static Exemplar Memory server
pip install faiss-gpu  # or faiss-cpu
pip install fastapi uvicorn
pip install datasets
```

### Setup

1. Clone the repository
2. Install dependencies
3. Configure model path in `config/molecule_opt.yaml`:

```yaml
model_path: YOUR_MODEL_PATH  # Path to your pretrained model
```

## Usage

### Training

```bash
python train.py --config-name molecule_opt \
    molecule_opt_task=qed \
    train_size=128
```

### Supported Tasks

**Single-objective:**
- `qed` - Drug-likeness (QED) [maximize]
- `logp` - Lipophilicity (LogP) [maximize]
- `sa` - Synthetic Accessibility [minimize]
- `jnk3` - JNK3 inhibition [maximize]
- `drd2` - DRD2 inhibition [maximize]

**Multi-objective (use `+` to combine):**
- `qed+logp` - Optimize both QED and LogP
- `drd2+qed` - Optimize both DRD2 and QED
- `drd2+qed+sa` - Complex multi-objective

### Starting Static Exemplar Memory Server

```bash
bash start_static_exemplar_server.sh --port 8000 --gpu 0
```

Options:
- `--port PORT` - Server port (default: 8000)
- `--gpu GPU_ID` - GPU for FAISS index (default: 0)
- `--nprobe NPROBE` - FAISS search parameter (default: 800)
- `--data-dir DIR` - Data directory (default: ./retriever_data)

## Configuration

### Dual-Memory System

Configure in `config/molecule_opt.yaml`:

```yaml
# Static Exemplar Memory (Retrieval)
static_exemplar_memory:
  enabled: true
  url: "http://127.0.0.1:8000/retrieve"
  topk: 5
  timeout: 5.0
  trigger_mode: "on_stuck"  # "always", "on_stuck", "never"
  similarity_threshold: 0.4

# Evolving Skill Memory
evolving_skill_memory:
  enabled: true
  max_size: 10000
  save_path: "${trainer.default_local_dir}/skill_memory.pkl"
  min_score_delta: 0.01
```

### Training Parameters

```yaml
trainer:
  experiment_name: molmem-${train_size}-${molecule_opt_task}
  total_training_steps: 100
  save_freq: 20

agent_proxy:
  max_turn: 5  # Maximum modification steps per episode
```

## Data Preparation

### Static Exemplar Memory Index

The Static Exemplar Memory requires a FAISS index of molecular fingerprints. Data files are not included due to size.

Required files in `retriever_data/`:
- `molecular_faiss.index` - FAISS index
- `molecular_metadata.pkl` - Molecule metadata (SMILES, properties)
- `molecular_corpus.jsonl` - Corpus for retrieval results

### Training Data

Prepare training data as Parquet files:

```
data/
└── {task}/
    ├── train/
    │   └── {task}_train_{size}.parquet
    └── val/
        └── {task}_val_32.parquet
```

Each Parquet file should contain:
- `smiles` - Initial molecule SMILES
- `seed` - Random seed for reproducibility

## Environment Variables

For Evolving Skill Memory GPT summarization (optional):

```bash
export AZURE_OPENAI_API_KEY=your_api_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
```

## License

This project is released for academic research purposes.
