#!/bin/bash
# Start Static Exemplar Memory Server (MolMem)
# Usage:
#   bash start_static_exemplar_server.sh [OPTIONS]

set -e

DATA_DIR="./retriever_data"
INDEX_PATH="${DATA_DIR}/molecular_faiss.index"
METADATA_PATH="${DATA_DIR}/molecular_metadata.pkl"
CORPUS_PATH="${DATA_DIR}/molecular_corpus.jsonl"
PORT=8000
NPROBE=800
RETRIEVER_GPU=0
PYTHON_BIN="${RETRIEVER_PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --nprobe)
            NPROBE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            INDEX_PATH="${DATA_DIR}/molecular_faiss.index"
            METADATA_PATH="${DATA_DIR}/molecular_metadata.pkl"
            CORPUS_PATH="${DATA_DIR}/molecular_corpus.jsonl"
            shift 2
            ;;
        --gpu)
            RETRIEVER_GPU="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT        Server port (default: 8000)"
            echo "  --nprobe NPROBE    FAISS nprobe parameter (default: 800)"
            echo "  --data-dir DIR     Directory containing data files (default: ./retriever_data)"
            echo "  --gpu GPU_ID       GPU to use for retrieval (default: 0)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ ! -f "$INDEX_PATH" ]; then
    echo "Error: Index file not found: $INDEX_PATH"
    echo "Please prepare data files first. See README for instructions."
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found: $METADATA_PATH"
    exit 1
fi

if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found: $CORPUS_PATH"
    exit 1
fi

echo "================================================================================"
echo "Starting Static Exemplar Memory Server (MolMem)"
echo "================================================================================"
echo "Index: $INDEX_PATH"
echo "Metadata: $METADATA_PATH"
echo "Corpus: $CORPUS_PATH"
echo "Port: $PORT"
echo "nprobe: $NPROBE"
echo "GPU: $RETRIEVER_GPU"
echo "Python: $PYTHON_BIN"
echo "================================================================================"
echo ""

export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"

if ! "$PYTHON_BIN" -c 'import faiss, sys; n = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0; ok = n > 0 and hasattr(faiss, "StandardGpuResources"); print("FAISS {}; visible GPUs: {}".format(getattr(faiss, "__version__", "unknown"), n)); sys.exit(0 if ok else 1)'; then
    echo "Error: --faiss_gpu requires a working FAISS-GPU build, but none is available."
    echo "See retrieval/FAISS_GPU.md. faiss-cpu is not sufficient."
    exit 1
fi

"$PYTHON_BIN" retrieval/server/static_exemplar_memory.py \
    --index_path "$INDEX_PATH" \
    --metadata_path "$METADATA_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --port $PORT \
    --topk 50 \
    --threshold 0.4 \
    --nprobe $NPROBE \
    --faiss_gpu

echo ""
echo "Server stopped."
