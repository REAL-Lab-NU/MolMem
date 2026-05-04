#!/bin/bash
#
# Start Static Exemplar Memory Server (MolMem)
#
# This script starts the Static Exemplar Memory retrieval server.
# Before running, prepare the required data files (see README for details).
#
# Usage:
#   bash start_static_exemplar_server.sh [OPTIONS]
#
# Examples:
#   # Start with defaults
#   bash start_static_exemplar_server.sh
#
#   # Custom port
#   bash start_static_exemplar_server.sh --port 8001
#

set -e  # Exit on error

# Configuration - Update these paths to your data files
DATA_DIR="./retriever_data"
INDEX_PATH="${DATA_DIR}/molecular_faiss.index"
METADATA_PATH="${DATA_DIR}/molecular_metadata.pkl"
CORPUS_PATH="${DATA_DIR}/molecular_corpus.jsonl"
PORT=8000
NPROBE=800  # Optimized nprobe value for IndexIVFFlat
RETRIEVER_GPU=0  # GPU for retrieval (set to available GPU)

# Parse command line arguments
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

# Check if required files exist
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
echo "================================================================================"
echo ""

# Set GPU for retrieval service
export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"

# Start the server with GPU acceleration
python retrieval/server/static_exemplar_memory.py \
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
