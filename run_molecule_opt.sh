#!/usr/bin/env bash
set -euo pipefail

TRAINER_GPUS="${TRAINER_GPUS:-1,2}"
PYTHON_BIN="${TRAINER_PYTHON:-python}"
RETRIEVER_HEALTH_URL="${RETRIEVER_HEALTH_URL:-http://127.0.0.1:8000/cache_stats}"

if ! curl -fsS --max-time 5 "$RETRIEVER_HEALTH_URL" >/dev/null; then
    echo "Error: Static Exemplar Memory is required but is not reachable at:"
    echo "  $RETRIEVER_HEALTH_URL"
    echo "Start it first on a separate GPU with start_static_exemplar_server.sh."
    exit 1
fi

IFS=',' read -r -a trainer_gpu_ids <<< "$TRAINER_GPUS"
trainer_gpu_count="${#trainer_gpu_ids[@]}"
if [ "$trainer_gpu_count" -lt 1 ]; then
    echo "Error: TRAINER_GPUS must contain at least one GPU id."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$TRAINER_GPUS"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

cuda_override="system.CUDA_VISIBLE_DEVICES=\"${TRAINER_GPUS}\""

echo "Starting MolMem training"
echo "  trainer GPUs: $TRAINER_GPUS"
echo "  retriever:    $RETRIEVER_HEALTH_URL"

exec "$PYTHON_BIN" train.py \
    --config-name molecule_opt \
    "number_of_gpus=${trainer_gpu_count}" \
    "$cuda_override" \
    "$@"
