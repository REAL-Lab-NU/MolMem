# FAISS-GPU runtime and 1+2 GPU launch

MolMem uses separate processes and environments for retrieval and training:

```text
physical GPU 0      FAISS-GPU Static Exemplar Memory (persistent HTTP server)
physical GPUs 1,2  veRL/vLLM trainer (logical GPUs 0,1 inside the process)
```

The retrieval process must be started first. Training calls it over
`http://127.0.0.1:8000/retrieve`. The server intentionally rejects a CPU-only
FAISS installation when launched with `--faiss_gpu`.

Build and validate `retriever_data/` first by following
[`BUILD_INDEX.md`](BUILD_INDEX.md).

## Base retrieval environment

```bash
conda create -n retriever-gpu python=3.11 pip -y
conda activate retriever-gpu
pip install -r requirements-retrieval.txt
```

Install GPU FAISS separately. A CUDA 12 wheel may be sufficient on older GPU
architectures, but verify it with the smoke test below before loading the full
index.

## Blackwell GPUs (`sm_120`)

Prebuilt FAISS wheels that do not contain `sm_120` kernels fail at runtime with
`no kernel image is available for execution on the device`; a successful import
is not enough. Build official FAISS with CUDA 12.8 or newer and explicitly
target architecture 120. Set `CUDA_TOOLKIT_ROOT` to the local CUDA toolkit
directory before configuring the build:

```bash
conda activate retriever-gpu
conda install -c conda-forge -y cmake ninja swig libopenblas-devel

export CUDA_TOOLKIT_ROOT=/path/to/cuda
export FAISS_SOURCE_DIR=/tmp/faiss-1.14.2

git clone --depth 1 --branch v1.14.2 \
    https://github.com/facebookresearch/faiss.git \
    "$FAISS_SOURCE_DIR"

cmake -S "$FAISS_SOURCE_DIR" -B "$FAISS_SOURCE_DIR/build" -G Ninja \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_CUVS=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    -DBLA_VENDOR=OpenBLAS \
    -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"

cmake --build "$FAISS_SOURCE_DIR/build" \
    --target faiss swigfaiss -j32

python -m pip install --force-reinstall --no-deps \
    "$FAISS_SOURCE_DIR/build/faiss/python"
```

## GPU smoke test

Run this before loading the 22 GiB IVF index:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import faiss
import numpy as np

print("faiss", faiss.__version__, "visible GPUs", faiss.get_num_gpus())
assert faiss.get_num_gpus() == 1
assert hasattr(faiss, "StandardGpuResources")

resources = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(2048)
gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
vectors = np.random.default_rng(0).random((32, 2048), dtype=np.float32)
faiss.normalize_L2(vectors)
gpu_index.add(vectors)
print(gpu_index.search(vectors[:1], 3))
PY
```

## Start and verify the full service

```bash
conda activate retriever-gpu
RETRIEVER_PYTHON="$CONDA_PREFIX/bin/python" \
    bash start_static_exemplar_server.sh --gpu 0 --nprobe 800

curl -sS http://127.0.0.1:8000/cache_stats
nvidia-smi
```

Startup should report `2,854,654` molecules, `FAISS GPU: True`, and a ready
server on port 8000. The full float16 GPU clone requires approximately 25 GiB
of GPU memory. Start training in a second shell:

```bash
conda activate ragen
TRAINER_GPUS=1,2 bash run_molecule_opt.sh \
    molecule_opt_task=qed \
    model_path=/path/to/model
```
