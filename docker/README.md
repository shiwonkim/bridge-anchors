# Bridge Anchors — Docker Environment

Environment-only image for the Bridge Anchors project. Contains Python 3.10, PyTorch 2.4.1+cu118, and all dependencies. Code and data are mounted at runtime via volumes.

## Environment

- **Base**: `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime`
- **Python**: 3.11 (from base image; our conda env uses 3.10 but all deps are compatible)
- **PyTorch**: 2.4.1+cu118 (CUDA 11.8 runtime, compatible with CUDA 12.x host drivers)
- **Key packages**: sentence-transformers, transformers, timm, numpy, matplotlib, umap-learn, scikit-learn

## Quick Setup (New Server)

The easiest way to get started on a new server:

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/bridge-anchors.git
cd bridge-anchors

# 2. Run the setup script (pulls image, checks GPU, launches container)
bash docker/setup_server.sh
```

The script accepts optional arguments for custom paths:
```bash
# Default: code = repo root, data = repo/data
bash docker/setup_server.sh

# Custom paths
bash docker/setup_server.sh /path/to/bridge-anchors /path/to/data
```

**Prerequisites** on the new server:
- Docker with `nvidia-container-toolkit` installed
- NVIDIA GPU with driver supporting CUDA >= 11.8

**Docker Hub image**: `shiwonkim/bridge-anchors:v2`

## Build (from source)

Only needed if you want to rebuild the image locally:
```bash
cd docker
docker build -t bridge-anchors:v2 .
```

## Run (manual)

### Basic (code mount only)
```bash
docker run --gpus all -it --shm-size=16g \
  -v /path/to/bridge-anchors:/workspace/bridge-anchors \
  -e PYTHONPATH=/workspace/bridge-anchors \
  -w /workspace/bridge-anchors \
  bridge-anchors:v2 bash
```

### With separate data mount
```bash
docker run --gpus all -it --shm-size=16g \
  -v /path/to/bridge-anchors:/workspace/bridge-anchors \
  -v /path/to/data:/workspace/bridge-anchors/data \
  -e PYTHONPATH=/workspace/bridge-anchors \
  -w /workspace/bridge-anchors \
  bridge-anchors:v2 bash
```

### With docker-compose
Edit `docker-compose.yml` to set the correct paths, then:
```bash
cd docker
docker compose up -d
docker compose exec bridge-anchors bash
```

## Inside the container

```bash
# Training
python -m src.train --config configs/default.yaml --model bridge_anchors --num-anchors 128

# Evaluation
python -m src.eval.retrieval --checkpoint path/to/ckpt

# Token embedding extraction
PYTHONPATH=/workspace/bridge-anchors python scripts/extract_token_embeddings_full.py
```

## Notes

- **CUDA compatibility**: The cu118 runtime is forward-compatible with CUDA 12.x host drivers. Tested on NVIDIA A40 (CUDA 11.4 driver) and expected to work on Quadro RTX 8000 (CUDA 12.4 driver).
- **Shared memory**: `--shm-size=16g` is needed for PyTorch DataLoader with `num_workers > 0`.
- **Data directory**: Embeddings and datasets are large (~100+ GB). Mount them from a fast local disk, not NFS, for extraction workloads.
- **Symlinks**: If your code repo uses symlinks to data on a separate mount (e.g., NAS), you must also mount that path in the container so symlinks resolve correctly. Example: `-v /mnt/nas/data:/mnt/nas/data:ro`
- **Verified**: 1-epoch baseline produces mR=12.64, matching native conda environment exactly.
