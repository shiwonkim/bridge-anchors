#!/usr/bin/env bash
# Bridge Anchors — New server setup script
# Usage: bash docker/setup_server.sh [code_dir] [data_dir]
#
# Pulls the Docker image and launches an interactive container
# with code and data mounted as volumes.

set -euo pipefail

IMAGE="shiwonkim/bridge-anchors:v2"

# Defaults: code = current repo root, data = code/data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${1:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${2:-${CODE_DIR}/data}"

echo "=== Bridge Anchors Server Setup ==="
echo "Image:    ${IMAGE}"
echo "Code dir: ${CODE_DIR}"
echo "Data dir: ${DATA_DIR}"
echo ""

# Verify directories exist
if [ ! -d "$CODE_DIR" ]; then
    echo "ERROR: Code directory not found: ${CODE_DIR}"
    echo "Usage: bash docker/setup_server.sh /path/to/bridge-anchors /path/to/data"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "WARNING: Data directory not found: ${DATA_DIR}"
    echo "Container will start but training won't work without data."
    echo ""
fi

# Check Docker and GPU support
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|Default Runtime.*nvidia"; then
    echo "WARNING: NVIDIA container runtime may not be configured."
    echo "Make sure nvidia-container-toolkit is installed."
    echo ""
fi

# Pull image
echo "Pulling ${IMAGE}..."
docker pull "${IMAGE}"
echo ""

# Quick GPU sanity check
echo "Running GPU sanity check..."
docker run --gpus all --rm "${IMAGE}" python -c \
    "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected!')"
echo ""

# Launch interactive container
echo "Launching interactive container..."
echo "  (Ctrl+D or 'exit' to leave)"
echo ""

docker run --gpus all -it --shm-size=16g \
    --name bridge-anchors \
    -v "${CODE_DIR}":/workspace/bridge-anchors \
    -v "${DATA_DIR}":/workspace/bridge-anchors/data \
    -e PYTHONPATH=/workspace/bridge-anchors \
    -w /workspace/bridge-anchors \
    "${IMAGE}" bash
