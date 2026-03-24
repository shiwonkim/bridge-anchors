# Bridge Anchors — Environment-only Docker image
#
# Base: PyTorch 2.4.1 + CUDA 11.8 + cuDNN 9 (matches current conda env exactly)
# Note: CUDA 11.8 runtime works on hosts with CUDA 12.x drivers (backward compatible).
# Tested on: Quadro RTX 8000 with CUDA 12.4 driver.
#
# Code is NOT included — mount via volume at runtime.
# Data is NOT included — mount via volume at runtime.

FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /workspace

# Install pip dependencies (torch/torchvision already in base image)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install Node.js 20 for Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code

# Verify key packages
RUN python -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); \
import sentence_transformers; \
import transformers; \
import numpy; \
import matplotlib; \
import sklearn; \
import umap; \
print('All key packages OK')"

# Default to bash
CMD ["bash"]
