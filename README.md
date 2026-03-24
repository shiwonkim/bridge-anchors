# Bridge Anchors: Cross-Modal Alignment for Independently Trained Unimodal Encoders

A parameter-efficient method for aligning frozen unimodal encoders using learnable anchor points instead of projection layers.

## Overview

Cross-modal retrieval typically requires either joint training of encoders or learning a projection layer to map one embedding space into another. Both approaches have drawbacks: joint training is expensive, and projection layers scale linearly with the embedding dimension.

**Bridge Anchors** takes a different approach. Instead of learning a transformation between two embedding spaces, we learn a set of *K* anchor points in each modality's space. Each embedding is then represented by its cosine similarity profile to these anchors, producing a *K*-dimensional vector. Because anchors in both modalities are trained jointly via contrastive loss, the resulting similarity profiles are directly comparable across modalities — no explicit space transformation is needed.

This enables cross-modal comparison in a shared anchor-similarity space with significantly fewer learnable parameters than projection-based methods. The method also extends to **token-level alignment**, where patch tokens from vision transformers are individually measured against anchors before aggregation, capturing richer spatial information.

## Method

Given frozen image encoder *f* and text encoder *g*, Bridge Anchors learns anchor matrices **A_img** ∈ ℝ^(K×d) and **A_txt** ∈ ℝ^(K×d):

```
bridge(img) = L2_normalize( cos_sim(f(img), A_img) )  →  (B, K)
bridge(txt) = L2_normalize( cos_sim(g(txt), A_txt) )  →  (B, K)
```

Both outputs live in the same *K*-dimensional space. Training minimizes InfoNCE loss on paired image-text embeddings.

**Token-Level Bridge Anchors** extends this to operate on patch token sequences. For an image with *T* tokens (CLS + patches), per-token anchor similarities are computed and then aggregated via mean pooling:

```
token_sims = cos_sim(tokens, A_img)   →  (B, T, K)
bridge(img) = L2_normalize( mean(token_sims, dim=T) )  →  (B, K)
```

## Project Structure

```
bridge-anchors/
├── src/
│   ├── models/
│   │   ├── bridge_anchors.py          # BridgeAnchorAligner
│   │   ├── token_bridge_anchors.py    # TokenBridgeAnchorAligner
│   │   ├── baselines.py               # LinearProjection, MLPProjection, FixedRelativeRep
│   │   └── losses.py                  # InfoNCE, load balancing, per-anchor contrastive
│   ├── data/
│   │   ├── extract_embeddings.py      # Pre-extract encoder outputs
│   │   ├── coco_dataset.py            # MSCOCO paired embedding dataset
│   │   ├── chunked_token_dataset.py   # Chunked token-level dataset for large-scale
│   │   └── eval_datasets.py           # Flickr30k, ImageNet eval loaders
│   ├── eval/
│   │   ├── retrieval.py               # Image-text retrieval (R@1, R@5, R@10)
│   │   ├── zeroshot.py                # ImageNet zero-shot classification
│   │   └── anchor_analysis.py         # Learned anchor analysis
│   └── train.py                       # Training entry point
├── scripts/
│   ├── extract_token_embeddings.py    # Token-level embedding extraction
│   └── extract_token_embeddings_full.py  # Full-scale chunked extraction
├── configs/
│   └── default.yaml                   # Default hyperparameters
├── docker/                            # Docker environment (see Setup)
├── IMPLEMENTATION.md                  # Detailed implementation specification
└── CLAUDE.md                          # Development instructions
```

## Setup

### Option 1: Docker (recommended)

```bash
# Pull the pre-built environment image
docker pull shiwonkim/bridge-anchors:v1

# Run with code and data mounted
docker run --gpus all -it --shm-size=16g \
  -v $(pwd):/workspace/bridge-anchors \
  -v /path/to/data:/workspace/bridge-anchors/data \
  -e PYTHONPATH=/workspace/bridge-anchors \
  -w /workspace/bridge-anchors \
  shiwonkim/bridge-anchors:v1 bash
```

See [`docker/README.md`](docker/README.md) for details.

### Option 2: Manual

Requires Python 3.10+, PyTorch 2.4.1 with CUDA support.

```bash
# Install PyTorch (adjust CUDA version to match your driver)
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r docker/requirements.txt
```

## Quick Start

```bash
# 1. Extract embeddings from frozen encoders (run once)
python src/data/extract_embeddings.py --dataset coco --split train
python src/data/extract_embeddings.py --dataset flickr30k

# 2. Train Bridge Anchors (K=128)
python -m src.train --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128

# 3. Train Token-Level Bridge Anchors
python -m src.train --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --token-level --token-pool mean

# 4. Evaluate
python -m src.eval.retrieval --checkpoint results/checkpoints/default/best.pt
python -m src.eval.zeroshot --checkpoint results/checkpoints/default/best.pt

# 5. Analyze learned anchors
python -m src.eval.anchor_analysis --checkpoint results/checkpoints/default/best.pt
```

## Encoders

| Modality | Model | Embedding Dim | Source |
|----------|-------|---------------|--------|
| Vision | DINOv2 ViT-B/14 | 768 | `torch.hub` (facebookresearch/dinov2) |
| Language | all-mpnet-base-v2 | 768 | `sentence-transformers` |

Both encoders are frozen. Training operates entirely on pre-extracted embeddings.

## References

- **Relative Representations** — Moschella et al., "Relative representations enable zero-shot latent space communication", ICLR 2023. Uses fixed anchors from data for same-modality model stitching; Bridge Anchors extends this with learnable anchors for cross-modal alignment.
- **Modality Gap** — Liang et al., "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning", NeurIPS 2022.
- **LiT** — Zhai et al., "LiT: Zero-Shot Transfer with Locked-image text Tuning", CVPR 2022. Locked-image tuning with frozen vision encoders.
