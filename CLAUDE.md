# Bridge Anchors: Parameter-Efficient Cross-Modal Alignment

## Project Overview

Research project implementing **Bridge Anchors**, a parameter-efficient method for aligning independently trained unimodal encoders (e.g., DINOv2 vision + sentence-transformers text) using learnable anchor points instead of projection layers.

**Core idea**: Instead of learning a transformation function (projection) between two embedding spaces, we learn *measurement reference points* (anchors) in each space. Each embedding is converted to a K-dimensional "distance profile" (cosine similarities to K anchors), making cross-modal comparison possible without any space transformation.

**Key distinction from Relative Representations (Moschella et al., ICLR 2023)**: Relative Representations use fixed anchors selected from data and were designed for same-modality model stitching (isometry assumption). Our method uses learnable anchors for cross-modal alignment of independently trained encoders (no isometry assumption). Anchors are free vectors, not constrained to the data manifold.

## Tech Stack
- Python 3.10+
- PyTorch 2.x
- HuggingFace transformers + sentence-transformers
- DINOv2 (torch.hub)
- Datasets: MSCOCO Captions (train), Flickr30k (eval), ImageNet (eval)

## Project Structure
```
bridge-anchors/
├── CLAUDE.md                 # This file
├── IMPLEMENTATION.md         # Detailed implementation spec
├── configs/
│   └── default.yaml          # Hyperparameters
├── data/
│   ├── embeddings/           # Pre-extracted encoder embeddings (.pt files)
│   └── datasets/             # Raw datasets (gitignored)
├── src/
│   ├── models/
│   │   ├── bridge_anchors.py # Core model: BridgeAnchorAligner
│   │   ├── baselines.py      # LinearProjection, MLPProjection, FixedRelativeRep
│   │   └── losses.py         # InfoNCE loss
│   ├── data/
│   │   ├── extract_embeddings.py  # Pre-extract & save encoder outputs
│   │   ├── coco_dataset.py        # MSCOCO paired embedding dataset
│   │   └── eval_datasets.py       # Flickr30k, ImageNet eval loaders
│   ├── eval/
│   │   ├── retrieval.py      # Image-text retrieval (R@1, R@5, R@10)
│   │   ├── zeroshot.py       # ImageNet zero-shot classification
│   │   └── anchor_analysis.py # Analyze learned anchor positions (Direction A)
│   └── utils/
│       ├── config.py
│       └── logging.py
├── scripts/
│   ├── extract_embeddings.sh
│   ├── train.sh
│   └── eval.sh
├── experiments/              # Experiment configs and results
│   ├── exp_a_main/
│   ├── exp_b_k_ablation/
│   ├── exp_c_data_efficiency/
│   └── exp_d_fixed_vs_learnable/
└── results/                  # Saved metrics, plots
```

## Key Commands
- Extract embeddings: `python src/data/extract_embeddings.py --dataset coco --split train`
- Train: `python -m src.train --config configs/default.yaml`
- Evaluate retrieval: `python -m src.eval.retrieval --checkpoint path/to/ckpt`
- Evaluate zero-shot: `python -m src.eval.zeroshot --checkpoint path/to/ckpt`
- Analyze anchors: `python -m src.eval.anchor_analysis --checkpoint path/to/ckpt`

## Code Style
- Type hints on all function signatures
- Docstrings for all public functions (Google style)
- Use dataclasses for configs
- PyTorch naming: snake_case for functions/variables, CamelCase for classes
- Keep modules focused: one model class per file for core models
- Use `torch.no_grad()` for all eval/embedding extraction

## Important Design Decisions
1. **Embeddings are pre-extracted and saved as .pt files** — encoders are frozen, so we never need to run them during training. This makes training extremely fast.
2. **Training loop operates on embedding tensors only** — no image/text processing during training.
3. **Cosine similarity for anchor distances** — matching the relative representation convention.
4. **InfoNCE loss with temperature 0.07** — standard for contrastive alignment.
5. **L2 normalization on bridged representations** — ensures cosine similarity is well-behaved.

## GPU Environment
- Single GPU: A40 (48GB) or Quadro RTX 8000 (48GB)
- All encoder inference (embedding extraction) fits easily in memory
- Training is on pre-extracted embeddings, very lightweight

## Environment Setup
- **Conda env**: `bridge-anchors` — activate with `conda activate bridge-anchors`
- **Python**: 3.10.20
- **PyTorch**: 2.4.1+cu118 (CUDA 11.8, cuDNN 90100)
- **GPU**: 2× NVIDIA A40 (44.6 GB each)
- **CUDA driver**: 470.256.02 (max CUDA 11.4, but bundled cu118 runtime works)
- **Key deps**: transformers 5.3.0, sentence-transformers 5.3.0, numpy 1.26.4
- Always activate this environment before running any commands: `eval "$(conda shell.bash hook)" && conda activate bridge-anchors`

## Development Process
- After completing each task, always update PROJECT_LOG.md with what was done, key decisions, and any issues. Keep entries concise but informative.

## Current Phase
Phase 1: Base implementation of Bridge Anchors + baselines + evaluation pipeline.
Future: Direction B (structural regularization on anchors) and Direction C (residual combination with fixed relative rep).
