"""Main training script for Bridge Anchors and baselines.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --model linear_projection
    python -m src.train --config configs/default.yaml --num-samples 5000 --seed 1
    python -m src.train --config configs/default.yaml --img-input tokens --chunked
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.coco_dataset import PairedEmbeddingDataset
from src.data.eval_datasets import Flickr30kEmbeddings
from src.eval.retrieval import evaluate_retrieval
from src.models.baselines import FixedRelativeRep, LinearProjection, MLPProjection
from src.models.bridge_anchors import BridgeAnchorAligner
from src.models.freeze_align import FreezeAlignProjector
from src.models.losses import (
    anchor_isometry_loss,
    anchor_orthogonality_loss,
    hierarchical_attention_diversity_loss,
    info_nce_loss,
    load_balancing_loss,
    per_anchor_contrastive_loss,
    per_anchor_info_nce_loss,
    reconstruction_loss,
    token_matching_loss,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set RNG seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN — small perf cost but guarantees reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu: int | None) -> torch.device:
    """Return a torch device from an optional GPU index."""
    if gpu is not None:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the config dict (in-place)."""
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.num_anchors is not None:
        cfg["model"]["num_anchors"] = args.num_anchors
    if getattr(args, "dim_img", None) is not None:
        cfg["model"]["dim_img"] = args.dim_img
    if getattr(args, "dim_txt", None) is not None:
        cfg["model"]["dim_txt"] = args.dim_txt
    if args.init_method is not None:
        cfg["model"]["init_method"] = args.init_method
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_samples is not None:
        cfg["data"]["num_samples"] = args.num_samples
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.ortho_lambda is not None:
        cfg["training"]["ortho_lambda"] = args.ortho_lambda
    if args.lb_lambda is not None:
        cfg["training"]["lb_lambda"] = args.lb_lambda
    if args.pa_lambda is not None:
        cfg["training"]["pa_lambda"] = args.pa_lambda
    if args.ca_lambda is not None:
        cfg["training"]["ca_lambda"] = args.ca_lambda
    if args.iso_lambda is not None:
        cfg["training"]["iso_lambda"] = args.iso_lambda
    if args.token_match_lambda is not None:
        cfg["training"]["token_match_lambda"] = args.token_match_lambda
    if getattr(args, "diversity_lambda", None) is not None:
        cfg["training"]["diversity_lambda"] = args.diversity_lambda
    if getattr(args, "diversity_sigma", None) is not None:
        cfg["training"]["diversity_sigma"] = args.diversity_sigma
    if getattr(args, "recon_lambda", None) is not None:
        cfg["training"]["recon_lambda"] = args.recon_lambda
    if args.top_k is not None:
        cfg["model"]["top_k"] = args.top_k
    if args.experiment_name is not None:
        cfg["logging"]["experiment_name"] = args.experiment_name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    """Build training and evaluation data for both CLS and token-level modes.

    Returns a dict with keys:
        train_loader: DataLoader or None (when using chunked loading)
        train_chunked: ChunkedTokenDataset or None
        flickr_img: Tensor of Flickr30k image embeddings (or None)
        flickr_txt: Tensor of Flickr30k text embeddings (or None)
        flickr_txt_mask: Tensor of Flickr30k text masks (or None)
        steps_per_epoch: int
        train_dataset: PairedEmbeddingDataset or None (for CLS path)
        val_dataset: PairedEmbeddingDataset or None (for CLS path)
    """
    img_input = args.img_input
    txt_input = args.txt_input
    uses_tokens = (img_input == "tokens" or txt_input == "tokens")
    batch_size = cfg["training"]["batch_size"]

    emb_dir = getattr(args, "embedding_dir", None)
    if emb_dir is not None:
        # Custom embedding directory: tokens and CLS in same dir
        token_dir = PROJECT_ROOT / emb_dir
        cls_dir = PROJECT_ROOT / emb_dir
    else:
        token_dir = PROJECT_ROOT / "data" / "embeddings" / "all_tokens"
        cls_dir = PROJECT_ROOT / "data" / "embeddings" / "cls"

    result: dict[str, Any] = {
        "train_loader": None,
        "train_chunked": None,
        "flickr_img": None,
        "flickr_txt": None,
        "flickr_txt_mask": None,
        "flickr_img_cls_attn": None,
        "flickr_txt_cls_attn": None,
        "steps_per_epoch": 0,
        "train_dataset": None,
        "val_dataset": None,
    }

    if uses_tokens:
        # --- Token-level path ---
        txt_token_level = (txt_input == "tokens")
        chunked = getattr(args, "chunked", False)

        # Flickr30k eval
        if img_input == "tokens":
            result["flickr_img"] = torch.load(
                token_dir / "flickr30k_test_img.pt", weights_only=True,
            ).float()
        else:
            result["flickr_img"] = torch.load(
                cls_dir / "flickr30k_test_img.pt", weights_only=True,
            ).float()

        if txt_input == "tokens":
            result["flickr_txt"] = torch.load(
                token_dir / "flickr30k_test_txt_tokens.pt", weights_only=True,
            ).float()
            result["flickr_txt_mask"] = torch.load(
                token_dir / "flickr30k_test_txt_mask.pt", weights_only=True,
            )
        else:
            result["flickr_txt"] = torch.load(
                token_dir / "flickr30k_test_txt.pt", weights_only=True,
            ).float()

        logger.info(
            "Flickr30k: img %s, txt %s",
            tuple(result["flickr_img"].shape),
            tuple(result["flickr_txt"].shape),
        )

        # Load CLS attention priors if requested
        cls_attn_prior = getattr(args, "cls_attn_prior", "none")
        needs_cls_attn = (
            cls_attn_prior != "none"
            or getattr(args, "attn_mask_groups", None) is not None
            or getattr(args, "expert_soft_mask", False)
            or (getattr(args, "diversity_lambda", None) or 0) > 0
        )
        if needs_cls_attn:
            flickr_img_attn_path = token_dir / "flickr30k_test_img_cls_attn.pt"
            flickr_txt_attn_path = token_dir / "flickr30k_test_txt_cls_attn.pt"
            if flickr_img_attn_path.exists():
                result["flickr_img_cls_attn"] = torch.load(
                    flickr_img_attn_path, weights_only=True,
                ).float()
                logger.info("Flickr img CLS attn: %s",
                            tuple(result["flickr_img_cls_attn"].shape))
            if flickr_txt_attn_path.exists():
                result["flickr_txt_cls_attn"] = torch.load(
                    flickr_txt_attn_path, weights_only=True,
                ).float()
                logger.info("Flickr txt CLS attn: %s",
                            tuple(result["flickr_txt_cls_attn"].shape))

        # Training data
        if img_input == "tokens" and chunked:
            from src.data.chunked_token_dataset import ChunkedTokenDataset

            txt_path = token_dir / "coco_train_txt.pt"
            chunked_kwargs: dict[str, Any] = {}
            if needs_cls_attn:
                chunked_kwargs["img_cls_attn_path"] = token_dir / "coco_train_img_cls_attn.pt"
                chunked_kwargs["txt_cls_attn_path"] = token_dir / "coco_train_txt_cls_attn.pt"
            train_chunked = ChunkedTokenDataset(
                chunk_dir=token_dir, text_emb_path=txt_path,
                batch_size=batch_size, seed=seed, split="train",
                text_token_level=txt_token_level,
                **chunked_kwargs,
            )
            result["train_chunked"] = train_chunked
            result["steps_per_epoch"] = train_chunked.n_batches_approx
            logger.info(
                "Chunked training: ~%d batches/epoch (bs=%d, txt_tokens=%s)",
                result["steps_per_epoch"], batch_size, txt_token_level,
            )
        else:
            from torch.utils.data import TensorDataset

            if img_input == "tokens":
                train_img = torch.load(
                    token_dir / "coco_train_10k_img.pt", weights_only=True,
                )
                train_txt_cls = torch.load(
                    token_dir / "coco_train_10k_txt.pt", weights_only=True,
                )
            else:
                train_img = torch.load(
                    cls_dir / "coco_train_img.pt", weights_only=True,
                )
                train_txt_cls = torch.load(
                    cls_dir / "coco_train_txt.pt", weights_only=True,
                )

            logger.info(
                "Train: img %s, txt %s",
                tuple(train_img.shape), tuple(train_txt_cls.shape),
            )

            n = train_img.shape[0]
            n_val = max(1, int(n * 0.05))
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=gen)
            train_idx = perm[n_val:]

            train_ds = TensorDataset(train_img[train_idx], train_txt_cls[train_idx])
            result["train_loader"] = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
                generator=torch.Generator().manual_seed(seed),
            )
            result["steps_per_epoch"] = len(result["train_loader"])
            logger.info(
                "Training: %d pairs (%d batches, bs=%d)",
                len(train_ds), result["steps_per_epoch"], batch_size,
            )

    else:
        # --- CLS-only path ---
        train_dataset = PairedEmbeddingDataset(
            img_emb_path=PROJECT_ROOT / cfg["data"]["img_emb_path"],
            txt_emb_path=PROJECT_ROOT / cfg["data"]["txt_emb_path"],
            num_samples=cfg["data"]["num_samples"],
            seed=seed,
            split="train",
        )
        val_dataset = PairedEmbeddingDataset(
            img_emb_path=PROJECT_ROOT / cfg["data"]["img_emb_path"],
            txt_emb_path=PROJECT_ROOT / cfg["data"]["txt_emb_path"],
            seed=seed,
            split="val",
        )
        result["train_dataset"] = train_dataset
        result["val_dataset"] = val_dataset

        result["train_loader"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        result["steps_per_epoch"] = len(result["train_loader"])

        logger.info(
            "Train: %d pairs (%d batches) | Val: %d pairs",
            len(train_dataset), result["steps_per_epoch"], len(val_dataset),
        )

        # Flickr30k eval embeddings
        flickr_img_path = PROJECT_ROOT / cfg["eval"]["flickr_img_emb_path"]
        flickr_txt_path = PROJECT_ROOT / cfg["eval"]["flickr_txt_emb_path"]
        if flickr_img_path.exists() and flickr_txt_path.exists():
            flickr = Flickr30kEmbeddings(flickr_img_path, flickr_txt_path)
            fi, ft = flickr.get_all()
            result["flickr_img"] = fi
            result["flickr_txt"] = ft
        else:
            logger.warning(
                "Flickr30k embeddings not found — skipping retrieval evaluation. "
                "Run extract_embeddings.py --dataset flickr30k first."
            )

    return result


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    train_dataset: PairedEmbeddingDataset | None,
    device: torch.device,
    seed: int,
) -> torch.nn.Module:
    """Instantiate the model specified by config.

    Handles both CLS-only and token-level inputs via the same model classes.
    The img_input/txt_input mode is configured at init time.

    Args:
        cfg: Full config dict.
        args: Parsed CLI args (for img_input/txt_input/token_pool).
        train_dataset: Training dataset (needed for prototype init and
            FixedRelativeRep anchor selection). May be None for token-level.
        device: Target device.
        seed: Random seed (for anchor selection when train_dataset is None).

    Returns:
        Model moved to ``device``.
    """
    model_name = cfg["model"]["name"]
    dim_img = cfg["model"]["dim_img"]
    dim_txt = cfg["model"]["dim_txt"]

    if model_name == "bridge_anchors":
        num_anchors = cfg["model"]["num_anchors"]
        init_method = cfg["model"]["init_method"]
        top_k = cfg["model"].get("top_k", 0)

        proto_img, proto_txt = None, None
        needs_data_init = init_method in ("prototype", "kmeans", "fps")

        # If train_dataset is None (chunked token path) but we need data
        # for init, load CLS embeddings as a temporary dataset.
        ds_for_init = train_dataset
        if ds_for_init is None and needs_data_init:
            cls_dir = PROJECT_ROOT / "data" / "embeddings" / "cls"
            logger.info(
                "Loading CLS embeddings for %s init (chunked token path)...",
                init_method,
            )
            ds_for_init = PairedEmbeddingDataset(
                img_emb_path=cls_dir / "coco_train_img.pt",
                txt_emb_path=cls_dir / "coco_train_txt.pt",
                seed=seed,
            )

        if ds_for_init is not None:
            if init_method == "prototype":
                proto_img, proto_txt = _compute_prototypes(
                    ds_for_init, num_anchors, seed,
                )
            elif init_method == "kmeans":
                proto_img, proto_txt = _compute_kmeans_centroids(
                    ds_for_init, num_anchors, seed,
                )
            elif init_method == "fps":
                proto_img, proto_txt = _compute_fps_anchors(
                    ds_for_init, num_anchors, seed,
                )

        model = BridgeAnchorAligner(
            dim_img=dim_img,
            dim_txt=dim_txt,
            num_anchors=num_anchors,
            init_method=init_method,
            proto_img=proto_img,
            proto_txt=proto_txt,
            top_k=top_k,
            token_pool=args.token_pool,
            pool_temperature=getattr(args, "pool_temperature", 0.1),
            learnable_tau=getattr(args, "learnable_tau", False),
            cls_attn_prior=getattr(args, "cls_attn_prior", "none"),
            cls_attn_beta=getattr(args, "cls_attn_beta", 1.0),
            group_taus=getattr(args, "group_taus", None),
            group_norm=getattr(args, "group_norm", False),
            group_gating=getattr(args, "group_gating", False),
            attn_mask_groups=getattr(args, "attn_mask_groups", None),
            projector_dim=getattr(args, "projector_dim", 0),
            stacked_anchors_dim=getattr(args, "stacked_anchors_dim", 0),
            profile_proj_dim=getattr(args, "profile_proj_dim", 0),
            cls_anchors=getattr(args, "cls_anchors", 0),
            num_experts=getattr(args, "num_experts", 1),
            expert_soft_mask=getattr(args, "expert_soft_mask", False),
            expert_k=getattr(args, "expert_k", 0),
            recon_loss=getattr(args, "recon_loss", False),
            img_input=args.img_input,
            txt_input=args.txt_input,
            ca_exclude_cls=getattr(args, "ca_exclude_cls", False),
            anchor_mediated=getattr(args, "anchor_mediated", False),
            selection_mode=getattr(args, "selection_mode", "soft"),
            am_cls_weight=getattr(args, "am_cls_weight", 0.0),
        )

    elif model_name == "linear_projection":
        model = LinearProjection(
            dim_img=dim_img, dim_txt=dim_txt,
            img_input=args.img_input, txt_input=args.txt_input,
        )

    elif model_name == "mlp_projection":
        model = MLPProjection(
            dim_img=dim_img,
            dim_txt=dim_txt,
            hidden_dim=cfg["baseline"]["mlp_hidden_dim"],
            img_input=args.img_input,
            txt_input=args.txt_input,
        )

    elif model_name == "freeze_align":
        model = FreezeAlignProjector(
            dim_img=dim_img, dim_txt=dim_txt,
            img_input=args.img_input, txt_input=args.txt_input,
        )

    elif model_name == "shared_anchor":
        from src.models.shared_anchor import SharedAnchorAligner
        model = SharedAnchorAligner(
            dim_img=dim_img,
            dim_txt=dim_txt,
            dim_shared=getattr(args, "dim_shared", 256),
            num_anchors=cfg["model"]["num_anchors"],
            hidden_dim=getattr(args, "hidden_dim", 256),
            projector_type=getattr(args, "projector_type", "mlp"),
            img_input=args.img_input,
            txt_input=args.txt_input,
        )

    elif model_name == "fixed_relative_rep":
        num_anchors = cfg["model"]["num_anchors"]
        if train_dataset is not None:
            init_method = cfg["model"]["init_method"]
            if init_method == "prototype":
                anchors_img, anchors_txt = _compute_prototypes(
                    train_dataset, num_anchors, seed,
                )
            else:
                anchors_img, anchors_txt = _select_fixed_anchors(
                    train_dataset, num_anchors, seed,
                )
        else:
            # Token-level path: load CLS embeddings for anchor selection
            cls_dir = PROJECT_ROOT / "data" / "embeddings" / "cls"
            img_data = torch.load(
                cls_dir / "coco_train_img.pt", weights_only=True,
            )
            txt_data = torch.load(
                cls_dir / "coco_train_txt.pt", weights_only=True,
            )
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(img_data.shape[0], generator=gen)[:num_anchors]
            anchors_img = img_data[idx]
            anchors_txt = txt_data[idx]
            del img_data, txt_data
        model = FixedRelativeRep(
            anchors_img=anchors_img, anchors_txt=anchors_txt,
            img_input=args.img_input, txt_input=args.txt_input,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | params: %s total, %s trainable | img=%s txt=%s",
        model_name, f"{n_params:,}", f"{n_train:,}",
        args.img_input, args.txt_input,
    )
    return model


def _compute_prototypes(
    dataset: PairedEmbeddingDataset,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K prototype anchors via K-means-style random partitioning.

    Randomly assigns all samples to K clusters and computes cluster means.
    This is a cheap approximation to K-means that gives reasonable
    prototypes for anchor initialisation.

    Returns:
        proto_img: (K, dim_img) image prototypes.
        proto_txt: (K, dim_txt) text prototypes.
    """
    img_embs, txt_embs = dataset.get_all()
    n = img_embs.shape[0]
    gen = torch.Generator().manual_seed(seed)
    assignments = torch.randint(0, k, (n,), generator=gen)

    proto_img = torch.zeros(k, img_embs.shape[1])
    proto_txt = torch.zeros(k, txt_embs.shape[1])
    for i in range(k):
        mask = assignments == i
        if mask.sum() == 0:
            # Empty cluster — fall back to a random sample
            idx = torch.randint(0, n, (1,), generator=gen).item()
            proto_img[i] = img_embs[idx]
            proto_txt[i] = txt_embs[idx]
        else:
            proto_img[i] = img_embs[mask].mean(dim=0)
            proto_txt[i] = txt_embs[mask].mean(dim=0)

    return proto_img, proto_txt


def _compute_kmeans_centroids(
    dataset: PairedEmbeddingDataset,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K anchor centroids via scikit-learn K-means clustering.

    Runs K-means independently on image and text embeddings, using the
    converged cluster centroids as anchor initialisation.

    Returns:
        centroids_img: (K, dim_img) image centroids.
        centroids_txt: (K, dim_txt) text centroids.
    """
    from sklearn.cluster import KMeans

    img_embs, txt_embs = dataset.get_all()

    logger.info("Running K-means (K=%d) on %d image embeddings...", k, img_embs.shape[0])
    km_img = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=100)
    km_img.fit(img_embs.numpy())

    logger.info("Running K-means (K=%d) on %d text embeddings...", k, txt_embs.shape[0])
    km_txt = KMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=100)
    km_txt.fit(txt_embs.numpy())

    centroids_img = torch.from_numpy(km_img.cluster_centers_).float()
    centroids_txt = torch.from_numpy(km_txt.cluster_centers_).float()

    return centroids_img, centroids_txt


def _compute_fps_anchors(
    dataset: PairedEmbeddingDataset,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select K anchor points via Farthest Point Sampling (FPS).

    Greedily selects points that maximize minimum distance to already
    selected points. Starts with the embedding closest to the mean.
    Runs independently for image and text embeddings.

    Returns:
        anchors_img: (K, dim_img) image anchor embeddings.
        anchors_txt: (K, dim_txt) text anchor embeddings.
    """
    def _fps(embs: torch.Tensor, k: int) -> torch.Tensor:
        embs_norm = torch.nn.functional.normalize(embs, dim=-1)
        n = embs_norm.shape[0]

        # Start with the point closest to the mean
        mean = embs_norm.mean(dim=0, keepdim=True)
        sims_to_mean = (embs_norm @ mean.T).squeeze()
        first_idx = sims_to_mean.argmax().item()

        selected = [first_idx]
        # min_dist[i] = max cosine sim from point i to any selected point
        # (we want to select the point with LOWEST max sim = farthest)
        min_sim = (embs_norm @ embs_norm[first_idx]).clone()  # (N,)

        for _ in range(k - 1):
            # Select the point with lowest similarity to any selected point
            # (= farthest from all selected points in cosine space)
            min_sim[selected] = float("inf")  # exclude already selected
            next_idx = min_sim.argmin().item()
            selected.append(next_idx)

            # Update min similarities
            new_sim = embs_norm @ embs_norm[next_idx]
            min_sim = torch.max(min_sim, new_sim)

        return embs[selected]

    img_embs, txt_embs = dataset.get_all()
    logger.info("Computing FPS anchors (K=%d) on %d embeddings...", k, img_embs.shape[0])

    anchors_img = _fps(img_embs, k)
    anchors_txt = _fps(txt_embs, k)

    return anchors_img, anchors_txt


def _select_fixed_anchors(
    dataset: PairedEmbeddingDataset,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select K random paired samples as fixed anchors.

    Returns:
        anchors_img: (K, dim_img) image anchor embeddings.
        anchors_txt: (K, dim_txt) text anchor embeddings.
    """
    img_embs, txt_embs = dataset.get_all()
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(img_embs.shape[0], generator=gen)[:k]
    return img_embs[indices], txt_embs[indices]


# ---------------------------------------------------------------------------
# Scheduler with linear warmup
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Cosine annealing with linear warmup.

    Args:
        optimizer: Optimiser instance.
        epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs (linear ramp from 0 to lr).
        steps_per_epoch: Number of optimiser steps per epoch.

    Returns:
        A scheduler that should be stepped once per optimiser step.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    if warmup_steps > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    return scheduler


# ---------------------------------------------------------------------------
# Weights & Biases logging
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Weights & Biases logger with graceful degradation.

    If wandb is not installed or is disabled (``WANDB_MODE=disabled``),
    training proceeds normally without logging.
    """

    def __init__(self, experiment_name: str, cfg: dict[str, Any]) -> None:
        self.run = None
        try:
            import wandb

            project = cfg.get("logging", {}).get(
                "wandb_project", "moa",
            )
            # Allow env-var override
            project = os.environ.get("WANDB_PROJECT", project)

            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=cfg,
                reinit=True,
            )
            logger.info(
                "wandb logging enabled (project=%s, run=%s).",
                project, self.run.name,
            )
        except ImportError:
            logger.warning(
                "wandb is not installed — metrics will not be logged. "
                "Install with: pip install wandb"
            )
        except Exception as e:
            logger.warning("wandb init failed: %s — continuing without logging.", e)

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of metrics at a given step."""
        if self.run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def set_summary(self, key: str, value: float | int | str) -> None:
        """Set a summary metric (shown in the runs table)."""
        if self.run is not None:
            self.run.summary[key] = value

    def close(self) -> None:
        """Finish the wandb run."""
        if self.run is not None:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    data_iter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    temperature: float,
    device: torch.device,
    grad_clip: float,
    ortho_lambda: float = 0.0,
    lb_lambda: float = 0.0,
    pa_lambda: float = 0.0,
    iso_lambda: float = 0.0,
    token_match_lambda: float = 0.0,
    ca_lambda: float = 0.0,
    diversity_lambda: float = 0.0,
    diversity_sigma: float = 0.2,
    diversity_modality: str = "both",
    recon_lambda: float = 0.0,
    num_experts: int = 1,
    *,
    is_chunked: bool = False,
    txt_token_level: bool = False,
    img_token_level: bool = False,
    strip_cls: bool = False,
) -> dict[str, float]:
    """Run one training epoch.

    Handles both CLS-only batches (img, txt) and token-level batches
    which may include text token tensors and masks.

    Args:
        data_iter: DataLoader or chunked epoch iterator.
        is_chunked: True when iterating from ChunkedTokenDataset (batches
            are already on device and may contain txt tokens/masks).
        txt_token_level: True when text input uses token-level embeddings.
        img_token_level: True when image input uses token-level embeddings.
        token_match_lambda: Weight for token-level matching loss (only
            active when both img and txt are token-level).

    Returns:
        Dict with ``loss`` (epoch average), ``lr`` (final step LR),
        and optionally aux loss values if their lambdas > 0.
    """
    model.train()
    total_loss = 0.0
    total_ortho = 0.0
    total_lb = 0.0
    total_pa = 0.0
    total_iso = 0.0
    total_token_match = 0.0
    total_ca = 0.0
    total_diversity = 0.0
    total_recon = 0.0
    # Per-expert monitoring totals (HME mode only)
    per_expert_recon: list[float] = []
    per_expert_infonce: list[float] = []
    per_expert_attn_entropy: list[float] = []
    num_batches = 0

    has_anchors = hasattr(model, "anchors_img")
    is_anchor_mediated = getattr(model, "anchor_mediated", False)
    need_raw_sims = (lb_lambda > 0 or pa_lambda > 0) and has_anchors and not is_anchor_mediated
    need_token_sims = (
        token_match_lambda > 0
        and has_anchors
        and img_token_level
        and txt_token_level
    )
    need_cls_and_ca = (
        ca_lambda > 0
        and has_anchors
        and (img_token_level or txt_token_level)
    )
    need_expert_attns = diversity_lambda > 0 and num_experts > 1
    need_expert_profiles = recon_lambda > 0 and num_experts > 1
    need_both_aux = need_expert_attns and need_expert_profiles

    for batch in data_iter:
        # --- Unpack batch ---
        img_cls_attn, txt_cls_attn = None, None
        if is_chunked:
            if txt_token_level:
                img_emb, txt_emb, txt_tok, txt_mask, img_cls_attn, txt_cls_attn = batch
            else:
                img_emb, txt_emb, img_cls_attn, txt_cls_attn = batch
                txt_tok, txt_mask = None, None
        else:
            img_emb, txt_emb = batch
            img_emb = img_emb.to(device, non_blocking=True)
            txt_emb = txt_emb.to(device, non_blocking=True)
            txt_tok, txt_mask = None, None

        # Optional: strip CLS token from token-level inputs
        if strip_cls:
            if img_token_level and img_emb.dim() == 3:
                img_emb = img_emb[:, 1:, :]
            if txt_token_level and txt_tok is not None and txt_tok.dim() == 3:
                txt_tok = txt_tok[:, 1:, :]
                if txt_mask is not None:
                    txt_mask = txt_mask[:, 1:]

        # Select text input: use token embeddings when available
        txt_for_model = txt_tok if txt_token_level and txt_tok is not None else txt_emb
        mask_for_model = txt_mask if txt_token_level else None

        optimizer.zero_grad()

        # --- Forward ---
        fwd_kwargs: dict[str, Any] = {}
        if mask_for_model is not None:
            fwd_kwargs["txt_mask"] = mask_for_model
        if img_cls_attn is not None:
            fwd_kwargs["img_cls_attn"] = img_cls_attn
        if txt_cls_attn is not None:
            fwd_kwargs["txt_cls_attn"] = txt_cls_attn

        if is_anchor_mediated:
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            if len(out) == 4:
                p_img, p_txt, b_cls_img, b_cls_txt = out
            else:
                p_img, p_txt = out
                b_cls_img, b_cls_txt = None, None
        elif need_cls_and_ca:
            fwd_kwargs["return_cls_and_ca"] = True
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            b_cls_img, b_cls_txt, b_ca_img, b_ca_txt = out
            b_img, b_txt = b_cls_img, b_cls_txt  # primary profiles for aux losses
        elif need_raw_sims:
            fwd_kwargs["return_raw_sims"] = True
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            b_img, b_txt, raw_img, raw_txt = out
        elif need_token_sims:
            fwd_kwargs["return_token_sims"] = True
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            b_img, b_txt, tok_sims_img, tok_sims_txt = out
        elif need_expert_attns:
            fwd_kwargs["return_expert_attns"] = True
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            b_img, b_txt, expert_attns_img, expert_attns_txt = out
        elif need_expert_profiles:
            fwd_kwargs["return_expert_profiles"] = True
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            (
                b_img, b_txt,
                expert_profiles_img, expert_profiles_txt,
                expert_attns_img, expert_attns_txt,
            ) = out
        else:
            out = model(img_emb, txt_for_model, **fwd_kwargs)
            b_img, b_txt = out

        # Use model's learnable temperature if available (e.g. FreezeAlign)
        effective_temp = model.temp if hasattr(model, "temp") else temperature
        if is_anchor_mediated:
            am_cls_w = getattr(model, "am_cls_weight", 0.0)
            loss = per_anchor_info_nce_loss(
                p_img, p_txt, temperature=effective_temp,
                b_cls_img=b_cls_img, b_cls_txt=b_cls_txt,
                cls_weight=am_cls_w,
            )
        else:
            loss = info_nce_loss(b_img, b_txt, temperature=effective_temp)

        ortho_loss_val = 0.0
        if ortho_lambda > 0 and has_anchors:
            ortho_loss = anchor_orthogonality_loss(
                model.anchors_img, model.anchors_txt,
            )
            loss = loss + ortho_lambda * ortho_loss
            ortho_loss_val = ortho_loss.item()

        iso_loss_val = 0.0
        if iso_lambda > 0 and has_anchors:
            iso_loss = anchor_isometry_loss(
                model.anchors_img, model.anchors_txt,
            )
            loss = loss + iso_lambda * iso_loss
            iso_loss_val = iso_loss.item()

        lb_loss_val = 0.0
        if lb_lambda > 0 and need_raw_sims:
            lb_loss = load_balancing_loss(raw_img, raw_txt)
            loss = loss + lb_lambda * lb_loss
            lb_loss_val = lb_loss.item()

        pa_loss_val = 0.0
        if pa_lambda > 0 and need_raw_sims:
            pa_loss = per_anchor_contrastive_loss(raw_img, raw_txt)
            loss = loss + pa_lambda * pa_loss
            pa_loss_val = pa_loss.item()

        ca_loss_val = 0.0
        if need_cls_and_ca:
            ca_loss = info_nce_loss(b_ca_img, b_ca_txt, temperature=effective_temp)
            loss = loss + ca_lambda * ca_loss
            ca_loss_val = ca_loss.item()

        token_match_loss_val = 0.0
        if need_token_sims and tok_sims_img is not None and tok_sims_txt is not None:
            tm_loss = token_matching_loss(
                tok_sims_img, tok_sims_txt, txt_mask=mask_for_model,
            )
            loss = loss + token_match_lambda * tm_loss
            token_match_loss_val = tm_loss.item()

        diversity_loss_val = 0.0
        if need_expert_attns:
            use_img = diversity_modality in ("both", "img_only")
            use_txt = diversity_modality in ("both", "txt_only")

            div_img = None
            div_txt = None
            if use_img and img_cls_attn is not None:
                div_img = hierarchical_attention_diversity_loss(
                    expert_attns_img, img_cls_attn, num_experts=num_experts,
                    sigma=diversity_sigma, mask=None,
                )
            if use_txt and txt_cls_attn is not None and len(expert_attns_txt) > 0:
                div_txt = hierarchical_attention_diversity_loss(
                    expert_attns_txt, txt_cls_attn, num_experts=num_experts,
                    sigma=diversity_sigma, mask=mask_for_model,
                )

            if div_img is not None and div_txt is not None:
                div_loss = (div_img + div_txt) / 2
            elif div_img is not None:
                div_loss = div_img
            elif div_txt is not None:
                div_loss = div_txt
            else:
                div_loss = None

            if div_loss is not None:
                loss = loss + diversity_lambda * div_loss
                diversity_loss_val = div_loss.item()

        recon_loss_val = 0.0
        if need_expert_profiles and hasattr(model, "decoders_img"):
            # Targets: detached, L2-normalized CLS embeddings of original encoder
            if img_emb.dim() == 3:
                target_img = F.normalize(img_emb[:, 0, :], dim=-1).detach()
            else:
                target_img = F.normalize(img_emb, dim=-1).detach()
            txt_target_src = txt_for_model
            if txt_target_src.dim() == 3:
                target_txt = F.normalize(txt_target_src[:, 0, :], dim=-1).detach()
            else:
                target_txt = F.normalize(txt_target_src, dim=-1).detach()

            r_loss = reconstruction_loss(
                expert_profiles_img, expert_profiles_txt,
                model.decoders_img, model.decoders_txt,
                target_img, target_txt,
            )
            loss = loss + recon_lambda * r_loss
            recon_loss_val = r_loss.item()

        # --- Per-expert monitoring metrics (no gradient impact) ---
        if need_expert_profiles:
            with torch.no_grad():
                G = len(expert_profiles_img)
                if not per_expert_recon:
                    per_expert_recon = [0.0] * G
                    per_expert_infonce = [0.0] * G
                    per_expert_attn_entropy = [0.0] * G

                # Targets for monitoring recon (compute even if recon_lambda=0)
                if img_emb.dim() == 3:
                    target_img_mon = F.normalize(img_emb[:, 0, :], dim=-1)
                else:
                    target_img_mon = F.normalize(img_emb, dim=-1)
                txt_src_mon = txt_for_model
                if txt_src_mon.dim() == 3:
                    target_txt_mon = F.normalize(txt_src_mon[:, 0, :], dim=-1)
                else:
                    target_txt_mon = F.normalize(txt_src_mon, dim=-1)

                has_decoders = hasattr(model, "decoders_img")
                for g in range(G):
                    # Per-expert recon
                    if has_decoders:
                        r_img_g = model.decoders_img[g](expert_profiles_img[g])
                        r_txt_g = model.decoders_txt[g](expert_profiles_txt[g])
                        rec_g = (
                            F.mse_loss(r_img_g, target_img_mon)
                            + F.mse_loss(r_txt_g, target_txt_mon)
                        ) / 2
                        per_expert_recon[g] += rec_g.item()

                    # Per-expert InfoNCE on this expert's sub-profile
                    p_img_g = F.normalize(expert_profiles_img[g], dim=-1)
                    p_txt_g = F.normalize(expert_profiles_txt[g], dim=-1)
                    nce_g = info_nce_loss(p_img_g, p_txt_g, temperature=effective_temp)
                    per_expert_infonce[g] += nce_g.item()

                    # Per-expert attention entropy
                    if g < len(expert_attns_img) and expert_attns_img[g] is not None:
                        # attn shape: (B, T, K_g) — distribution over T per anchor
                        attn = expert_attns_img[g]
                        ent = -(attn * torch.log(attn + 1e-8)).sum(dim=1).mean()
                        per_expert_attn_entropy[g] += ent.item()

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_ortho += ortho_loss_val
        total_lb += lb_loss_val
        total_pa += pa_loss_val
        total_iso += iso_loss_val
        total_token_match += token_match_loss_val
        total_ca += ca_loss_val
        total_diversity += diversity_loss_val
        total_recon += recon_loss_val
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    current_lr = scheduler.get_last_lr()[0]
    result = {"loss": avg_loss, "lr": current_lr}
    if ortho_lambda > 0:
        result["ortho_loss"] = total_ortho / max(num_batches, 1)
    if iso_lambda > 0:
        result["iso_loss"] = total_iso / max(num_batches, 1)
    if lb_lambda > 0:
        result["lb_loss"] = total_lb / max(num_batches, 1)
    if pa_lambda > 0:
        result["pa_loss"] = total_pa / max(num_batches, 1)
    if token_match_lambda > 0:
        result["token_match_loss"] = total_token_match / max(num_batches, 1)
    if ca_lambda > 0:
        result["ca_loss"] = total_ca / max(num_batches, 1)
    if diversity_lambda > 0:
        result["diversity_loss"] = total_diversity / max(num_batches, 1)
    if recon_lambda > 0:
        result["recon_loss"] = total_recon / max(num_batches, 1)
    # Per-expert monitoring metrics
    if per_expert_infonce:
        nb = max(num_batches, 1)
        for g in range(len(per_expert_infonce)):
            result[f"infonce_expert_{g}"] = per_expert_infonce[g] / nb
            result[f"recon_loss_expert_{g}"] = per_expert_recon[g] / nb
            result[f"attn_entropy_expert_{g}"] = per_expert_attn_entropy[g] / nb
    return result


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    cfg: dict[str, Any],
    path: Path,
) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


@torch.no_grad()
def _compute_val_loss(
    model: torch.nn.Module,
    val_dataset: PairedEmbeddingDataset,
    temperature: float,
    device: torch.device,
) -> float:
    """Compute InfoNCE loss on the validation split."""
    model.eval()
    img_embs, txt_embs = val_dataset.get_all()
    img_embs = img_embs.to(device)
    txt_embs = txt_embs.to(device)

    # Process in chunks to avoid OOM on large val sets
    chunk_size = 2048
    total_loss = 0.0
    n_chunks = 0
    for i in range(0, img_embs.shape[0], chunk_size):
        b_img, b_txt = model(img_embs[i:i + chunk_size], txt_embs[i:i + chunk_size])
        loss = info_nce_loss(b_img, b_txt, temperature=temperature)
        total_loss += loss.item()
        n_chunks += 1

    return total_loss / max(n_chunks, 1)


def _log_retrieval(metrics: dict[str, float], epoch: int) -> None:
    """Log retrieval metrics at a given epoch."""
    logger.info(
        "Epoch %d retrieval | i2t R@1=%.1f R@5=%.1f R@10=%.1f | "
        "t2i R@1=%.1f R@5=%.1f R@10=%.1f | mR=%.1f",
        epoch,
        metrics["i2t_r1"], metrics["i2t_r5"], metrics["i2t_r10"],
        metrics["t2i_r1"], metrics["t2i_r5"], metrics["t2i_r10"],
        metrics["mean_recall"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = resolve_device(getattr(args, "gpu", None))
    uses_tokens = (args.img_input == "tokens" or args.txt_input == "tokens")
    txt_token_level = (args.txt_input == "tokens")
    model_name = cfg["model"]["name"]

    logger.info(
        "Device: %s | Seed: %d | img=%s txt=%s | model=%s",
        device, seed, args.img_input, args.txt_input, model_name,
    )

    # --- Data ---
    data = build_dataloaders(cfg, args, seed)
    train_loader = data["train_loader"]
    train_chunked = data["train_chunked"]
    flickr_img = data["flickr_img"]
    flickr_txt = data["flickr_txt"]
    flickr_txt_mask = data["flickr_txt_mask"]
    flickr_img_cls_attn = data["flickr_img_cls_attn"]
    flickr_txt_cls_attn = data["flickr_txt_cls_attn"]

    # Strip CLS tokens from eval data if requested
    if getattr(args, "strip_cls", False):
        if flickr_img is not None and flickr_img.dim() == 3:
            flickr_img = flickr_img[:, 1:, :]
            logger.info("strip_cls: removed CLS from Flickr img → %s", tuple(flickr_img.shape))
        if flickr_txt is not None and flickr_txt.dim() == 3:
            flickr_txt = flickr_txt[:, 1:, :]
            if flickr_txt_mask is not None:
                flickr_txt_mask = flickr_txt_mask[:, 1:]
            logger.info("strip_cls: removed CLS from Flickr txt → %s", tuple(flickr_txt.shape))

    steps_per_epoch = data["steps_per_epoch"]
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]

    # --- Model ---
    model = build_model(cfg, args, train_dataset, device, seed)

    if getattr(args, "group_taus", None) is not None:
        logger.info(
            "Using group temperatures: %s (pool_temperature ignored for cross_attn)",
            args.group_taus,
        )

    # FixedRelativeRep has no learnable params — eval-only
    if model_name == "fixed_relative_rep":
        logger.info("FixedRelativeRep has no trainable parameters. Running evaluation only.")
        model.eval()
        with torch.no_grad():
            if flickr_img is not None:
                eval_kwargs: dict[str, Any] = {}
                if flickr_txt_mask is not None:
                    eval_kwargs["txt_mask"] = flickr_txt_mask
                if uses_tokens:
                    eval_kwargs["batch_size"] = 64
                metrics = evaluate_retrieval(
                    model, flickr_img, flickr_txt, **eval_kwargs,
                )
                _log_retrieval(metrics, epoch=0)
        logger.info("Done.")
        return

    # --- LR finder (optional) ---
    if getattr(args, "lr_find", False):
        from src.utils.lr_finder import find_lr

        logger.info("Running LR finder...")
        # Build a data iterator for the LR finder (one epoch)
        if train_chunked is not None:
            lr_data_iter = train_chunked.epoch_iterator(0, device=device)
        else:
            lr_data_iter = train_loader
        exp_name_for_plot = cfg["logging"].get("experiment_name", "default")
        plot_dir = PROJECT_ROOT / "experiments" / "lr_finder"
        plot_dir.mkdir(parents=True, exist_ok=True)
        suggested_lr = find_lr(
            model, lr_data_iter, device,
            batch_size=cfg["training"]["batch_size"],
            temperature=cfg["training"]["temperature"],
            num_iter=100,
            subset_size=5000,
            txt_token_level=txt_token_level,
            save_plot=plot_dir / f"lr_finder_{exp_name_for_plot}.png",
        )
        logger.info("LR finder suggests: %.2e (was %.2e)", suggested_lr,
                     cfg["training"]["lr"])
        cfg["training"]["lr"] = suggested_lr

    # --- Optimiser & scheduler ---
    epochs = cfg["training"]["epochs"]
    optimizer = Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer, epochs, cfg["training"]["warmup_epochs"], steps_per_epoch,
    )

    # --- Metrics logger (wandb) ---
    metrics_logger = MetricsLogger(
        experiment_name=cfg["logging"]["experiment_name"],
        cfg=cfg,
    )

    # Log model info to wandb
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_logger.set_summary("model_name", model_name)
    metrics_logger.set_summary("params_total", n_params)
    metrics_logger.set_summary("params_trainable", n_train)
    metrics_logger.set_summary("img_input", args.img_input)
    metrics_logger.set_summary("txt_input", args.txt_input)

    # --- Checkpoint dir ---
    exp_name = cfg["logging"]["experiment_name"]
    save_dir = PROJECT_ROOT / cfg["logging"]["save_dir"] / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = cfg["training"].get("grad_clip", 1.0)
    temperature = cfg["training"]["temperature"]
    ortho_lambda = cfg["training"].get("ortho_lambda", 0.0)
    lb_lambda = cfg["training"].get("lb_lambda", 0.0)
    pa_lambda = cfg["training"].get("pa_lambda", 0.0)
    ca_lambda = cfg["training"].get("ca_lambda", 0.0)
    iso_lambda = cfg["training"].get("iso_lambda", 0.0)
    token_match_lambda = cfg["training"].get("token_match_lambda", 0.0)
    diversity_lambda = cfg["training"].get("diversity_lambda", 0.0)
    diversity_sigma = cfg["training"].get("diversity_sigma", 0.2)
    recon_lambda = cfg["training"].get("recon_lambda", 0.0)
    img_token_level = (args.img_input == "tokens")
    eval_every = cfg["eval"]["eval_every"]

    # --- Training loop ---
    best_mean_recall = 0.0
    best_epoch = 0
    logger.info("Starting training for %d epochs...", epochs)
    logger.info(
        "Config: model=%s K=%s bs=%d lr=%.1e temp=%.3f grad_clip=%.1f ortho=%.3f lb=%.3f",
        model_name,
        cfg["model"].get("num_anchors", "N/A"),
        cfg["training"]["batch_size"],
        cfg["training"]["lr"],
        temperature,
        grad_clip,
        ortho_lambda,
        lb_lambda,
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Select data iterator for this epoch
        if train_chunked is not None:
            data_iter = train_chunked.epoch_iterator(epoch, device=device)
        else:
            data_iter = train_loader

        # Train
        train_metrics = train_one_epoch(
            model, data_iter, optimizer, scheduler,
            temperature, device, grad_clip, ortho_lambda, lb_lambda, pa_lambda,
            iso_lambda, token_match_lambda, ca_lambda,
            diversity_lambda=diversity_lambda,
            diversity_sigma=diversity_sigma,
            diversity_modality=getattr(args, "diversity_modality", "both"),
            recon_lambda=recon_lambda,
            num_experts=getattr(args, "num_experts", 1),
            is_chunked=(train_chunked is not None),
            txt_token_level=txt_token_level,
            img_token_level=img_token_level,
            strip_cls=getattr(args, "strip_cls", False),
        )
        epoch_time = time.time() - t0

        # Build per-epoch log dict
        global_step = epoch * steps_per_epoch
        log_dict: dict[str, float] = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/lr": train_metrics["lr"],
        }
        for aux_key in ("ortho_loss", "iso_loss", "lb_loss", "pa_loss",
                        "token_match_loss", "ca_loss", "diversity_loss",
                        "recon_loss"):
            if aux_key in train_metrics:
                log_dict[f"train/{aux_key}"] = train_metrics[aux_key]
        # Per-expert monitoring metrics
        for k, v in train_metrics.items():
            if k.startswith(("infonce_expert_", "recon_loss_expert_",
                              "attn_entropy_expert_")):
                log_dict[f"train/{k}"] = v

        # Log per-anchor τ statistics if using learnable temperatures
        if getattr(model, "learnable_tau", False):
            with torch.no_grad():
                tau_vals = model.log_pool_temperature.exp()
                log_dict["tau/mean"] = tau_vals.mean().item()
                log_dict["tau/min"] = tau_vals.min().item()
                log_dict["tau/max"] = tau_vals.max().item()
                log_dict["tau/std"] = tau_vals.std().item()

        # Validation loss (CLS path only — token path has no val split)
        val_loss_str = ""
        if val_dataset is not None:
            val_loss = _compute_val_loss(model, val_dataset, temperature, device)
            log_dict["val/loss"] = val_loss
            val_loss_str = f" val={val_loss:.4f}"

        # Auxiliary loss strings for console output
        aux_parts = []
        for key, label in [("ca_loss", "ca"), ("iso_loss", "iso"),
                           ("token_match_loss", "tm")]:
            if key in train_metrics:
                aux_parts.append(f" {label}={train_metrics[key]:.4f}")
        aux_str = "".join(aux_parts)

        tau_str = ""
        if "tau/mean" in log_dict:
            tau_str = (
                f" τ={log_dict['tau/mean']:.4f}"
                f"[{log_dict['tau/min']:.4f},{log_dict['tau/max']:.4f}]"
            )

        log_line = (
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={train_metrics['loss']:.4f}{val_loss_str}{aux_str}{tau_str} | "
            f"lr={train_metrics['lr']:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # Retrieval evaluation
        retrieval_metrics: dict[str, float] | None = None
        if flickr_img is not None and epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_kwargs = {}
                if flickr_txt_mask is not None:
                    eval_kwargs["txt_mask"] = flickr_txt_mask
                if flickr_img_cls_attn is not None:
                    eval_kwargs["img_cls_attn"] = flickr_img_cls_attn
                if flickr_txt_cls_attn is not None:
                    eval_kwargs["txt_cls_attn"] = flickr_txt_cls_attn
                if uses_tokens:
                    eval_kwargs["batch_size"] = 64
                retrieval_metrics = evaluate_retrieval(
                    model, flickr_img, flickr_txt, **eval_kwargs,
                )
            for k, v in retrieval_metrics.items():
                log_dict[f"flickr/{k}"] = v
            log_line += (
                f" | R@1={retrieval_metrics['i2t_r1']:.1f}/{retrieval_metrics['t2i_r1']:.1f}"
                f" R@5={retrieval_metrics['i2t_r5']:.1f}/{retrieval_metrics['t2i_r5']:.1f}"
                f" mR={retrieval_metrics['mean_recall']:.1f}"
            )

        # Log all metrics for this epoch in a single wandb call
        metrics_logger.log(log_dict, step=global_step)

        logger.info(log_line)

        # Checkpoint: save latest + best
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            retrieval_metrics or {},
            cfg,
            save_dir / "latest.pt",
        )
        if retrieval_metrics and retrieval_metrics["mean_recall"] > best_mean_recall:
            best_mean_recall = retrieval_metrics["mean_recall"]
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                retrieval_metrics,
                cfg,
                save_dir / "best.pt",
            )
            logger.info("  ↑ New best mean recall: %.2f", best_mean_recall)

    # --- Final summary ---
    metrics_logger.set_summary("best_mean_recall", best_mean_recall)
    metrics_logger.set_summary("best_epoch", best_epoch if best_mean_recall > 0 else 0)
    metrics_logger.close()
    logger.info("Training complete. Best mean recall: %.2f", best_mean_recall)
    logger.info("Checkpoints saved to: %s", save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Bridge Anchors or baseline alignment models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file.",
    )
    # --- Overrides (all optional — override specific config values) ---
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["bridge_anchors", "linear_projection", "mlp_projection",
                 "fixed_relative_rep", "freeze_align", "shared_anchor"],
        help="Model type (overrides config).",
    )
    parser.add_argument(
        "--num-anchors", type=int, default=None,
        help="Number of anchors K (overrides config).",
    )
    parser.add_argument(
        "--init-method", type=str, default=None,
        choices=["random", "prototype", "kmeans", "fps"],
        help="Anchor initialisation method (overrides config).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Subsample training data to this many pairs (overrides config).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--ortho-lambda", type=float, default=None,
        help="Orthogonality regularization weight (overrides config).",
    )
    parser.add_argument(
        "--lb-lambda", type=float, default=None,
        help="Load-balancing loss weight (overrides config).",
    )
    parser.add_argument(
        "--pa-lambda", type=float, default=None,
        help="Per-anchor contrastive loss weight (overrides config).",
    )
    parser.add_argument(
        "--iso-lambda", type=float, default=None,
        help="Anchor isometry loss weight (Gram matrix matching).",
    )
    parser.add_argument(
        "--token-match-lambda", type=float, default=None,
        help="Weight for token-level matching loss. Only active when both "
             "img_input=tokens and txt_input=tokens.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-K sparse gating (0 = disabled, use all anchors).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name for logging/checkpoints (overrides config).",
    )
    # --- Input mode control ---
    parser.add_argument(
        "--img-input", type=str, default="cls",
        choices=["cls", "tokens"],
        help="Image input type: 'cls' for CLS-only (B, 768), "
             "'tokens' for token-level (B, 257, 768).",
    )
    parser.add_argument(
        "--txt-input", type=str, default="cls",
        choices=["cls", "tokens"],
        help="Text input type: 'cls' for CLS-only (B, 768), "
             "'tokens' for token-level (B, S, 768) with attention mask.",
    )
    parser.add_argument(
        "--token-pool", type=str, default="mean",
        choices=["mean", "max", "cross_attn"],
        help="Token aggregation method for token-level inputs.",
    )
    parser.add_argument(
        "--pool-temperature", type=float, default=0.1,
        help="Temperature for cross-attention pooling softmax.",
    )
    parser.add_argument(
        "--ca-lambda", type=float, default=None,
        help="Weight for cross-attention InfoNCE loss (dual-loss mode). "
             "When > 0, model produces both CLS and cross-attention profiles.",
    )
    parser.add_argument(
        "--ca-exclude-cls", action="store_true", default=False,
        help="Exclude CLS token from cross-attention pooling in dual-loss mode. "
             "When True, CA uses patches only (img[:, 1:, :]).",
    )
    parser.add_argument(
        "--strip-cls", action="store_true", default=False,
        help="Strip CLS token from all token-level inputs before model forward. "
             "Affects both training and eval. Use for CLS-excluded experiments.",
    )
    parser.add_argument(
        "--anchor-mediated", action="store_true", default=False,
        help="Use anchor-mediated token representation instead of pooling.",
    )
    parser.add_argument(
        "--selection-mode", type=str, default="soft",
        choices=["soft", "hard"],
        help="Token selection mode for anchor-mediated: 'soft' or 'hard'.",
    )
    parser.add_argument(
        "--am-cls-weight", type=float, default=0.0,
        help="Weight for CLS profile similarity in anchor-mediated sim. "
             "Combined as: sim = sim_anchor + am_cls_weight * sim_cls.",
    )
    parser.add_argument(
        "--learnable-tau", action="store_true", default=False,
        help="Use per-anchor learnable temperature for cross-attention pooling. "
             "Each anchor gets its own τ, initialised to pool-temperature.",
    )
    parser.add_argument(
        "--group-taus", nargs="+", type=float, default=None,
        help="Per-group fixed temperatures for cross-attention pooling. "
             "K must be divisible by len(group_taus). Overrides pool-temperature.",
    )
    parser.add_argument(
        "--lr-find", action="store_true", default=False,
        help="Run learning rate finder before training and use the suggested "
             "LR. Overrides --lr.",
    )
    parser.add_argument(
        "--projector-dim", type=int, default=0,
        help="Bottleneck dimension for lightweight projector before anchor "
             "similarity. 0 = no projector (default).",
    )
    parser.add_argument(
        "--cls-anchors", type=int, default=0,
        help="Number of CLS-path anchors for dual profile. CLS gets its own "
             "projector and anchors. 0 = off (default).",
    )
    parser.add_argument(
        "--num-experts", type=int, default=1,
        help="Number of expert projectors. Each expert projects tokens into a "
             "different space with its own anchor group. 1 = single projector.",
    )
    parser.add_argument(
        "--expert-soft-mask", action="store_true", default=False,
        help="Enable learnable Gaussian soft masks for multi-expert. Each "
             "expert's attention is biased toward different CLS attention ranges.",
    )
    parser.add_argument(
        "--expert-k", type=int, default=0,
        help="Anchors PER expert (HME). When >0, total anchors = num_experts * expert_k. "
             "Each expert gets its own anchor parameters and projector.",
    )
    parser.add_argument(
        "--diversity-lambda", type=float, default=None,
        help="Weight for hierarchical attention diversity loss (HME only).",
    )
    parser.add_argument(
        "--diversity-sigma", type=float, default=None,
        help="Gaussian spread for tier target distributions in diversity loss.",
    )
    parser.add_argument(
        "--diversity-modality", type=str, default="both",
        choices=["both", "img_only", "txt_only"],
        help="Which modalities to apply HME diversity loss on.",
    )
    parser.add_argument(
        "--recon-loss", action="store_true", default=False,
        help="Enable per-expert reconstruction decoders (ReconBA).",
    )
    parser.add_argument(
        "--recon-lambda", type=float, default=None,
        help="Weight for reconstruction loss (ReconBA).",
    )
    parser.add_argument(
        "--stacked-anchors-dim", type=int, default=0,
        help="Layer 2 meta-anchor count (K2) for stacked anchor measurement. "
             "Meta-anchors live in K1-dim profile space. 0 = off (default).",
    )
    parser.add_argument(
        "--profile-proj-dim", type=int, default=0,
        help="Bottleneck dim for residual MLP projector on K-dim profile. "
             "0 = off (default). Mutually exclusive with --stacked-anchors-dim.",
    )
    parser.add_argument(
        "--attn-mask-groups", nargs="+", type=int, default=None,
        help="CLS attention masking: each group sees a different percentile "
             "slice of tokens sorted by CLS attention. E.g. '30 30 40 100' = "
             "top-30%%, next-30%%, bottom-40%%, all-tokens. K must be divisible "
             "by number of groups.",
    )
    parser.add_argument(
        "--group-norm", action="store_true", default=False,
        help="L2-normalize each group's sub-profile independently before "
             "concatenation. Requires --group-taus.",
    )
    parser.add_argument(
        "--group-gating", action="store_true", default=False,
        help="MoE-style gating: CLS embedding routes to groups via learned "
             "linear gate. Requires --group-taus.",
    )
    parser.add_argument(
        "--cls-attn-prior", type=str, default="none",
        choices=["none", "multiply", "additive"],
        help="CLS attention prior for cross-attention pooling. "
             "'multiply' uses shared beta, 'additive' uses per-anchor learnable betas.",
    )
    parser.add_argument(
        "--cls-attn-beta", type=float, default=1.0,
        help="Beta for 'multiply' CLS attention prior mode.",
    )
    parser.add_argument(
        "--chunked", action="store_true", default=False,
        help="Use chunked data loading for full-scale token-level training (118K).",
    )
    parser.add_argument(
        "--dim-shared", type=int, default=256,
        help="Shared space dimension for SharedAnchorAligner.",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension for SharedAnchorAligner projector MLPs.",
    )
    parser.add_argument(
        "--projector-type", type=str, default="mlp",
        choices=["mlp", "linear", "residual", "residual_shared"],
        help="Projector design for SharedAnchorAligner.",
    )
    parser.add_argument(
        "--dim-img", type=int, default=None,
        help="Image embedding dimension (overrides config). E.g. 1536 for ViT-G.",
    )
    parser.add_argument(
        "--dim-txt", type=int, default=None,
        help="Text embedding dimension (overrides config). E.g. 1024 for RoBERTa-large.",
    )
    parser.add_argument(
        "--embedding-dir", type=str, default=None,
        help="Custom embedding directory (contains chunks, CLS, and Flickr files). "
             "Overrides the default data/embeddings/all_tokens + cls dirs.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (e.g. 0 or 1). Defaults to CUDA:0 if available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
