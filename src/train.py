"""Main training script for Bridge Anchors and baselines.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --model linear_projection
    python -m src.train --config configs/default.yaml --num-samples 5000 --seed 1
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
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.coco_dataset import PairedEmbeddingDataset
from src.data.eval_datasets import Flickr30kEmbeddings
from src.eval.retrieval import evaluate_retrieval
from src.models.baselines import FixedRelativeRep, LinearProjection, MLPProjection
from src.models.bridge_anchors import BridgeAnchorAligner
from src.models.spectral_align import SpectralAligner
from src.models.token_bridge_anchors import TokenBridgeAnchorAligner
from src.models.losses import (
    anchor_orthogonality_loss,
    info_nce_loss,
    load_balancing_loss,
    per_anchor_contrastive_loss,
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
    if args.top_k is not None:
        cfg["model"]["top_k"] = args.top_k
    if args.pca_dim is not None:
        cfg["model"]["pca_dim"] = args.pca_dim
    if args.experiment_name is not None:
        cfg["logging"]["experiment_name"] = args.experiment_name


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(
    cfg: dict[str, Any],
    train_dataset: PairedEmbeddingDataset,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the model specified by config.

    Args:
        cfg: Full config dict.
        train_dataset: Training dataset (needed for prototype init and
            FixedRelativeRep anchor selection).
        device: Target device.

    Returns:
        Model moved to ``device``.
    """
    model_name = cfg["model"]["name"]
    dim_img = cfg["model"]["dim_img"]
    dim_txt = cfg["model"]["dim_txt"]

    if model_name == "bridge_anchors":
        num_anchors = cfg["model"]["num_anchors"]
        init_method = cfg["model"]["init_method"]

        proto_img, proto_txt = None, None
        if init_method == "prototype":
            proto_img, proto_txt = _compute_prototypes(
                train_dataset, num_anchors, cfg["training"]["seed"]
            )
        elif init_method == "kmeans":
            proto_img, proto_txt = _compute_kmeans_centroids(
                train_dataset, num_anchors, cfg["training"]["seed"]
            )
        elif init_method == "fps":
            proto_img, proto_txt = _compute_fps_anchors(
                train_dataset, num_anchors, cfg["training"]["seed"]
            )

        top_k = cfg["model"].get("top_k", 0)
        model = BridgeAnchorAligner(
            dim_img=dim_img,
            dim_txt=dim_txt,
            num_anchors=num_anchors,
            init_method=init_method,
            proto_img=proto_img,
            proto_txt=proto_txt,
            top_k=top_k,
        )

    elif model_name == "linear_projection":
        model = LinearProjection(dim_img=dim_img, dim_txt=dim_txt)

    elif model_name == "mlp_projection":
        model = MLPProjection(
            dim_img=dim_img,
            dim_txt=dim_txt,
            hidden_dim=cfg["baseline"]["mlp_hidden_dim"],
        )

    elif model_name == "spectral_aligner":
        num_anchors = cfg["model"]["num_anchors"]  # reuse K setting
        eigvecs_img, eigvecs_txt, mean_img, mean_txt = _compute_pca(
            train_dataset, num_anchors,
        )
        model = SpectralAligner(
            k=num_anchors,
            eigvecs_img=eigvecs_img,
            eigvecs_txt=eigvecs_txt,
            mean_img=mean_img,
            mean_txt=mean_txt,
        )

    elif model_name == "fixed_relative_rep":
        num_anchors = cfg["model"]["num_anchors"]
        init_method = cfg["model"]["init_method"]
        if init_method == "prototype":
            anchors_img, anchors_txt = _compute_prototypes(
                train_dataset, num_anchors, cfg["training"]["seed"]
            )
        else:
            anchors_img, anchors_txt = _select_fixed_anchors(
                train_dataset, num_anchors, cfg["training"]["seed"]
            )
        model = FixedRelativeRep(anchors_img=anchors_img, anchors_txt=anchors_txt)

    else:
        raise ValueError(f"Unknown model name: {model_name!r}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | params: %s total, %s trainable",
        model_name, f"{n_params:,}", f"{n_train:,}",
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


def _compute_pca(
    dataset: PairedEmbeddingDataset,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-K PCA eigenvectors for image and text embeddings.

    Returns:
        eigvecs_img: (dim_img, K) top-K eigenvectors for images.
        eigvecs_txt: (dim_txt, K) top-K eigenvectors for text.
        mean_img: (dim_img,) mean of image embeddings.
        mean_txt: (dim_txt,) mean of text embeddings.
    """
    img_embs, txt_embs = dataset.get_all()

    logger.info("Computing PCA (K=%d) on %d embeddings...", k, img_embs.shape[0])

    # Image PCA
    mean_img = img_embs.mean(dim=0)
    centered_img = img_embs - mean_img
    _, _, Vt_img = torch.linalg.svd(centered_img, full_matrices=False)
    eigvecs_img = Vt_img[:k].T  # (dim_img, K)

    # Text PCA
    mean_txt = txt_embs.mean(dim=0)
    centered_txt = txt_embs - mean_txt
    _, _, Vt_txt = torch.linalg.svd(centered_txt, full_matrices=False)
    eigvecs_txt = Vt_txt[:k].T  # (dim_txt, K)

    return eigvecs_img, eigvecs_txt, mean_img, mean_txt


def _apply_pca_reduction(
    pca_dim: int,
    train_dataset: PairedEmbeddingDataset,
    val_dataset: PairedEmbeddingDataset,
    flickr: Flickr30kEmbeddings | None,
) -> None:
    """Project all embedding tensors to a lower dimension via PCA (in-place).

    Computes PCA on the training set and applies the projection to train, val,
    and eval datasets. Modifies tensor data in-place on the dataset objects.

    Args:
        pca_dim: Target dimension.
        train_dataset: Training dataset (PCA is fit on this).
        val_dataset: Validation dataset.
        flickr: Optional Flickr30k eval dataset.
    """
    img_embs, txt_embs = train_dataset.get_all()
    logger.info(
        "Applying PCA reduction: %d → %d dims", img_embs.shape[1], pca_dim,
    )

    # Compute PCA on training data
    mean_img = img_embs.mean(dim=0)
    _, _, Vt_img = torch.linalg.svd(img_embs - mean_img, full_matrices=False)
    proj_img = Vt_img[:pca_dim].T  # (768, pca_dim)

    mean_txt = txt_embs.mean(dim=0)
    _, _, Vt_txt = torch.linalg.svd(txt_embs - mean_txt, full_matrices=False)
    proj_txt = Vt_txt[:pca_dim].T  # (768, pca_dim)

    # Project training data
    train_dataset.img_embs = (train_dataset.img_embs - mean_img) @ proj_img
    train_dataset.txt_embs = (train_dataset.txt_embs - mean_txt) @ proj_txt

    # Project validation data
    val_dataset.img_embs = (val_dataset.img_embs - mean_img) @ proj_img
    val_dataset.txt_embs = (val_dataset.txt_embs - mean_txt) @ proj_txt

    # Project Flickr30k eval data
    if flickr is not None:
        flickr.img_embs = (flickr.img_embs - mean_img) @ proj_img
        flickr.txt_embs = (flickr.txt_embs - mean_txt) @ proj_txt

    logger.info(
        "PCA reduction complete. Train img: %s, txt: %s",
        tuple(train_dataset.img_embs.shape),
        tuple(train_dataset.txt_embs.shape),
    )


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
# TensorBoard / W&B logging
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Thin wrapper that logs to TensorBoard and optionally W&B."""

    def __init__(self, log_dir: str | Path, experiment_name: str, cfg: dict[str, Any]) -> None:
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        # W&B (optional — only if WANDB_PROJECT env var is set)
        self.wandb_run = None
        wandb_project = os.environ.get("WANDB_PROJECT")
        if wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=cfg,
                    dir=str(self.log_dir),
                )
                logger.info("W&B logging enabled (project=%s).", wandb_project)
            except ImportError:
                logger.warning("WANDB_PROJECT is set but wandb is not installed.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to TensorBoard (and W&B if active)."""
        self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars under a common group."""
        for key, val in values.items():
            self.log_scalar(f"{main_tag}/{key}", val, step)

    def close(self) -> None:
        """Flush and close writers."""
        self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    temperature: float,
    device: torch.device,
    grad_clip: float,
    ortho_lambda: float = 0.0,
    lb_lambda: float = 0.0,
    pa_lambda: float = 0.0,
) -> dict[str, float]:
    """Run one training epoch.

    Returns:
        Dict with ``loss`` (epoch average), ``lr`` (final step LR),
        and optionally aux loss values if their lambdas > 0.
    """
    model.train()
    total_loss = 0.0
    total_ortho = 0.0
    total_lb = 0.0
    total_pa = 0.0
    num_batches = 0

    has_anchors = hasattr(model, "anchors_img")
    need_raw_sims = (lb_lambda > 0 or pa_lambda > 0) and has_anchors

    for img_emb, txt_emb in dataloader:
        img_emb = img_emb.to(device, non_blocking=True)
        txt_emb = txt_emb.to(device, non_blocking=True)

        optimizer.zero_grad()

        if need_raw_sims:
            b_img, b_txt, raw_img, raw_txt = model(
                img_emb, txt_emb, return_raw_sims=True,
            )
        else:
            b_img, b_txt = model(img_emb, txt_emb)

        loss = info_nce_loss(b_img, b_txt, temperature=temperature)

        ortho_loss_val = 0.0
        if ortho_lambda > 0 and has_anchors:
            ortho_loss = anchor_orthogonality_loss(
                model.anchors_img, model.anchors_txt,
            )
            loss = loss + ortho_lambda * ortho_loss
            ortho_loss_val = ortho_loss.item()

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

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_ortho += ortho_loss_val
        total_lb += lb_loss_val
        total_pa += pa_loss_val
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    current_lr = scheduler.get_last_lr()[0]
    result = {"loss": avg_loss, "lr": current_lr}
    if ortho_lambda > 0:
        result["ortho_loss"] = total_ortho / max(num_batches, 1)
    if lb_lambda > 0:
        result["lb_loss"] = total_lb / max(num_batches, 1)
    if pa_lambda > 0:
        result["pa_loss"] = total_pa / max(num_batches, 1)
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


def _run_token_level(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Completely independent token-level BridgeAnchors training path.

    Supports two data modes:
    - Pilot (default): loads 10K subset from coco_train_10k_img.pt
    - Chunked (--chunked): streams 118K from chunked files on NAS
    """
    from torch.utils.data import TensorDataset

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunked = getattr(args, "chunked", False)
    logger.info("TOKEN-LEVEL mode | %s | Device: %s | Seed: %d",
                "CHUNKED (118K)" if chunked else "PILOT (10K)", device, seed)

    token_dir = PROJECT_ROOT / "data" / "embeddings" / "token"
    num_anchors = cfg["model"]["num_anchors"]
    token_pool = args.token_pool
    batch_size = min(cfg["training"]["batch_size"], 128)

    # --- Load Flickr30k eval data ---
    flickr_img = torch.load(token_dir / "flickr30k_img.pt", weights_only=True)
    flickr_txt = torch.load(token_dir / "flickr30k_txt.pt", weights_only=True)
    logger.info("Flickr30k: img %s, txt %s", tuple(flickr_img.shape), tuple(flickr_txt.shape))

    # --- Load training data ---
    if chunked:
        from src.data.chunked_token_dataset import ChunkedTokenDataset
        chunk_dir = PROJECT_ROOT / "data" / "embeddings" / "token_full"
        txt_path = chunk_dir / "coco_train_txt.pt"
        train_chunked = ChunkedTokenDataset(
            chunk_dir=chunk_dir, text_emb_path=txt_path,
            batch_size=batch_size, seed=seed, split="train",
        )
        train_loader = None  # will use epoch_iterator
        steps_per_epoch = train_chunked.n_batches_approx
        logger.info("Chunked training: ~%d batches/epoch (bs=%d)",
                    steps_per_epoch, batch_size)
    else:
        train_img = torch.load(token_dir / "coco_train_10k_img.pt", weights_only=True)
        train_txt = torch.load(token_dir / "coco_train_10k_txt.pt", weights_only=True)
        logger.info("Pilot train: img %s, txt %s", tuple(train_img.shape), tuple(train_txt.shape))

        n = train_img.shape[0]
        n_val = max(1, int(n * 0.05))
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        train_idx = perm[n_val:]

        train_ds = TensorDataset(train_img[train_idx], train_txt[train_idx])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        steps_per_epoch = len(train_loader)
        logger.info("Pilot training: %d pairs (%d batches, bs=%d)",
                    len(train_ds), steps_per_epoch, batch_size)

    # --- Model ---
    model = TokenBridgeAnchorAligner(
        dim_img=768, dim_txt=768,
        num_anchors=num_anchors,
        token_pool=token_pool,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: TokenBridgeAnchorAligner | K=%d pool=%s | params=%s",
                num_anchors, token_pool, f"{n_params:,}")

    # --- Optimizer & scheduler ---
    epochs = cfg["training"]["epochs"]
    optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"],
                    weight_decay=cfg["training"]["weight_decay"])
    scheduler = build_scheduler(optimizer, epochs, cfg["training"]["warmup_epochs"], steps_per_epoch)

    temperature = cfg["training"]["temperature"]
    grad_clip = cfg["training"].get("grad_clip", 1.0)

    exp_name = cfg["logging"]["experiment_name"]
    save_dir = PROJECT_ROOT / cfg["logging"]["save_dir"] / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_mean_recall = 0.0
    logger.info("Starting training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Select data iterator
        if chunked:
            data_iter = train_chunked.epoch_iterator(epoch, device=device)
        else:
            data_iter = train_loader

        for batch_img, batch_txt in data_iter:
            if not chunked:
                batch_img = batch_img.to(device, non_blocking=True)
                batch_txt = batch_txt.to(device, non_blocking=True)

            optimizer.zero_grad()
            b_img, b_txt = model(batch_img, batch_txt)
            loss = info_nce_loss(b_img, b_txt, temperature=temperature)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - t0

        # Flickr eval (batched to avoid OOM with token-level tensors)
        model.eval()
        with torch.no_grad():
            metrics = evaluate_retrieval(model, flickr_img, flickr_txt,
                                        batch_size=64)

        log_line = (
            f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
            f"{epoch_time:.1f}s | "
            f"R@1={metrics['i2t_r1']:.1f}/{metrics['t2i_r1']:.1f} "
            f"R@5={metrics['i2t_r5']:.1f}/{metrics['t2i_r5']:.1f} "
            f"mR={metrics['mean_recall']:.1f}"
        )
        logger.info(log_line)

        save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, save_dir / "latest.pt")
        if metrics["mean_recall"] > best_mean_recall:
            best_mean_recall = metrics["mean_recall"]
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, save_dir / "best.pt")
            logger.info("  ↑ New best mean recall: %.2f", best_mean_recall)

    logger.info("Training complete. Best mean recall: %.2f", best_mean_recall)
    logger.info("Checkpoints saved to: %s", save_dir)


def main() -> None:
    args = parse_args()

    # --- Config ---
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    # --- Token-level path (completely independent) ---
    if args.token_level:
        _run_token_level(args, cfg)
        return

    seed = cfg["training"]["seed"]
    seed_everything(seed)

    # --- Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Seed: %d", device, seed)

    # --- Data ---
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(
        "Train: %d pairs (%d batches) | Val: %d pairs",
        len(train_dataset), len(train_loader), len(val_dataset),
    )

    # Flickr30k eval embeddings (optional — skip if files don't exist)
    flickr: Flickr30kEmbeddings | None = None
    flickr_img_path = PROJECT_ROOT / cfg["eval"]["flickr_img_emb_path"]
    flickr_txt_path = PROJECT_ROOT / cfg["eval"]["flickr_txt_emb_path"]
    if flickr_img_path.exists() and flickr_txt_path.exists():
        flickr = Flickr30kEmbeddings(flickr_img_path, flickr_txt_path)
    else:
        logger.warning(
            "Flickr30k embeddings not found — skipping retrieval evaluation. "
            "Run extract_embeddings.py --dataset flickr30k first."
        )

    # --- Optional PCA reduction ---
    pca_dim = cfg["model"].get("pca_dim", 0)
    if pca_dim and pca_dim > 0:
        _apply_pca_reduction(pca_dim, train_dataset, val_dataset, flickr)
        cfg["model"]["dim_img"] = pca_dim
        cfg["model"]["dim_txt"] = pca_dim

    # --- Model ---
    model = build_model(cfg, train_dataset, device)

    # FixedRelativeRep has no learnable params — just evaluate and exit
    if cfg["model"]["name"] == "fixed_relative_rep":
        logger.info("FixedRelativeRep has no trainable parameters. Running evaluation only.")
        if flickr is not None:
            fi, ft = flickr.get_all()
            metrics = evaluate_retrieval(model, fi, ft)
            _log_retrieval(metrics, epoch=0)
        logger.info("Done.")
        return

    # --- Optimiser & scheduler ---
    epochs = cfg["training"]["epochs"]
    optimizer = Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer, epochs, cfg["training"]["warmup_epochs"], steps_per_epoch,
    )

    # --- Metrics logger ---
    metrics_logger = MetricsLogger(
        log_dir=PROJECT_ROOT / cfg["logging"]["log_dir"],
        experiment_name=cfg["logging"]["experiment_name"],
        cfg=cfg,
    )

    # --- Checkpoint dir ---
    save_dir = PROJECT_ROOT / cfg["logging"]["save_dir"] / cfg["logging"]["experiment_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = cfg["training"].get("grad_clip", 1.0)
    temperature = cfg["training"]["temperature"]
    ortho_lambda = cfg["training"].get("ortho_lambda", 0.0)
    lb_lambda = cfg["training"].get("lb_lambda", 0.0)
    pa_lambda = cfg["training"].get("pa_lambda", 0.0)
    eval_every = cfg["eval"]["eval_every"]

    # --- Training loop ---
    best_mean_recall = 0.0
    logger.info("Starting training for %d epochs...", epochs)
    logger.info("Config: model=%s K=%s bs=%d lr=%.1e temp=%.3f grad_clip=%.1f ortho=%.3f lb=%.3f",
                cfg["model"]["name"],
                cfg["model"].get("num_anchors", "N/A"),
                cfg["training"]["batch_size"],
                cfg["training"]["lr"],
                temperature,
                grad_clip,
                ortho_lambda,
                lb_lambda)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            temperature, device, grad_clip, ortho_lambda, lb_lambda, pa_lambda,
        )
        epoch_time = time.time() - t0

        # Log training metrics
        global_step = epoch * steps_per_epoch
        metrics_logger.log_scalar("train/loss", train_metrics["loss"], global_step)
        metrics_logger.log_scalar("train/lr", train_metrics["lr"], global_step)

        # Validation loss on held-out split
        val_loss = _compute_val_loss(model, val_dataset, temperature, device)
        metrics_logger.log_scalar("val/loss", val_loss, global_step)

        log_line = (
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={train_metrics['loss']:.4f} val={val_loss:.4f} | "
            f"lr={train_metrics['lr']:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # Retrieval evaluation
        retrieval_metrics: dict[str, float] | None = None
        if flickr is not None and epoch % eval_every == 0:
            fi, ft = flickr.get_all()
            retrieval_metrics = evaluate_retrieval(model, fi, ft)
            metrics_logger.log_scalars("flickr", retrieval_metrics, global_step)
            log_line += (
                f" | R@1={retrieval_metrics['i2t_r1']:.1f}/{retrieval_metrics['t2i_r1']:.1f}"
                f" R@5={retrieval_metrics['i2t_r5']:.1f}/{retrieval_metrics['t2i_r5']:.1f}"
                f" mR={retrieval_metrics['mean_recall']:.1f}"
            )

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
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                retrieval_metrics,
                cfg,
                save_dir / "best.pt",
            )
            logger.info("  ↑ New best mean recall: %.2f", best_mean_recall)

    # --- Final summary ---
    metrics_logger.close()
    logger.info("Training complete. Best mean recall: %.2f", best_mean_recall)
    logger.info("Checkpoints saved to: %s", save_dir)


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
        choices=["bridge_anchors", "linear_projection", "mlp_projection", "fixed_relative_rep", "spectral_aligner"],
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
        "--top-k", type=int, default=None,
        help="Top-K sparse gating (0 = disabled, use all anchors).",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=None,
        help="PCA reduction dimension (0 or None = disabled). When enabled, "
             "all embeddings are projected to this dimension before training.",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name for logging/checkpoints (overrides config).",
    )
    # --- Token-level BridgeAnchors ---
    parser.add_argument(
        "--token-level", action="store_true", default=False,
        help="Enable token-level BridgeAnchors (loads from data/embeddings/token/).",
    )
    parser.add_argument(
        "--token-pool", type=str, default="mean",
        choices=["mean", "max"],
        help="Token aggregation method for token-level BA.",
    )
    parser.add_argument(
        "--chunked", action="store_true", default=False,
        help="Use chunked data loading for full-scale token-level training (118K).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
