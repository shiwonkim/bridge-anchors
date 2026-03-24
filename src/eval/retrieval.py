"""Image-text retrieval evaluation on Flickr30k.

Computes Recall@K for both image-to-text and text-to-image directions.
Supports both full-matrix and batched computation for large datasets.

Usage as standalone script:
    python -m src.eval.retrieval --checkpoint path/to/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.eval._utils import get_model_device, load_model_from_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 0,
) -> dict[str, float]:
    """Compute image-text retrieval recall metrics.

    Ground truth assumes 1-to-1 pairing: the *i*-th image is matched
    with the *i*-th text.

    Args:
        model: Alignment model with
            ``forward(img_emb, txt_emb) -> (b_img, b_txt)``.
        img_embs: (N, dim_img) raw image embeddings.
        txt_embs: (N, dim_txt) raw text embeddings.
        ks: Recall cutoffs to report.
        batch_size: If > 0, compute bridged representations in chunks of
            this size to limit GPU memory.  The similarity matrix is still
            computed on the full set.  Set to 0 (default) to process
            everything in one shot.

    Returns:
        Dict with keys ``i2t_r1``, ``i2t_r5``, ``i2t_r10``,
        ``t2i_r1``, ``t2i_r5``, ``t2i_r10``, ``mean_recall``.
        Values are percentages in [0, 100].
    """
    model.eval()
    device = get_model_device(model)

    # --- Compute bridged representations ---
    b_img, b_txt = _bridge_batched(model, img_embs, txt_embs, device, batch_size)

    # --- Similarity matrix (on CPU to avoid OOM for large N) ---
    # Both outputs are already L2-normalised, so dot product = cosine sim.
    # A 31K×31K float32 matrix is ~3.8 GB — fits in RAM but not alongside
    # model + embeddings on a single GPU.
    b_img_cpu = b_img.cpu()
    b_txt_cpu = b_txt.cpu()
    sims = b_img_cpu @ b_txt_cpu.T  # (N, N)

    # --- Recall computation (CPU) ---
    n = sims.shape[0]
    gt = torch.arange(n)  # diagonal ground truth
    metrics: dict[str, float] = {}

    # Image-to-text: for each image, find rank of correct text
    i2t_pos = _get_gt_ranks(sims, gt)
    for k in ks:
        metrics[f"i2t_r{k}"] = (i2t_pos < k).float().mean().item() * 100.0

    # Text-to-image: for each text, find rank of correct image
    t2i_pos = _get_gt_ranks(sims.T, gt)
    for k in ks:
        metrics[f"t2i_r{k}"] = (t2i_pos < k).float().mean().item() * 100.0

    # Mean recall across all six R@K values
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)

    return metrics


@torch.no_grad()
def compute_retrieval_ranks(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-query rank positions (useful for detailed analysis).

    Args:
        model: Alignment model.
        img_embs: (N, dim_img) raw image embeddings.
        txt_embs: (N, dim_txt) raw text embeddings.
        batch_size: Chunk size for bridging (0 = all at once).

    Returns:
        i2t_pos: (N,) rank of correct text for each image (0-indexed).
        t2i_pos: (N,) rank of correct image for each text (0-indexed).
    """
    model.eval()
    device = get_model_device(model)
    b_img, b_txt = _bridge_batched(model, img_embs, txt_embs, device, batch_size)
    b_img_cpu = b_img.cpu()
    b_txt_cpu = b_txt.cpu()
    sims = b_img_cpu @ b_txt_cpu.T
    n = sims.shape[0]
    gt = torch.arange(n)

    i2t_pos = _get_gt_ranks(sims, gt)
    t2i_pos = _get_gt_ranks(sims.T, gt)

    return i2t_pos, t2i_pos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_gt_ranks(sims: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Get the rank of each ground-truth match without a full argsort.

    For each row *i*, computes the rank of column ``gt[i]`` by counting
    how many columns have a higher similarity score.  This avoids
    allocating an (N, N) int64 index tensor from ``argsort``.

    Args:
        sims: (N, N) similarity matrix (CPU).
        gt: (N,) ground-truth column index per row.

    Returns:
        (N,) 0-indexed rank of the correct match per row.
    """
    gt_scores = sims[torch.arange(sims.shape[0]), gt]  # (N,)
    # rank = number of items with strictly higher similarity
    ranks = (sims > gt_scores.unsqueeze(1)).sum(dim=1)  # (N,)
    return ranks


def _bridge_batched(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bridged representations, optionally in batches.

    Returns:
        b_img: (N, D) on ``device``.
        b_txt: (N, D) on ``device``.
    """
    n = img_embs.shape[0]
    if batch_size <= 0 or batch_size >= n:
        img_embs = img_embs.to(device)
        txt_embs = txt_embs.to(device)
        return model(img_embs, txt_embs)

    b_img_parts: list[torch.Tensor] = []
    b_txt_parts: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bi, bt = model(
            img_embs[start:end].to(device),
            txt_embs[start:end].to(device),
        )
        b_img_parts.append(bi)
        b_txt_parts.append(bt)

    return torch.cat(b_img_parts, dim=0), torch.cat(b_txt_parts, dim=0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on Flickr30k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--flickr-img", type=str,
                        default="data/embeddings/flickr30k_test_img.pt",
                        help="Path to Flickr30k image embeddings.")
    parser.add_argument("--flickr-txt", type=str,
                        default="data/embeddings/flickr30k_test_txt.pt",
                        help="Path to Flickr30k text embeddings.")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for bridging (0 = all at once).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model, cfg = load_model_from_checkpoint(args.checkpoint)
    logger.info("Model: %s", cfg["model"]["name"])

    from src.data.eval_datasets import Flickr30kEmbeddings
    flickr = Flickr30kEmbeddings(args.flickr_img, args.flickr_txt)
    img_embs, txt_embs = flickr.get_all()

    metrics = evaluate_retrieval(model, img_embs, txt_embs, batch_size=args.batch_size)

    logger.info("Flickr30k Retrieval Results:")
    logger.info("  Image→Text  R@1=%.2f  R@5=%.2f  R@10=%.2f",
                metrics["i2t_r1"], metrics["i2t_r5"], metrics["i2t_r10"])
    logger.info("  Text→Image  R@1=%.2f  R@5=%.2f  R@10=%.2f",
                metrics["t2i_r1"], metrics["t2i_r5"], metrics["t2i_r10"])
    logger.info("  Mean Recall: %.2f", metrics["mean_recall"])


if __name__ == "__main__":
    main()
