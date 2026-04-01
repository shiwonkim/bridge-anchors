"""Direction A: Analyse what learned bridge anchors represent.

Three complementary analyses:
1. **Nearest neighbours** — which training samples are closest to each anchor.
2. **Anchor similarity structure** — do image-space and text-space anchor
   inter-relationships mirror each other?
3. **Class alignment** — which semantic classes does each anchor specialise in?

Usage as standalone script:
    python -m src.eval.anchor_analysis --checkpoint path/to/best.pt \
        --coco-img data/embeddings/cls/coco_train_img.pt \
        --coco-txt data/embeddings/cls/coco_train_txt.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.eval._utils import get_model_device, load_model_from_checkpoint

logger = logging.getLogger(__name__)


# ===================================================================
# Analysis 1 — Nearest Neighbours
# ===================================================================


@torch.no_grad()
def nearest_neighbours(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    top_k: int = 10,
) -> dict[str, Any]:
    """Find the nearest training samples to each anchor.

    For each of the K image anchors the ``top_k`` training images with
    the highest cosine similarity are returned, and likewise for text
    anchors.

    Args:
        model: A ``BridgeAnchorAligner`` (must have ``anchors_img`` and
            ``anchors_txt`` attributes).
        img_embs: (N, dim_img) training image embeddings.
        txt_embs: (N, dim_txt) training text embeddings.
        top_k: Number of neighbours to retrieve per anchor.

    Returns:
        Dict with:
        - ``img_indices``: (K, top_k) indices of nearest images per anchor.
        - ``img_sims``: (K, top_k) cosine similarities.
        - ``txt_indices``: (K, top_k) indices of nearest texts per anchor.
        - ``txt_sims``: (K, top_k) cosine similarities.
        - ``cross_modal_overlap``: (K,) Jaccard overlap between the
          neighbour sets of corresponding image/text anchors — high
          values suggest the paired anchors attend to the same concepts.
    """
    anchors_img, anchors_txt = _get_anchors(model)  # (K, dim_img), (K, dim_txt)

    # Normalise everything for cosine similarity
    a_img = F.normalize(anchors_img, dim=-1)  # (K, dim_img)
    a_txt = F.normalize(anchors_txt, dim=-1)  # (K, dim_txt)
    img_embs = F.normalize(img_embs, dim=-1)  # (N, dim_img)
    txt_embs = F.normalize(txt_embs, dim=-1)  # (N, dim_txt)

    # (K, N) similarities
    img_sims_all = a_img @ img_embs.T  # (K, N)
    txt_sims_all = a_txt @ txt_embs.T  # (K, N)

    img_topk_sims, img_topk_idx = img_sims_all.topk(top_k, dim=1)  # (K, top_k)
    txt_topk_sims, txt_topk_idx = txt_sims_all.topk(top_k, dim=1)

    # Cross-modal overlap: Jaccard between img-NN and txt-NN sets
    k = anchors_img.shape[0]
    overlap = torch.zeros(k)
    for i in range(k):
        img_set = set(img_topk_idx[i].tolist())
        txt_set = set(txt_topk_idx[i].tolist())
        union = len(img_set | txt_set)
        overlap[i] = len(img_set & txt_set) / union if union > 0 else 0.0

    return {
        "img_indices": img_topk_idx.cpu(),
        "img_sims": img_topk_sims.cpu(),
        "txt_indices": txt_topk_idx.cpu(),
        "txt_sims": txt_topk_sims.cpu(),
        "cross_modal_overlap": overlap,
    }


# ===================================================================
# Analysis 2 — Anchor Similarity Structure
# ===================================================================


@torch.no_grad()
def anchor_similarity_structure(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Compare inter-anchor similarity in image vs text space.

    If the learned alignment is working well, the (K, K) anchor–anchor
    similarity matrix should be similar across modalities — meaning
    anchors that are close in image space are also close in text space.

    Returns:
        Dict with:
        - ``sim_img``: (K, K) cosine similarity between image anchors.
        - ``sim_txt``: (K, K) cosine similarity between text anchors.
        - ``frobenius_diff``: Frobenius norm of the difference.
        - ``pearson_r``: Pearson correlation between the upper-triangle
          entries (excluding diagonal).
        - ``cka``: Centred Kernel Alignment between the two matrices.
    """
    anchors_img, anchors_txt = _get_anchors(model)

    a_img = F.normalize(anchors_img, dim=-1)
    a_txt = F.normalize(anchors_txt, dim=-1)

    sim_img = a_img @ a_img.T  # (K, K)
    sim_txt = a_txt @ a_txt.T  # (K, K)

    # Upper-triangle (excluding diagonal) for correlation
    k = sim_img.shape[0]
    triu_idx = torch.triu_indices(k, k, offset=1)
    vec_img = sim_img[triu_idx[0], triu_idx[1]]
    vec_txt = sim_txt[triu_idx[0], triu_idx[1]]

    # Frobenius norm of difference
    frob = (sim_img - sim_txt).norm(p="fro").item()

    # Pearson correlation
    pearson_r = _pearson(vec_img, vec_txt)

    # Linear CKA
    cka = _linear_cka(sim_img, sim_txt)

    return {
        "sim_img": sim_img.cpu(),
        "sim_txt": sim_txt.cpu(),
        "frobenius_diff": frob,
        "pearson_r": pearson_r,
        "cka": cka,
    }


# ===================================================================
# Analysis 3 — Class Alignment
# ===================================================================


@torch.no_grad()
def class_alignment(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    labels: torch.Tensor,
    top_classes: int = 5,
) -> dict[str, Any]:
    """For each anchor, find which semantic classes are closest.

    Computes per-class mean cosine similarity to each anchor and reports
    the top classes, revealing whether anchors specialise by concept.

    Args:
        model: ``BridgeAnchorAligner`` with ``anchors_img`` / ``anchors_txt``.
        img_embs: (N, dim_img) image embeddings.
        txt_embs: (N, dim_txt) text embeddings.
        labels: (N,) integer class labels.
        top_classes: Number of closest classes to report per anchor.

    Returns:
        Dict with:
        - ``img_class_sims``: (K, C) mean similarity of each anchor to
          each class's images.
        - ``txt_class_sims``: (K, C) mean similarity of each anchor to
          each class's texts.
        - ``img_top_classes``: (K, top_classes) top class indices per
          image anchor.
        - ``txt_top_classes``: (K, top_classes) top class indices per
          text anchor.
        - ``anchor_specialisation``: (K,) entropy of the class
          distribution per anchor (lower = more specialised).
    """
    anchors_img, anchors_txt = _get_anchors(model)
    a_img = F.normalize(anchors_img, dim=-1)
    a_txt = F.normalize(anchors_txt, dim=-1)
    img_embs = F.normalize(img_embs, dim=-1)
    txt_embs = F.normalize(txt_embs, dim=-1)
    labels = labels.long()

    num_classes = labels.max().item() + 1
    k_anchors = a_img.shape[0]

    # (K, N) raw similarities
    img_sims = a_img @ img_embs.T
    txt_sims = a_txt @ txt_embs.T

    # Aggregate per-class means: (K, C)
    img_class_sims = torch.zeros(k_anchors, num_classes)
    txt_class_sims = torch.zeros(k_anchors, num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        img_class_sims[:, c] = img_sims[:, mask].mean(dim=1).cpu()
        txt_class_sims[:, c] = txt_sims[:, mask].mean(dim=1).cpu()

    # Top classes per anchor
    _, img_top = img_class_sims.topk(top_classes, dim=1)
    _, txt_top = txt_class_sims.topk(top_classes, dim=1)

    # Specialisation: entropy of softmax(class_sims) per anchor
    # Lower entropy = anchor concentrates on fewer classes
    img_probs = F.softmax(img_class_sims, dim=1)  # (K, C)
    entropy = -(img_probs * (img_probs + 1e-12).log()).sum(dim=1)  # (K,)

    return {
        "img_class_sims": img_class_sims,
        "txt_class_sims": txt_class_sims,
        "img_top_classes": img_top,
        "txt_top_classes": txt_top,
        "anchor_specialisation": entropy,
    }


# ===================================================================
# Visualisation helpers
# ===================================================================


def plot_similarity_matrices(
    results: dict[str, Any],
    save_path: str | Path | None = None,
) -> None:
    """Plot image-space vs text-space anchor similarity heatmaps.

    Args:
        results: Output of ``anchor_similarity_structure()``.
        save_path: If given, save the figure to this path.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sim_img = results["sim_img"].numpy()
    sim_txt = results["sim_txt"].numpy()
    diff = sim_img - sim_txt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(sim_img, ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True)
    axes[0].set_title("Image-space anchor similarity")

    sns.heatmap(sim_txt, ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True)
    axes[1].set_title("Text-space anchor similarity")

    sns.heatmap(diff, ax=axes[2], cmap="RdBu_r", center=0, square=True)
    axes[2].set_title(
        f"Difference  (Frob={results['frobenius_diff']:.3f}, "
        f"r={results['pearson_r']:.3f})"
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved similarity matrices plot to %s", save_path)
    plt.close(fig)


def plot_class_alignment(
    results: dict[str, Any],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot per-anchor class specialisation heatmap.

    Args:
        results: Output of ``class_alignment()``.
        class_names: Optional list of class names for axis labels.
        save_path: If given, save the figure to this path.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Show img_class_sims for top-K most specialised anchors
    entropy = results["anchor_specialisation"]
    k = entropy.shape[0]
    # Sort anchors by specialisation (lowest entropy first)
    order = entropy.argsort()

    sims = results["img_class_sims"][order].numpy()
    # Only show top-20 classes per row for readability
    top_class_idx = set()
    for row in range(min(k, 16)):
        top_class_idx.update(results["img_top_classes"][order[row]].tolist()[:5])
    top_class_idx = sorted(top_class_idx)

    sub_sims = sims[:min(k, 16), :][:, top_class_idx]
    ylabels = [f"Anchor {order[i].item()}" for i in range(min(k, 16))]
    if class_names:
        xlabels = [class_names[c] for c in top_class_idx]
    else:
        xlabels = [str(c) for c in top_class_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(xlabels) * 0.5), min(k, 16) * 0.5))
    sns.heatmap(sub_sims, ax=ax, cmap="YlOrRd",
                xticklabels=xlabels, yticklabels=ylabels)
    ax.set_title("Anchor–class similarity (anchors sorted by specialisation)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved class alignment plot to %s", save_path)
    plt.close(fig)


def plot_cross_modal_overlap(
    nn_results: dict[str, Any],
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of per-anchor cross-modal neighbour overlap.

    Args:
        nn_results: Output of ``nearest_neighbours()``.
        save_path: If given, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    overlap = nn_results["cross_modal_overlap"].numpy()
    k = len(overlap)

    fig, ax = plt.subplots(figsize=(max(8, k * 0.3), 4))
    ax.bar(range(k), overlap, color="steelblue")
    ax.set_xlabel("Anchor index")
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Cross-modal NN overlap (higher = paired anchors agree)")
    ax.set_xticks(range(k))
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved cross-modal overlap plot to %s", save_path)
    plt.close(fig)


# ===================================================================
# Internal helpers
# ===================================================================


def _get_anchors(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract anchor parameters/buffers from the model.

    Works for both ``BridgeAnchorAligner`` (parameters) and
    ``FixedRelativeRep`` (buffers).

    Returns:
        anchors_img: (K, dim_img) tensor (detached, on CPU).
        anchors_txt: (K, dim_txt) tensor (detached, on CPU).
    """
    if hasattr(model, "anchors_img") and hasattr(model, "anchors_txt"):
        return model.anchors_img.detach().cpu(), model.anchors_txt.detach().cpu()
    raise ValueError(
        "Model does not have anchors_img / anchors_txt attributes. "
        "Anchor analysis is only supported for BridgeAnchorAligner and "
        "FixedRelativeRep."
    )


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = (x.norm() * y.norm()).clamp(min=1e-8)
    return (num / den).item()


def _linear_cka(K1: torch.Tensor, K2: torch.Tensor) -> float:
    """Linear Centred Kernel Alignment between two kernel matrices.

    CKA measures representational similarity; values in [0, 1].
    """
    # Centre kernels
    n = K1.shape[0]
    H = torch.eye(n, device=K1.device) - 1.0 / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H

    hsic_12 = (K1c * K2c).sum()
    hsic_11 = (K1c * K1c).sum()
    hsic_22 = (K2c * K2c).sum()

    denom = (hsic_11 * hsic_22).sqrt().clamp(min=1e-12)
    return (hsic_12 / denom).item()


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse learned bridge anchors (Direction A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--coco-img", type=str,
                        default="data/embeddings/cls/coco_train_img.pt",
                        help="Path to COCO training image embeddings.")
    parser.add_argument("--coco-txt", type=str,
                        default="data/embeddings/cls/coco_train_txt.pt",
                        help="Path to COCO training text embeddings.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Optional path to per-sample labels .pt for class alignment.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of nearest neighbours per anchor.")
    parser.add_argument("--output-dir", type=str, default="results/anchor_analysis",
                        help="Directory for output plots and data.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model, cfg = load_model_from_checkpoint(args.checkpoint)
    model_name = cfg["model"]["name"]
    logger.info("Model: %s", model_name)

    if not hasattr(model, "anchors_img"):
        logger.error("Anchor analysis requires a model with anchors "
                      "(BridgeAnchorAligner or FixedRelativeRep).")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_embs = torch.load(args.coco_img, weights_only=True)
    txt_embs = torch.load(args.coco_txt, weights_only=True)
    logger.info("Loaded training embeddings: img %s, txt %s",
                tuple(img_embs.shape), tuple(txt_embs.shape))

    # --- Analysis 1: Nearest neighbours ---
    logger.info("Running nearest-neighbour analysis (top_k=%d)...", args.top_k)
    nn_results = nearest_neighbours(model, img_embs, txt_embs, top_k=args.top_k)
    mean_overlap = nn_results["cross_modal_overlap"].mean().item()
    logger.info("  Mean cross-modal NN overlap (Jaccard): %.4f", mean_overlap)
    torch.save(nn_results, output_dir / "nn_results.pt")
    plot_cross_modal_overlap(nn_results, save_path=output_dir / "cross_modal_overlap.png")

    # --- Analysis 2: Anchor similarity structure ---
    logger.info("Running anchor similarity structure analysis...")
    struct_results = anchor_similarity_structure(model)
    logger.info("  Frobenius diff: %.4f", struct_results["frobenius_diff"])
    logger.info("  Pearson r:      %.4f", struct_results["pearson_r"])
    logger.info("  Linear CKA:     %.4f", struct_results["cka"])
    torch.save(struct_results, output_dir / "structure_results.pt")
    plot_similarity_matrices(struct_results, save_path=output_dir / "similarity_matrices.png")

    # --- Analysis 3: Class alignment (if labels provided) ---
    if args.labels is not None:
        logger.info("Running class alignment analysis...")
        labels = torch.load(args.labels, weights_only=True)
        cls_results = class_alignment(model, img_embs, txt_embs, labels)
        entropy = cls_results["anchor_specialisation"]
        logger.info("  Mean anchor specialisation entropy: %.4f (lower = more specialised)",
                    entropy.mean().item())
        torch.save(cls_results, output_dir / "class_alignment_results.pt")
        plot_class_alignment(cls_results, save_path=output_dir / "class_alignment.png")
    else:
        logger.info("No labels provided — skipping class alignment analysis.")

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
