"""Post-training analysis for Load Balancing loss experiments.

For each lambda, loads the best checkpoint and runs:
- Anchor usage distribution (dead anchor detection)
- UMAP coverage
- Cross-modal correspondence

Also generates comparison plots and summary table.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_step1_lb_loss"

COCO_IMG_EMB = PROJECT_ROOT / "data/embeddings/cls/coco_train_img.pt"
COCO_TXT_EMB = PROJECT_ROOT / "data/embeddings/cls/coco_train_txt.pt"
CAPTIONS_JSON = PROJECT_ROOT / "data/datasets/coco/annotations/captions_train2017.json"
INSTANCES_JSON = PROJECT_ROOT / "data/datasets/coco/annotations/instances_train2017.json"

LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
LAMBDA_LABELS = ["0p0", "0p01", "0p05", "0p1", "0p5", "1p0"]


def load_shared_data() -> dict:
    """Load embeddings and COCO annotations (shared across all lambdas)."""
    logger.info("Loading embeddings...")
    img_embs = torch.load(COCO_IMG_EMB, weights_only=True)
    txt_embs = torch.load(COCO_TXT_EMB, weights_only=True)

    logger.info("Loading COCO annotations...")
    with open(CAPTIONS_JSON) as f:
        cap_data = json.load(f)
    with open(INSTANCES_JSON) as f:
        inst_data = json.load(f)

    image_id_list = [img["id"] for img in cap_data["images"]]
    cat_id_to_info = {c["id"]: c for c in inst_data["categories"]}

    image_id_to_supercats = defaultdict(set)
    image_id_to_cats = defaultdict(set)
    for ann in inst_data["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_info[ann["category_id"]]
        image_id_to_cats[iid].add(cat["name"])
        image_id_to_supercats[iid].add(cat["supercategory"])

    return {
        "img_embs": img_embs,
        "txt_embs": txt_embs,
        "image_id_list": image_id_list,
        "image_id_to_cats": image_id_to_cats,
        "image_id_to_supercats": image_id_to_supercats,
    }


def load_checkpoint(lb_label: str) -> tuple:
    """Load model from checkpoint."""
    from src.eval._utils import load_model_from_checkpoint

    if lb_label == "0p0":
        ckpt_path = PROJECT_ROOT / "results/checkpoints/exp_b_k128_s42/best.pt"
    else:
        ckpt_path = PROJECT_ROOT / f"results/checkpoints/exp_lb_{lb_label}/best.pt"

    model, cfg = load_model_from_checkpoint(str(ckpt_path), device=torch.device("cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt["metrics"]
    return model, metrics


@torch.no_grad()
def compute_anchor_usage(model, img_embs, txt_embs) -> dict:
    """Compute per-anchor usage counts."""
    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1)
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1)
    img_norm = F.normalize(img_embs, dim=-1)
    txt_norm = F.normalize(txt_embs, dim=-1)

    K = anchors_img.shape[0]
    N = img_embs.shape[0]
    img_usage = torch.zeros(K, dtype=torch.long)
    txt_usage = torch.zeros(K, dtype=torch.long)

    chunk_size = 4096
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        img_closest = (img_norm[start:end] @ anchors_img.T).argmax(dim=1)
        txt_closest = (txt_norm[start:end] @ anchors_txt.T).argmax(dim=1)
        for idx in img_closest:
            img_usage[idx] += 1
        for idx in txt_closest:
            txt_usage[idx] += 1

    return {
        "img_usage": img_usage,
        "txt_usage": txt_usage,
        "img_dead": (img_usage == 0).sum().item(),
        "txt_dead": (txt_usage == 0).sum().item(),
        "img_std": img_usage.float().std().item(),
        "txt_std": txt_usage.float().std().item(),
        "img_max": img_usage.max().item(),
        "txt_max": txt_usage.max().item(),
        "img_min": img_usage.min().item(),
        "txt_min": txt_usage.min().item(),
    }


@torch.no_grad()
def compute_cross_modal_overlap(model, shared_data, top_k=10) -> dict:
    """Compute category overlap between paired image/text anchor neighbors."""
    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1)
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1)
    img_norm = F.normalize(shared_data["img_embs"], dim=-1)
    txt_norm = F.normalize(shared_data["txt_embs"], dim=-1)

    K = anchors_img.shape[0]
    img_sims = anchors_img @ img_norm.T
    txt_sims = anchors_txt @ txt_norm.T
    _, img_topk_idx = img_sims.topk(top_k, dim=1)
    _, txt_topk_idx = txt_sims.topk(top_k, dim=1)

    image_id_list = shared_data["image_id_list"]
    image_id_to_cats = shared_data["image_id_to_cats"]
    image_id_to_supercats = shared_data["image_id_to_supercats"]

    cat_overlaps = []
    supercat_overlaps = []
    for k in range(K):
        img_cats = set()
        img_sup = set()
        for j in range(top_k):
            iid = image_id_list[img_topk_idx[k, j].item()]
            img_cats.update(image_id_to_cats.get(iid, set()))
            img_sup.update(image_id_to_supercats.get(iid, set()))

        txt_cats = set()
        txt_sup = set()
        for j in range(top_k):
            iid = image_id_list[txt_topk_idx[k, j].item()]
            txt_cats.update(image_id_to_cats.get(iid, set()))
            txt_sup.update(image_id_to_supercats.get(iid, set()))

        union_cat = img_cats | txt_cats
        cat_overlaps.append(len(img_cats & txt_cats) / len(union_cat) if union_cat else 0.0)
        union_sup = img_sup | txt_sup
        supercat_overlaps.append(len(img_sup & txt_sup) / len(union_sup) if union_sup else 0.0)

    return {
        "cat_overlap_mean": np.mean(cat_overlaps),
        "supercat_overlap_mean": np.mean(supercat_overlaps),
    }


@torch.no_grad()
def compute_umap_coverage(model, shared_data, out_dir: Path, lb_label: str) -> None:
    """UMAP plot of anchors among training embeddings."""
    try:
        from umap import UMAP
    except ImportError:
        from sklearn.manifold import TSNE
        logger.warning("umap not available, using t-SNE")

    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1).numpy()
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1).numpy()
    K = anchors_img.shape[0]

    gen = torch.Generator().manual_seed(42)
    n_samples = 2000
    indices = torch.randperm(shared_data["img_embs"].shape[0], generator=gen)[:n_samples]
    img_sub = F.normalize(shared_data["img_embs"][indices], dim=-1).numpy()
    txt_sub = F.normalize(shared_data["txt_embs"][indices], dim=-1).numpy()

    image_id_list = shared_data["image_id_list"]
    image_id_to_supercats = shared_data["image_id_to_supercats"]
    supercat_names = []
    for idx in indices.tolist():
        iid = image_id_list[idx]
        scats = image_id_to_supercats.get(iid, set())
        supercat_names.append(sorted(scats)[0] if scats else "none")

    for modality, emb_sub, anch in [("image", img_sub, anchors_img), ("text", txt_sub, anchors_txt)]:
        combined = np.vstack([emb_sub, anch])
        try:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        except NameError:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = reducer.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:n_samples, 0], coords[:n_samples, 1], c="lightgray", s=4,
                  alpha=0.3, rasterized=True, label="Training samples")
        ax.scatter(coords[n_samples:, 0], coords[n_samples:, 1], c="red", s=60,
                  marker="*", edgecolors="black", linewidths=0.5, zorder=10,
                  label=f"Anchors (K={K})")
        ax.set_title(f"{modality.title()} Space — lb_lambda={lb_label.replace('p', '.')}", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / f"coverage_{modality}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    shared_data = load_shared_data()

    all_results = {}

    for lb, lb_label in zip(LAMBDAS, LAMBDA_LABELS):
        logger.info("=== Analyzing lb_lambda=%s ===", lb)

        out_dir = EXP_DIR / f"analysis_lb{lb_label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        model, metrics = load_checkpoint(lb_label)

        # Retrieval metrics
        result = {
            "lb_lambda": lb,
            "i2t_r1": metrics.get("i2t_r1", 0),
            "i2t_r5": metrics.get("i2t_r5", 0),
            "i2t_r10": metrics.get("i2t_r10", 0),
            "t2i_r1": metrics.get("t2i_r1", 0),
            "t2i_r5": metrics.get("t2i_r5", 0),
            "t2i_r10": metrics.get("t2i_r10", 0),
            "mean_recall": metrics.get("mean_recall", 0),
        }

        # Anchor usage
        usage = compute_anchor_usage(model, shared_data["img_embs"], shared_data["txt_embs"])
        result.update({
            "img_dead": usage["img_dead"],
            "txt_dead": usage["txt_dead"],
            "img_usage_std": usage["img_std"],
            "txt_usage_std": usage["txt_std"],
            "img_usage_min": usage["img_min"],
            "img_usage_max": usage["img_max"],
        })

        # Per-lambda usage plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        img_sorted = usage["img_usage"].sort(descending=True).values.numpy()
        txt_sorted = usage["txt_usage"].sort(descending=True).values.numpy()
        N = shared_data["img_embs"].shape[0]
        K = 128
        for ax, data_sorted, title, color in [
            (axes[0], img_sorted, "Image", "#2196F3"),
            (axes[1], txt_sorted, "Text", "#E91E63"),
        ]:
            ax.bar(range(K), data_sorted, color=color, width=1.0)
            ax.axhline(y=N/K, color="red", linestyle="--", linewidth=1.5, label=f"Uniform={N//K}")
            ax.set_xlabel("Anchor (sorted)")
            ax.set_ylabel("# Samples")
            ax.set_title(f"{title} Anchor Usage — lb={lb}")
            ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "anchor_usage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Cross-modal correspondence
        overlap = compute_cross_modal_overlap(model, shared_data)
        result.update(overlap)

        # UMAP coverage
        compute_umap_coverage(model, shared_data, out_dir, lb_label)

        all_results[lb_label] = result
        logger.info("  mR=%.2f, img_dead=%d, txt_dead=%d, cat_overlap=%.3f, supercat_overlap=%.3f",
                    result["mean_recall"], result["img_dead"], result["txt_dead"],
                    result["cat_overlap_mean"], result["supercat_overlap_mean"])

    # ===================================================================
    # Comparison plots
    # ===================================================================

    # 1. Overlay anchor usage: baseline vs each lambda
    logger.info("Generating comparison plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Anchor Usage Distribution: Baseline (lb=0) vs Load Balancing", fontsize=13)
    baseline_usage = None

    for i, (lb, lb_label) in enumerate(zip(LAMBDAS, LAMBDA_LABELS)):
        model, _ = load_checkpoint(lb_label)
        usage = compute_anchor_usage(model, shared_data["img_embs"], shared_data["txt_embs"])
        sorted_usage = usage["img_usage"].sort(descending=True).values.numpy()

        if lb == 0.0:
            baseline_usage = sorted_usage

        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.bar(range(K), sorted_usage, color="#2196F3", width=1.0, alpha=0.7, label=f"lb={lb}")
        if baseline_usage is not None and lb > 0:
            ax.bar(range(K), baseline_usage, color="gray", width=1.0, alpha=0.3, label="baseline (lb=0)")
        ax.axhline(y=N/K, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"lb_lambda = {lb}", fontsize=11)
        ax.set_ylim(0, max(baseline_usage.max() if baseline_usage is not None else 2000, sorted_usage.max()) * 1.1)
        ax.legend(fontsize=8)
        ax.set_xlabel("Anchor (sorted)")
        ax.set_ylabel("# Samples")

    plt.tight_layout()
    fig.savefig(EXP_DIR / "comparison_usage_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Summary metrics bar chart
    labels_disp = [str(lb) for lb in LAMBDAS]
    mr_vals = [all_results[l]["mean_recall"] for l in LAMBDA_LABELS]
    cat_vals = [all_results[l]["cat_overlap_mean"] for l in LAMBDA_LABELS]
    sup_vals = [all_results[l]["supercat_overlap_mean"] for l in LAMBDA_LABELS]
    img_std_vals = [all_results[l]["img_usage_std"] for l in LAMBDA_LABELS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean recall
    bars = axes[0, 0].bar(range(len(LAMBDAS)), mr_vals, color="#2196F3")
    axes[0, 0].set_xticks(range(len(LAMBDAS)))
    axes[0, 0].set_xticklabels(labels_disp)
    axes[0, 0].set_xlabel("lb_lambda")
    axes[0, 0].set_ylabel("Mean Recall")
    axes[0, 0].set_title("Flickr30k Mean Recall vs lb_lambda")
    for bar, val in zip(bars, mr_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f"{val:.2f}", ha="center", fontsize=9)

    # Usage std (lower = more uniform)
    bars = axes[0, 1].bar(range(len(LAMBDAS)), img_std_vals, color="#FF9800")
    axes[0, 1].set_xticks(range(len(LAMBDAS)))
    axes[0, 1].set_xticklabels(labels_disp)
    axes[0, 1].set_xlabel("lb_lambda")
    axes[0, 1].set_ylabel("Std Dev of Anchor Usage")
    axes[0, 1].set_title("Image Anchor Usage Uniformity (lower = more uniform)")
    for bar, val in zip(bars, img_std_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{val:.0f}", ha="center", fontsize=9)

    # Category overlap
    bars = axes[1, 0].bar(range(len(LAMBDAS)), cat_vals, color="#4CAF50")
    axes[1, 0].set_xticks(range(len(LAMBDAS)))
    axes[1, 0].set_xticklabels(labels_disp)
    axes[1, 0].set_xlabel("lb_lambda")
    axes[1, 0].set_ylabel("Mean Jaccard Overlap")
    axes[1, 0].set_title("Category Overlap (Cross-Modal Correspondence)")
    for bar, val in zip(bars, cat_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f"{val:.3f}", ha="center", fontsize=9)

    # Supercategory overlap
    bars = axes[1, 1].bar(range(len(LAMBDAS)), sup_vals, color="#E91E63")
    axes[1, 1].set_xticks(range(len(LAMBDAS)))
    axes[1, 1].set_xticklabels(labels_disp)
    axes[1, 1].set_xlabel("lb_lambda")
    axes[1, 1].set_ylabel("Mean Jaccard Overlap")
    axes[1, 1].set_title("Supercategory Overlap (Cross-Modal Correspondence)")
    for bar, val in zip(bars, sup_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(EXP_DIR / "comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # Summary markdown
    # ===================================================================
    lines = [
        "# Step 1: Load Balancing Loss — Results\n",
        "**Date:** 2026-03-23",
        "**Training:** COCO 118K, BridgeAnchors K=128, 20 epochs, seed=42",
        "**Eval:** Flickr30k retrieval",
        f"**Lambda sweep:** {LAMBDAS}\n",
        "## Summary Table\n",
        "| lb_lambda | mR | I→T R@1 | I→T R@5 | T→I R@1 | T→I R@5 | Img Dead | Txt Dead | Usage Std (img) | Cat Overlap | Supercat Overlap |",
        "|-----------|------|---------|---------|---------|---------|----------|----------|----------------|-------------|------------------|",
    ]
    for lb_label_i in LAMBDA_LABELS:
        r = all_results[lb_label_i]
        lines.append(
            f"| {r['lb_lambda']} | {r['mean_recall']:.2f} | "
            f"{r['i2t_r1']:.2f} | {r['i2t_r5']:.2f} | "
            f"{r['t2i_r1']:.2f} | {r['t2i_r5']:.2f} | "
            f"{r['img_dead']} | {r['txt_dead']} | "
            f"{r['img_usage_std']:.0f} | "
            f"{r['cat_overlap_mean']:.3f} | {r['supercat_overlap_mean']:.3f} |"
        )

    lines.append("")
    lines.append("## Per-Lambda Analysis Directories\n")
    for lb, lb_label_i in zip(LAMBDAS, LAMBDA_LABELS):
        lines.append(f"- `analysis_lb{lb_label_i}/` — anchor_usage.png, coverage_image.png, coverage_text.png")
    lines.append("")
    lines.append("## Comparison Plots\n")
    lines.append("- `comparison_usage_overlay.png` — baseline vs each lambda, anchor usage")
    lines.append("- `comparison_summary.png` — 4-panel: mR, usage std, cat overlap, supercat overlap")

    with open(EXP_DIR / "results_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Summary written to %s", EXP_DIR / "results_summary.md")
    logger.info("All analyses complete.")


if __name__ == "__main__":
    main()
