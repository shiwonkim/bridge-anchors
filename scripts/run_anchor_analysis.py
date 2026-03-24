"""Comprehensive anchor analysis for BridgeAnchors K=128 (Experiment B, seed=42).

Performs 5 analyses:
1. Nearest neighbor analysis with COCO captions and categories
2. Cross-modal anchor correspondence by category overlap
3. Anchor similarity structure (Gram matrices, Pearson, CKA)
4. Anchor coverage visualization (UMAP)
5. Dead anchor detection

Outputs: experiments/exp_anchor_analysis/
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
import seaborn as sns
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_anchor_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ──
CKPT_PATH = PROJECT_ROOT / "results/checkpoints/exp_b_k128_s42/best.pt"
COCO_IMG_EMB = PROJECT_ROOT / "data/embeddings/coco_train_img.pt"
COCO_TXT_EMB = PROJECT_ROOT / "data/embeddings/coco_train_txt.pt"
CAPTIONS_JSON = PROJECT_ROOT / "data/datasets/coco/annotations/captions_train2017.json"
INSTANCES_JSON = PROJECT_ROOT / "data/datasets/coco/annotations/instances_train2017.json"


# ===================================================================
# Load data
# ===================================================================

def load_all() -> dict:
    """Load model, embeddings, and COCO annotations."""
    from src.eval._utils import load_model_from_checkpoint

    logger.info("Loading checkpoint: %s", CKPT_PATH)
    model, cfg = load_model_from_checkpoint(str(CKPT_PATH), device=torch.device("cpu"))

    logger.info("Loading embeddings...")
    img_embs = torch.load(COCO_IMG_EMB, weights_only=True)
    txt_embs = torch.load(COCO_TXT_EMB, weights_only=True)
    logger.info("  img: %s, txt: %s", tuple(img_embs.shape), tuple(txt_embs.shape))

    logger.info("Loading COCO captions...")
    with open(CAPTIONS_JSON) as f:
        cap_data = json.load(f)

    logger.info("Loading COCO instances...")
    with open(INSTANCES_JSON) as f:
        inst_data = json.load(f)

    # Build image_id → index mapping
    # The embeddings are extracted in the order of cap_data["images"]
    image_id_list = [img["id"] for img in cap_data["images"]]
    image_id_to_idx = {iid: i for i, iid in enumerate(image_id_list)}

    # image_id → first caption (matches how embeddings were extracted)
    # Sort annotations by id to get deterministic first caption
    cap_anns = sorted(cap_data["annotations"], key=lambda a: a["id"])
    image_id_to_caption = {}
    for ann in cap_anns:
        iid = ann["image_id"]
        if iid not in image_id_to_caption:
            image_id_to_caption[iid] = ann["caption"]

    # Category info
    cat_id_to_info = {c["id"]: c for c in inst_data["categories"]}

    # image_id → set of category names and supercategory names
    image_id_to_cats = defaultdict(set)
    image_id_to_supercats = defaultdict(set)
    for ann in inst_data["annotations"]:
        iid = ann["image_id"]
        cat = cat_id_to_info[ann["category_id"]]
        image_id_to_cats[iid].add(cat["name"])
        image_id_to_supercats[iid].add(cat["supercategory"])

    return {
        "model": model,
        "img_embs": img_embs,
        "txt_embs": txt_embs,
        "image_id_list": image_id_list,
        "image_id_to_idx": image_id_to_idx,
        "image_id_to_caption": image_id_to_caption,
        "image_id_to_cats": image_id_to_cats,
        "image_id_to_supercats": image_id_to_supercats,
        "cat_id_to_info": cat_id_to_info,
    }


# ===================================================================
# Analysis 1: Nearest Neighbor with captions and categories
# ===================================================================

@torch.no_grad()
def analysis_1_nearest_neighbors(data: dict, top_k: int = 10) -> dict:
    """Find nearest training samples to each anchor with COCO metadata."""
    logger.info("=== Analysis 1: Nearest Neighbors ===")

    model = data["model"]
    img_embs = data["img_embs"]
    txt_embs = data["txt_embs"]
    image_id_list = data["image_id_list"]
    image_id_to_caption = data["image_id_to_caption"]
    image_id_to_cats = data["image_id_to_cats"]

    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1)
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1)
    img_norm = F.normalize(img_embs, dim=-1)
    txt_norm = F.normalize(txt_embs, dim=-1)

    K = anchors_img.shape[0]

    # Compute similarities
    img_sims = anchors_img @ img_norm.T  # (K, N)
    txt_sims = anchors_txt @ txt_norm.T  # (K, N)

    img_topk_sims, img_topk_idx = img_sims.topk(top_k, dim=1)
    txt_topk_sims, txt_topk_idx = txt_sims.topk(top_k, dim=1)

    # Build readable output
    anchor_details = []
    for k in range(K):
        # Image neighbors
        img_neighbors = []
        img_cats_counter = Counter()
        for j in range(top_k):
            idx = img_topk_idx[k, j].item()
            sim = img_topk_sims[k, j].item()
            iid = image_id_list[idx]
            caption = image_id_to_caption.get(iid, "N/A")
            cats = sorted(image_id_to_cats.get(iid, set()))
            img_neighbors.append({
                "rank": j + 1,
                "index": idx,
                "image_id": iid,
                "cosine_sim": round(sim, 4),
                "categories": cats,
                "caption": caption,
            })
            for c in cats:
                img_cats_counter[c] += 1

        # Text neighbors
        txt_neighbors = []
        txt_cats_counter = Counter()
        for j in range(top_k):
            idx = txt_topk_idx[k, j].item()
            sim = txt_topk_sims[k, j].item()
            iid = image_id_list[idx]
            caption = image_id_to_caption.get(iid, "N/A")
            cats = sorted(image_id_to_cats.get(iid, set()))
            txt_neighbors.append({
                "rank": j + 1,
                "index": idx,
                "image_id": iid,
                "cosine_sim": round(sim, 4),
                "categories": cats,
                "caption": caption,
            })
            for c in cats:
                txt_cats_counter[c] += 1

        anchor_details.append({
            "anchor_id": k,
            "img_top_categories": img_cats_counter.most_common(5),
            "txt_top_categories": txt_cats_counter.most_common(5),
            "img_neighbors": img_neighbors,
            "txt_neighbors": txt_neighbors,
        })

    # Save as JSON
    out_path = OUTPUT_DIR / "nearest_neighbors.json"
    with open(out_path, "w") as f:
        json.dump(anchor_details, f, indent=2)
    logger.info("  Saved nearest neighbor details to %s", out_path)

    # Also save a compact markdown summary
    md_lines = ["# Anchor Nearest Neighbors — BridgeAnchors K=128\n"]
    for a in anchor_details:
        k = a["anchor_id"]
        img_cats = ", ".join(f"{c}({n})" for c, n in a["img_top_categories"])
        txt_cats = ", ".join(f"{c}({n})" for c, n in a["txt_top_categories"])
        md_lines.append(f"## Anchor {k}")
        md_lines.append(f"**Image top categories:** {img_cats}")
        md_lines.append(f"**Text top categories:** {txt_cats}")
        md_lines.append("")
        md_lines.append("| | Image Neighbor | Text Neighbor |")
        md_lines.append("|---|---|---|")
        for j in range(min(5, top_k)):
            in_ = a["img_neighbors"][j]
            tn_ = a["txt_neighbors"][j]
            ic = ", ".join(in_["categories"][:3]) if in_["categories"] else "—"
            tc = ", ".join(tn_["categories"][:3]) if tn_["categories"] else "—"
            i_cap = in_["caption"][:60] + "..." if len(in_["caption"]) > 60 else in_["caption"]
            t_cap = tn_["caption"][:60] + "..." if len(tn_["caption"]) > 60 else tn_["caption"]
            md_lines.append(f"| {j+1} | [{ic}] {i_cap} | [{tc}] {t_cap} |")
        md_lines.append("")

    md_path = OUTPUT_DIR / "nearest_neighbors.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("  Saved markdown summary to %s", md_path)

    return {
        "img_topk_idx": img_topk_idx,
        "txt_topk_idx": txt_topk_idx,
        "img_topk_sims": img_topk_sims,
        "txt_topk_sims": txt_topk_sims,
        "anchor_details": anchor_details,
    }


# ===================================================================
# Analysis 2: Cross-modal anchor correspondence
# ===================================================================

def analysis_2_cross_modal_correspondence(data: dict, nn_results: dict) -> dict:
    """Compute category overlap between paired image/text anchors."""
    logger.info("=== Analysis 2: Cross-Modal Anchor Correspondence ===")

    image_id_list = data["image_id_list"]
    image_id_to_cats = data["image_id_to_cats"]
    image_id_to_supercats = data["image_id_to_supercats"]
    anchor_details = nn_results["anchor_details"]

    K = len(anchor_details)
    cat_overlaps = []
    supercat_overlaps = []

    for k in range(K):
        # Collect categories from image neighbors
        img_cats = set()
        for nb in anchor_details[k]["img_neighbors"]:
            img_cats.update(nb["categories"])

        # Collect categories from text neighbors
        txt_cats = set()
        for nb in anchor_details[k]["txt_neighbors"]:
            txt_cats.update(nb["categories"])

        # Jaccard overlap on categories
        union = img_cats | txt_cats
        cat_overlap = len(img_cats & txt_cats) / len(union) if union else 0.0
        cat_overlaps.append(cat_overlap)

        # Same for supercategories
        img_sup = set()
        for nb in anchor_details[k]["img_neighbors"]:
            iid = nb["image_id"]
            img_sup.update(image_id_to_supercats.get(iid, set()))
        txt_sup = set()
        for nb in anchor_details[k]["txt_neighbors"]:
            iid = nb["image_id"]
            txt_sup.update(image_id_to_supercats.get(iid, set()))
        union_sup = img_sup | txt_sup
        supercat_overlap = len(img_sup & txt_sup) / len(union_sup) if union_sup else 0.0
        supercat_overlaps.append(supercat_overlap)

    cat_overlaps = np.array(cat_overlaps)
    supercat_overlaps = np.array(supercat_overlaps)

    logger.info("  Category overlap — mean: %.3f, median: %.3f",
                cat_overlaps.mean(), np.median(cat_overlaps))
    logger.info("  Supercategory overlap — mean: %.3f, median: %.3f",
                supercat_overlaps.mean(), np.median(supercat_overlaps))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].bar(range(K), cat_overlaps, color="#2196F3", alpha=0.8, width=1.0)
    axes[0].axhline(y=cat_overlaps.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {cat_overlaps.mean():.3f}")
    axes[0].set_xlabel("Anchor Index", fontsize=11)
    axes[0].set_ylabel("Jaccard Overlap", fontsize=11)
    axes[0].set_title("Category Overlap: Image vs Text Anchor Neighbors", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)

    axes[1].bar(range(K), supercat_overlaps, color="#4CAF50", alpha=0.8, width=1.0)
    axes[1].axhline(y=supercat_overlaps.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {supercat_overlaps.mean():.3f}")
    axes[1].set_xlabel("Anchor Index", fontsize=11)
    axes[1].set_ylabel("Jaccard Overlap", fontsize=11)
    axes[1].set_title("Supercategory Overlap: Image vs Text Anchor Neighbors", fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_modal_correspondence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved cross-modal correspondence plot")

    return {
        "cat_overlaps": cat_overlaps,
        "supercat_overlaps": supercat_overlaps,
    }


# ===================================================================
# Analysis 3: Anchor similarity structure
# ===================================================================

@torch.no_grad()
def analysis_3_similarity_structure(data: dict) -> dict:
    """Compute and visualize anchor-anchor Gram matrices."""
    logger.info("=== Analysis 3: Anchor Similarity Structure ===")

    model = data["model"]
    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1)
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1)

    sim_img = (anchors_img @ anchors_img.T).numpy()
    sim_txt = (anchors_txt @ anchors_txt.T).numpy()

    # Metrics on upper triangle (excluding diagonal)
    K = sim_img.shape[0]
    triu = np.triu_indices(K, k=1)
    vec_img = sim_img[triu]
    vec_txt = sim_txt[triu]

    # Pearson correlation
    pearson_r = np.corrcoef(vec_img, vec_txt)[0, 1]

    # Linear CKA
    from src.eval.anchor_analysis import _linear_cka
    cka = _linear_cka(torch.from_numpy(sim_img), torch.from_numpy(sim_txt))

    # Frobenius norm of difference
    frob = np.linalg.norm(sim_img - sim_txt, "fro")

    logger.info("  Pearson r (off-diag): %.4f", pearson_r)
    logger.info("  Linear CKA: %.4f", cka)
    logger.info("  Frobenius diff: %.4f", frob)

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    vmin, vmax = -0.3, 0.3  # anchors should be near-orthogonal

    im0 = axes[0].imshow(sim_img, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[0].set_title("Image Anchor Gram Matrix\n$A_{img} A_{img}^T$", fontsize=12)
    axes[0].set_xlabel("Anchor Index")
    axes[0].set_ylabel("Anchor Index")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(sim_txt, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[1].set_title("Text Anchor Gram Matrix\n$A_{txt} A_{txt}^T$", fontsize=12)
    axes[1].set_xlabel("Anchor Index")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff = sim_img - sim_txt
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
    axes[2].set_title(f"Difference\nPearson r={pearson_r:.3f}, CKA={cka:.3f}", fontsize=12)
    axes[2].set_xlabel("Anchor Index")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gram_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved Gram matrices plot")

    # Distribution of off-diagonal similarities
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    axes2[0].hist(vec_img, bins=50, alpha=0.7, color="#2196F3", label="Image anchors")
    axes2[0].hist(vec_txt, bins=50, alpha=0.7, color="#E91E63", label="Text anchors")
    axes2[0].set_xlabel("Pairwise Cosine Similarity", fontsize=11)
    axes2[0].set_ylabel("Count", fontsize=11)
    axes2[0].set_title("Distribution of Off-Diagonal Anchor Similarities", fontsize=11)
    axes2[0].legend(fontsize=9)
    axes2[0].axvline(x=0, color="black", linestyle="--", alpha=0.3)

    axes2[1].scatter(vec_img, vec_txt, alpha=0.05, s=3, color="#666")
    axes2[1].plot([-0.4, 0.4], [-0.4, 0.4], "r--", alpha=0.5, label="y=x")
    axes2[1].set_xlabel("Image Anchor Pair Similarity", fontsize=11)
    axes2[1].set_ylabel("Text Anchor Pair Similarity", fontsize=11)
    axes2[1].set_title(f"Anchor Pair Similarity Correlation (r={pearson_r:.3f})", fontsize=11)
    axes2[1].legend(fontsize=9)
    axes2[1].set_aspect("equal")

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "similarity_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  Saved similarity distribution plot")

    return {
        "sim_img": sim_img,
        "sim_txt": sim_txt,
        "pearson_r": pearson_r,
        "cka": cka,
        "frob": frob,
    }


# ===================================================================
# Analysis 4: Anchor coverage (UMAP)
# ===================================================================

@torch.no_grad()
def analysis_4_coverage(data: dict, n_samples: int = 2000) -> None:
    """UMAP visualization of anchors among training embeddings."""
    logger.info("=== Analysis 4: Anchor Coverage (UMAP) ===")

    try:
        from umap import UMAP
    except ImportError:
        logger.warning("  umap-learn not installed, trying sklearn TSNE instead")
        from sklearn.manifold import TSNE as DimReducer
        use_umap = False
    else:
        use_umap = True

    model = data["model"]
    img_embs = data["img_embs"]
    txt_embs = data["txt_embs"]
    image_id_list = data["image_id_list"]
    image_id_to_supercats = data["image_id_to_supercats"]

    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1).numpy()
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1).numpy()

    K = anchors_img.shape[0]

    # Subsample training embeddings
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(img_embs.shape[0], generator=gen)[:n_samples]
    img_sub = F.normalize(img_embs[indices], dim=-1).numpy()
    txt_sub = F.normalize(txt_embs[indices], dim=-1).numpy()

    # Get supercategories for sampled embeddings
    supercat_names = []
    all_supercats = set()
    for idx in indices.tolist():
        iid = image_id_list[idx]
        scats = image_id_to_supercats.get(iid, set())
        # Use the first supercategory (most images have one dominant object)
        sc = sorted(scats)[0] if scats else "none"
        supercat_names.append(sc)
        all_supercats.add(sc)

    # Map supercategories to colors
    supercat_list = sorted(all_supercats)
    cmap = plt.cm.get_cmap("tab20", len(supercat_list))
    sc_to_color = {sc: cmap(i) for i, sc in enumerate(supercat_list)}
    sample_colors = [sc_to_color[sc] for sc in supercat_names]

    # Fit dimensionality reduction
    for modality, emb_sub, anch, label in [
        ("image", img_sub, anchors_img, "Image"),
        ("text", txt_sub, anchors_txt, "Text"),
    ]:
        logger.info("  Computing %s embedding for %s space...",
                    "UMAP" if use_umap else "t-SNE", label)

        combined = np.vstack([emb_sub, anch])  # (n_samples + K, dim)

        if use_umap:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        else:
            reducer = DimReducer(n_components=2, random_state=42, perplexity=30)

        coords = reducer.fit_transform(combined)
        emb_coords = coords[:n_samples]
        anch_coords = coords[n_samples:]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot embeddings colored by supercategory
        for sc in supercat_list:
            mask = [supercat_names[i] == sc for i in range(n_samples)]
            if any(mask):
                pts = emb_coords[mask]
                ax.scatter(pts[:, 0], pts[:, 1], c=[sc_to_color[sc]], s=6, alpha=0.3,
                          label=sc, rasterized=True)

        # Plot anchors as large red stars
        ax.scatter(anch_coords[:, 0], anch_coords[:, 1],
                  c="red", s=80, marker="*", edgecolors="black", linewidths=0.5,
                  zorder=10, label=f"Anchors (K={K})")

        method = "UMAP" if use_umap else "t-SNE"
        ax.set_title(f"{label} Space: {K} Anchors Among {n_samples} Training Embeddings ({method})",
                    fontsize=12)
        ax.set_xlabel(f"{method} 1", fontsize=11)
        ax.set_ylabel(f"{method} 2", fontsize=11)

        # Legend with smaller entries
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, fontsize=7, loc="upper right",
                 markerscale=2, ncol=2, framealpha=0.8)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"coverage_{modality}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s coverage plot", modality)


# ===================================================================
# Analysis 5: Dead anchor detection
# ===================================================================

@torch.no_grad()
def analysis_5_dead_anchors(data: dict) -> dict:
    """Check if any anchors are never the closest for any training sample."""
    logger.info("=== Analysis 5: Dead Anchor Detection ===")

    model = data["model"]
    img_embs = data["img_embs"]
    txt_embs = data["txt_embs"]

    anchors_img = F.normalize(model.anchors_img.detach().cpu(), dim=-1)
    anchors_txt = F.normalize(model.anchors_txt.detach().cpu(), dim=-1)
    img_norm = F.normalize(img_embs, dim=-1)
    txt_norm = F.normalize(txt_embs, dim=-1)

    K = anchors_img.shape[0]
    N = img_embs.shape[0]

    # Process in chunks to avoid OOM
    chunk_size = 4096
    img_usage = torch.zeros(K, dtype=torch.long)
    txt_usage = torch.zeros(K, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        # (chunk, K)
        img_sims = img_norm[start:end] @ anchors_img.T
        txt_sims = txt_norm[start:end] @ anchors_txt.T

        # Closest anchor for each sample
        img_closest = img_sims.argmax(dim=1)  # (chunk,)
        txt_closest = txt_sims.argmax(dim=1)

        for idx in img_closest:
            img_usage[idx] += 1
        for idx in txt_closest:
            txt_usage[idx] += 1

    img_dead = (img_usage == 0).sum().item()
    txt_dead = (txt_usage == 0).sum().item()
    img_active = K - img_dead
    txt_active = K - txt_dead

    logger.info("  Image anchors: %d/%d active (%d dead)", img_active, K, img_dead)
    logger.info("  Text anchors:  %d/%d active (%d dead)", txt_active, K, txt_dead)
    logger.info("  Image anchor usage — min: %d, max: %d, median: %d",
                img_usage.min().item(), img_usage.max().item(),
                img_usage.median().item())
    logger.info("  Text anchor usage — min: %d, max: %d, median: %d",
                txt_usage.min().item(), txt_usage.max().item(),
                txt_usage.median().item())

    # Plot usage distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by usage for visual clarity
    img_sorted = img_usage.sort(descending=True).values.numpy()
    txt_sorted = txt_usage.sort(descending=True).values.numpy()

    axes[0].bar(range(K), img_sorted, color="#2196F3", width=1.0)
    axes[0].axhline(y=N/K, color="red", linestyle="--", linewidth=1.5,
                    label=f"Uniform = {N/K:.0f}")
    axes[0].set_xlabel("Anchor (sorted by usage)", fontsize=11)
    axes[0].set_ylabel("# Samples Assigned", fontsize=11)
    axes[0].set_title(f"Image Anchor Usage ({img_active}/{K} active, {img_dead} dead)",
                     fontsize=11)
    axes[0].legend(fontsize=9)

    axes[1].bar(range(K), txt_sorted, color="#E91E63", width=1.0)
    axes[1].axhline(y=N/K, color="red", linestyle="--", linewidth=1.5,
                    label=f"Uniform = {N/K:.0f}")
    axes[1].set_xlabel("Anchor (sorted by usage)", fontsize=11)
    axes[1].set_ylabel("# Samples Assigned", fontsize=11)
    axes[1].set_title(f"Text Anchor Usage ({txt_active}/{K} active, {txt_dead} dead)",
                     fontsize=11)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "anchor_usage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved anchor usage plot")

    return {
        "img_usage": img_usage,
        "txt_usage": txt_usage,
        "img_dead": img_dead,
        "txt_dead": txt_dead,
        "img_active": img_active,
        "txt_active": txt_active,
    }


# ===================================================================
# Summary report
# ===================================================================

def write_summary(
    nn_res: dict,
    corr_res: dict,
    struct_res: dict,
    dead_res: dict,
) -> None:
    """Write a consolidated summary report."""
    lines = [
        "# Anchor Analysis — BridgeAnchors K=128 (Exp B, seed=42)\n",
        f"**Checkpoint:** `results/checkpoints/exp_b_k128_s42/best.pt`",
        f"**Training data:** COCO 2017 train (118,287 pairs)",
        f"**Anchors:** 128 image anchors (128×768) + 128 text anchors (128×768)\n",
        "## 1. Nearest Neighbor Analysis",
        "",
        "Each anchor's 10 nearest training samples (by cosine similarity) are recorded with",
        "their COCO image IDs, captions, and object categories.",
        "",
        f"Full details: `nearest_neighbors.json` and `nearest_neighbors.md`\n",
        "## 2. Cross-Modal Anchor Correspondence",
        "",
        "For each anchor k, we compare the COCO categories present in its image-space",
        "neighbors vs its text-space neighbors using Jaccard overlap.",
        "",
        f"- **Category overlap** — mean: {corr_res['cat_overlaps'].mean():.3f}, "
        f"median: {np.median(corr_res['cat_overlaps']):.3f}",
        f"- **Supercategory overlap** — mean: {corr_res['supercat_overlaps'].mean():.3f}, "
        f"median: {np.median(corr_res['supercat_overlaps']):.3f}",
        "",
        "Higher values indicate that paired image/text anchors attend to the same semantic",
        "concepts, confirming that the learned anchors develop cross-modal correspondence.\n",
        "## 3. Anchor Similarity Structure",
        "",
        "Compares the inter-anchor Gram matrices $A_{img} A_{img}^T$ and $A_{txt} A_{txt}^T$.",
        "",
        f"- **Pearson correlation** (off-diagonal): {struct_res['pearson_r']:.4f}",
        f"- **Linear CKA**: {struct_res['cka']:.4f}",
        f"- **Frobenius diff**: {struct_res['frob']:.4f}",
        "",
        "High Pearson/CKA means anchors organized into parallel structures across modalities —",
        "anchors that are close in image space are also close in text space.\n",
        "## 4. Anchor Coverage",
        "",
        "UMAP/t-SNE visualization of 128 anchors among 2,000 training embeddings, colored by",
        "COCO supercategory. Shows whether anchors spread across the embedding space or cluster.\n",
        "## 5. Dead Anchor Detection",
        "",
        f"- **Image anchors**: {dead_res['img_active']}/{dead_res['img_active'] + dead_res['img_dead']} "
        f"active ({dead_res['img_dead']} dead — never the closest anchor for any training sample)",
        f"- **Text anchors**: {dead_res['txt_active']}/{dead_res['txt_active'] + dead_res['txt_dead']} "
        f"active ({dead_res['txt_dead']} dead)",
        "",
        f"- Image usage range: {dead_res['img_usage'].min().item()} – {dead_res['img_usage'].max().item()} "
        f"(uniform would be {118287 // 128})",
        f"- Text usage range: {dead_res['txt_usage'].min().item()} – {dead_res['txt_usage'].max().item()}",
        "",
        "## Output Files",
        "",
        "- `nearest_neighbors.json` — full NN details with captions and categories",
        "- `nearest_neighbors.md` — readable markdown summary (top 5 per anchor)",
        "- `cross_modal_correspondence.png` — category/supercategory overlap bar charts",
        "- `gram_matrices.png` — side-by-side Gram matrix heatmaps + difference",
        "- `similarity_distributions.png` — off-diagonal similarity distributions + correlation",
        "- `coverage_image.png` — UMAP/t-SNE of image anchors + embeddings",
        "- `coverage_text.png` — UMAP/t-SNE of text anchors + embeddings",
        "- `anchor_usage.png` — per-anchor assignment counts (dead anchor detection)",
        "- `results_summary.md` — this file",
    ]

    summary_path = OUTPUT_DIR / "results_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Summary written to %s", summary_path)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    logger.info("Starting comprehensive anchor analysis...")
    data = load_all()

    nn_res = analysis_1_nearest_neighbors(data)
    corr_res = analysis_2_cross_modal_correspondence(data, nn_res)
    struct_res = analysis_3_similarity_structure(data)
    analysis_4_coverage(data)
    dead_res = analysis_5_dead_anchors(data)

    write_summary(nn_res, corr_res, struct_res, dead_res)
    logger.info("All analyses complete. Results in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
