"""Qualitative analysis of CLS attention saliency tiers.

Visualizes whether DINOv2 (image) and all-mpnet-base-v2 (text) CLS attention
maps produce meaningful 4-tier saliency assignments for HME diversity loss.

Usage:
    python scripts/analyze_cls_saliency.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings" / "all_tokens"
OUT_DIR = PROJECT_ROOT / "experiments" / "exp_hme_diversity"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tier colors (RGBA, semi-transparent overlays)
TIER_COLORS_RGB = [
    (1.0, 0.0, 0.0),  # Tier 0: RED — salient
    (1.0, 0.85, 0.0), # Tier 1: YELLOW — secondary
    (0.0, 0.4, 1.0),  # Tier 2: BLUE — context
    (0.5, 0.5, 0.5),  # Tier 3: GRAY — peripheral
]
TIER_NAMES = ["Salient (top-25%)", "Secondary (25-50%)",
              "Context (50-75%)", "Peripheral (bot-25%)"]

# Common English stopwords (no NLTK dependency)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "doing", "will", "would", "should", "could",
    "can", "may", "might", "must", "shall", "this", "that", "these", "those",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "s", "t", "d", "ll", "m", "o", "re", "ve", "y", "as",
    ",", ".", "'", '"', "!", "?", ":", ";", "-", "(", ")", "'s",
}


def load_train_caption_map() -> dict:
    """Return {image_id: first_caption} for COCO train2017."""
    ann_file = DATA_DIR / "coco" / "annotations" / "captions_train2017.json"
    with open(ann_file) as f:
        data = json.load(f)
    img_id_to_fn = {img["id"]: img["file_name"] for img in data["images"]}
    first_cap: dict[int, str] = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in first_cap:
            first_cap[ann["image_id"]] = ann["caption"]
    # Filenames sorted alphabetically — matches extraction order
    sorted_fnames = sorted([img_id_to_fn[iid] for iid in first_cap.keys()])
    fname_to_idx = {fn: i for i, fn in enumerate(sorted_fnames)}
    # Build fname → (idx, caption)
    result = {}
    for iid, cap in first_cap.items():
        fn = img_id_to_fn[iid]
        if fn in fname_to_idx:
            result[fn] = (fname_to_idx[fn], cap, iid)
    return result


def pick_diverse_images(fname_to_info: dict, n: int = 12) -> list[tuple]:
    """Pick n diverse COCO images spread across the dataset.

    Returns list of (fname, idx, caption).
    """
    sorted_fnames = sorted(fname_to_info.keys())
    N = len(sorted_fnames)
    # Spread selections: pick every N/n indices
    step = N // n
    picks = []
    for i in range(n):
        idx = (i * step + 1000) % N  # offset to avoid earliest files
        fname = sorted_fnames[idx]
        file_idx, caption, iid = fname_to_info[fname]
        picks.append((fname, file_idx, caption, iid))
    return picks


def get_tier_from_rank(ranks: np.ndarray) -> np.ndarray:
    """Convert normalized ranks (0..1) to tier indices (0..3).

    Tier 0: rank < 0.25 (salient)
    Tier 1: 0.25 <= rank < 0.5
    Tier 2: 0.5 <= rank < 0.75
    Tier 3: rank >= 0.75 (peripheral)
    """
    tiers = np.zeros_like(ranks, dtype=np.int32)
    tiers[ranks >= 0.25] = 1
    tiers[ranks >= 0.50] = 2
    tiers[ranks >= 0.75] = 3
    return tiers


def load_image(fname: str) -> np.ndarray:
    """Load COCO image and resize to 224x224 (matches DINOv2 input)."""
    path = DATA_DIR / "coco" / "train2017" / fname
    img = Image.open(path).convert("RGB")
    # Match our preprocessing: resize 256 → center crop 224
    img = img.resize((256, 256), Image.BICUBIC)
    w, h = img.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    return np.array(img)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap on an image. heatmap is (H, W) in [0, 1]."""
    # Resize heatmap from 16x16 to 224x224
    from matplotlib.cm import get_cmap
    hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    hm_pil = hm_pil.resize((224, 224), Image.BILINEAR)
    hm_arr = np.array(hm_pil).astype(np.float32) / 255.0
    # Apply colormap
    cmap = get_cmap("jet")
    hm_rgb = (cmap(hm_arr)[..., :3] * 255).astype(np.uint8)
    # Blend
    result = (img.astype(np.float32) * (1 - alpha) + hm_rgb.astype(np.float32) * alpha)
    return result.astype(np.uint8)


def overlay_tiers(img: np.ndarray, tiers: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay 4-tier colored regions on image.

    tiers: (256,) array with values 0-3 mapping to 16x16 patch grid.
    """
    tier_grid = tiers.reshape(16, 16)
    # Build colored tier image by upsampling each patch
    tier_rgb = np.zeros((16, 16, 3), dtype=np.float32)
    for t in range(4):
        mask = tier_grid == t
        for c in range(3):
            tier_rgb[:, :, c][mask] = TIER_COLORS_RGB[t][c]
    tier_pil = Image.fromarray((tier_rgb * 255).astype(np.uint8))
    tier_pil = tier_pil.resize((224, 224), Image.NEAREST)
    tier_arr = np.array(tier_pil)
    # Blend
    result = (img.astype(np.float32) * (1 - alpha) + tier_arr.astype(np.float32) * alpha)
    return result.astype(np.uint8)


# ────────────────────────────────────────────────────────────────────
# Part 1: Image saliency visualization
# ────────────────────────────────────────────────────────────────────

def part1_image_saliency(picks: list[tuple], img_cls_attn: torch.Tensor) -> dict:
    """Generate image saliency visualization and return per-image info."""
    logger.info("Part 1: Image saliency visualization (%d images)", len(picks))

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    fig.suptitle("DINOv2 CLS Attention Saliency Tiers", fontsize=14, y=0.995)

    per_image_info = {}

    for row, (fname, file_idx, caption, iid) in enumerate(picks):
        img = load_image(fname)
        attn = img_cls_attn[file_idx].numpy()  # (256,)

        # Compute ranks (0 = highest attention)
        order = np.argsort(-attn)
        ranks_norm = np.empty_like(attn)
        ranks_norm[order] = np.arange(len(attn)) / (len(attn) - 1)
        tiers = get_tier_from_rank(ranks_norm)

        # Heatmap (normalized attention)
        attn_normed = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        heatmap = attn_normed.reshape(16, 16)

        # Col 1: original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{fname}\n{caption[:60]}", fontsize=7)
        axes[row, 0].axis("off")

        # Col 2: continuous heatmap
        axes[row, 1].imshow(overlay_heatmap(img, heatmap, alpha=0.55))
        if row == 0:
            axes[row, 1].set_title("CLS attention (jet)", fontsize=9)
        axes[row, 1].axis("off")

        # Col 3: tier assignment
        axes[row, 2].imshow(overlay_tiers(img, tiers, alpha=0.5))
        if row == 0:
            axes[row, 2].set_title("4-tier assignment", fontsize=9)
        axes[row, 2].axis("off")

        # Check spatial coherence of top-25% tier
        top_tier_mask = (tiers == 0).reshape(16, 16)
        # Count connected components (crude)
        from scipy.ndimage import label
        _, num_components = label(top_tier_mask)
        per_image_info[fname] = {
            "caption": caption,
            "num_top_tier_components": int(num_components),
            "top_tier_attn_sum": float(attn[tiers == 0].sum()),
            "tier_counts": {t: int((tiers == t).sum()) for t in range(4)},
        }

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS_RGB[i], label=TIER_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    out_path = OUT_DIR / "img_saliency_tiers.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)

    return per_image_info


# ────────────────────────────────────────────────────────────────────
# Part 2: Text saliency visualization
# ────────────────────────────────────────────────────────────────────

def part2_text_saliency(
    picks: list[tuple],
    txt_cls_attn: torch.Tensor,
    txt_mask: torch.Tensor,
    tokenizer,
) -> dict:
    """Generate text saliency visualization and return per-caption info."""
    logger.info("Part 2: Text saliency visualization (%d captions)", len(picks))

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(14, n * 1.2))
    fig.suptitle("all-mpnet CLS Attention Saliency Tiers (per token)",
                 fontsize=14, y=1.00)

    per_caption_info = {}

    for row, (fname, file_idx, caption, iid) in enumerate(picks):
        # Tokenize the caption to get actual token strings
        enc = tokenizer(caption, padding=False, truncation=True, max_length=128,
                        return_tensors="pt")
        token_ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Get CLS attention for this caption
        # Note: cls attention is saved with shape (N, max_len-1) — CLS excluded
        attn = txt_cls_attn[file_idx].numpy()  # (max_len - 1,)
        mask = txt_mask[file_idx].numpy()  # (max_len,) — includes CLS

        # CLS attention skips CLS token (index 0). Word tokens are at positions 1..L-1.
        # Take only valid (non-padding) positions
        valid_mask = mask[1:] > 0  # word tokens validity
        n_valid = valid_mask.sum()
        # Number of actual word tokens from tokenizer (excluding CLS)
        n_tokens_tok = len(tokens) - 1  # skip [CLS] at index 0 (for BERT-style) or <s>
        # Align lengths: use min of tokens-1 and n_valid
        n_use = min(n_tokens_tok, int(n_valid))
        attn_valid = attn[:n_use]
        word_tokens = tokens[1:1 + n_use]  # skip first special token

        # Compute ranks within valid tokens
        if n_use > 1:
            order = np.argsort(-attn_valid)
            ranks_norm = np.empty_like(attn_valid)
            ranks_norm[order] = np.arange(n_use) / (n_use - 1)
        else:
            ranks_norm = np.zeros(n_use)
        tiers = get_tier_from_rank(ranks_norm)

        # Bar chart
        ax = axes[row] if n > 1 else axes
        x_pos = np.arange(n_use)
        colors = [TIER_COLORS_RGB[t] for t in tiers]
        ax.bar(x_pos, attn_valid, color=colors)
        # Clean token labels: replace special chars
        clean_tokens = []
        for tok in word_tokens:
            t = tok.replace("▁", "").replace("Ġ", "").replace("</s>", "[/s]")
            t = t.replace("<s>", "[s]").replace("<pad>", "[P]")
            clean_tokens.append(t if t else "_")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clean_tokens, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{fname[:20]}: {caption[:70]}", fontsize=8, loc="left")
        ax.set_ylabel("attn", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)

        # Content word analysis
        is_content = [t.lower() not in STOPWORDS and len(t.strip()) > 0
                       for t in clean_tokens]
        tier0_content = sum(1 for i, c in enumerate(is_content)
                            if tiers[i] == 0 and c)
        tier0_total = int((tiers == 0).sum())
        tier3_content = sum(1 for i, c in enumerate(is_content)
                            if tiers[i] == 3 and c)
        tier3_total = int((tiers == 3).sum())

        per_caption_info[fname] = {
            "caption": caption,
            "tokens": clean_tokens,
            "tiers": tiers.tolist(),
            "attn_values": attn_valid.tolist(),
            "tier0_content_ratio": tier0_content / max(tier0_total, 1),
            "tier3_content_ratio": tier3_content / max(tier3_total, 1),
            "tier0_tokens": [clean_tokens[i] for i in range(n_use) if tiers[i] == 0],
            "tier3_tokens": [clean_tokens[i] for i in range(n_use) if tiers[i] == 3],
        }

    plt.tight_layout(rect=[0, 0.0, 1, 0.98])
    out_path = OUT_DIR / "txt_saliency_tiers.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)

    return per_caption_info


# ────────────────────────────────────────────────────────────────────
# Part 3: Cross-modal figure
# ────────────────────────────────────────────────────────────────────

def part3_cross_modal(picks: list[tuple], img_cls_attn, txt_cls_attn, txt_mask, tokenizer):
    """Combined image + text tier figure."""
    logger.info("Part 3: Cross-modal saliency comparison")

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 3),
                              gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle("Cross-Modal Saliency Tiers (Image ↔ Text)", fontsize=14, y=0.995)

    for row, (fname, file_idx, caption, iid) in enumerate(picks):
        img = load_image(fname)

        # Image tiers
        attn_i = img_cls_attn[file_idx].numpy()
        order_i = np.argsort(-attn_i)
        ranks_i = np.empty_like(attn_i)
        ranks_i[order_i] = np.arange(len(attn_i)) / (len(attn_i) - 1)
        tiers_i = get_tier_from_rank(ranks_i)

        axes[row, 0].imshow(overlay_tiers(img, tiers_i, alpha=0.5))
        axes[row, 0].set_title(fname, fontsize=8)
        axes[row, 0].axis("off")

        # Text tiers
        enc = tokenizer(caption, padding=False, truncation=True, max_length=128,
                        return_tensors="pt")
        token_ids = enc["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        attn_t = txt_cls_attn[file_idx].numpy()
        mask_t = txt_mask[file_idx].numpy()
        valid = mask_t[1:] > 0
        n_valid = int(valid.sum())
        n_use = min(len(tokens) - 1, n_valid)
        attn_valid = attn_t[:n_use]
        word_tokens = tokens[1:1 + n_use]
        clean = [w.replace("▁", "").replace("Ġ", "") or "_" for w in word_tokens]

        if n_use > 1:
            order_t = np.argsort(-attn_valid)
            ranks_t = np.empty_like(attn_valid)
            ranks_t[order_t] = np.arange(n_use) / (n_use - 1)
        else:
            ranks_t = np.zeros(n_use)
        tiers_t = get_tier_from_rank(ranks_t)

        ax = axes[row, 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Render tokens as colored text
        x, y = 0.01, 0.7
        for i, tok in enumerate(clean):
            color = TIER_COLORS_RGB[tiers_t[i]]
            ax.text(x, y, tok, color=color, fontsize=10, fontweight="bold",
                    transform=ax.transAxes)
            x += 0.015 * (len(tok) + 1.5)
            if x > 0.95:
                x = 0.01
                y -= 0.25

        # Show tier 0 and tier 3 token lists below
        tier0_toks = [clean[i] for i in range(n_use) if tiers_t[i] == 0]
        tier3_toks = [clean[i] for i in range(n_use) if tiers_t[i] == 3]
        ax.text(0.01, 0.25,
                f"Tier 0 (salient): {', '.join(tier0_toks)}",
                color="red", fontsize=8, transform=ax.transAxes)
        ax.text(0.01, 0.10,
                f"Tier 3 (peripheral): {', '.join(tier3_toks)}",
                color="gray", fontsize=8, transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0.0, 1, 0.98])
    out_path = OUT_DIR / "cross_modal_tiers.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ────────────────────────────────────────────────────────────────────
# Part 4: Quantitative summary
# ────────────────────────────────────────────────────────────────────

def part4_write_summary(img_info: dict, txt_info: dict):
    logger.info("Part 4: Writing quantitative summary")

    # Aggregate stats
    avg_num_components = np.mean([v["num_top_tier_components"]
                                   for v in img_info.values()])
    avg_top_tier_mass = np.mean([v["top_tier_attn_sum"]
                                  for v in img_info.values()])

    avg_tier0_content = np.mean([v["tier0_content_ratio"]
                                  for v in txt_info.values()])
    avg_tier3_content = np.mean([v["tier3_content_ratio"]
                                  for v in txt_info.values()])

    lines = [
        "# CLS Attention Saliency Analysis",
        "",
        "Diagnostic check: do the CLS attention rankings from DINOv2 (image) and "
        "all-mpnet-base-v2 (text) produce meaningful tier assignments for the HME "
        "diversity loss?",
        "",
        "## Image Saliency (DINOv2 ViT-B/14)",
        "",
        f"- Images analyzed: {len(img_info)}",
        f"- Avg connected components in top-25% tier: **{avg_num_components:.2f}**",
        f"- Avg attention mass in top-25% tier: **{avg_top_tier_mass:.4f}** "
        f"(vs uniform 0.25 of total ≈ 0.25)",
        "",
        "Interpretation: lower component count = more spatially coherent salient regions.",
        "Higher attention mass = top-25% dominates.",
        "",
        "### Per-image details",
        "",
        "| Image | Num components (top-25%) | Top-25% attn mass | Caption |",
        "|-------|-------------------------|-------------------|---------|",
    ]
    for fname, info in img_info.items():
        cap = info["caption"][:50].replace("|", "\\|")
        lines.append(
            f"| {fname} | {info['num_top_tier_components']} | "
            f"{info['top_tier_attn_sum']:.4f} | {cap} |"
        )

    lines.extend([
        "",
        "## Text Saliency (all-mpnet-base-v2)",
        "",
        f"- Captions analyzed: {len(txt_info)}",
        f"- Avg content-word ratio in Tier 0 (salient): **{avg_tier0_content:.2%}**",
        f"- Avg content-word ratio in Tier 3 (peripheral): **{avg_tier3_content:.2%}**",
        "",
        "Interpretation: if text CLS attention is semantically meaningful, Tier 0 "
        "should have higher content-word ratio than Tier 3. If ratios are similar, "
        "CLS attention is NOT a good saliency signal for text.",
        "",
        "### Per-caption tier assignments",
        "",
        "| Image | Tier 0 (salient) tokens | Tier 3 (peripheral) tokens |",
        "|-------|-------------------------|----------------------------|",
    ])
    for fname, info in txt_info.items():
        t0 = ", ".join(info["tier0_tokens"])[:60]
        t3 = ", ".join(info["tier3_tokens"])[:60]
        lines.append(f"| {fname} | {t0} | {t3} |")

    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    if avg_tier0_content > avg_tier3_content + 0.15:
        lines.append(
            "**Text CLS attention IS meaningful:** Tier 0 has substantially higher "
            f"content-word ratio ({avg_tier0_content:.2%}) than Tier 3 "
            f"({avg_tier3_content:.2%}). Semantically important words (nouns, verbs) "
            "are ranked higher than function words."
        )
    elif avg_tier0_content < avg_tier3_content - 0.15:
        lines.append(
            "**Text CLS attention is INVERTED:** Tier 0 (highest attention) has LOWER "
            f"content-word ratio ({avg_tier0_content:.2%}) than Tier 3 "
            f"({avg_tier3_content:.2%}). The encoder attends more to function words "
            "(articles, prepositions) than content words — a known artifact of some "
            "language models. Our HME tier assignment would be backward for text."
        )
    else:
        lines.append(
            "**Text CLS attention is NEUTRAL:** Content-word ratio is similar in "
            f"Tier 0 ({avg_tier0_content:.2%}) and Tier 3 ({avg_tier3_content:.2%}). "
            "CLS attention does not meaningfully separate content from function "
            "words — the tier assignment provides little semantic signal for text."
        )

    lines.append("")
    if avg_num_components <= 3:
        lines.append(
            "**Image CLS attention IS meaningful:** Top-25% tier forms spatially "
            f"coherent regions (avg {avg_num_components:.2f} components). The "
            "encoder focuses on localized salient objects."
        )
    else:
        lines.append(
            "**Image CLS attention is FRAGMENTED:** Top-25% tier is scattered "
            f"across the image (avg {avg_num_components:.2f} components). This may "
            "still be semantically meaningful (object parts), but the tier boundaries "
            "are not a simple foreground/background split."
        )

    out_path = OUT_DIR / "saliency_analysis.md"
    out_path.write_text("\n".join(lines))
    logger.info("Saved %s", out_path)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading caption map...")
    fname_to_info = load_train_caption_map()
    logger.info("Loaded %d captions", len(fname_to_info))

    logger.info("Loading pre-extracted CLS attention maps...")
    img_cls_attn = torch.load(
        EMB_DIR / "coco_train_img_cls_attn.pt", map_location="cpu", weights_only=True,
    )
    txt_cls_attn = torch.load(
        EMB_DIR / "coco_train_txt_cls_attn.pt", map_location="cpu", weights_only=True,
    )
    txt_mask = torch.load(
        EMB_DIR / "coco_train_txt_mask.pt", map_location="cpu", weights_only=True,
    )
    logger.info("img_cls_attn: %s, txt_cls_attn: %s, txt_mask: %s",
                tuple(img_cls_attn.shape), tuple(txt_cls_attn.shape),
                tuple(txt_mask.shape))

    logger.info("Loading all-mpnet tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    picks = pick_diverse_images(fname_to_info, n=12)
    logger.info("Selected %d images:", len(picks))
    for fname, idx, cap, iid in picks:
        logger.info("  [%6d] %s: %s", idx, fname, cap[:60])

    img_info = part1_image_saliency(picks, img_cls_attn)
    txt_info = part2_text_saliency(picks, txt_cls_attn, txt_mask, tokenizer)
    part3_cross_modal(picks, img_cls_attn, txt_cls_attn, txt_mask, tokenizer)
    part4_write_summary(img_info, txt_info)

    logger.info("All done! Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
