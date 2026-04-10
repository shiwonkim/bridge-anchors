"""Visualize cross-modal attention maps from the tok/tok CA model.

Shows both image spatial attention and text word attention for each anchor,
revealing cross-modal correspondence.
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.bridge_anchors import BridgeAnchorAligner

OUT_DIR = Path("experiments/exp_cross_attention/attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = Path("results/checkpoints/1_2a_toktok_ca/best.pt")
TOKEN_DIR = Path("data/embeddings/all_tokens")
FLICKR_IMG_DIR = Path("data/datasets/flickr30k/flickr30k_images")
FLICKR_CAPTION_FILE = Path("data/datasets/flickr30k/results_20130124.token")

SAMPLE_INDICES = [0, 500, 3000, 10000, 25000]
TAU = 0.05


def load_flickr_index() -> list[tuple[str, str]]:
    """Load sorted Flickr30k filenames and first captions."""
    first_caption: dict[str, str] = {}
    with open(FLICKR_CAPTION_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            key, caption = parts
            filename = key.rsplit("#", 1)[0]
            if filename not in first_caption:
                first_caption[filename] = caption.strip()
    sorted_fnames = sorted(first_caption.keys())
    result = []
    for fname in sorted_fnames:
        if (FLICKR_IMG_DIR / fname).exists():
            result.append((fname, first_caption[fname]))
    return result


def load_model() -> BridgeAnchorAligner:
    ckpt = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
    model = BridgeAnchorAligner(
        dim_img=768, dim_txt=768, num_anchors=128,
        token_pool="cross_attn", pool_temperature=TAU,
        img_input="tokens", txt_input="tokens",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def tokenize_caption(caption: str, max_len: int) -> list[str]:
    """Tokenize caption using the same tokenizer as sentence-transformers."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoded = tokenizer(caption, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
    token_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens


def compute_attentions(
    model: BridgeAnchorAligner,
    img_tokens: torch.Tensor,
    txt_tokens: torch.Tensor,
    txt_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute image and text attention weights.

    Returns:
        attn_img: (257, K) image attention
        attn_txt: (M, K) text attention
    """
    with torch.no_grad():
        a_img = F.normalize(model.anchors_img, dim=-1)
        a_txt = F.normalize(model.anchors_txt, dim=-1)

        img_n = F.normalize(img_tokens, dim=-1)
        sim_img = img_n @ a_img.T
        attn_img = F.softmax(sim_img / TAU, dim=1)

        txt_n = F.normalize(txt_tokens, dim=-1)
        sim_txt = txt_n @ a_txt.T
        sim_txt_masked = sim_txt.masked_fill(
            ~txt_mask.bool().unsqueeze(-1), float("-inf"),
        )
        attn_txt = F.softmax(sim_txt_masked / TAU, dim=1)

    return attn_img[0], attn_txt[0]  # (257, K), (M, K)


def plot_cross_modal_attention(
    attn_img: torch.Tensor,
    attn_txt: torch.Tensor,
    txt_mask: torch.Tensor,
    word_tokens: list[str],
    img_path: Path,
    caption: str,
    flickr_idx: int,
    sample_idx: int,
    out_path: Path,
) -> None:
    """Plot cross-modal attention for top 4 anchors."""
    orig_img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    orig_arr = np.array(orig_img)

    patch_attn = attn_img[1:, :]  # (256, K)
    max_attn_per_anchor = patch_attn.max(dim=0).values
    top4 = max_attn_per_anchor.topk(4).indices.tolist()

    # Valid text tokens
    mask = txt_mask.bool()
    n_valid = mask.sum().item()

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.6, 1, 1, 1, 1],
                           hspace=0.35, wspace=0.3)

    # Top: caption
    ax_cap = fig.add_subplot(gs[0, :])
    cap_display = caption[:120] + "..." if len(caption) > 120 else caption
    ax_cap.text(0.5, 0.5, f'Flickr30k #{flickr_idx}\n"{cap_display}"',
                ha="center", va="center", fontsize=11, wrap=True,
                style="italic")
    ax_cap.axis("off")

    for i, anchor_k in enumerate(top4):
        # Image attention (left column)
        ax_img = fig.add_subplot(gs[i + 1, 0])
        ax_img.imshow(orig_arr)
        hmap = patch_attn[:, anchor_k].reshape(1, 1, 16, 16)
        hmap_224 = F.interpolate(hmap, size=(224, 224), mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
        ax_img.imshow(hmap_224, cmap="jet", alpha=0.5, vmin=0, vmax=hmap_224.max())
        ax_img.set_title(f"Anchor {anchor_k} — Image", fontsize=10)
        ax_img.axis("off")

        # Text attention (right column)
        ax_txt = fig.add_subplot(gs[i + 1, 1])
        txt_weights = attn_txt[:n_valid, anchor_k].numpy()
        valid_tokens = word_tokens[:n_valid]

        # Clean up tokens for display
        display_tokens = []
        for t in valid_tokens:
            if t in ("<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"):
                display_tokens.append(t)
            elif t.startswith("▁"):
                display_tokens.append(t[1:])
            else:
                display_tokens.append(t)

        colors = ["#d62728" if w == txt_weights.max() else "#1f77b4"
                  for w in txt_weights]
        y_pos = np.arange(len(display_tokens))
        ax_txt.barh(y_pos, txt_weights, color=colors, height=0.7)
        ax_txt.set_yticks(y_pos)
        ax_txt.set_yticklabels(display_tokens, fontsize=8)
        ax_txt.invert_yaxis()
        ax_txt.set_title(f"Anchor {anchor_k} — Text", fontsize=10)
        ax_txt.set_xlabel("Attention weight", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_anchor_roles(
    model: BridgeAnchorAligner,
    flickr_txt: torch.Tensor,
    flickr_mask: torch.Tensor,
    flickr_index: list[tuple[str, str]],
    n_samples: int = 200,
) -> dict[int, Counter]:
    """For each anchor, collect most-attended text tokens across samples."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    K = model.num_anchors
    anchor_words: dict[int, Counter] = {k: Counter() for k in range(K)}

    # Use evenly spaced samples
    indices = np.linspace(0, len(flickr_index) - 1, n_samples, dtype=int)

    with torch.no_grad():
        a_txt = F.normalize(model.anchors_txt, dim=-1)

        for idx in indices:
            txt_tok = flickr_txt[idx:idx + 1].float()
            mask = flickr_mask[idx:idx + 1]

            txt_n = F.normalize(txt_tok, dim=-1)
            sim = txt_n @ a_txt.T
            sim_masked = sim.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
            attn = F.softmax(sim_masked / TAU, dim=1)[0]  # (M, K)

            # Get word tokens
            _, caption = flickr_index[idx]
            max_len = txt_tok.shape[1]
            encoded = tokenizer(caption, padding="max_length", truncation=True,
                                max_length=max_len, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            n_valid = mask[0].sum().item()

            for k in range(K):
                top_idx = attn[:n_valid, k].argmax().item()
                word = tokens[top_idx]
                # Clean up
                if word.startswith("▁"):
                    word = word[1:]
                if word not in ("<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", ""):
                    anchor_words[k][word] += 1

    return anchor_words


def plot_anchor_roles(
    anchor_words: dict[int, Counter],
    out_path: Path,
    top_n_anchors: int = 10,
    top_n_words: int = 5,
) -> None:
    """Plot anchor semantic role summary."""
    # Find most-used anchors (by total word count)
    anchor_usage = {k: sum(c.values()) for k, c in anchor_words.items()}
    top_anchors = sorted(anchor_usage, key=anchor_usage.get, reverse=True)[:top_n_anchors]

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    for i, k in enumerate(top_anchors):
        top_words = anchor_words[k].most_common(top_n_words)
        word_str = ", ".join(f"{w} ({c})" for w, c in top_words)
        labels.append(f"Anchor {k:>3d}: {word_str}")

    y_pos = np.arange(len(labels))
    counts = [anchor_usage[k] for k in top_anchors]
    ax.barh(y_pos, counts, color="#1f77b4", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Total top-word assignments (across 200 samples)")
    ax.set_title("Top 10 Anchors — Most Commonly Attended Text Words")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_anchor_usage(
    model: BridgeAnchorAligner,
    flickr_img: torch.Tensor,
    out_path: Path,
) -> None:
    """Plot anchor dominance histogram for tok/tok model."""
    K = model.num_anchors
    dominant_counts = torch.zeros(K, dtype=torch.long)

    with torch.no_grad():
        a_img = F.normalize(model.anchors_img, dim=-1)
        for start in range(0, flickr_img.shape[0], 256):
            batch = F.normalize(flickr_img[start:start + 256].float(), dim=-1)
            sim = batch @ a_img.T
            attn = F.softmax(sim / TAU, dim=1)
            patch_attn = attn[:, 1:, :]
            max_per_anchor = patch_attn.max(dim=1).values
            dominant = max_per_anchor.argmax(dim=1)
            for d in dominant:
                dominant_counts[d] += 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(K), dominant_counts.numpy(), color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Anchor Index")
    ax.set_ylabel("# Images Where This Anchor Is Dominant")
    ax.set_title(
        f"tok/tok CA — Anchor Dominance Histogram ({flickr_img.shape[0]} images)\n"
        f"Active: {(dominant_counts > 0).sum()}/{K}, "
        f"Max: {dominant_counts.max()}, Min: {dominant_counts.min()}"
    )
    ax.set_xlim(-1, K)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Active anchors: {(dominant_counts > 0).sum()}/{K}")


def main():
    print("Loading Flickr30k index...")
    flickr_index = load_flickr_index()
    print(f"  {len(flickr_index)} images")

    print("Loading model...")
    model = load_model()

    print("Loading data...")
    flickr_img = torch.load(TOKEN_DIR / "flickr30k_test_img.pt", weights_only=True)
    flickr_txt = torch.load(TOKEN_DIR / "flickr30k_test_txt_tokens.pt", weights_only=True)
    flickr_mask = torch.load(TOKEN_DIR / "flickr30k_test_txt_mask.pt", weights_only=True)
    print(f"  img: {flickr_img.shape}, txt: {flickr_txt.shape}, mask: {flickr_mask.shape}")

    max_len = flickr_txt.shape[1]

    # Mapping table
    print(f"\n{'Sample':>6}  {'Idx':>6}  {'Filename':<25}  Caption")
    print("-" * 85)
    for i, idx in enumerate(SAMPLE_INDICES):
        fname, cap = flickr_index[idx]
        print(f"{i:>6}  {idx:>6}  {fname:<25}  {cap[:45]}...")

    # Per-sample cross-modal attention
    for i, idx in enumerate(SAMPLE_INDICES):
        fname, caption = flickr_index[idx]
        print(f"\nSample {i} (#{idx}, {fname})...")

        img_tok = flickr_img[idx:idx + 1].float()
        txt_tok = flickr_txt[idx:idx + 1].float()
        mask = flickr_mask[idx:idx + 1]

        attn_img, attn_txt = compute_attentions(model, img_tok, txt_tok, mask)
        word_tokens = tokenize_caption(caption, max_len)

        out_path = OUT_DIR / f"toktok_attn_flickr{idx:05d}_sample{i}.png"
        plot_cross_modal_attention(
            attn_img, attn_txt, mask[0], word_tokens,
            FLICKR_IMG_DIR / fname, caption, idx, i, out_path,
        )
        print(f"  Saved {out_path}")

    # Anchor usage histogram
    print("\nComputing anchor usage histogram...")
    plot_anchor_usage(model, flickr_img, OUT_DIR.parent / "toktok_anchor_usage_histogram.png")

    # Anchor semantic roles
    print("\nComputing anchor semantic roles (200 samples)...")
    anchor_words = compute_anchor_roles(model, flickr_txt, flickr_mask, flickr_index, n_samples=200)
    plot_anchor_roles(anchor_words, OUT_DIR.parent / "anchor_roles_summary.png")
    print("  Saved anchor_roles_summary.png")

    # Print top-10 anchor roles
    print("\nTop 10 anchor roles:")
    anchor_usage = {k: sum(c.values()) for k, c in anchor_words.items()}
    top10 = sorted(anchor_usage, key=anchor_usage.get, reverse=True)[:10]
    for k in top10:
        top_words = anchor_words[k].most_common(5)
        word_str = ", ".join(f"{w}({c})" for w, c in top_words)
        print(f"  Anchor {k:>3d}: {word_str}")

    print(f"\nAll saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
