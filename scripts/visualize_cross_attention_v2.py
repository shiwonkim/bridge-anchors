"""Visualize cross-attention maps overlaid on original Flickr30k images."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.bridge_anchors import BridgeAnchorAligner

OUT_DIR = Path("experiments/exp_cross_attention/attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = Path("results/checkpoints/ca_tau_0.05/best.pt")
TOKEN_DIR = Path("data/embeddings/all_tokens")
FLICKR_IMG_DIR = Path("data/datasets/flickr30k/flickr30k_images")
FLICKR_CAPTION_FILE = Path("data/datasets/flickr30k/results_20130124.token")

SAMPLE_INDICES = [0, 500, 3000, 10000, 25000]


def load_flickr_index() -> list[tuple[str, str]]:
    """Load sorted Flickr30k filenames and first captions.

    Returns list of (filename, first_caption) in the same order as embeddings.
    """
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
        p = FLICKR_IMG_DIR / fname
        if p.exists():
            result.append((fname, first_caption[fname]))
    return result


def load_model() -> BridgeAnchorAligner:
    """Load the best cross-attention model."""
    ckpt = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
    model = BridgeAnchorAligner(
        dim_img=768, dim_txt=768, num_anchors=128,
        token_pool="cross_attn", pool_temperature=0.05,
        img_input="tokens", txt_input="cls",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_attention(
    model: BridgeAnchorAligner,
    img_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute attention weights: (257, K)."""
    with torch.no_grad():
        a_img = F.normalize(model.anchors_img, dim=-1)
        img_emb = F.normalize(img_tokens, dim=-1)
        sim = img_emb @ a_img.T
        attn = F.softmax(sim / model.pool_temperature, dim=1)
    return attn[0]


def plot_attention_overlay(
    attn: torch.Tensor,
    img_path: Path,
    caption: str,
    flickr_idx: int,
    sample_idx: int,
    out_path: Path,
) -> None:
    """Plot top 6 anchor attention heatmaps overlaid on original image."""
    # Load and resize original image
    orig_img = Image.open(img_path).convert("RGB")
    orig_img = orig_img.resize((224, 224), Image.BILINEAR)
    orig_arr = np.array(orig_img)

    # Patch attention (skip CLS)
    patch_attn = attn[1:, :]  # (256, K)

    # Top 6 anchors by max attention
    max_attn_per_anchor = patch_attn.max(dim=0).values
    top6 = max_attn_per_anchor.topk(6).indices.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    # Truncate caption for display
    cap_display = caption[:100] + "..." if len(caption) > 100 else caption
    fig.suptitle(
        f"Flickr30k #{flickr_idx} — \"{cap_display}\"\n"
        f"Top 6 Anchor Attention Maps (tau=0.05)",
        fontsize=11, y=0.98,
    )

    for i, anchor_k in enumerate(top6):
        ax = axes[i // 3, i % 3]

        # Show original image
        ax.imshow(orig_arr)

        # Upsample 16×16 attention to 224×224
        hmap_16 = patch_attn[:, anchor_k].reshape(1, 1, 16, 16)
        hmap_224 = F.interpolate(
            hmap_16, size=(224, 224), mode="bilinear", align_corners=False,
        )[0, 0].numpy()

        # Overlay heatmap
        ax.imshow(hmap_224, cmap="jet", alpha=0.5, vmin=0, vmax=hmap_224.max())

        vmin, vmax = patch_attn[:, anchor_k].min().item(), patch_attn[:, anchor_k].max().item()
        ax.set_title(f"Anchor {anchor_k}  (max={vmax:.4f})", fontsize=10)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading Flickr30k index...")
    flickr_index = load_flickr_index()
    print(f"  {len(flickr_index)} images indexed")

    print("Loading model...")
    model = load_model()

    print("Loading Flickr30k image tokens...")
    flickr_img = torch.load(
        TOKEN_DIR / "flickr30k_test_img.pt", weights_only=True,
    ).float()
    print(f"  Shape: {flickr_img.shape}")

    # Print mapping table
    print(f"\n{'Sample':>6}  {'Flickr Idx':>10}  {'Filename':<25}  Caption")
    print("-" * 90)
    for i, idx in enumerate(SAMPLE_INDICES):
        fname, caption = flickr_index[idx]
        cap_short = caption[:50] + "..." if len(caption) > 50 else caption
        print(f"{i:>6}  {idx:>10}  {fname:<25}  {cap_short}")

    # Generate attention maps
    for i, idx in enumerate(SAMPLE_INDICES):
        fname, caption = flickr_index[idx]
        img_path = FLICKR_IMG_DIR / fname
        print(f"\nProcessing sample {i} (Flickr #{idx}, {fname})...")

        img_tokens = flickr_img[idx:idx + 1]
        attn = compute_attention(model, img_tokens)

        out_name = f"attn_flickr{idx:05d}_sample{i}.png"
        out_path = OUT_DIR / out_name
        plot_attention_overlay(attn, img_path, caption, idx, i, out_path)
        print(f"  Saved {out_path}")

    print(f"\nAll visualizations saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
