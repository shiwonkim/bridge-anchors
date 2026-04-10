"""Visualize cross-attention maps from the best cross-attention pooling model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.bridge_anchors import BridgeAnchorAligner

OUT_DIR = Path("experiments/exp_cross_attention/attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = Path("results/checkpoints/ca_tau_0.05/best.pt")
TOKEN_DIR = Path("data/embeddings/all_tokens")

# Diverse sample indices from Flickr30k test set (31783 images)
SAMPLE_INDICES = [0, 500, 3000, 10000, 25000]


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
    """Compute attention weights for image tokens.

    Args:
        model: Trained BridgeAnchorAligner.
        img_tokens: (1, 257, 768) image token embeddings.

    Returns:
        attn: (257, K) attention weights.
    """
    with torch.no_grad():
        a_img = F.normalize(model.anchors_img, dim=-1)  # (K, D)
        img_emb = F.normalize(img_tokens, dim=-1)        # (1, 257, D)
        sim = img_emb @ a_img.T                           # (1, 257, K)
        attn = F.softmax(sim / model.pool_temperature, dim=1)  # (1, 257, K)
    return attn[0]  # (257, K)


def plot_attention_maps(
    attn: torch.Tensor,
    sample_idx: int,
    out_path: Path,
) -> None:
    """Plot top 6 anchor attention heatmaps for one image.

    Args:
        attn: (257, K) attention weights.
        sample_idx: Image index (for title).
        out_path: Output path.
    """
    # Patch attention only (skip CLS token at index 0)
    patch_attn = attn[1:, :]  # (256, K)

    # Find top 6 anchors by max attention value
    max_attn_per_anchor = patch_attn.max(dim=0).values  # (K,)
    top6 = max_attn_per_anchor.topk(6).indices.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Image {sample_idx} — Top 6 Anchor Attention Maps (tau=0.05)",
        fontsize=14,
    )

    for i, anchor_k in enumerate(top6):
        ax = axes[i // 3, i % 3]
        hmap = patch_attn[:, anchor_k].reshape(16, 16).numpy()
        im = ax.imshow(hmap, cmap="hot", interpolation="bilinear")
        vmin, vmax = hmap.min(), hmap.max()
        ax.set_title(
            f"Anchor {anchor_k}\n"
            f"min={vmin:.4f}, max={vmax:.4f}",
            fontsize=10,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_anchor_usage_histogram(
    all_flickr_img: torch.Tensor,
    model: BridgeAnchorAligner,
    out_path: Path,
) -> None:
    """Plot histogram of dominant anchor usage across all Flickr30k images.

    Args:
        all_flickr_img: (N, 257, 768) all Flickr30k image token embeddings.
        model: Trained model.
        out_path: Output path.
    """
    K = model.num_anchors
    dominant_counts = torch.zeros(K, dtype=torch.long)

    with torch.no_grad():
        a_img = F.normalize(model.anchors_img, dim=-1)  # (K, D)
        # Process in batches
        batch_size = 256
        for start in range(0, all_flickr_img.shape[0], batch_size):
            batch = all_flickr_img[start:start + batch_size]  # (B, 257, D)
            batch = F.normalize(batch, dim=-1)
            sim = batch @ a_img.T  # (B, 257, K)
            attn = F.softmax(sim / model.pool_temperature, dim=1)  # (B, 257, K)
            # For each image, which anchor has the highest max-attention?
            patch_attn = attn[:, 1:, :]  # (B, 256, K) — skip CLS
            max_per_anchor = patch_attn.max(dim=1).values  # (B, K)
            dominant = max_per_anchor.argmax(dim=1)  # (B,)
            for d in dominant:
                dominant_counts[d] += 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(K), dominant_counts.numpy(), color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Anchor Index")
    ax.set_ylabel("# Images Where This Anchor Is Dominant")
    ax.set_title(
        f"Anchor Dominance Histogram — Flickr30k Test ({all_flickr_img.shape[0]} images)\n"
        f"Active: {(dominant_counts > 0).sum()}/{K} anchors, "
        f"Max: {dominant_counts.max()}, Min: {dominant_counts.min()}"
    )
    ax.set_xlim(-1, K)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Active anchors: {(dominant_counts > 0).sum()}/{K}")
    print(f"Dominance range: {dominant_counts.min()} — {dominant_counts.max()}")
    top5 = dominant_counts.topk(5)
    print(f"Top 5 dominant anchors: {list(zip(top5.indices.tolist(), top5.values.tolist()))}")


def main():
    print("Loading model...")
    model = load_model()

    print("Loading Flickr30k image tokens...")
    flickr_img = torch.load(
        TOKEN_DIR / "flickr30k_test_img.pt", weights_only=True,
    ).float()  # (31783, 257, 768)
    print(f"Flickr30k images: {flickr_img.shape}")

    # Step 2-3: Attention maps for 5 sample images
    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"\nProcessing sample {i} (Flickr index {idx})...")
        img_tokens = flickr_img[idx:idx + 1]  # (1, 257, 768)
        attn = compute_attention(model, img_tokens)
        out_path = OUT_DIR / f"sample_{i}_attn.png"
        plot_attention_maps(attn, idx, out_path)
        print(f"  Saved {out_path}")

        # Print CLS token attention stats
        cls_attn = attn[0, :]  # (K,)
        print(f"  CLS token total attention: {cls_attn.sum():.4f}")
        print(f"  CLS top anchor: {cls_attn.argmax()} (attn={cls_attn.max():.4f})")

    # Step 4: Anchor usage histogram
    print("\nComputing anchor usage histogram...")
    plot_anchor_usage_histogram(
        flickr_img, model,
        OUT_DIR.parent / "anchor_usage_histogram.png",
    )

    print("\nAll visualizations saved to", OUT_DIR)


if __name__ == "__main__":
    main()
