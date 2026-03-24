"""Cross-modal CKA analysis across encoder layers.

Computes linear CKA between all (DINOv2_layer_i, MPNet_layer_j) pairs
to find which intermediate layers have the highest cross-modal similarity.

Usage:
    python -m src.eval.layer_cka --n-samples 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
OUT_DIR = PROJECT_ROOT / "experiments" / "exp_intermediate_layer"


# ===================================================================
# Encoder loading with intermediate layer extraction
# ===================================================================


def load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14."""
    logger.info("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_dinov2_all_layers(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> list[torch.Tensor]:
    """Extract CLS token from every DINOv2 block (12 blocks + final norm).

    Returns:
        List of 13 tensors, each (N, 768). Index 0 = after block 0, ...,
        index 11 = after block 11, index 12 = final output (after norm).
    """
    n_blocks = len(model.blocks)  # 12
    all_layer_embs = [[] for _ in range(n_blocks + 1)]

    for start in tqdm(range(0, len(images), batch_size), desc="DINOv2 layers"):
        batch = images[start:start + batch_size].to(device)

        # get_intermediate_layers with n=int returns last n blocks
        # We want all 12 blocks, so n=n_blocks
        intermediates = model.get_intermediate_layers(
            batch, n=n_blocks,
            return_class_token=True, norm=False,
        )
        # intermediates is a tuple of (patch_tokens, cls_token) tuples
        for i, (_, cls_tok) in enumerate(intermediates):
            all_layer_embs[i].append(cls_tok.cpu())

        # Final output (after norm) = just the last block with norm=True
        final = model.get_intermediate_layers(
            batch, n=1, return_class_token=True, norm=True,
        )
        all_layer_embs[n_blocks].append(final[0][1].cpu())

    return [torch.cat(embs, dim=0) for embs in all_layer_embs]


@torch.no_grad()
def extract_mpnet_all_layers(
    texts: list[str],
    batch_size: int = 128,
) -> list[torch.Tensor]:
    """Extract pooled output from every MPNet encoder layer + final.

    Returns:
        List of 13 tensors, each (N, 768). Index 0 = after layer 0, ...,
        index 11 = after layer 11, index 12 = final sentence-transformers output.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers all-mpnet-base-v2...")
    st_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer = st_model.tokenizer
    mpnet = st_model[0].auto_model
    pooling = st_model[1]  # Pooling layer

    device = next(mpnet.parameters()).device
    n_layers = mpnet.config.num_hidden_layers  # 12

    all_layer_embs = [[] for _ in range(n_layers + 1)]

    for start in tqdm(range(0, len(texts), batch_size), desc="MPNet layers"):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = mpnet(**encoded, output_hidden_states=True)
        # outputs.hidden_states: tuple of (n_layers+1) tensors (B, seq_len, 768)
        # Index 0 = embedding layer output, 1..12 = transformer layer outputs
        hidden_states = outputs.hidden_states

        attention_mask = encoded["attention_mask"]

        # Mean pooling for each hidden state (layers 1..12)
        for i in range(n_layers):
            hs = hidden_states[i + 1]  # skip embedding layer (index 0)
            # Mean pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            all_layer_embs[i].append(pooled.cpu())

        # Final sentence-transformer output (layer 12 + pooling + normalize)
        final_hs = hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        final_pooled = (final_hs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        final_normed = F.normalize(final_pooled, dim=-1)
        all_layer_embs[n_layers].append(final_normed.cpu())

    return [torch.cat(embs, dim=0) for embs in all_layer_embs]


# ===================================================================
# CKA computation
# ===================================================================


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (N, D1) representation matrix.
        Y: (N, D2) representation matrix.

    Returns:
        CKA score in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    K = X @ X.T  # (N, N)
    L = Y @ Y.T  # (N, N)

    # Center kernels
    N = K.shape[0]
    H = torch.eye(N) - 1.0 / N
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_kl = (Kc * Lc).sum()
    hsic_kk = (Kc * Kc).sum()
    hsic_ll = (Lc * Lc).sum()

    denom = (hsic_kk * hsic_ll).sqrt().clamp(min=1e-12)
    return (hsic_kl / denom).item()


def compute_cka_matrix(
    img_layers: list[torch.Tensor],
    txt_layers: list[torch.Tensor],
) -> np.ndarray:
    """Compute CKA between all (img_layer_i, txt_layer_j) pairs.

    Returns:
        (n_img_layers, n_txt_layers) CKA matrix.
    """
    n_img = len(img_layers)
    n_txt = len(txt_layers)
    cka_mat = np.zeros((n_img, n_txt))

    for i in range(n_img):
        for j in range(n_txt):
            cka_mat[i, j] = linear_cka(img_layers[i], txt_layers[j])
            logger.debug("CKA[img=%d, txt=%d] = %.4f", i, j, cka_mat[i, j])

    return cka_mat


# ===================================================================
# Visualization
# ===================================================================


def plot_cka_heatmap(
    cka_mat: np.ndarray,
    save_path: Path,
    top_pairs: list[tuple[int, int, float]],
) -> None:
    """Plot and save CKA heatmap."""
    n_img, n_txt = cka_mat.shape

    img_labels = [f"Block {i}" for i in range(n_img - 1)] + ["Final"]
    txt_labels = [f"Layer {i}" for i in range(n_txt - 1)] + ["Final"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=cka_mat.max())

    ax.set_xticks(range(n_txt))
    ax.set_xticklabels(txt_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_img))
    ax.set_yticklabels(img_labels, fontsize=8)
    ax.set_xlabel("MPNet (text) Layer", fontsize=11)
    ax.set_ylabel("DINOv2 (image) Block", fontsize=11)
    ax.set_title("Cross-Modal Linear CKA: DINOv2 × MPNet Layer Pairs", fontsize=12)

    # Annotate all cells
    for i in range(n_img):
        for j in range(n_txt):
            color = "white" if cka_mat[i, j] > cka_mat.max() * 0.7 else "black"
            ax.text(j, i, f"{cka_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=6, color=color)

    # Highlight top pair
    best_i, best_j, best_cka = top_pairs[0]
    rect = plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                         linewidth=3, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="Linear CKA", shrink=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved CKA heatmap to %s", save_path)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-modal CKA layer analysis.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-img", type=int, default=64)
    parser.add_argument("--batch-size-txt", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, N=%d, seed=%d", device, args.n_samples, args.seed)

    # --- Load COCO subset ---
    logger.info("Loading COCO training data (first %d samples)...", args.n_samples)
    ann_file = DATA_DIR / "coco" / "annotations" / "captions_train2017.json"
    img_dir = DATA_DIR / "coco" / "train2017"
    with open(ann_file) as f:
        cap_data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in cap_data["images"]}
    first_caption = {}
    for ann in cap_data["annotations"]:
        iid = ann["image_id"]
        if iid not in first_caption:
            first_caption[iid] = ann["caption"]

    sorted_ids = sorted(first_caption.keys())
    # Subsample deterministically
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(sorted_ids), generator=gen)[:args.n_samples]

    image_paths = []
    captions = []
    for idx in perm.tolist():
        iid = sorted_ids[idx]
        image_paths.append(img_dir / id_to_file[iid])
        captions.append(first_caption[iid])

    logger.info("Selected %d COCO image-caption pairs.", len(image_paths))

    # --- Preprocess images ---
    from src.data.extract_embeddings import get_image_transform, ImageListDataset
    transform = get_image_transform()
    ds = ImageListDataset(image_paths, transform)
    loader = DataLoader(ds, batch_size=args.batch_size_img, shuffle=False,
                       num_workers=4, pin_memory=True)

    logger.info("Preprocessing images...")
    all_imgs = []
    for imgs, _ in tqdm(loader, desc="Load images"):
        all_imgs.append(imgs)
    all_imgs = torch.cat(all_imgs, dim=0)
    logger.info("Image tensor: %s", tuple(all_imgs.shape))

    # --- Extract DINOv2 intermediate layers ---
    dinov2 = load_dinov2(device)
    logger.info("Extracting DINOv2 intermediate layers...")
    img_layers = extract_dinov2_all_layers(dinov2, all_imgs, device, args.batch_size_img)
    logger.info("DINOv2: %d layers, each %s", len(img_layers), tuple(img_layers[0].shape))
    del dinov2
    torch.cuda.empty_cache()

    # --- Extract MPNet intermediate layers ---
    logger.info("Extracting MPNet intermediate layers...")
    txt_layers = extract_mpnet_all_layers(captions, args.batch_size_txt)
    logger.info("MPNet: %d layers, each %s", len(txt_layers), tuple(txt_layers[0].shape))

    # --- Compute CKA matrix ---
    logger.info("Computing CKA matrix (%d × %d)...", len(img_layers), len(txt_layers))
    cka_mat = compute_cka_matrix(img_layers, txt_layers)

    # Save raw matrix
    np.save(OUT_DIR / "cka_matrix.npy", cka_mat)

    # Find top-5 pairs
    flat_indices = np.argsort(cka_mat.ravel())[::-1]
    top_pairs = []
    for flat_idx in flat_indices[:5]:
        i, j = divmod(flat_idx, cka_mat.shape[1])
        top_pairs.append((int(i), int(j), float(cka_mat[i, j])))

    logger.info("Top-5 layer pairs by CKA:")
    img_names = [f"Block {i}" for i in range(12)] + ["Final"]
    txt_names = [f"Layer {j}" for j in range(12)] + ["Final"]
    for rank, (i, j, cka) in enumerate(top_pairs, 1):
        logger.info("  %d. DINOv2 %s × MPNet %s: CKA=%.4f",
                    rank, img_names[i], txt_names[j], cka)

    # Save top pairs
    top_pairs_data = [
        {"rank": r + 1, "img_layer": i, "txt_layer": j, "cka": c,
         "img_name": img_names[i], "txt_name": txt_names[j]}
        for r, (i, j, c) in enumerate(top_pairs)
    ]
    with open(OUT_DIR / "top_layer_pairs.json", "w") as f:
        json.dump(top_pairs_data, f, indent=2)

    # Plot
    plot_cka_heatmap(cka_mat, OUT_DIR / "cka_heatmap.png", top_pairs)

    logger.info("Phase 1 complete. Results in %s", OUT_DIR)


if __name__ == "__main__":
    main()
