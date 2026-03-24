"""Extract token-level DINOv2 embeddings (CLS + 256 patch tokens).

For token-level BridgeAnchors pilot:
- COCO train 10K subset: (10K, 257, 768) image tokens + (10K, 768) text CLS
- Flickr30k test: (31783, 257, 768) image tokens + (31783, 768) text CLS

Text stays CLS-level (single-sentence pooled output from MPNet).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
TOKEN_DIR = PROJECT_ROOT / "data" / "embeddings" / "token"

SEED = 42
N_SUBSET = 10000


def load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14."""
    logger.info("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_token_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract all tokens (CLS + patches) from DINOv2's final block.

    Returns:
        (N, 257, 768) tensor — 1 CLS + 256 patch tokens, L2-normalized.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False)

    n_blocks = len(model.blocks)
    all_tokens = []

    for imgs, indices in tqdm(loader, desc="  DINOv2 tokens"):
        imgs = imgs.to(device, non_blocking=True)

        # get_intermediate_layers returns (patch_tokens, cls_token) tuples
        # n=1 with norm=True gives the final output
        out = model.get_intermediate_layers(
            imgs, n=1, return_class_token=True, norm=True,
        )
        patch_tokens = out[0][0]  # (B, 256, 768)
        cls_token = out[0][1]     # (B, 768)

        # Concatenate CLS + patches → (B, 257, 768)
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        tokens = F.normalize(tokens, dim=-1)
        all_tokens.append(tokens.cpu())

    return torch.cat(all_tokens, dim=0)


def main() -> None:
    from src.data.extract_embeddings import (
        get_image_transform, ImageListDataset, load_text_encoder,
        _load_coco_annotations, _load_flickr30k_annotations,
    )

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dinov2 = load_dinov2(device)
    transform = get_image_transform()

    # ── COCO train 10K subset ──
    logger.info("=" * 60)
    logger.info("COCO train 10K subset")
    logger.info("=" * 60)

    image_paths, captions = _load_coco_annotations("train")

    # Deterministic 10K subset
    gen = torch.Generator().manual_seed(SEED)
    all_indices = torch.randperm(len(image_paths), generator=gen)
    subset_indices = all_indices[:N_SUBSET]

    sub_paths = [image_paths[i] for i in subset_indices.tolist()]
    sub_captions = [captions[i] for i in subset_indices.tolist()]
    logger.info("Selected %d COCO pairs (seed=%d)", len(sub_paths), SEED)

    # Save indices for reproducibility
    idx_path = TOKEN_DIR / "coco_train_10k_indices.pt"
    torch.save(subset_indices, idx_path)
    logger.info("Saved indices → %s", idx_path)

    # Image tokens
    img_path = TOKEN_DIR / "coco_train_10k_img.pt"
    if img_path.exists():
        logger.info("Already exists: %s", img_path)
    else:
        logger.info("Extracting DINOv2 tokens for 10K COCO images...")
        ds = ImageListDataset(sub_paths, transform)
        img_tokens = extract_token_embeddings(dinov2, ds, device, batch_size=32)
        torch.save(img_tokens, img_path)
        logger.info("Saved %s → %s (%.1f MB)",
                    tuple(img_tokens.shape), img_path,
                    img_path.stat().st_size / 1e6)

    # Text CLS (reuse existing extraction)
    txt_path = TOKEN_DIR / "coco_train_10k_txt.pt"
    if txt_path.exists():
        logger.info("Already exists: %s", txt_path)
    else:
        logger.info("Extracting MPNet text embeddings for 10K COCO captions...")
        text_model = load_text_encoder()
        txt_embs = text_model.encode(
            sub_captions, batch_size=256, show_progress_bar=True,
            convert_to_tensor=True, normalize_embeddings=True,
        ).cpu()
        torch.save(txt_embs, txt_path)
        logger.info("Saved %s → %s", tuple(txt_embs.shape), txt_path)

    # ── Flickr30k test ──
    logger.info("=" * 60)
    logger.info("Flickr30k test (full)")
    logger.info("=" * 60)

    flickr_img_path = TOKEN_DIR / "flickr30k_img.pt"
    flickr_txt_path = TOKEN_DIR / "flickr30k_txt.pt"

    image_paths_f, captions_f = _load_flickr30k_annotations()

    if flickr_img_path.exists():
        logger.info("Already exists: %s", flickr_img_path)
    else:
        logger.info("Extracting DINOv2 tokens for %d Flickr30k images...", len(image_paths_f))
        ds_f = ImageListDataset(image_paths_f, transform)
        flickr_img_tokens = extract_token_embeddings(dinov2, ds_f, device, batch_size=32)
        torch.save(flickr_img_tokens, flickr_img_path)
        logger.info("Saved %s → %s (%.1f MB)",
                    tuple(flickr_img_tokens.shape), flickr_img_path,
                    flickr_img_path.stat().st_size / 1e6)

    if flickr_txt_path.exists():
        logger.info("Already exists: %s", flickr_txt_path)
    else:
        logger.info("Extracting MPNet text embeddings for %d Flickr30k captions...", len(captions_f))
        if 'text_model' not in dir():
            text_model = load_text_encoder()
        txt_embs_f = text_model.encode(
            captions_f, batch_size=256, show_progress_bar=True,
            convert_to_tensor=True, normalize_embeddings=True,
        ).cpu()
        torch.save(txt_embs_f, flickr_txt_path)
        logger.info("Saved %s → %s", tuple(txt_embs_f.shape), flickr_txt_path)

    logger.info("=" * 60)
    logger.info("All token-level embeddings extracted.")
    logger.info("Files in %s:", TOKEN_DIR)
    for f in sorted(TOKEN_DIR.iterdir()):
        logger.info("  %s (%.1f MB)", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
