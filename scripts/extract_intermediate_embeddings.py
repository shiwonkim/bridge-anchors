"""Extract embeddings from intermediate encoder layers for the top-3 CKA pairs.

Extracts from:
  - Top 1: DINOv2 Block 9 × MPNet Layer 10
  - Top 2: DINOv2 Block 9 × MPNet Layer 11
  - Top 3: DINOv2 Block 10 × MPNet Layer 10

For all datasets: COCO train, Flickr30k test, ImageNet val.
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
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"

# Top-3 pairs: (dino_block, mpnet_layer)
LAYER_PAIRS = [
    (9, 10),   # Top 1: CKA=0.586
    (9, 11),   # Top 2: CKA=0.552
    (10, 10),  # Top 3: CKA=0.514 (different img block for diversity)
]


# ===================================================================
# DINOv2 single-layer extraction
# ===================================================================

@torch.no_grad()
def extract_dinov2_layer(
    model: torch.nn.Module,
    dataset,
    block_idx: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract CLS token from a specific DINOv2 block."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False)

    all_embs = []
    n_blocks = len(model.blocks)

    for imgs, indices in tqdm(loader, desc=f"  DINOv2 block {block_idx}", leave=False):
        imgs = imgs.to(device, non_blocking=True)

        if block_idx < n_blocks:
            # Get specific block output (no norm)
            intermediates = model.get_intermediate_layers(
                imgs, n=n_blocks, return_class_token=True, norm=False,
            )
            cls_tok = intermediates[block_idx][1]  # (B, 768)
        else:
            # Final (after norm) — block_idx == n_blocks means "Final"
            intermediates = model.get_intermediate_layers(
                imgs, n=1, return_class_token=True, norm=True,
            )
            cls_tok = intermediates[0][1]

        embs = F.normalize(cls_tok, dim=-1)
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ===================================================================
# MPNet single-layer extraction
# ===================================================================

@torch.no_grad()
def extract_mpnet_layer(
    texts: list[str],
    layer_idx: int,
    batch_size: int = 128,
) -> torch.Tensor:
    """Extract mean-pooled embeddings from a specific MPNet layer."""
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer = st_model.tokenizer
    mpnet = st_model[0].auto_model
    device = next(mpnet.parameters()).device
    n_layers = mpnet.config.num_hidden_layers  # 12

    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"  MPNet layer {layer_idx}", leave=False):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = mpnet(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        attention_mask = encoded["attention_mask"]

        if layer_idx < n_layers:
            hs = hidden_states[layer_idx + 1]  # +1 to skip embedding layer
        else:
            hs = hidden_states[-1]  # Final

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        embs = F.normalize(pooled, dim=-1)
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


# ===================================================================
# Dataset loading helpers (reused from extract_embeddings.py)
# ===================================================================

def load_coco_data(split: str) -> tuple:
    """Load COCO image paths and captions."""
    split_name = {"train": "train2017", "val": "val2017"}[split]
    ann_file = DATA_DIR / "coco" / "annotations" / f"captions_{split_name}.json"
    img_dir = DATA_DIR / "coco" / split_name

    with open(ann_file) as f:
        data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    first_caption = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in first_caption:
            first_caption[iid] = ann["caption"]

    sorted_ids = sorted(first_caption.keys())
    image_paths = [img_dir / id_to_file[iid] for iid in sorted_ids]
    captions = [first_caption[iid] for iid in sorted_ids]
    return image_paths, captions


def load_flickr30k_data() -> tuple:
    """Load Flickr30k image paths and captions."""
    base = DATA_DIR / "flickr30k"
    img_dir = None
    for candidate in ["flickr30k_images", "flickr30k-images", "images"]:
        p = base / candidate
        if p.is_dir():
            img_dir = p
            break

    token_file = None
    for candidate in ["results_20130124.token", "results.token", "captions.token"]:
        p = base / candidate
        if p.exists():
            token_file = p
            break

    first_caption = {}
    with open(token_file, encoding="utf-8") as f:
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
                first_caption[filename] = caption

    sorted_filenames = sorted(first_caption.keys())
    image_paths = [img_dir / fname for fname in sorted_filenames if (img_dir / fname).exists()]
    captions = [first_caption[fname] for fname in sorted_filenames if (img_dir / fname).exists()]
    return image_paths, captions


def load_imagenet_data() -> tuple:
    """Load ImageNet val image paths and class text prompts."""
    val_dir = DATA_DIR / "imagenet" / "val"
    exts = {".jpg", ".jpeg", ".png", ".JPEG"}

    subdirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    image_paths = []
    for subdir in subdirs:
        imgs = sorted(p for p in subdir.iterdir() if p.suffix in exts)
        image_paths.extend(imgs)

    # Class text prompts
    from src.data.extract_embeddings import _get_imagenet_class_names
    class_names = _get_imagenet_class_names()
    prompts = [f"a photo of a {name}" for name in class_names]
    return image_paths, prompts


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    from src.data.extract_embeddings import (
        get_image_transform, ImageListDataset, load_dinov2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load DINOv2 once
    dinov2 = load_dinov2(device)
    transform = get_image_transform()

    # Unique layer indices to extract
    dino_blocks = sorted(set(p[0] for p in LAYER_PAIRS))
    mpnet_layers = sorted(set(p[1] for p in LAYER_PAIRS))

    logger.info("DINOv2 blocks to extract: %s", dino_blocks)
    logger.info("MPNet layers to extract: %s", mpnet_layers)

    # ── COCO train ──
    logger.info("=" * 60)
    logger.info("COCO train")
    logger.info("=" * 60)
    image_paths, captions = load_coco_data("train")
    ds = ImageListDataset(image_paths, transform)

    for blk in dino_blocks:
        out_path = EMB_DIR / f"coco_train_img_dino_block{blk}.pt"
        if out_path.exists():
            logger.info("  Already exists: %s", out_path)
            continue
        logger.info("  Extracting DINOv2 block %d for COCO train (%d images)...", blk, len(ds))
        embs = extract_dinov2_layer(dinov2, ds, blk, 64, device)
        torch.save(embs, out_path)
        logger.info("  Saved %s → %s", tuple(embs.shape), out_path)

    for lay in mpnet_layers:
        out_path = EMB_DIR / f"coco_train_txt_mpnet_layer{lay}.pt"
        if out_path.exists():
            logger.info("  Already exists: %s", out_path)
            continue
        logger.info("  Extracting MPNet layer %d for COCO train (%d texts)...", lay, len(captions))
        embs = extract_mpnet_layer(captions, lay)
        torch.save(embs, out_path)
        logger.info("  Saved %s → %s", tuple(embs.shape), out_path)

    # ── Flickr30k test ──
    logger.info("=" * 60)
    logger.info("Flickr30k test")
    logger.info("=" * 60)
    image_paths, captions = load_flickr30k_data()
    ds = ImageListDataset(image_paths, transform)

    for blk in dino_blocks:
        out_path = EMB_DIR / f"flickr30k_test_img_dino_block{blk}.pt"
        if out_path.exists():
            logger.info("  Already exists: %s", out_path)
            continue
        logger.info("  Extracting DINOv2 block %d for Flickr30k (%d images)...", blk, len(ds))
        embs = extract_dinov2_layer(dinov2, ds, blk, 64, device)
        torch.save(embs, out_path)
        logger.info("  Saved %s → %s", tuple(embs.shape), out_path)

    for lay in mpnet_layers:
        out_path = EMB_DIR / f"flickr30k_test_txt_mpnet_layer{lay}.pt"
        if out_path.exists():
            logger.info("  Already exists: %s", out_path)
            continue
        logger.info("  Extracting MPNet layer %d for Flickr30k (%d texts)...", lay, len(captions))
        embs = extract_mpnet_layer(captions, lay)
        torch.save(embs, out_path)
        logger.info("  Saved %s → %s", tuple(embs.shape), out_path)

    logger.info("=" * 60)
    logger.info("All intermediate embeddings extracted.")


if __name__ == "__main__":
    main()
