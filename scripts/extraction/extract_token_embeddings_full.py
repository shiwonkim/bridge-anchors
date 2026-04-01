"""Extract full-scale token-level DINOv2 embeddings for COCO 118K + Flickr30k.

Saves to data/embeddings/all_tokens/ on local SSD.
COCO chunks: (10000, 257, 768) float16 — ~3.7GB per chunk, ~44GB total.
Last chunk is smaller (8287 samples).
Flickr30k: (31783, 257, 768) float16 — single file, ~12GB.

Text: copies existing CLS text embeddings from data/embeddings/cls/.

Usage:
    python scripts/extraction/extract_token_embeddings_full.py
"""

from __future__ import annotations

import json
import logging
import time
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "all_tokens"
CLS_DIR = PROJECT_ROOT / "data" / "embeddings" / "cls"

CHUNK_SIZE = 10000
TOTAL_SAMPLES = 118287
NUM_CHUNKS = (TOTAL_SAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE  # 12

PILOT_SEED = 42
PILOT_N = 10000


def load_dinov2(device: torch.device) -> torch.nn.Module:
    logger.info("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_chunk_tokens(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract all tokens (CLS + patches) from DINOv2. Returns float16."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, drop_last=False)

    all_tokens = []
    for imgs, _ in tqdm(loader, desc="    Batch", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        out = model.get_intermediate_layers(
            imgs, n=1, return_class_token=True, norm=True,
        )
        patch_tokens = out[0][0]  # (B, 256, 768)
        cls_token = out[0][1]     # (B, 768)
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        tokens = F.normalize(tokens, dim=-1)
        all_tokens.append(tokens.half().cpu())  # float16

    return torch.cat(all_tokens, dim=0)


def main() -> None:
    from src.data.extract_embeddings import (
        get_image_transform, ImageListDataset, _load_flickr30k_annotations,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Output dir: %s (float16)", OUT_DIR)
    logger.info("Expected COCO total: ~%.1f GB", TOTAL_SAMPLES * 257 * 768 * 2 / 1e9)

    # ── COCO 118K chunked extraction ──
    logger.info("=" * 60)
    logger.info("COCO train 118K — chunked token extraction")
    logger.info("=" * 60)

    ann_file = PROJECT_ROOT / "data" / "datasets" / "coco" / "annotations" / "captions_train2017.json"
    img_dir = PROJECT_ROOT / "data" / "datasets" / "coco" / "train2017"

    logger.info("Loading COCO annotations...")
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
    logger.info("COCO train: %d images", len(image_paths))
    assert len(image_paths) == TOTAL_SAMPLES

    # Check existing chunks
    existing = set()
    for i in range(NUM_CHUNKS):
        p = OUT_DIR / f"coco_train_chunk_{i:02d}_img.pt"
        if p.exists():
            existing.add(i)
    if existing:
        logger.info("Skipping %d existing chunks: %s", len(existing), sorted(existing))

    # Load DINOv2
    dinov2 = load_dinov2(device)
    transform = get_image_transform()

    t_start = time.time()
    total_bytes = 0

    for chunk_idx in range(NUM_CHUNKS):
        chunk_path = OUT_DIR / f"coco_train_chunk_{chunk_idx:02d}_img.pt"

        if chunk_idx in existing:
            total_bytes += chunk_path.stat().st_size
            continue

        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, TOTAL_SAMPLES)
        chunk_paths = image_paths[start:end]

        logger.info("Chunk %02d/%02d — [%d:%d] (%d samples)...",
                    chunk_idx, NUM_CHUNKS - 1, start, end, len(chunk_paths))

        ds = ImageListDataset(chunk_paths, transform)
        chunk_tokens = extract_chunk_tokens(dinov2, ds, device, batch_size=64)

        torch.save(chunk_tokens, chunk_path)
        sz = chunk_path.stat().st_size
        total_bytes += sz
        elapsed = time.time() - t_start

        logger.info("  Chunk %d/%d done, %.2f GB saved, elapsed: %.1f min",
                    chunk_idx + 1, NUM_CHUNKS, total_bytes / 1e9, elapsed / 60)

        del chunk_tokens, ds
        torch.cuda.empty_cache()

    coco_time = time.time() - t_start

    # Save COCO chunk metadata
    metadata = {
        "num_chunks": NUM_CHUNKS,
        "chunk_size": CHUNK_SIZE,
        "total_samples": TOTAL_SAMPLES,
        "tokens_per_image": 257,
        "embedding_dim": 768,
        "dtype": "float16",
        "chunks": [],
    }
    for i in range(NUM_CHUNKS):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, TOTAL_SAMPLES)
        metadata["chunks"].append({
            "index": i,
            "filename": f"coco_train_chunk_{i:02d}_img.pt",
            "start_idx": start,
            "end_idx": end,
            "num_samples": end - start,
        })

    with open(OUT_DIR / "chunk_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved chunk_metadata.json")

    # ── Flickr30k test token extraction ──
    logger.info("=" * 60)
    logger.info("Flickr30k test — full token extraction")
    logger.info("=" * 60)

    flickr_img_path = OUT_DIR / "flickr30k_test_img.pt"
    if flickr_img_path.exists():
        logger.info("Already exists: %s", flickr_img_path)
    else:
        image_paths_f, _ = _load_flickr30k_annotations()
        logger.info("Extracting DINOv2 tokens for %d Flickr30k images...", len(image_paths_f))
        t_flickr = time.time()
        ds_f = ImageListDataset(image_paths_f, transform)
        flickr_tokens = extract_chunk_tokens(dinov2, ds_f, device, batch_size=64)
        torch.save(flickr_tokens, flickr_img_path)
        flickr_sz = flickr_img_path.stat().st_size
        flickr_time = time.time() - t_flickr
        logger.info("  Flickr30k done: %s, %.2f GB, %.1f min",
                    tuple(flickr_tokens.shape), flickr_sz / 1e9, flickr_time / 60)
        total_bytes += flickr_sz
        del flickr_tokens

    # ── Copy text embeddings from cls/ ──
    logger.info("=" * 60)
    logger.info("Copying text embeddings from cls/")
    logger.info("=" * 60)

    import shutil
    for name in ["coco_train_txt.pt", "flickr30k_test_txt.pt"]:
        dst = OUT_DIR / name
        src = CLS_DIR / name
        if dst.exists():
            logger.info("Already exists: %s", dst)
        elif src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied %s → %s", src.name, dst)
        else:
            logger.warning("Source not found: %s", src)

    # ── Save pilot 10K indices for reference ──
    gen = torch.Generator().manual_seed(PILOT_SEED)
    pilot_perm = torch.randperm(TOTAL_SAMPLES, generator=gen)
    pilot_indices = pilot_perm[:PILOT_N]
    torch.save(pilot_indices, OUT_DIR / "pilot_10k_indices.pt")
    logger.info("Saved pilot_10k_indices.pt (%d indices, seed=%d)", PILOT_N, PILOT_SEED)

    # ── Final verification ──
    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("ALL DONE.")
    logger.info("  COCO chunks: %d, %.1f GB, %.1f min",
                NUM_CHUNKS, sum(
                    (OUT_DIR / f"coco_train_chunk_{i:02d}_img.pt").stat().st_size
                    for i in range(NUM_CHUNKS)
                ) / 1e9, coco_time / 60)
    logger.info("  Total time: %.1f min", total_time / 60)

    x = torch.load(OUT_DIR / "coco_train_chunk_00_img.pt", weights_only=True)
    logger.info("  Verify chunk_00: shape=%s, dtype=%s", tuple(x.shape), x.dtype)
    assert x.shape == (CHUNK_SIZE, 257, 768)
    assert x.dtype == torch.float16

    logger.info("Files in %s:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        logger.info("  %s (%.2f GB)", f.name, f.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
