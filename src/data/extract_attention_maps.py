"""Extract CLS attention maps from DINOv2 and all-mpnet-base-v2.

For each encoder, extracts the CLS → token attention weights from the
last transformer layer, averaged over heads. These serve as a prior for
cross-attention pooling in Bridge Anchors.

Image (DINOv2): CLS → 256 patches → (N, 256) attention weights
Text (all-mpnet-base-v2): CLS → word tokens → (N, M) padded attention weights

Usage:
    python -m src.data.extract_attention_maps
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings" / "all_tokens"

DINOV2_IMAGE_SIZE = 224
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
TEXT_MODEL_NAME = "all-mpnet-base-v2"


# ── Image helpers ──────────────────────────────────────────────────────

def get_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(DINOV2_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


class ImageListDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def _get_coco_image_paths() -> list[Path]:
    """Get sorted COCO train2017 image paths."""
    img_dir = DATA_DIR / "coco" / "train2017"
    paths = sorted(img_dir.glob("*.jpg"))
    logger.info("COCO train: %d images", len(paths))
    return paths


def _get_flickr_image_paths() -> list[Path]:
    """Get sorted Flickr30k image paths (matching embedding order)."""
    base = DATA_DIR / "flickr30k"
    for candidate in ["flickr30k_images", "flickr30k-images", "images"]:
        p = base / candidate
        if p.is_dir():
            img_dir = p
            break
    else:
        raise FileNotFoundError("Flickr30k images not found")

    token_file = base / "results_20130124.token"
    first_caption: dict[str, str] = {}
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
    paths = [img_dir / fname for fname in sorted_filenames if (img_dir / fname).exists()]
    logger.info("Flickr30k: %d images", len(paths))
    return paths


@torch.no_grad()
def extract_dinov2_cls_attention(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract CLS → patch attention from DINOv2 last layer.

    Returns:
        cls_attn: (N, 256) float32 — CLS attention over 256 patches,
            averaged over heads, softmax-normalized.
    """
    logger.info("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()

    # Get the last block's attention module
    last_block = model.blocks[-1]
    attn_module = last_block.attn

    transform = get_image_transform()
    dataset = ImageListDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_cls_attn = []

    for batch in tqdm(loader, desc="DINOv2 CLS attention"):
        batch = batch.to(device, non_blocking=True)

        # Run through all blocks except the last one
        x = model.prepare_tokens_with_masks(batch)
        for blk in model.blocks[:-1]:
            x = blk(x)

        # At the last block, manually compute attention
        # x: (B, 257, 768)
        B, N, C = x.shape
        # Apply layer norm (pre-norm architecture)
        x_normed = last_block.norm1(x)

        # Compute QKV
        qkv = attn_module.qkv(x_normed).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
        q, k, v = torch.unbind(qkv, 2)  # each (B, N, H, D_h)
        q = q.transpose(1, 2)  # (B, H, N, D_h)
        k = k.transpose(1, 2)

        # Compute attention weights
        scale = (C // attn_module.num_heads) ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # CLS token (index 0) attention to patches (indices 1:)
        cls_to_patches = attn_weights[:, :, 0, 1:]  # (B, H, 256)
        # Average over heads
        cls_attn = cls_to_patches.mean(dim=1)  # (B, 256)
        # Renormalize (after removing CLS→CLS)
        cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        all_cls_attn.append(cls_attn.cpu())

    return torch.cat(all_cls_attn, dim=0)  # (N, 256)


# ── Text helpers ───────────────────────────────────────────────────────

def _get_coco_captions() -> list[str]:
    """Load first caption per COCO train image (matching embedding order)."""
    import json
    ann_file = DATA_DIR / "coco" / "annotations" / "captions_train2017.json"
    with open(ann_file) as f:
        data = json.load(f)

    # Group by image_id, pick first caption, sort by image filename
    img_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    first_cap: dict[str, str] = {}
    for ann in data["annotations"]:
        fname = img_id_to_filename[ann["image_id"]]
        if fname not in first_cap:
            first_cap[fname] = ann["caption"]

    sorted_filenames = sorted(first_cap.keys())
    captions = [first_cap[fname] for fname in sorted_filenames]
    logger.info("COCO train: %d captions", len(captions))
    return captions


def _get_flickr_captions() -> list[str]:
    """Load first caption per Flickr30k image (matching embedding order)."""
    base = DATA_DIR / "flickr30k"
    token_file = base / "results_20130124.token"
    first_caption: dict[str, str] = {}
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

    img_dir = None
    for candidate in ["flickr30k_images", "flickr30k-images", "images"]:
        p = base / candidate
        if p.is_dir():
            img_dir = p
            break

    sorted_filenames = sorted(first_caption.keys())
    captions = [first_caption[fname] for fname in sorted_filenames
                if img_dir is None or (img_dir / fname).exists()]
    logger.info("Flickr30k: %d captions", len(captions))
    return captions


@torch.no_grad()
def extract_text_cls_attention(
    captions: list[str],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 256,
) -> tuple[torch.Tensor, int]:
    """Extract CLS → word token attention from all-mpnet-base-v2 last layer.

    Returns:
        cls_attn: (N, max_seq_len) float32 — CLS attention over word tokens,
            averaged over heads, zero-padded for padding tokens.
        actual_max_len: The actual max sequence length encountered.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading text model: %s", TEXT_MODEL_NAME)
    st_model = SentenceTransformer(TEXT_MODEL_NAME)
    transformer = st_model[0].auto_model.to(device).eval()
    tokenizer = st_model[0].tokenizer

    all_cls_attn = []
    actual_max_len = 0

    for i in tqdm(range(0, len(captions), batch_size), desc="Text CLS attention"):
        batch_texts = captions[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward with attention output
        outputs = transformer(**encoded, output_attentions=True)
        # outputs.attentions: tuple of (B, H, S, S) per layer
        last_layer_attn = outputs.attentions[-1]  # (B, H, S, S)

        seq_len = last_layer_attn.shape[-1]
        actual_max_len = max(actual_max_len, seq_len)

        # CLS (index 0) attention to all other tokens (indices 1:)
        cls_to_words = last_layer_attn[:, :, 0, 1:]  # (B, H, S-1)
        # Average over heads
        cls_attn = cls_to_words.mean(dim=1)  # (B, S-1)

        # Zero out padding positions
        # attention_mask: (B, S) — 1 for valid, 0 for padding
        word_mask = encoded["attention_mask"][:, 1:]  # (B, S-1)
        cls_attn = cls_attn * word_mask.float()

        # Renormalize
        cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        all_cls_attn.append(cls_attn.cpu())

    # Pad all batches to the same max length (S-1, since we stripped CLS)
    padded = []
    for t in all_cls_attn:
        if t.shape[1] < actual_max_len - 1:
            pad = torch.zeros(t.shape[0], actual_max_len - 1 - t.shape[1])
            t = torch.cat([t, pad], dim=1)
        padded.append(t)

    return torch.cat(padded, dim=0), actual_max_len


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Image attention (DINOv2) ──
    for name, get_paths in [("coco_train", _get_coco_image_paths),
                             ("flickr30k_test", _get_flickr_image_paths)]:
        out_path = EMB_DIR / f"{name}_img_cls_attn.pt"
        if out_path.exists():
            logger.info("Already exists: %s — skipping", out_path)
            continue
        paths = get_paths()
        cls_attn = extract_dinov2_cls_attention(paths, device)
        logger.info("Saving %s: %s", out_path.name, tuple(cls_attn.shape))
        torch.save(cls_attn, out_path)

    # ── Text attention (all-mpnet-base-v2) ──
    for name, get_captions in [("coco_train", _get_coco_captions),
                                ("flickr30k_test", _get_flickr_captions)]:
        out_path = EMB_DIR / f"{name}_txt_cls_attn.pt"
        if out_path.exists():
            logger.info("Already exists: %s — skipping", out_path)
            continue
        captions = get_captions()
        cls_attn, max_len = extract_text_cls_attention(captions, device)
        logger.info("Saving %s: %s (max_len=%d)", out_path.name,
                     tuple(cls_attn.shape), max_len)
        torch.save(cls_attn, out_path)

    logger.info("Done! All attention maps saved to %s", EMB_DIR)


if __name__ == "__main__":
    main()
