"""Extract and save embeddings from frozen encoders.

Loads DINOv2 ViT-B/14 (image) and all-mpnet-base-v2 (text), extracts
embeddings for COCO train / Flickr30k test / ImageNet val, and saves
them as .pt files under data/embeddings/.

Usage:
    python src/data/extract_embeddings.py --dataset coco --split train
    python src/data/extract_embeddings.py --dataset flickr30k --split test
    python src/data/extract_embeddings.py --dataset imagenet --split val
    python src/data/extract_embeddings.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"

DINOV2_DIM = 768
TEXT_DIM = 768
DINOV2_MODEL = "dinov2_vitb14"
TEXT_MODEL_NAME = "all-mpnet-base-v2"

# DINOv2 ViT-B/14 expects 518×518 (14×37=518) but works with 224 too.
# Using 224 for efficiency — standard ImageNet size, works well in practice.
DINOV2_IMAGE_SIZE = 224
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

IMAGENET_PROMPT_TEMPLATE = "a photo of a {}"

# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------


def get_image_transform() -> transforms.Compose:
    """Standard transform for DINOv2: resize, center crop, normalize."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(DINOV2_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class ImageListDataset(Dataset):
    """Dataset that loads images from a list of file paths."""

    def __init__(self, image_paths: list[Path], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx
        except Exception as e:
            logger.warning("Failed to load image %s: %s", path, e)
            # Return a black image so the batch doesn't break
            return torch.zeros(3, DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE), idx


class ImageFolderFlat(Dataset):
    """Load images from a flat directory (no class subdirs)."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        self.image_paths = sorted(
            p for p in root.iterdir() if p.suffix.lower() in exts
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), idx
        except Exception as e:
            logger.warning("Failed to load image %s: %s", path, e)
            return torch.zeros(3, DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE), idx


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------


def load_dinov2(device: torch.device) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14 from torch.hub."""
    logger.info("Loading DINOv2 ViT-B/14 from torch.hub...")
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL)
    model = model.to(device)
    model.eval()
    logger.info("DINOv2 loaded (dim=%d).", DINOV2_DIM)
    return model


def load_text_encoder() -> "SentenceTransformer":
    """Load sentence-transformers text encoder."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading text encoder: %s ...", TEXT_MODEL_NAME)
    model = SentenceTransformer(TEXT_MODEL_NAME)
    logger.info("Text encoder loaded (dim=%d).", TEXT_DIM)
    return model


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_image_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> torch.Tensor:
    """Extract DINOv2 CLS-token embeddings for all images in dataset.

    Args:
        model: DINOv2 model in eval mode.
        dataset: Dataset returning (image_tensor, index) tuples.
        batch_size: Batch size for inference.
        device: Torch device.
        num_workers: DataLoader workers.

    Returns:
        Tensor of shape (N, 768) with L2-normalized embeddings.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_embs = torch.zeros(len(dataset), DINOV2_DIM)
    for imgs, indices in tqdm(loader, desc="  images", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        embs = model(imgs)  # (B, 768) — CLS token
        embs = F.normalize(embs, dim=-1)
        all_embs[indices] = embs.cpu()

    return all_embs


def extract_text_embeddings(
    model: "SentenceTransformer",
    texts: list[str],
    batch_size: int,
) -> torch.Tensor:
    """Encode a list of strings with the sentence-transformer model.

    Args:
        model: SentenceTransformer model.
        texts: List of strings to encode.
        batch_size: Batch size for encoding.

    Returns:
        Tensor of shape (N, 768) with L2-normalized embeddings.
    """
    # sentence-transformers .encode() returns numpy array, handles batching
    # and shows its own progress bar when show_progress_bar=True.
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embs.cpu()


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def _load_coco_annotations(split: str) -> tuple[list[Path], list[str]]:
    """Parse COCO captions JSON, return (image_paths, captions).

    Uses first caption per image, sorted by image id for reproducibility.
    """
    split_name = {"train": "train2017", "val": "val2017"}[split]
    ann_file = DATA_DIR / "coco" / "annotations" / f"captions_{split_name}.json"
    img_dir = DATA_DIR / "coco" / split_name

    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_file}. "
            f"Expected directory layout: data/datasets/coco/annotations/ and data/datasets/coco/{split_name}/"
        )
    if not img_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found at {img_dir}")

    logger.info("Loading COCO annotations from %s ...", ann_file)
    with open(ann_file) as f:
        data = json.load(f)

    # Build image_id -> filename map
    id_to_file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}

    # Group captions by image, keep first
    first_caption: dict[int, str] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in first_caption:
            first_caption[img_id] = ann["caption"]

    # Sort by image id for deterministic ordering
    sorted_ids = sorted(first_caption.keys())

    image_paths: list[Path] = []
    captions: list[str] = []
    skipped = 0
    for img_id in sorted_ids:
        p = img_dir / id_to_file[img_id]
        if not p.exists():
            skipped += 1
            continue
        image_paths.append(p)
        captions.append(first_caption[img_id])

    if skipped:
        logger.warning("Skipped %d images with missing files.", skipped)
    logger.info("COCO %s: %d image-caption pairs.", split, len(image_paths))
    return image_paths, captions


def extract_coco(
    split: str,
    dinov2: torch.nn.Module,
    text_model: "SentenceTransformer",
    device: torch.device,
    batch_size_img: int,
    batch_size_txt: int,
    num_workers: int,
) -> None:
    """Extract and save COCO embeddings."""
    image_paths, captions = _load_coco_annotations(split)
    transform = get_image_transform()

    out_img = EMB_DIR / f"coco_{split}_img.pt"
    out_txt = EMB_DIR / f"coco_{split}_txt.pt"

    # --- Image embeddings ---
    if out_img.exists():
        logger.info("Image embeddings already exist at %s, skipping.", out_img)
        img_embs = torch.load(out_img, weights_only=True)
    else:
        logger.info("Extracting COCO %s image embeddings...", split)
        ds = ImageListDataset(image_paths, transform)
        img_embs = extract_image_embeddings(dinov2, ds, batch_size_img, device, num_workers)
        torch.save(img_embs, out_img)
        logger.info("Saved image embeddings %s to %s", tuple(img_embs.shape), out_img)

    # --- Text embeddings ---
    if out_txt.exists():
        logger.info("Text embeddings already exist at %s, skipping.", out_txt)
        txt_embs = torch.load(out_txt, weights_only=True)
    else:
        logger.info("Extracting COCO %s text embeddings...", split)
        txt_embs = extract_text_embeddings(text_model, captions, batch_size_txt)
        torch.save(txt_embs, out_txt)
        logger.info("Saved text embeddings %s to %s", tuple(txt_embs.shape), out_txt)

    assert img_embs.shape[0] == txt_embs.shape[0], (
        f"Mismatch: {img_embs.shape[0]} images vs {txt_embs.shape[0]} texts"
    )
    logger.info("COCO %s done — %d pairs, img %s, txt %s",
                split, img_embs.shape[0], tuple(img_embs.shape), tuple(txt_embs.shape))


# ---------------------------------------------------------------------------
# Flickr30k
# ---------------------------------------------------------------------------


def _load_flickr30k_annotations() -> tuple[list[Path], list[str]]:
    """Load Flickr30k test split image paths and first captions.

    Expected layout:
        data/datasets/flickr30k/images/         — JPEG images
        data/datasets/flickr30k/results_20130124.token   — caption file
    OR
        data/datasets/flickr30k/flickr30k_images/
        data/datasets/flickr30k/results_20130124.token

    The .token file format: <image_name>#<idx>\t<caption>
    """
    base = DATA_DIR / "flickr30k"

    # Find image directory (two common layouts)
    img_dir: Optional[Path] = None
    for candidate in ["flickr30k_images", "flickr30k-images", "images"]:
        p = base / candidate
        if p.is_dir():
            img_dir = p
            break
    if img_dir is None:
        raise FileNotFoundError(
            f"Flickr30k images not found. Expected one of: "
            f"{base}/images/, {base}/flickr30k_images/, {base}/flickr30k-images/"
        )

    # Find caption file
    token_file: Optional[Path] = None
    for candidate in ["results_20130124.token", "results.token", "captions.token"]:
        p = base / candidate
        if p.exists():
            token_file = p
            break
    if token_file is None:
        raise FileNotFoundError(
            f"Flickr30k captions not found. Expected results_20130124.token in {base}"
        )

    logger.info("Loading Flickr30k captions from %s ...", token_file)

    # Parse: each line is "imagefilename#idx\tcaption"
    first_caption: dict[str, str] = {}
    with open(token_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on tab
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            key, caption = parts
            # key format: "1000092795.jpg#0"
            filename = key.rsplit("#", 1)[0]
            if filename not in first_caption:
                first_caption[filename] = caption

    # Sort for deterministic ordering
    sorted_filenames = sorted(first_caption.keys())

    image_paths: list[Path] = []
    captions: list[str] = []
    skipped = 0
    for fname in sorted_filenames:
        p = img_dir / fname
        if not p.exists():
            skipped += 1
            continue
        image_paths.append(p)
        captions.append(first_caption[fname])

    if skipped:
        logger.warning("Skipped %d Flickr30k images with missing files.", skipped)
    logger.info("Flickr30k: %d image-caption pairs.", len(image_paths))
    return image_paths, captions


def extract_flickr30k(
    dinov2: torch.nn.Module,
    text_model: "SentenceTransformer",
    device: torch.device,
    batch_size_img: int,
    batch_size_txt: int,
    num_workers: int,
) -> None:
    """Extract and save Flickr30k test embeddings."""
    image_paths, captions = _load_flickr30k_annotations()
    transform = get_image_transform()

    out_img = EMB_DIR / "flickr30k_test_img.pt"
    out_txt = EMB_DIR / "flickr30k_test_txt.pt"

    if out_img.exists():
        logger.info("Flickr30k image embeddings already exist at %s, skipping.", out_img)
        img_embs = torch.load(out_img, weights_only=True)
    else:
        logger.info("Extracting Flickr30k image embeddings...")
        ds = ImageListDataset(image_paths, transform)
        img_embs = extract_image_embeddings(dinov2, ds, batch_size_img, device, num_workers)
        torch.save(img_embs, out_img)
        logger.info("Saved image embeddings %s to %s", tuple(img_embs.shape), out_img)

    if out_txt.exists():
        logger.info("Flickr30k text embeddings already exist at %s, skipping.", out_txt)
        txt_embs = torch.load(out_txt, weights_only=True)
    else:
        logger.info("Extracting Flickr30k text embeddings...")
        txt_embs = extract_text_embeddings(text_model, captions, batch_size_txt)
        torch.save(txt_embs, out_txt)
        logger.info("Saved text embeddings %s to %s", tuple(txt_embs.shape), out_txt)

    assert img_embs.shape[0] == txt_embs.shape[0], (
        f"Mismatch: {img_embs.shape[0]} images vs {txt_embs.shape[0]} texts"
    )
    logger.info("Flickr30k done — %d pairs, img %s, txt %s",
                img_embs.shape[0], tuple(img_embs.shape), tuple(txt_embs.shape))


# ---------------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------------


def _get_imagenet_class_names() -> list[str]:
    """Return the 1000 ImageNet class names in synset order.

    Tries to load from a local file first. Falls back to torchvision's
    built-in mapping.
    """
    local_file = DATA_DIR / "imagenet" / "imagenet_classes.txt"
    if local_file.exists():
        logger.info("Loading ImageNet class names from %s", local_file)
        with open(local_file) as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == 1000:
            return names
        logger.warning("Expected 1000 classes in %s, got %d. Falling back.", local_file, len(names))

    # Fallback: use torchvision's built-in ImageNet class names via the
    # IMAGENET1K_V1 weights meta information.
    try:
        from torchvision.models import ResNet50_Weights
        meta = ResNet50_Weights.IMAGENET1K_V1.meta
        names = meta["categories"]
        if len(names) == 1000:
            logger.info("Loaded 1000 ImageNet class names from torchvision metadata.")
            return names
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not determine ImageNet class names. Place a file with one "
        "class name per line (1000 lines) at data/datasets/imagenet/imagenet_classes.txt"
    )


def _collect_imagenet_val_images(val_dir: Path) -> tuple[list[Path], list[int]]:
    """Collect ImageNet val images and their class indices.

    Supports two layouts:
      1. Subdirectory per class: val/n01440764/ILSVRC*.JPEG
      2. Flat directory with ILSVRC2012_val_ground_truth.txt alongside
    """
    exts = {".jpg", ".jpeg", ".png", ".JPEG"}

    # Layout 1: class subdirectories
    subdirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    if subdirs:
        logger.info("ImageNet val: detected class-subdirectory layout (%d classes).", len(subdirs))
        image_paths: list[Path] = []
        labels: list[int] = []
        for class_idx, subdir in enumerate(subdirs):
            imgs = sorted(p for p in subdir.iterdir() if p.suffix in exts)
            image_paths.extend(imgs)
            labels.extend([class_idx] * len(imgs))
        logger.info("ImageNet val: %d images across %d classes.", len(image_paths), len(subdirs))
        return image_paths, labels

    # Layout 2: flat directory + ground truth file
    gt_file = val_dir.parent / "ILSVRC2012_val_ground_truth.txt"
    if not gt_file.exists():
        gt_file = val_dir.parent / "ground_truth.txt"
    if gt_file.exists():
        logger.info("ImageNet val: detected flat layout with ground truth file.")
        all_imgs = sorted(p for p in val_dir.iterdir() if p.suffix in exts)
        with open(gt_file) as f:
            gt_labels = [int(line.strip()) for line in f if line.strip()]
        if len(all_imgs) != len(gt_labels):
            raise ValueError(
                f"Image count ({len(all_imgs)}) != ground truth count ({len(gt_labels)})"
            )
        return all_imgs, gt_labels

    raise FileNotFoundError(
        f"Cannot parse ImageNet val at {val_dir}. Expected either class "
        f"subdirectories or a flat directory with a ground truth file."
    )


def extract_imagenet(
    dinov2: torch.nn.Module,
    text_model: "SentenceTransformer",
    device: torch.device,
    batch_size_img: int,
    batch_size_txt: int,
    num_workers: int,
) -> None:
    """Extract and save ImageNet val embeddings + class text embeddings."""
    val_dir = DATA_DIR / "imagenet" / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet val directory not found at {val_dir}. "
            f"Expected layout: data/datasets/imagenet/val/"
        )

    transform = get_image_transform()
    out_img = EMB_DIR / "imagenet_val_img.pt"
    out_labels = EMB_DIR / "imagenet_val_labels.pt"
    out_txt = EMB_DIR / "imagenet_val_txt.pt"

    # --- Image embeddings ---
    image_paths, labels = _collect_imagenet_val_images(val_dir)

    if out_img.exists() and out_labels.exists():
        logger.info("ImageNet image embeddings already exist at %s, skipping.", out_img)
        img_embs = torch.load(out_img, weights_only=True)
    else:
        logger.info("Extracting ImageNet val image embeddings (%d images)...", len(image_paths))
        ds = ImageListDataset(image_paths, transform)
        img_embs = extract_image_embeddings(dinov2, ds, batch_size_img, device, num_workers)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        torch.save(img_embs, out_img)
        torch.save(labels_tensor, out_labels)
        logger.info("Saved image embeddings %s to %s", tuple(img_embs.shape), out_img)
        logger.info("Saved labels %s to %s", tuple(labels_tensor.shape), out_labels)

    # --- Class name text embeddings ---
    if out_txt.exists():
        logger.info("ImageNet text embeddings already exist at %s, skipping.", out_txt)
        txt_embs = torch.load(out_txt, weights_only=True)
    else:
        class_names = _get_imagenet_class_names()
        prompts = [IMAGENET_PROMPT_TEMPLATE.format(name) for name in class_names]
        logger.info("Extracting ImageNet class text embeddings (%d classes)...", len(prompts))
        txt_embs = extract_text_embeddings(text_model, prompts, batch_size_txt)
        torch.save(txt_embs, out_txt)
        logger.info("Saved text embeddings %s to %s", tuple(txt_embs.shape), out_txt)

    logger.info("ImageNet done — %d images, %d class texts, img %s, txt %s",
                img_embs.shape[0], txt_embs.shape[0],
                tuple(img_embs.shape), tuple(txt_embs.shape))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from frozen encoders and save as .pt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["coco", "flickr30k", "imagenet", "all"],
        help="Which dataset to extract embeddings for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (e.g. train, val, test). "
             "Required for coco. Ignored for 'all'.",
    )
    parser.add_argument(
        "--batch-size-img",
        type=int,
        default=64,
        help="Batch size for image embedding extraction.",
    )
    parser.add_argument(
        "--batch-size-txt",
        type=int,
        default=256,
        help="Batch size for text embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate split arg
    if args.dataset == "coco" and args.split is None:
        args.split = "train"
        logger.info("No split specified for COCO, defaulting to 'train'.")
    if args.dataset == "coco" and args.split not in ("train", "val"):
        logger.error("COCO split must be 'train' or 'val', got '%s'.", args.split)
        sys.exit(1)

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Create output directory
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # If --force, remove existing files for the requested dataset
    if args.force:
        _remove_existing(args.dataset, args.split)

    # Load encoders (only once, even for --dataset all)
    dinov2 = load_dinov2(device)
    text_model = load_text_encoder()

    common_kwargs = dict(
        dinov2=dinov2,
        text_model=text_model,
        device=device,
        batch_size_img=args.batch_size_img,
        batch_size_txt=args.batch_size_txt,
        num_workers=args.num_workers,
    )

    if args.dataset in ("coco", "all"):
        split = args.split or "train"
        logger.info("=" * 60)
        logger.info("Extracting COCO %s embeddings", split)
        logger.info("=" * 60)
        extract_coco(split=split, **common_kwargs)

    if args.dataset in ("flickr30k", "all"):
        logger.info("=" * 60)
        logger.info("Extracting Flickr30k test embeddings")
        logger.info("=" * 60)
        extract_flickr30k(**common_kwargs)

    if args.dataset in ("imagenet", "all"):
        logger.info("=" * 60)
        logger.info("Extracting ImageNet val embeddings")
        logger.info("=" * 60)
        extract_imagenet(**common_kwargs)

    logger.info("All done.")


def _remove_existing(dataset: str, split: Optional[str]) -> None:
    """Remove existing embedding files when --force is set."""
    patterns: list[str] = []
    if dataset in ("coco", "all"):
        s = split or "train"
        patterns += [f"coco_{s}_img.pt", f"coco_{s}_txt.pt"]
    if dataset in ("flickr30k", "all"):
        patterns += ["flickr30k_test_img.pt", "flickr30k_test_txt.pt"]
    if dataset in ("imagenet", "all"):
        patterns += ["imagenet_val_img.pt", "imagenet_val_labels.pt", "imagenet_val_txt.pt"]

    for name in patterns:
        p = EMB_DIR / name
        if p.exists():
            p.unlink()
            logger.info("Removed existing file: %s", p)


if __name__ == "__main__":
    main()
