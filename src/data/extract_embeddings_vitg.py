"""Extract embeddings from DINOv2 ViT-Giant + RoBERTa-large.

Produces the same file layout as the existing all_tokens/ directory
but with ViT-G (1536-dim) and RoBERTa-large (1024-dim) embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m src.data.extract_embeddings_vitg
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "datasets"
OUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "vitg_roberta"

IMAGE_SIZE = 224
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
CHUNK_SIZE = 5000  # Smaller chunks for 1536-dim (vs 10000 for 768-dim)


# ── Dataset helpers ────────────────────────────────────────────────────

class ImageListDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), idx


def get_image_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


def _load_coco_annotations() -> tuple[list[Path], list[str]]:
    """Load COCO train2017 — same ordering as extract_embeddings.py."""
    ann_file = DATA_DIR / "coco" / "annotations" / "captions_train2017.json"
    with open(ann_file) as f:
        data = json.load(f)
    img_id_to_fn = {img["id"]: img["file_name"] for img in data["images"]}
    first_cap: dict[int, str] = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in first_cap:
            first_cap[ann["image_id"]] = ann["caption"]
    sorted_ids = sorted(first_cap.keys())
    img_dir = DATA_DIR / "coco" / "train2017"
    paths, captions = [], []
    for iid in sorted_ids:
        p = img_dir / img_id_to_fn[iid]
        if p.exists():
            paths.append(p)
            captions.append(first_cap[iid])
    logger.info("COCO train: %d pairs", len(paths))
    return paths, captions


def _load_flickr_annotations() -> tuple[list[Path], list[str]]:
    """Load Flickr30k test — same ordering as extract_embeddings.py."""
    base = DATA_DIR / "flickr30k"
    img_dir = None
    for candidate in ["flickr30k_images", "flickr30k-images", "images"]:
        p = base / candidate
        if p.is_dir():
            img_dir = p
            break
    token_file = base / "results_20130124.token"
    first_cap: dict[str, str] = {}
    with open(token_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            key, caption = parts
            fname = key.rsplit("#", 1)[0]
            if fname not in first_cap:
                first_cap[fname] = caption
    sorted_fnames = sorted(first_cap.keys())
    paths, captions = [], []
    for fn in sorted_fnames:
        p = img_dir / fn
        if p.exists():
            paths.append(p)
            captions.append(first_cap[fn])
    logger.info("Flickr30k: %d pairs", len(paths))
    return paths, captions


# ── DINOv2 ViT-Giant extraction ───────────────────────────────────────

@torch.no_grad()
def extract_dinov2_vitg(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Extract CLS + patch tokens from DINOv2 ViT-Giant.

    Returns:
        cls_embs: (N, 1536) float32
        chunk_list: list of (chunk_size, 257, 1536) float16 tensors
    """
    logger.info("Loading DINOv2 ViT-Giant (dinov2_vitg14)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    model = model.to(device).eval()

    transform = get_image_transform()
    dataset = ImageListDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_cls = []
    all_tokens = []
    current_chunk = []
    chunk_list = []

    for batch_imgs, _ in tqdm(loader, desc="DINOv2 ViT-G"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        # get_intermediate_layers returns [(patch_tokens, cls_token)]
        out = model.get_intermediate_layers(
            batch_imgs, n=1, return_class_token=True, norm=True,
        )
        patch_tokens = out[0][0]  # (B, 256, 1536)
        cls_token = out[0][1]     # (B, 1536)

        # L2 normalize
        cls_normed = F.normalize(cls_token, dim=-1)
        tokens = torch.cat([cls_normed.unsqueeze(1),
                            F.normalize(patch_tokens, dim=-1)], dim=1)  # (B, 257, 1536)

        all_cls.append(cls_normed.cpu())
        current_chunk.append(tokens.half().cpu())

        # Check if chunk is full
        total_in_chunk = sum(t.shape[0] for t in current_chunk)
        if total_in_chunk >= CHUNK_SIZE:
            chunk_tensor = torch.cat(current_chunk, dim=0)
            chunk_list.append(chunk_tensor[:CHUNK_SIZE])
            # Carry over remainder
            if chunk_tensor.shape[0] > CHUNK_SIZE:
                current_chunk = [chunk_tensor[CHUNK_SIZE:]]
            else:
                current_chunk = []

    # Final chunk
    if current_chunk:
        chunk_list.append(torch.cat(current_chunk, dim=0))

    cls_embs = torch.cat(all_cls, dim=0)  # (N, 1536) float32
    return cls_embs, chunk_list


# ── RoBERTa-large extraction ──────────────────────────────────────────

@torch.no_grad()
def extract_roberta(
    captions: list[str],
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 77,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract CLS (mean-pooled) + token embeddings from RoBERTa-large.

    Returns:
        cls_embs: (N, 1024) float32 — mean-pooled sentence embeddings
        token_embs: (N, max_seq_len, 1024) float16
        masks: (N, max_seq_len) bool
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info("Loading RoBERTa-large...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModel.from_pretrained("roberta-large").to(device).eval()

    all_cls = []
    all_tokens = []
    all_masks = []
    actual_max_len = 0

    for i in tqdm(range(0, len(captions), batch_size), desc="RoBERTa-large"):
        batch_texts = captions[i:i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        hidden = outputs.last_hidden_state  # (B, L, 1024)

        # Mean pooling (masked)
        attn_mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        mean_pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)  # (B, 1024)
        mean_pooled = F.normalize(mean_pooled, dim=-1)

        # L2 normalize tokens
        token_normed = F.normalize(hidden, dim=-1)

        seq_len = hidden.shape[1]
        actual_max_len = max(actual_max_len, seq_len)

        all_cls.append(mean_pooled.cpu())
        all_tokens.append(token_normed.half().cpu())
        all_masks.append(encoded["attention_mask"].bool().cpu())

    # Pad all batches to same max length
    padded_tokens = []
    padded_masks = []
    for t, m in zip(all_tokens, all_masks):
        if t.shape[1] < actual_max_len:
            pad_t = torch.zeros(t.shape[0], actual_max_len - t.shape[1], t.shape[2],
                                dtype=t.dtype)
            t = torch.cat([t, pad_t], dim=1)
            pad_m = torch.zeros(m.shape[0], actual_max_len - m.shape[1], dtype=m.dtype)
            m = torch.cat([m, pad_m], dim=1)
        padded_tokens.append(t)
        padded_masks.append(m)

    cls_embs = torch.cat(all_cls, dim=0)       # (N, 1024)
    token_embs = torch.cat(padded_tokens, dim=0)  # (N, max_len, 1024)
    masks = torch.cat(padded_masks, dim=0)      # (N, max_len)

    return cls_embs, token_embs, masks


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── COCO train ──
    coco_paths, coco_captions = _load_coco_annotations()

    # Image embeddings (DINOv2 ViT-G)
    coco_cls_img_path = OUT_DIR / "coco_train_img.pt"
    if not coco_cls_img_path.exists():
        cls_img, chunks = extract_dinov2_vitg(coco_paths, device, batch_size=32)
        logger.info("COCO img CLS: %s", tuple(cls_img.shape))
        torch.save(cls_img, coco_cls_img_path)

        # Save chunks + metadata
        metadata = {
            "num_chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "total_samples": cls_img.shape[0],
            "tokens_per_image": 257,
            "embedding_dim": 1536,
            "dtype": "float16",
            "chunks": [],
        }
        offset = 0
        for i, chunk in enumerate(chunks):
            fname = f"coco_train_chunk_{i:02d}_img.pt"
            torch.save(chunk, OUT_DIR / fname)
            metadata["chunks"].append({
                "index": i,
                "filename": fname,
                "start_idx": offset,
                "end_idx": offset + chunk.shape[0],
                "num_samples": chunk.shape[0],
            })
            offset += chunk.shape[0]
            logger.info("Saved %s: %s", fname, tuple(chunk.shape))

        with open(OUT_DIR / "chunk_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved chunk_metadata.json (%d chunks)", len(chunks))
        del cls_img, chunks
        torch.cuda.empty_cache()
    else:
        logger.info("COCO img already extracted, skipping")

    # Text embeddings (RoBERTa-large)
    coco_cls_txt_path = OUT_DIR / "coco_train_txt.pt"
    if not coco_cls_txt_path.exists():
        cls_txt, tok_txt, masks_txt = extract_roberta(coco_captions, device)
        logger.info("COCO txt CLS: %s, tokens: %s, masks: %s",
                     tuple(cls_txt.shape), tuple(tok_txt.shape), tuple(masks_txt.shape))
        torch.save(cls_txt, coco_cls_txt_path)
        torch.save(tok_txt, OUT_DIR / "coco_train_txt_tokens.pt")
        torch.save(masks_txt, OUT_DIR / "coco_train_txt_mask.pt")
        del cls_txt, tok_txt, masks_txt
        torch.cuda.empty_cache()
    else:
        logger.info("COCO txt already extracted, skipping")

    # ── Flickr30k test ──
    flickr_paths, flickr_captions = _load_flickr_annotations()

    # Image — save tokens as flickr30k_test_img.pt (matches train.py loading convention)
    flickr_img_path = OUT_DIR / "flickr30k_test_img.pt"
    if not flickr_img_path.exists():
        cls_img, chunks = extract_dinov2_vitg(flickr_paths, device, batch_size=32)
        # Flickr is small enough to save as one file
        all_tokens = torch.cat(chunks, dim=0)  # (31783, 257, 1536) float16
        torch.save(all_tokens, flickr_img_path)
        logger.info("Flickr img: CLS %s, tokens %s",
                     tuple(cls_img.shape), tuple(all_tokens.shape))
        del cls_img, chunks, all_tokens
        torch.cuda.empty_cache()
    else:
        logger.info("Flickr img already extracted, skipping")

    # Text — CLS as flickr30k_test_txt.pt, tokens separately
    flickr_txt_path = OUT_DIR / "flickr30k_test_txt.pt"
    if not flickr_txt_path.exists():
        cls_txt, tok_txt, masks_txt = extract_roberta(flickr_captions, device)
        torch.save(cls_txt, flickr_txt_path)  # (N, 1024) CLS/mean-pooled
        torch.save(tok_txt, OUT_DIR / "flickr30k_test_txt_tokens.pt")
        torch.save(masks_txt, OUT_DIR / "flickr30k_test_txt_mask.pt")
        logger.info("Flickr txt: CLS %s, tokens %s, masks %s",
                     tuple(cls_txt.shape), tuple(tok_txt.shape), tuple(masks_txt.shape))
    else:
        logger.info("Flickr txt already extracted, skipping")

    logger.info("All done! Embeddings saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
