"""Extract text token-level embeddings from all-mpnet-base-v2.

Extracts ALL token outputs from the transformer (before pooling), not just
the pooled CLS output. Saves token embeddings and attention masks for
masked mean pooling during training.

Output files (in data/embeddings/all_tokens/):
    coco_train_txt_tokens.pt    — (118287, max_seq_len, 768) float16
    coco_train_txt_mask.pt      — (118287, max_seq_len) bool
    flickr30k_test_txt_tokens.pt
    flickr30k_test_txt_mask.pt

Usage:
    python scripts/extraction/extract_text_token_embeddings.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "all_tokens"


@torch.no_grad()
def extract_text_tokens(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 256,
    max_length: int = 128,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract token-level embeddings from sentence-transformers model.

    Uses the underlying transformer directly (not the pooling layer) to get
    all token outputs.

    Args:
        model: SentenceTransformer model.
        tokenizer: The model's tokenizer.
        texts: List of strings.
        batch_size: Batch size.
        max_length: Maximum sequence length for tokenization.
        device: Target device for model inference.

    Returns:
        token_embs: (N, max_seq_len, 768) float16 — padded token embeddings
        attention_masks: (N, max_seq_len) bool — attention masks
    """
    # Get the underlying transformer
    transformer = model[0].auto_model
    if device is not None:
        transformer = transformer.to(device)
    transformer.eval()

    all_token_embs = []
    all_masks = []
    actual_max_len = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="  text tokens"):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward through transformer (no pooling)
        outputs = transformer(**encoded)
        token_outputs = outputs.last_hidden_state  # (B, seq_len, 768)

        # L2 normalize each token
        token_outputs = torch.nn.functional.normalize(token_outputs, dim=-1)

        seq_len = token_outputs.shape[1]
        actual_max_len = max(actual_max_len, seq_len)

        all_token_embs.append(token_outputs.half().cpu())
        all_masks.append(encoded["attention_mask"].bool().cpu())

    # Pad all batches to the same max length
    logger.info("Max sequence length across all batches: %d", actual_max_len)

    padded_embs = []
    padded_masks = []
    dim = all_token_embs[0].shape[-1]

    for embs, masks in zip(all_token_embs, all_masks):
        seq_len = embs.shape[1]
        if seq_len < actual_max_len:
            pad_len = actual_max_len - seq_len
            embs = torch.cat([
                embs,
                torch.zeros(embs.shape[0], pad_len, dim, dtype=embs.dtype),
            ], dim=1)
            masks = torch.cat([
                masks,
                torch.zeros(masks.shape[0], pad_len, dtype=masks.dtype),
            ], dim=1)
        padded_embs.append(embs)
        padded_masks.append(masks)

    token_embs = torch.cat(padded_embs, dim=0)    # (N, max_seq_len, 768) float16
    attention_masks = torch.cat(padded_masks, dim=0)  # (N, max_seq_len) bool

    return token_embs, attention_masks


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from src.data.extract_embeddings import (
        _load_coco_annotations,
        _load_flickr30k_annotations,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model and tokenizer
    logger.info("Loading all-mpnet-base-v2...")
    model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer = model.tokenizer

    t_start = time.time()

    # --- COCO train ---
    coco_emb_path = OUT_DIR / "coco_train_txt_tokens.pt"
    coco_mask_path = OUT_DIR / "coco_train_txt_mask.pt"

    if coco_emb_path.exists() and coco_mask_path.exists():
        logger.info("COCO text tokens already exist, skipping.")
    else:
        logger.info("=" * 60)
        logger.info("Extracting COCO train text tokens...")
        logger.info("=" * 60)

        _, captions = _load_coco_annotations("train")
        logger.info("COCO train: %d captions", len(captions))

        coco_tokens, coco_masks = extract_text_tokens(
            model, tokenizer, captions, batch_size=256, device=device,
        )

        torch.save(coco_tokens, coco_emb_path)
        torch.save(coco_masks, coco_mask_path)
        logger.info("Saved COCO text tokens: %s (%.2f GB)",
                    tuple(coco_tokens.shape),
                    coco_emb_path.stat().st_size / 1e9)
        logger.info("Saved COCO text masks: %s (%.2f MB)",
                    tuple(coco_masks.shape),
                    coco_mask_path.stat().st_size / 1e6)

    # --- Flickr30k test ---
    flickr_emb_path = OUT_DIR / "flickr30k_test_txt_tokens.pt"
    flickr_mask_path = OUT_DIR / "flickr30k_test_txt_mask.pt"

    if flickr_emb_path.exists() and flickr_mask_path.exists():
        logger.info("Flickr30k text tokens already exist, skipping.")
    else:
        logger.info("=" * 60)
        logger.info("Extracting Flickr30k test text tokens...")
        logger.info("=" * 60)

        _, captions_f = _load_flickr30k_annotations()
        logger.info("Flickr30k: %d captions", len(captions_f))

        flickr_tokens, flickr_masks = extract_text_tokens(
            model, tokenizer, captions_f, batch_size=256, device=device,
        )

        torch.save(flickr_tokens, flickr_emb_path)
        torch.save(flickr_masks, flickr_mask_path)
        logger.info("Saved Flickr30k text tokens: %s (%.2f GB)",
                    tuple(flickr_tokens.shape),
                    flickr_emb_path.stat().st_size / 1e9)
        logger.info("Saved Flickr30k text masks: %s (%.2f MB)",
                    tuple(flickr_masks.shape),
                    flickr_mask_path.stat().st_size / 1e6)

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("ALL DONE. Total time: %.1f min", total_time / 60)

    # Verify
    logger.info("Files in %s:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        if "txt_token" in f.name or "txt_mask" in f.name:
            logger.info("  %s (%.2f GB)", f.name, f.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
