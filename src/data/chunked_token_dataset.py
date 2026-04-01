"""Chunked token-level embedding dataset for full-scale token BridgeAnchors.

Loads token-level image embeddings from chunks stored as float16 on local SSD.
Each chunk (~3.7GB) is loaded one at a time, cast to float32 on GPU, iterated
in shuffled batches, then freed before loading the next chunk.

Optionally loads text token-level embeddings and attention masks for models
that process text tokens (Freeze-Align, Token BA with text tokens).

Usage:
    dataset = ChunkedTokenDataset(
        chunk_dir="data/embeddings/all_tokens",
        text_emb_path="data/embeddings/all_tokens/coco_train_txt.pt",
        batch_size=128,
        seed=42,
    )
    for epoch in range(20):
        for batch in dataset.epoch_iterator(epoch, device):
            img_tokens, txt_embs = batch[:2]
            txt_tokens, txt_mask = batch[2], batch[3]  # if text_token_level=True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ChunkedTokenDataset:
    """Chunked token-level embedding dataset with float16 storage.

    Chunks are stored as float16 on disk. When loaded for a batch, they
    are cast to float32 before being sent to the GPU.

    Args:
        chunk_dir: Directory containing chunk .pt files and metadata JSON.
        text_emb_path: Path to full text embedding file (N, 768) float32.
        batch_size: Training batch size.
        seed: Base seed for reproducibility.
        val_fraction: Fraction of data reserved for validation.
        split: ``'train'`` or ``'val'``.
        text_token_level: If True, also loads text token embeddings and masks.
        text_token_path: Path to text token embeddings (N, S, 768) float16.
        text_mask_path: Path to text attention masks (N, S) bool.
    """

    def __init__(
        self,
        chunk_dir: str | Path,
        text_emb_path: str | Path,
        batch_size: int = 128,
        seed: int = 42,
        val_fraction: float = 0.05,
        split: str = "train",
        text_token_level: bool = False,
        text_token_path: str | Path | None = None,
        text_mask_path: str | Path | None = None,
    ) -> None:
        self.chunk_dir = Path(chunk_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.text_token_level = text_token_level

        # Load metadata
        meta_path = self.chunk_dir / "chunk_metadata.json"
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.num_chunks = self.metadata["num_chunks"]
        self.total_samples = self.metadata["total_samples"]

        # Load CLS text embeddings (small: 118K × 768 float32 = ~347MB)
        logger.info("Loading text embeddings from %s...", text_emb_path)
        self.txt_embs = torch.load(text_emb_path, weights_only=True).float()
        assert self.txt_embs.shape[0] == self.total_samples, (
            f"Text embeddings ({self.txt_embs.shape[0]}) != "
            f"total samples ({self.total_samples})"
        )

        # Optionally load text token-level embeddings
        self.txt_tokens = None
        self.txt_masks = None
        if text_token_level:
            if text_token_path is None:
                text_token_path = self.chunk_dir / "coco_train_txt_tokens.pt"
            if text_mask_path is None:
                text_mask_path = self.chunk_dir / "coco_train_txt_mask.pt"

            logger.info("Loading text token embeddings from %s...", text_token_path)
            self.txt_tokens = torch.load(text_token_path, weights_only=True)
            logger.info("Loading text attention masks from %s...", text_mask_path)
            self.txt_masks = torch.load(text_mask_path, weights_only=True)

            assert self.txt_tokens.shape[0] == self.total_samples
            assert self.txt_masks.shape[0] == self.total_samples
            logger.info("Text tokens: %s %s, masks: %s %s",
                       tuple(self.txt_tokens.shape), self.txt_tokens.dtype,
                       tuple(self.txt_masks.shape), self.txt_masks.dtype)

        # Deterministic train/val split at the sample level
        n_val = max(1, int(self.total_samples * val_fraction))
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.total_samples, generator=gen)
        if split == "val":
            self._valid_indices = set(perm[:n_val].tolist())
        else:
            self._valid_indices = set(perm[n_val:].tolist())

        # Precompute per-chunk valid global indices
        self._chunk_global_indices: list[list[int]] = []
        for chunk_info in self.metadata["chunks"]:
            start = chunk_info["start_idx"]
            end = chunk_info["end_idx"]
            global_idxs = [i for i in range(start, end) if i in self._valid_indices]
            self._chunk_global_indices.append(global_idxs)

        self._n_samples = len(self._valid_indices)
        logger.info(
            "ChunkedTokenDataset: %s split, %d samples across %d chunks, bs=%d, txt_tokens=%s",
            split, self._n_samples, self.num_chunks, batch_size, text_token_level,
        )

    def __len__(self) -> int:
        return self._n_samples

    @property
    def n_batches_approx(self) -> int:
        """Approximate batches per epoch (for LR scheduler)."""
        return self._n_samples // self.batch_size

    def epoch_iterator(
        self,
        epoch: int,
        device: torch.device | None = None,
    ):
        """Yield batches for one epoch.

        Chunks are shuffled each epoch. Within each chunk, samples are
        shuffled. Chunks are loaded as float16 from disk and cast to
        float32 when batching to GPU.

        Args:
            epoch: Current epoch (determines shuffle seed).
            device: Target device for batches.

        Yields:
            If text_token_level=False:
                (img_tokens, txt_embs) — both float32.
            If text_token_level=True:
                (img_tokens, txt_embs, txt_tokens, txt_mask) — all float32/bool.
        """
        gen = torch.Generator().manual_seed(self.seed + epoch)
        chunk_order = torch.randperm(self.num_chunks, generator=gen).tolist()

        for chunk_idx in chunk_order:
            global_indices = self._chunk_global_indices[chunk_idx]
            if not global_indices:
                continue

            # Load chunk from local SSD (float16, ~3.7GB per chunk)
            chunk_info = self.metadata["chunks"][chunk_idx]
            chunk_path = self.chunk_dir / chunk_info["filename"]
            chunk_img = torch.load(chunk_path, weights_only=True)  # (C, 257, 768) float16
            chunk_start = chunk_info["start_idx"]

            # Shuffle within chunk
            local_perm = torch.randperm(len(global_indices), generator=gen)
            shuffled_globals = [global_indices[i] for i in local_perm.tolist()]

            # Yield batches
            for batch_start in range(0, len(shuffled_globals), self.batch_size):
                batch_globals = shuffled_globals[batch_start:batch_start + self.batch_size]
                if len(batch_globals) < self.batch_size // 2:
                    continue  # skip tiny tail batches

                batch_locals = [g - chunk_start for g in batch_globals]

                # Cast float16 → float32 when creating batch
                img_batch = chunk_img[batch_locals].float()   # (B, 257, 768) float32
                txt_batch = self.txt_embs[batch_globals]       # (B, 768) float32

                if device is not None:
                    img_batch = img_batch.to(device, non_blocking=True)
                    txt_batch = txt_batch.to(device, non_blocking=True)

                if self.text_token_level:
                    txt_tok_batch = self.txt_tokens[batch_globals].float()  # (B, S, 768)
                    txt_mask_batch = self.txt_masks[batch_globals]           # (B, S) bool

                    if device is not None:
                        txt_tok_batch = txt_tok_batch.to(device, non_blocking=True)
                        txt_mask_batch = txt_mask_batch.to(device, non_blocking=True)

                    yield img_batch, txt_batch, txt_tok_batch, txt_mask_batch
                else:
                    yield img_batch, txt_batch

            # Free chunk memory before loading next
            del chunk_img
