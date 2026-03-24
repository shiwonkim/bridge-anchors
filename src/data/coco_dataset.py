"""MSCOCO paired embedding dataset for training.

Loads pre-extracted image and text embedding tensors from .pt files.
Supports deterministic subsampling for data efficiency experiments
(Experiment C) and optional train/val splitting for monitoring
training-set overfitting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PairedEmbeddingDataset(Dataset):
    """Loads pre-extracted embedding pairs from ``.pt`` files.

    Each ``.pt`` file contains a single tensor of shape ``(N, D)`` where
    every row corresponds to one sample.  The *i*-th image embedding is
    paired with the *i*-th text embedding (1-to-1 correspondence produced
    by ``extract_embeddings.py``).

    Args:
        img_emb_path: Path to the image embeddings ``.pt`` file.
        txt_emb_path: Path to the text embeddings ``.pt`` file.
        num_samples: If set, randomly subsample this many pairs.  A fixed
            ``seed`` ensures reproducibility across runs.
        seed: RNG seed used for subsampling.
        split: ``'train'``, ``'val'``, or ``None``.  When set, the full
            dataset is deterministically partitioned: the first
            ``1 - val_fraction`` rows go to train, the rest to val.
            Applied *before* ``num_samples`` subsampling.
        val_fraction: Fraction of data reserved for the val split
            (default 0.05 ≈ 5 900 pairs for COCO 118 K).
    """

    def __init__(
        self,
        img_emb_path: str | Path,
        txt_emb_path: str | Path,
        num_samples: int | None = None,
        seed: int = 42,
        split: str | None = None,
        val_fraction: float = 0.05,
    ) -> None:
        img_emb_path = Path(img_emb_path)
        txt_emb_path = Path(txt_emb_path)

        if not img_emb_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {img_emb_path}")
        if not txt_emb_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {txt_emb_path}")

        img_embs: torch.Tensor = torch.load(img_emb_path, weights_only=True)
        txt_embs: torch.Tensor = torch.load(txt_emb_path, weights_only=True)

        if img_embs.shape[0] != txt_embs.shape[0]:
            raise ValueError(
                f"Pair count mismatch: {img_embs.shape[0]} images vs "
                f"{txt_embs.shape[0]} texts"
            )

        n_total = img_embs.shape[0]
        logger.info(
            "Loaded embeddings: img %s, txt %s from %s, %s",
            tuple(img_embs.shape),
            tuple(txt_embs.shape),
            img_emb_path.name,
            txt_emb_path.name,
        )

        # ----- deterministic train / val split -----
        if split is not None:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train' or 'val', got {split!r}")
            if not 0.0 < val_fraction < 1.0:
                raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

            n_val = max(1, int(n_total * val_fraction))
            n_train = n_total - n_val

            # Use a fixed permutation so train and val never overlap
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n_total, generator=gen)

            if split == "train":
                indices = perm[:n_train]
            else:
                indices = perm[n_train:]

            img_embs = img_embs[indices]
            txt_embs = txt_embs[indices]
            logger.info(
                "Split '%s': %d / %d pairs (val_fraction=%.3f)",
                split,
                img_embs.shape[0],
                n_total,
                val_fraction,
            )

        # ----- subsampling for data efficiency experiments -----
        if num_samples is not None:
            current_n = img_embs.shape[0]
            if num_samples > current_n:
                logger.warning(
                    "Requested %d samples but only %d available after split; "
                    "using all %d.",
                    num_samples,
                    current_n,
                    current_n,
                )
                num_samples = current_n

            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(current_n, generator=gen)[:num_samples]
            img_embs = img_embs[indices]
            txt_embs = txt_embs[indices]
            logger.info("Subsampled to %d pairs.", num_samples)

        self.img_embs = img_embs
        self.txt_embs = txt_embs

    def __len__(self) -> int:
        return self.img_embs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single (img_embedding, txt_embedding) pair.

        Returns:
            img_emb: (dim_img,) tensor.
            txt_emb: (dim_txt,) tensor.
        """
        return self.img_embs[idx], self.txt_embs[idx]

    @property
    def dim_img(self) -> int:
        """Dimension of image embeddings."""
        return self.img_embs.shape[1]

    @property
    def dim_txt(self) -> int:
        """Dimension of text embeddings."""
        return self.txt_embs.shape[1]

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full embedding matrices (useful for evaluation).

        Returns:
            img_embs: (N, dim_img) tensor.
            txt_embs: (N, dim_txt) tensor.
        """
        return self.img_embs, self.txt_embs
