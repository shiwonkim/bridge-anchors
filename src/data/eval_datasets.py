"""Evaluation dataset loaders for Flickr30k retrieval and ImageNet zero-shot.

Both classes load pre-extracted ``.pt`` embedding files produced by
``extract_embeddings.py`` and expose the tensors needed by the
evaluation functions in ``src/eval/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Flickr30kEmbeddings:
    """Flickr30k test-set embeddings for image-text retrieval evaluation.

    Loads paired image / text embedding tensors.  The *i*-th image is
    matched with the *i*-th text (first caption per image, as produced
    by ``extract_embeddings.py``).

    Args:
        img_emb_path: Path to ``flickr30k_test_img.pt`` — shape ``(N, dim_img)``.
        txt_emb_path: Path to ``flickr30k_test_txt.pt`` — shape ``(N, dim_txt)``.

    Attributes:
        img_embs: (N, dim_img) image embedding tensor.
        txt_embs: (N, dim_txt) text embedding tensor.
        num_samples: Number of image-text pairs.
    """

    def __init__(
        self,
        img_emb_path: str | Path,
        txt_emb_path: str | Path,
    ) -> None:
        img_emb_path = Path(img_emb_path)
        txt_emb_path = Path(txt_emb_path)

        if not img_emb_path.exists():
            raise FileNotFoundError(
                f"Flickr30k image embeddings not found: {img_emb_path}"
            )
        if not txt_emb_path.exists():
            raise FileNotFoundError(
                f"Flickr30k text embeddings not found: {txt_emb_path}"
            )

        self.img_embs: torch.Tensor = torch.load(img_emb_path, weights_only=True)
        self.txt_embs: torch.Tensor = torch.load(txt_emb_path, weights_only=True)

        if self.img_embs.shape[0] != self.txt_embs.shape[0]:
            raise ValueError(
                f"Pair count mismatch: {self.img_embs.shape[0]} images vs "
                f"{self.txt_embs.shape[0]} texts"
            )

        self.num_samples: int = self.img_embs.shape[0]
        logger.info(
            "Flickr30k: loaded %d pairs — img %s, txt %s",
            self.num_samples,
            tuple(self.img_embs.shape),
            tuple(self.txt_embs.shape),
        )

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full embedding matrices.

        Returns:
            img_embs: (N, dim_img) tensor.
            txt_embs: (N, dim_txt) tensor.
        """
        return self.img_embs, self.txt_embs


class ImageNetEmbeddings:
    """ImageNet validation embeddings for zero-shot classification.

    Loads image embeddings, per-image ground-truth labels, and class-name
    text embeddings (one per class, generated from ``"a photo of a {class}"``
    prompts by ``extract_embeddings.py``).

    Args:
        img_emb_path: Path to ``imagenet_val_img.pt`` — shape ``(N, dim_img)``.
        txt_emb_path: Path to ``imagenet_val_txt.pt`` — shape ``(C, dim_txt)``
            where C = 1000.
        labels_path: Path to ``imagenet_val_labels.pt`` — shape ``(N,)``
            with integer class indices in ``[0, C)``.

    Attributes:
        img_embs: (N, dim_img) image embedding tensor.
        txt_embs: (C, dim_txt) class text embedding tensor.
        labels: (N,) integer label tensor.
        num_images: Number of validation images.
        num_classes: Number of classes (should be 1000).
    """

    def __init__(
        self,
        img_emb_path: str | Path,
        txt_emb_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        img_emb_path = Path(img_emb_path)
        txt_emb_path = Path(txt_emb_path)
        labels_path = Path(labels_path)

        for path, name in [
            (img_emb_path, "image embeddings"),
            (txt_emb_path, "text embeddings"),
            (labels_path, "labels"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"ImageNet {name} not found: {path}"
                )

        self.img_embs: torch.Tensor = torch.load(img_emb_path, weights_only=True)
        self.txt_embs: torch.Tensor = torch.load(txt_emb_path, weights_only=True)
        self.labels: torch.Tensor = torch.load(labels_path, weights_only=True)

        self.num_images: int = self.img_embs.shape[0]
        self.num_classes: int = self.txt_embs.shape[0]

        # --- Validation ---
        if self.labels.shape[0] != self.num_images:
            raise ValueError(
                f"Label count ({self.labels.shape[0]}) != image count "
                f"({self.num_images})"
            )
        if self.labels.max().item() >= self.num_classes:
            raise ValueError(
                f"Max label index ({self.labels.max().item()}) >= number of "
                f"class texts ({self.num_classes})"
            )
        if self.labels.min().item() < 0:
            raise ValueError(
                f"Negative label index found: {self.labels.min().item()}"
            )

        logger.info(
            "ImageNet: %d images, %d classes — img %s, txt %s, labels %s",
            self.num_images,
            self.num_classes,
            tuple(self.img_embs.shape),
            tuple(self.txt_embs.shape),
            tuple(self.labels.shape),
        )

    def get_image_embeddings(self) -> torch.Tensor:
        """Return all image embeddings.

        Returns:
            (N, dim_img) tensor.
        """
        return self.img_embs

    def get_class_embeddings(self) -> torch.Tensor:
        """Return per-class text embeddings.

        Returns:
            (C, dim_txt) tensor where C is the number of classes.
        """
        return self.txt_embs

    def get_labels(self) -> torch.Tensor:
        """Return ground-truth labels.

        Returns:
            (N,) long tensor with values in ``[0, C)``.
        """
        return self.labels
