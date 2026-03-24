"""Core Bridge Anchor Aligner model.

Aligns two independently trained encoder spaces using learnable bridge
anchors.  Each embedding is converted to a K-dimensional vector of cosine
similarities to K learnable anchor points, producing representations that
can be directly compared across modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeAnchorAligner(nn.Module):
    """Aligns two encoder spaces via learnable bridge anchors.

    Each embedding is projected to a K-dimensional "distance profile" —
    cosine similarities to K learnable anchor points — producing cross-modal
    comparable representations without any space transformation.

    Args:
        dim_img: Dimension of image encoder output (e.g., 768 for DINOv2 ViT-B).
        dim_txt: Dimension of text encoder output (e.g., 768 for all-mpnet-base-v2).
        num_anchors: Number of anchor points K.
        init_method: ``'random'``, ``'prototype'``, or ``'kmeans'`` for
            data-driven initialisation.
        proto_img: Optional (K, dim_img) tensor of image centroids/prototypes
            for ``'prototype'`` or ``'kmeans'`` init.
        proto_txt: Optional (K, dim_txt) tensor of text centroids/prototypes
            for ``'prototype'`` or ``'kmeans'`` init.
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        num_anchors: int = 32,
        init_method: str = "random",
        proto_img: torch.Tensor | None = None,
        proto_txt: torch.Tensor | None = None,
        top_k: int = 0,
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.num_anchors = num_anchors
        self.init_method = init_method
        self.top_k = top_k  # 0 = disabled (use all anchors)

        # Learnable anchors: (K, dim_img) and (K, dim_txt)
        self.anchors_img = nn.Parameter(torch.empty(num_anchors, dim_img))
        self.anchors_txt = nn.Parameter(torch.empty(num_anchors, dim_txt))

        self._init_anchors(init_method, proto_img, proto_txt)

    def _init_anchors(
        self,
        method: str,
        proto_img: torch.Tensor | None,
        proto_txt: torch.Tensor | None,
    ) -> None:
        """Initialise anchor parameters.

        Args:
            method: ``'random'``, ``'prototype'``, or ``'kmeans'``.
            proto_img: (K, dim_img) image centroids/prototypes
                (required if method='prototype' or 'kmeans').
            proto_txt: (K, dim_txt) text centroids/prototypes
                (required if method='prototype' or 'kmeans').
        """
        if method == "random":
            nn.init.normal_(self.anchors_img)
            nn.init.normal_(self.anchors_txt)
            # L2-normalise so anchors start on the unit sphere
            with torch.no_grad():
                self.anchors_img.data = F.normalize(self.anchors_img.data, dim=-1)
                self.anchors_txt.data = F.normalize(self.anchors_txt.data, dim=-1)
        elif method in ("prototype", "kmeans", "fps"):
            if proto_img is None or proto_txt is None:
                raise ValueError(
                    f"proto_img and proto_txt must be provided for '{method}' init."
                )
            if proto_img.shape != (self.num_anchors, self.dim_img):
                raise ValueError(
                    f"proto_img shape {proto_img.shape} does not match "
                    f"({self.num_anchors}, {self.dim_img})"
                )
            if proto_txt.shape != (self.num_anchors, self.dim_txt):
                raise ValueError(
                    f"proto_txt shape {proto_txt.shape} does not match "
                    f"({self.num_anchors}, {self.dim_txt})"
                )
            with torch.no_grad():
                self.anchors_img.data = F.normalize(proto_img.clone(), dim=-1)
                self.anchors_txt.data = F.normalize(proto_txt.clone(), dim=-1)
        else:
            raise ValueError(
                f"Unknown init_method: {method!r}. "
                f"Use 'random', 'prototype', 'kmeans', or 'fps'."
            )

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        return_raw_sims: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bridged representations for a batch of pairs.

        Args:
            img_emb: (B, dim_img) image embeddings.
            txt_emb: (B, dim_txt) text embeddings.
            return_raw_sims: If True, also return raw cosine similarities
                before L2 normalisation (needed for load-balancing loss).

        Returns:
            If return_raw_sims is False:
                Tuple of (b_img, b_txt), each (B, K) L2-normalised.
            If return_raw_sims is True:
                Tuple of (b_img, b_txt, raw_sim_img, raw_sim_txt), where
                raw_sim_* are (B, K) cosine similarities before normalisation.

        Shapes (example with B=64, dim_img=768, dim_txt=768, K=32)::

            img_emb:      (64, 768)
            txt_emb:      (64, 768)
            anchors_img:  (32, 768)  ← normalised
            anchors_txt:  (32, 768)  ← normalised
            b_img:        (64, 32)   ← cosine sims, then normalised
            b_txt:        (64, 32)   ← cosine sims, then normalised
        """
        # Step 1: L2-normalise anchors (on the forward pass so gradients
        #         flow through the original parameters)
        a_img = F.normalize(self.anchors_img, dim=-1)  # (K, dim_img)
        a_txt = F.normalize(self.anchors_txt, dim=-1)  # (K, dim_txt)

        # Step 2: L2-normalise input embeddings
        img_emb = F.normalize(img_emb, dim=-1)  # (B, dim_img)
        txt_emb = F.normalize(txt_emb, dim=-1)  # (B, dim_txt)

        # Step 3: Cosine similarities to anchors
        raw_img = img_emb @ a_img.T  # (B, K)
        raw_txt = txt_emb @ a_txt.T  # (B, K)

        # Step 3.5: Optional top-k sparse gating (straight-through estimator)
        if self.top_k > 0 and self.top_k < self.num_anchors:
            raw_img = self._sparse_topk(raw_img, self.top_k)
            raw_txt = self._sparse_topk(raw_txt, self.top_k)

        # Step 4: L2-normalise bridged representations
        b_img = F.normalize(raw_img, dim=-1)  # (B, K)
        b_txt = F.normalize(raw_txt, dim=-1)  # (B, K)

        if return_raw_sims:
            return b_img, b_txt, raw_img, raw_txt
        return b_img, b_txt

    @staticmethod
    def _sparse_topk(sim: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k values per row, zero the rest (straight-through).

        Uses straight-through estimator: forward applies the sparse mask,
        backward passes gradients to all anchors as if unmasked.
        """
        _, topk_idx = sim.topk(k, dim=-1)                    # (B, k)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)
        # Straight-through: masked_sim in forward, sim in backward
        masked_sim = sim * mask
        return sim + (masked_sim - sim).detach()

    def extra_repr(self) -> str:
        top_k_str = f", top_k={self.top_k}" if self.top_k > 0 else ""
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"num_anchors={self.num_anchors}, init_method='{self.init_method}'"
            f"{top_k_str}"
        )
