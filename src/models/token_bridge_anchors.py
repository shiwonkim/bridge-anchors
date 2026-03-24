"""Token-level Bridge Anchor Aligner.

Extends BridgeAnchors to operate on full token sequences (CLS + patch
tokens) from vision encoders, rather than just the CLS token. Text side
remains CLS-level (pooled sentence embedding).

Image input: (B, T, D) where T = 257 (1 CLS + 256 patches for ViT-B/14)
Text input: (B, D) — standard pooled embedding

Aggregation across tokens is done AFTER computing per-token anchor
similarities, allowing the model to attend to spatially-specific features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBridgeAnchorAligner(nn.Module):
    """Token-level Bridge Anchor Aligner.

    Like BridgeAnchorAligner but processes image token sequences.
    For each image token, computes cosine similarity to all K anchors,
    then aggregates across tokens via mean or max pooling.

    Args:
        dim_img: Dimension of image token embeddings (768).
        dim_txt: Dimension of text embeddings (768).
        num_anchors: Number of anchor points K.
        token_pool: Aggregation method across image tokens: ``'mean'`` or ``'max'``.
    """

    def __init__(
        self,
        dim_img: int = 768,
        dim_txt: int = 768,
        num_anchors: int = 128,
        token_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.num_anchors = num_anchors
        self.token_pool = token_pool

        if token_pool not in ("mean", "max"):
            raise ValueError(f"token_pool must be 'mean' or 'max', got {token_pool!r}")

        # Same anchor structure as standard BridgeAnchors
        self.anchors_img = nn.Parameter(torch.empty(num_anchors, dim_img))
        self.anchors_txt = nn.Parameter(torch.empty(num_anchors, dim_txt))

        # Random init, L2-normalized
        nn.init.normal_(self.anchors_img)
        nn.init.normal_(self.anchors_txt)
        with torch.no_grad():
            self.anchors_img.data = F.normalize(self.anchors_img.data, dim=-1)
            self.anchors_txt.data = F.normalize(self.anchors_txt.data, dim=-1)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bridged representations.

        Args:
            img_emb: (B, T, D) token-level image embeddings, or (B, D) CLS-only.
            txt_emb: (B, D) text embeddings.

        Returns:
            Tuple of (b_img, b_txt), each (B, K) L2-normalised.
        """
        a_img = F.normalize(self.anchors_img, dim=-1)  # (K, D)
        a_txt = F.normalize(self.anchors_txt, dim=-1)  # (K, D)

        # --- Image side ---
        if img_emb.dim() == 3:
            # Token-level: (B, T, D)
            img_emb = F.normalize(img_emb, dim=-1)      # (B, T, D)
            # Per-token similarities: (B, T, K)
            sim_img = img_emb @ a_img.T                  # (B, T, K)

            if self.token_pool == "mean":
                raw_img = sim_img.mean(dim=1)             # (B, K)
            else:  # max
                raw_img = sim_img.max(dim=1).values       # (B, K)
        else:
            # CLS-only fallback: (B, D)
            img_emb = F.normalize(img_emb, dim=-1)
            raw_img = img_emb @ a_img.T                   # (B, K)

        # --- Text side (always CLS-level) ---
        txt_emb = F.normalize(txt_emb, dim=-1)            # (B, D)
        raw_txt = txt_emb @ a_txt.T                       # (B, K)

        # L2-normalise bridged representations
        b_img = F.normalize(raw_img, dim=-1)              # (B, K)
        b_txt = F.normalize(raw_txt, dim=-1)              # (B, K)

        return b_img, b_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"num_anchors={self.num_anchors}, token_pool='{self.token_pool}'"
        )
