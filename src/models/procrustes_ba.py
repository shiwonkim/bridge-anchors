"""Procrustes Bridge Anchor Aligner.

Aligns image embeddings to text space via a fixed orthogonal Procrustes
rotation, then measures both modalities against SHARED learnable anchors.

Unlike SA-BA (which used MLP projectors and destroyed geometry), orthogonal
rotation preserves all pairwise distances, cosine similarities, and norms
with zero information loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcrustesBA(nn.Module):
    """BridgeAnchors with Procrustes pre-alignment and shared anchors.

    1. Rotate image embeddings into text space via fixed orthogonal R*.
    2. Measure both modalities against SHARED learnable anchors.
    3. L2-normalise profiles, train with InfoNCE.

    Args:
        dim: Embedding dimension (must be same for img and txt).
        num_anchors: Number of shared anchor points K.
        R_matrix: (D, D) pre-computed Procrustes rotation matrix.
            Registered as a buffer (fixed, not learnable).
        img_input: ``'cls'`` or ``'tokens'``.
        txt_input: ``'cls'`` or ``'tokens'``.
    """

    def __init__(
        self,
        dim: int,
        num_anchors: int = 128,
        R_matrix: torch.Tensor | None = None,
        img_input: str = "cls",
        txt_input: str = "cls",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_anchors = num_anchors
        self.img_input = img_input
        self.txt_input = txt_input

        # Procrustes rotation — fixed, NOT learnable
        if R_matrix is not None:
            self.register_buffer("R", R_matrix)
        else:
            self.register_buffer("R", torch.eye(dim))

        # SHARED anchors — one set for both modalities
        self.anchors = nn.Parameter(torch.empty(num_anchors, dim))
        nn.init.normal_(self.anchors)
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, dim=-1)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bridged representations.

        Args:
            img_emb: (B, D) CLS image embeddings.
            txt_emb: (B, D) CLS text embeddings.

        Returns:
            b_img: (B, K) L2-normalised image profiles.
            b_txt: (B, K) L2-normalised text profiles.
        """
        # Step 1: Rotate image into text space (no gradient through R)
        img_aligned = img_emb @ self.R  # (B, D)

        # Step 2: Shared anchor profiles
        a = F.normalize(self.anchors, dim=-1)  # (K, D)

        profile_img = F.normalize(img_aligned, dim=-1) @ a.T  # (B, K)
        profile_txt = F.normalize(txt_emb, dim=-1) @ a.T  # (B, K)

        b_img = F.normalize(profile_img, dim=-1)
        b_txt = F.normalize(profile_txt, dim=-1)

        return b_img, b_txt

    def extra_repr(self) -> str:
        is_identity = torch.allclose(
            self.R, torch.eye(self.dim, device=self.R.device), atol=1e-5,
        )
        r_str = "R=identity" if is_identity else "R=procrustes"
        return (
            f"dim={self.dim}, num_anchors={self.num_anchors}, "
            f"{r_str}, img_input='{self.img_input}', "
            f"txt_input='{self.txt_input}'"
        )
