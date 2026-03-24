"""Baseline alignment models for comparison with Bridge Anchors.

All models implement the same interface:
    forward(img_emb, txt_emb) -> (b_img, b_txt)

where b_img and b_txt are L2-normalised representations that can be
compared via cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    """Projects image embeddings into text embedding space via a single linear layer.

    Args:
        dim_img: Dimension of image encoder output.
        dim_txt: Dimension of text encoder output.

    Learnable parameters: dim_img * dim_txt (e.g., 768*768 = 590,592).
    """

    def __init__(self, dim_img: int, dim_txt: int) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.proj = nn.Linear(dim_img, dim_txt, bias=False)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project image embeddings and normalise both modalities.

        Args:
            img_emb: (B, dim_img) image embeddings.
            txt_emb: (B, dim_txt) text embeddings.

        Returns:
            Tuple of (proj_img, norm_txt), each (B, dim_txt) L2-normalised.
        """
        # Project image embeddings into text space
        proj_img = self.proj(img_emb)         # (B, dim_txt)
        proj_img = F.normalize(proj_img, dim=-1)  # (B, dim_txt)
        norm_txt = F.normalize(txt_emb, dim=-1)   # (B, dim_txt)
        return proj_img, norm_txt

    def extra_repr(self) -> str:
        return f"dim_img={self.dim_img}, dim_txt={self.dim_txt}"


class MLPProjection(nn.Module):
    """Two-layer MLP with bottleneck that projects image embeddings into text space.

    Architecture: dim_img -> hidden_dim (ReLU) -> dim_txt

    Args:
        dim_img: Dimension of image encoder output.
        dim_txt: Dimension of text encoder output.
        hidden_dim: Bottleneck dimension.

    Learnable parameters: dim_img*hidden + hidden*dim_txt
        (e.g., 768*256 + 256*768 = 393,216).
    """

    def __init__(self, dim_img: int, dim_txt: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim_img, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_txt, bias=False),
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project image embeddings through MLP and normalise both modalities.

        Args:
            img_emb: (B, dim_img) image embeddings.
            txt_emb: (B, dim_txt) text embeddings.

        Returns:
            Tuple of (proj_img, norm_txt), each (B, dim_txt) L2-normalised.
        """
        proj_img = self.mlp(img_emb)              # (B, dim_txt)
        proj_img = F.normalize(proj_img, dim=-1)   # (B, dim_txt)
        norm_txt = F.normalize(txt_emb, dim=-1)    # (B, dim_txt)
        return proj_img, norm_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"hidden_dim={self.hidden_dim}"
        )


class FixedRelativeRep(nn.Module):
    """Relative Representations baseline (Moschella et al., ICLR 2023).

    Uses fixed (non-learnable) anchors selected from paired training data.
    Embeddings are converted to cosine-similarity profiles against these
    fixed reference points — same computation as BridgeAnchorAligner but
    with zero learnable parameters.

    Args:
        anchors_img: (K, dim_img) image anchor embeddings from training data.
        anchors_txt: (K, dim_txt) text anchor embeddings from training data.

    Learnable parameters: 0.
    """

    def __init__(
        self,
        anchors_img: torch.Tensor,
        anchors_txt: torch.Tensor,
    ) -> None:
        super().__init__()
        if anchors_img.shape[0] != anchors_txt.shape[0]:
            raise ValueError(
                f"Anchor count mismatch: img has {anchors_img.shape[0]}, "
                f"txt has {anchors_txt.shape[0]}"
            )
        self.num_anchors = anchors_img.shape[0]
        self.dim_img = anchors_img.shape[1]
        self.dim_txt = anchors_txt.shape[1]

        # Registered buffers: saved in state_dict but receive no gradients
        self.register_buffer(
            "anchors_img", F.normalize(anchors_img.clone(), dim=-1)
        )  # (K, dim_img)
        self.register_buffer(
            "anchors_txt", F.normalize(anchors_txt.clone(), dim=-1)
        )  # (K, dim_txt)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fixed relative representations.

        Same cosine-similarity computation as BridgeAnchorAligner, but
        anchors are frozen buffers (no gradient).

        Args:
            img_emb: (B, dim_img) image embeddings.
            txt_emb: (B, dim_txt) text embeddings.

        Returns:
            Tuple of (b_img, b_txt), each (B, K) L2-normalised.
        """
        # L2-normalise inputs
        img_emb = F.normalize(img_emb, dim=-1)  # (B, dim_img)
        txt_emb = F.normalize(txt_emb, dim=-1)  # (B, dim_txt)

        # Cosine similarities to fixed anchors
        b_img = img_emb @ self.anchors_img.T  # (B, K)
        b_txt = txt_emb @ self.anchors_txt.T  # (B, K)

        # L2-normalise bridged representations
        b_img = F.normalize(b_img, dim=-1)  # (B, K)
        b_txt = F.normalize(b_txt, dim=-1)  # (B, K)

        return b_img, b_txt

    def extra_repr(self) -> str:
        return (
            f"num_anchors={self.num_anchors}, dim_img={self.dim_img}, "
            f"dim_txt={self.dim_txt}, learnable=False"
        )
