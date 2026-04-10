"""Baseline alignment models for comparison with Bridge Anchors.

All models implement the same interface:
    forward(img_emb, txt_emb, txt_mask=None) -> (b_img, b_txt)

where b_img and b_txt are L2-normalised representations that can be
compared via cosine similarity.

Token-level support: when configured with img_input='tokens' or
txt_input='tokens', models apply their core operation per-token then
mean-pool to get (B, D). Text attention masks are used for masked mean
pooling of variable-length text sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool a 3D tensor along dim=1, optionally using an attention mask.

    Args:
        x: (B, S, D) token-level features.
        mask: (B, S) boolean attention mask, or None (all tokens valid).

    Returns:
        (B, D) mean-pooled features.
    """
    if mask is None:
        return x.mean(dim=1)
    # mask: (B, S) -> (B, S, 1) for broadcasting
    mask_f = mask.unsqueeze(-1).float()  # (B, S, 1)
    summed = (x * mask_f).sum(dim=1)     # (B, D)
    counts = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
    return summed / counts


class LinearProjection(nn.Module):
    """Projects image embeddings into text embedding space via a single linear layer.

    Args:
        dim_img: Dimension of image encoder output.
        dim_txt: Dimension of text encoder output.
        img_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level images.
        txt_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level text.

    Learnable parameters: dim_img * dim_txt (e.g., 768*768 = 590,592).
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        img_input: str = "cls",
        txt_input: str = "cls",
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.img_input = img_input
        self.txt_input = txt_input
        self.proj = nn.Linear(dim_img, dim_txt, bias=False)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project image embeddings and normalise both modalities.

        Args:
            img_emb: (B, dim_img) when img_input='cls', or
                (B, S_img, dim_img) when img_input='tokens'.
            txt_emb: (B, dim_txt) when txt_input='cls', or
                (B, S_txt, dim_txt) when txt_input='tokens'.
            txt_mask: (B, S_txt) boolean attention mask for text tokens.

        Returns:
            Tuple of (proj_img, norm_txt), each (B, dim_txt) L2-normalised.
        """
        # Image: project, pool if token-level
        if self.img_input == "tokens":
            assert img_emb.ndim == 3, (
                f"img_input='tokens' but got {img_emb.ndim}D tensor"
            )
            proj_img = self.proj(img_emb)                    # (B, S, dim_txt)
            proj_img = _masked_mean_pool(proj_img, None)     # (B, dim_txt)
        else:
            assert img_emb.ndim == 2, (
                f"img_input='cls' but got {img_emb.ndim}D tensor"
            )
            proj_img = self.proj(img_emb)                    # (B, dim_txt)
        proj_img = F.normalize(proj_img, dim=-1)

        # Text: just pool if token-level (no projection on text side)
        if self.txt_input == "tokens":
            assert txt_emb.ndim == 3, (
                f"txt_input='tokens' but got {txt_emb.ndim}D tensor"
            )
            norm_txt = _masked_mean_pool(txt_emb, txt_mask)  # (B, dim_txt)
        else:
            assert txt_emb.ndim == 2, (
                f"txt_input='cls' but got {txt_emb.ndim}D tensor"
            )
            norm_txt = txt_emb
        norm_txt = F.normalize(norm_txt, dim=-1)

        return proj_img, norm_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"img_input='{self.img_input}', txt_input='{self.txt_input}'"
        )


class MLPProjection(nn.Module):
    """Two-layer MLP with bottleneck that projects image embeddings into text space.

    Architecture: dim_img -> hidden_dim (ReLU) -> dim_txt

    Args:
        dim_img: Dimension of image encoder output.
        dim_txt: Dimension of text encoder output.
        hidden_dim: Bottleneck dimension.
        img_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level images.
        txt_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level text.

    Learnable parameters: dim_img*hidden + hidden*dim_txt
        (e.g., 768*256 + 256*768 = 393,216).
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        hidden_dim: int = 256,
        img_input: str = "cls",
        txt_input: str = "cls",
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.hidden_dim = hidden_dim
        self.img_input = img_input
        self.txt_input = txt_input
        self.mlp = nn.Sequential(
            nn.Linear(dim_img, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_txt, bias=False),
        )

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project image embeddings through MLP and normalise both modalities.

        Args:
            img_emb: (B, dim_img) when img_input='cls', or
                (B, S_img, dim_img) when img_input='tokens'.
            txt_emb: (B, dim_txt) when txt_input='cls', or
                (B, S_txt, dim_txt) when txt_input='tokens'.
            txt_mask: (B, S_txt) boolean attention mask for text tokens.

        Returns:
            Tuple of (proj_img, norm_txt), each (B, dim_txt) L2-normalised.
        """
        # Image: MLP per-token, pool if token-level
        if self.img_input == "tokens":
            assert img_emb.ndim == 3, (
                f"img_input='tokens' but got {img_emb.ndim}D tensor"
            )
            proj_img = self.mlp(img_emb)                     # (B, S, dim_txt)
            proj_img = _masked_mean_pool(proj_img, None)     # (B, dim_txt)
        else:
            assert img_emb.ndim == 2, (
                f"img_input='cls' but got {img_emb.ndim}D tensor"
            )
            proj_img = self.mlp(img_emb)                     # (B, dim_txt)
        proj_img = F.normalize(proj_img, dim=-1)

        # Text: just pool if token-level
        if self.txt_input == "tokens":
            assert txt_emb.ndim == 3, (
                f"txt_input='tokens' but got {txt_emb.ndim}D tensor"
            )
            norm_txt = _masked_mean_pool(txt_emb, txt_mask)  # (B, dim_txt)
        else:
            assert txt_emb.ndim == 2, (
                f"txt_input='cls' but got {txt_emb.ndim}D tensor"
            )
            norm_txt = txt_emb
        norm_txt = F.normalize(norm_txt, dim=-1)

        return proj_img, norm_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"hidden_dim={self.hidden_dim}, "
            f"img_input='{self.img_input}', txt_input='{self.txt_input}'"
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
        img_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level images.
        txt_input: ``'cls'`` for CLS-only or ``'tokens'`` for token-level text.

    Learnable parameters: 0.
    """

    def __init__(
        self,
        anchors_img: torch.Tensor,
        anchors_txt: torch.Tensor,
        img_input: str = "cls",
        txt_input: str = "cls",
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
        self.img_input = img_input
        self.txt_input = txt_input

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
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fixed relative representations.

        Same cosine-similarity computation as BridgeAnchorAligner, but
        anchors are frozen buffers (no gradient).

        Args:
            img_emb: (B, dim_img) when img_input='cls', or
                (B, S_img, dim_img) when img_input='tokens'.
            txt_emb: (B, dim_txt) when txt_input='cls', or
                (B, S_txt, dim_txt) when txt_input='tokens'.
            txt_mask: (B, S_txt) boolean attention mask for text tokens.

        Returns:
            Tuple of (b_img, b_txt), each (B, K) L2-normalised.
        """
        # Image: per-token anchor sims, then pool if token-level
        if self.img_input == "tokens":
            assert img_emb.ndim == 3, (
                f"img_input='tokens' but got {img_emb.ndim}D tensor"
            )
            img_emb = F.normalize(img_emb, dim=-1)          # (B, S, dim_img)
            b_img = img_emb @ self.anchors_img.T             # (B, S, K)
            b_img = _masked_mean_pool(b_img, None)           # (B, K)
        else:
            assert img_emb.ndim == 2, (
                f"img_input='cls' but got {img_emb.ndim}D tensor"
            )
            img_emb = F.normalize(img_emb, dim=-1)           # (B, dim_img)
            b_img = img_emb @ self.anchors_img.T             # (B, K)
        b_img = F.normalize(b_img, dim=-1)

        # Text: per-token anchor sims, then masked pool if token-level
        if self.txt_input == "tokens":
            assert txt_emb.ndim == 3, (
                f"txt_input='tokens' but got {txt_emb.ndim}D tensor"
            )
            txt_emb = F.normalize(txt_emb, dim=-1)           # (B, S, dim_txt)
            b_txt = txt_emb @ self.anchors_txt.T             # (B, S, K)
            b_txt = _masked_mean_pool(b_txt, txt_mask)       # (B, K)
        else:
            assert txt_emb.ndim == 2, (
                f"txt_input='cls' but got {txt_emb.ndim}D tensor"
            )
            txt_emb = F.normalize(txt_emb, dim=-1)           # (B, dim_txt)
            b_txt = txt_emb @ self.anchors_txt.T             # (B, K)
        b_txt = F.normalize(b_txt, dim=-1)

        return b_img, b_txt

    def extra_repr(self) -> str:
        return (
            f"num_anchors={self.num_anchors}, dim_img={self.dim_img}, "
            f"dim_txt={self.dim_txt}, learnable=False, "
            f"img_input='{self.img_input}', txt_input='{self.txt_input}'"
        )
