"""Shared Anchor BridgeAnchors (SA-BA) model.

Variant of Bridge Anchors where both modalities are projected into a single
shared space and measured against ONE set of shared anchors. Enforces
cross-modal consistency by construction — the same anchor vector measures
both image and text profiles.

Supports multiple projector designs:
    - "mlp": Linear → GELU → Linear (standard MLP, destructive)
    - "linear": Single Linear layer (no nonlinearity)
    - "residual": BottleneckProjector (residual MLP in original space) → Linear down
    - "residual_shared": BottleneckProjector only (d_shared must equal dim_img/txt)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckProjector(nn.Module):
    """Lightweight bottleneck projector with residual connection.

    Initialized so output ≈ input at start (zero-init up projection).
    Matches the BottleneckProjector in bridge_anchors.py.
    """

    def __init__(self, d_in: int, d_mid: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_in, d_mid)
        self.act = nn.GELU()
        self.up = nn.Linear(d_mid, d_in)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class SharedAnchorAligner(nn.Module):
    """Aligns two encoder spaces via shared anchors in a common projected space.

    Args:
        dim_img: Input image embedding dim.
        dim_txt: Input text embedding dim.
        dim_shared: Shared space dimension (for residual_shared must equal input dims).
        num_anchors: Number of shared anchor points K.
        hidden_dim: Hidden dimension of MLP projectors ("mlp" mode).
        projector_type: Projector design. One of:
            'mlp', 'linear', 'residual', 'residual_shared'.
        bottleneck_dim: Bottleneck dim for BottleneckProjector
            (residual/residual_shared modes).
        img_input: 'cls' or 'tokens'.
        txt_input: 'cls' or 'tokens'.
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        dim_shared: int = 256,
        num_anchors: int = 128,
        hidden_dim: int = 256,
        projector_type: str = "mlp",
        bottleneck_dim: int = 32,
        img_input: str = "cls",
        txt_input: str = "cls",
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.dim_shared = dim_shared
        self.num_anchors = num_anchors
        self.hidden_dim = hidden_dim
        self.projector_type = projector_type
        self.img_input = img_input
        self.txt_input = txt_input

        if projector_type not in ("mlp", "linear", "residual", "residual_shared"):
            raise ValueError(
                f"projector_type must be one of "
                f"'mlp'|'linear'|'residual'|'residual_shared', got {projector_type!r}"
            )

        if projector_type == "mlp":
            self.proj_img = nn.Sequential(
                nn.Linear(dim_img, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim_shared),
            )
            self.proj_txt = nn.Sequential(
                nn.Linear(dim_txt, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim_shared),
            )
        elif projector_type == "linear":
            self.proj_img = nn.Linear(dim_img, dim_shared)
            self.proj_txt = nn.Linear(dim_txt, dim_shared)
        elif projector_type == "residual":
            self.refine_img = BottleneckProjector(dim_img, bottleneck_dim)
            self.refine_txt = BottleneckProjector(dim_txt, bottleneck_dim)
            self.down_img = nn.Linear(dim_img, dim_shared)
            self.down_txt = nn.Linear(dim_txt, dim_shared)
        elif projector_type == "residual_shared":
            # d_shared must match both input dims
            if dim_shared != dim_img or dim_shared != dim_txt:
                raise ValueError(
                    f"residual_shared requires dim_shared ({dim_shared}) == "
                    f"dim_img ({dim_img}) == dim_txt ({dim_txt})"
                )
            self.refine_img = BottleneckProjector(dim_img, bottleneck_dim)
            self.refine_txt = BottleneckProjector(dim_txt, bottleneck_dim)

        # Single set of shared anchors in the shared space
        self.shared_anchors = nn.Parameter(torch.empty(num_anchors, dim_shared))
        nn.init.normal_(self.shared_anchors)
        with torch.no_grad():
            self.shared_anchors.data = F.normalize(self.shared_anchors.data, dim=-1)

    def _project(self, emb: torch.Tensor, is_img: bool) -> torch.Tensor:
        """Project an embedding to the shared space."""
        if self.projector_type in ("mlp", "linear"):
            proj = self.proj_img if is_img else self.proj_txt
            return proj(emb)
        elif self.projector_type == "residual":
            refine = self.refine_img if is_img else self.refine_txt
            down = self.down_img if is_img else self.down_txt
            return down(refine(emb))
        elif self.projector_type == "residual_shared":
            refine = self.refine_img if is_img else self.refine_txt
            return refine(emb)
        raise ValueError(f"unknown projector_type: {self.projector_type}")

    def _to_cls(self, emb: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Reduce a (possibly token-level) embedding to a single CLS vector."""
        if input_mode == "tokens":
            assert emb.dim() == 3, f"tokens mode expects 3D, got {emb.dim()}D"
            return emb[:, 0, :]
        assert emb.dim() == 2, f"cls mode expects 2D, got {emb.dim()}D"
        return emb

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_cls = self._to_cls(img_emb, self.img_input)
        txt_cls = self._to_cls(txt_emb, self.txt_input)

        img_shared = self._project(img_cls, is_img=True)
        txt_shared = self._project(txt_cls, is_img=False)

        img_shared = F.normalize(img_shared, dim=-1)
        txt_shared = F.normalize(txt_shared, dim=-1)
        anchors = F.normalize(self.shared_anchors, dim=-1)  # (K, d_s)

        profile_img = img_shared @ anchors.T  # (B, K)
        profile_txt = txt_shared @ anchors.T  # (B, K)

        b_img = F.normalize(profile_img, dim=-1)
        b_txt = F.normalize(profile_txt, dim=-1)

        return b_img, b_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"dim_shared={self.dim_shared}, num_anchors={self.num_anchors}, "
            f"projector_type='{self.projector_type}', hidden_dim={self.hidden_dim}, "
            f"img_input='{self.img_input}', txt_input='{self.txt_input}'"
        )
