"""Freeze-Align projector baseline (Maniparambil et al., CVPR 2025).

Faithful reproduction of the official Freeze-Align implementation.
Reference: freeze-align/train/models/clip_adjustable_combined_vis_cls.py
Config: freeze-align/train/configs-v2/dinov2-mpnet-wds-combined.yaml

Architecture (dinov2-mpnet combined_vis_cls config):
    Vision side:
        1. Extract CLS token from original embeddings
        2. local_vision_proj (LN+Dropout+PatchProjection) on ALL tokens
        3. Mean pool projected patches (excluding CLS at index 0)
        4. cls_vision_proj (LN+Dropout+PatchProjection) on original CLS
        5. image_feat = local_feat + cls_feat
        6. L2 normalize  (NO global projector for vision)
    Text side (token-level, text_pooling='mean'):
        1. local_text_proj (LN+Dropout+PatchProjection) on ALL tokens (incl CLS)
        2. Attention-masked mean pool (CLS included in pool)
        3. text_proj (ProjectionHead/MLP) on pooled result
        4. L2 normalize
    Text side (CLS-only fallback):
        1. text_proj (ProjectionHead/MLP) on CLS embedding
        2. L2 normalize
    Temperature: learnable nn.Parameter, clamped to [0.001, 0.5]

Key design notes from code audit:
    - PatchProjection: Linear(x) + [Linear(x)->GELU->Linear(x)]
      The linear branch IS the residual path. No explicit x+ skip.
      Activation is GELU (not ReLU despite paper figure legend).
    - Vision has NO global projector (vision_proj = Identity in reference).
    - Text has NO separate CLS projector. CLS is included in the token
      mean pool through local_text_proj, then text_proj (MLP) is applied.
    - Weight sharing: local projectors are shared across tokens (applied
      via matmul), CLS projectors have separate weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchProjection(nn.Module):
    """Token Projector: residual sum of linear and non-linear branches.

    Architecture: output = Linear(x) + [Linear(x) -> GELU -> Linear(x)]
    The linear branch serves as the residual/skip path.

    Args:
        embedding_dim: Input dimension.
        projection_dim: Output dimension.
    """

    def __init__(self, embedding_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.linear_projection = nn.Linear(embedding_dim, projection_dim)
        self.non_linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_projection(x) + self.non_linear_projection(x)


class ProjectionHead(nn.Module):
    """MLP projection head with residual connection and LayerNorm.

    Architecture: projected = Linear(x); h = GELU(projected) -> Linear -> Dropout;
                  output = LayerNorm(h + projected)

    Args:
        embedding_dim: Input dimension.
        projection_dim: Output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class FreezeAlignProjector(nn.Module):
    """Freeze-Align projector for cross-modal alignment.

    Faithfully reproduces the vis_cls combined model from the official
    implementation with the dinov2-mpnet config.

    Supports all 4 input combinations:
      - CLS img + CLS txt
      - Token img + CLS txt
      - CLS img + Token txt
      - Token img + Token txt

    Args:
        dim_img: Image encoder output dimension (768 for DINOv2 ViT-B).
        dim_txt: Text encoder output dimension (768 for MPNet).
        embed_dim: Shared projection dimension (768).
        init_temp: Initial temperature value (0.07).
    """

    def __init__(
        self,
        dim_img: int = 768,
        dim_txt: int = 768,
        embed_dim: int = 768,
        init_temp: float = 0.07,
    ) -> None:
        super().__init__()

        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.embed_dim = embed_dim

        # --- Vision projectors ---
        # local_vision_proj: applied to ALL tokens, then patches mean-pooled
        self.local_vision_proj = nn.Sequential(
            nn.LayerNorm(dim_img),
            nn.Dropout(0.1),
            PatchProjection(dim_img, embed_dim),
        )
        # cls_vision_proj: applied to ORIGINAL CLS token (before local proj)
        self.cls_vision_proj = nn.Sequential(
            nn.LayerNorm(dim_img),
            nn.Dropout(0.1),
            PatchProjection(dim_img, embed_dim),
        )
        # NO global vision projector (vision_proj = Identity in reference)

        # --- Text projectors ---
        # local_text_proj: PatchProjection on ALL text tokens before pooling
        # (config: local_text_projection = 'patch')
        self.local_text_proj = nn.Sequential(
            nn.LayerNorm(dim_txt),
            nn.Dropout(0.1),
            PatchProjection(dim_txt, embed_dim),
        )
        # text_proj: MLP applied AFTER pooling (config: text_projection = 'mlp')
        # Input dim is embed_dim (output of local_text_proj) when token-level,
        # or dim_txt when CLS-only
        self.text_proj = ProjectionHead(
            embedding_dim=embed_dim,
            projection_dim=embed_dim,
            dropout=0.1,
        )
        # NO separate cls_text_proj — CLS is included in the token mean pool

        # --- Learnable temperature ---
        self.temp = nn.Parameter(torch.ones([]) * init_temp)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            img_emb: (B, dim_img) CLS-only or (B, T, dim_img) token-level.
            txt_emb: (B, dim_txt) CLS-only or (B, S, dim_txt) token-level.
            txt_mask: (B, S) attention mask. Required when txt_emb is 3D.

        Returns:
            (image_feat, text_feat), both (B, embed_dim), L2-normalized.
        """
        # Clamp learnable temperature
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # --- Vision ---
        if img_emb.ndim == 2:
            # CLS-only: only cls_vision_proj
            image_feat = self.cls_vision_proj(img_emb)
        elif img_emb.ndim == 3:
            # Token-level (reference order of operations):
            # 1. Extract CLS from ORIGINAL embeddings
            cls_token_orig = img_emb[:, 0, :]
            # 2. Project ALL tokens through local_vision_proj
            all_projected = self.local_vision_proj(img_emb)  # (B, T, embed_dim)
            # 3. Mean pool patches (exclude CLS at index 0)
            local_feat = all_projected[:, 1:, :].mean(dim=1)
            # 4. Project original CLS through cls_vision_proj
            cls_feat = self.cls_vision_proj(cls_token_orig)
            # 5. Combine (both alphas = 1.0)
            image_feat = local_feat + cls_feat
        else:
            raise ValueError(f"img_emb must be 2D or 3D, got {img_emb.ndim}D")

        # NO global vision projector — just normalize
        image_feat = F.normalize(image_feat, dim=-1)

        # --- Text ---
        if txt_emb.ndim == 3:
            # Token-level: local_text_proj -> masked mean pool -> text_proj
            if txt_mask is None:
                raise ValueError("txt_mask required when txt_emb is 3D")
            # 1. Project ALL tokens (CLS included)
            txt_projected = self.local_text_proj(txt_emb)
            # 2. Attention-masked mean pooling (CLS included in pool)
            mask_expanded = txt_mask.unsqueeze(-1).float()
            txt_pooled = (txt_projected * mask_expanded).sum(dim=1)
            txt_pooled = txt_pooled / txt_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            # 3. Apply text_proj MLP on pooled result
            text_feat = self.text_proj(txt_pooled)
        elif txt_emb.ndim == 2:
            # CLS-only: text_proj directly (skip local_text_proj)
            text_feat = self.text_proj(txt_emb)
        else:
            raise ValueError(f"txt_emb must be 2D or 3D, got {txt_emb.ndim}D")

        text_feat = F.normalize(text_feat, dim=-1)

        return image_feat, text_feat
