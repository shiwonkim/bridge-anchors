"""Hard-Routing Multi-Expert BridgeAnchors (HardRoute-BA).

True MoE architecture where each token is hard-assigned to exactly one expert
via a learned router with straight-through estimator. Each expert only processes
tokens routed to it, creating structural specialization pressure.

Architecture:
    img_patches (256, 768) → router → hard_gate (256, G)
    For expert g:
        routed_patches → proj_g → CAP with anchors_g → sub_profile_g (K_g)
    Concat all sub_profiles → L2 norm → InfoNCE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bridge_anchors import BottleneckProjector


class TokenRouter(nn.Module):
    """Learned router that assigns each token to top-k of G experts."""

    def __init__(
        self, dim: int, num_experts: int, temperature: float = 1.0,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.temperature = temperature
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T, D)
            mask: (B, T) optional padding mask, 1=valid 0=pad

        Returns:
            hard_gate: (B, T, G) — top-k assignment with STE gradient
            soft_gate: (B, T, G) — soft routing probabilities (for LB loss)
        """
        logits = self.gate(tokens) / self.temperature  # (B, T, G)
        soft_gate = F.softmax(logits, dim=-1)            # (B, T, G)

        if self.top_k == 1:
            # Top-1: argmax + STE
            hard_idx = soft_gate.argmax(dim=-1)               # (B, T)
            hard_gate = F.one_hot(hard_idx, self.num_experts).float()
        else:
            # Top-k: select top-k experts per token
            _, topk_idx = soft_gate.topk(self.top_k, dim=-1)  # (B, T, k)
            hard_gate = torch.zeros_like(soft_gate)
            hard_gate.scatter_(-1, topk_idx, 1.0)             # (B, T, G) with k ones

        # STE: hard_gate in forward, soft_gate in backward
        hard_gate = hard_gate - soft_gate.detach() + soft_gate

        # Zero out padded tokens
        if mask is not None:
            hard_gate = hard_gate * mask.unsqueeze(-1)
            soft_gate = soft_gate * mask.unsqueeze(-1)

        return hard_gate, soft_gate


class HardRouteBridgeAnchors(nn.Module):
    """Hard-routing MoE BridgeAnchors.

    Each token is hard-assigned to exactly one expert. Each expert has its
    own projector, anchors, and performs CAP only on its assigned tokens.

    Args:
        dim_img: Image encoder output dimension.
        dim_txt: Text encoder output dimension.
        num_experts: Number of experts (G).
        expert_k: Anchors per expert (K_g). Total anchors = G * K_g.
        projector_dim: Bottleneck dimension for BottleneckProjector.
        pool_temperature: Temperature for cross-attention pooling.
        route_temperature: Temperature for router softmax.
        img_input: 'cls' or 'tokens'.
        txt_input: 'cls' or 'tokens'.
    """

    def __init__(
        self,
        dim_img: int = 768,
        dim_txt: int = 768,
        num_experts: int = 4,
        expert_k: int = 128,
        projector_dim: int = 32,
        pool_temperature: float = 0.05,
        route_temperature: float = 1.0,
        top_k_route: int = 1,
        img_input: str = "tokens",
        txt_input: str = "tokens",
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.num_experts = num_experts
        self.expert_k = expert_k
        self.num_anchors = num_experts * expert_k  # for compat with logging
        self.projector_dim = projector_dim
        self.pool_temperature = pool_temperature
        self.route_temperature = route_temperature
        self.top_k_route = top_k_route
        self.img_input = img_input
        self.txt_input = txt_input

        # Routers
        self.router_img = TokenRouter(dim_img, num_experts, route_temperature, top_k=top_k_route)
        self.router_txt = TokenRouter(dim_txt, num_experts, route_temperature, top_k=top_k_route)

        # Per-expert projectors
        self.expert_projs_img = nn.ModuleList([
            BottleneckProjector(dim_img, projector_dim)
            for _ in range(num_experts)
        ])
        self.expert_projs_txt = nn.ModuleList([
            BottleneckProjector(dim_txt, projector_dim)
            for _ in range(num_experts)
        ])

        # Per-expert anchors
        self.expert_anchors_img = nn.ParameterList([
            nn.Parameter(torch.empty(expert_k, dim_img))
            for _ in range(num_experts)
        ])
        self.expert_anchors_txt = nn.ParameterList([
            nn.Parameter(torch.empty(expert_k, dim_txt))
            for _ in range(num_experts)
        ])
        for p in list(self.expert_anchors_img) + list(self.expert_anchors_txt):
            nn.init.normal_(p)
            with torch.no_grad():
                p.data = F.normalize(p.data, dim=-1)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
        return_routing: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Args:
            img_emb: (B, T, dim_img) when img_input='tokens'.
            txt_emb: (B, S, dim_txt) when txt_input='tokens'.
            txt_mask: (B, S) padding mask.
            return_routing: If True, also return soft/hard gate tensors.

        Returns:
            Default: (b_img, b_txt) each (B, G*K_g) L2-normalized.
            With return_routing: (b_img, b_txt, soft_gate_img, soft_gate_txt,
                                  hard_gate_img, hard_gate_txt,
                                  expert_profiles_img, expert_profiles_txt)
        """
        G = self.num_experts

        # Separate CLS from patches/words
        if img_emb.dim() == 3:
            img_patches = img_emb[:, 1:, :]   # (B, T-1, D) — exclude CLS
        else:
            # CLS-only: unsqueeze to treat as single token
            img_patches = img_emb.unsqueeze(1)

        if txt_emb.dim() == 3:
            txt_tokens = txt_emb[:, 1:, :]    # (B, S-1, D)
            if txt_mask is not None:
                txt_mask_rest = txt_mask[:, 1:]
            else:
                txt_mask_rest = None
        else:
            txt_tokens = txt_emb.unsqueeze(1)
            txt_mask_rest = None

        # Route tokens to experts
        hard_gate_img, soft_gate_img = self.router_img(img_patches)
        hard_gate_txt, soft_gate_txt = self.router_txt(txt_tokens, mask=txt_mask_rest)

        # Expert-wise projection + masked CAP
        expert_profiles_img = []
        expert_profiles_txt = []

        for g in range(G):
            route_mask_img = hard_gate_img[:, :, g]  # (B, T-1)
            route_mask_txt = hard_gate_txt[:, :, g]  # (B, S-1)

            # Project
            img_g = self.expert_projs_img[g](img_patches)
            txt_g = self.expert_projs_txt[g](txt_tokens)

            # Anchors
            a_img_g = F.normalize(self.expert_anchors_img[g], dim=-1)
            a_txt_g = F.normalize(self.expert_anchors_txt[g], dim=-1)

            # Image CAP with routing mask
            img_g_n = F.normalize(img_g, dim=-1)
            sim_img_g = img_g_n @ a_img_g.T  # (B, T-1, K_g)
            logits_img_g = sim_img_g / self.pool_temperature
            # STE-compatible masking via log(mask + eps)
            logits_img_g = logits_img_g + torch.log(route_mask_img.unsqueeze(-1) + 1e-8)
            attn_img_g = F.softmax(logits_img_g, dim=1)
            attn_img_g = attn_img_g.nan_to_num(0.0)
            profile_img_g = (attn_img_g * sim_img_g).sum(dim=1)  # (B, K_g)

            # Text CAP with routing + padding mask
            txt_g_n = F.normalize(txt_g, dim=-1)
            sim_txt_g = txt_g_n @ a_txt_g.T
            logits_txt_g = sim_txt_g / self.pool_temperature
            if txt_mask_rest is not None:
                combined_mask = route_mask_txt * txt_mask_rest.float()
            else:
                combined_mask = route_mask_txt
            logits_txt_g = logits_txt_g + torch.log(combined_mask.unsqueeze(-1) + 1e-8)
            attn_txt_g = F.softmax(logits_txt_g, dim=1)
            attn_txt_g = attn_txt_g.nan_to_num(0.0)
            profile_txt_g = (attn_txt_g * sim_txt_g).sum(dim=1)

            expert_profiles_img.append(profile_img_g)
            expert_profiles_txt.append(profile_txt_g)

        # Concat and normalize
        raw_img = torch.cat(expert_profiles_img, dim=-1)  # (B, G*K_g)
        raw_txt = torch.cat(expert_profiles_txt, dim=-1)

        b_img = F.normalize(raw_img, dim=-1)
        b_txt = F.normalize(raw_txt, dim=-1)

        if return_routing:
            return (
                b_img, b_txt,
                soft_gate_img, soft_gate_txt,
                hard_gate_img, hard_gate_txt,
                expert_profiles_img, expert_profiles_txt,
            )
        return b_img, b_txt

    def extra_repr(self) -> str:
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"num_experts={self.num_experts}, expert_k={self.expert_k}, "
            f"projector_dim={self.projector_dim}, "
            f"pool_temperature={self.pool_temperature}, "
            f"route_temperature={self.route_temperature}, "
            f"top_k_route={self.top_k_route}"
        )
