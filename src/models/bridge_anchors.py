"""Core Bridge Anchor Aligner model.

Aligns two independently trained encoder spaces using learnable bridge
anchors.  Each embedding is converted to a K-dimensional vector of cosine
similarities to K learnable anchor points, producing representations that
can be directly compared across modalities.

Supports both CLS-only (B, D) and token-level (B, T, D) inputs.  The input
mode is configured at init time via ``img_input`` and ``txt_input``.  When
token-level inputs are configured, per-token anchor similarities are
computed and then aggregated via mean, max, or cross-attention pooling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProfileDecoder(nn.Module):
    """Linear decoder: reconstructs original CLS embedding from profile.

    Intentionally a single linear layer — a stronger decoder would let the
    profile be lazy about encoding information.
    """

    def __init__(self, profile_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(profile_dim, output_dim)

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        return self.linear(profile)


class BottleneckProjector(nn.Module):
    """Lightweight bottleneck projector with residual connection.

    Initialized so output = input at start (zero-init up projection).
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


class BridgeAnchorAligner(nn.Module):
    """Aligns two encoder spaces via learnable bridge anchors.

    Each embedding is projected to a K-dimensional "distance profile" —
    cosine similarities to K learnable anchor points — producing cross-modal
    comparable representations without any space transformation.

    Args:
        dim_img: Dimension of image encoder output (e.g., 768 for DINOv2 ViT-B).
        dim_txt: Dimension of text encoder output (e.g., 768 for all-mpnet-base-v2).
        num_anchors: Number of anchor points K.
        init_method: ``'random'``, ``'prototype'``, ``'kmeans'``, or ``'fps'``
            for data-driven initialisation.
        proto_img: Optional (K, dim_img) tensor of image centroids/prototypes
            for ``'prototype'``, ``'kmeans'``, or ``'fps'`` init.
        proto_txt: Optional (K, dim_txt) tensor of text centroids/prototypes
            for ``'prototype'``, ``'kmeans'``, or ``'fps'`` init.
        top_k: If > 0, keep only top-k anchor similarities per sample
            (sparse gating with straight-through estimator).  0 = disabled.
        token_pool: Aggregation method across tokens when input is token-level:
            ``'mean'``, ``'max'``, or ``'cross_attn'``.
        pool_temperature: Temperature for cross-attention softmax.  Only used
            when ``token_pool='cross_attn'``.  Lower = sharper attention.
        learnable_tau: If True, replace the scalar pool_temperature with
            per-anchor learnable temperatures stored in log-space.  Each
            anchor gets its own τ, initialised to ``pool_temperature``.
        cls_attn_prior: How to incorporate encoder CLS attention as a prior
            for cross-attention pooling.  ``'none'`` = no prior (default),
            ``'multiply'`` = add ``beta * log(cls_attn)`` to logits (shared
            beta), ``'additive'`` = per-anchor learnable betas for each
            modality.
        cls_attn_beta: Shared beta for ``'multiply'`` mode.  Also used
            as init value for per-anchor betas in ``'additive'`` mode.
        group_taus: If provided, a list of G temperature values.  K
            anchors are divided into G equal groups, each using a different
            fixed τ.  K must be divisible by G.  Overrides
            ``pool_temperature`` for cross-attention pooling.
        img_input: ``'cls'`` for CLS-only (B, D) or ``'tokens'`` for
            token-level (B, T, D) image inputs.
        txt_input: ``'cls'`` for CLS-only (B, D) or ``'tokens'`` for
            token-level (B, S, D) text inputs.
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
        token_pool: str = "mean",
        pool_temperature: float = 0.1,
        learnable_tau: bool = False,
        cls_attn_prior: str = "none",
        cls_attn_beta: float = 1.0,
        group_taus: list[float] | None = None,
        group_norm: bool = False,
        group_gating: bool = False,
        attn_mask_groups: list[int] | None = None,
        img_input: str = "cls",
        txt_input: str = "cls",
        ca_exclude_cls: bool = False,
        anchor_mediated: bool = False,
        selection_mode: str = "soft",
        am_cls_weight: float = 0.0,
        projector_dim: int = 0,
        stacked_anchors_dim: int = 0,
        profile_proj_dim: int = 0,
        cls_anchors: int = 0,
        num_experts: int = 1,
        expert_soft_mask: bool = False,
        expert_k: int = 0,
        recon_loss: bool = False,
    ) -> None:
        super().__init__()
        self.dim_img = dim_img
        self.dim_txt = dim_txt
        self.num_anchors = num_anchors
        self.init_method = init_method
        self.top_k = top_k  # 0 = disabled (use all anchors)
        self.token_pool = token_pool
        self.pool_temperature = pool_temperature
        self.learnable_tau = learnable_tau
        self.cls_attn_prior = cls_attn_prior
        self.cls_attn_beta = cls_attn_beta
        self.group_taus = group_taus
        self.group_norm = group_norm
        self.group_gating = group_gating
        self.attn_mask_groups = attn_mask_groups
        self.img_input = img_input
        self.txt_input = txt_input
        self.ca_exclude_cls = ca_exclude_cls
        self.anchor_mediated = anchor_mediated
        self.selection_mode = selection_mode
        self.am_cls_weight = am_cls_weight

        if token_pool not in ("mean", "max", "cross_attn"):
            raise ValueError(
                f"token_pool must be 'mean', 'max', or 'cross_attn', "
                f"got {token_pool!r}"
            )
        if img_input not in ("cls", "tokens"):
            raise ValueError(f"img_input must be 'cls' or 'tokens', got {img_input!r}")
        if txt_input not in ("cls", "tokens"):
            raise ValueError(f"txt_input must be 'cls' or 'tokens', got {txt_input!r}")

        # Learnable anchors: (K, dim_img) and (K, dim_txt)
        # In HME mode (expert_k > 0), per-expert anchors are created below
        # and the shared anchors_img/txt are unused dead weight, so skip them.
        self._is_hme = (expert_k > 0 and num_experts > 1)
        if not self._is_hme:
            self.anchors_img = nn.Parameter(torch.empty(num_anchors, dim_img))
            self.anchors_txt = nn.Parameter(torch.empty(num_anchors, dim_txt))
            self._init_anchors(init_method, proto_img, proto_txt)

        # Group temperatures: fixed per-group τ values (not learnable)
        if group_taus is not None:
            G = len(group_taus)
            if num_anchors % G != 0:
                raise ValueError(
                    f"num_anchors ({num_anchors}) must be divisible by "
                    f"len(group_taus) ({G})"
                )
            apg = num_anchors // G
            tau_vec = torch.zeros(num_anchors)
            for g, tau_g in enumerate(group_taus):
                tau_vec[g * apg : (g + 1) * apg] = tau_g
            # Register as buffer (not a parameter — fixed, not learned)
            self.register_buffer("_group_tau_vec", tau_vec)

            if group_gating:
                self.img_gate = nn.Linear(dim_img, G, bias=False)
                self.txt_gate = nn.Linear(dim_txt, G, bias=False)

        # Attention mask groups: each group sees different tokens
        if attn_mask_groups is not None:
            G = len(attn_mask_groups)
            if num_anchors % G != 0:
                raise ValueError(
                    f"num_anchors ({num_anchors}) must be divisible by "
                    f"len(attn_mask_groups) ({G})"
                )

        # Per-anchor learnable temperature (log-space for positivity)
        if learnable_tau:
            self.log_pool_temperature = nn.Parameter(
                torch.full((num_anchors,), math.log(pool_temperature))
            )

        # CLS attention prior for cross-attention pooling
        if cls_attn_prior not in ("none", "multiply", "additive"):
            raise ValueError(
                f"cls_attn_prior must be 'none', 'multiply', or 'additive', "
                f"got {cls_attn_prior!r}"
            )
        if cls_attn_prior == "additive":
            self.cls_attn_betas_img = nn.Parameter(
                torch.full((num_anchors,), cls_attn_beta)
            )
            self.cls_attn_betas_txt = nn.Parameter(
                torch.full((num_anchors,), cls_attn_beta)
            )

        # Optional lightweight projector (Position A: before anchors)
        # In HME mode, per-expert hme_projs_* are used instead — skip shared.
        self.projector_dim = projector_dim
        if projector_dim > 0 and not self._is_hme:
            self.proj_img = BottleneckProjector(dim_img, projector_dim)
            self.proj_txt = BottleneckProjector(dim_txt, projector_dim)

        # Stacked anchors: Layer 2 meta-anchors in profile space
        self.stacked_anchors_dim = stacked_anchors_dim
        if stacked_anchors_dim > 0:
            assert profile_proj_dim == 0, \
                "Cannot use both stacked_anchors_dim and profile_proj_dim"
            K2 = stacked_anchors_dim
            self.meta_anchors_img = nn.Parameter(torch.empty(K2, num_anchors))
            self.meta_anchors_txt = nn.Parameter(torch.empty(K2, num_anchors))
            nn.init.normal_(self.meta_anchors_img, std=0.02)
            nn.init.normal_(self.meta_anchors_txt, std=0.02)

        # Profile projector: residual MLP in profile space
        self.profile_proj_dim = profile_proj_dim
        if profile_proj_dim > 0:
            assert stacked_anchors_dim == 0, \
                "Cannot use both stacked_anchors_dim and profile_proj_dim"
            K = num_anchors
            self.profile_proj_img = nn.Sequential(
                nn.Linear(K, profile_proj_dim),
                nn.GELU(),
                nn.Linear(profile_proj_dim, K),
            )
            self.profile_proj_txt = nn.Sequential(
                nn.Linear(K, profile_proj_dim),
                nn.GELU(),
                nn.Linear(profile_proj_dim, K),
            )
            # Zero-init last layer for residual start
            nn.init.zeros_(self.profile_proj_img[-1].weight)
            nn.init.zeros_(self.profile_proj_img[-1].bias)
            nn.init.zeros_(self.profile_proj_txt[-1].weight)
            nn.init.zeros_(self.profile_proj_txt[-1].bias)

        # --- Step 1: CLS anchors (dual profile) ---
        self.cls_anchors_k = cls_anchors
        if cls_anchors > 0:
            self.cls_anchors_img = nn.Parameter(torch.empty(cls_anchors, dim_img))
            self.cls_anchors_txt = nn.Parameter(torch.empty(cls_anchors, dim_txt))
            nn.init.normal_(self.cls_anchors_img)
            nn.init.normal_(self.cls_anchors_txt)
            with torch.no_grad():
                self.cls_anchors_img.data = F.normalize(self.cls_anchors_img.data, dim=-1)
                self.cls_anchors_txt.data = F.normalize(self.cls_anchors_txt.data, dim=-1)
            if projector_dim > 0:
                self.cls_proj_img = BottleneckProjector(dim_img, projector_dim)
                self.cls_proj_txt = BottleneckProjector(dim_txt, projector_dim)

        # --- Step 2: Multi-expert projectors (old path, shared anchors) ---
        # In HME mode (expert_k > 0), we use hme_projs_* instead.
        self.num_experts = num_experts
        if num_experts > 1 and projector_dim > 0 and not self._is_hme:
            # Expert 0 reuses self.proj_img/proj_txt (already created)
            # Experts 1..G-1 get new projectors
            self.extra_expert_projs_img = nn.ModuleList(
                [BottleneckProjector(dim_img, projector_dim) for _ in range(num_experts - 1)]
            )
            self.extra_expert_projs_txt = nn.ModuleList(
                [BottleneckProjector(dim_txt, projector_dim) for _ in range(num_experts - 1)]
            )

        # --- Step 3: Expert soft masks ---
        self.expert_soft_mask = expert_soft_mask
        if expert_soft_mask and num_experts > 1:
            init_centers = torch.linspace(0.8, 0.2, num_experts)
            self.mask_centers_img = nn.Parameter(init_centers.clone())
            self.mask_widths_img = nn.Parameter(torch.full((num_experts,), 0.2))
            self.mask_centers_txt = nn.Parameter(init_centers.clone())
            self.mask_widths_txt = nn.Parameter(torch.full((num_experts,), 0.2))

        # --- HME: Hierarchical Multi-Expert (each expert has own anchors + proj) ---
        self.expert_k = expert_k
        if expert_k > 0 and num_experts > 1:
            # Own anchor parameters per expert
            self.expert_anchors_img = nn.ParameterList([
                nn.Parameter(torch.empty(expert_k, dim_img)) for _ in range(num_experts)
            ])
            self.expert_anchors_txt = nn.ParameterList([
                nn.Parameter(torch.empty(expert_k, dim_txt)) for _ in range(num_experts)
            ])
            for p in self.expert_anchors_img:
                nn.init.normal_(p)
                with torch.no_grad():
                    p.data = F.normalize(p.data, dim=-1)
            for p in self.expert_anchors_txt:
                nn.init.normal_(p)
                with torch.no_grad():
                    p.data = F.normalize(p.data, dim=-1)
            # Own projectors per expert (separate from shared proj_img/proj_txt)
            if projector_dim > 0:
                self.hme_projs_img = nn.ModuleList([
                    BottleneckProjector(dim_img, projector_dim) for _ in range(num_experts)
                ])
                self.hme_projs_txt = nn.ModuleList([
                    BottleneckProjector(dim_txt, projector_dim) for _ in range(num_experts)
                ])

        # ReconBA: per-expert reconstruction decoders (HME mode only)
        self.recon_loss = recon_loss
        if recon_loss and self._is_hme:
            self.decoders_img = nn.ModuleList([
                ProfileDecoder(expert_k, dim_img) for _ in range(num_experts)
            ])
            self.decoders_txt = nn.ModuleList([
                ProfileDecoder(expert_k, dim_txt) for _ in range(num_experts)
            ])

    def _init_anchors(
        self,
        method: str,
        proto_img: torch.Tensor | None,
        proto_txt: torch.Tensor | None,
    ) -> None:
        """Initialise anchor parameters.

        Args:
            method: ``'random'``, ``'prototype'``, ``'kmeans'``, or ``'fps'``.
            proto_img: (K, dim_img) image centroids/prototypes
                (required if method is not 'random').
            proto_txt: (K, dim_txt) text centroids/prototypes
                (required if method is not 'random').
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

    def _compute_profile(
        self,
        emb: torch.Tensor,
        anchors: torch.Tensor,
        pool: str,
        mask: torch.Tensor | None = None,
        cls_attn: torch.Tensor | None = None,
        cls_attn_betas: torch.Tensor | None = None,
        gate_emb: torch.Tensor | None = None,
        is_img: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute anchor similarity profile from embeddings.

        Args:
            emb: (B, D) for CLS-only or (B, S, D) for token-level inputs.
            anchors: (K, D) L2-normalised anchor parameters.
            pool: ``'cls'``, ``'mean'``, ``'max'``, or ``'cross_attn'``.
                ``'cls'`` is used for 2D inputs (no pooling needed).
            mask: (B, S) attention mask for variable-length text tokens.
                Required when pool != 'cls' and input is text tokens.
            cls_attn: (B, S) encoder CLS attention prior (patches only,
                no CLS token).  Used when ``cls_attn_prior != 'none'``.
            cls_attn_betas: (K,) per-anchor learnable betas for
                ``'additive'`` mode.

        Returns:
            Tuple of (raw_profile, token_sims):
                raw_profile: (B, K) raw cosine similarities (before L2 norm).
                token_sims: (B, S, K) per-token similarities if available,
                    None for CLS inputs.
        """
        if emb.dim() == 2:
            # CLS-only: (B, D)
            emb = F.normalize(emb, dim=-1)
            return emb @ anchors.T, None  # (B, K), None

        # Token-level: (B, S, D)
        emb = F.normalize(emb, dim=-1)       # (B, S, D)
        sim = emb @ anchors.T                 # (B, S, K)

        if pool == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)         # (B, S, 1)
                raw = (sim * mask_expanded).sum(dim=1)     # (B, K)
                raw = raw / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                raw = sim.mean(dim=1)                      # (B, K)
        elif pool == "max":
            if mask is not None:
                sim_masked = sim.masked_fill(
                    ~mask.bool().unsqueeze(-1), float("-inf"),
                )
                raw = sim_masked.max(dim=1).values         # (B, K)
            else:
                raw = sim.max(dim=1).values                # (B, K)
        elif pool == "cross_attn":
            # Precompute CLS log-prior if needed (shared across groups)
            cls_log_prior_padded = None
            if self.cls_attn_prior != "none" and cls_attn is not None:
                S = sim.shape[1]
                P = cls_attn.shape[1]
                if P < S:
                    pad = torch.zeros(cls_attn.shape[0], S - P,
                                      device=cls_attn.device)
                    cls_attn_padded = torch.cat([pad, cls_attn], dim=1)
                else:
                    cls_attn_padded = cls_attn[:, :S]
                cls_log_prior_padded = torch.log(
                    cls_attn_padded + 1e-8,
                ).unsqueeze(-1)  # (B, S, 1)

            if self.attn_mask_groups is not None and cls_attn is not None:
                # --- Attention mask groups: each group sees different tokens ---
                G = len(self.attn_mask_groups)
                K = sim.shape[2]
                S = sim.shape[1]
                apg = K // G

                # Pad cls_attn to match S if needed (CLS token at pos 0)
                P = cls_attn.shape[1]
                if P < S:
                    pad_attn = torch.zeros(cls_attn.shape[0], S - P,
                                           device=cls_attn.device)
                    cls_attn_full = torch.cat([pad_attn, cls_attn], dim=1)
                else:
                    cls_attn_full = cls_attn[:, :S]

                # Zero out padding positions before ranking
                if mask is not None:
                    cls_attn_full = cls_attn_full * mask.float()

                # Count valid (non-padding) tokens per sample
                if mask is not None:
                    n_valid = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
                else:
                    n_valid = torch.full((sim.shape[0], 1), float(S),
                                         device=sim.device)

                # Compute token ranks by CLS attention (0 = highest attention)
                ranks = cls_attn_full.argsort(dim=1, descending=True).argsort(dim=1)

                raw_parts = []
                cumulative_pct = 0

                for g, pct in enumerate(self.attn_mask_groups):
                    s_idx = g * apg
                    e_idx = (g + 1) * apg
                    sim_g = sim[:, :, s_idx:e_idx]
                    logits_g = sim_g / self.pool_temperature

                    if pct < 100:
                        # Per-sample boundaries based on valid token count
                        start_frac = cumulative_pct / 100.0
                        end_frac = (cumulative_pct + pct) / 100.0
                        start_rank = (n_valid * start_frac).long()  # (B, 1)
                        end_rank = (n_valid * end_frac).clamp(min=1).long()  # (B, 1)

                        # (B, S) mask: ranks in [start_rank, end_rank)
                        ranks_exp = ranks.unsqueeze(-1)  # (B, S, 1) for broadcast
                        token_mask = (ranks_exp >= start_rank.unsqueeze(1)) & \
                                     (ranks_exp < end_rank.unsqueeze(1))
                        token_mask = token_mask.squeeze(-1)  # (B, S)

                        logits_g = logits_g.masked_fill(
                            ~token_mask.unsqueeze(-1), float("-inf"),
                        )
                        cumulative_pct += pct

                    # Apply text padding mask
                    if mask is not None:
                        logits_g = logits_g.masked_fill(
                            ~mask.bool().unsqueeze(-1), float("-inf"),
                        )

                    # CLS prior on top of masking
                    if cls_log_prior_padded is not None:
                        if self.cls_attn_prior == "multiply":
                            logits_g = logits_g + self.cls_attn_beta * cls_log_prior_padded
                        elif self.cls_attn_prior == "additive":
                            betas_g = cls_attn_betas[s_idx:e_idx].unsqueeze(0).unsqueeze(0)
                            logits_g = logits_g + betas_g * cls_log_prior_padded

                    attn_g = F.softmax(logits_g, dim=1)
                    # Safety: if all logits are -inf, softmax gives NaN → zero
                    attn_g = attn_g.nan_to_num(0.0)
                    raw_parts.append((attn_g * sim_g).sum(dim=1))

                raw = torch.cat(raw_parts, dim=-1)
                return raw, sim

            if self.group_taus is not None:
                # --- Group-wise pooling with optional norm and gating ---
                G = len(self.group_taus)
                K = sim.shape[2]
                apg = K // G

                # Compute gate weights if gating is enabled
                gate_weights = None
                if self.group_gating and gate_emb is not None:
                    gate_linear = self.img_gate if is_img else self.txt_gate
                    gate_weights = F.softmax(gate_linear(gate_emb), dim=-1)  # (B, G)

                raw_parts = []
                for g, tau_g in enumerate(self.group_taus):
                    s_idx = g * apg
                    e_idx = (g + 1) * apg
                    sim_g = sim[:, :, s_idx:e_idx]       # (B, S, apg)
                    logits_g = sim_g / tau_g

                    # CLS prior (group-specific beta slice)
                    if cls_log_prior_padded is not None:
                        if self.cls_attn_prior == "multiply":
                            logits_g = logits_g + self.cls_attn_beta * cls_log_prior_padded
                        elif self.cls_attn_prior == "additive":
                            betas_g = cls_attn_betas[s_idx:e_idx].unsqueeze(0).unsqueeze(0)
                            logits_g = logits_g + betas_g * cls_log_prior_padded

                    if mask is not None:
                        logits_g = logits_g.masked_fill(
                            ~mask.bool().unsqueeze(-1), float("-inf"),
                        )
                    attn_g = F.softmax(logits_g, dim=1)  # (B, S, apg)
                    raw_g = (attn_g * sim_g).sum(dim=1)   # (B, apg)

                    if self.group_norm:
                        raw_g = F.normalize(raw_g, dim=-1)

                    if gate_weights is not None:
                        raw_g = raw_g * gate_weights[:, g:g+1]  # (B, apg) * (B, 1)

                    raw_parts.append(raw_g)

                raw = torch.cat(raw_parts, dim=-1)       # (B, K)
                return raw, sim

            # --- Standard single-softmax path ---
            if self.learnable_tau:
                tau = self.log_pool_temperature.exp().unsqueeze(0).unsqueeze(0)
            else:
                tau = self.pool_temperature
            logits = sim / tau                               # (B, S, K)

            # Apply CLS attention prior
            if cls_log_prior_padded is not None:
                if self.cls_attn_prior == "multiply":
                    logits = logits + self.cls_attn_beta * cls_log_prior_padded
                elif self.cls_attn_prior == "additive":
                    betas = cls_attn_betas.unsqueeze(0).unsqueeze(0)
                    logits = logits + betas * cls_log_prior_padded

            if mask is not None:
                logits = logits.masked_fill(
                    ~mask.bool().unsqueeze(-1), float("-inf"),
                )
            attn = F.softmax(logits, dim=1)                  # (B, S, K)
            raw = (attn * sim).sum(dim=1)                   # (B, K)
        else:
            raise ValueError(f"Unknown pool mode: {pool!r}")

        return raw, sim

    def _anchor_mediated_forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        a_img: torch.Tensor,
        a_txt: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Anchor-mediated token representation.

        Each anchor k selects a representative token from each modality
        (CLS excluded — patches/words only), then computes per-anchor
        profile similarities.

        Args:
            img_emb: (B, S, D) image tokens (S includes CLS at index 0).
            txt_emb: (B, M, D) text tokens (M includes CLS at index 0).
            a_img: (K, D) L2-normalised image anchors.
            a_txt: (K, D) L2-normalised text anchors.
            txt_mask: (B, M) optional text attention mask.

        Returns:
            p_img: (B, K, K) per-anchor L2-normalised profiles for images.
            p_txt: (B, K, K) per-anchor L2-normalised profiles for texts.
        """
        # Strip CLS token (index 0) — force selection among patches/words
        img_patches = img_emb[:, 1:, :]  # (B, S-1, D)
        txt_words = txt_emb[:, 1:, :]    # (B, M-1, D)
        if txt_mask is not None:
            txt_mask = txt_mask[:, 1:]    # (B, M-1)

        img_n = F.normalize(img_patches, dim=-1)  # (B, S-1, D)
        txt_n = F.normalize(txt_words, dim=-1)    # (B, M-1, D)

        sim_img = img_n @ a_img.T  # (B, S, K)
        sim_txt = txt_n @ a_txt.T  # (B, M, K)
        if txt_mask is not None:
            sim_txt = sim_txt.masked_fill(
                ~txt_mask.bool().unsqueeze(-1), float("-inf"),
            )

        if self.selection_mode == "soft":
            if self.learnable_tau:
                tau = self.log_pool_temperature.exp().unsqueeze(0).unsqueeze(0)
            else:
                tau = self.pool_temperature
            attn_img = F.softmax(sim_img / tau, dim=1)  # (B, S, K)
            attn_txt = F.softmax(sim_txt / tau, dim=1)  # (B, M, K)
            r_img = torch.einsum("bsk,bsd->bkd", attn_img, img_n)  # (B, K, D)
            r_txt = torch.einsum("bmk,bmd->bkd", attn_txt, txt_n)  # (B, K, D)
        else:  # hard
            D = img_n.shape[-1]
            idx_img = sim_img.argmax(dim=1)  # (B, K)
            idx_txt = sim_txt.argmax(dim=1)  # (B, K)
            r_img = torch.gather(
                img_n, 1, idx_img.unsqueeze(-1).expand(-1, -1, D),
            )  # (B, K, D)
            r_txt = torch.gather(
                txt_n, 1, idx_txt.unsqueeze(-1).expand(-1, -1, D),
            )  # (B, K, D)

        r_img_n = F.normalize(r_img, dim=-1)  # (B, K, D)
        r_txt_n = F.normalize(r_txt, dim=-1)  # (B, K, D)

        p_img = torch.einsum("bkd,ld->bkl", r_img_n, a_img)  # (B, K, K)
        p_txt = torch.einsum("bkd,ld->bkl", r_txt_n, a_txt)  # (B, K, K)

        p_img = F.normalize(p_img, dim=-1)  # (B, K, K)
        p_txt = F.normalize(p_txt, dim=-1)  # (B, K, K)

        return p_img, p_txt

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        txt_mask: torch.Tensor | None = None,
        img_cls_attn: torch.Tensor | None = None,
        txt_cls_attn: torch.Tensor | None = None,
        return_raw_sims: bool = False,
        return_token_sims: bool = False,
        return_cls_and_ca: bool = False,
        return_expert_attns: bool = False,
        return_expert_profiles: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ):
        """Compute bridged representations for a batch of pairs.

        Args:
            img_emb: (B, dim_img) when img_input='cls', or
                (B, T, dim_img) when img_input='tokens'.
            txt_emb: (B, dim_txt) when txt_input='cls', or
                (B, S, dim_txt) when txt_input='tokens'.
            txt_mask: (B, S) attention mask for text tokens.  Required when
                txt_input='tokens'.  1 = valid token, 0 = padding.
            return_raw_sims: If True, also return raw cosine similarities
                before L2 normalisation (needed for load-balancing loss).
            return_token_sims: If True, also return per-token similarity
                matrices before pooling.  Only meaningful when both
                img_input='tokens' and txt_input='tokens'.
            return_cls_and_ca: If True, return both CLS-based and
                cross-attention-based profiles for dual-loss training.
                Returns (b_cls_img, b_cls_txt, b_ca_img, b_ca_txt).

        Returns:
            Default: (b_img, b_txt), each (B, K) L2-normalised.
            With return_raw_sims: (b_img, b_txt, raw_sim_img, raw_sim_txt).
            With return_token_sims: (b_img, b_txt, token_sims_img, token_sims_txt).
            With return_cls_and_ca: (b_cls_img, b_cls_txt, b_ca_img, b_ca_txt).
        """
        # Store original embeddings for CLS path before projecting
        img_emb_orig = img_emb
        txt_emb_orig = txt_emb

        # Optional projector: transform embeddings before anchor similarity
        if self.projector_dim > 0 and self.num_experts <= 1:
            # Single-expert: project once (original behavior)
            img_emb = self.proj_img(img_emb)
            txt_emb = self.proj_txt(txt_emb)

        # L2-normalise anchors (on the forward pass so gradients flow
        # through the original parameters). Not used in HME mode.
        if self._is_hme:
            a_img = a_txt = None
        else:
            a_img = F.normalize(self.anchors_img, dim=-1)  # (K, dim_img)
            a_txt = F.normalize(self.anchors_txt, dim=-1)  # (K, dim_txt)

        # --- Anchor-mediated path ---
        if self.anchor_mediated:
            assert img_emb.dim() == 3 and txt_emb.dim() == 3, (
                "anchor_mediated requires token-level inputs (3D tensors)"
            )
            p_img, p_txt = self._anchor_mediated_forward(
                img_emb, txt_emb, a_img, a_txt, txt_mask=txt_mask,
            )

            if self.am_cls_weight > 0:
                cls_img = F.normalize(img_emb[:, 0, :], dim=-1)
                cls_txt = F.normalize(txt_emb[:, 0, :], dim=-1)
                b_cls_img = F.normalize(cls_img @ a_img.T, dim=-1)
                b_cls_txt = F.normalize(cls_txt @ a_txt.T, dim=-1)
                return p_img, p_txt, b_cls_img, b_cls_txt

            return p_img, p_txt

        # --- Dual CLS + Cross-Attention path ---
        if return_cls_and_ca:
            # CLS profiles: use first token (CLS) for images, CLS for text
            if self.img_input == "tokens":
                assert img_emb.dim() == 3
                cls_img_raw, _ = self._compute_profile(
                    img_emb[:, 0, :], a_img, pool="cls",
                )
                ca_img_input = img_emb[:, 1:, :] if self.ca_exclude_cls else img_emb
                ca_img_raw, _ = self._compute_profile(
                    ca_img_input, a_img, pool="cross_attn",
                )
            else:
                assert img_emb.dim() == 2
                cls_img_raw, _ = self._compute_profile(
                    img_emb, a_img, pool="cls",
                )
                ca_img_raw = cls_img_raw  # no tokens to attend over

            if self.txt_input == "tokens":
                assert txt_emb.dim() == 3
                if txt_mask is None:
                    raise ValueError("txt_mask required when txt_input='tokens'")
                cls_txt_raw, _ = self._compute_profile(
                    txt_emb[:, 0, :], a_txt, pool="cls",
                )
                ca_txt_raw, _ = self._compute_profile(
                    txt_emb, a_txt, pool="cross_attn", mask=txt_mask,
                )
            else:
                assert txt_emb.dim() == 2
                cls_txt_raw, _ = self._compute_profile(
                    txt_emb, a_txt, pool="cls",
                )
                ca_txt_raw = cls_txt_raw

            b_cls_img = F.normalize(cls_img_raw, dim=-1)
            b_cls_txt = F.normalize(cls_txt_raw, dim=-1)
            b_ca_img = F.normalize(ca_img_raw, dim=-1)
            b_ca_txt = F.normalize(ca_txt_raw, dim=-1)
            return b_cls_img, b_cls_txt, b_ca_img, b_ca_txt

        # --- Standard single-output path ---
        img_pool = self.token_pool if self.img_input == "tokens" else "cls"
        txt_pool = self.token_pool if self.txt_input == "tokens" else "cls"
        if self.img_input == "tokens":
            assert img_emb.dim() == 3, (
                f"img_input='tokens' but got {img_emb.dim()}D tensor"
            )
        else:
            assert img_emb.dim() == 2, (
                f"img_input='cls' but got {img_emb.dim()}D tensor"
            )
        if self.txt_input == "tokens":
            assert txt_emb.dim() == 3, (
                f"txt_input='tokens' but got {txt_emb.dim()}D tensor"
            )
            if txt_mask is None:
                raise ValueError("txt_mask required when txt_input='tokens'")
        else:
            assert txt_emb.dim() == 2, (
                f"txt_input='cls' but got {txt_emb.dim()}D tensor"
            )

        # Gate embeddings (CLS tokens) for group gating
        img_gate_emb = None
        txt_gate_emb = None
        if self.group_gating and self.img_input == "tokens" and img_emb.dim() == 3:
            img_gate_emb = F.normalize(img_emb[:, 0, :], dim=-1)
        if self.group_gating and self.txt_input == "tokens" and txt_emb.dim() == 3:
            txt_gate_emb = F.normalize(txt_emb[:, 0, :], dim=-1)

        sim_img, sim_txt = None, None  # for return_token_sims
        expert_attn_maps_img: list[torch.Tensor] = []
        expert_attn_maps_txt: list[torch.Tensor] = []
        # Per-expert raw profiles (HME mode only) — exposed via return_expert_profiles
        raw_img_parts: list[torch.Tensor] = []
        raw_txt_parts: list[torch.Tensor] = []

        if self.expert_k > 0 and self.num_experts > 1 and self.projector_dim > 0:
            # --- HME path: each expert has own anchors + projector ---
            G = self.num_experts

            for g in range(G):
                # Project with this expert's own projector
                img_g = self.hme_projs_img[g](img_emb_orig)
                txt_g = self.hme_projs_txt[g](txt_emb_orig)

                # Expert-specific anchors
                a_img_g = F.normalize(self.expert_anchors_img[g], dim=-1)
                a_txt_g = F.normalize(self.expert_anchors_txt[g], dim=-1)

                # Image CAP
                img_g_n = F.normalize(img_g, dim=-1)
                sim_g = img_g_n @ a_img_g.T  # (B, T, K_g)
                logits_g = sim_g / self.pool_temperature
                attn_g = F.softmax(logits_g, dim=1)  # (B, T, K_g)
                profile_img_g = (attn_g * sim_g).sum(dim=1)  # (B, K_g)
                raw_img_parts.append(profile_img_g)
                expert_attn_maps_img.append(attn_g)

                # Text CAP with padding mask
                txt_g_n = F.normalize(txt_g, dim=-1)
                sim_t = txt_g_n @ a_txt_g.T
                logits_t = sim_t / self.pool_temperature
                if txt_mask is not None:
                    logits_t = logits_t.masked_fill(
                        ~txt_mask.bool().unsqueeze(-1), float("-inf"),
                    )
                attn_t = F.softmax(logits_t, dim=1)
                attn_t = attn_t.nan_to_num(0.0)
                profile_txt_g = (attn_t * sim_t).sum(dim=1)
                raw_txt_parts.append(profile_txt_g)
                expert_attn_maps_txt.append(attn_t)

            raw_img = torch.cat(raw_img_parts, dim=-1)
            raw_txt = torch.cat(raw_txt_parts, dim=-1)

        elif self.num_experts > 1 and self.projector_dim > 0:
            # --- Multi-expert path ---
            K_t = self.num_anchors
            G = self.num_experts
            apg = K_t // G

            # Collect all expert projectors (expert 0 = self.proj_img, rest = extra)
            img_projs = [self.proj_img] + list(self.extra_expert_projs_img)
            txt_projs = [self.proj_txt] + list(self.extra_expert_projs_txt)

            # Precompute percentile ranks for soft mask (if needed)
            img_cls_ranks = None
            txt_cls_ranks = None
            if self.expert_soft_mask:
                if img_cls_attn is not None:
                    P = img_cls_attn.shape[1]
                    img_cls_ranks = (
                        img_cls_attn.argsort(dim=1).argsort(dim=1).float()
                        / max(P - 1, 1)
                    )  # (B, P), range [0, 1]
                if txt_cls_attn is not None:
                    P_t = txt_cls_attn.shape[1]
                    txt_cls_ranks = (
                        txt_cls_attn.argsort(dim=1).argsort(dim=1).float()
                        / max(P_t - 1, 1)
                    )  # (B, P_t), range [0, 1]

            raw_img_parts = []
            raw_txt_parts = []

            for g in range(G):
                s_idx, e_idx = g * apg, (g + 1) * apg
                a_img_g = a_img[s_idx:e_idx]
                a_txt_g = a_txt[s_idx:e_idx]

                # Project with this expert's projector
                img_g = img_projs[g](img_emb_orig)
                txt_g = txt_projs[g](txt_emb_orig)

                # Compute CAP profile for this group's anchors
                # Inline simplified CAP: normalize, sim, softmax, weighted sum
                if img_g.dim() == 3:
                    img_g_n = F.normalize(img_g, dim=-1)
                    sim_g = img_g_n @ a_img_g.T  # (B, S, apg)
                    logits_g = sim_g / self.pool_temperature

                    # Expert soft mask (percentile-rank space)
                    if self.expert_soft_mask and img_cls_ranks is not None:
                        c = self.mask_centers_img[g].clamp(0.01, 0.99)
                        w = F.softplus(self.mask_widths_img[g]) + 0.01
                        S = logits_g.shape[1]
                        P = img_cls_ranks.shape[1]
                        if P < S:
                            pad = torch.zeros(img_cls_ranks.shape[0], S - P,
                                              device=img_cls_ranks.device)
                            ranks_padded = torch.cat([pad, img_cls_ranks], dim=1)
                        else:
                            ranks_padded = img_cls_ranks[:, :S]
                        soft_mask = torch.exp(-((ranks_padded - c) ** 2) / (w ** 2))
                        logits_g = logits_g + torch.log(soft_mask + 1e-8).unsqueeze(-1)

                    attn_g = F.softmax(logits_g, dim=1)
                    raw_img_g = (attn_g * sim_g).sum(dim=1)
                else:
                    img_g_n = F.normalize(img_g, dim=-1)
                    raw_img_g = img_g_n @ a_img_g.T

                if txt_g.dim() == 3:
                    txt_g_n = F.normalize(txt_g, dim=-1)
                    sim_t = txt_g_n @ a_txt_g.T
                    logits_t = sim_t / self.pool_temperature

                    if self.expert_soft_mask and txt_cls_ranks is not None:
                        c = self.mask_centers_txt[g].clamp(0.01, 0.99)
                        w = F.softplus(self.mask_widths_txt[g]) + 0.01
                        S_t = logits_t.shape[1]
                        P_t = txt_cls_ranks.shape[1]
                        if P_t < S_t:
                            pad = torch.zeros(txt_cls_ranks.shape[0], S_t - P_t,
                                              device=txt_cls_ranks.device)
                            ranks_padded_t = torch.cat([pad, txt_cls_ranks], dim=1)
                        else:
                            ranks_padded_t = txt_cls_ranks[:, :S_t]
                        soft_mask_t = torch.exp(-((ranks_padded_t - c) ** 2) / (w ** 2))
                        logits_t = logits_t + torch.log(soft_mask_t + 1e-8).unsqueeze(-1)

                    if txt_mask is not None:
                        logits_t = logits_t.masked_fill(
                            ~txt_mask.bool().unsqueeze(-1), float("-inf"),
                        )
                    attn_t = F.softmax(logits_t, dim=1)
                    attn_t = attn_t.nan_to_num(0.0)
                    raw_txt_g = (attn_t * sim_t).sum(dim=1)
                else:
                    txt_g_n = F.normalize(txt_g, dim=-1)
                    raw_txt_g = txt_g_n @ a_txt_g.T

                raw_img_parts.append(raw_img_g)
                raw_txt_parts.append(raw_txt_g)

            raw_img = torch.cat(raw_img_parts, dim=-1)
            raw_txt = torch.cat(raw_txt_parts, dim=-1)

        else:
            # --- Single-expert path (original) ---
            raw_img, sim_img = self._compute_profile(
                img_emb, a_img, pool=img_pool,
                cls_attn=img_cls_attn,
                cls_attn_betas=getattr(self, "cls_attn_betas_img", None),
                gate_emb=img_gate_emb,
                is_img=True,
            )
            raw_txt, sim_txt = self._compute_profile(
                txt_emb, a_txt, pool=txt_pool,
                mask=txt_mask if self.txt_input == "tokens" else None,
                cls_attn=txt_cls_attn,
                cls_attn_betas=getattr(self, "cls_attn_betas_txt", None),
                gate_emb=txt_gate_emb,
                is_img=False,
            )

        # Optional top-k sparse gating (straight-through estimator)
        if self.top_k > 0 and self.top_k < self.num_anchors:
            raw_img = self._sparse_topk(raw_img, self.top_k)
            raw_txt = self._sparse_topk(raw_txt, self.top_k)

        # L2-normalise token profile
        b_img = F.normalize(raw_img, dim=-1)  # (B, K_t)
        b_txt = F.normalize(raw_txt, dim=-1)  # (B, K_t)

        # --- CLS anchors: dual profile (Step 1) ---
        if self.cls_anchors_k > 0 and self.img_input == "tokens" and img_emb_orig.dim() == 3:
            cls_img_raw = F.normalize(img_emb_orig[:, 0, :], dim=-1)
            cls_txt_raw = F.normalize(txt_emb_orig[:, 0, :], dim=-1)
            if hasattr(self, "cls_proj_img"):
                cls_img_raw = self.cls_proj_img(cls_img_raw)
                cls_txt_raw = self.cls_proj_txt(cls_txt_raw)
            ca_img = F.normalize(self.cls_anchors_img, dim=-1)
            ca_txt = F.normalize(self.cls_anchors_txt, dim=-1)
            cls_prof_img = F.normalize(
                F.normalize(cls_img_raw, dim=-1) @ ca_img.T, dim=-1,
            )
            cls_prof_txt = F.normalize(
                F.normalize(cls_txt_raw, dim=-1) @ ca_txt.T, dim=-1,
            )
            b_img = F.normalize(torch.cat([b_img, cls_prof_img], dim=-1), dim=-1)
            b_txt = F.normalize(torch.cat([b_txt, cls_prof_txt], dim=-1), dim=-1)

        # Stacked anchors: Layer 2 in profile space
        if self.stacked_anchors_dim > 0:
            meta_img = F.normalize(self.meta_anchors_img, dim=-1)  # (K2, K1)
            meta_txt = F.normalize(self.meta_anchors_txt, dim=-1)  # (K2, K1)
            b_img = F.normalize(b_img @ meta_img.T, dim=-1)  # (B, K2)
            b_txt = F.normalize(b_txt @ meta_txt.T, dim=-1)  # (B, K2)

        # Profile projector: residual MLP in profile space
        if self.profile_proj_dim > 0:
            b_img = F.normalize(b_img + self.profile_proj_img(b_img), dim=-1)
            b_txt = F.normalize(b_txt + self.profile_proj_txt(b_txt), dim=-1)

        if return_expert_attns:
            return b_img, b_txt, expert_attn_maps_img, expert_attn_maps_txt
        if return_expert_profiles:
            # Returns 6 items: profiles + per-expert attn maps (HME path only)
            return (
                b_img, b_txt,
                raw_img_parts, raw_txt_parts,
                expert_attn_maps_img, expert_attn_maps_txt,
            )
        if return_raw_sims:
            return b_img, b_txt, raw_img, raw_txt
        if return_token_sims:
            sim_img_normed = F.normalize(sim_img, dim=-1) if sim_img is not None else None
            sim_txt_normed = F.normalize(sim_txt, dim=-1) if sim_txt is not None else None
            return b_img, b_txt, sim_img_normed, sim_txt_normed
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
        pool_str = f", token_pool='{self.token_pool}'"
        temp_str = ""
        if self.token_pool == "cross_attn":
            if self.group_taus is not None:
                extras = []
                if self.group_norm:
                    extras.append("norm")
                if self.group_gating:
                    extras.append("gating")
                ext = f"+{'+'  .join(extras)}" if extras else ""
                temp_str = f", group_taus={self.group_taus}{ext}"
            elif self.learnable_tau:
                temp_str = f", learnable_tau=True (init={self.pool_temperature})"
            else:
                temp_str = f", pool_temperature={self.pool_temperature}"
        ca_cls_str = ", ca_exclude_cls=True" if self.ca_exclude_cls else ""
        prior_str = ""
        if self.cls_attn_prior != "none":
            prior_str = f", cls_attn_prior='{self.cls_attn_prior}'"
            if self.cls_attn_prior == "multiply":
                prior_str += f", beta={self.cls_attn_beta}"
        am_str = ""
        if self.anchor_mediated:
            am_str = f", anchor_mediated=True, selection='{self.selection_mode}'"
        proj_str = ""
        if self.projector_dim > 0:
            proj_str = f", projector_dim={self.projector_dim}"
        return (
            f"dim_img={self.dim_img}, dim_txt={self.dim_txt}, "
            f"num_anchors={self.num_anchors}, init_method='{self.init_method}'"
            f"{top_k_str}{pool_str}{temp_str}{prior_str}{ca_cls_str}{am_str}"
            f"{proj_str}, "
            f"img_input='{self.img_input}', txt_input='{self.txt_input}'"
        )
