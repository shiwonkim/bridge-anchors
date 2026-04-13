"""Shared utilities for evaluation modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Infer the device a model lives on.

    Checks parameters first, then buffers (for models like
    ``FixedRelativeRep`` that have zero learnable parameters).
    Falls back to CPU.
    """
    params = list(model.parameters())
    if params:
        return params[0].device
    buffers = list(model.buffers())
    if buffers:
        return buffers[0].device
    return torch.device("cpu")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a model from a training checkpoint.

    The checkpoint must contain ``config`` and ``model_state_dict`` keys
    (as produced by ``src.train.save_checkpoint``).

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to place the model on.  Defaults to CUDA if
            available, else CPU.

    Returns:
        (model, config) — the model in eval mode and the training config.
    """
    from src.models.baselines import FixedRelativeRep, LinearProjection, MLPProjection
    from src.models.bridge_anchors import BridgeAnchorAligner

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    model_name = cfg["model"]["name"]

    if model_name == "bridge_anchors":
        # Detect optional features from state_dict
        has_learnable_tau = "log_pool_temperature" in state_dict
        has_cls_attn_betas = "cls_attn_betas_img" in state_dict
        # Detect projector from state_dict
        has_projector = "proj_img.down.weight" in state_dict
        proj_dim = 0
        if has_projector:
            proj_dim = state_dict["proj_img.down.weight"].shape[0]
        # Detect fixed anchors (Hybrid Anchor Pool)
        has_fixed = "fixed_anchors_img" in state_dict
        fixed_anchors = 0
        fixed_proto_img, fixed_proto_txt = None, None
        if has_fixed:
            fixed_proto_img = state_dict["fixed_anchors_img"]
            fixed_proto_txt = state_dict["fixed_anchors_txt"]
            fixed_anchors = fixed_proto_img.shape[0]
        model: torch.nn.Module = BridgeAnchorAligner(
            dim_img=cfg["model"]["dim_img"],
            dim_txt=cfg["model"]["dim_txt"],
            num_anchors=cfg["model"]["num_anchors"],
            learnable_tau=has_learnable_tau,
            cls_attn_prior="additive" if has_cls_attn_betas else "none",
            projector_dim=proj_dim,
            fixed_anchors=fixed_anchors,
            fixed_proto_img=fixed_proto_img,
            fixed_proto_txt=fixed_proto_txt,
        )
    elif model_name == "linear_projection":
        model = LinearProjection(
            dim_img=cfg["model"]["dim_img"],
            dim_txt=cfg["model"]["dim_txt"],
        )
    elif model_name == "mlp_projection":
        model = MLPProjection(
            dim_img=cfg["model"]["dim_img"],
            dim_txt=cfg["model"]["dim_txt"],
            hidden_dim=cfg["baseline"]["mlp_hidden_dim"],
        )
    elif model_name == "fixed_relative_rep":
        # FixedRelativeRep needs anchor shapes from state_dict
        anchors_img = state_dict["anchors_img"]
        anchors_txt = state_dict["anchors_txt"]
        model = FixedRelativeRep(
            anchors_img=anchors_img,
            anchors_txt=anchors_txt,
        )
    else:
        raise ValueError(f"Unknown model name in checkpoint: {model_name!r}")

    # strict=False to handle optional buffers like _group_tau_vec
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("Loaded %s from %s (epoch %d).",
                model_name, checkpoint_path.name, ckpt.get("epoch", -1))
    return model, cfg
