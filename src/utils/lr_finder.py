"""Learning Rate Finder for Bridge Anchors training.

Log-spaced LR sweep to find optimal learning rate before training.
Based on STRUCTURE's implementation: collects a fixed 5000-sample subset,
then samples random batches from it for each LR step.

Usage:
    from src.utils.lr_finder import find_lr
    suggested_lr = find_lr(model, data_iter, device, batch_size, temperature)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from src.models.losses import info_nce_loss

logger = logging.getLogger(__name__)


def _collect_subset(
    data_iter,
    subset_size: int,
    txt_token_level: bool,
) -> dict[str, torch.Tensor]:
    """Collect a fixed subset of samples from the data iterator.

    Accumulates batches until subset_size samples are collected.
    All tensors are stored on CPU.

    Returns dict with keys: img, txt, txt_tok, txt_mask,
    img_cls_attn, txt_cls_attn (some may be None).
    """
    imgs, txts = [], []
    txt_toks, txt_masks = [], []
    img_attns, txt_attns = [], []
    collected = 0

    for batch in data_iter:
        if txt_token_level:
            img_emb, txt_emb, txt_tok, txt_mask = batch[:4]
            img_cls_attn = batch[4] if len(batch) > 4 else None
            txt_cls_attn = batch[5] if len(batch) > 5 else None
        else:
            img_emb, txt_emb = batch[:2]
            txt_tok, txt_mask = None, None
            img_cls_attn = batch[2] if len(batch) > 2 else None
            txt_cls_attn = batch[3] if len(batch) > 3 else None

        bs = img_emb.shape[0]
        need = subset_size - collected
        if need <= 0:
            break

        sl = slice(0, min(bs, need))
        imgs.append(img_emb[sl].cpu())
        txts.append(txt_emb[sl].cpu())
        if txt_tok is not None:
            txt_toks.append(txt_tok[sl].cpu())
        if txt_mask is not None:
            txt_masks.append(txt_mask[sl].cpu())
        if img_cls_attn is not None:
            img_attns.append(img_cls_attn[sl].cpu())
        if txt_cls_attn is not None:
            txt_attns.append(txt_cls_attn[sl].cpu())

        collected += min(bs, need)
        if collected >= subset_size:
            break

    result = {
        "img": torch.cat(imgs),
        "txt": torch.cat(txts),
        "txt_tok": torch.cat(txt_toks) if txt_toks else None,
        "txt_mask": torch.cat(txt_masks) if txt_masks else None,
        "img_cls_attn": torch.cat(img_attns) if img_attns else None,
        "txt_cls_attn": torch.cat(txt_attns) if txt_attns else None,
    }
    logger.info("LR finder: collected %d samples for subset", result["img"].shape[0])
    return result


def find_lr(
    model: torch.nn.Module,
    data_iter,
    device: torch.device,
    batch_size: int = 1024,
    temperature: float = 0.07,
    num_iter: int = 100,
    subset_size: int = 5000,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    smooth_factor: float = 0.05,
    diverge_factor: float = 5.0,
    lr_divisor: float = 5.0,
    txt_token_level: bool = False,
    save_plot: str | Path | None = None,
) -> float:
    """Run LR range test and return suggested learning rate.

    Collects a fixed ``subset_size`` sample subset (default 5000),
    then sweeps LR from ``start_lr`` to ``end_lr`` on a log scale
    over ``num_iter`` steps, sampling random batches from the subset.

    Args:
        model: The alignment model (state is saved and restored).
        data_iter: Iterable yielding batches.
        device: Target device.
        batch_size: Batch size for each LR step.
        temperature: InfoNCE temperature.
        num_iter: Number of LR steps.
        subset_size: Number of samples to collect for the subset.
        start_lr: Starting (minimum) learning rate.
        end_lr: Ending (maximum) learning rate.
        smooth_factor: Exponential smoothing weight for previous value.
        diverge_factor: Stop if loss exceeds best × this factor.
        lr_divisor: Divide steepest-descent LR by this for the suggestion.
        txt_token_level: If True, batches contain text tokens + masks.
        save_plot: If given, save the LR finder plot to this path.

    Returns:
        Suggested learning rate (float).
    """
    # Collect fixed subset
    subset = _collect_subset(data_iter, subset_size, txt_token_level)
    n = subset["img"].shape[0]
    bs = min(batch_size, n)

    # Save model state
    model_state = deepcopy(model.state_dict())
    model.train()

    optimizer = Adam(model.parameters(), lr=start_lr)
    mult = (end_lr / start_lr) ** (1.0 / num_iter)

    lrs: list[float] = []
    losses: list[float] = []
    best_loss = float("inf")

    try:
        for i in range(num_iter):
            # Sample random batch from subset
            idx = torch.randperm(n)[:bs]

            img_emb = subset["img"][idx].to(device)
            txt_for_model = img_emb  # placeholder, overwritten below

            fwd_kwargs: dict[str, torch.Tensor] = {}

            if txt_token_level and subset["txt_tok"] is not None:
                txt_for_model = subset["txt_tok"][idx].to(device)
                fwd_kwargs["txt_mask"] = subset["txt_mask"][idx].to(device)
            else:
                txt_for_model = subset["txt"][idx].to(device)

            if subset["img_cls_attn"] is not None:
                fwd_kwargs["img_cls_attn"] = subset["img_cls_attn"][idx].to(device)
            if subset["txt_cls_attn"] is not None:
                fwd_kwargs["txt_cls_attn"] = subset["txt_cls_attn"][idx].to(device)

            optimizer.zero_grad()

            b_img, b_txt = model(img_emb, txt_for_model, **fwd_kwargs)

            effective_temp = model.temp if hasattr(model, "temp") else temperature
            loss = info_nce_loss(b_img, b_txt, temperature=effective_temp)
            loss.backward()

            lr = optimizer.param_groups[0]["lr"]
            loss_val = loss.item()
            lrs.append(lr)
            losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val

            if loss_val > diverge_factor * best_loss or not np.isfinite(loss_val):
                logger.info("LR finder: loss diverged at lr=%.2e (iter %d), stopping",
                            lr, i)
                break

            optimizer.step()

            for pg in optimizer.param_groups:
                pg["lr"] *= mult

            if i % 20 == 0:
                logger.info("  LR finder iter %d/%d: lr=%.2e, loss=%.4f",
                            i, num_iter, lr, loss_val)

    except Exception as e:
        logger.error("LR finder failed: %s", e)
        model.load_state_dict(model_state)
        raise

    # Restore model state
    model.load_state_dict(model_state)

    if len(losses) <= 5:
        logger.warning("LR finder: too few data points (%d). Using start_lr × 10.",
                        len(losses))
        return start_lr * 10

    lrs_arr = np.array(lrs)
    losses_arr = np.array(losses)

    # Smooth losses
    smoothed = np.empty_like(losses_arr)
    smoothed[0] = losses_arr[0]
    for i in range(1, len(losses_arr)):
        smoothed[i] = smoothed[i - 1] * smooth_factor + losses_arr[i] * (1 - smooth_factor)

    # Gradient of smoothed loss w.r.t. log10(lr)
    gradients = np.gradient(smoothed, np.log10(lrs_arr))

    # Find steepest descent before the minimum loss point
    min_loss_idx = int(np.argmin(smoothed))
    valid_indices = list(range(1, min(min_loss_idx, len(gradients) - 1)))

    if len(valid_indices) > 0:
        steep_idx = valid_indices[int(np.argmin(gradients[valid_indices]))]
    else:
        steep_idx = max(0, min_loss_idx - 1)

    steep_lr = lrs_arr[steep_idx]
    suggested_lr = max(steep_lr / lr_divisor, start_lr * 10)

    logger.info("LR finder results:")
    logger.info("  Steepest descent at lr=%.2e (loss=%.4f)", steep_lr, smoothed[steep_idx])
    logger.info("  Min loss at lr=%.2e (loss=%.4f)", lrs_arr[min_loss_idx], smoothed[min_loss_idx])
    logger.info("  Suggested lr=%.2e (steep / %.1f)", suggested_lr, lr_divisor)

    if save_plot is not None:
        _plot_lr_finder(
            lrs_arr, losses_arr, smoothed, gradients,
            min_loss_idx, steep_idx, suggested_lr,
            save_path=save_plot,
        )

    return suggested_lr


def _plot_lr_finder(
    lrs: np.ndarray,
    losses: np.ndarray,
    smoothed: np.ndarray,
    gradients: np.ndarray,
    min_loss_idx: int,
    steep_idx: int,
    suggested_lr: float,
    save_path: str | Path,
) -> None:
    """Save LR finder visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]},
    )

    # Top: Loss vs LR
    ax1.plot(lrs, losses, "o-", color="lightblue", alpha=0.6, markersize=3,
             label="Raw loss")
    ax1.plot(lrs, smoothed, "-", color="blue", linewidth=2, label="Smoothed loss")

    ax1.scatter([lrs[min_loss_idx]], [smoothed[min_loss_idx]],
                color="darkgreen", s=100, zorder=5, label="Min loss")
    ax1.scatter([lrs[steep_idx]], [smoothed[steep_idx]],
                color="purple", s=100, zorder=5, label="Steepest descent")
    ax1.axvline(x=suggested_lr, color="red", linestyle="--", linewidth=2,
                label=f"Suggested LR: {suggested_lr:.2e}")
    ax1.axvspan(suggested_lr / 3, suggested_lr * 3, alpha=0.1, color="green",
                label="Recommended range")

    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Learning Rate Finder", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    min_smooth = np.min(smoothed)
    median = np.median(smoothed[smoothed < min_smooth * 2])
    ax1.set_ylim(min_smooth * 0.8, min(np.max(smoothed), median * 2))

    # Bottom: Gradient
    ax2.plot(lrs, gradients, "-", color="crimson", linewidth=2)
    ax2.scatter([lrs[steep_idx]], [gradients[steep_idx]],
                color="purple", s=100, zorder=5)
    ax2.axvline(x=suggested_lr, color="red", linestyle="--", linewidth=2)

    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate", fontsize=12)
    ax2.set_ylabel("Gradient (dL/d(log LR))", fontsize=12)
    ax2.set_title("Rate of Change in Loss", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved LR finder plot to %s", save_path)
