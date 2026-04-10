"""Plot training dynamics for anchor isometry experiments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tensorboard.backend.event_processing import event_accumulator


OUT_DIR = Path("experiments/exp_anchor_isometry")
LOG_DIR = Path("results/logs")
CKPT_DIR = Path("results/checkpoints")

RUNS = [
    ("iso_sweep_0", 0.0),
    ("iso_sweep_0.001", 0.001),
    ("iso_sweep_0.01", 0.01),
    ("iso_sweep_0.1", 0.1),
    ("iso_sweep_1.0", 1.0),
    ("iso_sweep_10.0", 10.0),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_scalars(run_name: str, tag: str) -> tuple[list[int], list[float]]:
    """Load scalar events from TensorBoard log dir."""
    ea = event_accumulator.EventAccumulator(str(LOG_DIR / run_name))
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def steps_to_epochs(steps: list[int], steps_per_epoch: int = 13) -> list[float]:
    """Convert global steps to epoch numbers."""
    return [s / steps_per_epoch for s in steps]


def setup_plot():
    """Configure matplotlib for clean publication-style plots."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })


def plot_training_loss():
    """Plot 1: Training loss curves."""
    setup_plot()
    fig, ax = plt.subplots()

    for (run_name, lam), color in zip(RUNS, COLORS):
        steps, values = load_scalars(run_name, "train/loss")
        if not steps:
            continue
        epochs = steps_to_epochs(steps)
        label = f"λ={lam}" if lam > 0 else "λ=0 (baseline)"
        ax.plot(epochs, values, color=color, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (InfoNCE + λ·iso)")
    ax.set_title("Training Loss vs Epoch — Anchor Isometry Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_loss_curves.png")
    plt.close(fig)
    print("Saved training_loss_curves.png")


def plot_val_loss():
    """Plot 4: Validation loss curves."""
    setup_plot()
    fig, ax = plt.subplots()

    for (run_name, lam), color in zip(RUNS, COLORS):
        steps, values = load_scalars(run_name, "val/loss")
        if not steps:
            continue
        epochs = steps_to_epochs(steps)
        label = f"λ={lam}" if lam > 0 else "λ=0 (baseline)"
        ax.plot(epochs, values, color=color, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (InfoNCE only)")
    ax.set_title("Validation Loss vs Epoch — Anchor Isometry Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_loss_curves.png")
    plt.close(fig)
    print("Saved val_loss_curves.png")


def plot_retrieval():
    """Plot 3: Mean recall during training."""
    setup_plot()
    fig, ax = plt.subplots()

    for (run_name, lam), color in zip(RUNS, COLORS):
        steps, values = load_scalars(run_name, "flickr/mean_recall")
        if not steps:
            continue
        epochs = steps_to_epochs(steps)
        label = f"λ={lam}" if lam > 0 else "λ=0 (baseline)"
        ax.plot(epochs, values, color=color, label=label, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flickr30k Mean Recall")
    ax.set_title("Retrieval Performance vs Epoch — Anchor Isometry Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retrieval_during_training.png")
    plt.close(fig)
    print("Saved retrieval_during_training.png")


def plot_isometry_loss():
    """Plot 2: Isometry loss curves.

    Since iso_loss isn't in TensorBoard, compute it from the total training
    loss minus the InfoNCE-only baseline.

    For each iso_lambda>0 run:
        iso_loss ≈ (total_loss - baseline_total_loss) / lambda

    This is approximate (optimizer dynamics differ), so also compute the
    actual ||G_img - G_txt||²_F from checkpoints at the final epoch.
    We'll use the val_loss as a proxy for "pure InfoNCE" and compute
    the isometry component from (train_loss - val_loss_interpolated).

    Actually, the cleanest approach: use val_loss (which is pure InfoNCE
    on held-out data) to show how the *contrastive* objective is affected,
    and compute final iso_loss from checkpoints.

    Let's just show estimated iso component = train_loss - baseline_train_loss
    for each lambda, acknowledging it's approximate.
    """
    setup_plot()
    fig, ax = plt.subplots()

    # Load baseline training loss
    base_steps, base_values = load_scalars("iso_sweep_0", "train/loss")
    base_epochs = steps_to_epochs(base_steps)

    for (run_name, lam), color in zip(RUNS[1:], COLORS[1:]):  # skip lambda=0
        steps, values = load_scalars(run_name, "train/loss")
        if not steps or lam == 0:
            continue
        epochs = steps_to_epochs(steps)
        # Estimated iso component = (total_loss - baseline_loss) / lambda
        # Both have same epochs, so align directly
        n = min(len(values), len(base_values))
        iso_component = [(values[i] - base_values[i]) / lam for i in range(n)]
        label = f"λ={lam}"
        ax.plot(epochs[:n], iso_component, color=color, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated ||G_img − G_txt||²_F")
    ax.set_title("Isometry Loss vs Epoch (estimated from total loss difference)")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "isometry_loss_curves.png")
    plt.close(fig)
    print("Saved isometry_loss_curves.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_loss()
    plot_val_loss()
    plot_retrieval()
    plot_isometry_loss()
    print("\nAll plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
