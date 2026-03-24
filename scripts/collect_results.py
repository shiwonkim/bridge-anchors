"""Collect and summarise results from all experiments.

Scans checkpoint directories for best.pt files, extracts stored metrics
and config, and writes a summary CSV and console table.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --exp exp_a
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"


def collect_experiment(prefix: str) -> list[dict[str, Any]]:
    """Collect results for all runs matching a name prefix.

    Args:
        prefix: Experiment name prefix (e.g. ``'exp_a'``).

    Returns:
        List of dicts, one per run, with config + metrics fields.
    """
    rows: list[dict[str, Any]] = []
    if not CKPT_DIR.exists():
        return rows

    for run_dir in sorted(CKPT_DIR.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        best = run_dir / "best.pt"
        latest = run_dir / "latest.pt"
        ckpt_path = best if best.exists() else (latest if latest.exists() else None)
        if ckpt_path is None:
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        metrics = ckpt.get("metrics", {})

        row: dict[str, Any] = {
            "experiment": run_dir.name,
            "model": cfg.get("model", {}).get("name", "?"),
            "num_anchors": cfg.get("model", {}).get("num_anchors", "N/A"),
            "init_method": cfg.get("model", {}).get("init_method", "N/A"),
            "num_samples": cfg.get("data", {}).get("num_samples", "all"),
            "seed": cfg.get("training", {}).get("seed", "?"),
            "epoch": ckpt.get("epoch", "?"),
            "checkpoint": ckpt_path.name,
        }
        # Flatten metrics
        for key, val in metrics.items():
            row[key] = round(val, 2) if isinstance(val, float) else val

        rows.append(row)

    return rows


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    """Print a formatted table to the console."""
    if not rows:
        print(f"\n{title}: no results found.\n")
        return

    # Determine columns to show
    metric_keys = [k for k in rows[0] if k not in {
        "experiment", "model", "num_anchors", "init_method",
        "num_samples", "seed", "epoch", "checkpoint",
    }]
    headers = ["experiment", "model", "K", "init", "N", "seed"] + metric_keys

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header_line = (
        f"{'experiment':<40s} {'model':<22s} {'K':>4s} {'init':<6s} "
        f"{'N':>7s} {'seed':>4s}"
    )
    for mk in metric_keys:
        header_line += f" {mk:>8s}"
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        line = (
            f"{row['experiment']:<40s} {row['model']:<22s} "
            f"{str(row['num_anchors']):>4s} {str(row['init_method']):<6s} "
            f"{str(row.get('num_samples', 'all')):>7s} {str(row['seed']):>4s}"
        )
        for mk in metric_keys:
            val = row.get(mk, "")
            if isinstance(val, float):
                line += f" {val:>8.2f}"
            else:
                line += f" {str(val):>8s}"
        print(line)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment results.")
    parser.add_argument("--exp", type=str, default=None,
                        choices=["exp_a", "exp_b", "exp_c", "exp_d"],
                        help="Collect only this experiment (default: all).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to save CSV summary.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    experiments = {
        "exp_a": "Experiment A: Main Comparison",
        "exp_b": "Experiment B: K Ablation",
        "exp_c": "Experiment C: Data Efficiency",
        "exp_d": "Experiment D: Fixed vs Learnable",
    }

    if args.exp:
        experiments = {args.exp: experiments[args.exp]}

    all_rows: list[dict[str, Any]] = []
    for prefix, title in experiments.items():
        rows = collect_experiment(prefix)
        print_table(rows, title)
        all_rows.extend(rows)

    if args.csv:
        save_csv(all_rows, Path(args.csv))
    elif all_rows:
        default_csv = PROJECT_ROOT / "results" / "all_results.csv"
        save_csv(all_rows, default_csv)


if __name__ == "__main__":
    main()
