"""Compute CKA between image and text anchor Gram matrices from checkpoints."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def gram_cka(anchors_img: torch.Tensor, anchors_txt: torch.Tensor) -> float:
    """Compute CKA between Gram matrices of image and text anchors."""
    a_img = F.normalize(anchors_img, dim=-1)
    a_txt = F.normalize(anchors_txt, dim=-1)
    g_img = a_img @ a_img.T  # (K, K)
    g_txt = a_txt @ a_txt.T  # (K, K)

    # Linear CKA: HSIC(G_img, G_txt) / sqrt(HSIC(G_img, G_img) * HSIC(G_txt, G_txt))
    # For centered Gram matrices
    n = g_img.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n  # centering matrix
    g_img_c = H @ g_img @ H
    g_txt_c = H @ g_txt @ H

    hsic_xy = (g_img_c * g_txt_c).sum()
    hsic_xx = (g_img_c * g_img_c).sum()
    hsic_yy = (g_txt_c * g_txt_c).sum()

    cka = hsic_xy / (hsic_xx * hsic_yy).sqrt().clamp(min=1e-8)
    return cka.item()


def frob_diff(anchors_img: torch.Tensor, anchors_txt: torch.Tensor) -> float:
    """Compute ||G_img - G_txt||_F^2."""
    a_img = F.normalize(anchors_img, dim=-1)
    a_txt = F.normalize(anchors_txt, dim=-1)
    g_img = a_img @ a_img.T
    g_txt = a_txt @ a_txt.T
    return (g_img - g_txt).pow(2).sum().item()


def main():
    ckpt_dir = Path("results/checkpoints")
    runs = [
        ("random, iso=0", "iso_sweep_0"),
        ("random, iso=0.001", "iso_sweep_0.001"),
        ("random, iso=0.01", "iso_sweep_0.01"),
        ("random, iso=0.1", "iso_sweep_0.1"),
        ("random, iso=1.0", "iso_sweep_1.0"),
        ("random, iso=10.0", "iso_sweep_10.0"),
        ("kmeans, iso=0", "iso_kmeans_base"),
    ]

    print(f"{'Condition':<25} {'CKA':>8} {'||G_img-G_txt||_F^2':>20}")
    print("-" * 58)
    for label, name in runs:
        ckpt_path = ckpt_dir / name / "best.pt"
        if not ckpt_path.exists():
            print(f"{label:<25}  NOT FOUND")
            continue
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        sd = ckpt["model_state_dict"]
        a_img = sd["anchors_img"]
        a_txt = sd["anchors_txt"]
        cka = gram_cka(a_img, a_txt)
        frob = frob_diff(a_img, a_txt)
        mr = ckpt["metrics"].get("mean_recall", 0.0)
        print(f"{label:<25} {cka:>8.4f} {frob:>20.2f}   mR={mr:.2f}")


if __name__ == "__main__":
    main()
