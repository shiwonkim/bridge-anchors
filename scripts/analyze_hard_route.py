"""Diagnostic analysis comparing HardRoute-BA vs HME soft routing.

Four analyses:
1. Routing visualization — which patches go to which expert (14x14 grid)
2. Per-expert retrieval — each expert's sub-profile evaluated independently
3. Attention pattern comparison — anchor attention maps across models
4. Missed token analysis — Jaccard overlap of top-attending patches

Usage:
    python scripts/analyze_hard_route.py --gpu 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from src.models.hard_route_ba import HardRouteBridgeAnchors
from src.models.bridge_anchors import BridgeAnchorAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = Path("data/embeddings/all_tokens")
CKPT_DIR = Path("results/checkpoints")
OUT_DIR = Path("experiments/exp_hard_route")

FLICKR_IMG = EMB_DIR / "flickr30k_test_img.pt"
FLICKR_TXT_TOK = EMB_DIR / "flickr30k_test_txt_tokens.pt"
FLICKR_TXT_MASK = EMB_DIR / "flickr30k_test_txt_mask.pt"

CKPT_TOP1 = CKPT_DIR / "hr_topk1_cos40_clamp" / "best.pt"
CKPT_TOP2 = CKPT_DIR / "hr_topk2_cos40_clamp" / "best.pt"
CKPT_HME = CKPT_DIR / "hme_g4k128_divlam0.0" / "best.pt"

# 12 diverse sample indices from Flickr30k test set (31783 images)
# Chosen to cover varied scenes: people, animals, landscapes, objects, crowds
SAMPLE_INDICES = [0, 100, 500, 1000, 2000, 5000, 8000, 12000, 16000, 20000, 25000, 30000]


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_hard_route(ckpt_path: Path, device: torch.device) -> HardRouteBridgeAnchors:
    """Load a HardRouteBridgeAnchors model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    # Detect top_k_route from state dict or config
    # All HR models used: G=4, K_g=128, proj_d=32, τ_pool=0.05, τ_route=2.0
    # Detect top_k from checkpoint name
    top_k = 2 if "topk2" in str(ckpt_path) else 1

    model = HardRouteBridgeAnchors(
        dim_img=cfg["model"]["dim_img"],
        dim_txt=cfg["model"]["dim_txt"],
        num_experts=4,
        expert_k=128,
        projector_dim=32,
        pool_temperature=0.05,
        route_temperature=2.0,
        top_k_route=top_k,
        img_input="tokens",
        txt_input="tokens",
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info("Loaded HardRoute (top-%d) from %s (epoch %d)", top_k, ckpt_path.name, ckpt["epoch"])
    return model


def load_hme_soft(ckpt_path: Path, device: torch.device) -> BridgeAnchorAligner:
    """Load an HME soft BridgeAnchorAligner from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    model = BridgeAnchorAligner(
        dim_img=cfg["model"]["dim_img"],
        dim_txt=cfg["model"]["dim_txt"],
        num_anchors=cfg["model"]["num_anchors"],
        token_pool="cross_attn",
        pool_temperature=0.05,
        img_input="tokens",
        txt_input="tokens",
        projector_dim=32,
        num_experts=4,
        expert_k=128,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    logger.info("Loaded HME soft from %s (epoch %d)", ckpt_path.name, ckpt["epoch"])
    return model


def load_flickr_data(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Flickr30k token embeddings.

    Returns:
        img_embs: (N, 257, 768) float16
        txt_embs: (N, 83, 768) float16
        txt_mask: (N, 83) bool
    """
    img = torch.load(FLICKR_IMG, map_location="cpu", weights_only=True)
    txt = torch.load(FLICKR_TXT_TOK, map_location="cpu", weights_only=True)
    mask = torch.load(FLICKR_TXT_MASK, map_location="cpu", weights_only=True)
    logger.info("Flickr30k: img %s, txt %s, mask %s", img.shape, txt.shape, mask.shape)
    return img, txt, mask


# ---------------------------------------------------------------------------
# Analysis 1: Routing Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def analysis_routing_visualization(
    model_top1: HardRouteBridgeAnchors,
    model_top2: HardRouteBridgeAnchors,
    img_embs: torch.Tensor,
    device: torch.device,
) -> None:
    """Visualize expert assignments on 14x14 patch grids for 12 images."""
    logger.info("=== Analysis 1: Routing Visualization ===")
    fig, axes = plt.subplots(len(SAMPLE_INDICES), 2, figsize=(8, 3 * len(SAMPLE_INDICES)))
    fig.suptitle("Expert Assignment per Patch (14×14 grid)", fontsize=14, y=1.01)

    expert_cmap = plt.cm.Set1
    colors_4 = [expert_cmap(i / 4) for i in range(4)]

    for row, idx in enumerate(SAMPLE_INDICES):
        img = img_embs[idx : idx + 1].float().to(device)  # (1, 257, 768)

        for col, (model, name) in enumerate(
            [(model_top1, "Top-1"), (model_top2, "Top-2")]
        ):
            patches = img[:, 1:, :]  # (1, 256, 768) — exclude CLS
            hard_gate, _ = model.router_img(patches)  # (1, 256, G)

            # Get dominant expert per patch
            assignments = hard_gate[0].argmax(dim=-1).cpu().numpy()  # (256,)
            grid = assignments.reshape(16, 16)

            ax = axes[row, col]
            # Create RGB image from expert assignments
            rgb = np.zeros((16, 16, 3))
            for g in range(4):
                mask = grid == g
                rgb[mask] = colors_4[g][:3]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"img {idx}", fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_4[g][:3], label=f"Expert {g}") for g in range(4)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10)

    plt.tight_layout()
    out_path = OUT_DIR / "routing_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved routing visualization to %s", out_path)

    # Also compute routing statistics
    stats_lines = ["# Routing Statistics\n"]
    for name, model in [("Top-1", model_top1), ("Top-2", model_top2)]:
        # Run on all 12 samples
        imgs = img_embs[SAMPLE_INDICES].float().to(device)
        patches = imgs[:, 1:, :]
        hard_gate, soft_gate = model.router_img(patches)
        assignments = hard_gate.argmax(dim=-1)  # (12, 256)
        for g in range(4):
            frac = (assignments == g).float().mean().item()
            stats_lines.append(f"{name} Expert {g}: {frac*100:.1f}% of patches")
        stats_lines.append("")
    logger.info("\n".join(stats_lines))


# ---------------------------------------------------------------------------
# Analysis 2: Per-Expert Retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def analysis_per_expert_retrieval(
    model_top1: HardRouteBridgeAnchors,
    model_top2: HardRouteBridgeAnchors,
    model_hme: BridgeAnchorAligner,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    txt_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Run retrieval using each expert's K_g=128 sub-profile independently."""
    logger.info("=== Analysis 2: Per-Expert Retrieval ===")

    results_lines = [
        "# Per-Expert Retrieval Analysis",
        "",
        "Each expert's 128-dim sub-profile evaluated independently on Flickr30k.",
        "Similar mR across experts = collapse, different = specialization.",
        "",
        "| Model | Expert | i2t R@1 | t2i R@1 | mR |",
        "|-------|--------|---------|---------|------|",
    ]

    models = [
        ("Top-1", model_top1, True),
        ("Top-2", model_top2, True),
        ("HME soft", model_hme, False),
    ]

    batch_size = 512
    N = img_embs.shape[0]

    for model_name, model, is_hard_route in models:
        # Compute full profiles in batches
        all_expert_img = [[] for _ in range(4)]
        all_expert_txt = [[] for _ in range(4)]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            img_batch = img_embs[start:end].float().to(device)
            txt_batch = txt_embs[start:end].float().to(device)
            mask_batch = txt_mask[start:end].to(device)

            if is_hard_route:
                out = model(img_batch, txt_batch, txt_mask=mask_batch, return_routing=True)
                # expert_profiles are at indices 6 and 7
                expert_profiles_img = out[6]  # list of (B, K_g)
                expert_profiles_txt = out[7]  # list of (B, K_g)
            else:
                # HME soft: forward with return_expert_profiles
                out = model(img_batch, txt_batch, txt_mask=mask_batch,
                           return_expert_profiles=True)
                # BridgeAnchorAligner returns (b_img, b_txt, raw_img_parts, raw_txt_parts, attn_img, attn_txt)
                expert_profiles_img = out[2]
                expert_profiles_txt = out[3]

            for g in range(4):
                all_expert_img[g].append(expert_profiles_img[g].cpu())
                all_expert_txt[g].append(expert_profiles_txt[g].cpu())

        # Concat and evaluate per expert
        for g in range(4):
            ep_img = F.normalize(torch.cat(all_expert_img[g], dim=0), dim=-1)
            ep_txt = F.normalize(torch.cat(all_expert_txt[g], dim=0), dim=-1)

            # Compute retrieval metrics
            sim = ep_img @ ep_txt.T  # (N, N)
            N_eval = sim.shape[0]

            # i2t: for each image, rank texts
            ranks_i2t = (sim.argsort(dim=1, descending=True) == torch.arange(N_eval).unsqueeze(1)).nonzero()[:, 1]
            i2t_r1 = (ranks_i2t < 1).float().mean().item() * 100

            # t2i: for each text, rank images
            ranks_t2i = (sim.T.argsort(dim=1, descending=True) == torch.arange(N_eval).unsqueeze(1)).nonzero()[:, 1]
            t2i_r1 = (ranks_t2i < 1).float().mean().item() * 100

            mR = (i2t_r1 + t2i_r1) / 2
            results_lines.append(
                f"| {model_name} | {g} | {i2t_r1:.1f} | {t2i_r1:.1f} | {mR:.1f} |"
            )
            logger.info("%s Expert %d: i2t_r1=%.1f, t2i_r1=%.1f, mR=%.1f",
                        model_name, g, i2t_r1, t2i_r1, mR)

        results_lines.append("")

    out_path = OUT_DIR / "per_expert_retrieval.md"
    out_path.write_text("\n".join(results_lines))
    logger.info("Saved per-expert retrieval to %s", out_path)


# ---------------------------------------------------------------------------
# Analysis 3: Attention Pattern Comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def analysis_attention_comparison(
    model_top1: HardRouteBridgeAnchors,
    model_top2: HardRouteBridgeAnchors,
    model_hme: BridgeAnchorAligner,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    txt_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Compare anchor attention maps across top-1, top-2, and HME soft."""
    logger.info("=== Analysis 3: Attention Pattern Comparison ===")

    # Use first 4 sample images for detailed visualization
    sample_idx = SAMPLE_INDICES[:4]
    n_samples = len(sample_idx)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    fig.suptitle("Mean Anchor Attention (averaged over K_g=128 anchors, Expert 0)",
                 fontsize=14, y=1.01)

    models = [
        ("Top-1", model_top1, True),
        ("Top-2", model_top2, True),
        ("HME soft", model_hme, False),
    ]

    for col, (model_name, model, is_hard_route) in enumerate(models):
        for row, idx in enumerate(sample_idx):
            img = img_embs[idx : idx + 1].float().to(device)
            txt = txt_embs[idx : idx + 1].float().to(device)
            mask = txt_mask[idx : idx + 1].to(device)

            if is_hard_route:
                # Get expert 0's attention pattern
                G = model.num_experts
                img_patches = img[:, 1:, :]  # (1, 256, 768)

                # Project through expert 0
                img_g = model.expert_projs_img[0](img_patches)
                a_img_g = F.normalize(model.expert_anchors_img[0], dim=-1)
                img_g_n = F.normalize(img_g, dim=-1)
                sim = img_g_n @ a_img_g.T  # (1, 256, 128)
                logits = sim / model.pool_temperature
                attn = F.softmax(logits, dim=1)  # (1, 256, 128)

                # Mean attention across anchors
                mean_attn = attn[0].mean(dim=1).cpu().numpy()  # (256,)
            else:
                # HME soft: get expert 0's attention
                img_orig = img
                img_g = model.hme_projs_img[0](img_orig)
                a_img_g = F.normalize(model.expert_anchors_img[0], dim=-1)
                img_g_n = F.normalize(img_g, dim=-1)
                # For HME, img includes CLS — use all tokens for CAP
                sim = img_g_n @ a_img_g.T  # (1, T, 128)
                logits = sim / model.pool_temperature
                attn = F.softmax(logits, dim=1)  # (1, T, 128)

                # Mean attention across anchors, skip CLS token
                mean_attn = attn[0, 1:, :].mean(dim=1).cpu().numpy()  # (256,)

            grid = mean_attn.reshape(16, 16)
            ax = axes[row, col]
            im = ax.imshow(grid, cmap="hot", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(model_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"img {idx}", fontsize=9)

    plt.tight_layout()
    out_path = OUT_DIR / "attention_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention comparison to %s", out_path)


# ---------------------------------------------------------------------------
# Analysis 4: Missed Token Analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analysis_missed_tokens(
    model_top2: HardRouteBridgeAnchors,
    model_hme: BridgeAnchorAligner,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    txt_mask: torch.Tensor,
    device: torch.device,
    n_eval: int = 1000,
    top_k_patches: int = 16,
) -> None:
    """Compute Jaccard overlap of top-attending patches between top-2 and HME soft.

    For each anchor in each expert, identify the top-k patches by attention weight.
    Compare with HME soft's corresponding anchor. Low overlap = hard routing
    forces anchors to use suboptimal tokens.

    Args:
        n_eval: Number of images to evaluate (subset for speed).
        top_k_patches: Number of top patches to compare per anchor.
    """
    logger.info("=== Analysis 4: Missed Token Analysis ===")

    results_lines = [
        "# Missed Token Analysis",
        "",
        f"Comparing top-{top_k_patches} attending patches per anchor between",
        "Top-2 HardRoute and HME soft on first %d Flickr30k images." % n_eval,
        "",
    ]

    jaccard_per_expert = {g: [] for g in range(4)}

    batch_size = 256
    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        img = img_embs[start:end].float().to(device)
        txt = txt_embs[start:end].float().to(device)
        mask = txt_mask[start:end].to(device)
        B = img.shape[0]

        for g in range(4):
            # --- Top-2 attention ---
            img_patches = img[:, 1:, :]
            img_g = model_top2.expert_projs_img[g](img_patches)
            a_img_g = F.normalize(model_top2.expert_anchors_img[g], dim=-1)
            img_g_n = F.normalize(img_g, dim=-1)
            sim_hr = img_g_n @ a_img_g.T  # (B, 256, 128)
            logits_hr = sim_hr / model_top2.pool_temperature
            attn_hr = F.softmax(logits_hr, dim=1)  # (B, 256, 128)

            # --- HME soft attention ---
            img_g_hme = model_hme.hme_projs_img[g](img)
            a_img_g_hme = F.normalize(model_hme.expert_anchors_img[g], dim=-1)
            img_g_hme_n = F.normalize(img_g_hme, dim=-1)
            sim_hme = img_g_hme_n @ a_img_g_hme.T  # (B, T, 128)
            logits_hme = sim_hme / model_hme.pool_temperature
            attn_hme = F.softmax(logits_hme, dim=1)  # (B, T, 128)
            # Skip CLS for HME to match patch indices
            attn_hme = attn_hme[:, 1:, :]  # (B, 256, 128)

            # Get top-k patches per anchor
            _, top_hr = attn_hr.topk(top_k_patches, dim=1)  # (B, top_k, 128)
            _, top_hme = attn_hme.topk(top_k_patches, dim=1)  # (B, top_k, 128)

            # Compute Jaccard per anchor per image
            K_g = attn_hr.shape[2]
            for k in range(K_g):
                hr_sets = top_hr[:, :, k]  # (B, top_k)
                hme_sets = top_hme[:, :, k]  # (B, top_k)
                for b in range(B):
                    hr_s = set(hr_sets[b].cpu().tolist())
                    hme_s = set(hme_sets[b].cpu().tolist())
                    intersection = len(hr_s & hme_s)
                    union = len(hr_s | hme_s)
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_per_expert[g].append(jaccard)

    # Summarize
    results_lines.append("| Expert | Mean Jaccard | Std | Min | Max |")
    results_lines.append("|--------|-------------|-----|-----|-----|")
    overall_jaccards = []
    for g in range(4):
        j = np.array(jaccard_per_expert[g])
        overall_jaccards.extend(j.tolist())
        results_lines.append(
            f"| {g} | {j.mean():.3f} | {j.std():.3f} | {j.min():.3f} | {j.max():.3f} |"
        )
        logger.info("Expert %d: mean Jaccard=%.3f, std=%.3f", g, j.mean(), j.std())

    overall = np.array(overall_jaccards)
    results_lines.append("")
    results_lines.append(f"**Overall mean Jaccard: {overall.mean():.3f}** (1.0 = identical attention, 0.0 = no overlap)")
    results_lines.append("")
    results_lines.append("Interpretation:")
    results_lines.append("- Jaccard < 0.3: Hard routing forces anchors to attend to substantially different patches than HME soft — information restriction is real.")
    results_lines.append("- Jaccard > 0.7: Hard routing doesn't change which patches anchors attend to — the loss must come from gradient quality (STE noise).")
    results_lines.append("- Jaccard ~0.5: Partial overlap — routing restricts some but not all informative patches.")

    out_path = OUT_DIR / "missed_token_analysis.md"
    out_path.write_text("\n".join(results_lines))
    logger.info("Saved missed token analysis to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HardRoute-BA diagnostic analysis")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["all", "routing", "retrieval", "attention", "missed"],
                        help="Which analysis to run (default: all)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info("Loading models...")
    model_top1 = load_hard_route(CKPT_TOP1, device)
    model_top2 = load_hard_route(CKPT_TOP2, device)
    model_hme = load_hme_soft(CKPT_HME, device)

    # Load data
    logger.info("Loading Flickr30k embeddings...")
    img_embs, txt_embs, txt_mask = load_flickr_data(device)

    # Run analyses
    if args.analysis in ("all", "routing"):
        analysis_routing_visualization(model_top1, model_top2, img_embs, device)

    if args.analysis in ("all", "retrieval"):
        analysis_per_expert_retrieval(
            model_top1, model_top2, model_hme,
            img_embs, txt_embs, txt_mask, device,
        )

    if args.analysis in ("all", "attention"):
        analysis_attention_comparison(
            model_top1, model_top2, model_hme,
            img_embs, txt_embs, txt_mask, device,
        )

    if args.analysis in ("all", "missed"):
        analysis_missed_tokens(
            model_top2, model_hme,
            img_embs, txt_embs, txt_mask, device,
        )

    logger.info("All analyses complete.")


if __name__ == "__main__":
    main()
