"""Evaluate dual CLS + CA models with three evaluation strategies.

For each checkpoint, computes:
1. mR using CLS profiles only
2. mR using CA profiles only
3. mR using combined profiles: L2_norm(b_cls + b_ca)
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.eval.retrieval import _get_gt_ranks
from src.models.bridge_anchors import BridgeAnchorAligner

TOKEN_DIR = Path("data/embeddings/all_tokens")
CLS_DIR = Path("data/embeddings/cls")


def load_model(ckpt_path: str, ca_exclude_cls: bool = False) -> BridgeAnchorAligner:
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = BridgeAnchorAligner(
        dim_img=cfg.get("model", {}).get("dim_img", 768),
        dim_txt=cfg.get("model", {}).get("dim_txt", 768),
        num_anchors=cfg.get("model", {}).get("num_anchors", 128),
        token_pool="cross_attn",
        pool_temperature=0.05,
        img_input="tokens",
        txt_input="cls",
        ca_exclude_cls=ca_exclude_cls,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_recalls(b_img: torch.Tensor, b_txt: torch.Tensor) -> dict[str, float]:
    """Compute retrieval recalls from bridged representations."""
    sims = b_img @ b_txt.T
    n = sims.shape[0]
    gt = torch.arange(n)
    metrics = {}
    i2t_pos = _get_gt_ranks(sims, gt)
    t2i_pos = _get_gt_ranks(sims.T, gt)
    for k in (1, 5, 10):
        metrics[f"i2t_r{k}"] = (i2t_pos < k).float().mean().item() * 100.0
        metrics[f"t2i_r{k}"] = (t2i_pos < k).float().mean().item() * 100.0
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
    return metrics


@torch.no_grad()
def evaluate_triple(
    model: BridgeAnchorAligner,
    flickr_img: torch.Tensor,
    flickr_txt: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, dict[str, float]]:
    """Evaluate with CLS-only, CA-only, and combined profiles.

    Returns dict with keys 'cls', 'ca', 'combined', each mapping to recall metrics.
    """
    model = model.to(device)
    n = flickr_img.shape[0]

    all_cls_img, all_cls_txt = [], []
    all_ca_img, all_ca_txt = [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img_batch = flickr_img[start:end].to(device)
        txt_batch = flickr_txt[start:end].to(device)

        b_cls_img, b_cls_txt, b_ca_img, b_ca_txt = model(
            img_batch, txt_batch, return_cls_and_ca=True,
        )
        all_cls_img.append(b_cls_img.cpu())
        all_cls_txt.append(b_cls_txt.cpu())
        all_ca_img.append(b_ca_img.cpu())
        all_ca_txt.append(b_ca_txt.cpu())

    cls_img = torch.cat(all_cls_img)
    cls_txt = torch.cat(all_cls_txt)
    ca_img = torch.cat(all_ca_img)
    ca_txt = torch.cat(all_ca_txt)

    # Combined: L2_norm(cls + ca)
    comb_img = F.normalize(cls_img + ca_img, dim=-1)
    comb_txt = F.normalize(cls_txt + ca_txt, dim=-1)

    return {
        "cls": compute_recalls(cls_img, cls_txt),
        "ca": compute_recalls(ca_img, ca_txt),
        "combined": compute_recalls(comb_img, comb_txt),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ca-exclude-cls", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(args.checkpoint, ca_exclude_cls=args.ca_exclude_cls)

    print("Loading Flickr30k data...")
    flickr_img = torch.load(TOKEN_DIR / "flickr30k_test_img.pt", weights_only=True).float()
    flickr_txt = torch.load(CLS_DIR / "flickr30k_test_txt.pt", weights_only=True).float()
    print(f"  img: {flickr_img.shape}, txt: {flickr_txt.shape}")

    print("Evaluating...")
    results = evaluate_triple(model, flickr_img, flickr_txt, device)

    print(f"\n{'Method':<12} {'i2t R@1':>8} {'i2t R@5':>8} {'t2i R@1':>8} {'t2i R@5':>8} {'mR':>8}")
    print("-" * 55)
    for method, metrics in results.items():
        print(
            f"{method:<12} {metrics['i2t_r1']:>8.1f} {metrics['i2t_r5']:>8.1f} "
            f"{metrics['t2i_r1']:>8.1f} {metrics['t2i_r5']:>8.1f} {metrics['mean_recall']:>8.2f}"
        )


if __name__ == "__main__":
    main()
