"""ImageNet zero-shot classification evaluation.

Predicts classes by matching bridged image representations against
bridged class-name text representations (generated from prompts like
``"a photo of a {class}"`` by ``extract_embeddings.py``).

Usage as standalone script:
    python -m src.eval.zeroshot --checkpoint path/to/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.eval._utils import get_model_device, load_model_from_checkpoint

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_zeroshot(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Zero-shot classification via bridge-anchor alignment.

    For each image, the predicted class is the one whose bridged text
    representation has the highest cosine similarity with the bridged
    image representation.

    Because image and class-text embeddings have different counts (N images
    vs C classes), the model is applied separately to each set.  A dummy
    tensor of matching size is passed for the unused modality; only the
    relevant output is kept.

    Args:
        model: Alignment model with
            ``forward(img_emb, txt_emb) -> (b_img, b_txt)``.
        img_embs: (N, dim_img) validation image embeddings.
        txt_embs: (C, dim_txt) per-class text embeddings.
        labels: (N,) ground-truth class indices in ``[0, C)``.
        batch_size: Process images in chunks of this size to limit
            GPU memory.

    Returns:
        Dict with ``top1`` and ``top5`` accuracy as percentages [0, 100].

    Shapes::

        img_embs:       (N, dim_img)    e.g. (50000, 768)
        txt_embs:       (C, dim_txt)    e.g. (1000, 768)
        labels:         (N,)
        b_img:          (N, D)          bridged image reps
        b_txt:          (C, D)          bridged class text reps
        sims:           (N, C)          cosine similarity matrix
    """
    model.eval()
    device = get_model_device(model)

    # --- Bridge class text embeddings (small, do all at once) ---
    b_txt = _bridge_texts(model, txt_embs, device)  # (C, D)

    # --- Bridge image embeddings in batches and classify ---
    n = img_embs.shape[0]
    labels = labels.to(device)
    top1_correct = 0
    top5_correct = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_img = _bridge_images(model, img_embs[start:end], device)  # (B, D)

        # Cosine similarity: (B, C)
        sims = b_img @ b_txt.T

        batch_labels = labels[start:end]  # (B,)

        # Top-1
        preds = sims.argmax(dim=1)  # (B,)
        top1_correct += (preds == batch_labels).sum().item()

        # Top-5
        _, top5_preds = sims.topk(5, dim=1)  # (B, 5)
        top5_correct += (top5_preds == batch_labels.unsqueeze(1)).any(dim=1).sum().item()

    top1_acc = top1_correct / n * 100.0
    top5_acc = top5_correct / n * 100.0

    return {"top1": top1_acc, "top5": top5_acc}


@torch.no_grad()
def per_class_accuracy(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Compute top-1 accuracy per class (useful for analysis).

    Args:
        model: Alignment model.
        img_embs: (N, dim_img) image embeddings.
        txt_embs: (C, dim_txt) class text embeddings.
        labels: (N,) ground-truth class indices.
        batch_size: Chunk size for image bridging.

    Returns:
        (C,) tensor of per-class accuracy in [0, 1].
    """
    model.eval()
    device = get_model_device(model)
    b_txt = _bridge_texts(model, txt_embs, device)

    n = img_embs.shape[0]
    c = txt_embs.shape[0]
    labels = labels.to(device)

    correct = torch.zeros(c, device=device)
    total = torch.zeros(c, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_img = _bridge_images(model, img_embs[start:end], device)
        sims = b_img @ b_txt.T
        preds = sims.argmax(dim=1)
        batch_labels = labels[start:end]

        for cls_idx in range(c):
            mask = batch_labels == cls_idx
            if mask.any():
                total[cls_idx] += mask.sum()
                correct[cls_idx] += (preds[mask] == cls_idx).sum()

    return (correct / total.clamp(min=1)).cpu()


# ---------------------------------------------------------------------------
# Helpers — bridge each modality independently
# ---------------------------------------------------------------------------

def _bridge_images(
    model: torch.nn.Module,
    img_embs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Bridge image embeddings through the model.

    Because the model interface requires both modalities, a zero-tensor
    is passed as the text input; only ``b_img`` is used.

    Returns:
        (B, D) bridged image representations on ``device``.
    """
    img_embs = img_embs.to(device)
    # Determine the text dimension the model expects
    txt_dim = _get_txt_dim(model)
    dummy_txt = torch.zeros(img_embs.shape[0], txt_dim, device=device)
    b_img, _ = model(img_embs, dummy_txt)
    return b_img


def _bridge_texts(
    model: torch.nn.Module,
    txt_embs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Bridge text embeddings through the model.

    A zero-tensor is passed as the image input; only ``b_txt`` is used.

    Returns:
        (C, D) bridged text representations on ``device``.
    """
    txt_embs = txt_embs.to(device)
    img_dim = _get_img_dim(model)
    dummy_img = torch.zeros(txt_embs.shape[0], img_dim, device=device)
    _, b_txt = model(dummy_img, txt_embs)
    return b_txt


def _get_img_dim(model: torch.nn.Module) -> int:
    """Infer image embedding dimension from model attributes."""
    if hasattr(model, "dim_img"):
        return model.dim_img
    # LinearProjection / MLPProjection: first layer input features
    if hasattr(model, "proj"):
        return model.proj.in_features
    if hasattr(model, "mlp"):
        return model.mlp[0].in_features
    raise ValueError("Cannot infer dim_img from model.")


def _get_txt_dim(model: torch.nn.Module) -> int:
    """Infer text embedding dimension from model attributes."""
    if hasattr(model, "dim_txt"):
        return model.dim_txt
    if hasattr(model, "proj"):
        return model.proj.out_features
    if hasattr(model, "mlp"):
        return model.mlp[-1].out_features
    raise ValueError("Cannot infer dim_txt from model.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot ImageNet classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--imagenet-img", type=str,
                        default="data/embeddings/imagenet_val_img.pt",
                        help="Path to ImageNet val image embeddings.")
    parser.add_argument("--imagenet-txt", type=str,
                        default="data/embeddings/imagenet_val_txt.pt",
                        help="Path to ImageNet class text embeddings.")
    parser.add_argument("--imagenet-labels", type=str,
                        default="data/embeddings/imagenet_val_labels.pt",
                        help="Path to ImageNet val labels.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for image bridging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    model, cfg = load_model_from_checkpoint(args.checkpoint)
    logger.info("Model: %s", cfg["model"]["name"])

    from src.data.eval_datasets import ImageNetEmbeddings
    inet = ImageNetEmbeddings(args.imagenet_img, args.imagenet_txt, args.imagenet_labels)

    metrics = evaluate_zeroshot(
        model,
        inet.get_image_embeddings(),
        inet.get_class_embeddings(),
        inet.get_labels(),
        batch_size=args.batch_size,
    )

    logger.info("ImageNet Zero-Shot Results:")
    logger.info("  Top-1 Accuracy: %.2f%%", metrics["top1"])
    logger.info("  Top-5 Accuracy: %.2f%%", metrics["top5"])


if __name__ == "__main__":
    main()
