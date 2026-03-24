"""Loss functions for cross-modal alignment training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    img_features: torch.Tensor,
    txt_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss for cross-modal contrastive alignment.

    Treats the diagonal of the (B, B) similarity matrix as positive pairs
    and all off-diagonal entries as negatives. The final loss is the average
    of the image-to-text and text-to-image cross-entropy losses.

    Args:
        img_features: (B, D) L2-normalised image representations.
        txt_features: (B, D) L2-normalised text representations.
        temperature: Scalar temperature that scales the logits.

    Returns:
        Scalar loss (average of i2t and t2i directions).

    Shapes::

        logits:   (B, B) — scaled cosine similarity matrix
        labels:   (B,)   — [0, 1, ..., B-1], diagonal is positive
        loss_i2t: scalar
        loss_t2i: scalar
        loss:     scalar — (loss_i2t + loss_t2i) / 2
    """
    # (B, B) cosine similarity scaled by temperature
    logits = img_features @ txt_features.T / temperature

    # Positive pairs lie on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


def load_balancing_loss(
    sim_img: torch.Tensor,
    sim_txt: torch.Tensor,
) -> torch.Tensor:
    """Switch Transformer-style load-balancing loss for anchor usage.

    Encourages uniform anchor utilisation by penalising correlation between
    hard assignment frequency and soft routing probability.

    For each modality:
        p_k = fraction of batch where anchor k is the argmax (hard assignment)
        f_k = mean of softmax(sim)_k across batch (soft routing probability)
        L = K * sum_k(p_k * f_k)

    Returns the average across image and text modalities.

    Args:
        sim_img: (B, K) raw cosine similarities (before L2 normalisation)
            from image embeddings to image anchors.
        sim_txt: (B, K) raw cosine similarities from text embeddings to
            text anchors.

    Returns:
        Scalar load-balancing loss.
    """
    def _lb_one_modality(sim: torch.Tensor) -> torch.Tensor:
        B, K = sim.shape
        # p_k: fraction of batch assigned to anchor k (hard)
        assignments = sim.argmax(dim=-1)                    # (B,)
        counts = torch.zeros(K, device=sim.device)
        counts.scatter_add_(0, assignments, torch.ones(B, device=sim.device))
        p = counts / B                                      # (K,)

        # f_k: mean softmax routing probability per anchor (soft)
        routing_probs = F.softmax(sim, dim=-1)              # (B, K)
        f = routing_probs.mean(dim=0)                       # (K,)

        return K * (p * f).sum()

    return (_lb_one_modality(sim_img) + _lb_one_modality(sim_txt)) / 2


def per_anchor_contrastive_loss(
    sim_img: torch.Tensor,
    sim_txt: torch.Tensor,
) -> torch.Tensor:
    """Per-anchor cross-modal consistency loss.

    For each anchor k, computes Pearson correlation between the image
    and text similarity vectors across the batch. For matched pairs,
    if an image is close to anchor k, its paired text should also be
    close to anchor k. The loss is the negative mean correlation.

    Args:
        sim_img: (B, K) raw cosine similarities from images to image anchors.
        sim_txt: (B, K) raw cosine similarities from texts to text anchors.

    Returns:
        Scalar loss (negative mean Pearson correlation across anchors).
    """
    # (B, K) → compute per-column (per-anchor) Pearson correlation
    sim_img_c = sim_img - sim_img.mean(dim=0, keepdim=True)  # (B, K)
    sim_txt_c = sim_txt - sim_txt.mean(dim=0, keepdim=True)  # (B, K)

    num = (sim_img_c * sim_txt_c).sum(dim=0)                 # (K,)
    den = (sim_img_c.norm(dim=0) * sim_txt_c.norm(dim=0)).clamp(min=1e-8)  # (K,)
    correlations = num / den                                  # (K,)

    return -correlations.mean()


def anchor_orthogonality_loss(
    anchors_img: torch.Tensor,
    anchors_txt: torch.Tensor,
) -> torch.Tensor:
    """Penalise non-orthogonality among anchor vectors.

    Computes the Frobenius norm of the off-diagonal elements of the
    Gram matrix ``A @ A.T`` for each set of anchors (image and text),
    encouraging the K anchor directions to be mutually orthogonal.

    Args:
        anchors_img: (K, D_img) image anchor parameters.
        anchors_txt: (K, D_txt) text anchor parameters.

    Returns:
        Scalar loss (mean of image and text orthogonality penalties).
    """
    def _off_diag_frob(anchors: torch.Tensor) -> torch.Tensor:
        a_norm = F.normalize(anchors, dim=-1)          # (K, D)
        gram = a_norm @ a_norm.T                        # (K, K)
        # Zero out diagonal — we only penalise off-diagonal similarities
        mask = 1.0 - torch.eye(gram.shape[0], device=gram.device)
        return (gram * mask).pow(2).sum() / mask.sum()  # mean squared off-diag

    return (_off_diag_frob(anchors_img) + _off_diag_frob(anchors_txt)) / 2
