"""Spectral Alignment model for cross-modal alignment.

Aligns two independently trained encoder spaces by:
1. Computing PCA eigenvectors on training data (pre-computation, not learned).
2. Projecting embeddings onto top-K eigenvectors to get spectral coordinates.
3. Learning a soft permutation (temperature-scaled softmax) and per-component
   scaling to align image spectral coordinates to text spectral coordinates.

Only the image side is permuted+scaled; the text side stays as-is.

Status: This approach did NOT converge meaningfully in experiments. See
experiments/exp_spectral_k_ablation/results_summary.md for analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAligner(nn.Module):
    """Aligns two encoder spaces via PCA projection + learned permutation/scaling.

    Pre-computes PCA eigenvectors from training data (stored as frozen buffers),
    then learns a soft permutation matrix and K scaling factors to map image
    spectral coordinates onto text spectral coordinates.

    Args:
        k: Number of PCA components (spectral dimensions).
        eigvecs_img: (dim_img, K) top-K PCA eigenvectors for image embeddings.
        eigvecs_txt: (dim_txt, K) top-K PCA eigenvectors for text embeddings.
        mean_img: (dim_img,) mean of image training embeddings.
        mean_txt: (dim_txt,) mean of text training embeddings.
        tau: Softmax temperature for the soft permutation (lower = crisper).
    """

    def __init__(
        self,
        k: int,
        eigvecs_img: torch.Tensor,
        eigvecs_txt: torch.Tensor,
        mean_img: torch.Tensor,
        mean_txt: torch.Tensor,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.tau = tau

        # Frozen PCA bases — (dim, K) eigenvectors and (dim,) means
        self.register_buffer("eigvecs_img", eigvecs_img)  # (dim_img, K)
        self.register_buffer("eigvecs_txt", eigvecs_txt)  # (dim_txt, K)
        self.register_buffer("mean_img", mean_img)         # (dim_img,)
        self.register_buffer("mean_txt", mean_txt)         # (dim_txt,)

        # Learnable: soft permutation logits (K, K) and scaling factors (K,)
        self.perm_logits = nn.Parameter(torch.zeros(k, k))
        self.scales = nn.Parameter(torch.ones(k))

        # Initialise permutation logits to slight identity bias
        with torch.no_grad():
            self.perm_logits.data += torch.eye(k) * 2.0

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spectrally aligned representations.

        Args:
            img_emb: (B, dim_img) image embeddings.
            txt_emb: (B, dim_txt) text embeddings.

        Returns:
            Tuple of (aligned_img, aligned_txt), each (B, K) L2-normalised.
        """
        # Step 1: Center and project onto PCA eigenvectors → (B, K)
        spec_img = (img_emb - self.mean_img) @ self.eigvecs_img  # (B, K)
        spec_txt = (txt_emb - self.mean_txt) @ self.eigvecs_txt  # (B, K)

        # Step 2: Compute soft permutation matrix
        if self.training:
            # Temperature-scaled softmax: differentiable soft permutation
            perm = F.softmax(self.perm_logits / self.tau, dim=-1)  # (K, K)
        else:
            # Hard argmax: true permutation at inference
            idx = self.perm_logits.argmax(dim=-1)  # (K,)
            perm = torch.zeros_like(self.perm_logits)
            perm.scatter_(1, idx.unsqueeze(1), 1.0)

        # Step 3: Permute and scale image spectral coordinates
        aligned_img = (spec_img @ perm) * self.scales  # (B, K)

        # Step 4: L2-normalise both sides
        aligned_img = F.normalize(aligned_img, dim=-1)  # (B, K)
        aligned_txt = F.normalize(spec_txt, dim=-1)      # (B, K)

        return aligned_img, aligned_txt

    def extra_repr(self) -> str:
        n_params = self.k * self.k + self.k
        return f"k={self.k}, learnable_params={n_params}, tau={self.tau}"
