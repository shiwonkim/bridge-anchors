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


def per_anchor_info_nce_loss(
    p_img: torch.Tensor,
    p_txt: torch.Tensor,
    temperature: float = 0.07,
    b_cls_img: torch.Tensor | None = None,
    b_cls_txt: torch.Tensor | None = None,
    cls_weight: float = 0.0,
) -> torch.Tensor:
    """InfoNCE loss with per-anchor similarity sum.

    The similarity between image i and text j is the mean of per-anchor
    cosine similarities: sim(i, j) = (1/K) * sum_k cos(p_k^img_i, p_k^txt_j).

    Optionally combines with CLS profile similarity:
    sim(i, j) = sim_anchor(i, j) + cls_weight * cos(b_cls_img_i, b_cls_txt_j)

    Args:
        p_img: (B, K, K) per-anchor L2-normalised profiles for images.
        p_txt: (B, K, K) per-anchor L2-normalised profiles for texts.
        temperature: Scalar temperature for logit scaling.
        b_cls_img: (B, K) optional CLS-based L2-normalised profiles for images.
        b_cls_txt: (B, K) optional CLS-based L2-normalised profiles for texts.
        cls_weight: Weight for CLS similarity contribution.

    Returns:
        Scalar loss (average of i2t and t2i directions).
    """
    K = p_img.shape[1]
    # sim_matrix[i, j] = (1/K) * sum_k cos(p_k^img_i, p_k^txt_j)
    # Dividing by K normalises to [-1, 1], matching standard cosine sim scale
    sim_matrix = torch.einsum("bkd,ckd->bc", p_img, p_txt) / K  # (B, B)

    if cls_weight > 0 and b_cls_img is not None and b_cls_txt is not None:
        sim_matrix = sim_matrix + cls_weight * (b_cls_img @ b_cls_txt.T)

    sim_matrix = sim_matrix / temperature

    labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.T, labels)

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


def token_matching_loss(
    token_sims_img: torch.Tensor,
    token_sims_txt: torch.Tensor,
    txt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bidirectional max-matching loss over per-token anchor profiles.

    For each positive pair, computes:
        img→txt: (1/S) * sum_s max_m cos(img[s], txt[m])
        txt→img: (1/M') * sum_m max_s cos(img[s], txt[m])
    where M' is the number of valid text tokens.

    Loss = -mean over batch of (img2txt + txt2img).

    Processes one sample at a time to avoid materialising a large
    (B, S, M) tensor.

    Args:
        token_sims_img: (B, S, K) L2-normalised per-token image profiles.
        token_sims_txt: (B, M, K) L2-normalised per-token text profiles.
        txt_mask: (B, M) attention mask (1 = valid, 0 = padding).

    Returns:
        Scalar loss.
    """
    B, S, K = token_sims_img.shape
    M = token_sims_txt.shape[1]

    total_sim = torch.tensor(0.0, device=token_sims_img.device)

    for i in range(B):
        # (S, K) @ (K, M) -> (S, M)
        cross_sim = token_sims_img[i] @ token_sims_txt[i].T

        if txt_mask is not None:
            mask_i = txt_mask[i].bool()  # (M,)
            mask_f = mask_i.float()
            n_valid = mask_f.sum().clamp(min=1)
            # Mask out padded text tokens with -inf for img→txt max
            cross_sim_masked = cross_sim.masked_fill(
                (~mask_i).unsqueeze(0), float("-inf"),
            )  # (S, M)
            # img→txt: for each image token, max over valid text tokens
            img2txt = cross_sim_masked.max(dim=1).values.mean()  # mean over S
            # txt→img: for each valid text token, max over image tokens
            cross_sim_t = cross_sim.T  # (M, S)
            txt2img_per_token = cross_sim_t.max(dim=1).values  # (M,)
            txt2img = (txt2img_per_token * mask_f).sum() / n_valid
        else:
            img2txt = cross_sim.max(dim=1).values.mean()
            txt2img = cross_sim.max(dim=0).values.mean()

        total_sim = total_sim + img2txt + txt2img

    return -total_sim / B


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


def anchor_isometry_loss(
    anchors_img: torch.Tensor,
    anchors_txt: torch.Tensor,
) -> torch.Tensor:
    """Gromov-Wasserstein inspired loss aligning anchor geometry across modalities.

    Forces image and text anchors to form the same geometric shape by matching
    their Gram matrices (pairwise cosine similarity structure).

    Args:
        anchors_img: (K, D_img) image anchor parameters.
        anchors_txt: (K, D_txt) text anchor parameters.

    Returns:
        Scalar loss = ||G_img - G_txt||_F^2.
    """
    a_img = F.normalize(anchors_img, dim=-1)  # (K, D_img)
    a_txt = F.normalize(anchors_txt, dim=-1)  # (K, D_txt)
    g_img = a_img @ a_img.T                   # (K, K)
    g_txt = a_txt @ a_txt.T                   # (K, K)
    return (g_img - g_txt).pow(2).sum()


def hierarchical_attention_diversity_loss(
    expert_attns: list[torch.Tensor],
    cls_attn: torch.Tensor,
    num_experts: int,
    sigma: float = 0.2,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL-divergence diversity loss that pushes each expert toward a
    different saliency tier (based on encoder CLS attention ranks).

    Args:
        expert_attns: list of G tensors, each (B, T, K_g) — per-expert CAP
            attention distributions over tokens.
        cls_attn: (B, P) encoder CLS attention scores. If P < T, it is
            zero-prepended to align with the attention map's T axis.
        num_experts: G.
        sigma: Gaussian width for tier target distributions.
        mask: Optional (B, T) text padding mask. 1 = valid, 0 = padding.
            Padded positions are excluded from both q_g and h_g.

    Returns:
        Scalar KL divergence loss, averaged over experts and batch.
    """
    G = num_experts
    B, T, _ = expert_attns[0].shape
    device = expert_attns[0].device

    # Pad cls_attn to T (prepend zeros for CLS/special tokens)
    P = cls_attn.shape[1]
    if P < T:
        pad = torch.zeros(B, T - P, device=device)
        cls_attn_full = torch.cat([pad, cls_attn], dim=1)
    else:
        cls_attn_full = cls_attn[:, :T]

    # Ranks descending: highest attention → rank 0.0, lowest → rank 1.0
    # For padded positions (if mask given), force them to rank 1.0 (peripheral).
    denom = max(T - 1, 1)
    if mask is not None:
        # Mark padded positions as lowest attention (so they rank last).
        cls_for_rank = torch.where(
            mask.bool(), cls_attn_full, torch.full_like(cls_attn_full, -1e9),
        )
    else:
        cls_for_rank = cls_attn_full
    ranks = cls_for_rank.argsort(dim=1, descending=True).argsort(dim=1).float() / denom
    # ranks shape: (B, T)

    total_loss = torch.zeros((), device=device)
    for g in range(G):
        mu_g = g / max(G - 1, 1)  # tier center: 0 → salient, 1 → peripheral
        # Target distribution q_g(t): Gaussian around mu_g
        q_g = torch.exp(-((ranks - mu_g) ** 2) / (2 * sigma ** 2))  # (B, T)
        if mask is not None:
            q_g = q_g * mask.float()
        q_g = q_g / (q_g.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # Expert attention averaged over anchor axis: h_g(t) = mean_k attn[:, t, k]
        h_g = expert_attns[g].mean(dim=2)  # (B, T)
        if mask is not None:
            h_g = h_g * mask.float()
            h_g = h_g / (h_g.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # KL(q_g || h_g) = sum_t q_g * log(q_g / h_g)
        # F.kl_div expects log(input), target ; computes target * (log(target) - input)
        h_g_log = torch.log(h_g + 1e-8)
        kl_g = F.kl_div(h_g_log, q_g, reduction="batchmean")
        total_loss = total_loss + kl_g

    return total_loss / G


def reconstruction_loss(
    expert_profiles_img: list[torch.Tensor],
    expert_profiles_txt: list[torch.Tensor],
    decoders_img,
    decoders_txt,
    target_img: torch.Tensor,
    target_txt: torch.Tensor,
) -> torch.Tensor:
    """Per-expert profile reconstruction of the original CLS embedding.

    Each expert's K_g-dim profile is decoded back to the original encoder
    dimension (768) via a per-expert linear decoder. Targets are detached
    so gradients only flow through the decoder/profile path.

    L = (1/G) * sum_g [MSE(decoder_img_g(profile_img_g), target_img)
                       + MSE(decoder_txt_g(profile_txt_g), target_txt)]

    Args:
        expert_profiles_img: list of G tensors, each (B, K_g) — raw profiles
            from HME forward (before L2 normalization).
        expert_profiles_txt: list of G tensors, each (B, K_g).
        decoders_img: nn.ModuleList of G ProfileDecoders for image.
        decoders_txt: nn.ModuleList of G ProfileDecoders for text.
        target_img: (B, dim_img) detached CLS image embedding (L2-normalized).
        target_txt: (B, dim_txt) detached CLS text embedding (L2-normalized).

    Returns:
        Scalar reconstruction loss (averaged over experts and modalities).
    """
    G = len(expert_profiles_img)
    total = 0.0
    for g in range(G):
        recon_img = decoders_img[g](expert_profiles_img[g])
        recon_txt = decoders_txt[g](expert_profiles_txt[g])
        total = total + F.mse_loss(recon_img, target_img)
        total = total + F.mse_loss(recon_txt, target_txt)
    return total / G


def routing_load_balance_loss(
    soft_gate: torch.Tensor,
    hard_gate: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Switch Transformer-style load balancing loss for hard routing.

    Encourages equal token distribution across experts.
    L = G * sum_g(f_g * p_g)
    where f_g = fraction of tokens assigned to expert g (hard)
          p_g = average routing probability for expert g (soft)

    Args:
        soft_gate: (B, T, G) soft routing probabilities.
        hard_gate: (B, T, G) hard one-hot assignment.
        mask: (B, T) optional padding mask, 1=valid 0=pad.

    Returns:
        Scalar load balancing loss.
    """
    if mask is not None:
        n_valid = mask.sum().clamp(min=1)
        hard_masked = hard_gate * mask.unsqueeze(-1)
        soft_masked = soft_gate * mask.unsqueeze(-1)
        f = hard_masked.sum(dim=[0, 1]) / n_valid
        p = soft_masked.sum(dim=[0, 1]) / n_valid
    else:
        f = hard_gate.mean(dim=[0, 1])
        p = soft_gate.mean(dim=[0, 1])

    G = soft_gate.shape[-1]
    return G * (f * p).sum()


def structure_preservation_loss(
    original_embeddings: torch.Tensor,
    aligned_embeddings: torch.Tensor,
    temperature: float = 0.05,
    structure_levels: int = 1,
) -> torch.Tensor:
    """JS divergence between pairwise similarity distributions.

    Preserves neighborhood structure from the original encoder space in
    the aligned (profile) space.  For each sample, converts its pairwise
    cosine similarities to a probability distribution (row-wise softmax)
    and minimises JS divergence between original and aligned distributions.

    Args:
        original_embeddings: (B, D) frozen encoder output (CLS embeddings).
        aligned_embeddings: (B, K) BA profile output.
        temperature: Softmax temperature for pairwise similarity.
        structure_levels: Multi-scale levels.  1 = direct similarities
            only.  l > 1 also includes l-hop similarities via matrix
            powers of the similarity matrix.

    Returns:
        Scalar JS divergence loss (mean over all samples and levels).
    """
    orig_norm = F.normalize(original_embeddings, dim=-1)
    align_norm = F.normalize(aligned_embeddings, dim=-1)

    S_orig = orig_norm @ orig_norm.T  # (B, B)
    S_align = align_norm @ align_norm.T  # (B, B)

    # Mask out self-similarity (diagonal)
    mask = ~torch.eye(
        S_orig.shape[0], dtype=torch.bool, device=S_orig.device,
    )

    def _js_div(S_o: torch.Tensor, S_a: torch.Tensor) -> torch.Tensor:
        # Row-wise softmax over non-self neighbours
        logits_o = S_o.masked_fill(~mask, float("-inf")) / temperature
        logits_a = S_a.masked_fill(~mask, float("-inf")) / temperature

        P = F.softmax(logits_o, dim=-1)
        Q = F.softmax(logits_a, dim=-1)

        # JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
        # Compute manually to avoid log(0) issues:
        # KL(P||M) = sum_j P_j * log(P_j / M_j)
        M = 0.5 * (P + Q)  # M > 0 wherever P or Q > 0
        # Clamp for numerical safety
        eps = 1e-8
        kl_pm = (P * ((P + eps).log() - (M + eps).log())).sum(dim=-1)
        kl_qm = (Q * ((Q + eps).log() - (M + eps).log())).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)  # (B,)
        return js.mean()

    js = _js_div(S_orig, S_align)

    if structure_levels > 1:
        for _l in range(2, structure_levels + 1):
            S_orig_l = torch.linalg.matrix_power(S_orig, _l)
            S_align_l = torch.linalg.matrix_power(S_align, _l)
            js = js + _js_div(S_orig_l, S_align_l)

    return js


def atcr_loss(
    original_embeddings: torch.Tensor,
    profiles: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Anchor Territory Coherence Regularization.

    Encourages each anchor's "territory" (samples with high profile
    values for that anchor) to be semantically coherent in the original
    encoder space.  Computes weighted pairwise coherence per anchor
    using soft territory membership, then maximises mean coherence.

    Args:
        original_embeddings: (B, D) frozen encoder output (should be
            detached — no gradient through originals).
        profiles: (B, K) L2-normalised BA profiles (gradient flows
            through these).
        temperature: Temperature for column-wise softmax that converts
            profile values to soft territory weights.

    Returns:
        Scalar loss (negative mean coherence — minimise to maximise
        territory coherence).
    """
    # Original pairwise similarity (no gradient)
    orig_norm = F.normalize(original_embeddings, dim=-1)
    S_orig = orig_norm @ orig_norm.T  # (B, B)

    # Soft territory weights: column-wise softmax over samples
    # W[:, k] gives how much each sample belongs to anchor k's territory
    W = F.softmax(profiles / temperature, dim=0)  # (B, K)

    # Coherence per anchor: diag(W^T @ S_orig @ W)
    # = for each k: sum_{i,j} W[i,k] * S_orig[i,j] * W[j,k]
    # Efficient: compute (S_orig @ W) first, then element-wise multiply
    SW = S_orig @ W  # (B, K)
    coherence = (W * SW).sum(dim=0)  # (K,)

    # Maximise coherence → minimise negative coherence
    return -coherence.mean()
