# Bridge Anchors — Development Log

## Current Status & Next Steps

**Activate environment:** `eval "$(conda shell.bash hook)" && conda activate bridge-anchors`

### Infrastructure
- Full pipeline operational: code, datasets, embeddings, training, evaluation — all tested end-to-end
- **New server (2026-03-25):** Docker environment, Python 3.10, PyTorch 2.4.1+cu118, 2× Quadro RTX 8000 / A40 (46–49 GB each)
- Datasets: COCO 2017, Flickr30k, ImageNet val — all downloaded, embeddings pre-extracted (1.1 GB CLS in `data/embeddings/cls/`)
- Token embeddings: COCO train 118K image tokens extracted (12 chunks × ~3.7 GB, float16, 56 GB total in `data/embeddings/all_tokens/`)
- Each training run takes ~75s (20 epochs on 118K COCO pairs)

### Experiment A — COMPLETE (5 models × 3 seeds = 15 runs)
Main comparison on COCO 118K, evaluated on Flickr30k retrieval + ImageNet zero-shot.

| Model | Params | Flickr30k mR | ImageNet top-1 |
|-------|--------|-------------|----------------|
| LinearProjection | 589K | **23.5** | **17.4%** |
| MLPProjection | 393K | 20.4 | 15.0% |
| BridgeAnchors (K=32, prototype) | 49K | 16.9 | 11.3% |
| BridgeAnchors (K=32, random) | 49K | 16.8 | 11.3% |
| FixedRelativeRep (K=32, random) | 0 | 0.3 | ~0.1% |

BridgeAnchors at K=32 achieves meaningful alignment with 12× fewer params than LinearProjection, but performance is limited by the low-dimensional (32-dim) bridged representation. FixedRelativeRep at chance level confirms that **learnable anchors are essential** — random measurement points in independently trained spaces provide no cross-modal alignment. Details: `experiments/exp_a_main/results_summary.md`

### Experiment B — COMPLETE (7 K values × 1 seed = 7 runs)
K ablation for BridgeAnchors on COCO 118K, Flickr30k retrieval. Seed=42.

| K | Params | Flickr30k mR |
|---|--------|-------------|
| 4 | 6K | 0.5 |
| 8 | 12K | 3.4 |
| 16 | 25K | 9.8 |
| 32 | 49K | 16.7 |
| 64 | 98K | 21.7 |
| **128** | **197K** | **23.7** |
| **256** | **393K** | **24.0** |

**BridgeAnchors K=128 (197K params) surpasses LinearProjection (590K params) with mR 23.66 vs 23.57 — 3× fewer parameters.** K=128 is the sweet spot: K=64→128 adds +1.95 mR, but K=128→256 adds only +0.36. The K=32 bottleneck in Exp A was purely a dimensionality issue, not a limitation of the anchor approach. Details: `experiments/exp_b_k_ablation/results_summary.md`, plot: `experiments/exp_b_k_ablation/mean_recall_vs_k.png`

### Experiment C — COMPLETE (4 models × 6 data sizes × 1 seed = 24 runs)
Data efficiency: BridgeAnchors K=128 (random vs prototype vs kmeans init, 197K params) vs LinearProjection (590K params).

| N (pairs) | BA random mR | BA prototype mR | BA kmeans mR | LP mR |
|-----------|-------------|----------------|-------------|-------|
| 500 | 0.8 | 1.3 | 0.5 | 1.9 |
| 1,000 | 2.3 | 2.9 | 1.9 | 4.6 |
| 5,000 | 7.9 | 8.9 | 9.0 | 11.3 |
| 10,000 | 11.6 | 11.9 | **12.3** | 14.5 |
| 50,000 | 20.3 | 19.9 | 20.4 | 21.4 |
| **118,287** | **23.7** | 23.2 | 23.6 | 23.6 |

**K-means init shows a distinctive crossover pattern**: worst at tiny scales (0.48 at N=500 — centroids overfit with ~4 samples/cluster), but best BA variant at medium scales (12.31 at 10K, 20.36 at 50K). Prototype init helps at N≤1K. **No init strategy closes the gap with LinearProjection at small scales.** All four converge within ~0.5 mR at 118K. Details: `experiments/exp_c_data_efficiency/results_summary.md`, plot: `experiments/exp_c_data_efficiency/mean_recall_vs_data.png`

### Experiment B2 — COMPLETE (Direction B: orthogonal anchor regularization)
Orthogonal regularization (λ=0.1) K ablation on COCO 118K, Flickr30k retrieval. Seed=42.

| K | BA mR | BA+ortho mR | Δ |
|---|-------|-------------|---|
| 4 | 0.52 | 0.52 | 0.00 |
| 8 | 3.42 | 3.44 | +0.02 |
| 16 | 9.79 | 9.76 | -0.03 |
| 32 | 16.69 | 16.70 | +0.01 |
| 64 | 21.71 | 21.71 | 0.00 |
| 128 | 23.66 | 23.67 | +0.01 |
| 256 | 24.02 | 24.03 | +0.01 |

**Null result: ortho reg has zero measurable effect (max Δ = 0.03 mR).** In 768-d space, random anchors are already near-orthogonal, and InfoNCE implicitly encourages diversity. Simple orthogonal regularization solves a non-existent problem. Details: `experiments/exp_b2_ortho_k_ablation/results_summary.md`, plot: `experiments/exp_b2_ortho_k_ablation/mean_recall_vs_k_ortho.png`

### SpectralAligner — FAILED (Idea 3: spectral alignment via learned permutation)
PCA projection + soft permutation + scaling. K ablation on COCO 118K, Flickr30k retrieval. Seed=42.

| K | SA params | SA mR | BA mR | Ratio |
|---|-----------|-------|-------|-------|
| 4 | 20 | 0.03 | 0.52 | 17× |
| 32 | 1K | 0.12 | 16.69 | 139× |
| 128 | 17K | 0.34 | 23.66 | 70× |
| 256 | 66K | 0.50 | 24.02 | 48× |

**Total failure: 0.03–0.50 mR across all K (chance level).** The assumption that PCA axes of independently trained encoders have a 1-to-1 correspondence recoverable by permutation is wrong. The relationship is a dense linear transform, not a permutation. Additionally, soft permutation optimization is fundamentally difficult (K! search space, train-eval mismatch from soft→hard). Tested tau={0.1, 1.0}, lr={1e-3, 1e-2, 1e-1} — none converged. This validates the BridgeAnchors design: learning free anchor positions is far more effective than constraining alignment to axis reordering. Details: `experiments/exp_spectral_k_ablation/results_summary.md`, plot: `experiments/exp_spectral_k_ablation/mean_recall_vs_k.png`

### PCA-reduced BridgeAnchors — COMPLETE (pca_dim × K sweep)
BridgeAnchors with PCA-reduced embeddings. 2D sweep on COCO 118K, Flickr30k retrieval. Seed=42.

| pca_dim \ K | 32 | 64 | 128 |
|-------------|------|------|------|
| 32 | 2.5 (2K) | — | — |
| 64 | 6.0 (4K) | 6.4 (8K) | — |
| 128 | 10.2 (8K) | 12.7 (16K) | 13.1 (33K) |
| 256 | 14.0 (16K) | 18.0 (33K) | **19.2 (66K)** |

Best PCA-BA: d=256, K=128 → 19.2 mR (66K params) vs standard BA K=128 → 23.7 mR (197K params) vs LP → 23.6 mR (590K params). **PCA reduction trades 4.4 mR for 3× param savings — not worthwhile.** PCA dimension is the dominant factor (variance-preserving basis discards cross-modal signal). Standard BA in full 768-d is more efficient in performance-per-parameter. Details: `experiments/exp_pca_ba/results_summary.md`, heatmap: `experiments/exp_pca_ba/heatmap_pca_ba.png`

### Full-Scale Comparison — COMPLETE (15 runs + CLS baselines)
All models × {cls/cls, tok/cls, tok/tok}. COCO 118K, Flickr30k eval, seed=42, 20 epochs, per-model optimal BS/LR.

| Rank | Model | Input | Params | **mR** |
|------|-------|-------|--------|--------|
| 1 | **FreezeAlign** | tok/cls | 6.5M | **29.11** |
| 2 | **MLPProj** | tok/tok | 393K | **28.85** |
| 3 | **MLPProj** | tok/cls | 393K | **28.79** |
| 4 | Token BA K=512 | tok/cls | 787K | 27.52 |
| 5 | FreezeAlign | tok/tok | 6.5M | 27.61 |
| 6 | Token BA K=128 | tok/cls | 197K | 27.27 |
| 7 | LP | tok/cls | 590K | 26.51 |
| 8 | Token BA K=64 | tok/cls | 98K | 25.34 |
| 9 | CLS BA K=128 (opt) | cls/cls | 197K | 25.05 |
| 10 | LP | cls/cls | 590K | 23.77 |
| 11 | FA | cls/cls | 6.5M | 21.23 |
| 12 | MLP | cls/cls | 393K | 20.77 |

**Key findings:**
- **MLPProj tok/cls (393K, 28.79 mR) is the new lightweight champion** — within 0.32 of FA (6.5M) at 17× fewer params. The bottleneck regularization + token input is a powerful combo.
- Token-level input helps ALL models: MLP gains +8.02, FA +7.88, LP +2.74, BA +2.22
- tok/tok still doesn't help any model (text CLS is sufficient)
- BA K=128 tok/cls (197K, 27.27) remains best at ultra-low param budgets

Details: `experiments/exp_full_comparison/results_summary.md`

### Immediate next steps
1. Experiment D — fixed vs learnable anchors, random vs prototype init
2. Attention-weighted token pooling (learnable query)
3. Multi-seed validation of key results

### Direction A Anchor Analysis — COMPLETE (5 analyses on BA K=128)
Comprehensive analysis of learned anchors from Experiment B (K=128, seed=42, COCO 118K).

**Key findings:**
- **Cross-modal correspondence**: Supercategory overlap 70.3% — paired image/text anchors attend to the same broad semantic concepts. Category-level overlap 22.5% (expected to be lower due to finer granularity).
- **Parallel structure**: Pearson r=0.660, CKA=0.859 between Gram matrices — anchors organized into strongly parallel structures across modalities. Anchor pairs that are close in image space are also close in text space.
- **No dead anchors**: 128/128 active in both modalities. Usage range 322–1990 (image), 185–2017 (text) vs uniform=924. Distribution is non-uniform but no anchor is unused.
- **Coverage**: UMAP shows anchors cluster in the center of the embedding space rather than spreading to the periphery — they occupy a densely-connected semantic core. Individual supercategory clusters in the periphery are reached via similarity to nearby anchors rather than dedicated anchors.

Details: `experiments/exp_anchor_analysis/results_summary.md`, nearest neighbors: `nearest_neighbors.md`

### Step 1: Load Balancing Loss — COMPLETE (lambda sweep)
Switch Transformer-style LB loss for anchor usage uniformity. K=128, COCO 118K, Flickr30k retrieval.

| lb_lambda | mR | Usage Std (img) | Cat Overlap | Supercat Overlap |
|-----------|------|----------------|-------------|------------------|
| 0.0 (baseline) | 23.66 | 320 | 0.225 | 0.703 |
| 0.01 | 23.66 | 320 | 0.225 | 0.704 |
| 0.1 | 23.66 | 316 | 0.226 | 0.706 |
| 0.5 | 23.66 | 301 | 0.223 | 0.702 |
| 1.0 | 23.65 | 283 | 0.223 | 0.702 |

**Zero impact on retrieval (23.65–23.66 mR across all lambdas). Modest improvement in usage uniformity** (std 320→283 at lb=1.0, ~12%). No dead anchors at any lambda. Cross-modal correspondence unaffected. The LB loss is safe but provides limited benefit — InfoNCE already distributes anchor usage naturally in 768-d space. Details: `experiments/exp_step1_lb_loss/results_summary.md`

### Intermediate Layer Alignment — COMPLETE (negative result)
CKA analysis + training on top-3 layer pairs. COCO 118K, BA K=128, Flickr30k retrieval.

| Layer Pair | CKA | mR | Δ vs Baseline |
|------------|------|------|---------------|
| Final × Final (baseline) | 0.461 | **23.66** | — |
| Block 10 × Layer 10 | 0.514 | 17.82 | -5.84 |
| Block 9 × Layer 11 | 0.552 | 15.79 | -7.87 |
| Block 9 × Layer 10 | **0.586** | 15.72 | -7.94 |

**Final layers are best — higher CKA actually predicts *worse* alignment performance.** Intermediate layers with higher cross-modal CKA capture generic, shared features but lack the discriminative semantics needed for retrieval. The hypothesis that "highest-CKA layer pair → better alignment" is not confirmed. Details: `experiments/exp_intermediate_layer/results_summary.md`, CKA heatmap: `cka_heatmap.png`

### Steps 2/3/4 — COMPLETE (sparse gating, per-anchor loss, FPS init)

| Method | mR | Δ | Verdict |
|--------|------|------|---------|
| Baseline (random, all anchors, InfoNCE only) | 23.66 | — | — |
| **Step 4: FPS init** | **23.72** | **+0.06** | marginal positive |
| Step 3: pa=0.01 | 23.65 | -0.01 | neutral/negative |
| Step 2: top_k=96 (best sparse) | 12.85 | -10.81 | severe negative |

**Step 2 (Sparse Gating)**: Devastating — all top-k values destroy performance (8.9–12.9 mR). Straight-through estimator creates forward/backward mismatch; L2 norm after sparsification amplifies noise.
**Step 3 (Per-Anchor Contrastive)**: Slightly negative at all lambdas (-0.01 to -1.64 mR). InfoNCE already implicitly enforces cross-modal anchor correspondence; explicit Pearson correlation is redundant.
**Step 4 (FPS Init)**: Only improvement (+0.06 mR). Maximally spread initial anchors give a marginally better starting point.

**Key insight**: BridgeAnchors K=128 + InfoNCE is already well-optimized. Auxiliary losses and sparsity don't help. Details: `experiments/steps_2_3_4_summary.md`

### Token-Level BridgeAnchors Pilot — POSITIVE RESULT
Token-level BA using patch tokens (CLS + 256 patches) vs CLS-only. 10K COCO subset, Flickr30k eval.

| Model | Params | mR | Δ vs CLS BA |
|-------|--------|------|-------------|
| CLS BA K=128 | 197K | 11.58 | — |
| CLS LinearProj | 590K | 14.48 | +2.90 |
| **Token BA K=128 mean** | **197K** | **13.82** | **+2.24** |
| Token BA K=128 max | 197K | 13.41 | +1.83 |
| **Token BA K=256 mean** | **393K** | **15.00** | **+3.42** |

**Token-level BA is the first substantial improvement over standard BA.** Token BA K=128 gains +2.24 mR (19% relative) at identical param count. Token BA K=256 (15.00 mR) surpasses CLS LinearProjection (14.48 mR) with 33% fewer params. Mean pooling > max pooling. More anchors help with richer token input. Details: `experiments/exp_token_level_pilot/results_summary.md`

### Future work

1. **Token-level BA full-scale** — scale to 118K COCO (requires chunked storage for ~88GB token embeddings)
2. **Experiment D** — fixed vs learnable anchors, random vs prototype init
3. **Direction C** — residual combination with fixed relative representations
4. Attention-weighted token pooling (learnable query over patches)

---

Reverse-chronological record of development activity. Newest entries first.

---

## 2026-03-26 — Token-level support for all baselines (Prompt 40)

**What was done:**

Extended the `--img-input {cls,tokens}` / `--txt-input {cls,tokens}` flag system to all 3 baseline models:
- **LinearProjection**: applies projection per-token, then mean pools
- **MLPProjection**: applies MLP per-token, then mean pools
- **FixedRelativeRep**: computes per-token anchor similarities, then mean pools

Key changes:
- `src/models/baselines.py`: Added `txt_mask` parameter and `_masked_mean_pool` helper. All 3 models auto-detect 2D (CLS) vs 3D (token) input. Masked mean pooling for variable-length text.
- `src/train.py`: Added LP, MLP, FRR to `_run_token_level` model creation. Added FRR eval-only early-return (matching CLS path behavior).
- Zero new parameters — same weights applied per-token.
- Backward compatible: cls/cls produces identical results (verified: loss=2.6296, mR=16.5 matches exactly).

Smoke test results (1 epoch, BS=256, LR=1e-3):

| Model | cls/cls | tok/cls | cls/tok | tok/tok |
|-------|---------|---------|---------|---------|
| LinearProj | 16.5 | 19.3 | 16.5 | 19.3 |
| MLPProj | 17.4 | 21.0 | 17.5 | 21.0 |
| FixedRelRep | 1.1 | 1.0 | 1.0 | 1.0 |

Pattern: tok/cls gives +2.8–3.6 mR boost; tok/tok ≈ tok/cls (text tokens don't help, consistent with BA/FA findings). FRR is at chance with random anchors as expected.

---

## 2026-03-26 — LP/MLP full-scale experiments + comprehensive comparison (Prompt 41)

**What was done:**

### BS/LR tuning for LP and MLP
- 3-epoch probes: BS={256, 512, 1024, 2048} with linear LR scaling
- Both LP and MLP converge on **BS=1024, LR=4e-3** as optimal (different from BA's BS=1024/LR=8e-3 due to different base BS)
- For CLS path: tested BS={256, 1024, 4096, 8192}, BS=1024/LR=4e-3 best for both

### Full 20-epoch runs (6 new experiments)

| Model | Input | Params | mR |
|-------|-------|--------|------|
| LP | cls/cls | 590K | 23.77 |
| LP | tok/cls | 590K | 26.51 |
| LP | tok/tok | 590K | 26.48 |
| MLP | cls/cls | 393K | 20.77 |
| **MLP** | **tok/cls** | **393K** | **28.79** |
| MLP | tok/tok | 393K | 28.85 |

### Key discovery: MLPProj tok/cls is the new lightweight champion
- **28.79 mR (393K params)** — within 0.32 of FreezeAlign (29.11, 6.5M) at 17× fewer params
- Beats Token BA K=128 (27.27, 197K) by +1.52 mR with 2× params
- MLP's bottleneck (768→256→768) regularizes well + token input provides spatial info the bottleneck was starved of
- MLP has the largest cls→tok gain: +8.02 mR (+38.6%), indicating CLS was the bottleneck, not model capacity

### Updated comprehensive comparison (all 5 model families)
- Updated results_summary.md with Optimal Hyperparameters reference table
- Created 3 plots: comparison_plot.png, param_efficiency_plot.png, token_improvement_plot.png
- Updated results.csv with all 16 configurations

**Files changed:**
- `experiments/exp_full_comparison/results_summary.md` — comprehensive rewrite with all models
- `experiments/exp_full_comparison/results.csv` — all 16 configs
- `experiments/exp_full_comparison/comparison_plot.png` — updated with LP/MLP
- `experiments/exp_full_comparison/param_efficiency_plot.png` — updated
- `experiments/exp_full_comparison/token_improvement_plot.png` — NEW

---

## 2026-03-26 — Full-scale token-level comparison experiments (Prompt 39)

**What was done:**

### Text token extraction
- Extracted all-mpnet-base-v2 token-level embeddings (before pooling) for COCO train (118,287 captions) and Flickr30k test (31,783 captions)
- COCO: `[118287, 59, 768]` float16, 10.72 GB + 6.98 MB masks
- Flickr30k: `[31783, 83, 768]` float16, 4.05 GB + 2.64 MB masks
- Extraction time: 2.1 minutes. All tokens L2-normalized.

### Token-level batch size / LR optimization
- Removed hardcoded BS=128 cap in `_run_token_level` (line 682)
- All batch sizes {128, 256, 512, 1024, 2048, 4096} fit on A40 (46 GB)
- Linear LR scaling (base: BS=128, LR=1e-3) tested at 3-epoch probes
- **Optimal: BS=1024, LR=8e-3** (mR=25.6 @ 3 epochs)

### Full-scale experiments (9 new runs)
6 Token BA runs + 3 FreezeAlign runs, all 20 epochs, COCO 118K, Flickr30k eval.

| Model | Input | K | Params | mR |
|-------|-------|---|--------|------|
| Token BA | tok/cls | 64 | 98K | 25.34 |
| **Token BA** | **tok/cls** | **128** | **197K** | **27.27** |
| Token BA | tok/cls | 256 | 393K | 27.45 |
| Token BA | tok/cls | 512 | 787K | 27.52 |
| Token BA | tok/tok | 128 | 197K | 27.26 |
| Token BA | tok/tok | 256 | 393K | 27.47 |
| FreezeAlign | cls/cls | - | 6.5M | 21.23 |
| **FreezeAlign** | **tok/cls** | **-** | **6.5M** | **29.11** |
| FreezeAlign | tok/tok | - | 6.5M | 27.61 |

### Key findings
1. **Token input is transformative**: BA tok/cls K=128 (27.27) vs CLS BA K=128 (25.05) = +2.22 mR at same params
2. **BA is 33× more parameter-efficient than FA**: 197K params → 27.27 mR (94% of FA's 29.11 at 6.5M)
3. **tok/tok doesn't help either model** — text CLS is sufficient
4. **FA needs tokens**: FA cls/cls (21.23) is worst of all models
5. **K=128 is the sweet spot**: K=128→512 gains only +0.25 mR total

**Files changed/created:**
- `src/train.py` — removed BS=128 cap in token-level path
- `experiments/exp_full_comparison/results_summary.md` — comprehensive analysis
- `experiments/exp_full_comparison/results.csv` — all metrics
- `experiments/exp_full_comparison/comparison_plot.png` — grouped bar chart
- `experiments/exp_full_comparison/param_efficiency_plot.png` — params vs mR scatter

---

## 2026-03-25 — Server migration, batch/LR optimization, token extraction (Prompt 38)

**What was done:**

### Server migration & baseline reproduction
- Moved to new server: 2× Quadro RTX 8000 (46–49 GB each), Docker environment with base conda (no named env — packages installed directly)
- CLS-only baselines reproduced on new server:
  - BA K=128: 23.66 mR (matches previous server exactly)
  - LinearProj: 23.57 mR (matches)

### Batch size sweep (CLS BA K=128, COCO 118K)
- Swept BS={256, 512, 1024, 2048, 4096, 8192, 16384, 32768} at default LR=1e-3
- **BS=1024 optimal at default LR**: mR=24.34 (+0.68 over BS=256 baseline)
- Clear inverted-U: performance rises 256→1024, drops sharply above 2048
- Results: `experiments/exp_batch_size_sweep/results_summary.md`

### LR scaling experiment (CLS BA K=128, COCO 118K)
- Tested linear and sqrt LR scaling rules at BS={1024, 2048, 4096, 8192}
- **BS=8192 + LR=32e-3 (linear scaling) achieves mR=25.05** — best CLS-only result, +1.39 over original baseline
- BS=4096 + LR=16e-3 nearly identical (25.04) at half the memory — practical recommendation
- Linear scaling completely fixes large-batch degradation; sqrt under-compensates
- Results: `experiments/exp_lr_scaling/results_summary.md`

### FreezeAlign audit & unified CLI
- FreezeAlign implementation audited against reference code — all 4 checks passed, matches exactly
- Added unified `--img-input {cls,tokens}` and `--txt-input {cls,tokens}` CLI flags
- Both BA and FreezeAlign support all 4 input combos (cls/cls, tok/cls, cls/tok, tok/tok)
- Paper vs code differences documented

### Image token embedding extraction
- Full COCO train 118K image tokens extracted to `data/embeddings/all_tokens/`
- 12 chunks (00–11): chunks 00–10 = 10,000 images, chunk 11 = 8,287 → 118,287 total
- Shape per chunk: `[N, 257, 768]` (CLS + 256 patches), dtype=float16
- Total disk: ~56 GB (including Flickr30k test tokens + text CLS + metadata)
- Extraction verified: all chunks present, shapes correct, total count matches

### Current best results
| Config | Params | mR | Notes |
|--------|--------|------|-------|
| CLS BA K=128, BS=8192, LR=32e-3 | 197K | **25.05** | Best CLS-only |
| CLS BA K=128, BS=4096, LR=16e-3 | 197K | 25.04 | Practical best (half memory) |
| CLS BA K=128, BS=1024, LR=1e-3 | 197K | 24.34 | Best at default LR |
| CLS BA K=128, BS=256, LR=1e-3 | 197K | 23.66 | Original baseline |
| CLS FreezeAlign, BS=256, LR=1e-3 | 6.5M | ~17.11 | 1-epoch only, needs full training |

### Next steps (2026-03-26)
1. Extract text token embeddings (for tok/tok experiments)
2. Full-scale token-level BA experiments: K={64, 128, 256, 512} with optimal BS/LR
3. Full-scale FreezeAlign experiments: all 4 input combos
4. Comprehensive comparison: BA vs FreezeAlign across all settings

---

## 2026-03-25 — FreezeAlign audit + unified CLI flags (Prompt 37)

**What was done:**

**Task 1: FreezeAlign audit** — checked all 4 issues against reference code:
1. **PatchProjection residual skip**: Reference has `Linear(x) + [Linear→GELU→Linear](x)`. No explicit `x+` skip — linear branch IS the residual. GELU not ReLU. **Our code: correct.**
2. **Text CLS Token Projector**: Reference has NO separate text CLS projector. CLS is included in the token mean pool through `local_text_proj`. **Our code: correct.**
3. **Vision global projector**: Reference uses `nn.Identity()` — NO global vision MLP. **Our code: correct.**
4. **Weight sharing**: Automatic via batched matmul. **Our code: correct.**

All checks passed — FreezeAlignProjector matches the reference implementation.

**Task 2: Unified CLI flags** — added `--img-input {cls,tokens}` and `--txt-input {cls,tokens}`:
- 4 combinations: cls/cls, tok/cls, cls/tok, tok/tok
- Both BA and FreezeAlign support all 4 combos
- BA auto-selects TokenBridgeAnchorAligner when either input uses tokens
- Legacy `--token-level` translated to `--img-input tokens --txt-input cls --chunked`
- Updated `evaluate_retrieval` and `_bridge_batched` to pass through `txt_mask`
- Refactored `_run_token_level` to handle all combos and both model types

**Smoke test results:**

| Model | img/txt | 1-epoch mR | Params |
|-------|---------|-----------|--------|
| BA K=128 | cls/cls | 12.64 | 196,608 |
| Token BA K=128 | tok/cls chunked | 17.02 | 196,608 |
| FreezeAlign | cls/cls | 17.11 | 6,502,657 |
| FreezeAlign | tok/cls chunked | 18.54 | 6,502,657 |
| Both models × all 4 combos | mock data | shapes correct | — |

**Files changed:**
- `src/models/freeze_align.py` — audited, confirmed correct
- `src/train.py` — new `--img-input`/`--txt-input` flags, refactored `_run_token_level`
- `src/eval/retrieval.py` — `txt_mask` passthrough in `evaluate_retrieval` and `_bridge_batched`
- `experiments/exp_freeze_align_baseline/implementation_notes.md` — updated to v3

---

## 2026-03-25 — Freeze-Align audit + text token-level support (Prompt 36)

**What was done:**
- Audited FreezeAlignProjector against reference implementation — found 5 differences and fixed all
- Added text token-level support to FreezeAlignProjector and TokenBridgeAnchorAligner
- Created text token extraction script
- Updated ChunkedTokenDataset for optional text token loading

**Differences fixed:**

| Issue | v1 | v2 (fixed) |
|-------|-----|-----------|
| Temperature | Fixed 0.07 | Learnable nn.Parameter, clamped [0.001, 0.5] |
| Text projection | ProjectionHead on CLS only | local_text_proj (PatchProj on tokens) + text_proj (MLP after pooling) |
| Text input | CLS-only | Supports CLS (B, 768) and tokens (B, S, 768) + mask |
| Vision token order | Separate first, project patches | Project ALL tokens first, then separate (matches reference) |
| text_proj input dim | dim_txt | embed_dim (after local_text_proj) |

**Updated parameter counts:**

| Model | Params |
|-------|--------|
| BridgeAnchors K=128 | 196,608 |
| LinearProjection | 589,824 |
| FreezeAlign (full) | **6,502,657** (33× BA) |
| Token BA K=128 | 196,608 (unchanged — no extra params for text tokens) |

**TokenBridgeAnchorAligner extension:**
- Now accepts txt_emb as (B, S, D) + txt_mask → per-token anchor sims → masked mean → (B, K)
- No additional parameters — same A_txt anchors used on all tokens
- Backward compatible: (B, D) input works as before

**Files changed/created:**
- `src/models/freeze_align.py` — complete rewrite matching reference exactly
- `src/models/token_bridge_anchors.py` — added txt_mask parameter, text token path
- `src/data/chunked_token_dataset.py` — added text_token_level, text_token_path, text_mask_path
- `scripts/extraction/extract_text_token_embeddings.py` — new extraction script
- `src/train.py` — learnable temp support via `model.temp` attribute

**Smoke tests:** All passed (FreezeAlign CLS/token, Token BA CLS/token, temp gradient flow)

---

## 2026-03-25 — Freeze-Align baseline implementation (Prompt 35)

**What was done:**
- Read Freeze-Align reference implementation (`freeze-align/train/models/clip_adjustable_combined_vis_cls.py`)
- Implemented `FreezeAlignProjector` in `src/models/freeze_align.py`
- Integrated into training pipeline with `--model freeze_align` CLI option
- 1-epoch smoke test passed: mR=17.89, loss converging normally

**Architecture (from reference):**
- **PatchProjection (Token Projector):** `Linear(x) + [Linear(x) → GELU → Linear(x)]` — residual sum of linear and non-linear branches
- **Vision:** Separate CLS projector + local patch projector (both PatchProjection wrapped in LayerNorm+Dropout), combined by addition, then L2-normalized. Mean pooling for patches.
- **Text:** ProjectionHead MLP (Linear → GELU → Linear → Dropout → residual → LayerNorm), L2-normalized
- **Loss:** Standard symmetric InfoNCE (same as ours)

**Parameter comparison:**

| Model | Params |
|-------|--------|
| BridgeAnchors K=128 | 197K |
| LinearProjection | 590K |
| FreezeAlignProjector | **4,729K** (24× BA, 8× LP) |

**Key design decisions:**
- Followed reference 'patch' config with LayerNorm+Dropout wrapping (not simplified)
- CLS-only fallback for (B, 768) input uses cls_vision_proj only
- No local_text_proj (our text is pre-pooled CLS, not raw tokens)
- Fixed temp=0.07 for fair comparison (reference uses learnable temp)

**Files:** `src/models/freeze_align.py`, `experiments/exp_freeze_align_baseline/implementation_notes.md`

---

## 2026-03-25 — Full-scale token-level embedding extraction (Prompt 34)

**What was done:**
- Extracted DINOv2 ViT-B/14 token embeddings (CLS + 256 patches = 257 tokens × 768 dim) for COCO train 118K and Flickr30k test 31K
- All saved to `data/embeddings/all_tokens/` in float16
- Updated extraction script (`scripts/extraction/extract_token_embeddings_full.py`) with new output dir, Flickr30k extraction, text copying, pilot indices
- Updated all code paths: `src/data/chunked_token_dataset.py`, `src/train.py` — all token-level references now point to `all_tokens/`
- CLS-level code paths unchanged (still `data/embeddings/cls/`)

**Files created:**

| File | Shape | Size |
|------|-------|------|
| coco_train_chunk_00..10_img.pt | (10000, 257, 768) fp16 | 3.7 GB each |
| coco_train_chunk_11_img.pt | (8287, 257, 768) fp16 | 3.1 GB |
| flickr30k_test_img.pt | (31783, 257, 768) fp16 | 12 GB |
| coco_train_txt.pt | (118287, 768) fp32 | 347 MB |
| flickr30k_test_txt.pt | (31783, 768) fp32 | 94 MB |
| chunk_metadata.json | 12 chunks metadata | 2 KB |
| pilot_10k_indices.pt | (10000,) indices | 1 MB |

**Total: ~56 GB** (44 GB COCO chunks + 12 GB Flickr30k)

**Timing:** ~15 min total (COCO: ~12 min at ~1 min/chunk, Flickr30k: ~3 min), batch_size=64

**Verification:** All shapes, dtypes (float16 images, float32 text), and chunk boundaries confirmed correct.

---

## 2026-03-25 — LR scaling with larger batch sizes (Prompt 33)

**What was done:**
- Tested linear and sqrt LR scaling to recover large-batch performance for CLS BA K=128
- Linear rule: lr = 1e-3 × (bs/256). Sqrt rule: lr = 1e-3 × sqrt(bs/256)
- 7 new runs: linear {1024, 2048, 4096, 8192} + sqrt {2048, 4096, 8192}

**Results:**

| Batch Size | Original mR | Linear LR mR | Sqrt LR mR |
|-----------|-------------|-------------|------------|
| 256 | 23.66 | — | — |
| 1024 | 24.34 | 24.59 | — |
| 2048 | 24.24 | 24.90 | 24.81 |
| 4096 | 23.33 | **25.04** | 24.92 |
| 8192 | 20.58 | **25.05** | 24.46 |

**Key findings:**
- **Linear LR scaling completely fixes large-batch degradation** — mR increases monotonically to 25.05 at bs=8192
- **New CLS BA best: mR=25.05** (+1.39 over original bs=256 baseline, +0.71 over bs=1024 best)
- Diminishing returns above bs=4096 (+0.01 from 4096→8192), so **bs=4096 lr=16e-3 is the practical optimum** (same perf, half memory)
- Sqrt scaling effective but weaker — under-compensates at bs=8192 (24.46 vs 25.05)
- Val loss minima (2.529–2.531) align perfectly with best mR configs

**Timing:** ~13 min total (7 runs × ~2 min each)

**Files:** `experiments/exp_lr_scaling/results_summary.md`, `results.csv`, `scaling_plot.png`

---

## 2026-03-25 — Batch size sweep for CLS BA K=128 (Prompt 32)

**What was done:**
- Found max batch size: 32768 (65536 OOMs on RTX 8000 — 16 GB needed for B×B similarity matrix)
- Ran full sweep: bs={256, 512, 1024, 2048, 4096, 8192, 32768}, all 20 epochs, seed=42, COCO 118K

**Results:**

| Batch Size | mR | Δ vs 256 | Epoch Time |
|-----------|------|----------|------------|
| 256 | 23.66 | — | 2.3s |
| 512 | 24.10 | +0.44 | 1.2s |
| **1024** | **24.34** | **+0.68** | **1.0s** |
| 2048 | 24.24 | +0.58 | 0.9s |
| 4096 | 23.33 | -0.33 | 1.0s |
| 8192 | 20.58 | -3.08 | 1.3s |
| 32768 | 9.53 | -14.13 | 2.8s |

**Key findings:**
- **Optimal batch size: 1024 (mR=24.34)** — free +0.68 mR gain over baseline with no extra parameters
- Clear inverted-U pattern: more negatives help up to 1024, then too few gradient steps per epoch hurt (118K/32768 = only 3.6 batches)
- bs=1024 is also fastest (1.0s/epoch) and has lowest val loss (2.554)
- Very large batch sizes would need LR scaling/LARS, but unlikely to beat 1024 at this dataset size
- **Recommendation: update default batch size from 256 to 1024**

**Timing:** ~14 min total (7 runs × ~2 min each)

**Files:** `experiments/exp_batch_size_sweep/results_summary.md`, `results.csv`, `scaling_plot.png`

---

## 2026-03-25 — New server setup + CLS baseline validation (Prompt 31)

**What was done:**
- Set up new Docker server (2× Quadro RTX 8000, CUDA 12.4, PyTorch 2.4.1+cu118)
- Downloaded COCO 2017 (train2017: 118,287 images, val2017: 5,000 images, annotations) and Flickr30k (31,783 images + captions) from scratch
- Extracted CLS embeddings: DINOv2 ViT-B/14 (images) + all-mpnet-base-v2 (text) for COCO train and Flickr30k test
- Trained and validated both CLS baselines on full COCO 118K, seed=42

**Results — exact reproduction:**

| Model | Params | New Server mR | Previous Server mR | Δ |
|-------|--------|---------------|-------------------|---|
| BridgeAnchors K=128 | 196,608 | 23.66 | 23.66 | 0.00 |
| LinearProjection | 589,824 | 23.57 | 23.57 | 0.00 |

**Key details:**
- NAS (`/mnt/2021_NIA_data/`) not accessible from Docker container — all data downloaded fresh
- No conda env in Docker — deps installed system-level via pip in pytorch base image
- COCO download took ~4.5 hours (server throttling, multiple restarts with `wget -c`)
- Flickr30k downloaded via Kaggle API in ~3 seconds; captions converted from CSV to .token format
- Embedding extraction: COCO train ~10 min, Flickr30k ~2 min
- Training: ~2 min per run (20 epochs), comparable to previous server

**Files created:**
- `data/embeddings/coco_train_img.pt`, `coco_train_txt.pt` (347 MB each)
- `data/embeddings/flickr30k_test_img.pt`, `flickr30k_test_txt.pt` (93 MB each)
- `experiments/exp_new_server_validation/results_summary.md`

---

## 2026-03-24 — Docker environment image (Prompt 30)

**What was done:**
- Created `docker/` directory with Dockerfile, requirements.txt, docker-compose.yml, .dockerignore, README.md
- Base image: `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime` (Python 3.11, PyTorch 2.4.1+cu118)
- Image contains environment only — code is volume-mounted at runtime, data mounted separately
- Updated `.gitignore` to exclude `data/embeddings/`, `experiments/`, `results/`, `*.pt`

**Build & verification:**
- `docker build -t bridge-anchors:v1 .` — successful
- CUDA check: `torch.cuda.is_available() = True`, PyTorch 2.4.1+cu118, GPU detected (NVIDIA A40)
- 1-epoch baseline: mR=12.64 — exact match with native conda environment

**Key decisions:**
- Used cu118 base image (matching current env) rather than cu124 — CUDA 11.8 runtime is forward-compatible with CUDA 12.x host drivers, so this works on both current server (CUDA 11.4 driver) and target server (Quadro RTX 8000, CUDA 12.4 driver)
- Base image uses Python 3.11 (not 3.10 like conda env) — all pinned deps installed successfully, no compatibility issues
- Data must be mounted separately; if using symlinks to NAS, the NAS path must also be mounted in the container

**Files created:**
```
docker/
├── Dockerfile          # pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime + all pip deps
├── requirements.txt    # Pinned pip freeze (excluding torch/nvidia packages)
├── docker-compose.yml  # GPU-enabled service with volume mounts
├── .dockerignore       # Exclude data/code from build context
└── README.md           # Build/run instructions
```

**Image size:** 7.8 GB (base ~5.5 GB + deps ~2.3 GB)

---

## 2026-03-24 — Data migration to NAS (Prompt 29)

**What was done:**
- Moved large data files from local disk (`/home`, 20TB, was 100% full) to NAS mount (`/mnt/2021_NIA_data/`, 64TB, 11TB free)
- All moves used `mv` (not `cp`) to free space immediately. Created symlinks so all existing code paths work unchanged.

**What was moved:**
- `data/embeddings/*.pt` (CLS embeddings, ~2.8 GB) → NAS `cls/` + top-level symlinks
- `data/embeddings/token/` (pilot token embeddings, ~31 GB) → NAS `token/` symlink
- `data/datasets/coco/` (~18 GB, 118K train + 5K val images + annotations) → NAS symlink
- `data/datasets/flickr30k/` (~5 GB) → NAS symlink
- `data/datasets/imagenet/` (~7 GB) → NAS symlink

**Disk space freed: ~64 GB** (13 GB → 173 GB available on `/home`)

**Verification:**
- `torch.load('data/embeddings/coco_train_img.pt')` → shape [118287, 768] ✓
- 1-epoch baseline training: mR=12.6 (exact match with previous runs) ✓
- Token embeddings, dataset annotations, all accessible via symlinks ✓

**Symlink structure:**
```
data/embeddings/coco_train_img.pt → cls/coco_train_img.pt → /mnt/.../cls/coco_train_img.pt
data/embeddings/cls → /mnt/2021_NIA_data/bridge-anchors-data/embeddings/cls
data/embeddings/token → /mnt/2021_NIA_data/bridge-anchors-data/embeddings/token
data/datasets/coco → /mnt/2021_NIA_data/bridge-anchors-data/datasets/coco
data/datasets/flickr30k → /mnt/2021_NIA_data/bridge-anchors-data/datasets/flickr30k
data/datasets/imagenet → /mnt/2021_NIA_data/bridge-anchors-data/datasets/imagenet
```

**Note:** NAS access is over NFS — file I/O will be slower than local SSD. Training on pre-extracted embeddings (small files, loaded once) should be minimally affected. Embedding extraction (many small image reads) will be slower.

---

## 2026-03-23 — Token-Level BridgeAnchors Pilot (Prompt 28)

**What was done:**
- **Step 0 — Organization**: Created `data/embeddings/cls/` (symlinks to existing files) and `data/embeddings/token/` for new token-level data. New model in `src/models/token_bridge_anchors.py`. Added `--token-level` and `--token-pool` flags to train.py with fully independent code path. Reproducibility verified: baseline epoch 1 mR=12.64 (matches exactly).
- **Phase 1 — Extraction**: Created `scripts/extract_token_embeddings.py`. Extracted (10K, 257, 768) image tokens + (10K, 768) text CLS for COCO 10K subset, and full Flickr30k tokens. Sizes: COCO 10K → 7.4 GB, Flickr30k → 24 GB.
- **Phase 2 — Model**: `TokenBridgeAnchorAligner` processes (B, 257, 768) image tokens by computing per-token anchor similarities (B, 257, K), then aggregating via mean or max pooling → (B, K). Text side unchanged. Also handles (B, 768) CLS-only input via shape detection.
- **Phase 3 — Pilot**: 6 experiments on 10K COCO, Flickr30k eval.

**Results — first substantial improvement:**

| Model | Params | mR | Δ vs CLS BA |
|-------|--------|------|-------------|
| CLS BA K=128 | 197K | 11.58 | — |
| CLS LinearProj | 590K | 14.48 | +2.90 |
| Token BA K=128 mean | 197K | 13.82 | +2.24 |
| Token BA K=128 max | 197K | 13.41 | +1.83 |
| Token BA K=64 mean | 98K | 12.21 | +0.63 |
| Token BA K=256 mean | 393K | 15.00 | +3.42 |

**Key findings:**
- **Token BA K=128 mean gains +2.24 mR (19% relative) at identical param count** vs CLS BA. Patch tokens provide richer spatial info that anchors leverage.
- **Token BA K=256 mean (15.00) beats CLS LinearProjection (14.48)** with 33% fewer params. First time BA surpasses LP at the 10K data scale.
- **Mean > max** pooling (+0.41 mR) — mean averages spatial evidence more robustly than noisy max
- **More anchors help** with token input (K: 64→128→256 gives +1.61, +1.18 mR) — 257 tokens provide more spatial diversity for more anchors to specialize

**Timing:** Phase 1 ~25 min (extraction), Phase 3 ~30 min (6 training runs with batched eval)

**Issues:** Flickr30k token file is 24 GB — required batched eval (batch_size=64) to avoid OOM.

---

## 2026-03-23 — Steps 2/3/4: Sparse Gating, Per-Anchor Loss, FPS Init (Prompt 27)

**What was done:**
- **Step 2 — Sparse Gating**: Added `top_k` param to `BridgeAnchorAligner` with straight-through estimator. After computing cosine similarities, zeros out all but top-k per sample, then L2-normalizes. CLI: `--top-k`. Ran sweep: top_k={16, 32, 48, 64, 96} with K=128.
- **Step 3 — Per-Anchor Contrastive Loss**: Added `per_anchor_contrastive_loss(sim_img, sim_txt)` to losses.py — computes per-anchor Pearson correlation between image and text similarity vectors, penalizes low correlation. CLI: `--pa-lambda`. Ran sweep: pa_lambda={0.01, 0.05, 0.1, 0.5, 1.0}.
- **Step 4 — FPS Init**: Added `_compute_fps_anchors()` using Farthest Point Sampling (greedy, cosine distance) independently on image and text embeddings. CLI: `--init-method fps`. Ran fps vs random.
- All plug-and-play with zero impact when disabled.

**Results:**

| Method | mR | Δ |
|--------|------|------|
| Baseline | 23.66 | — |
| FPS init | 23.72 | +0.06 |
| pa=0.01 | 23.65 | -0.01 |
| pa=1.0 | 22.02 | -1.64 |
| top_k=96 | 12.85 | -10.81 |
| top_k=16 | 10.82 | -12.84 |

**Key findings:**
- **Step 2 is catastrophic**: All sparse gating levels destroy performance. Straight-through estimator fundamentally incompatible with the BridgeAnchors architecture — the forward/backward mismatch prevents learning under sparsity, and L2 normalization after masking amplifies noise in retained dimensions.
- **Step 3 is redundant**: InfoNCE already implicitly aligns per-anchor response patterns. Explicit Pearson correlation loss adds nothing (pa=0.01) or hurts (pa≥0.5) by over-constraining.
- **Step 4 gives marginal improvement**: FPS init (+0.06 mR) is the only positive result. Consistent with Exp C finding that init matters little at full data scale.
- **Meta-insight**: BridgeAnchors K=128 with random init and InfoNCE is a remarkably robust baseline. The simplicity of the method is its strength — auxiliary losses and architectural modifications provide no benefit at this scale.

**Timing:** ~20 min for all 11 training runs

**Issues:** None.

---

## 2026-03-23 — Intermediate Layer Alignment (Prompt 26)

**What was done:**
- **Phase 1 — CKA Analysis**: Created `src/eval/layer_cka.py` to compute cross-modal linear CKA between all (DINOv2 block, MPNet layer) pairs. Ran on 5,000 COCO samples. Used DINOv2's `get_intermediate_layers()` for CLS tokens from all 12 blocks + final, and MPNet's `output_hidden_states` for all 12 layers + final.
- **Phase 2 — Embedding Extraction**: Created `scripts/extract_intermediate_embeddings.py`. Extracted embeddings from top-3 layer pairs for COCO train (118K) and Flickr30k test (31K). 8 new .pt files in `data/embeddings/`.
- **Phase 3 — Training**: Trained BA K=128 on each of 3 layer pairs using custom config files with patched embedding paths.

**CKA heatmap key finding**: DINOv2 Block 9 dominates the top-4 pairs. Peak CKA = 0.586 (Block 9 × Layer 10). Baseline Final × Final CKA = 0.461.

**Results:**

| Layer Pair | CKA | mR | Δ |
|------------|------|------|-----|
| Final × Final (baseline) | 0.461 | 23.66 | — |
| Block 10 × Layer 10 | 0.514 | 17.82 | -5.84 |
| Block 9 × Layer 11 | 0.552 | 15.79 | -7.87 |
| Block 9 × Layer 10 | 0.586 | 15.72 | -7.94 |

**Key findings:**
- **Negative result**: All intermediate pairs underperform the baseline by 5.8–7.9 mR
- **Higher CKA predicts WORSE performance**: The pair with highest CKA (0.586) has the lowest mR (15.72). The inverse correlation suggests high CKA captures generic shared features, not discriminative semantics
- **Why**: Final layers contain the most compressed, task-relevant semantics. Intermediate layers retain positional/structural information that doesn't help retrieval. BridgeAnchors needs discriminative representations, not similar ones.
- **Practical conclusion**: Use final-layer embeddings. Cross-modal representational similarity is not the bottleneck for BridgeAnchors alignment.

**Timing:** Phase 1 ~5 min (CKA on 5K samples), Phase 2 ~35 min (embedding extraction), Phase 3 ~6 min (3 training runs)

**Issues:** None.

---

## 2026-03-23 — Step 1: Load Balancing Loss (Prompt 25)

**What was done:**
- Added `load_balancing_loss()` to `src/models/losses.py` — Switch Transformer-style: penalizes correlation between hard assignment frequency (p_k) and soft routing probability (f_k) per anchor
- Modified `BridgeAnchorAligner.forward()` to optionally return raw cosine similarities (before L2 norm) via `return_raw_sims=True` flag — backward compatible
- Added `--lb-lambda` CLI arg and `lb_lambda: 0.0` to config — zero impact when disabled
- Ran lambda sweep: {0.01, 0.05, 0.1, 0.5, 1.0} with K=128 on COCO 118K, seed=42
- Per-lambda anchor analysis: usage distribution, UMAP coverage, cross-modal correspondence

**Results:**

| lb_lambda | mR | Usage Std (img) | Change |
|-----------|------|----------------|--------|
| 0.0 | 23.66 | 320 | — |
| 0.01 | 23.66 | 320 | 0% |
| 0.05 | 23.66 | 318 | -1% |
| 0.1 | 23.66 | 316 | -1% |
| 0.5 | 23.66 | 301 | -6% |
| 1.0 | 23.65 | 283 | -12% |

**Key findings:**
- **Zero impact on retrieval**: All lambdas produce 23.65–23.66 mR — the LB loss is truly plug-and-play
- **Modest usage uniformity improvement**: Std drops 320→283 at lb=1.0 (~12%), but baseline already has 0 dead anchors and reasonable distribution
- **Cross-modal correspondence unaffected**: Category overlap 0.223–0.226, supercategory overlap 0.702–0.706 — all within noise
- **Why modest effect**: BridgeAnchors at K=128 in 768-d already has good anchor coverage. InfoNCE naturally encourages diversity. The LB loss addresses mild non-uniformity, not a severe problem. Would likely be more impactful at very high K or in lower-dim spaces.

**Timing:** ~10 min training (5 runs) + ~3 min analysis

**Issues:** None.

---

## 2026-03-23 — Direction A: comprehensive anchor analysis (Prompt 24)

**What was done:**
- Wrote `scripts/run_anchor_analysis.py` — comprehensive 5-part analysis of learned anchors from BridgeAnchors K=128 (Exp B, seed=42)
- Used COCO captions + instance annotations for semantic grounding
- Installed `umap-learn` for coverage visualization

**Analysis 1 — Nearest Neighbors with COCO metadata:**
- For each of 128 image/text anchors, found 10 nearest training samples with captions and categories
- Output: `nearest_neighbors.json` (full details) + `nearest_neighbors.md` (readable summary)
- Anchors show clear semantic specialization — e.g., some anchors attract person+sports images, others attract food+kitchen scenes

**Analysis 2 — Cross-Modal Anchor Correspondence:**
- Category overlap (Jaccard): mean 0.225, median 0.216
- Supercategory overlap: mean 0.703, median 0.727
- Paired image/text anchors strongly agree at the supercategory level — anchor k in image space and anchor k in text space point to the same broad semantic concepts

**Analysis 3 — Anchor Similarity Structure (Gram matrices):**
- Pearson r = 0.660 between off-diagonal elements of image and text Gram matrices
- Linear CKA = 0.859
- Frobenius diff = 7.90
- Anchors develop strongly parallel inter-anchor structure across modalities — the geometry of anchor relationships is preserved cross-modally

**Analysis 4 — Anchor Coverage (UMAP):**
- 128 anchors visualized among 2,000 training embeddings colored by COCO supercategory
- Anchors cluster in the center of the embedding space (densely connected semantic core) rather than spreading to the periphery
- Individual category clusters in the periphery are covered via similarity to nearby anchors

**Analysis 5 — Dead Anchor Detection:**
- Image: 128/128 active, usage range 322–1990 (median 899, uniform=924)
- Text: 128/128 active, usage range 185–2017 (median 841)
- No dead anchors — all 128 anchors are used in both modalities
- Usage distribution is non-uniform (Zipf-like) but healthy — some anchors cover larger regions

**Timing:** ~1 min total (UMAP is the bottleneck)

**Issues:** None.

---

## 2026-03-23 — PCA-reduced BridgeAnchors (Prompt 23)

**What was done:**
- Added `--pca-dim` CLI arg to `src/train.py` (default 0 = disabled)
- When enabled, computes PCA on training embeddings (768 → pca_dim), projects all datasets in-place before model construction
- No model changes needed — BridgeAnchors naturally adapts to the reduced input dimension
- Ran 2D sweep: pca_dim ∈ {32, 64, 128, 256} × K ∈ {32, 64, 128} (9 valid combos where K ≤ pca_dim)

**Results:**

| pca_dim | K | Params | mR | vs BA K=128 (197K, 23.66) |
|---------|---|--------|------|---------------------------|
| 32 | 32 | 2K | 2.53 | -21.1 |
| 64 | 64 | 8K | 6.43 | -17.2 |
| 128 | 64 | 16K | 12.70 | -11.0 |
| 128 | 128 | 33K | 13.10 | -10.6 |
| 256 | 64 | 33K | 17.97 | -5.7 |
| 256 | 128 | 66K | 19.22 | -4.4 |

**Key findings:**
- **PCA dim is the dominant factor** — increasing pca_dim has far more impact than increasing K. At K=32: d=32→256 gives 2.5→14.0 mR (5.6×). At d=256: K=32→128 gives 14.0→19.2 (1.4×)
- **Significant performance cost**: Best PCA-BA (d=256, K=128, 66K params) achieves 19.2 mR — 4.4 mR below standard BA K=128 (197K params, 23.7 mR). Not worthwhile.
- **PCA discards cross-modal signal**: PCA optimizes for variance preservation per modality, not cross-modal alignment. Low-variance dimensions dropped by PCA may carry important alignment information.
- **Standard BA dominates at comparable param counts**: BA K=32 (49K, 16.7 mR) beats PCA-BA d=128,K=128 (33K, 13.1 mR). Operating in full 768-d is more param-efficient for alignment.
- **Conclusion**: PCA reduction is not recommended — the 768-d embedding space is efficiently utilized by standard BridgeAnchors, and the reduced space loses too much signal.

**Timing:** ~12 min for 9 runs (including PCA computation)

**Issues:** None.

---

## 2026-03-23 — Idea 3: SpectralAligner implementation + K ablation (Prompt 22)

**What was done:**
- Implemented `SpectralAligner` in `src/models/spectral_align.py`:
  - Pre-computes PCA eigenvectors (top-K) on training embeddings for each modality (frozen buffers)
  - Projects embeddings to K-dim spectral coordinates via PCA
  - Learns a soft permutation matrix (temperature-scaled softmax, tau configurable) + K scaling factors to align image spectral coords to text
  - Hard argmax permutation at inference; soft permutation during training
  - Params: K² + K (e.g., K=128 → 16,512 params)
- Integrated into `src/train.py`: added `_compute_pca()`, `spectral_aligner` model branch, CLI choice
- Ran K ablation: K={4, 8, 16, 32, 64, 128, 256} on full COCO 118K, seed=42
- Also tested with lr={1e-3, 1e-2, 1e-1} and tau={0.1, 1.0} — none converged

**Results — FAILED:**

| K | Params | SpectralAligner mR | BridgeAnchors mR |
|---|--------|-------------------|-----------------|
| 4 | 20 | 0.03 | 0.52 |
| 8 | 72 | 0.04 | 3.42 |
| 16 | 272 | 0.07 | 9.79 |
| 32 | 1,056 | 0.12 | 16.69 |
| 64 | 4,160 | 0.23 | 21.71 |
| 128 | 16,512 | 0.34 | 23.66 |
| 256 | 65,792 | 0.50 | 24.02 |

**Why it failed:**
1. **Wrong assumption**: PCA axes of independently trained encoders (DINOv2 vs sentence-transformers) have no 1-to-1 correspondence — the cross-modal relationship is a dense linear transform, not a permutation
2. **Optimization difficulty**: K×K soft permutation has K! discrete solutions; temperature-scaled softmax is either too uniform (tau=1.0) or has vanishing gradients (tau=0.1)
3. **Train-eval mismatch**: soft permutation during training ≠ hard argmax at inference — model trains on blurred representation it never actually produces at test time
4. **Too constrained**: even a perfect permutation can only reorder axes, not mix them — fundamentally less expressive than BridgeAnchors' free anchor positions

**Key insight**: BridgeAnchors succeeds because it learns *where to measure* (free anchor positions in 768-d space) rather than *how to permute fixed measurements* (constrained axis reordering). BA at K=4 (6K params, mR=0.52) already matches SpectralAligner at K=256 (66K params, mR=0.50).

**Timing:** ~9 min for 7 runs + additional hyperparameter sweeps

**Issues:** None (clean negative result).

---

## 2026-03-23 — Direction B: orthogonal anchor regularization (Prompt 21)

**What was done:**
- Implemented orthogonal anchor regularization as a plug-and-play addition:
  - `src/models/losses.py`: added `anchor_orthogonality_loss()` — penalizes off-diagonal elements of normalized Gram matrix `A @ A.T` for image and text anchors independently
  - `src/train.py`: integrated into `train_one_epoch()` with `ortho_lambda` param; added `--ortho-lambda` CLI arg
  - `configs/default.yaml`: added `ortho_lambda: 0.0` (disabled by default)
- When `ortho_lambda=0`, model behaves identically to before (no code path changes)
- Ran K ablation: BridgeAnchors + ortho (λ=0.1) for K={4,8,16,32,64,128,256} on full COCO 118K, seed=42

**Results — null result:**

| K | BA mR | BA+ortho mR | Δ |
|---|-------|-------------|---|
| 4 | 0.52 | 0.52 | 0.00 |
| 8 | 3.42 | 3.44 | +0.02 |
| 16 | 9.79 | 9.76 | -0.03 |
| 32 | 16.69 | 16.70 | +0.01 |
| 64 | 21.71 | 21.71 | 0.00 |
| 128 | 23.66 | 23.67 | +0.01 |
| 256 | 24.02 | 24.03 | +0.01 |

**Key findings:**
- Orthogonal regularization at λ=0.1 has **zero measurable effect** — max Δ across all K is 0.03 mR, well within noise
- The two curves are indistinguishable in the plot
- **Why it doesn't help**: In 768-d space, K random unit vectors (even K=256) are already near-orthogonal (expected pairwise cosine ≈ 0). The regularization is solving a non-existent problem.
- Additionally, InfoNCE implicitly encourages anchor diversity — redundant anchors provide no discriminative signal
- Conclusion: simple orthogonal reg is not useful for BridgeAnchors in high-dimensional spaces. More structured regularization (e.g., encouraging semantic alignment) would be needed for improvement.

**Timing:** ~10 min for all 7 runs

**Issues:** None.

---

## 2026-03-23 — Experiment C: kmeans init + data efficiency (Prompt 20)

**What was done:**
- Added `'kmeans'` init method to `BridgeAnchorAligner` — runs scikit-learn K-means independently on image and text training embeddings to produce K centroids as initial anchors
- Code changes: `src/models/bridge_anchors.py` (accept `'kmeans'` in `_init_anchors`), `src/train.py` (new `_compute_kmeans_centroids` function, wired into `build_model`, added CLI choice)
- Ran BridgeAnchors K=128 with kmeans init across 6 data sizes {500, 1K, 5K, 10K, 50K, 118K}, seed=42
- Updated Experiment C with 4th line: CSV, results_summary.md, 4-line plot

**Results:**

| N | BA random | BA proto | BA kmeans | LP |
|---|-----------|----------|-----------|-----|
| 500 | 0.81 | 1.31 | 0.48 | 1.92 |
| 1,000 | 2.28 | 2.89 | 1.94 | 4.56 |
| 5,000 | 7.89 | 8.93 | 8.96 | 11.30 |
| 10,000 | 11.58 | 11.91 | 12.31 | 14.48 |
| 50,000 | 20.27 | 19.91 | 20.36 | 21.35 |
| 118,287 | 23.66 | 23.22 | 23.58 | 23.57 |

**Key findings:**
- K-means init has a distinctive crossover: **worst at tiny scales** (0.48 at N=500 — K=128 centroids from 500 points are just individual data points), **best BA variant at medium scales** (12.31 at 10K, 20.36 at 50K)
- The crossover happens around N=5K (~40 samples/cluster), where K-means starts finding real structure
- Init strategy ranking depends on data scale: prototype best at N<5K, kmeans best at 5K–50K, all converge at 118K
- No init strategy makes BridgeAnchors beat LinearProjection at small scales — LP's simpler parameterization is fundamentally easier to learn with limited data
- K-means adds ~5–30s overhead per run (sklearn on CPU), negligible vs training time

**Timing:** ~8 min for all 6 runs (including K-means clustering)

**Issues:** None.

---

## 2026-03-23 — Experiment C+: prototype init data efficiency (Prompt 19)

**What was done:**
- Ran BridgeAnchors K=128 with **prototype init** across 6 data sizes {500, 1K, 5K, 10K, 50K, 118K}, seed=42
- Evaluated all 6 runs on Flickr30k retrieval
- Added results to existing Experiment C data: updated CSV, results_summary.md, and regenerated plot with 3 lines

**Results:**

| N | BA random mR | BA proto mR | LP mR | Proto vs Random |
|---|-------------|-------------|-------|-----------------|
| 500 | 0.81 | 1.31 | 1.92 | +0.50 |
| 1,000 | 2.28 | 2.89 | 4.56 | +0.61 |
| 5,000 | 7.89 | 8.93 | 11.30 | +1.04 |
| 10,000 | 11.58 | 11.91 | 14.48 | +0.33 |
| 50,000 | 20.27 | 19.91 | 21.35 | -0.36 |
| 118,287 | 23.66 | 23.22 | 23.57 | -0.44 |

**Key findings:**
- Prototype init helps at small scales: +0.50 to +1.04 mR for N ≤ 5K (data-informed starting positions give anchors a head start)
- Advantage fades and reverses at large scales: at 50K and 118K, random init is slightly better (-0.36 and -0.44 mR)
- **Prototype init does NOT close the gap with LinearProjection** — LP still wins at all scales below 118K
- The hypothesis that prototype init would make BridgeAnchors more data-efficient than LP was not confirmed
- Practical takeaway unchanged: LP for small data, BridgeAnchors for abundant data

**Timing:** ~6 min for all 6 runs

**Issues:** None.

---

## 2026-03-21 — Experiment C: data efficiency (Prompt 18)

**What was done:**
- Ran Experiment C: BridgeAnchors (K=128) vs LinearProjection across 6 data sizes {500, 1K, 5K, 10K, 50K, 118K}, seed=42
- Evaluated all 12 runs on Flickr30k retrieval
- Generated results summary, CSV, and data efficiency plot

**Results:**

| N | BA (K=128, 197K) mR | LP (590K) mR | Difference |
|---|---------------------|--------------|------------|
| 500 | 0.81 | 1.92 | -1.11 |
| 1,000 | 2.28 | 4.56 | -2.28 |
| 5,000 | 7.89 | 11.30 | -3.41 |
| 10,000 | 11.58 | 14.48 | -2.90 |
| 50,000 | 20.27 | 21.35 | -1.08 |
| 118,287 | 23.66 | 23.57 | +0.09 |

**Key findings:**
- LinearProjection is more data-efficient at small scales — its simpler 768→768 linear map learns faster with limited examples
- BridgeAnchors has a steeper learning curve — the gap narrows from -3.41 (5K) to -1.08 (50K) to +0.09 (118K)
- At full scale (118K), BridgeAnchors matches LinearProjection with 3× fewer parameters
- The crossover happens around 100K pairs — below that, the simpler model wins
- Practical implication: use LinearProjection when data is scarce (<10K), BridgeAnchors when data is abundant (>50K)

**Timing:** ~8 min for all 12 runs

**Issues:** None.

---

## 2026-03-21 — Experiment A prototype init + Experiment B K ablation (Prompt 17)

**What was done:**
- Added BridgeAnchors with **prototype init** to Experiment A (3 seeds × Flickr30k + ImageNet eval)
- Ran Experiment B: K ablation with K = {4, 8, 16, 32, 64, 128, 256}, seed=42 only, with Flickr30k retrieval eval
- Generated results summary, CSV, and Mean Recall vs K plot for Experiment B

**Experiment A — prototype init results (mean across 3 seeds):**

| Model | Params | Flickr30k mR | ImageNet top-1 | ImageNet top-5 |
|-------|--------|-------------|----------------|----------------|
| BridgeAnchors (K=32, prototype) | 49,152 | 16.88 | 11.32% | 26.19% |
| BridgeAnchors (K=32, random) | 49,152 | 16.79 | 11.25% | 26.52% |

Prototype init provides a marginal improvement (~0.1 mR, ~0.07% top-1) — within noise. With sufficient training data (118K pairs), initialization strategy matters little at K=32.

**Experiment B — K ablation results (seed=42):**

| K | Params | Flickr30k mR |
|---|--------|-------------|
| 4 | 6,144 | 0.52 |
| 8 | 12,288 | 3.42 |
| 16 | 24,576 | 9.79 |
| 32 | 49,152 | 16.69 |
| 64 | 98,304 | 21.71 |
| 128 | 196,608 | 23.66 |
| 256 | 393,216 | 24.02 |

**Key findings:**
- BridgeAnchors **surpasses LinearProjection** (mR=23.52) at K=128 (mR=23.66) with 3× fewer parameters (197K vs 590K)
- Strong monotonic improvement: 46× mR increase from K=4 to K=256
- Diminishing returns above K=128: K=128→256 adds only +0.36 mR, while K=64→128 adds +1.95
- K=128 is the sweet spot for parameter efficiency — matches projection baselines with fewer params
- The K=32 performance gap in Experiment A was due to representation dimensionality bottleneck, not fundamental limitations of the anchor-distance approach

**Timing:** ~5 min for Exp A prototype runs, ~10 min for Exp B (7 K values × train + eval)

**Issues:** None.

---

## 2026-03-21 — Experiment A: main comparison (Prompt 16)

**What was done:**
- Fixed CUDA OOM bug in `src/eval/retrieval.py` that crashed all 12 runs in the background attempt
- Re-ran all 12 Experiment A runs interactively: 4 models × 3 seeds
- Evaluated all models on Flickr30k retrieval (R@1/5/10) and ImageNet zero-shot (top-1/5)
- Generated `experiments/exp_a_main/results_summary.md` and `results.csv`
- Reverted `scripts/run_exp_a.sh` to the clean Prompt 7 version

**Bug fix — CUDA OOM in retrieval eval:**
- **Root cause**: `evaluate_retrieval()` computed the 31,783×31,783 similarity matrix and then called `argsort()` on it, all on GPU. The similarity matrix alone is 3.8 GB float32, and argsort allocates a same-sized int64 index tensor (7.6 GB). Combined with model weights and embeddings, this exceeded the A40's 44.6 GB.
- **Fix**: Moved similarity computation and rank calculation to CPU. Replaced `argsort`-based ranking with a counting approach (`_get_gt_ranks`): for each query, count how many items score higher than the ground truth. This avoids allocating the N×N int64 index tensor entirely. Also fixed `compute_retrieval_ranks()` with the same approach.

**Results (mean across 3 seeds):**

| Model | Params | Flickr30k mR | ImageNet top-1 | ImageNet top-5 |
|-------|--------|-------------|----------------|----------------|
| LinearProjection | 589,824 | 23.52 | 17.39% | 35.85% |
| MLPProjection | 393,472 | 20.44 | 15.00% | 32.57% |
| BridgeAnchors (K=32) | 49,152 | 16.79 | 11.25% | 26.52% |
| FixedRelativeRep (K=32) | 0 | 0.30 | ~0.1% | ~0.5% |

**Key observations:**
- LinearProjection strongest overall — expected with 12× more parameters
- BridgeAnchors achieves meaningful alignment with only 49K params — concept validated
- FixedRelativeRep at chance level confirms learnable anchors are essential
- All models very stable across seeds (std < 0.3 on all metrics)
- Each training run took ~75s (20 epochs on 118K COCO pairs) — extremely fast since embeddings are pre-extracted

**Timing:** ~15 min total for all 12 runs + evaluations

**Issues:**
- Background tmux run failed on all 12 runs due to the OOM bug. Fixed and re-ran interactively.

---

## 2026-03-21 — Extract Flickr30k and ImageNet embeddings (Prompt 15)

**What was done:**
- Extracted Flickr30k test embeddings: 31,783 image-caption pairs
  - Images: ~1.5 min (128 batch, 8 workers), Text: ~10s → `flickr30k_test_img.pt` + `flickr30k_test_txt.pt`
- Extracted ImageNet val embeddings: 50,000 images + 1,000 class texts
  - Images: ~3 min (128 batch, 8 workers), Text: <1s → `imagenet_val_img.pt` + `imagenet_val_txt.pt` + `imagenet_val_labels.pt`
- Verified all 9 embedding files: shapes, dtypes, L2 normalization — all checks passed
- Total embedding storage: 1.1 GB in `data/embeddings/`

**Verification results:**
| File | Shape | dtype | L2 norm |
|------|-------|-------|---------|
| coco_train_img | (118287, 768) | float32 | 1.0000 |
| coco_train_txt | (118287, 768) | float32 | 1.0000 |
| coco_val_img | (5000, 768) | float32 | 1.0000 |
| coco_val_txt | (5000, 768) | float32 | 1.0000 |
| flickr30k_test_img | (31783, 768) | float32 | 1.0000 |
| flickr30k_test_txt | (31783, 768) | float32 | 1.0000 |
| imagenet_val_img | (50000, 768) | float32 | 1.0000 |
| imagenet_val_txt | (1000, 768) | float32 | 1.0000 |
| imagenet_val_labels | (50000,) | int64 | — |

**Issues:**
- None. All embeddings match expected shapes exactly.

---

## 2026-03-21 — ImageNet validation set setup (Prompt 14)

**What was done:**
- Extracted `~/ILSVRC2012_img_val.tar` (6.3 GB) to `data/datasets/imagenet/val/` — 50,000 flat JPEG files
- Downloaded official `valprep.sh` from soumith/imagenetloader.torch (50,000 mv + 1,000 mkdir commands)
- Ran valprep.sh to reorganize into standard class-subdirectory structure: `val/nXXXXXXXX/ILSVRC2012_val_NNNNNNN.JPEG`
- Verified result: 1,000 subdirectories, 50 images each, 50,000 total — correct
- Confirmed `extract_embeddings.py` `_collect_imagenet_val_images()` will detect this layout (Layout 1: class subdirectories, sorted synset order → class indices 0–999)
- Deleted original tar (~6.3 GB) and valprep.sh to save disk space

**Key decisions:**
- Used the canonical valprep.sh from soumith's GitHub rather than rolling our own ground truth mapping — this is the de facto standard used by PyTorch/torchvision and ensures synset ordering matches torchvision's ImageNet class index convention
- Class-subdirectory layout chosen over flat+ground_truth.txt because it's the standard PyTorch convention and what torchvision.datasets.ImageFolder expects

**Issues:**
- None

---

## 2026-03-21 — Download Flickr30k dataset (Prompt 13)

**What was done:**
- Found existing Kaggle API token at `~/.kaggle/kaggle.json` (permissions already 600)
- Installed `kaggle` CLI (v1.7.4.5) in the bridge-anchors conda env
- Searched Kaggle and selected `eeshawn/flickr30k` (4.1 GB, usability 1.0) over `hsankesara/flickr-image-dataset` (8.7 GB, double-nested dirs, no captions file)
- Downloaded and extracted to `data/datasets/flickr30k/`:
  - `flickr30k_images/`: 31,783 JPEG images
  - `captions.txt`: original CSV format from Kaggle
  - `results_20130124.token`: converted to expected `filename#idx\tcaption` format (158,915 lines, 5 captions per image)
- Verified `extract_embeddings.py` correctly parses all 31,783 image-caption pairs
- Deleted `flickr30k.zip` to save ~4.1 GB disk space

**Key decisions:**
- Chose `eeshawn/flickr30k` over `hsankesara/flickr-image-dataset` because: half the download size, cleaner directory structure (single-level `flickr30k_images/` vs double-nested), includes captions file, and 1.0 usability rating
- Converted Kaggle's CSV caption format (`image_name,comment_number,comment`) to the standard `.token` format (`filename#idx\tcaption`) that our extraction code expects — named it `results_20130124.token` to match the primary filename the code searches for
- Kept the original `captions.txt` as well (only 13 MB) for reference

**Issues:**
- None. Download took ~3 seconds at 1.7 GB/s.

---

## 2026-03-21 — Extract COCO embeddings (Prompt 12)

**What was done:**
- Extracted COCO train embeddings (118,287 image-caption pairs) using DINOv2 ViT-B/14 (images) and all-mpnet-base-v2 (text)
- Extracted COCO val embeddings (5,000 image-caption pairs) in the same run session
- Saved 4 files to `data/embeddings/`: `coco_train_img.pt`, `coco_train_txt.pt`, `coco_val_img.pt`, `coco_val_txt.pt`
- Verified all tensors: correct shapes (N, 768), float32, mean L2 norm = 1.0000

**Timings:**
- DINOv2 model download (first run, cached for future): ~2s (from torch.hub cache)
- DINOv2 weights download: ~1s (329 MB, already cached)
- COCO train images (118,287 @ batch_size=128, 8 workers): ~6.5 minutes (~2.3 batches/sec on A40)
- COCO train text (118,287 @ batch_size=256): ~30 seconds
- COCO val images (5,000 @ batch_size=128): ~20 seconds
- COCO val text (5,000 @ batch_size=256): ~1 second
- Total train+val extraction: ~8 minutes

**Key details:**
- Used `--batch-size-img 128 --num-workers 8` for throughput (A40 handles 128 images easily)
- xFormers not installed — DINOv2 warns but works fine without it (standard attention fallback)
- Train embeddings: 347 MB each (img + txt), val: 15 MB each
- Total embedding storage: 723 MB

**Issues:**
- None

---

## 2026-03-21 — Dataset download and setup (Prompt 11)

**What was done:**
- Downloaded COCO 2017 dataset to `data/datasets/coco/`:
  - `train2017/`: 118,287 images (19 GB) — verified exact count
  - `val2017/`: 5,000 images (788 MB) — verified exact count
  - `annotations/captions_train2017.json`: 118,287 images, 591,753 captions
  - `annotations/captions_val2017.json`: 5,000 images, 25,014 captions
- Verified zip integrity before extraction, deleted zips after to save ~20GB
- Created `data/datasets/flickr30k/README.md` with download instructions (Kaggle or official form)
- Created `data/datasets/imagenet/README.md` with download instructions (academic access required)
- Created directory structures for Flickr30k and ImageNet placeholders
- Searched machine for existing datasets: found only `tiny-imagenet-200` at `/mnt/2021_NIA_data/tiny-imagenet-200/` (200 classes, not suitable for standard eval)

**Key decisions:**
- Used `wget -c` for resume support on the 18GB COCO train download — download was interrupted/stalled once and resumed cleanly
- Deleted zip files after verified extraction to conserve disk (253GB remaining on /home)
- Flickr30k and ImageNet require manual download (access restrictions) — created READMEs with clear instructions and expected directory layouts
- Noted tiny-imagenet-200 existence in ImageNet README but marked as unsuitable for standard zero-shot eval

**Issues:**
- COCO train2017 download speed fluctuated significantly (1-10 MB/s), stalled once at 45% and needed restart with `wget -c` resume. Total download time ~35 minutes.
- Flickr30k and ImageNet not available on this machine — manual download required

---

## 2026-03-20 — Comprehensive integration test (Prompt 10)

**What was done:**
- Ran full integration test with dummy data (256 train, 64 Flickr-like, 128 ImageNet-like with 10 classes)
- Tested all 6 model variants through training, retrieval eval, zero-shot eval, and anchor analysis:
  1. BridgeAnchorAligner (random init, K=32) — 49,152 params
  2. BridgeAnchorAligner (prototype init, K=32) — 49,152 params
  3. LinearProjection — 589,824 params
  4. MLPProjection (hidden=256) — 393,472 params
  5. FixedRelativeRep (random anchors, K=32) — 0 params
  6. FixedRelativeRep (prototype anchors, K=32) — 0 params
- Verified loss decreases over 3 epochs for all trainable models
- Verified output shapes at every stage (b_img, b_txt, retrieval metrics, zero-shot accuracy, anchor analysis tensors)
- Ran CLI integration: train.py -> eval.retrieval -> eval.zeroshot -> eval.anchor_analysis for all 4 model types
- All tests passed on GPU (NVIDIA A40, CUDA 11.8, PyTorch 2.4.1)

**Key results (dummy data — confirms pipeline correctness, not model quality):**
- BridgeAnchors (random): loss 7.00 → 3.18, mR=12.5, top1=10.2%
- BridgeAnchors (prototype): loss 1.20 → 0.76, mR=10.9, top1=3.9% (lower starting loss due to data-informed init)
- LinearProjection: loss 4.31 → 1.46, mR=15.9, top1=8.6%
- MLPProjection: loss 4.26 → 1.88, mR=11.2, top1=7.0%
- FixedRelativeRep (random): no training, mR=6.2, top1=8.6%
- FixedRelativeRep (prototype): no training, mR=10.2, top1=6.2%
- Anchor analysis: NN indices (32,5), cross-modal overlap computed, structure matrices (32,32), CKA ~0.96, class alignment (32, num_classes)

**Issues:**
- None. All 6 models × all evaluation modules × both direct API and CLI paths pass without errors.

---

## 2026-03-20 — Environment lock-down and reproducibility (Prompt 9)

**What was done:**
- Exported full conda environment to `environment.yaml` (exact reproduction) with prefix path stripped
- Created `environment_minimal.yaml` with just Python 3.10 + pip install instructions as comments
- Updated `requirements.txt` with exact pinned versions, grouped by purpose (Core ML, Numerical, Data, Visualization, Logging), with torch/torchvision excluded (installed separately via `--index-url`)
- Created `README.md` with: environment setup (two options: conda+pip or full env), PyTorch CUDA version selection table, three documented gotchas (NumPy <2, transformers ≥5 requires torch ≥2.4, driver vs toolkit CUDA versions), installation verification command, and quick-start guide

**Key decisions:**
- torch/torchvision deliberately excluded from requirements.txt — they require a system-specific `--index-url` for CUDA, so including them would install CPU-only versions and break GPU support
- environment.yaml is the "exact reproduction" path; requirements.txt is the "cross-platform" path that accommodates different CUDA versions
- environment_minimal.yaml kept as a minimal scaffold with comments pointing to the README for actual install steps — avoids a half-working conda env that's missing CUDA torch
- Three known gotchas documented from issues we actually encountered during setup, not theoretical

**Issues:**
- None

---

## 2026-03-20 — Environment setup (Prompt 8)

**What was done:**
- Detected system: 2× NVIDIA A40 (44.6 GB each), Driver 470.256.02 (CUDA 11.4 max), nvcc 11.6, Python 3.9.12
- Created conda environment `bridge-anchors` with Python 3.10.20
- Installed PyTorch 2.4.1+cu118 with CUDA 11.8 runtime (bundled, works despite driver reporting 11.4)
- Installed all dependencies: transformers 5.3.0, sentence-transformers 5.3.0, numpy 1.26.4, torchvision 0.19.1, etc.
- Ran full compatibility check: CUDA works, all imports pass, GPU forward+backward verified
- Added Environment Setup section to CLAUDE.md

**Key decisions:**
- Used cu118 (CUDA 11.8) PyTorch wheels — the CUDA runtime is bundled in the wheel, so it works even though the driver only reports CUDA 11.4 as the max toolkit version. Verified with actual GPU tensor operations.
- Started with PyTorch 2.0.1 but had to upgrade to 2.4.1 because `transformers>=5.0` and `sentence-transformers>=5.0` require PyTorch ≥ 2.4
- Pinned numpy<2 (1.26.4) because PyTorch 2.4.1 cu118 wheels were compiled against NumPy 1.x

**Issues resolved:**
- NumPy 2.x incompatibility: pip initially installed numpy 2.2.6 which caused `_ARRAY_API not found` error with PyTorch. Fixed by pinning `numpy<2`.
- transformers 5.3.0 requires PyTorch ≥ 2.4: `NameError: name 'nn' is not defined` in `transformers/integrations/accelerate.py` with PyTorch 2.0.1. Fixed by upgrading to PyTorch 2.4.1.

---

## 2026-03-20 — Experiment scripts and configs (Prompt 7)

**What was done:**
- Created `scripts/run_exp_a.sh` — Experiment A: main comparison (4 models × 3 seeds = 12 runs), with Flickr30k + ImageNet eval on best checkpoints
- Created `scripts/run_exp_b.sh` — Experiment B: K ablation (7 K values × 3 seeds = 21 runs)
- Created `scripts/run_exp_c.sh` — Experiment C: data efficiency (2 models × 6 sample counts × 3 seeds = 36 runs), including full-dataset runs with no `--num-samples`
- Created `scripts/run_exp_d.sh` — Experiment D: fixed vs learnable (4 strategies × 3 seeds = 12 runs)
- Created `scripts/run_all.sh` — master script that runs A→B→C→D sequentially with `--skip-X` and `--only X` flags, embedding-existence check, and total runtime display
- Created `scripts/collect_results.py` — scans checkpoints, extracts metrics/config, prints formatted table and writes CSV to `results/all_results.csv`
- Created `configs/exp_d_fixed_proto.yaml` — config variant with `init_method: prototype` for fixed-prototype anchor strategy
- Updated `src/train.py` `build_model()` to support `init_method` for `fixed_relative_rep` — when `prototype`, uses cluster-mean prototypes instead of random sample selection

**Key decisions:**
- 81 total training runs across all experiments, matching IMPLEMENTATION.md spec exactly
- All scripts use CLI overrides against `configs/default.yaml` rather than per-run config files — keeps configuration DRY
- Experiment D strategy 2 (fixed prototype) required both a config change and a train.py code change — `FixedRelativeRep` now respects `init_method` to choose between random sample selection and prototype-based anchors
- Exp C handles the "118K" (full dataset) case by omitting `--num-samples` entirely, letting the config's `null` take effect
- Each run logs to both `experiments/exp_X/` (stdout tee) and `results/logs/` (TensorBoard)
- `collect_results.py` reads metrics from `best.pt` checkpoint files, falling back to `latest.pt`
- `run_all.sh` checks for COCO embeddings before starting and warns (but continues) if Flickr30k is missing

**Issues:**
- None

---

## 2026-03-20 — Evaluation modules: retrieval, zero-shot, anchor analysis (Prompt 6)

**What was done:**
- Enhanced `src/eval/retrieval.py` — added batched bridging for large datasets, `compute_retrieval_ranks()` for detailed rank analysis, and standalone CLI (`python -m src.eval.retrieval --checkpoint ...`)
- Implemented `src/eval/zeroshot.py` — ImageNet zero-shot classification with top-1 and top-5 accuracy, batched image bridging, `per_class_accuracy()` for fine-grained analysis, and standalone CLI
- Implemented `src/eval/anchor_analysis.py` — Direction A with three analyses: nearest neighbours, anchor similarity structure, and class alignment; includes three visualisation functions and standalone CLI
- Created `src/eval/_utils.py` — shared helpers: `get_model_device()` (handles param-free models like FixedRelativeRep) and `load_model_from_checkpoint()` (reconstructs any model type from checkpoint)

**Key decisions:**
- Zero-shot uses separate bridging for images vs class texts since they have different counts (N vs C). Dummy tensors are passed for the unused modality since all models require both inputs — this is correct because the forward pass normalises each modality independently
- Anchor analysis extracts anchors via `_get_anchors()` helper that works for both `BridgeAnchorAligner` (nn.Parameter) and `FixedRelativeRep` (buffer)
- Three quantitative metrics for anchor structure: Frobenius norm of difference, Pearson correlation on upper-triangle, and linear CKA — together they measure whether anchors maintain consistent inter-relationships across modalities
- Class alignment uses softmax entropy to measure anchor specialisation (lower = more concept-specific)
- Cross-modal NN overlap uses Jaccard similarity between neighbour index sets of corresponding image/text anchors
- Retrieval batching only applies to the bridging step; the full (N, N) similarity matrix is computed at once since Flickr30k (31K) fits in memory
- `_utils.py` centralises checkpoint loading with full model reconstruction, avoiding code duplication across eval CLIs
- Verified train.py still works after retrieval.py import refactor

**Issues:**
- None

---

## 2026-03-20 — Training script and retrieval evaluation (Prompt 5)

**What was done:**
- Implemented `src/train.py` — full training loop with config loading, model construction, dataloading, optimisation, evaluation, checkpointing, and logging
- Implemented `src/eval/retrieval.py` — image-text retrieval evaluation with R@1/5/10 for both directions plus mean recall
- All four model types tested end-to-end: `bridge_anchors`, `linear_projection`, `mlp_projection`, `fixed_relative_rep`

**Key decisions:**
- CLI overrides for all major hyperparameters (`--model`, `--num-anchors`, `--lr`, `--epochs`, `--batch-size`, `--num-samples`, `--seed`, `--experiment-name`) so experiment scripts can reuse one config file
- Linear warmup + cosine annealing scheduler, stepping per optimiser step (not per epoch) for smooth warmup with `SequentialLR`
- Gradient clipping (default 1.0) via `clip_grad_norm_` for training stability
- `FixedRelativeRep` special-cased: detected as zero-parameter model, runs eval only and exits
- `prototype` init computes prototypes via random partition → cluster means (cheap K-means approximation)
- Validation loss computed on held-out 5% split (from `PairedEmbeddingDataset` split='val') every epoch
- Flickr30k retrieval eval is optional — gracefully skipped with warning if embeddings don't exist yet
- TensorBoard always enabled; W&B enabled only when `WANDB_PROJECT` env var is set
- Checkpoints save full state (model, optimiser, scheduler, metrics, config) as `latest.pt` + `best.pt` (by mean recall)
- Reproducibility: `seed_everything()` covers Python, NumPy, PyTorch, cuDNN deterministic mode, and DataLoader generator seed

**Issues:**
- None

---

## 2026-03-20 — Dataset loaders for training and evaluation (Prompt 4)

**What was done:**
- Implemented `src/data/coco_dataset.py` — `PairedEmbeddingDataset` class for loading pre-extracted `.pt` embedding pairs
- Implemented `src/data/eval_datasets.py` — `Flickr30kEmbeddings` and `ImageNetEmbeddings` loader classes

**Key decisions:**
- `PairedEmbeddingDataset` supports three composable features: train/val splitting (deterministic permutation-based), subsampling (for Experiment C data efficiency), and `get_all()` for bulk access. Split is applied *before* subsampling so subsamples are drawn only from the correct partition
- Train/val split uses a fixed `torch.randperm` with seed so train and val never overlap across instantiations
- Subsampling warns and clamps if `num_samples` exceeds available data after split
- Eval loaders are plain classes (not `Dataset` subclasses) since evaluation runs on full tensors in one pass — no need for batched iteration via DataLoader
- `ImageNetEmbeddings` validates label bounds against class count at load time to catch data mismatches early
- Both eval loaders match the file naming from `extract_embeddings.py` and `configs/default.yaml`

**Issues:**
- None

---

## 2026-03-20 — Core model implementation (Prompt 3)

**What was done:**
- Implemented `src/models/bridge_anchors.py` — `BridgeAnchorAligner` with learnable anchor points, supporting both `'random'` and `'prototype'` initialization
- Implemented `src/models/baselines.py` — `LinearProjection`, `MLPProjection`, and `FixedRelativeRep` baseline models
- Implemented `src/models/losses.py` — symmetric InfoNCE loss with configurable temperature
- All models share a common `(b_img, b_txt)` return interface
- Added development process rule to CLAUDE.md

**Key decisions:**
- All four models return the same `tuple[Tensor, Tensor]` shape `(B, K)` or `(B, dim_txt)` for interchangeable use in the training loop
- `BridgeAnchorAligner` normalizes anchors, inputs, *and* outputs (three-stage normalization per the spec)
- `FixedRelativeRep` uses `register_buffer` for anchors (no gradients, but included in `state_dict` for checkpointing)
- `MLPProjection` uses ReLU activation in the bottleneck, no bias on final layer to match `LinearProjection` convention
- `prototype` init accepts pre-computed prototype tensors; the caller is responsible for computing them from data

**Issues:**
- None

---

## 2026-03-20 — Embedding extraction script (Prompt 2)

**What was done:**
- Implemented `src/data/extract_embeddings.py` — full pipeline for extracting embeddings from DINOv2 ViT-B/14 (image) and all-mpnet-base-v2 (text)
- Supports COCO train/val, Flickr30k test, and ImageNet val datasets
- Includes argparse CLI, progress bars (tqdm + sentence-transformers built-in), batch processing, structured logging
- Outputs match `configs/default.yaml` paths exactly (e.g., `coco_train_img.pt`, `flickr30k_test_img.pt`)

**Key decisions:**
- Resume-friendly: skips datasets whose `.pt` files already exist (overridable with `--force`)
- COCO: uses first caption per image, sorted by image ID for reproducibility
- Flickr30k: auto-detects common directory layouts (`images/`, `flickr30k_images/`)
- ImageNet: supports both class-subdirectory and flat+ground-truth layouts; also saves `imagenet_val_labels.pt` for downstream accuracy eval
- ImageNet class text embeddings use `"a photo of a {class}"` prompt template
- Corrupt/missing images produce a logged warning + black-image placeholder (batch doesn't break)
- All embeddings are L2-normalized at extraction time

**Issues:**
- None

---

## 2026-03-20 — Project initialization (Prompt 1)

**What was done:**
- Created full directory structure per CLAUDE.md spec: `src/{models,data,eval,utils}`, `scripts/`, `configs/`, `data/{embeddings,datasets}`, `experiments/exp_{a,b,c,d}_*`, `results/`
- Added empty `__init__.py` files for all Python packages
- Created `requirements.txt` with all dependencies (torch, torchvision, transformers, sentence-transformers, datasets, PyYAML, tqdm, numpy, matplotlib, seaborn, tensorboard, scipy)
- Created `configs/default.yaml` with all hyperparameters from the implementation spec (K=32, batch_size=256, lr=1e-3, weight_decay=1e-4, 20 epochs, temperature=0.07, cosine scheduler, 2 warmup epochs)
- Created shell scripts (`extract_embeddings.sh`, `train.sh`, `eval.sh`) and made them executable
- Added `.gitignore` (datasets, .pt files, checkpoints, __pycache__, venvs)
- Added `.gitkeep` files for empty directories

**Key decisions:**
- Placeholder `.py` files created empty — implementation deferred to subsequent prompts
- Shell scripts are thin wrappers around the Python CLI entry points
- `.gitignore` excludes raw data and extracted embeddings (large binary files)

**Issues:**
- None
