#!/bin/bash
# Steps 2, 3, 4: Sparse Gating, Per-Anchor Contrastive Loss, FPS Init
# All K=128, COCO 118K, seed=42

set -e

# --- Step 2: Sparse Gating (top-k sweep) ---
echo "========== Step 2: Sparse Gating =========="
mkdir -p experiments/exp_step2_sparse_gating

for TK in 16 32 48 64 96; do
    echo "=== top_k=${TK} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors --num-anchors 128 --seed 42 \
        --top-k ${TK} \
        --experiment-name "exp_step2_topk${TK}" \
        2>&1 | tee "experiments/exp_step2_sparse_gating/topk${TK}.log"
done
# top_k=0 (all) is the baseline — already in exp_b_k128_s42

# --- Step 3: Per-Anchor Contrastive Loss (lambda sweep) ---
echo "========== Step 3: Per-Anchor Contrastive Loss =========="
mkdir -p experiments/exp_step3_per_anchor

for PA in 0.01 0.05 0.1 0.5 1.0; do
    PA_LABEL=$(echo "$PA" | tr '.' 'p')
    echo "=== pa_lambda=${PA} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors --num-anchors 128 --seed 42 \
        --pa-lambda ${PA} \
        --experiment-name "exp_step3_pa${PA_LABEL}" \
        2>&1 | tee "experiments/exp_step3_per_anchor/pa${PA_LABEL}.log"
done

# --- Step 4: FPS Init ---
echo "========== Step 4: FPS Init =========="
mkdir -p experiments/exp_step4_fps_init

echo "=== fps init ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --seed 42 \
    --init-method fps \
    --experiment-name "exp_step4_fps" \
    2>&1 | tee "experiments/exp_step4_fps_init/fps.log"

echo "========== All Steps 2/3/4 complete =========="
