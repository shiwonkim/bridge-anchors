#!/bin/bash
# Experiment B2: BridgeAnchors + orthogonal regularization, K ablation
# K values: 4, 8, 16, 32, 64, 128, 256
# ortho_lambda=0.1, full COCO 118K, seed=42

set -e

K_VALUES=(4 8 16 32 64 128 256)
EXP_DIR="experiments/exp_b2_ortho_k_ablation"

for K in "${K_VALUES[@]}"; do
    echo "=== Running BridgeAnchors+ortho K=${K} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors \
        --num-anchors ${K} \
        --init-method random \
        --ortho-lambda 0.1 \
        --seed 42 \
        --experiment-name "exp_b2_ortho_k${K}" \
        2>&1 | tee "${EXP_DIR}/exp_b2_ortho_k${K}_s42.log"
done

echo "=== All ortho K ablation runs complete ==="
