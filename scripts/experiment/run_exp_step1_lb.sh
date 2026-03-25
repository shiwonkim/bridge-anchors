#!/bin/bash
# Step 1: Load Balancing Loss — Lambda sweep
# lb_lambda: {0.01, 0.05, 0.1, 0.5, 1.0}, K=128, full COCO 118K, seed=42

set -e

LAMBDAS=(0.01 0.05 0.1 0.5 1.0)
EXP_DIR="experiments/exp_step1_lb_loss"

for LB in "${LAMBDAS[@]}"; do
    LB_LABEL=$(echo "$LB" | tr '.' 'p')
    echo "=== Training BridgeAnchors K=128, lb_lambda=${LB} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors \
        --num-anchors 128 \
        --init-method random \
        --lb-lambda ${LB} \
        --seed 42 \
        --experiment-name "exp_lb_${LB_LABEL}" \
        2>&1 | tee "${EXP_DIR}/train_lb${LB_LABEL}_s42.log"
done

echo "=== All LB training runs complete ==="
