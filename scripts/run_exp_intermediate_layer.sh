#!/bin/bash
# Phase 3: Train BridgeAnchors K=128 on top-3 intermediate layer pairs
# Uses custom embedding paths via config overrides

set -e

EXP_DIR="experiments/exp_intermediate_layer"

# Pair 1: DINOv2 Block 9 × MPNet Layer 10 (CKA=0.586)
echo "=== Training on Block9 × Layer10 ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --seed 42 \
    --experiment-name "exp_intlayer_b9_l10" \
    2>&1 | tee "${EXP_DIR}/train_b9_l10.log"

# Pair 2: DINOv2 Block 9 × MPNet Layer 11 (CKA=0.552)
echo "=== Training on Block9 × Layer11 ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --seed 42 \
    --experiment-name "exp_intlayer_b9_l11" \
    2>&1 | tee "${EXP_DIR}/train_b9_l11.log"

# Pair 3: DINOv2 Block 10 × MPNet Layer 10 (CKA=0.514)
echo "=== Training on Block10 × Layer10 ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --seed 42 \
    --experiment-name "exp_intlayer_b10_l10" \
    2>&1 | tee "${EXP_DIR}/train_b10_l10.log"

echo "=== All intermediate layer training runs complete ==="
