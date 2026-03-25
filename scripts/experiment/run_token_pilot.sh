#!/bin/bash
# Token-level BridgeAnchors pilot experiments
# All trained on 10K COCO subset, evaluated on Flickr30k
# seed=42, 20 epochs

set -e

EXP_DIR="experiments/exp_token_level_pilot"

# --- CLS-only baselines (trained on same 10K subset) ---
echo "=== CLS BA K=128 (10K baseline) ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors --num-anchors 128 --seed 42 \
    --num-samples 10000 \
    --experiment-name "exp_token_pilot_cls_ba128" \
    2>&1 | tee "${EXP_DIR}/cls_ba128.log"

echo "=== CLS LinearProjection (10K baseline) ==="
python -m src.train \
    --config configs/default.yaml \
    --model linear_projection --seed 42 \
    --num-samples 10000 \
    --experiment-name "exp_token_pilot_cls_lp" \
    2>&1 | tee "${EXP_DIR}/cls_lp.log"

# --- Token-level experiments ---
echo "=== Token BA K=128, mean pool ==="
python -m src.train \
    --config configs/default.yaml \
    --num-anchors 128 --seed 42 \
    --token-level --token-pool mean \
    --experiment-name "exp_token_pilot_mean_k128" \
    2>&1 | tee "${EXP_DIR}/token_mean_k128.log"

echo "=== Token BA K=128, max pool ==="
python -m src.train \
    --config configs/default.yaml \
    --num-anchors 128 --seed 42 \
    --token-level --token-pool max \
    --experiment-name "exp_token_pilot_max_k128" \
    2>&1 | tee "${EXP_DIR}/token_max_k128.log"

echo "=== Token BA K=64, mean pool ==="
python -m src.train \
    --config configs/default.yaml \
    --num-anchors 64 --seed 42 \
    --token-level --token-pool mean \
    --experiment-name "exp_token_pilot_mean_k64" \
    2>&1 | tee "${EXP_DIR}/token_mean_k64.log"

echo "=== Token BA K=256, mean pool ==="
python -m src.train \
    --config configs/default.yaml \
    --num-anchors 256 --seed 42 \
    --token-level --token-pool mean \
    --experiment-name "exp_token_pilot_mean_k256" \
    2>&1 | tee "${EXP_DIR}/token_mean_k256.log"

echo "=== All token pilot runs complete ==="
