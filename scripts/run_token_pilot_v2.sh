#!/bin/bash
# Token-level BridgeAnchors pilot experiments (token runs only, CLS baselines already done)
set -e

EXP_DIR="experiments/exp_token_level_pilot"

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
