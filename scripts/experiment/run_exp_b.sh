#!/bin/bash
# Experiment B: K ablation
# K = {4, 8, 16, 32, 64, 128, 256}
# BridgeAnchors only, COCO 118K, Flickr30k eval, 3 seeds
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
EXP_DIR="${PROJECT_ROOT}/experiments/exp_b_k_ablation"

SEEDS=(42 123 456)
K_VALUES=(4 8 16 32 64 128 256)

echo "========================================"
echo "Experiment B: K Ablation"
echo "========================================"

for K in "${K_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp_b_k${K}_s${SEED}"
        echo ""
        echo "--- ${EXP_NAME} ---"

        python -m src.train \
            --config "${CONFIG}" \
            --model bridge_anchors \
            --num-anchors "${K}" \
            --seed "${SEED}" \
            --experiment-name "${EXP_NAME}" \
            2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
    done
done

echo ""
echo "Experiment B complete. Logs in ${EXP_DIR}/"
