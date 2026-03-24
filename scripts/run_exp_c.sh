#!/bin/bash
# Experiment C: Data efficiency
# Training samples = {500, 1000, 5000, 10000, 50000, 118287}
# BridgeAnchors(K=32) vs LinearProjection, Flickr30k eval, 3 seeds
# Note: 118287 is the full COCO train set; passing no --num-samples uses all.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
EXP_DIR="${PROJECT_ROOT}/experiments/exp_c_data_efficiency"

SEEDS=(42 123 456)
MODELS=(bridge_anchors linear_projection)
SAMPLE_COUNTS=(500 1000 5000 10000 50000)

echo "========================================"
echo "Experiment C: Data Efficiency"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    # Subsample runs
    for N_SAMPLES in "${SAMPLE_COUNTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            EXP_NAME="exp_c_${MODEL}_n${N_SAMPLES}_s${SEED}"
            echo ""
            echo "--- ${EXP_NAME} ---"

            python -m src.train \
                --config "${CONFIG}" \
                --model "${MODEL}" \
                --num-samples "${N_SAMPLES}" \
                --seed "${SEED}" \
                --experiment-name "${EXP_NAME}" \
                2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
        done
    done

    # Full dataset (no --num-samples override)
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp_c_${MODEL}_nFull_s${SEED}"
        echo ""
        echo "--- ${EXP_NAME} ---"

        python -m src.train \
            --config "${CONFIG}" \
            --model "${MODEL}" \
            --seed "${SEED}" \
            --experiment-name "${EXP_NAME}" \
            2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
    done
done

echo ""
echo "Experiment C complete. Logs in ${EXP_DIR}/"
