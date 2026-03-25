#!/bin/bash
# Experiment A: Main comparison across all methods
# Models: bridge_anchors, linear_projection, mlp_projection, fixed_relative_rep
# 3 random seeds each, COCO 118K, Flickr30k + ImageNet eval
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
EXP_DIR="${PROJECT_ROOT}/experiments/exp_a_main"

SEEDS=(42 123 456)
MODELS=(bridge_anchors linear_projection mlp_projection fixed_relative_rep)

echo "========================================"
echo "Experiment A: Main Comparison"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="exp_a_${MODEL}_s${SEED}"
        echo ""
        echo "--- ${EXP_NAME} ---"

        python -m src.train \
            --config "${CONFIG}" \
            --model "${MODEL}" \
            --seed "${SEED}" \
            --experiment-name "${EXP_NAME}" \
            2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"

        # Run full eval on best checkpoint (if it exists)
        CKPT="${PROJECT_ROOT}/results/checkpoints/${EXP_NAME}/best.pt"
        if [ -f "${CKPT}" ]; then
            echo "  Evaluating ${CKPT} ..."
            python -m src.eval.retrieval --checkpoint "${CKPT}" \
                2>&1 | tee -a "${EXP_DIR}/${EXP_NAME}.log"
            python -m src.eval.zeroshot --checkpoint "${CKPT}" \
                2>&1 | tee -a "${EXP_DIR}/${EXP_NAME}.log"
        fi
    done
done

echo ""
echo "Experiment A complete. Logs in ${EXP_DIR}/"
