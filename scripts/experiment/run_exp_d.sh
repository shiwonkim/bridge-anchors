#!/bin/bash
# Experiment D: Fixed vs Learnable anchors
# K=32, four strategies:
#   (1) Fixed random    — fixed_relative_rep, init doesn't matter (random anchor selection)
#   (2) Fixed prototype — fixed_relative_rep with prototype-selected anchors
#                         (requires separate config; for now, uses default random selection
#                          since FixedRelativeRep anchor selection is seed-based in train.py)
#   (3) Learnable random    — bridge_anchors, init_method=random
#   (4) Learnable prototype — bridge_anchors, init_method=prototype
# COCO 118K, Flickr30k eval, 3 seeds
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG="${PROJECT_ROOT}/configs/default.yaml"
EXP_DIR="${PROJECT_ROOT}/experiments/exp_d_fixed_vs_learnable"

SEEDS=(42 123 456)

echo "========================================"
echo "Experiment D: Fixed vs Learnable Anchors"
echo "========================================"

# Strategy 1: Fixed random
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp_d_fixed_random_s${SEED}"
    echo ""
    echo "--- ${EXP_NAME} ---"

    python -m src.train \
        --config "${CONFIG}" \
        --model fixed_relative_rep \
        --num-anchors 32 \
        --seed "${SEED}" \
        --experiment-name "${EXP_NAME}" \
        2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
done

# Strategy 2: Fixed prototype
# Uses the prototype config variant — train.py's _select_fixed_anchors picks
# random paired samples. For a true "fixed prototype" we'd need prototype-
# selected anchors. We use a dedicated config that overrides this behavior.
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp_d_fixed_proto_s${SEED}"
    echo ""
    echo "--- ${EXP_NAME} ---"

    python -m src.train \
        --config "${PROJECT_ROOT}/configs/exp_d_fixed_proto.yaml" \
        --model fixed_relative_rep \
        --num-anchors 32 \
        --seed "${SEED}" \
        --experiment-name "${EXP_NAME}" \
        2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
done

# Strategy 3: Learnable random
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp_d_learnable_random_s${SEED}"
    echo ""
    echo "--- ${EXP_NAME} ---"

    python -m src.train \
        --config "${CONFIG}" \
        --model bridge_anchors \
        --num-anchors 32 \
        --init-method random \
        --seed "${SEED}" \
        --experiment-name "${EXP_NAME}" \
        2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
done

# Strategy 4: Learnable prototype
for SEED in "${SEEDS[@]}"; do
    EXP_NAME="exp_d_learnable_proto_s${SEED}"
    echo ""
    echo "--- ${EXP_NAME} ---"

    python -m src.train \
        --config "${CONFIG}" \
        --model bridge_anchors \
        --num-anchors 32 \
        --init-method prototype \
        --seed "${SEED}" \
        --experiment-name "${EXP_NAME}" \
        2>&1 | tee "${EXP_DIR}/${EXP_NAME}.log"
done

echo ""
echo "Experiment D complete. Logs in ${EXP_DIR}/"
