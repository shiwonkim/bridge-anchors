#!/bin/bash
# Experiment: SpectralAligner K ablation
# K values: 4, 8, 16, 32, 64, 128, 256
# Full COCO 118K, seed=42

set -e

K_VALUES=(4 8 16 32 64 128 256)
EXP_DIR="experiments/exp_spectral_k_ablation"

for K in "${K_VALUES[@]}"; do
    echo "=== Running SpectralAligner K=${K} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model spectral_aligner \
        --num-anchors ${K} \
        --seed 42 \
        --experiment-name "exp_spectral_k${K}" \
        2>&1 | tee "${EXP_DIR}/exp_spectral_k${K}_s42.log"
done

echo "=== All SpectralAligner K ablation runs complete ==="
