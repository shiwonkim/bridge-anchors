#!/bin/bash
# Experiment: PCA-reduced BridgeAnchors 2D sweep
# pca_dim: {32, 64, 128, 256} × K: {32, 64, 128}
# Full COCO 118K, seed=42

set -e

PCA_DIMS=(32 64 128 256)
K_VALUES=(32 64 128)
EXP_DIR="experiments/exp_pca_ba"

for PD in "${PCA_DIMS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        # Skip if K > pca_dim (anchors can't exceed embedding dimension)
        if [ "$K" -gt "$PD" ]; then
            echo "=== Skipping pca_dim=${PD}, K=${K} (K > pca_dim) ==="
            continue
        fi
        echo "=== Running PCA-BA pca_dim=${PD}, K=${K} ==="
        python -m src.train \
            --config configs/default.yaml \
            --model bridge_anchors \
            --num-anchors ${K} \
            --init-method random \
            --pca-dim ${PD} \
            --seed 42 \
            --experiment-name "exp_pca_ba_pd${PD}_k${K}" \
            2>&1 | tee "${EXP_DIR}/pca_ba_pd${PD}_k${K}_s42.log"
    done
done

echo "=== All PCA-BA runs complete ==="
