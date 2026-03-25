#!/bin/bash
# Experiment C: BridgeAnchors K=128 with KMEANS init, data efficiency
# Subsets: 500, 1000, 5000, 10000, 50000, 118287 (all)
# Seed: 42

set -e

SIZES=(500 1000 5000 10000 50000)
EXP_DIR="experiments/exp_c_data_efficiency"

for N in "${SIZES[@]}"; do
    echo "=== Running BridgeAnchors kmeans init, N=${N} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors \
        --num-anchors 128 \
        --init-method kmeans \
        --num-samples ${N} \
        --seed 42 \
        --experiment-name "exp_c_ba_kmeans_n${N}" \
        2>&1 | tee "${EXP_DIR}/exp_c_bridge_anchors_kmeans_n${N}_s42.log"
done

# Full dataset
echo "=== Running BridgeAnchors kmeans init, N=118287 (all) ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors \
    --num-anchors 128 \
    --init-method kmeans \
    --seed 42 \
    --experiment-name "exp_c_ba_kmeans_n118287" \
    2>&1 | tee "${EXP_DIR}/exp_c_bridge_anchors_kmeans_n118287_s42.log"

echo "=== All kmeans init runs complete ==="
