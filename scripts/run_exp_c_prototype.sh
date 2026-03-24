#!/bin/bash
# Experiment C+: BridgeAnchors K=128 with PROTOTYPE init, data efficiency
# Subsets: 500, 1000, 5000, 10000, 50000, 118287 (all)
# Seed: 42

set -e

SIZES=(500 1000 5000 10000 50000)
EXP_DIR="experiments/exp_c_data_efficiency"

for N in "${SIZES[@]}"; do
    echo "=== Running BridgeAnchors prototype init, N=${N} ==="
    python -m src.train \
        --config configs/default.yaml \
        --model bridge_anchors \
        --num-anchors 128 \
        --init-method prototype \
        --num-samples ${N} \
        --seed 42 \
        --experiment-name "exp_c_ba_proto_n${N}" \
        2>&1 | tee "${EXP_DIR}/exp_c_bridge_anchors_proto_n${N}_s42.log"
done

# Full dataset (no --num-samples override → uses all 118287)
echo "=== Running BridgeAnchors prototype init, N=118287 (all) ==="
python -m src.train \
    --config configs/default.yaml \
    --model bridge_anchors \
    --num-anchors 128 \
    --init-method prototype \
    --seed 42 \
    --experiment-name "exp_c_ba_proto_n118287" \
    2>&1 | tee "${EXP_DIR}/exp_c_bridge_anchors_proto_n118287_s42.log"

echo "=== All prototype init runs complete ==="
