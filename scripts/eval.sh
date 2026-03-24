#!/bin/bash
# Evaluate a trained checkpoint

set -e

CHECKPOINT=${1:?Usage: eval.sh <checkpoint_path>}

echo "Evaluating retrieval on Flickr30k..."
python -m src.eval.retrieval --checkpoint "$CHECKPOINT"

echo "Evaluating zero-shot on ImageNet..."
python -m src.eval.zeroshot --checkpoint "$CHECKPOINT"

echo "Done."
