#!/bin/bash
# Train Bridge Anchors model

set -e

python -m src.train --config configs/default.yaml "$@"
