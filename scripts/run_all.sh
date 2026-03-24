#!/bin/bash
# Run all experiments sequentially: A → B → C → D → collect results
#
# Prerequisites:
#   1. Embeddings extracted: ./scripts/extract_embeddings.sh
#   2. Dependencies installed: pip install -r requirements.txt
#
# Usage:
#   ./scripts/run_all.sh              # run everything
#   ./scripts/run_all.sh --skip-a     # skip experiment A
#   ./scripts/run_all.sh --only b     # run only experiment B
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Parse args ---
SKIP_A=false
SKIP_B=false
SKIP_C=false
SKIP_D=false
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-a) SKIP_A=true; shift ;;
        --skip-b) SKIP_B=true; shift ;;
        --skip-c) SKIP_C=true; shift ;;
        --skip-d) SKIP_D=true; shift ;;
        --only)   ONLY="$2"; shift 2 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If --only is set, skip everything else
if [ -n "$ONLY" ]; then
    SKIP_A=true; SKIP_B=true; SKIP_C=true; SKIP_D=true
    case "$ONLY" in
        a|A) SKIP_A=false ;;
        b|B) SKIP_B=false ;;
        c|C) SKIP_C=false ;;
        d|D) SKIP_D=false ;;
        *)   echo "Unknown experiment: $ONLY (use a/b/c/d)"; exit 1 ;;
    esac
fi

# --- Check embeddings ---
EMB_DIR="${PROJECT_ROOT}/data/embeddings"
if [ ! -f "${EMB_DIR}/coco_train_img.pt" ]; then
    echo "ERROR: COCO train embeddings not found."
    echo "Run: ./scripts/extract_embeddings.sh"
    exit 1
fi
if [ ! -f "${EMB_DIR}/flickr30k_test_img.pt" ]; then
    echo "WARNING: Flickr30k embeddings not found. Retrieval eval will be skipped."
fi

echo "========================================"
echo "  Bridge Anchors — Full Experiment Suite"
echo "========================================"
echo ""
START_TIME=$(date +%s)

# --- Experiment A ---
if [ "$SKIP_A" = false ]; then
    echo ">>> Starting Experiment A: Main Comparison"
    bash "${SCRIPT_DIR}/run_exp_a.sh"
    echo ""
fi

# --- Experiment B ---
if [ "$SKIP_B" = false ]; then
    echo ">>> Starting Experiment B: K Ablation"
    bash "${SCRIPT_DIR}/run_exp_b.sh"
    echo ""
fi

# --- Experiment C ---
if [ "$SKIP_C" = false ]; then
    echo ">>> Starting Experiment C: Data Efficiency"
    bash "${SCRIPT_DIR}/run_exp_c.sh"
    echo ""
fi

# --- Experiment D ---
if [ "$SKIP_D" = false ]; then
    echo ">>> Starting Experiment D: Fixed vs Learnable"
    bash "${SCRIPT_DIR}/run_exp_d.sh"
    echo ""
fi

# --- Collect results ---
echo ">>> Collecting results..."
python "${SCRIPT_DIR}/collect_results.py"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo "  Results:    results/all_results.csv"
echo "  Logs:       experiments/exp_*/  "
echo "========================================"
