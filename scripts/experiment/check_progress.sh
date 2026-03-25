#!/bin/bash
# Check progress of Experiment A runs.
# Usage: ./scripts/experiment/check_progress.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="${PROJECT_ROOT}/experiments/exp_a_main"
CKPT_DIR="${PROJECT_ROOT}/results/checkpoints"
LOGFILE="${EXP_DIR}/experiment_log.txt"

TOTAL=12
MODELS=(bridge_anchors linear_projection mlp_projection fixed_relative_rep)
SEEDS=(42 123 456)

echo "========================================"
echo "  Experiment A Progress"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Count completed runs (have a best.pt or latest.pt checkpoint)
COMPLETED=0
TRAINING=0
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        NAME="exp_a_${MODEL}_s${SEED}"
        BEST="${CKPT_DIR}/${NAME}/best.pt"
        LATEST="${CKPT_DIR}/${NAME}/latest.pt"
        LOG="${EXP_DIR}/${NAME}.log"

        if [ -f "${BEST}" ]; then
            STATUS="DONE"
            COMPLETED=$((COMPLETED + 1))
        elif [ -f "${LATEST}" ]; then
            STATUS="IN PROGRESS"
            TRAINING=$((TRAINING + 1))
        elif [ -f "${LOG}" ]; then
            STATUS="STARTED (no checkpoint yet)"
        else
            STATUS="PENDING"
        fi
        printf "  %-40s %s\n" "${NAME}" "${STATUS}"
    done
done

echo ""
echo "Completed: ${COMPLETED}/${TOTAL}  |  In progress: ${TRAINING}  |  Remaining: $(( TOTAL - COMPLETED - TRAINING ))"

# Show last few log lines
if [ -f "${LOGFILE}" ]; then
    echo ""
    echo "--- Last 10 lines of experiment_log.txt ---"
    tail -10 "${LOGFILE}"
fi

# Check if tmux session is still running
if tmux has-session -t exp_a 2>/dev/null; then
    echo ""
    echo "tmux session 'exp_a' is RUNNING."
else
    echo ""
    echo "tmux session 'exp_a' is NOT running."
    if [ ${COMPLETED} -eq ${TOTAL} ]; then
        echo "All runs completed!"
    elif [ -f "${LOGFILE}" ] && grep -q "All done" "${LOGFILE}"; then
        echo "Experiment finished (check log for details)."
    else
        echo "Experiment may have been interrupted."
    fi
fi
