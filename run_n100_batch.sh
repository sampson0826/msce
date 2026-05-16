#!/bin/bash
# n=100 batch experiment — 4 models across QuickRouter
# Usage: bash run_n100_batch.sh
set -e
cd "$(dirname "$0")"

export QUICKROUTER_API_KEY="${QUICKROUTER_API_KEY:-sk-0XqACr4qnHSionZmOLmiWegQSpVFNVH7M5sUF2dHN2Z80JVA}"

MODELS=(
    "gpt-4o-mini"
    "claude-haiku-4-5-20251001"
    "claude-sonnet-4-6"
    "claude-opus-4-6"
)

echo "================================================"
echo "n=100 Batch Experiment — ${#MODELS[@]} models"
echo "Seeds: 100, Generations: 3, Samples/model: 400"
echo "Start: $(date)"
echo "================================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "=== $model ==="
    python -m synthetic_decay_monitor.experiment_runner \
        --model "$model" \
        --provider quickrouter \
        --generations 3 \
        --seeds 100 \
        --temperature 0.8 \
        --output-dir experiment_data/n100 \
        --no-analyze 2>&1 | tail -20
    echo "Done: $model at $(date)"
done

echo ""
echo "================================================"
echo "All done: $(date)"
echo "================================================"
