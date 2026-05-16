#!/bin/bash
# Demo script — produces the terminal output shown in README.
# Record with: asciinema rec demo.cast --command "./demo.sh"
# Or: terminalizer record -c demo.yml

echo "============================================================"
echo "SYNTHETIC DATA DECAY MONITOR — Real Qwen2.5-7B Experiment"
echo "Model: Qwen2.5-7B-Instruct | 5 recursive generations"
echo "============================================================"
echo ""

python3 -m constraint_residual.synthetic_decay_monitor.cli \
    --input experiment_data/real_lineage.jsonl \
    --hybrid \
    --output /dev/null \
    2>&1

echo ""
echo "── Intervention priority ──────────────────────────────────"
echo "  1. 🔴 Calibration data — style diversity (most urgent)"
echo "  2. 🟡 Axiom data — formal logic chains"
echo "  3. 🟢 Boundary data — edge cases + rare facts (preventive)"
echo ""
echo "Run: decay-monitor watch --input your_data.jsonl"
