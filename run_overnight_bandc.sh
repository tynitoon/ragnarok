#!/bin/bash
# Band C N=10 extension: seeds 52-56 on primary pair cartpole_mcc
# Pre-registered in preregistration.md §13 v3.7 amendment (2026-04-17)
# Extends seeds 47-51 Band B data; final analysis pools all 10 seeds.

set -e
cd /f/dev/ragnarok

echo "=========================================="
echo "[$(date)] BAND C EXTENSION: seeds 52-56 on cartpole_mcc"
echo "  Pre-registered pass spec (ALL must hold on N=10):"
echo "    - ratio >= 1.30"
echo "    - log-rank p < 0.10 (asymptotic AND permutation)"
echo "    - LOO min ratio >= 1.15"
echo "  Kill spec (ANY triggers paper abandonment):"
echo "    - ratio < 1.20"
echo "    - log-rank p >= 0.20"
echo "    - LOO min ratio < 1.00"
echo "=========================================="

./venv310/Scripts/python.exe -m scripts.pilot_run \
    --base-seed 52 --seeds 5 \
    --pairs cartpole_mcc \
    --output pilot_bandc_results.json \
    >> pilot_bandc.log 2>&1

echo "=========================================="
echo "[$(date)] BAND C DONE. Run analysis with:"
echo "  python -m scripts.pilot_analysis pilot_bandc_results.json"
echo "  # Then merge seeds 47-51 + 52-56 for final N=10 verdict."
echo "=========================================="
