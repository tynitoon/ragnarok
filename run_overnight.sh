#!/bin/bash
# Overnight GPU pipeline: paire 3 cleanup + Band B rescue
# Runs sequentially to avoid GPU contention

set -e  # exit on any error
cd /f/dev/ragnarok

echo "=========================================="
echo "[$(date)] PHASE 1: Paire 3 re-run seeds 44+46 (source cap=300k)"
echo "=========================================="
./venv310/Scripts/python.exe -m scripts.pilot_run \
    --base-seed 44 --seeds 3 \
    --pairs pendulum_dmc_cartpole \
    --source-max-steps 300000 \
    --output pilot_results.json \
    >> pilot_paire3_rerun.log 2>&1

echo "=========================================="
echo "[$(date)] PHASE 1 DONE. Starting PHASE 2."
echo "=========================================="

echo "=========================================="
echo "[$(date)] PHASE 2: Band B rescue cell (seeds 47-51, primary, warmup=200)"
echo "=========================================="
./venv310/Scripts/python.exe -m scripts.pilot_run \
    --base-seed 47 --seeds 5 \
    --pairs cartpole_mcc \
    --output pilot_bandb_results.json \
    >> pilot_bandb.log 2>&1

echo "=========================================="
echo "[$(date)] PHASE 2 DONE. All overnight runs complete."
echo "=========================================="
