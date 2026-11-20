#!/usr/bin/env bash
# Recompute every readout that depends on the exported SIDs/checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
echo "== prefix retention (D0/D1/U/S)"
$PY results/research/directed_graphlet_role/evaluate.py 2>&1 | tail -6
echo "== relation gap (z vs q kNN recovery)"
$PY results/research/directed_graphlet_role/relation_gap.py 2>&1 | tail -6
echo "== directed-edge readout (AUC + direction)"
$PY results/research/directed_graphlet_role/directed_edge_predictor.py 2>&1 | tail -20
echo "== ablation table"
python3 results/research/directed_graphlet_role/ablation_table.py
