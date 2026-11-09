#!/bin/bash
cd /fs04/ar57/wenyu/GeneRec
TREES="TIGER_BEHAVIOR_R48 TIGER_BEHAVIOR_SAMPLE_RQ-VAE TIGER_bootstrap_code_stability TIGER_HYPERBOLIC_RQ-VAE TIGER_FORK_RQ-VAE TIGER_behavior_residual_geometry_transport TIGER_c106_directed_assignment_usage_balance TIGER_c106_directed_codeword_margin_geometry TIGER_c1_behaviour_conditioned_residual_target curvature_RQ-VAE curvature_RQ-VAE_iter48"
while true; do
  ts=$(date +%H:%M:%S); alive=$(pgrep -cf 'bin/python -u train_rqvae')
  line="$ts alive=$alive"
  for t in $TREES; do
    v=$(grep -oE 'epoch=[0-9]+' results/research/stage3_queue/retrain_$t.log | tail -1 | cut -d= -f2)
    [ -z "$v" ] && v=$(grep -oE 'global_step=[0-9]+' results/research/stage3_queue/retrain_$t.log | tail -1 | cut -d= -f2)
    line="$line ${t##*/}=${v:--}"
  done
  echo "$line $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ',')" >> results/research/stage3_queue/retrain_progress.log
  sleep 60
done
