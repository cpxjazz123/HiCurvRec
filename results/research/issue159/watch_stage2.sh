#!/usr/bin/env bash
# Wait for the #159 Stage2 export, then run the section 63 recording and the collision report.
set -u
cd /fs04/ar57/wenyu/GeneRec
SID="results/stage2_RQ-VAE/TIGER_plan_rounded_rq/item_sids.json"
for _ in $(seq 60); do
  if [ -f "$SID" ]; then
    sleep 20
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.research.science.sid_layer_analysis "$SID" \
      && /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/research/issue159/post_stage2.py
    exit 0
  fi
  sleep 55
done
echo "timeout: no export appeared"
exit 1
