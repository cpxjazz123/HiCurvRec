#!/usr/bin/env bash
# Wait for the #167 Stage2 export, then record section 63 and report both collision figures.
set -u
cd /fs04/ar57/wenyu/GeneRec
SID="results/stage2_RQ-VAE/TIGER_af3_curvature_rq/item_sids.json"
for _ in $(seq 400); do
  if [ -f "$SID" ]; then
    sleep 30
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.research.science.sid_layer_analysis "$SID" \
      && /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/research/issue167/post_stage2.py
    exit 0
  fi
  sleep 60
done
echo "timeout: no export appeared"
exit 1
