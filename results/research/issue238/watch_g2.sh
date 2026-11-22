#!/bin/bash
cd /fs04/ar57/wenyu/GeneRec
for i in $(seq 90); do
  F=$(find results/stage3_T5Train/TIGER_g237_sem_rq -name test_final.json 2>/dev/null | head -1)
  if [ -n "$F" ]; then
    cp "$F" results/research/issue238/g2_test_final.json
    echo "DONE $(date -Iseconds)" > results/research/issue238/g2_done.txt
    cat "$F"
    exit 0
  fi
  sleep 60
done
echo TIMEOUT
