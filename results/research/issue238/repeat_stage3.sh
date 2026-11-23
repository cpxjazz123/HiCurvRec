#!/bin/bash
# Run Stage3 twice more on the same baseline SID, serially, to get n=3 (plus the n=2 already on disk).
cd /fs04/ar57/wenyu/GeneRec || exit 1
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
LOG=results/research/issue238/repeat_stage3.log
SID=/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json
for REP in 2 3; do
  BEFORE=$(ls -d results/stage3_T5Train/TIGER_RQ-VAE/logs/Amazon_2023_Instruments/*/ 2>/dev/null | wc -l)
  echo "=== repeat $REP start $(date -Iseconds) (runs before: $BEFORE)" >> "$LOG"
  "$PY" - "$SID" >> "$LOG" 2>&1 <<'PY'
import json, subprocess, sys
from pathlib import Path
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
code = ("import json,sys;sys.path.insert(0,%r);sys.path.insert(0,%r);import server;"
        "print(json.dumps(server.tool_stage3_start(%r)))" % (str(REPO), str(REPO/"tools"/"genrec-mcp"), sys.argv[1]))
done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
print("STAGE3_START", done.stdout[-300:], done.stderr[-300:])
PY
  for i in $(seq 200); do
    NOW=$(ls -d results/stage3_T5Train/TIGER_RQ-VAE/logs/Amazon_2023_Instruments/*/ 2>/dev/null | wc -l)
    if [ "$NOW" -gt "$BEFORE" ]; then
      sleep 30
      ANY=$(pgrep -fc "run_stage3_param_TIGER_RQ-VAE" || true)
      [ "${ANY:-0}" -eq 0 ] && break
    fi
    sleep 15
  done
  echo "=== repeat $REP done $(date -Iseconds)" >> "$LOG"
done
echo "=== REPEATS COMPLETE ===" >> "$LOG"
