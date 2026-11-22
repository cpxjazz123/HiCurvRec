#!/bin/bash
cd /fs04/ar57/wenyu/GeneRec || exit 1
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
LOG=results/research/issue238/dt_run.log
SID=results/stage2_RQ-VAE/TIGER_degree_temp_rq/item_sids.json
MAN=results/stage2_RQ-VAE/TIGER_degree_temp_rq/export_manifest.json
echo "=== wait af3 stage2 $(date -Iseconds)" >> "$LOG"
for i in $(seq 240); do [ -f "$SID" ] && [ -f "$MAN" ] && break; sleep 15; done
[ -f "$SID" ] || { echo "no SID" >> "$LOG"; exit 1; }
echo "=== SID ready $(date -Iseconds)" >> "$LOG"
"$PY" - "$SID" >> "$LOG" 2>&1 <<'PY'
import json, subprocess, sys
from pathlib import Path
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
code = ("import json,sys;sys.path.insert(0,%r);sys.path.insert(0,%r);import server;"
        "print(json.dumps(server.tool_stage3_start(%r)))" % (str(REPO), str(REPO/"tools"/"genrec-mcp"), sys.argv[1]))
done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
print("STAGE3_START", done.stdout[-700:], done.stderr[-300:])
PY
for i in $(seq 220); do
  F=$(find results/stage3_T5Train/TIGER_degree_temp_rq -name test_final.json 2>/dev/null | head -1)
  [ -n "$F" ] && { cp "$F" results/research/issue238/TIGER_degree_temp_rq_stage3_result.json; echo "RESULT $(cat "$F")" >> "$LOG"; break; }
  sleep 15
done
echo "=== AF3 FLOW COMPLETE $(date -Iseconds)" >> "$LOG"
