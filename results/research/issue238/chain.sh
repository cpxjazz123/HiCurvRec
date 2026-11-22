#!/bin/bash
# Wait for H's Stage2 to export, run its Stage3, then run J end to end.
cd /fs04/ar57/wenyu/GeneRec || exit 1
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
LOG=results/research/issue238/chain.log
SID=results/stage2_RQ-VAE/TIGER_kappa_cost_rq/item_sids.json
MANIFEST=results/stage2_RQ-VAE/TIGER_kappa_cost_rq/export_manifest.json

echo "=== wait H stage2 $(date -Iseconds)" >> "$LOG"
# Wait for the export gate, not the trainer: the gate rewrites item_sids.json from the
# checkpoint and Stage 3 must read the rewritten table (CLAUDE.md 67).
for i in $(seq 240); do [ -f "$MANIFEST" ] && [ -f "$SID" ] && break; sleep 15; done
[ -f "$MANIFEST" ] || { echo "H: no export manifest" >> "$LOG"; exit 1; }
echo "=== H export ready: $(python3 -c 'import json;d=json.load(open("results/stage2_RQ-VAE/TIGER_kappa_cost_rq/export_manifest.json"));print("tokens_match",d.get("tokens_match"),"differing",d.get("original_differing_items"))') $(date -Iseconds)" >> "$LOG"

"$PY" - "$SID" >> "$LOG" 2>&1 <<'PY'
import json, subprocess, sys
from pathlib import Path
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
code = ("import json,sys;sys.path.insert(0,%r);sys.path.insert(0,%r);import server;"
        "print(json.dumps(server.tool_stage3_start(%r)))" % (str(REPO), str(REPO/"tools"/"genrec-mcp"), sys.argv[1]))
done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
print("STAGE3_START", done.stdout[-600:], done.stderr[-400:])
PY
for i in $(seq 200); do
  F=$(find results/stage3_T5Train/TIGER_kappa_cost_rq -name test_final.json 2>/dev/null | head -1)
  [ -n "$F" ] && { cp "$F" results/research/issue238/TIGER_kappa_cost_rq_stage3_result.json; echo "H RESULT $(cat "$F")" >> "$LOG"; break; }
  sleep 15
done

echo "=== J stage2+stage3 $(date -Iseconds)" >> "$LOG"
"$PY" results/research/issue238/run_arm.py TIGER_kappa_sk_only_rq >> "$LOG" 2>&1
echo "=== J exit=$? $(date -Iseconds)" >> "$LOG"
echo "=== CHAIN COMPLETE ===" >> "$LOG"
