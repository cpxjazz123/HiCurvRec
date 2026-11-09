#!/bin/sh
# The real command that produced tests.log: one interpreter invocation per file.
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
LOG=/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q15/tests.log
: > "$LOG"
{
  echo "# script: results/research/struct_mixed_rq/q15/run_suite.sh"
  echo "# script_sha256: $(sha256sum "$0" | cut -d' ' -f1)"
  echo "# interpreter: $PY"
  echo "# cwd: $(pwd)"
  echo "# start: $(date -Is)"
} >> "$LOG"
pass=0; fail=0; files=0
for f in tests/test_*.py; do
  files=$((files+1))
  echo "CMD: $PY $f" >> "$LOG"
  out=$("$PY" "$f" 2>&1); rc=$?
  echo "== $f rc=$rc" >> "$LOG"
  echo "$out" >> "$LOG"
  if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
{
  echo "# end: $(date -Is)"
  echo "files=$files pass=$pass fail=$fail"
} >> "$LOG"
tail -2 "$LOG"
