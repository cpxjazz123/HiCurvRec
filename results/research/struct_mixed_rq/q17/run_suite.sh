#!/bin/sh
# One run, one new receipt; refuses to overwrite and propagates any failure.
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
BASE=/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q17
STAMP=$(date +%Y%m%dT%H%M%S)
LOG=$BASE/tests_$STAMP.log
[ -e "$LOG" ] && { echo "receipt exists: $LOG" >&2; exit 99; }
: > "$LOG" || exit 98
{
  echo "# script: results/research/struct_mixed_rq/q17/run_suite.sh"
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
[ $fail -eq 0 ] || exit 1
exit 0
