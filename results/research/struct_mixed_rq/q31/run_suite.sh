#!/bin/sh
# One run, one new receipt; binds source/test identity before and after.
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
BASE=/fs04/ar57/wenyu/GeneRec/results/research/struct_mixed_rq/q31
STAMP=$(date +%Y%m%dT%H%M%S)
LOG=$BASE/tests_$STAMP.log
[ -e "$LOG" ] && { echo "receipt exists: $LOG" >&2; exit 99; }
: > "$LOG" || exit 98
PRE=$BASE/source_pre_$STAMP.txt
POST=$BASE/source_post_$STAMP.txt
{
  echo "# script: results/research/struct_mixed_rq/q31/run_suite.sh"
  echo "# script_sha256: $(sha256sum "$0" | cut -d' ' -f1)"
  echo "# interpreter: $PY"
  echo "# cwd: $(pwd)"
  echo "# start: $(date -Is)"
  echo "# source_pre: $PRE"
} >> "$LOG"
find . \( -name '*.py' -o -name 'math_spec.json' -o -name 'reference_contract.json' \) -not -path './__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$PRE"
pass=0; fail=0; files=0
for f in tests/test_*.py; do
  files=$((files+1))
  echo "CMD: $PY $f" >> "$LOG"
  out=$("$PY" "$f" 2>&1); rc=$?
  echo "== $f rc=$rc" >> "$LOG"
  echo "$out" >> "$LOG"
  if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
find . \( -name '*.py' -o -name 'math_spec.json' -o -name 'reference_contract.json' \) -not -path './__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$POST"
{
  echo "# end: $(date -Is)"
  echo "# source_post: $POST"
  echo "# source_drift: $(cmp -s "$PRE" "$POST" && echo none || echo CHANGED)"
  echo "files=$files pass=$pass fail=$fail"
} >> "$LOG"
[ $fail -eq 0 ] || exit 1
exit 0
