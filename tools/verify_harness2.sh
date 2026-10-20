#!/usr/bin/env bash
# The protected runner for the Harness2 / MVG infrastructure (issue #102).
#
# It runs the two suites that guard the work, in the two environments they need, and
# fails loudly if either does. No GPU, no network, no credentials.
#
#   tools/verify_harness2.sh
#
# The framework's own interpreter is needed for the Harness2 suite (the framework needs
# Python 3.12 and pytest); the project's interpreter is needed for the MVG suite (it
# needs torch and deliberately has no pytest).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK_PY="${HARNESS2_FRAMEWORK_PY:-/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python}"
PROJECT_PY="${GENREC_PY:-/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9}"

cd "$REPO_ROOT" || exit 1
status=0

echo "== the Harness2 pipeline (offline, scripted) =="
if [ -x "$FRAMEWORK_PY" ]; then
    PYTHONPATH=. "$FRAMEWORK_PY" -m pytest tools/harness2_genrec/tests -q || status=1
else
    echo "SKIPPED: no framework interpreter at $FRAMEWORK_PY"
    status=1
fi

echo
echo "== the independent checks (judges, mutation suite) =="
if [ -x "$PROJECT_PY" ]; then
    PYTHONPATH=. "$PROJECT_PY" tools/mvg/tests/run_mvg_tests.py || status=1
else
    echo "SKIPPED: no project interpreter at $PROJECT_PY"
    status=1
fi

echo
echo "== the frozen evaluation reproduces the frozen value =="
if [ -x "$PROJECT_PY" ] && [ -f results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json ]; then
    PYTHONPATH=. "$PROJECT_PY" tools/mvg/er_eval.py \
        --sid results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json \
        | python3 -c '
import json, sys
d = json.load(sys.stdin)
er, thr = d["l1_er"], d["frozen_threshold"]
print(f"   L1 ER {er!r} vs the frozen {thr!r}: " + ("MATCH" if abs(er - thr) < 1e-9 else "MISMATCH"))
sys.exit(0 if abs(er - thr) < 1e-9 else 1)
' || status=1
else
    echo "SKIPPED: the frozen baseline arm's SIDs are not here"
    status=1
fi

echo
if [ "$status" -eq 0 ]; then
    echo "ALL GREEN"
else
    echo "FAILURES ABOVE"
fi
exit "$status"
