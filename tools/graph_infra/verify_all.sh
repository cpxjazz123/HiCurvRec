#!/usr/bin/env bash
# One command that re-verifies everything the v2 infrastructure claims (issue #246).
# Usage: bash tools/graph_infra/verify_all.sh
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
status=0
run() {  # run <label> <command...>
  local label="$1"; shift
  echo "=== $label"
  if "$@"; then echo "--- $label: OK"; else echo "--- $label: FAILED"; status=1; fi
}
run "manifest v2"        "$PY" tools/graph_infra/verify_assets.py --version v2
for suite in test_known_answers test_reference test_topology test_graphlet test_pairs \
             test_metrics test_assets test_graph_distance test_v2_graph test_v2_ego \
             test_v2_graphlet test_v2_metrics test_v2_assets test_v2_pairs test_v2_ppr \
             test_v2_geometry test_v2_assets_real; do
  run "$suite" "$PY" "tools/graph_infra/tests/$suite.py"
done
run "mutations v2"       "$PY" tools/graph_infra/tests/test_mutation_v2.py
run "module checks"      "$PY" tools/graph_infra/module_checks.py
run "isolation audit"    "$PY" tools/graph_infra/audit_v2.py
run "tiger provenance"   "$PY" tools/graph_infra/tiger_audit.py
run "baseline v2"        "$PY" tools/graph_infra/baseline_v2.py
exit $status
