#!/usr/bin/env bash
# Issue #248: one command that rebuilds nothing and re-verifies everything.
#   - unit / contract tests for the full-resolution curvature
#   - mutation tests (each mutation must be caught by its named assertion)
#   - the asset manifest: every hash, every declared use, the full enumeration
# Usage: bash tools/graph_infra/verify_v4.sh
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
status=0
run() {
  local label="$1"; shift
  echo "=== $label"
  if "$@"; then echo "--- $label: OK"; else echo "--- $label: FAILED"; status=1; fi
}
run "full curvature tests" "$PY" -m unittest tools.graph_infra.tests.test_full_curvature_v4
run "v4 asset tests"      "$PY" -m unittest tools.graph_infra.tests.test_v4_assets
run "v4 mutations"        "$PY" -m unittest tools.graph_infra.tests.test_mutation_v4
run "v4 reload"           "$PY" -m unittest tools.graph_infra.tests.test_v4_reload
run "asset verification"  "$PY" -m tools.graph_infra.build_assets_v4 --verify
exit $status
