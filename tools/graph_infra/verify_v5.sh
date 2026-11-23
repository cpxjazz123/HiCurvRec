#!/usr/bin/env bash
# Issue #249: one deterministic command that re-verifies everything this round produced.
#
#   bash tools/graph_infra/verify_v5.sh
#
# No training, no PPR, no writes to the frozen assets: it only reads and checks.
set -u
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
status=0
run() {
  local label="$1"; shift
  echo "=== $label"
  if "$@"; then echo "--- $label: OK"; else echo "--- $label: FAILED"; status=1; fi
}
run "v5 differential tests" "$PY" -m unittest tools.graph_infra.tests.test_v5_diff_sectional
run "v5 infrastructure"    "$PY" -m unittest tools.graph_infra.tests.test_v5_infrastructure
run "v5 held-out"          "$PY" -m unittest tools.graph_infra.tests.test_v5_heldout
run "v5 mutations"         "$PY" -m unittest tools.graph_infra.tests.test_v5_mutations
run "v4 regressions"       "$PY" -m unittest tools.graph_infra.tests.test_full_curvature_v4 \
                                   tools.graph_infra.tests.test_v4_assets \
                                   tools.graph_infra.tests.test_v4_reload
run "v3 regressions"       "$PY" -m unittest tools.graph_infra.tests.test_sectional_v3 \
                                   tools.graph_infra.tests.test_interfaces_v3
run "asset manifest"       "$PY" -m tools.graph_infra.build_assets_v5 --verify
run "module checks"        "$PY" -m tools.graph_infra.module_checks_v5
exit $status
