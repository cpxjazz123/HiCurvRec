#!/usr/bin/env bash
# CPU-only, no PPR, no training, no frozen-asset writes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
"$PY" -m unittest tools.graph_infra.tests.test_sectional_v3 tools.graph_infra.tests.test_interfaces_v3 tools.graph_infra.tests.test_mutation_v3
"$PY" -m tools.graph_infra.prepare_v3
"$PY" -m tools.graph_infra.module_checks_v3
