#!/usr/bin/env bash
# Verify the first-party agent package. No third-party framework, no extra environment:
# tools/genrec_agents/ imports nothing outside this repository, so the project's own
# interpreter is all it needs.
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHON="${GENREC_PYTHON:-/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10}"
status=0

echo "== the package imports under the production interpreter =="
"$PYTHON" -c "import tools.genrec_agents.cli, tools.genrec_agents.dag, tools.genrec_agents.worker, tools.genrec_agents.executor, tools.genrec_agents.roles" || status=1

echo "== the package's tests =="
"$PYTHON" -m pytest tools/genrec_agents/tests -q || status=1

if [ "$status" -eq 0 ]; then
  echo "verify: PASS"
else
  echo "verify: FAIL"
fi
exit "$status"
