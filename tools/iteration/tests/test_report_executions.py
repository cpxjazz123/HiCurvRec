"""The report records how many validation commands ran and how long they took."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import report


def test_executions_section_and_counts(tmp_path):
    identity = {'arm': 'TIGER_x', 'stage2_run_id': 'r1', 'sid': {'sha256': 'abc'}}
    plan = {'run': [], 'rerun': [], 'reuse': [{'test_id': 'u'}]}
    executions = [{'unit_id': 'u', 'status': 'PASS', 'duration_s': 1.5},
                  {'unit_id': 'v', 'status': 'FAIL', 'duration_s': 0.5}]
    written = report.write_all(tmp_path, identity=identity, plan=plan, receipts=[],
                               analysis={'sid_chain': {}}, executions=executions)
    assert 'EXECUTIONS' in written
    payload = json.loads(Path(written['EXECUTIONS']).read_text())
    assert payload['counts'] == {'PASS': 1, 'FAIL': 1}
    assert payload['total'] == 2
    assert payload['duration_s'] == 2.0
    text = Path(written['REPORT']).read_text()
    assert 'executions: 2' in text
    assert 'total execution seconds: 2.0' in text
