"""Durable execution: every command leaves a persisted log, exit code and record."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import depgraph, execution
from tools.iteration.tests.support import write_tree


def test_success_persists_log_metadata_and_evidence(tmp_path):
    result = execution.run('ok_unit', [sys.executable, '-c', 'print("hello")'],
                           out_dir=tmp_path / 'checks', cwd=tmp_path, timeout=30)
    assert result.status == 'PASS'
    assert result.exit_code == 0
    assert Path(result.log).is_file() and 'hello' in Path(result.log).read_text()
    meta = json.loads((Path(result.record) / 'command.json').read_text())
    assert meta['unit_id'] == 'ok_unit'
    assert meta['command']
    assert meta['cwd'] == str(tmp_path)
    assert meta['started_at'] and meta['ended_at']
    assert result.duration_s >= 0
    ev = result.evidence()
    assert ev['exit_code'] == 0 and ev['log'] == result.log
    assert ev['output_sha256'] == depgraph.file_sha(result.log)


def test_nonzero_exit_is_failure_but_log_kept(tmp_path):
    result = execution.run('bad_unit', [sys.executable, '-c', 'print("boom"); raise SystemExit(3)'],
                           out_dir=tmp_path / 'checks', cwd=tmp_path, timeout=30)
    assert result.status == 'FAIL'
    assert result.exit_code == 3
    assert 'boom' in Path(result.log).read_text()


def test_timeout_is_incomplete_with_partial_log(tmp_path):
    code = 'import time; print("start", flush=True); time.sleep(5)'
    result = execution.run('slow_unit', [sys.executable, '-c', code],
                           out_dir=tmp_path / 'checks', cwd=tmp_path, timeout=1)
    assert result.status == 'INCOMPLETE'
    assert 'start' in Path(result.log).read_text()
    meta = json.loads((Path(result.record) / 'command.json').read_text())
    assert meta['status'] == 'INCOMPLETE'


def test_preflight_refusal_writes_nothing(tmp_path):
    blocker = tmp_path / 'not_a_dir'
    blocker.write_text('x')
    with pytest.raises(execution.ExecutionError):
        execution.run('u', [sys.executable, '-c', 'print(1)'], out_dir=blocker, cwd=tmp_path)
    assert blocker.read_text() == 'x'


def test_run_and_record_pass_then_reuse(tmp_path):
    write_tree(tmp_path, {'tools/test_u.py': 'def test_ok():\n    assert True\n'})
    spec = depgraph.Spec('exec_unit', tests=[tmp_path / 'tools/test_u.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    result = execution.run_and_record(cache, spec, [sys.executable, '-c', 'print("ok")'],
                                      out_dir=tmp_path / 'checks', repo=tmp_path,
                                      cwd=tmp_path, timeout=30)
    assert result.status == 'PASS'
    plan = depgraph.plan([spec], cache=cache, repo=tmp_path)
    assert [r['test_id'] for r in plan['reuse']] == ['exec_unit']


def test_run_and_record_failure_is_not_reused(tmp_path):
    write_tree(tmp_path, {'tools/test_u.py': 'def test_ok():\n    assert True\n'})
    spec = depgraph.Spec('exec_unit', tests=[tmp_path / 'tools/test_u.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    result = execution.run_and_record(cache, spec, [sys.executable, '-c', 'raise SystemExit(4)'],
                                      out_dir=tmp_path / 'checks', repo=tmp_path,
                                      cwd=tmp_path, timeout=30)
    assert result.status == 'FAIL'
    plan = depgraph.plan([spec], cache=cache, repo=tmp_path)
    assert plan['reuse'] == []
    assert plan['run'][0]['reason'].startswith('previous status')
