"""Saved validation metrics are read only when bound to the same run identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import analysis

IDENTITY = {'arm': 'TIGER_x', 'stage2_run_id': 'run-1', 'sid': {'sha256': 'abc', 'path': 'unused'}}


def artifact(path, *, run_id='run-1', sid='abc', metric=0.071):
    payload = {'identity': {'run_id': run_id, 'sid_sha256': sid},
               'metrics': {'test_recall@10': metric}}
    Path(path).write_text(json.dumps(payload))
    return path


def test_matching_artifact_is_read(tmp_path):
    path = artifact(tmp_path / 'validation.json')
    out = analysis.saved_validation(path, IDENTITY)
    assert out['status'] == 'PASS'
    assert out['metrics']['test_recall@10'] == 0.071


def test_mismatched_identity_is_not_available(tmp_path):
    path = artifact(tmp_path / 'validation.json', run_id='other')
    out = analysis.saved_validation(path, IDENTITY)
    assert out['status'] == 'NOT_AVAILABLE'
    assert 'identity' in out['reason']


def test_sid_mismatch_is_not_available(tmp_path):
    path = artifact(tmp_path / 'validation.json', sid='zzz')
    assert analysis.saved_validation(path, IDENTITY)['status'] == 'NOT_AVAILABLE'


def test_missing_artifact_is_not_available(tmp_path):
    out = analysis.saved_validation(tmp_path / 'absent.json', IDENTITY)
    assert out['status'] == 'NOT_AVAILABLE'
    assert 'missing' in out['reason']


def test_analyse_without_artifact_reports_the_gap(tmp_path):
    sid = tmp_path / 'sids.json'
    sid.write_text(json.dumps({'1': [0, 0, 0, 0]}))
    identity = dict(IDENTITY, sid={'sha256': 'abc', 'path': str(sid)})
    out = analysis.analyse(identity, repo=tmp_path)
    assert out['saved_validation']['status'] == 'NOT_AVAILABLE'
    assert 'no_artifact' in out['saved_validation']['reason']


def test_analyse_with_artifact_includes_metrics(tmp_path):
    sid = tmp_path / 'sids.json'
    sid.write_text(json.dumps({'1': [0, 0, 0, 0]}))
    identity = dict(IDENTITY, sid={'sha256': 'abc', 'path': str(sid)})
    path = artifact(tmp_path / 'validation.json')
    out = analysis.analyse(identity, repo=tmp_path, validation_artifact=path)
    assert out['saved_validation']['status'] == 'PASS'
    assert out['saved_validation']['metrics']['test_recall@10'] == 0.071
