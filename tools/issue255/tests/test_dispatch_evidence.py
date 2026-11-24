"""A corrected formal run cannot consume missing, failed, or stale credentials."""
import hashlib
import json

import pytest

from tools.issue255 import dispatch_evidence


def prepared(tmp_path):
    source = tmp_path / 'stage2_RQ-VAE/arm/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('candidate revision one\n')
    files = {str(source.relative_to(tmp_path)): hashlib.sha256(source.read_bytes()).hexdigest()}
    evidence = tmp_path / 'results/research/issue255'
    evidence.mkdir(parents=True)
    binding = {'files': files}
    records = {
        'TEST_RESULTS.json': {'passed': True, 'failed_suites': [], 'sources_sha256': files,
                             'suites': {'candidate': {'exit_code': 0, 'summary': ['3 passed']},
                                        'assets': {'exit_code': 0, 'summary': ['2 passed']}}},
        'ADMISSION.json': {'passed': True, 'source_hashes': binding,
                           'checks': [{'name': 'entry', 'status': 'PASS'},
                                      {'name': 'behavior_graph', 'status': 'PASS'}]},
        'VALIDATION.json': {'passed': True, 'source_hashes': binding, 'source_run_directory': 'validation/run'},
        'PERFORMANCE.json': {'ceiling_met': True, 'mean_seconds': .7,
                             'window': {'epochs': 30}, 'source_hashes': binding,
                             'source_run_directory': 'benchmark/run'},
        'MODULE_CHECKS.json': {'gradient_reached_every_module': True, 'optimizer_updated_every_module': True,
                               'source_run_directory': 'validation/run'},
    }
    for role in ('VALIDATION.json', 'PERFORMANCE.json'):
        directory = tmp_path / records[role]['source_run_directory']
        directory.mkdir(parents=True)
        for rank in range(4):
            (directory / ('rank_%d.json' % rank)).write_text(json.dumps({
                'rank': rank, 'status': 'pass', 'source_hashes': binding,
                'recipe': {'epochs': 3000, 'global_batch': 4096, 'world_size': 4, 'seed': 42}}))
    for name, value in records.items():
        (evidence / name).write_text(json.dumps(value))
    return source, files, evidence


def test_complete_actual_source_credential_is_accepted(tmp_path):
    _, files, _ = prepared(tmp_path)
    assert dispatch_evidence.verify(tmp_path, files)['source_files'] == files


def test_changed_source_cannot_reuse_green_receipts(tmp_path):
    source, files, _ = prepared(tmp_path)
    source.write_text('candidate revision two\n')
    with pytest.raises(ValueError, match='source'):
        dispatch_evidence.verify(tmp_path, files)


def test_missing_rank_is_not_four_rank_verification(tmp_path):
    _, files, _ = prepared(tmp_path)
    (tmp_path / 'validation/run/rank_3.json').unlink()
    with pytest.raises(ValueError, match='rank'):
        dispatch_evidence.verify(tmp_path, files)


def test_failed_test_or_different_source_is_not_admission(tmp_path):
    _, files, evidence = prepared(tmp_path)
    test = json.loads((evidence / 'TEST_RESULTS.json').read_text())
    test['passed'] = False
    (evidence / 'TEST_RESULTS.json').write_text(json.dumps(test))
    with pytest.raises(ValueError, match='test'):
        dispatch_evidence.verify(tmp_path, files)
