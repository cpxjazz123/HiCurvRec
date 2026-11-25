"""Profiler selection must reject stale inventories, not choose by age or speed."""
import json

import pytest

from tools.issue256.r03 import collect_evidence


def write_profile(directory, files):
    directory.mkdir(parents=True)
    for rank in range(4):
        row = {'rank': rank, 'status': 'PROFILED', 'source_unchanged': True,
               'source_hashes_before': files, 'source_hashes_after': files}
        (directory/('rank_%d.json'%rank)).write_text(json.dumps(row))


def configure(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    evidence = repo/'evidence'
    monkeypatch.setattr(collect_evidence, 'ROOT', repo)
    monkeypatch.setattr(collect_evidence, 'SOURCE', repo/'stage2_RQ-VAE/arm')
    monkeypatch.setattr(collect_evidence, 'EVIDENCE', evidence)
    files = {'stage2_RQ-VAE/arm/train_rqvae.py': 'a'*64,
             'tools/issue256/r02/controller.py': 'b'*64}
    return evidence/'steady_profile', files


@pytest.mark.parametrize('difference', ['removed_candidate', 'missing_controller', 'changed_controller'])
def test_only_complete_exact_profiler_inventory_can_be_selected(tmp_path, monkeypatch, difference):
    profiles, files = configure(tmp_path, monkeypatch)
    stale = dict(files)
    if difference == 'removed_candidate':
        stale['stage2_RQ-VAE/arm/execution_entry.py'] = 'c'*64
    elif difference == 'missing_controller':
        stale.pop('tools/issue256/r02/controller.py')
    else:
        stale['tools/issue256/r02/controller.py'] = 'd'*64
    write_profile(profiles/'a_obsolete', stale)
    write_profile(profiles/'z_current', files)
    directory, paths, reports = collect_evidence.select('steady_profile', files)
    assert directory == profiles/'z_current', 'stale profiler inventory must not shadow the exact generation'
    assert len(paths) == 4 and all(r['source_hashes_before'] == files for r in reports)


def test_two_exact_profiler_generations_remain_ambiguous(tmp_path, monkeypatch):
    profiles, files = configure(tmp_path, monkeypatch)
    write_profile(profiles/'a', files)
    write_profile(profiles/'z', files)
    with pytest.raises(ValueError, match='found 2'):
        collect_evidence.select('steady_profile', files)
