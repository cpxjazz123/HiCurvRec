"""Evidence, generation, deletion and runtime-identity rules of the cache.

A receipt is reusable only when its machine evidence is intact; a deleted or
renamed dependency is re-evaluated against the previous receipt's closure and
then stops forcing reruns once a fresh receipt exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import depgraph
from tools.iteration.tests.support import evidence, record_valid, write_tree


def repo_tree(root):
    write_tree(root, {
        'tools/shared.py': 'CONST = 1\n',
        'tools/gate.py': 'from tools.shared import CONST\n\ndef value():\n    return CONST\n',
        'tools/test_gate.py': 'from tools.gate import value\n\ndef test_value():\n    assert value() == 1\n',
    })
    return depgraph.Spec('gate_math', code=[root / 'tools/gate.py'],
                         tests=[root / 'tools/test_gate.py'], runtime=False)


def valid_plan(root, spec, cache):
    record_valid(cache, spec, root)
    return depgraph.plan([spec], cache=cache, repo=root)


def test_valid_evidence_reuses(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    plan = valid_plan(root, spec, cache)
    assert [r['test_id'] for r in plan['reuse']] == ['gate_math']
    assert plan['run'] == []


@pytest.mark.parametrize('mutate,needle', [
    ('delete_log', 'log'),
    ('empty_log', 'log'),
    ('tamper_output', 'sha'),
    ('drop_output_sha', 'sha'),
    ('drop_command', 'command'),
])
def test_broken_evidence_is_not_reused(tmp_path, mutate, needle):
    root = tmp_path
    spec = repo_tree(root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    ev = evidence(root, 'gate_math')
    record_valid(cache, spec, root, ev=ev)
    if mutate == 'delete_log':
        Path(ev['log']).unlink()
    elif mutate == 'empty_log':
        Path(ev['log']).write_text('')
    elif mutate == 'tamper_output':
        Path(ev['output']).write_text('{"ok": false}\n')
    elif mutate == 'drop_output_sha':
        ev.pop('output_sha256')
        cache = depgraph.Cache(tmp_path / 'receipts2')
        record_valid(cache, spec, root, ev=ev)
    elif mutate == 'drop_command':
        ev.pop('command')
        cache = depgraph.Cache(tmp_path / 'receipts3')
        record_valid(cache, spec, root, ev=ev)
    plan = depgraph.plan([spec], cache=cache, repo=root)
    assert plan['reuse'] == []
    assert plan['run'], 'invalid evidence must force a run'
    assert needle in plan['run'][0]['reason'].lower()


def test_nonzero_exit_and_failed_status_do_not_reuse(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    ev = evidence(root, 'gate_math')
    ev['exit_code'] = 2
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root, ev=ev)
    assert depgraph.plan([spec], cache=cache, repo=root)['reuse'] == []

    cache2 = depgraph.Cache(tmp_path / 'receipts2')
    record_valid(cache2, spec, root, status='FAIL')
    plan = depgraph.plan([spec], cache=cache2, repo=root)
    assert plan['reuse'] == [] and plan['run'][0]['reason'].startswith('previous status')


def test_legacy_receipt_without_fingerprint_is_invalid(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    fp = spec.fingerprint(repo=root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', fp['digest'], 'PASS', evidence=evidence(root, 'gate_math'))
    plan = depgraph.plan([spec], cache=cache, repo=root)
    assert plan['reuse'] == []
    assert 'fingerprint' in plan['run'][0]['reason']


def test_receipts_are_generations_not_overwrites(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    fp = spec.fingerprint(repo=root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', fp['digest'], 'PASS', evidence=evidence(root, 'gate_math'),
                 declared=spec.declared(), fingerprint=fp, generation='0001')
    cache.record('gate_math', fp['digest'], 'FAIL', evidence=evidence(root, 'gate_math'),
                 declared=spec.declared(), fingerprint=fp, generation='0002')
    assert len(cache.receipts('gate_math')) == 2
    plan = depgraph.plan([spec], cache=cache, repo=root)
    assert plan['reuse'] == [], 'the newest generation is FAIL and must win'
    assert plan['run'][0]['reason'].startswith('previous status')


def test_newest_valid_generation_is_reused(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    fp = spec.fingerprint(repo=root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', fp['digest'], 'FAIL', evidence=evidence(root, 'gate_math'),
                 declared=spec.declared(), fingerprint=fp, generation='0001')
    cache.record('gate_math', fp['digest'], 'PASS', evidence=evidence(root, 'gate_math'),
                 declared=spec.declared(), fingerprint=fp, generation='0002')
    assert depgraph.plan([spec], cache=cache, repo=root)['reuse']


def test_deleted_dependency_triggers_one_rerun_then_reuses(tmp_path):
    root = tmp_path
    write_tree(root, {
        'tools/a.py': 'from tools.b import B\nVALUE = B\n',
        'tools/b.py': 'B = 1\n',
        'tools/test_a.py': 'from tools.a import VALUE\n\ndef test_value():\n    assert VALUE == 1\n',
    })
    spec = depgraph.Spec('del_unit', code=[root / 'tools/a.py'],
                         tests=[root / 'tools/test_a.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root)

    (root / 'tools/b.py').unlink()
    (root / 'tools/a.py').write_text('VALUE = 1\n')
    plan = depgraph.plan([spec], cache=cache, repo=root, changed=[root / 'tools/b.py'])
    assert [r['test_id'] for r in plan['run']] == ['del_unit']
    assert 'previous closure' in plan['run'][0]['reason']

    record_valid(cache, spec, root)
    again = depgraph.plan([spec], cache=cache, repo=root, changed=[root / 'tools/b.py'])
    assert again['reuse'], 'a fresh receipt must stop the rerun loop'


def test_renamed_dependency_triggers_rerun(tmp_path):
    root = tmp_path
    write_tree(root, {
        'tools/a.py': 'from tools.old import X\nVALUE = X\n',
        'tools/old.py': 'X = 1\n',
        'tools/test_a.py': 'from tools.a import VALUE\n',
    })
    spec = depgraph.Spec('ren_unit', code=[root / 'tools/a.py'],
                         tests=[root / 'tools/test_a.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root)
    (root / 'tools/old.py').rename(root / 'tools/new.py')
    (root / 'tools/a.py').write_text('from tools.new import X\nVALUE = X\n')
    plan = depgraph.plan([spec], cache=cache, repo=root,
                         changed=[root / 'tools/old.py', root / 'tools/new.py'])
    assert [r['test_id'] for r in plan['run']] == ['ren_unit']
    assert 'previous closure' in plan['run'][0]['reason']


def test_missing_asset_is_visible_and_blocks_reuse(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    missing = root / 'assets' / 'table.npy'
    spec = depgraph.Spec('asset_unit', tests=[root / 'tools/test_gate.py'],
                         assets=[missing], runtime=False)
    fp = spec.fingerprint(repo=root)
    assert str(missing) in ' '.join(fp['missing'])
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root)
    plan = depgraph.plan([spec], cache=cache, repo=root)
    assert plan['reuse'] == []
    assert 'missing' in plan['run'][0]['reason']


def test_runtime_identity_is_scoped_per_unit(tmp_path, monkeypatch):
    root = tmp_path
    spec_gpu = depgraph.Spec('gpu_unit', tests=[root / 'tools/test_gate.py'],
                             runtime=True, runtime_keys=('torch', 'devices'))
    spec_cpu = depgraph.Spec('cpu_unit', tests=[root / 'tools/test_gate.py'], runtime=False)
    base = {'interpreter': '/py', 'implementation': 'cpython', 'python': '3.9',
            'torch': '2.0', 'cuda': '11.8', 'devices': 4, 'platform': 'linux'}
    monkeypatch.setattr(depgraph, 'runtime_identity', lambda: dict(base))
    gpu_before = spec_gpu.fingerprint(repo=root)['digest']
    cpu_before = spec_cpu.fingerprint(repo=root)['digest']
    changed = dict(base, torch='9.9', devices=1)
    monkeypatch.setattr(depgraph, 'runtime_identity', lambda: dict(changed))
    assert spec_gpu.fingerprint(repo=root)['digest'] != gpu_before
    assert spec_cpu.fingerprint(repo=root)['digest'] == cpu_before
    spec_py = depgraph.Spec('py_unit', tests=[root / 'tools/test_gate.py'],
                            runtime=True, runtime_keys=('python',))
    py_before = spec_py.fingerprint(repo=root)['digest']
    monkeypatch.setattr(depgraph, 'runtime_identity', lambda: dict(changed, python='3.9', torch='1.0'))
    assert spec_py.fingerprint(repo=root)['digest'] == py_before


def test_only_the_changed_unit_is_rerun(tmp_path):
    root = tmp_path
    write_tree(root, {
        'tools/one.py': 'A = 1\n',
        'tools/two.py': 'B = 2\n',
        'tools/test_one.py': 'from tools.one import A\n',
        'tools/test_two.py': 'from tools.two import B\n',
    })
    one = depgraph.Spec('unit_one', code=[root / 'tools/one.py'], tests=[root / 'tools/test_one.py'], runtime=False)
    two = depgraph.Spec('unit_two', code=[root / 'tools/two.py'], tests=[root / 'tools/test_two.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, one, root)
    record_valid(cache, two, root)
    (root / 'tools/one.py').write_text('A = 3\n')
    plan = depgraph.plan([one, two], cache=cache, repo=root, changed=[root / 'tools/one.py'])
    assert [r['test_id'] for r in plan['run']] == ['unit_one']
    assert [r['test_id'] for r in plan['reuse']] == ['unit_two']


def test_touched_but_unchanged_file_does_not_loop(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root)
    (root / 'tools/shared.py').write_text('CONST = 1\n')
    plan = depgraph.plan([spec], cache=cache, repo=root, changed=[root / 'tools/shared.py'])
    assert plan['reuse'], 'content identity, not the changed list, decides reuse'


def test_config_files_are_hashed_not_parsed_as_python(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    config = root / 'settings.yaml'
    config.write_text('{not valid python: [\n')
    spec = depgraph.Spec('config_unit', tests=[root / 'tools/test_gate.py'],
                         configs=[config], runtime=False)
    before = spec.fingerprint(repo=root)
    assert before['missing'] == []
    config.write_text('{still not python: [\n')
    assert spec.fingerprint(repo=root)['digest'] != before['digest']


def test_mandatory_gate_is_never_reused(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    gate = depgraph.Spec('admission_entry', code=[root / 'tools/gate.py'],
                         tests=[root / 'tools/test_gate.py'], cacheable=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, gate, root)
    record_valid(cache, spec, root)
    plan = depgraph.plan([gate, spec], cache=cache, repo=root)
    assert [r['test_id'] for r in plan['run']] == ['admission_entry']
    assert 'mandatory' in plan['run'][0]['reason']
    assert [r['test_id'] for r in plan['reuse']] == ['gate_math']


def test_directory_asset_inputs_are_expanded(tmp_path):
    root = tmp_path
    spec = repo_tree(root)
    tables = root / 'assets' / 'tables'
    tables.mkdir(parents=True)
    (tables / 'one.csv').write_text('a\n')
    (tables / 'two.csv').write_text('b\n')
    unit = depgraph.Spec('dir_unit', tests=[root / 'tools/test_gate.py'],
                         assets=[tables], runtime=False)
    fp = unit.fingerprint(repo=root)
    assert set(fp['assets']) == {'assets/tables/one.csv', 'assets/tables/two.csv'}


def test_same_name_different_content_does_not_reuse(tmp_path):
    root_a = tmp_path / 'candidate_a'
    root_b = tmp_path / 'candidate_b'
    for root, value in ((root_a, '1'), (root_b, '2')):
        write_tree(root, {
            'tools/u.py': 'A = %s\n' % value,
            'tools/test_u.py': 'from tools.u import A\n'})
    spec_a = depgraph.Spec('candidate', code=[root_a / 'tools/u.py'],
                           tests=[root_a / 'tools/test_u.py'], runtime=False)
    spec_b = depgraph.Spec('candidate', code=[root_b / 'tools/u.py'],
                           tests=[root_b / 'tools/test_u.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec_a, root_a)
    plan = depgraph.plan([spec_b], cache=cache, repo=root_b)
    assert plan['reuse'] == [], 'same unit name and layout must not reuse another candidate'
    assert plan['run'][0]['test_id'] == 'candidate'


def test_performance_and_mutation_specs_follow_the_compute_path(tmp_path):
    root = tmp_path
    write_tree(root, {
        'tools/kernel.py': 'K = 1\n',
        'tools/test_kernel.py': 'from tools.kernel import K\n',
        'report.md': 'notes\n'})
    benchmark = depgraph.Spec('performance_60_epoch', code=[root / 'tools/kernel.py'],
                              tests=[root / 'tools/test_kernel.py'], runtime=False)
    mutation = depgraph.Spec('mutation_kernel', code=[root / 'tools/kernel.py'],
                             tests=[root / 'tools/test_kernel.py'], runtime=False)
    prose = depgraph.Spec('prose_only', extra=[root / 'report.md'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    for unit in (benchmark, mutation, prose):
        record_valid(cache, unit, root)
    (root / 'report.md').write_text('notes edited\n')
    plan = depgraph.plan([benchmark, mutation, prose], cache=cache, repo=root,
                         changed=[root / 'report.md'])
    assert [r['test_id'] for r in plan['run']] == ['prose_only']
    assert [r['test_id'] for r in plan['reuse']] == ['performance_60_epoch', 'mutation_kernel']
    record_valid(cache, prose, root)
    (root / 'tools/kernel.py').write_text('K = 2\n')
    plan = depgraph.plan([benchmark, mutation, prose], cache=cache, repo=root,
                         changed=[root / 'tools/kernel.py'])
    assert [r['test_id'] for r in plan['run']] == ['performance_60_epoch', 'mutation_kernel']
    assert [r['test_id'] for r in plan['reuse']] == ['prose_only']
