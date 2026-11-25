#!/usr/bin/env python3.9
"""Fixed-fixture scenario matrix for the incremental validation path (#259).

Writes SCENARIOS.json / SCENARIOS.md and removes its scratch tree. It uses only
public interfaces: Spec/Cache/plan. The cache lives outside the scratch tree so
the cross-round scenario can open a second Cache over the same bytes.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from tools.iteration import depgraph  # noqa: E402

WORK = HERE / 'scenarios_work'
CACHE = HERE / 'scenario_cache'


def tree(name, files):
    root = WORK / name
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def write_evidence(root, unit):
    log = root / 'logs' / (unit + '.log')
    out = root / 'outputs' / (unit + '.json')
    log.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('ok\n')
    out.write_text('{"ok": true}\n')
    return {'command': 'python -m pytest', 'exit_code': 0, 'log': str(log),
            'output': str(out), 'output_sha256': depgraph.file_sha(out)}


def record(cache, spec, root, unit=None, status='PASS'):
    unit = unit or spec.test_id
    fp = spec.fingerprint(repo=root)
    cache.record(unit, fp['digest'], status, evidence=write_evidence(root, unit),
                 declared=spec.declared(), fingerprint=fp)
    return fp


def measure(specs, cache, root, changed=()):
    start = time.perf_counter()
    result = depgraph.plan(specs, cache=cache, repo=root, changed=list(changed))
    return result, round((time.perf_counter() - start) * 1000.0, 3)


def one_unit(root, name='unit', *, code='tools/u.py', test='tools/test_u.py', **kwargs):
    return depgraph.Spec(name, code=[root / code], tests=[root / test], runtime=False, **kwargs)


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    rows = []

    # 1. first run: no receipt.
    root = tree('first', {'tools/u.py': 'A = 1\n', 'tools/test_u.py': 'from tools.u import A\n'})
    spec = one_unit(root)
    plan, ms = measure([spec], depgraph.Cache(CACHE), root)
    rows.append({'scenario': 'first_run', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 2. no change: second Cache instance over the same bytes (cross-round lookup).
    cache = depgraph.Cache(CACHE)
    record(cache, spec, root)
    plan, ms = measure([spec], depgraph.Cache(CACHE), root)
    rows.append({'scenario': 'no_change_cross_round', 'expected_run': 0, 'expected_reuse': 1,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 3. single-unit change: only that unit reruns.
    shared_root = tree('single', {
        'tools/one.py': 'A = 1\n', 'tools/two.py': 'B = 2\n',
        'tools/test_one.py': 'from tools.one import A\n', 'tools/test_two.py': 'from tools.two import B\n'})
    one = depgraph.Spec('unit_one', code=[shared_root / 'tools/one.py'],
                        tests=[shared_root / 'tools/test_one.py'], runtime=False)
    two = depgraph.Spec('unit_two', code=[shared_root / 'tools/two.py'],
                        tests=[shared_root / 'tools/test_two.py'], runtime=False)
    cache = depgraph.Cache(CACHE / 'single')
    record(cache, one, shared_root)
    record(cache, two, shared_root)
    (shared_root / 'tools/one.py').write_text('A = 9\n')
    plan, ms = measure([one, two], cache, shared_root, [shared_root / 'tools/one.py'])
    rows.append({'scenario': 'single_unit_change', 'expected_run': 1, 'expected_reuse': 1,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 4. shared dependency change: every consumer reruns.
    shared_root = tree('shared', {
        'tools/shared.py': 'C = 1\n',
        'tools/a.py': 'from tools.shared import C\n', 'tools/test_a.py': 'from tools.a import C\n',
        'tools/b.py': 'from tools.shared import C\n', 'tools/test_b.py': 'from tools.b import C\n'})
    a = depgraph.Spec('shared_a', code=[shared_root / 'tools/a.py'],
                      tests=[shared_root / 'tools/test_a.py'], runtime=False)
    b = depgraph.Spec('shared_b', code=[shared_root / 'tools/b.py'],
                      tests=[shared_root / 'tools/test_b.py'], runtime=False)
    cache = depgraph.Cache(CACHE / 'shared')
    record(cache, a, shared_root)
    record(cache, b, shared_root)
    (shared_root / 'tools/shared.py').write_text('C = 2\n')
    plan, ms = measure([a, b], cache, shared_root, [shared_root / 'tools/shared.py'])
    rows.append({'scenario': 'shared_dependency_change', 'expected_run': 2, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 5. missing / corrupt evidence.
    root = tree('corrupt', {'tools/u.py': 'A = 1\n', 'tools/test_u.py': 'from tools.u import A\n'})
    spec = one_unit(root)
    cache = depgraph.Cache(CACHE / 'corrupt')
    fp = record(cache, spec, root)
    (root / 'logs' / 'unit.log').unlink()
    plan, ms = measure([spec], cache, root)
    rows.append({'scenario': 'missing_log', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})
    root2 = tree('missing_asset', {'tools/test_u.py': 'from tools.u import A\n'})
    missing = root2 / 'assets' / 'absent.npy'
    spec2 = depgraph.Spec('asset_unit', tests=[root2 / 'tools/test_u.py'], assets=[missing], runtime=False)
    plan, ms = measure([spec2], depgraph.Cache(CACHE / 'missing_asset'), root2)
    rows.append({'scenario': 'missing_asset', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 6. unknown dynamic import: no reuse even with valid evidence.
    root = tree('unknown', {
        'tools/loader.py': 'import importlib\n\ndef load(name):\n    return importlib.import_module(name)\n',
        'tools/test_loader.py': 'from tools.loader import load\n'})
    spec = depgraph.Spec('unknown_unit', code=[root / 'tools/loader.py'],
                         tests=[root / 'tools/test_loader.py'], runtime=False)
    cache = depgraph.Cache(CACHE / 'unknown')
    record(cache, spec, root)
    plan, ms = measure([spec], cache, root)
    rows.append({'scenario': 'unknown_dynamic_import', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 7. deleted dependency: one re-evaluation, then reuse.
    root = tree('deleted', {
        'tools/a.py': 'from tools.b import B\nVALUE = B\n',
        'tools/b.py': 'B = 1\n',
        'tools/test_a.py': 'from tools.a import VALUE\n'})
    spec = depgraph.Spec('del_unit', code=[root / 'tools/a.py'],
                         tests=[root / 'tools/test_a.py'], runtime=False)
    cache = depgraph.Cache(CACHE / 'deleted')
    record(cache, spec, root)
    (root / 'tools/b.py').unlink()
    (root / 'tools/a.py').write_text('VALUE = 1\n')
    plan, ms = measure([spec], cache, root, [root / 'tools/b.py'])
    rows.append({'scenario': 'deleted_dependency_first', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})
    record(cache, spec, root)
    plan, ms = measure([spec], cache, root, [root / 'tools/b.py'])
    rows.append({'scenario': 'deleted_dependency_after_receipt', 'expected_run': 0, 'expected_reuse': 1,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 8. mandatory gate never reuses.
    root = tree('mandatory', {'tools/u.py': 'A = 1\n', 'tools/test_u.py': 'from tools.u import A\n'})
    gate = depgraph.Spec('admission_entry', code=[root / 'tools/u.py'],
                         tests=[root / 'tools/test_u.py'], cacheable=False)
    cache = depgraph.Cache(CACHE / 'mandatory')
    record(cache, gate, root)
    plan, ms = measure([gate], cache, root)
    rows.append({'scenario': 'mandatory_gate', 'expected_run': 1, 'expected_reuse': 0,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    # 9. runtime scoping: torch-only change reruns the GPU unit, not the CPU unit.
    root = tree('runtime', {'tools/test_u.py': 'from tools.u import A\n'})
    gpu = depgraph.Spec('gpu_unit', tests=[root / 'tools/test_u.py'], runtime=True,
                        runtime_keys=('torch', 'devices'))
    cpu = depgraph.Spec('cpu_unit', tests=[root / 'tools/test_u.py'], runtime=False)
    cache = depgraph.Cache(CACHE / 'runtime')
    base = {'interpreter': '/py', 'implementation': 'cpython', 'python': '3.9',
            'torch': '2.0', 'cuda': '11.8', 'devices': 4, 'platform': 'linux'}
    original = depgraph.runtime_identity
    try:
        depgraph.runtime_identity = lambda: dict(base)
        record(cache, gpu, root)
        record(cache, cpu, root)
        depgraph.runtime_identity = lambda: dict(base, torch='9.9', devices=1)
        plan, ms = measure([gpu, cpu], cache, root)
    finally:
        depgraph.runtime_identity = original
    rows.append({'scenario': 'runtime_scoped', 'expected_run': 1, 'expected_reuse': 1,
                 'run': len(plan['run']), 'reuse': len(plan['reuse']), 'plan_ms': ms,
                 'reasons': [r['reason'] for r in plan['run']]})

    payload = {'schema': 'genrec.issue259.scenarios/1', 'rows': rows,
               'total_run': sum(row['run'] for row in rows),
               'total_reuse': sum(row['reuse'] for row in rows),
               'all_expected': all(row['run'] == row['expected_run'] and row['reuse'] == row['expected_reuse']
                                   for row in rows)}
    (HERE / 'SCENARIOS.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    lines = ['# #259 scenario matrix', '',
             '| scenario | run | reuse | expected run | expected reuse | plan ms |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| %s | %d | %d | %d | %d | %s |' % (
            row['scenario'], row['run'], row['reuse'], row['expected_run'],
            row['expected_reuse'], row['plan_ms']))
    lines += ['', 'Total: %d run, %d reuse, all expected: %s' %
              (payload['total_run'], payload['total_reuse'], payload['all_expected'])]
    (HERE / 'SCENARIOS.md').write_text('\n'.join(lines) + '\n')
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(CACHE, ignore_errors=True)
    print(json.dumps({'rows': len(rows), 'all_expected': payload['all_expected'],
                      'total_run': payload['total_run'], 'total_reuse': payload['total_reuse']}))
    return 0 if payload['all_expected'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
