"""Relative, package, transitive and dynamic imports must enter the closure.

These are the #259 import-coverage tests. They fail on the #257 implementation
(which only handled `from . import x` with no module part, and ignored dynamic
imports entirely).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import depgraph
from tools.iteration.tests.support import record_valid, write_tree


def package_repo(root):
    write_tree(root, {
        'pkg/__init__.py': '',
        'pkg/base.py': 'BASE = 1\n',
        'pkg/sub/__init__.py': '',
        'pkg/sub/sibling.py': 'SIBLING = 2\n',
        'pkg/sub/mod.py': (
            'from ..base import BASE\n'
            'from .sibling import SIBLING\n'
            'def value():\n'
            '    return BASE + SIBLING\n'),
        'pkg/tests/test_mod.py': (
            'from pkg.sub.mod import value\n'
            'def test_value():\n'
            '    assert value() == 3\n'),
    })
    return root


def closure(root):
    return depgraph.closure_report([root / 'pkg' / 'tests' / 'test_mod.py'], repo=root)


def test_multi_level_relative_and_sibling_imports(tmp_path):
    root = package_repo(tmp_path)
    report = closure(root)
    names = {str(p.relative_to(root)) for p in report['files']}
    assert 'pkg/sub/mod.py' in names
    assert 'pkg/base.py' in names, 'from ..base must resolve to pkg/base.py'
    assert 'pkg/sub/sibling.py' in names, 'from .sibling must resolve'
    assert 'pkg/__init__.py' in names and 'pkg/sub/__init__.py' in names
    assert report['unresolved'] == []


def test_from_package_import_module(tmp_path):
    root = package_repo(tmp_path)
    (root / 'pkg/consumer.py').write_text('from pkg.sub import sibling\n')
    report = depgraph.closure_report([root / 'pkg/consumer.py'], repo=root)
    names = {str(p.relative_to(root)) for p in report['files']}
    assert 'pkg/sub/sibling.py' in names


def test_static_dynamic_import_is_followed(tmp_path):
    root = package_repo(tmp_path)
    (root / 'pkg/dynamic.py').write_text(
        'import importlib\n'
        'def load():\n'
        '    return importlib.import_module("pkg.sub.sibling")\n')
    report = depgraph.closure_report([root / 'pkg/dynamic.py'], repo=root)
    names = {str(p.relative_to(root)) for p in report['files']}
    assert 'pkg/sub/sibling.py' in names
    assert report['unresolved'] == []


def test_nonliteral_dynamic_import_is_unresolved_and_forces_run(tmp_path):
    root = package_repo(tmp_path)
    (root / 'pkg/unknown.py').write_text(
        'import importlib\n'
        'def load(name):\n'
        '    return importlib.import_module(name)\n')
    spec = depgraph.Spec('dynamic_unit', code=[root / 'pkg/unknown.py'], runtime=False)
    fp = spec.fingerprint(repo=root)
    assert fp['unresolved'], 'a non-literal dynamic import must be marked unresolved'
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('dynamic_unit', fp['digest'], 'PASS', evidence=depgraph and {
        'command': 'x', 'exit_code': 0, 'log': str(_log(root)), 'output': str(_out(root)),
        'output_sha256': depgraph.file_sha(_out(root))}, declared=spec.declared(), fingerprint=fp)
    plan = depgraph.plan([spec], cache=cache, repo=root)
    assert [row['test_id'] for row in plan['run']] == ['dynamic_unit']
    assert 'unresolved' in plan['run'][0]['reason']


def _log(root):
    path = Path(root) / 'logs' / 'dynamic_unit.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('ok\n')
    return path


def _out(root):
    path = Path(root) / 'outputs' / 'dynamic_unit.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"ok": true}\n')
    return path


def test_relative_import_change_invalidates(tmp_path):
    root = package_repo(tmp_path)
    spec = depgraph.Spec('pkg_unit', code=[root / 'pkg/sub/mod.py'],
                         tests=[root / 'pkg/tests/test_mod.py'], runtime=False)
    cache = depgraph.Cache(tmp_path / 'receipts')
    record_valid(cache, spec, root)
    assert depgraph.plan([spec], cache=cache, repo=root)['reuse']
    (root / 'pkg/base.py').write_text('BASE = 9\n')
    plan = depgraph.plan([spec], cache=cache, repo=root, changed=[root / 'pkg/base.py'])
    assert [row['test_id'] for row in plan['run']] == ['pkg_unit']
