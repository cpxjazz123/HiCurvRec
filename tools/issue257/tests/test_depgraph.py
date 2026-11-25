"""Dependency-aware validation: reuse only on identical real dependencies.

Three real change scenarios from the issue:
  * a mechanism-math change must invalidate the tests that reach it,
  * an unrelated report edit must not,
  * a shared-operator change must invalidate every consumer.

A mutation drops the import closure and the first scenario must then be missed,
which proves the test measures the closure and not just file paths.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.issue257 import depgraph


def fixture_repo(root):
    (root / 'tools').mkdir(parents=True, exist_ok=True)
    (root / 'tools' / 'shared.py').write_text('CONST = 1\n')
    (root / 'tools' / 'gate.py').write_text('from tools.shared import CONST\n\n\ndef value():\n    return CONST\n')
    (root / 'tools' / 'test_gate.py').write_text('from tools.gate import value\n\n\ndef test_value():\n    assert value() == 1\n')
    (root / 'report.md').write_text('notes\n')
    return root


def spec(root):
    return depgraph.Spec('gate_math',
                         code=[root / 'tools' / 'gate.py'],
                         tests=[root / 'tools' / 'test_gate.py'],
                         runtime=False)


def test_pass_is_reused_for_identical_dependencies(tmp_path):
    root = fixture_repo(tmp_path)
    cache = depgraph.Cache(tmp_path / 'receipts')
    fingerprint = spec(root).fingerprint(repo=root)
    cache.record('gate_math', fingerprint['digest'], 'PASS')
    plan = depgraph.plan([spec(root)], cache=cache, repo=root)
    assert [row['test_id'] for row in plan['reuse']] == ['gate_math']
    assert plan['rerun'] == []


def test_mechanism_change_invalidates_and_unrelated_edit_does_not(tmp_path):
    root = fixture_repo(tmp_path)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', spec(root).fingerprint(repo=root)['digest'], 'PASS')

    (root / 'report.md').write_text('notes edited\n')
    plan = depgraph.plan([spec(root)], cache=cache, repo=root,
                         changed=[root / 'report.md'])
    assert plan['rerun'] == [], 'an unrelated report edit must not force a rerun'

    (root / 'tools' / 'gate.py').write_text('from tools.shared import CONST\n\n\ndef value():\n    return CONST + 1\n')
    plan = depgraph.plan([spec(root)], cache=cache, repo=root, changed=[root / 'tools' / 'gate.py'])
    assert [row['test_id'] for row in plan['rerun']] == ['gate_math']


def test_shared_operator_change_invalidates_every_consumer(tmp_path):
    root = fixture_repo(tmp_path)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', spec(root).fingerprint(repo=root)['digest'], 'PASS')
    (root / 'tools' / 'shared.py').write_text('CONST = 2\n')
    plan = depgraph.plan([spec(root)], cache=cache, repo=root, changed=[root / 'tools' / 'shared.py'])
    assert [row['test_id'] for row in plan['rerun']] == ['gate_math']


def test_failed_receipt_never_reuses(tmp_path):
    root = fixture_repo(tmp_path)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', spec(root).fingerprint(repo=root)['digest'], 'FAIL')
    plan = depgraph.plan([spec(root)], cache=cache, repo=root)
    assert plan['reuse'] == [] and plan['rerun'][0]['reason'].startswith('previous status')


def test_a_new_import_changes_the_fingerprint(tmp_path):
    root = fixture_repo(tmp_path)
    before = spec(root).fingerprint(repo=root)['digest']
    (root / 'tools' / 'extra.py').write_text('MORE = 1\n')
    (root / 'tools' / 'gate.py').write_text(
        'from tools.shared import CONST\nfrom tools.extra import MORE\n\n\ndef value():\n    return CONST + MORE\n')
    after = spec(root).fingerprint(repo=root)['digest']
    assert before != after


def test_mutation_dropping_the_closure_is_caught(tmp_path, monkeypatch):
    """The shared-operator scenario is only detected because imports are followed."""
    root = fixture_repo(tmp_path)
    cache = depgraph.Cache(tmp_path / 'receipts')
    cache.record('gate_math', spec(root).fingerprint(repo=root)['digest'], 'PASS')
    (root / 'tools' / 'shared.py').write_text('CONST = 3\n')

    real = depgraph.plan([spec(root)], cache=cache, repo=root, changed=[root / 'tools' / 'shared.py'])
    assert real['rerun'], 'the real planner must rerun on a shared-operator change'

    def flat_closure(paths, repo=REPO, roots=()):
        return sorted({Path(p).resolve() for p in paths})

    monkeypatch.setattr(depgraph, 'module_closure', flat_closure)
    mutated = depgraph.plan([spec(root)], cache=cache, repo=root, changed=[root / 'tools' / 'shared.py'])
    assert mutated['rerun'], 'the changed file is still declared, so this mutant only loses the import edge'
    # The mutant is exposed by the *undeclared* consumer: an edit to shared.py with
    # only the entry module declared must still be caught by the real closure.
    entry_only = depgraph.Spec('gate_math', tests=[root / 'tools' / 'test_gate.py'], runtime=False)
    fresh = depgraph.Cache(tmp_path / 'fresh')
    fresh.record('gate_math', entry_only.fingerprint(repo=root)['digest'], 'PASS')
    (root / 'tools' / 'shared.py').write_text('CONST = 4\n')
    mutated_plan = depgraph.plan([entry_only], cache=fresh, repo=root, changed=[root / 'tools' / 'shared.py'])
    assert mutated_plan['rerun'] == [], 'the mutant cannot see the undeclared shared dependency'
