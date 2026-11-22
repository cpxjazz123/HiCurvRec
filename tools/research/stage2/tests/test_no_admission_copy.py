#!/usr/bin/env python3
"""A conformance run judges the tree where it stands: no ``_admission_`` copy.

Owner instruction (2026-11-22): the promoted copy directory is not wanted. The
checks used to copy the tree to ``stage2_RQ-VAE/_admission_<name>_<digest>_<pid>_
<uuid>`` so that the repository marker resolved from the same place the trainer
runs. The trainer derives the root by walking up to a directory that holds
``stage2_RQ-VAE/``, which a tree under ``results/`` reaches as well, so the copy
bought nothing and left a directory behind next to the frozen baseline.

These tests pin the replacement: checks read the real tree, write their scratch
output under ``results/stage2_RQ-VAE/_conformance/``, and never create an
``_admission_*`` directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from tools.research.stage2 import checks, contract  # noqa: E402


def _no_admission_dirs() -> list[str]:
    root = contract.resolve_root()
    return sorted(p.name for p in (root / "stage2_RQ-VAE").glob("_admission_*"))


def test_run_all_returns_the_tree_it_was_given(tmp_path, monkeypatch):
    tree = tmp_path / "TIGER_demo"
    tree.mkdir()
    monkeypatch.setattr(checks, "check_entry",
                        lambda t: {"status": "PASS", "name": "entry", "detail": ""})
    monkeypatch.setattr(checks, "check_behavior_graph",
                        lambda t: {"status": "PASS", "name": "graph", "detail": ""})
    report = checks.run_all(tree)
    assert report["tree"] == str(tree.resolve())
    assert report["passed"] is True
    assert _no_admission_dirs() == []


def test_the_scratch_directory_is_shared_and_outside_the_tree(tmp_path):
    tree = tmp_path / "TIGER_demo"
    tree.mkdir()
    first = checks._scratch_dir(tree)
    second = checks._scratch_dir(tree)
    assert first == second
    assert first.name == "TIGER_demo"
    assert first.parent.name == "_conformance"
    assert tree not in first.parents
    assert tree.resolve() not in first.parents


def test_no_copy_helper_survives():
    # The helpers are gone, not merely unused: a later caller must not be able to
    # recreate the directory the owner refused.
    assert not hasattr(checks, "_admission_tree")
    assert not hasattr(checks, "_discard_admission_tree")
