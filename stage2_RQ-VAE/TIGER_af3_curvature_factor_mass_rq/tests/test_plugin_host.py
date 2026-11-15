"""Regression: the candidate ships NO plug-in host (the idea's touch list is distances/).

Earlier revisions let a tree-local ``mechanism.py`` register itself through
``distances/__init__.py`` and edited ``train_rqvae.py`` / added ``dataset_fields.py`` /
``runtime_context.py`` to publish row fields. The frozen surface allows neither: the
trainer, the dataset and the row-field channel are host-owned. These checks pin the
removed plug-in surface so it cannot come back, and pin that the trainer is the frozen
baseline with only ``ARM_NAME`` changed.

Everything here is source-level on purpose: it must not depend on how pytest put the
tree on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _find_tree():
    """The candidate root: probe from this file and from the entries pytest put on sys.path."""
    candidates = [Path(__file__).absolute().parent.parent]
    for entry in list(sys.path):
        start = Path(entry).absolute()
        candidates.append(start)
    for candidate in candidates:
        for parent in [candidate, *candidate.parents]:
            if (parent / "distances" / "distance.py").is_file():
                return parent
    return None


TREE = _find_tree()
pytestmark = pytest.mark.skipif(
    TREE is None,
    reason="the tree root is not reachable from this interpreter's cwd; "
           "source-level regression checks need it",
)


def _repository_root() -> Path:
    for parent in [TREE, *TREE.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise AssertionError("cannot locate the repository root above " + str(TREE))


def test_registry_is_static_and_trains_the_mechanism():
    source = (TREE / "distances" / "__init__.py").read_text(encoding="utf-8")
    assert "_register_from_mechanism" not in source
    assert "import mechanism" not in source
    assert 'DISTANCES.get(name)' in source
    assert '"af3_factor_mass"' in source and '"euclidean"' in source
    assert 'DEFAULT = "af3_factor_mass"' in source
    assert "raise DistanceError" in source


def test_mechanism_files_do_not_reference_a_plug_in_host():
    for name in ("distances/distance.py", "distances/__init__.py"):
        source = (TREE / name).read_text(encoding="utf-8")
        assert "import mechanism" not in source, name
        assert "mechanism_base" not in source, name
        assert "dataset_fields" not in source, name


def test_train_entry_is_the_frozen_baseline_plus_arm_name():
    tree_source = (TREE / "train_rqvae.py").read_text(encoding="utf-8")
    for banned in ("import mechanism", "mechanism_base", "dataset_fields",
                   "publish_full_table", "_mechanism_"):
        assert banned not in tree_source, banned

    baseline = (_repository_root() / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
                / "train_rqvae.py").read_text(encoding="utf-8")
    normalised = tree_source.replace(
        'ARM_NAME = "TIGER_af3_curvature_factor_mass_rq"',
        'ARM_NAME = "TIGER_RQ-VAE"',
    )
    assert normalised == baseline


def test_the_item_id_channel_is_read_lazily_from_the_host():
    """`distances/distance.py` reads `runtime_context.get("item_id")` lazily, not at import."""
    source = (TREE / "distances" / "distance.py").read_text(encoding="utf-8")
    top_level = source.split("\ndef ", 1)[0]
    assert "import runtime_context" not in top_level
    assert "def _runtime_context()" in source
    assert "def published_item_ids()" in source
    assert 'runtime_context.get("item_id")' in source or 'get("item_id")' in source
