"""Regression: the candidate ships NO plug-in host (the idea's touch list is distances/).

Earlier revisions let a tree-local ``mechanism.py`` register itself through
``distances/__init__.py`` and edited ``train_rqvae.py`` / added ``dataset_fields.py`` /
``runtime_context.py`` to publish row fields. The frozen surface allows neither: the
trainer, the dataset and the row-field channel are host-owned. These checks pin the
removed plug-in surface so it cannot come back, and pin that the trainer is the frozen
baseline with only ``ARM_NAME`` changed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from distances import DEFAULT, DISTANCES, distance as D  # noqa: E402

#: The candidate tree root (no `Path(__file__).parents[N]`, no absolute path).
TREE = Path(__file__.rsplit("/tests/", 1)[0])


def test_registry_is_static_and_trains_the_mechanism():
    import distances

    assert not hasattr(distances, "_register_from_mechanism")
    assert set(DISTANCES) == {"euclidean", "af3_factor_mass"}
    assert DEFAULT == "af3_factor_mass"
    assert DISTANCES["af3_factor_mass"]["function"] is D.distances
    assert DISTANCES["euclidean"]["function"] is D.euclidean


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

    baseline = (D.repository_root() / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
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
    assert callable(D.published_item_ids) and callable(D._runtime_context)
