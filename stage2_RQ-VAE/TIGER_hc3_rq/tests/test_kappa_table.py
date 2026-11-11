"""The frozen table load and the per-row resolution of issue #177."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import kappa_table as kt  # noqa: E402


def test_repo_root_is_found_by_markers_without_an_absolute_path():
    root = kt.repo_root()
    for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding"):
        assert (root / marker).is_dir()


def test_the_table_loads_and_carries_the_measured_shape():
    table = kt.load()
    assert table["level_1"]["kappa"] == 1.0
    assert table["level_2"]["prefixes"] == 240
    assert table["level_3"]["prefixes"] == 7408
    assert table["items_without_kappa_filled_with_mean"] == 31


def test_level_one_is_exactly_one_for_every_row():
    values = kt.kappa_for_rows([(0,), (5,), (255,)], level=1)
    assert values.shape == (3, 1)
    assert torch.equal(values, torch.ones(3, 1, dtype=torch.float64))


def test_levels_resolve_per_row_and_fall_back_for_unseen_prefixes():
    table = {"level_1": {"kappa": 1.0}, "mean_frozen": 0.75,
             "level_2": {"rows": {"(3,)": 0.25}}, "level_3": {"rows": {"(3, 7)": 2.5}}}
    two = kt.kappa_for_rows([(3,), (99,)], level=2, table=table)
    assert [float(x) for x in two.flatten()] == [0.25, 0.75]
    three = kt.kappa_for_rows([(3, 7), (3, 8)], level=3, table=table)
    assert [float(x) for x in three.flatten()] == [2.5, 0.75]


def test_tensor_prefixes_resolve_the_same_as_tuples():
    table = {"level_1": {"kappa": 1.0}, "mean_frozen": 1.0,
             "level_2": {"rows": {"(3,)": 0.25}}, "level_3": {"rows": {}}}
    from_tensor = kt.kappa_for_rows(torch.tensor([[3], [4]]), level=2, table=table)
    from_tuples = kt.kappa_for_rows([(3,), (4,)], level=2, table=table)
    assert torch.equal(from_tensor, from_tuples)


def test_rows_resolve_inside_the_measured_range():
    table = kt.load()
    values = kt.kappa_for_rows([(0,), (16,), (255,)], level=2)
    assert bool((values >= 1e-3).all()) and bool((values <= 8.27).all())
