"""`af3_item_potential`: the rank map, the missing-key rule, and the frozen artefact.

The graph used here is deliberately tiny and hand-checkable: two disjoint four-node
paths (`0-1-2-7` and `3-4-5-8`) over an item space of 12, so items `6, 9, 10, 11` have
degree 0 and must take the graph-mean rank.

    kappa_bar = [1, .5, .5, 1, .5, .5, .75, 1, 1, .75, .75, .75]
    rank      = [10.5, 2.5, 2.5, 10.5, 2.5, 2.5, 6.5, 10.5, 10.5, 6.5, 6.5, 6.5]
    u         = (rank - 0.5) / 12  ->  {1/6, 1/2, 5/6}
    u_miss    = (6.5 - 0.5) / 12 = 1/2      (math.json says this value is expected here)

The reuse is also tested: the potential must be exactly what the sibling arm's
`structural_potentials` returns, and the AF3 file must not re-implement it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import af3_item_potential as af3

NUM_ITEMS = 12


def _frame() -> pd.DataFrame:
    """Two rows whose `history ++ [target]` sequences are two disjoint four-node paths."""
    return pd.DataFrame({"history": [[0, 1, 2], [3, 4, 5]], "target": [7, 8]})


@pytest.fixture(scope="module")
def table() -> dict:
    return af3.build_table(_frame(), num_items=NUM_ITEMS)


# --------------------------------------------------------------------------------------
# the rank map
# --------------------------------------------------------------------------------------

def test_potential_comes_from_the_sibling_arm_unchanged(table):
    sibling = af3.sibling()
    for name in ("repository_root", "train_parquet", "sequences", "edge_arrays",
                 "adjacency", "edge_curvatures", "structural_potentials"):
        assert hasattr(sibling, name), name
    items = sibling.sequences(_frame())
    rows, cols = sibling.edge_arrays(items)
    matrix = sibling.adjacency(rows, cols, NUM_ITEMS)
    curvature = sibling.edge_curvatures(matrix, rows, cols)
    expected = sibling.structural_potentials(matrix, rows, cols, curvature)
    np.testing.assert_array_equal(table["potential"], expected)
    # the rank acts on the UNSTANDARDISED mean, not on `standardise`'s output
    assert not np.allclose(table["potential"], sibling.standardise(expected))


def test_hand_computed_ranks_and_u(table):
    np.testing.assert_allclose(
        table["rank"],
        [10.5, 2.5, 2.5, 10.5, 2.5, 2.5, 6.5, 10.5, 10.5, 6.5, 6.5, 6.5])
    np.testing.assert_allclose(table["u"], (table["rank"] - 0.5) / NUM_ITEMS)
    np.testing.assert_allclose(table["u"][[0, 1, 6]], [5.0 / 6.0, 1.0 / 6.0, 0.5])


def test_u_is_the_full_table_rank_not_a_batch_rank(table):
    """The denominator is the table size, so the values do not move with a batch."""
    # smallest/largest AVERAGE rank in this table (ties push them inwards)
    assert np.allclose(table["u"].min(), (2.5 - 0.5) / NUM_ITEMS)
    assert np.allclose(table["u"].max(), (10.5 - 0.5) / NUM_ITEMS)
    subset = af3.build_table(_frame(), num_items=NUM_ITEMS)
    np.testing.assert_array_equal(subset["u"], table["u"])


def test_ties_are_averaged_and_never_broken_by_item_id(table):
    tied = [1, 2, 4, 5]  # all kappa_bar = 0.5
    assert len(set(np.round(table["rank"][tied], 12))) == 1
    assert np.allclose(table["rank"][tied], 2.5)
    # a tie broken by item id would give 1,2,3,4 in id order
    assert not np.allclose(table["rank"][tied], [1.0, 2.0, 3.0, 4.0])


def test_u_is_strictly_inside_the_open_interval(table):
    assert np.all(table["u"] > 0.0) and np.all(table["u"] < 1.0)


def test_rank_is_monotone_in_the_potential(table):
    order = np.argsort(table["potential"], kind="stable")
    assert np.all(np.diff(table["rank"][order]) >= 0.0)


# --------------------------------------------------------------------------------------
# missing keys / degree 0
# --------------------------------------------------------------------------------------

def test_degree_zero_items_take_the_graph_mean_rank(table):
    missing = np.flatnonzero(table["is_missing"])
    np.testing.assert_array_equal(missing, [6, 9, 10, 11])
    assert np.allclose(table["potential"][missing], 0.75)  # the graph mean
    assert np.allclose(table["rank_of_graph_mean"], 6.5)
    u_miss = (table["rank_of_graph_mean"] - 0.5) / NUM_ITEMS
    assert np.allclose(u_miss, 0.5)  # math.json: this configuration sits at 0.5
    assert np.allclose(table["u"][missing], u_miss)


def test_touched_items_are_not_marked_missing(table):
    assert not table["is_missing"][:6].any()
    assert not table["is_missing"][7:9].any()


def test_every_item_id_appears_once(table):
    np.testing.assert_array_equal(table["item_ids"], np.arange(NUM_ITEMS))


# --------------------------------------------------------------------------------------
# the artefact: one file, validated, digest checked on read
# --------------------------------------------------------------------------------------

def test_save_and_load_roundtrip_keeps_the_digest(table, tmp_path):
    target = tmp_path / "af3_item_potential.npz"
    written = af3.save(table, target)
    assert written == target and target.is_file()
    loaded = af3.load(target)
    for name in af3.FIELDS:
        np.testing.assert_array_equal(np.asarray(loaded[name]), np.asarray(table[name]))
    assert loaded["digest"] == af3.digest({name: np.asarray(table[name])
                                           for name in af3.FIELDS})


def test_load_refuses_a_tampered_file(table, tmp_path):
    target = tmp_path / "af3_item_potential.npz"
    af3.save(table, target)
    with np.load(target) as data:
        payload = {name: np.asarray(data[name]) for name in af3.FIELDS}
    payload["u"] = payload["u"] + 0.01
    np.savez(target, digest=np.asarray("0" * 64), **payload)
    with pytest.raises(RuntimeError):
        af3.load(target)


def test_load_refuses_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        af3.load(tmp_path / "absent.npz")


@pytest.mark.parametrize("field,value", [
    ("u", lambda table: np.where(np.arange(NUM_ITEMS) == 0, 0.0, table["u"])),
    ("u", lambda table: np.where(np.arange(NUM_ITEMS) == 0, 1.0, table["u"])),
    ("rank", lambda table: table["rank"] + 100.0),
    ("potential", lambda table: np.where(np.arange(NUM_ITEMS) == 0, np.nan,
                                         table["potential"])),
])
def test_save_refuses_an_out_of_range_table(table, tmp_path, field, value):
    broken = {name: np.asarray(table[name]).copy() for name in af3.FIELDS}
    broken[field] = np.asarray(value(table))
    with pytest.raises(RuntimeError):
        af3.save(broken, tmp_path / "broken.npz")


def test_save_refuses_fields_of_different_lengths(table, tmp_path):
    broken = {name: np.asarray(table[name]).copy() for name in af3.FIELDS}
    broken["u"] = broken["u"][:-1]
    with pytest.raises(RuntimeError):
        af3.save(broken, tmp_path / "short.npz")


def test_rank_of_graph_mean_is_a_scalar_field(table, tmp_path):
    """The graph-mean rank is one value; a length-N vector is the wrong shape.

    On the old code the first `save` raised because the length check compared the
    scalar's size (1) with N, so the scalar field could never coexist with N > 1.
    """
    target = tmp_path / "scalar.npz"
    af3.save(table, target)
    assert target.is_file()

    broken = {name: np.asarray(table[name]).copy() for name in af3.FIELDS}
    broken["rank_of_graph_mean"] = np.full(NUM_ITEMS, table["rank_of_graph_mean"])
    with pytest.raises(RuntimeError):
        af3.save(broken, tmp_path / "broken.npz")


def test_output_path_is_the_eval_artefact_of_this_tree():
    path = af3.output_path()
    assert path.name == "af3_item_potential.npz"
    assert path.parent.name == "eval"
    assert path.parent.parent.name == "TIGER_af3_curvature_factor_mass_rq"


def test_u_to_p_is_not_computed_here():
    """The single `u -> p` map lives in factor_share_geometry.share, not in this file."""
    source = af3.build_table.__code__.co_names
    assert "share" not in source
    assert not hasattr(af3, "share")
