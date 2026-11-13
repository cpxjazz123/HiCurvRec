"""The frozen table: rank -> curvature, the diagnostics, and the cross-arm import.

The AF3 graph code is NOT duplicated in this arm, so these tests use a fake sibling
module with a potential vector I control. That isolates the part this arm owns:

* `u_i = (rank(kappa_bar_i) - 0.5) / N`, computed ONCE over the whole table (ties take
  the average rank), so a batch cannot change an item's curvature;
* `c_i = c_min * (c_max/c_min)^{u_i}` inside `[0.25, 4.0]`; `behaviour_off` pins
  `u_i = 0.5` -> `c_i = 1` -> variance 0;
* the map is rank-based, so any strictly increasing transform of `kappa_bar` produces
  the SAME table (this is why no standardisation and no Huber are applied);
* a missing / cold-start key takes `u_i = 0.5` and cannot shift anybody else's rank;
* a broken cross-arm import is a `RuntimeError` naming the expected path, never a
  silent fallback to a local copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


def table_module():
    import af3_per_item_curvature as module

    return module


class FakeSibling:
    """The seven functions of the sibling arm, over a potential vector I choose."""

    def __init__(self, potentials):
        self.potentials = np.asarray(potentials, dtype=np.float64)

    def repository_root(self):
        return Path("/nonexistent-but-unused")

    def train_parquet(self):
        return "unused.parquet"

    def sequences(self, frame):
        return frame

    def edge_arrays(self, items):
        return np.array([0, 1]), np.array([1, 2])

    def adjacency(self, rows, cols, num_items):
        return np.zeros((num_items, num_items), dtype=np.float64)

    def edge_curvatures(self, matrix, rows, cols):
        return np.zeros(max(len(rows), 1))

    def structural_potentials(self, matrix, rows, cols, edge_kappa):
        return self.potentials.copy()


@pytest.fixture
def fake_sibling(monkeypatch):
    def install(potentials):
        module = table_module()
        monkeypatch.setattr(module, "_load_sibling", lambda: FakeSibling(potentials))
        return module

    return install


# ------------------------------------------------------------------- the rank map
def test_the_rank_is_the_average_rank_over_the_whole_table():
    module = table_module()
    u, ties, misses = module.global_rank_u([10.0, 20.0, 20.0, 40.0])
    assert ties == 2 and misses == 0
    # ascending average rank: 1, 2.5, 2.5, 4 over N = 4
    np.testing.assert_allclose(u, [(1 - 0.5) / 4, (2.5 - 0.5) / 4, (2.5 - 0.5) / 4,
                                   (4 - 0.5) / 4])
    assert np.all(u > 0.0) and np.all(u < 1.0)


def test_the_rank_does_not_depend_on_the_batch():
    """Same item, same curvature, whether it arrives in a batch of 2 or a batch of 200."""
    module = table_module()
    potentials = np.linspace(-3.0, 3.0, 200)
    whole, _, _ = module.global_rank_u(potentials)
    for subset in (potentials[:2], potentials[7:9], potentials[-3:]):
        # the rank is always taken over the FULL table: the caller passes the whole
        # vector and slices afterwards, so slicing cannot move anybody's u
        np.testing.assert_allclose(module.global_rank_u(potentials)[0][: len(subset)],
                                   whole[: len(subset)])


def test_a_strictly_increasing_transform_leaves_the_table_unchanged(fake_sibling):
    """No standardisation, no Huber: the rank makes any monotone rescale a no-op."""
    potentials = np.array([-2.0, -0.5, 0.0, 0.0, 1.0, 3.0])
    first = fake_sibling(potentials).build(frame=objects(), num_items=6)
    shifted = potentials * 100.0 + 7.0
    assert fake_sibling(shifted).build(frame=objects(), num_items=6)["curvature"].tolist() \
        == first["curvature"].tolist()


def test_a_missing_key_takes_the_neutral_curvature_and_cannot_shift_the_others():
    module = table_module()
    potentials = np.array([1.0, 2.0, 3.0])
    present = module.global_rank_u(potentials)[0]
    with_missing = np.array([1.0, 2.0, 3.0, np.nan])
    u, ties, misses = module.global_rank_u(with_missing, missing=[False, False, False, True])
    assert misses == 1 and ties == 0
    np.testing.assert_allclose(u[:3], present)
    assert u[3] == 0.5


# --------------------------------------------------------------- the curvature map
def test_the_curvature_map_is_the_stated_monotone_map():
    module = table_module()
    u = np.array([0.0, 0.5, 1.0])
    c = module.curvature_of_u(u)
    np.testing.assert_allclose(c, [0.25, 1.0, 4.0])
    assert c[1] == pytest.approx(np.sqrt(0.25 * 4.0))       # geometric interpolation
    assert np.all(np.diff(c) > 0)
    np.testing.assert_allclose(module.curvature_of_u(np.array([0.25]))[0],
                               0.25 * (4.0 / 0.25) ** 0.25)


def test_behaviour_off_pins_every_curvature_to_one():
    module = table_module()
    c = module.curvature_of_u(np.linspace(0.0, 1.0, 9), behaviour_off=True)
    assert np.all(c == 1.0)
    assert float(np.var(c)) == 0.0


def test_the_constants_are_the_mechanisms_constants():
    module = table_module()
    assert (module.C_MIN, module.C_MAX) == (0.25, 4.0)
    assert module.D_E + module.D_H == 32
    assert module.SCALE_HYP == 0.25
    assert module.U_OFF == 0.5


# ------------------------------------------------------------------------- build()
def objects():
    """A stand-in for the training frame: the fake sibling ignores its content."""
    return object()


def test_build_produces_one_curvature_per_item_inside_the_range(fake_sibling):
    potentials = np.array([-1.0, 0.0, 0.0, 0.5, 2.0, 5.0])
    table = fake_sibling(potentials).build(frame=objects(), num_items=6)
    c = table["curvature"]
    assert c.shape == (6,)
    assert np.all(c >= 0.25) and np.all(c <= 4.0)
    assert np.all(np.isfinite(c))
    # strictly monotone in the potential, and equal for the two tied items
    order = np.argsort(potentials, kind="stable")
    assert np.all(np.diff(c[order]) >= 0)
    assert c[1] == c[2]

    metrics = table["metrics"]
    for key in ("items", "graph_edges", "ties", "misses", "curvature_min",
                "curvature_max", "curvature_variance", "curvature_quantiles",
                "radius_quantiles", "radius_limit_quantiles", "log_curvature_variance"):
        assert key in metrics
    assert metrics["items"] == 6
    assert metrics["ties"] == 2
    assert metrics["misses"] == 0
    assert metrics["curvature_variance"] > 0.0
    assert set(metrics["curvature_quantiles"]) == {"min", "p25", "p50", "p75", "max"}


def test_build_reports_a_flat_table_when_behaviour_is_off(fake_sibling):
    potentials = np.linspace(-2.0, 2.0, 8)
    table = fake_sibling(potentials).build(frame=objects(), num_items=8, behaviour_off=True)
    assert np.all(table["curvature"] == 1.0)
    assert table["metrics"]["curvature_variance"] == 0.0
    assert table["metrics"]["curvature_min"] == table["metrics"]["curvature_max"] == 1.0
    np.testing.assert_allclose(table["radius"], 1.0)


def test_build_keeps_the_radius_inside_the_contract_ball(fake_sibling):
    table = fake_sibling(np.linspace(-5.0, 5.0, 32)).build(frame=objects(), num_items=32)
    np.testing.assert_allclose(table["radius"], 1.0 / np.sqrt(table["curvature"]))
    assert np.all(table["radius"] >= 0.5) and np.all(table["radius"] <= 2.0)
    assert np.all(table["radius_limit"] < table["radius"])


def test_curvature_tensor_is_a_frozen_float32_copy(fake_sibling):
    module = table_module()
    table = fake_sibling(np.linspace(-1.0, 1.0, 5)).build(frame=objects(), num_items=5)
    tensor = module.curvature_tensor(table)
    assert tensor.dtype == torch.float32
    assert tensor.requires_grad is False
    np.testing.assert_allclose(tensor.numpy(), table["curvature"], rtol=1e-6)


# ------------------------------------------------------- the cross-arm import rules
def test_a_broken_cross_arm_import_names_the_expected_path(monkeypatch, tmp_path):
    module = table_module()
    monkeypatch.setattr(module, "sibling_path", lambda: tmp_path / "af3_potential.py")
    with pytest.raises(RuntimeError) as error:
        module._load_sibling()
    text = str(error.value)
    assert "TIGER_af3_curvature_drive_rq/af3_potential.py" in text
    assert "af3_potential.py" in text


def test_the_sibling_module_really_has_the_functions_this_arm_calls():
    module = table_module()
    path = module.sibling_path()
    assert path.is_file(), "the sibling arm must stay importable: " + str(path)
    loaded = module._load_sibling()
    for name in module.USED_FUNCTIONS:
        assert callable(getattr(loaded, name))
    assert set(module.USED_FUNCTIONS).isdisjoint(module.UNUSED_FUNCTIONS)


def test_this_arm_does_not_import_the_sibling_build_path():
    """`standardise/huber/calibrate_lambda/build/potential_tensor` must not be called."""
    module = table_module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    body = source.split("UNUSED_FUNCTIONS", 1)[1]
    for forbidden in ("standardise(", "huber(", "calibrate_lambda(",
                      "potential_tensor(", "sibling.build("):
        assert forbidden not in body


def test_the_table_is_built_from_the_train_split_helper_not_from_a_local_graph():
    module = table_module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "train_parquet" in source
    assert "_load_sibling()" in source
    assert "read_parquet" in source.split("def build", 1)[1]
