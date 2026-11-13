"""The registry face: names, the factory, `needs_rows`, and the untouched default.

`distances/__init__.py` is a fork, so these tests pin both halves of it: the new entry
exists and needs runtime data, and the baseline entry still returns the baseline
function unchanged (otherwise every framework probe that calls `build_distance()`
would start training something else).
"""

from __future__ import annotations

import pytest
import torch

from conftest import CODEBOOK_DIM


def test_the_entry_is_registered_and_describes_itself():
    from distances import DISTANCES

    entry = DISTANCES["per_item_curvature"]
    assert "factory" in entry and "function" not in entry
    assert entry["description"] == __import__("distances.per_item_curvature",
                                              fromlist=["DESCRIPTION"]).DESCRIPTION
    assert "curvature" in entry["description"].lower()


def test_the_module_names_itself():
    from distances import per_item_curvature as module

    assert module.NAME == "per_item_curvature"
    assert module.D == CODEBOOK_DIM
    assert module.D_E + module.D_H == module.D
    assert module.SCALE_HYP == 0.25
    assert module.PerItemCurvature.needs_rows is True


def test_the_factory_builds_a_row_dependent_metric(curvature_table):
    from distances import build_distance

    metric = build_distance(name="per_item_curvature", curvature_table=curvature_table)
    assert metric.needs_rows is True
    latent = torch.zeros(8, CODEBOOK_DIM)
    codebook = torch.zeros(4, CODEBOOK_DIM)
    cost = metric(latent, codebook, torch.arange(8))          # `__call__`
    assert cost.shape == (8, 4)
    torch.testing.assert_close(metric.distances(latent, codebook, torch.arange(8)), cost)


def test_the_factory_refuses_to_build_without_the_table():
    from distances import build_distance

    with pytest.raises(ValueError, match="curvature table"):
        build_distance(name="per_item_curvature")


def test_a_two_argument_call_is_refused_instead_of_guessing_rows(tiny_metric):
    with pytest.raises(ValueError, match="rows"):
        tiny_metric(torch.zeros(4, CODEBOOK_DIM), torch.zeros(3, CODEBOOK_DIM))


def test_the_metric_reads_the_row_curvature_it_is_given(curvature_table):
    from distances import build_distance

    metric = build_distance(name="per_item_curvature", curvature_table=curvature_table)
    codebook = torch.zeros(2, CODEBOOK_DIM)
    codebook[1, 0] = 1.0
    latent = torch.zeros(2, CODEBOOK_DIM)
    latent[:, CODEBOOK_DIM - 1] = 0.3
    first = metric.curvature_of(torch.tensor([0, 7]), latent)
    assert float(first[0]) == pytest.approx(0.25) and float(first[1]) == pytest.approx(4.0)


def test_the_default_distance_is_still_the_baseline_function():
    from distances import DEFAULT, build_distance
    from distances.distance import distances as baseline

    assert DEFAULT == "euclidean"
    assert build_distance() is baseline
    assert build_distance(None, "euclidean") is baseline
    latent = torch.randn(3, CODEBOOK_DIM, generator=torch.Generator().manual_seed(1))
    codebook = torch.randn(5, CODEBOOK_DIM, generator=torch.Generator().manual_seed(2))
    assert torch.equal(build_distance()(latent, codebook), baseline(latent, codebook))


def test_the_config_route_still_resolves_a_name():
    from distances import build_distance
    from distances.distance import distances as baseline

    assert build_distance({"distance": "euclidean"}) is baseline
    with pytest.raises(Exception):
        build_distance({"distance": "not_a_metric"})


def test_an_unknown_name_stops_with_the_registry_listed():
    from distances import DistanceError, build_distance

    with pytest.raises(DistanceError, match="per_item_curvature"):
        build_distance(name="per_item_curvature_v2")


def test_runtime_arguments_for_a_stateless_metric_are_refused():
    from distances import DistanceError, build_distance

    with pytest.raises(DistanceError, match="takes no runtime arguments"):
        build_distance(name="euclidean", curvature_table=torch.ones(3))


def test_the_stateless_baseline_module_is_untouched():
    """`distances/distance.py` must keep the exact expression `geometry_off` is compared to."""
    import inspect

    from distances import distance as module

    source = inspect.getsource(module.distances)
    assert "latent.pow(2).sum(1, keepdim=True)" in source
    assert "codebook.pow(2).sum(1, keepdim=True).t()" in source
    assert module.NAME == "euclidean"


def test_the_metric_object_holds_no_trainable_state(tiny_metric):
    assert not hasattr(tiny_metric, "parameters")
    assert tiny_metric.table.requires_grad is False
    assert tiny_metric.has_levels is False
