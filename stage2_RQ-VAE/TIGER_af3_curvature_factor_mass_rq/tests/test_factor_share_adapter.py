"""The registry entry `factor_share` and the adapter's interface.

Two groups:

* the adapter's own behaviour (runs today): it forwards to the ONE geometry
  implementation, and it refuses a missing / mis-shaped share instead of silently
  falling back to the Euclidean metric;
* the wiring into `distances/__init__.py` and `distances/distance.py`
  (`4_method.modified_files`), which is what makes the tree actually train on the
  product manifold. Those tests fail until that wiring is written -- by design, the
  failing test is the specification.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

import factor_share_geometry as geometry
from distances import factor_share as adapter


def _fixture(batch: int = 3, codes: int = 4, seed: int = 3):
    torch.manual_seed(seed)
    latent = torch.randn(batch, geometry.D_TOTAL, dtype=torch.float64)
    codebook = torch.randn(codes, geometry.D_TOTAL, dtype=torch.float64) * 0.3
    p = torch.tensor([0.2, 0.5, 0.8][:batch], dtype=torch.float64)
    return latent, codebook, p


# --------------------------------------------------------------------------------------
# the adapter (runs today)
# --------------------------------------------------------------------------------------

def test_adapter_declares_the_registry_fields():
    assert adapter.NAME == "factor_share"
    assert isinstance(adapter.DESCRIPTION, str) and adapter.DESCRIPTION


def test_adapter_forwards_to_the_one_geometry_implementation():
    latent, codebook, p = _fixture()
    got = adapter.distances(latent, codebook, p)
    expected = geometry.product_cost(latent, codebook, p)
    assert torch.equal(got, expected)
    # and it holds no formula of its own: the module source calls product_cost
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "product_cost" in source
    assert "expmap0" not in source and "squared_distance" not in source


def test_adapter_requires_the_share():
    latent, codebook, _ = _fixture()
    with pytest.raises(ValueError):
        adapter.distances(latent, codebook)
    with pytest.raises(ValueError):
        adapter.distances(latent, codebook, None)
    with pytest.raises(ValueError):
        adapter.distances(latent, codebook, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_adapter_geometry_off_equals_the_baseline_expansion():
    latent, codebook, p = _fixture()
    baseline = (latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t())
    assert torch.equal(adapter.distances(latent, codebook, p, mode="geometry_off"), baseline)


def test_adapter_does_not_import_the_model_package():
    """`model/__init__ -> model/layers -> distances` must stay acyclic."""
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "import model" not in source
    assert "from model" not in source


def test_adapter_accepts_the_two_argument_call_shape():
    """The baseline quantiser calls `distance(latent, codebook)`; p may be added."""
    parameters = list(inspect.signature(adapter.distances).parameters)
    assert parameters[:2] == ["latent", "codebook"]
    assert "p" in parameters


# --------------------------------------------------------------------------------------
# the registry wiring (implemented: `4_method.modified_files`, `distances/__init__.py`)
# --------------------------------------------------------------------------------------

def test_registry_holds_factor_share_alongside_euclidean():
    from distances import DISTANCES, build_distance

    assert "euclidean" in DISTANCES  # geometry_off needs it
    assert "factor_share" in DISTANCES
    entry = DISTANCES["factor_share"]
    assert entry["function"] is adapter.distances
    assert isinstance(entry["description"], str) and entry["description"]
    assert build_distance(name="factor_share") is adapter.distances


def test_registry_default_is_still_euclidean():
    from distances import build_distance, distance

    assert build_distance() is distance.distances


def test_registry_unknown_name_still_raises():
    from distances import DistanceError, build_distance

    with pytest.raises(DistanceError):
        build_distance(name="factor_share_typo")


def test_distance_module_has_the_three_argument_compatibility_wrapper():
    """`VQLayer` calls one 3-arg shape for either registry entry."""
    from distances import distance as distance_module

    parameters = list(inspect.signature(distance_module.distances).parameters)
    assert parameters[:3] == ["latent", "codebook", "p"]

    latent, codebook, p = _fixture()
    baseline = (latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t())
    # p is ignored, and the numbers do not move
    assert torch.equal(distance_module.distances(latent, codebook, p), baseline)
    assert torch.equal(distance_module.distances(latent, codebook), baseline)


def test_distance_module_numbers_are_unchanged_for_the_two_argument_call():
    from distances import distance as distance_module

    latent, codebook, _ = _fixture()
    assert torch.equal(distance_module.distances(latent, codebook),
                       distance_module.distances(latent, codebook, None))
