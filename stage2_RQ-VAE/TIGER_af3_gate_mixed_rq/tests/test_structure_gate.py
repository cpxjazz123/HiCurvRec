"""The structure gate: domain, neutral start, switches and per-item lookup.

The oracles here are the definitions themselves, written out independently of
the module: a sigmoid of a linear map, a zero output layer that must reproduce
the frozen baseline's ``alpha = 1/2`` exactly, and the c=0 limit of the
hyperbolic distance. Nothing here touches a GPU or the real data.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import structure_gate as gate_module  # noqa: E402
from model.mixed_rq import CURVATURE, AlphaTable, MixedRQ  # noqa: E402
from model.poincare_contract import squared_distance  # noqa: E402


def _table(rows: int = 6, dim: int = gate_module.DESCRIPTOR_DIMS):
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(rows, dim, generator=generator)
    observed = np.ones((rows, dim), dtype=bool)
    observed[2, 6] = False
    values[2, 6] = 0.0
    metadata = {"n_items": rows, "descriptor_names": list(gate_module.DESCRIPTOR_NAMES)}
    return gate_module.StructureTable(values, observed, metadata)


def test_alphas_lie_strictly_inside_the_open_unit_interval():
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[-1].weight.normal_(0.0, 1.0)
            head[-1].bias.normal_(0.0, 1.0)
    structure = _table().values
    alphas = gate(structure)
    assert len(alphas) == gate_module.LEVELS
    for alpha in alphas:
        assert alpha.shape == (structure.shape[0],)
        assert bool((alpha > 0).all()) and bool((alpha < 1).all())


def test_zero_initialised_output_layer_reproduces_the_frozen_alpha_initial():
    gate = gate_module.StructureGate()
    structure = _table().values
    alphas = gate(structure)
    for alpha in alphas:
        # sigma(0) = 1/2 exactly, so the start is the baseline's ALPHA_INITIAL
        assert torch.equal(alpha, torch.full_like(alpha, 0.5))
    assert gate_module.ALPHA_INIT == 0.5


def test_each_level_has_its_own_head():
    gate = gate_module.StructureGate()
    with torch.no_grad():
        gate.heads[0][-1].bias.fill_(2.0)
    structure = _table().values
    alphas = gate(structure)
    assert torch.allclose(alphas[0], torch.sigmoid(torch.tensor(2.0)).expand_as(alphas[0]))
    assert torch.equal(alphas[1], torch.full_like(alphas[1], 0.5))
    assert torch.equal(alphas[2], torch.full_like(alphas[2], 0.5))


def test_behaviour_off_makes_every_item_share_one_alpha_per_level(monkeypatch):
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[-1].weight.normal_(0.0, 1.0)
            head[-1].bias.normal_(0.0, 1.0)
    structure = _table().values
    monkeypatch.setattr(gate_module, "BEHAVIOR_OFF", True)
    alphas = gate(structure)
    for alpha in alphas:
        # one weight per level: the spread across items is at the float noise
        # floor of the linear layer, not a per-item value
        assert float(alpha.max() - alpha.min()) < 1e-6, alpha


def test_behaviour_off_uses_the_frozen_train_mean_not_a_per_item_value(monkeypatch):
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[0].weight.normal_(0.0, 1.0)
            head[-1].bias.normal_(0.0, 0.5)
    structure = _table().values
    monkeypatch.setattr(gate_module, "BEHAVIOR_OFF", True)
    off = gate(structure)
    # the frozen train mean is the zero row after standardisation
    on_mean = gate(torch.zeros(1, structure.shape[1]))
    for level, alpha in enumerate(off):
        assert torch.allclose(alpha, on_mean[level].expand_as(alpha), atol=0, rtol=0)


def test_gate_changes_alpha_when_the_descriptor_changes():
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[-1].weight.normal_(0.0, 1.0)
    base = torch.zeros(1, gate_module.DESCRIPTOR_DIMS)
    moved = base.clone()
    moved[0, 0] = 2.0
    assert not torch.allclose(gate(base)[0], gate(moved)[0])


def test_table_lookup_is_by_item_id_and_batch_independent():
    table = _table(rows=6)
    ids = torch.tensor([4, 1, 4, 0])
    rows = table.lookup(ids)
    assert torch.equal(rows[0], rows[2])
    assert torch.equal(rows[0], table.values[4])
    other = table.lookup(torch.tensor([0, 4]))
    assert torch.equal(other[1], rows[0])


def test_geometry_off_has_a_euclidean_hyperbolic_half():
    # the switch's cheap wiring check: at c = 0 the canonical d_c^2/4 is the
    # plain squared Euclidean distance, to 1e-6
    generator = torch.Generator().manual_seed(11)
    first = 0.1 * torch.randn(16, 16, generator=generator, dtype=torch.float64)
    second = 0.1 * torch.randn(16, 16, generator=generator, dtype=torch.float64)
    left = squared_distance_safe(first, second, 0.0) / 4.0
    right = (first - second).pow(2).sum(-1)
    torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
    # and the curvature-1 distance is not the Euclidean one, so the switch is
    # not a no-op
    curved = squared_distance(first, second, CURVATURE) / 4.0
    assert not torch.allclose(curved, right, atol=1e-3)


def squared_distance_safe(first, second, curvature):
    return squared_distance(first, second, curvature)


def test_switch_constants_exist_and_default_to_the_factory_configuration():
    assert gate_module.BEHAVIOR_OFF is False
    assert gate_module.GEOMETRY_OFF is False
    assert gate_module.LAMBDA_CLOSURE == 1.0
    assert gate_module.LAMBDA_BRIDGE == 1.0
    assert gate_module.LAMBDA_USAGE == 1.0


def test_quantizer_uses_the_gate_instead_of_the_frozen_table():
    table = _table(rows=8)
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[-1].bias.fill_(3.0)
    rq = MixedRQ([4, 4, 4], AlphaTable(), curvature=CURVATURE, gate=gate)
    rq.attach_structure(table)
    generator = torch.Generator().manual_seed(3)
    latent = 0.05 * torch.randn(5, 32, generator=generator)
    ids = torch.tensor([0, 1, 2, 3, 4])
    with torch.no_grad():
        rq(latent, item_ids=ids)
    alphas = rq.last_alphas
    assert alphas is not None and len(alphas) == 3
    expected = torch.sigmoid(torch.tensor(3.0))
    for alpha in alphas:
        assert torch.allclose(alpha, expected.expand_as(alpha), atol=1e-6)
    # a gate that is not attached raises rather than silently falling back
    bare = MixedRQ([4, 4, 4], AlphaTable(), curvature=CURVATURE, gate=gate_module.StructureGate())
    try:
        bare(latent)
    except ValueError as error:
        assert "item ids" in str(error)
    else:
        raise AssertionError("a gated quantizer must refuse a forward without item ids")


def test_gate_parameters_receive_gradient_from_the_quantisation_cost():
    table = _table(rows=8)
    gate = gate_module.StructureGate()
    with torch.no_grad():
        for head in gate.heads:
            head[-1].weight.normal_(0.0, 0.5)
    rq = MixedRQ([6, 6, 6], AlphaTable(), curvature=CURVATURE, gate=gate)
    rq.attach_structure(table)
    generator = torch.Generator().manual_seed(5)
    latent = 0.08 * torch.randn(7, 32, generator=generator)
    ids = torch.arange(7)
    readout, quant_loss, _, tokens = rq(latent, item_ids=ids)
    quant_loss.backward()
    norms = [float(head[-1].weight.grad.abs().sum()) for head in gate.heads]
    assert all(value > 0 for value in norms), norms
