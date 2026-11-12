"""Issue #207 acceptance 1-5 and 7: one curvature, one level, and the gate that stops a collapse.

The truth values come from the frozen contract, from the routing definition and from the geometry diff against F0, never from the implementation's own output.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
REPO_ROOT = TREE.parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import l3k_metric as m  # noqa: E402
from model import poincare_contract as pc  # noqa: E402

F0 = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0"
#: the actual geometry code: the layers and the Poincare contract, which must not move for this arm
GEOMETRY = ("model/layers.py", "model/poincare_contract.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative", GEOMETRY)
def test_every_geometry_file_is_byte_identical_to_the_start_point(relative):
    assert _sha(TREE / relative) == _sha(F0 / relative), f"{relative} was modified; this arm changes no geometry"


def test_the_frozen_cost_function_is_unchanged_and_the_only_new_definition_carries_the_curvature():
    """``mixed_rq.py`` has to change -- the level signature must carry the curvature -- so the case states exactly what stays: F0's ``mixed_cost`` body."""
    import re

    def names(source: str) -> set:
        return set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)", source, flags=re.M))

    def body(source: str, name: str) -> str:
        match = re.search(rf"^def {name}\(.*?(?=^def |\Z)", source, flags=re.M | re.S)
        assert match is not None, name
        return match.group(0)

    here = (TREE / "model/mixed_rq.py").read_text()
    frozen = (F0 / "model/mixed_rq.py").read_text()
    assert body(here, "mixed_cost") == body(frozen, "mixed_cost")
    assert names(here) - names(frozen) == {"mixed_cost_with_kappa"}
    assert names(frozen) - names(here) == set()


def test_the_curvature_acts_on_the_third_level_and_nowhere_else():
    saved = (m.BEHAVIOUR_OFF, m.LEVEL12_ON)
    try:
        m.BEHAVIOUR_OFF, m.LEVEL12_ON = False, False
        assert [m.level_uses_kappa(level) for level in (1, 2, 3)] == [False, False, True]
        m.LEVEL12_ON = True
        assert [m.level_uses_kappa(level) for level in (1, 2, 3)] == [True, True, True]
        m.BEHAVIOUR_OFF = True
        assert [m.level_uses_kappa(level) for level in (1, 2, 3)] == [False, False, False]
    finally:
        m.BEHAVIOUR_OFF, m.LEVEL12_ON = saved


def test_behaviour_off_returns_the_baseline_curvature_for_every_row():
    table = {"kappa": {"1": 4.0, "2": 0.25}, "mean_frozen": 0.5}
    saved = m.BEHAVIOUR_OFF
    try:
        m.BEHAVIOUR_OFF = True
        rows = m.kappa_rows(torch.tensor([1, 2, 999]), table)
    finally:
        m.BEHAVIOUR_OFF = saved
    assert rows.tolist() == [m.BASELINE_CURVATURE] * 3


def test_rows_come_from_the_table_with_the_blind_spot_value_and_a_floor():
    table = {"kappa": {"1": 4.0, "2": 0.0}, "mean_frozen": 0.5}
    saved = m.BEHAVIOUR_OFF
    try:
        m.BEHAVIOUR_OFF = False
        rows = m.kappa_rows(torch.tensor([1, 2, 999]), table)
    finally:
        m.BEHAVIOUR_OFF = saved
    assert rows[0] == pytest.approx(4.0)
    assert rows[1] == pytest.approx(m.KAPPA_FLOOR)
    assert rows[2] == pytest.approx(0.5)


def test_distance_at_curvature_one_is_the_frozen_contracts_distance():
    torch.manual_seed(0)
    points = pc.project(torch.randn(4, 8) * 0.05, 1.0).double()
    codes = pc.project(torch.randn(6, 8) * 0.05, 1.0).double()
    mine = m.curved_squared(points, codes, torch.ones(4, dtype=torch.float64)).sqrt()
    reference = pc.distance(points.unsqueeze(1), codes.unsqueeze(0), 1.0)
    assert torch.allclose(mine, reference, rtol=1e-6, atol=1e-8)


def test_a_coincident_pair_is_zero_up_to_the_ball_epsilon():
    torch.manual_seed(1)
    points = pc.project(torch.randn(3, 8) * 0.05, 1.0).double()
    paired = m.curved_squared_paired(points, points.clone(), torch.ones(3, dtype=torch.float64)).sqrt()
    assert torch.all(paired < 1e-5)


def test_gradients_survive_the_boundary_and_a_tiny_curvature():
    torch.manual_seed(2)
    points = pc.project(torch.randn(3, 8) * 0.5, 1.0).double().requires_grad_(True)
    codes = pc.project(torch.randn(5, 8) * 0.5, 1.0).double()
    values = m.curved_squared(points, codes, torch.tensor([0.001, 1.0, 8.2671], dtype=torch.float64))
    values.sum().backward()
    assert points.grad is not None and bool(torch.isfinite(points.grad).all())
    assert float(points.grad.abs().sum()) > 0.0

    boundary = torch.full((1, 8), 1.0 - 1e-4, dtype=torch.float64, requires_grad=True)
    out = m.curved_squared(boundary, codes, torch.tensor([1.0], dtype=torch.float64))
    out.sum().backward()
    assert boundary.grad is not None and bool(torch.isfinite(boundary.grad).all())


def test_gradcheck_on_the_point_argument():
    torch.manual_seed(3)
    codes = pc.project(torch.randn(4, 8) * 0.05, 1.0).double()
    kappa = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

    def f(points):
        return m.curved_squared(points, codes, kappa)

    start = pc.project(torch.randn(3, 8) * 0.05, 1.0).double().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (start,), eps=1e-6, atol=1e-4)


def test_level_one_and_two_codes_are_identical_with_and_without_level12_on():
    from model import RQVAE
    torch.manual_seed(0)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32)
    model.eval()
    ids = torch.arange(128)
    embeddings = torch.randn(128, 32)
    saved = (m.BEHAVIOUR_OFF, m.LEVEL12_ON)
    try:
        m.BEHAVIOUR_OFF, m.LEVEL12_ON = False, False
        shipped = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
        m.LEVEL12_ON = True
        with_levels = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
        assert torch.equal(shipped[:, 0], with_levels[:, 0]), "level 1 must not react"
        assert torch.equal(shipped[:, 1], with_levels[:, 1]), "level 2 must not react"
    finally:
        m.BEHAVIOUR_OFF, m.LEVEL12_ON = saved


def test_the_encoder_receives_gradient_and_the_curvature_is_not_a_parameter():
    from model import RQVAE
    torch.manual_seed(4)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32)
    embeddings = torch.randn(32, 32)
    reconstructed, quant_loss, _, _ = model(embeddings, item_ids=torch.arange(32))
    loss, _ = model.compute_loss(embeddings, reconstructed, quant_loss)
    loss.backward()
    encoder = [p.grad for n, p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None]
    assert encoder and any(float(g.abs().sum()) > 0 for g in encoder)
    for name, _ in model.named_parameters():
        assert "kappa" not in name


def test_the_collapse_gate_predicate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("l3k_trainer_under_test", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.collapse_gate_fires(0.9)
    assert module.collapse_gate_fires(0.512181)          # F0V's single-process reading
    assert module.collapse_gate_fires(0.823443)          # F0V's four-rank reading
    assert not module.collapse_gate_fires(0.3999)
    assert not module.collapse_gate_fires(0.257168)      # F0's own export
