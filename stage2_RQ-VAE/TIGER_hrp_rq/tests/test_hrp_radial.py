"""Issue #209: the radial map is the identity at r=1, invertible, direction-preserving, and only the selection sees it."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
REPO_ROOT = TREE.parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import hrp_radial as h  # noqa: E402

F0 = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative", ("model/layers.py", "model/poincare_contract.py"))
def test_the_geometry_layers_are_byte_identical_to_f0(relative):
    assert _sha(TREE / relative) == _sha(F0 / relative)


def test_the_frozen_cost_function_is_unchanged_and_the_radial_one_is_the_only_addition():
    def names(source):
        return set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)", source, flags=re.M))

    def body(source, name):
        match = re.search(rf"^def {name}\(.*?(?=^def |\Z)", source, flags=re.M | re.S)
        assert match is not None
        return match.group(0)

    here = (TREE / "model/mixed_rq.py").read_text()
    frozen = (F0 / "model/mixed_rq.py").read_text()
    assert body(here, "mixed_cost") == body(frozen, "mixed_cost")
    assert names(here) - names(frozen) == {"mixed_cost_radial"}


def test_radius_is_one_everywhere_when_the_switch_is_closed():
    saved = h.BEHAVIOUR_OFF
    try:
        h.BEHAVIOUR_OFF = True
        r = h.radius_rows(torch.arange(16))
    finally:
        h.BEHAVIOUR_OFF = saved
    assert torch.all(r == 1.0)


def test_radius_is_the_inverse_square_root_of_the_frozen_curvature():
    table = {"kappa": {"1": 4.0, "2": 0.25}, "mean_frozen": 1.0}
    r = h.radius_rows(torch.tensor([1, 2]), table)
    assert r.tolist() == pytest.approx([0.5, 2.0], rel=1e-6)


def test_r_one_is_the_identity_matrix():
    torch.manual_seed(0)
    points = torch.rand(6, 8) * 0.5
    moved = h.phi(torch.ones(6), points)
    assert torch.allclose(moved, points, rtol=1e-12, atol=1e-12)


def test_the_map_is_invertible():
    """In the precision it is evaluated in: the map is exact in double, and float32 storage costs about 1e-4 on a component near 0.2."""
    torch.manual_seed(1)
    raw = torch.rand(5, 8, dtype=torch.float64) + 0.05
    points = raw / raw.norm(dim=-1, keepdim=True) * 0.6      # inside the ball, where the map is defined
    r = torch.tensor([0.3, 0.7, 1.0, 2.0, 3.5], dtype=torch.float64)
    round_trip = h.phi_inverse(r, h.phi(r, points))
    assert torch.allclose(round_trip, points, rtol=1e-9, atol=1e-9)


def test_the_map_keeps_the_direction():
    """Double precision: the 4e-4 radian tilt seen in float32 is the storage rounding of the output, not the map."""
    torch.manual_seed(2)
    raw = torch.rand(5, 8, dtype=torch.float64) + 0.05
    points = raw / raw.norm(dim=-1, keepdim=True) * 0.8
    r = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    moved = h.phi(r, points)
    cos = torch.nn.functional.cosine_similarity(points, moved, dim=-1)
    angle = torch.arccos(cos.clamp(-1.0, 1.0))
    assert torch.all(angle < 1e-6)


def test_gradients_stay_finite_at_the_boundary_and_for_a_small_radius():
    points = torch.tensor([[1.0 - 1e-7] * 8], dtype=torch.float64, requires_grad=True)
    moved = h.phi(torch.tensor([1e-3], dtype=torch.float64), points)
    moved.sum().backward()
    assert points.grad is not None and bool(torch.isfinite(points.grad).all())


def test_gradcheck_on_phi_and_on_its_inverse():
    torch.manual_seed(3)
    r = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    start = (torch.rand(3, 6, dtype=torch.float64) * 0.4 + 0.1).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: h.phi(r, p), (start,), eps=1e-6, atol=1e-4)
    mid = h.phi(r, start).detach().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: h.phi_inverse(r, p), (mid,), eps=1e-6, atol=1e-4)


def test_with_radius_one_the_export_matches_the_radius_free_path():
    from model import RQVAE
    torch.manual_seed(4)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32).eval()
    embeddings = torch.randn(64, 32)
    ids = torch.arange(64)
    plain = model.get_indices(embeddings, infer_use_sk=False)
    saved = h.BEHAVIOUR_OFF
    try:
        h.BEHAVIOUR_OFF = True
        gated = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
    finally:
        h.BEHAVIOUR_OFF = saved
    assert torch.equal(plain, gated)


def test_the_collapse_gate_predicate():
    spec = importlib.util.spec_from_file_location("hrp_trainer_under_test", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.collapse_gate_fires(0.9)
    assert module.collapse_gate_fires(0.512181)
    assert not module.collapse_gate_fires(0.3999)
    assert not module.collapse_gate_fires(0.257168)


def test_the_encoder_receives_gradient_and_the_radius_is_not_a_parameter():
    from model import RQVAE
    torch.manual_seed(5)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32)
    embeddings = torch.randn(32, 32)
    reconstructed, quant_loss, _, _ = model(embeddings, item_ids=torch.arange(32) + 100)
    loss, _ = model.compute_loss(embeddings, reconstructed, quant_loss)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    for name, _ in model.named_parameters():
        assert "radius" not in name and "kappa" not in name
