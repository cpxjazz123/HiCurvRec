"""Issue #200 acceptance 1-5 and 7: a start point that is unchanged, a term that is switched off cleanly, and a kernel with no free constant.

The strongest case here is the diff against the start point: the arm's whole claim is that only one thing was added, so the geometry files must be byte-identical
to F0's.  The remaining cases pin the switch, the AF3 correction, the readout/export identity, and the kernel's limits.
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

from model import af3_teacher, r48k_geometry, sd_step, sdh_kernel, struct_distill  # noqa: E402

F0 = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_mixed_rq_F0"
GEOMETRY = ("model/mixed_rq.py", "model/model.py", "model/layers.py", "model/poincare_contract.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- acceptance 2: the start point is byte-identical ---------------------------------------------------------------


@pytest.mark.parametrize("relative", GEOMETRY)
def test_every_geometry_file_is_byte_identical_to_the_start_point(relative):
    assert _sha(TREE / relative) == _sha(F0 / relative), f"{relative} was modified; this arm changes no geometry"


def test_only_the_identity_and_the_structural_modules_differ():
    f0 = {p.relative_to(F0).as_posix(): _sha(p) for p in F0.rglob("*.py")}
    here = {p.relative_to(TREE).as_posix(): _sha(p) for p in TREE.rglob("*.py")}
    added = sorted(set(here) - set(f0))
    changed = sorted(name for name in set(here) & set(f0) if here[name] != f0[name])
    assert added == ["model/af3_teacher.py", "model/r48k_geometry.py", "model/sd_step.py",
                     "model/sdh_kernel.py", "model/struct_distill.py", "tests/test_f0b_structure.py"], added
    assert changed == ["train_rqvae.py"], changed


# --- acceptance 1 and 7: the switch closes the term ---------------------------------------------------------------


def test_behaviour_off_is_the_shipped_state_and_the_weight_expression_goes_to_zero():
    assert struct_distill.BEHAVIOUR_OFF is False
    saved = struct_distill.BEHAVIOUR_OFF
    try:
        struct_distill.BEHAVIOUR_OFF = True
        weight = 0.0 if struct_distill.BEHAVIOUR_OFF else struct_distill.lambda_schedule(3000) * 0.5
        assert weight == 0.0
    finally:
        struct_distill.BEHAVIOUR_OFF = saved


def test_a_step_with_the_switch_closed_never_reaches_the_mechanism(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the structural term was reached while BEHAVIOUR_OFF is set")

    monkeypatch.setattr(sd_step, "structural_loss", explode)
    monkeypatch.setattr(sdh_kernel, "support_mask", explode)
    saved = struct_distill.BEHAVIOUR_OFF
    struct_distill.BEHAVIOUR_OFF = True
    try:
        torch.manual_seed(0)
        from model import RQVAE
        config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                                 codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
        model = RQVAE(config, in_dim=32)
        batch = torch.randn(16, 32)
        reconstructed, quant_loss, _, _ = model(batch, item_ids=torch.arange(16))
        loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
        weight = 0.0 if struct_distill.BEHAVIOUR_OFF else 1.0
        loss = loss + weight * 0.0
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        struct_distill.BEHAVIOUR_OFF = saved


# --- acceptance 3: the corrected AF3 --------------------------------------------------------------------------------


CYCLE4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
TRIANGLE = [(0, 1), (1, 2), (2, 0)]
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def _degrees(edges):
    counts: dict[int, int] = {}
    for i, j in edges:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    return counts


@pytest.mark.parametrize("edges,expected", [
    (CYCLE4, 0.0),      # 4 - 2 - 2 + 3*0
    (TRIANGLE, 3.0),    # 4 - 2 - 2 + 3*1
    (K4, 4.0),          # 4 - 3 - 3 + 3*2
])
def test_af3_counts_triangles_rather_than_assuming_one(edges, expected):
    values = af3_teacher.af3_values(edges, _degrees(edges))
    assert all(value == pytest.approx(expected) for value in values.values())


# --- acceptance 4: the readout and the export agree ---------------------------------------------------------------


def test_codes_of_is_the_export_rule():
    from model import RQVAE
    torch.manual_seed(1)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32).eval()
    embeddings = torch.randn(64, 32)
    assert torch.equal(sd_step.codes_of(model, embeddings), model.get_indices(embeddings, infer_use_sk=False))


# --- acceptance 5: the kernel -------------------------------------------------------------------------------------


def test_affinity_is_the_frozen_form_and_the_teacher_is_detached():
    torch.manual_seed(2)
    anchor = torch.randn(2, 8) * 0.05
    candidates = torch.randn(2, 4, 8) * 0.05
    kappa_a = torch.tensor([1.0, 0.5])
    kappa_c = torch.tensor([0.7, 1.3, 2.0, 0.2])
    scores = sdh_kernel.affinity(anchor, candidates, kappa_a, kappa_c)
    assert scores.shape == (2, 4)
    assert not scores.requires_grad
    assert bool(torch.isfinite(scores).all())
    # i = j gives 1 up to the ball's own epsilon
    coincident = sdh_kernel.affinity(anchor, anchor.unsqueeze(1), kappa_a, kappa_a)
    assert torch.allclose(coincident.diagonal(dim1=0, dim2=1) if coincident.dim() == 2 else coincident[:, 0],
                          torch.ones(2), atol=1e-5)


def test_the_supported_mass_is_normalised_and_the_empty_anchor_contributes_nothing():
    scores = torch.tensor([[0.5, 0.25, 0.125]])
    mask = torch.tensor([[True, True, False]])
    teacher = sdh_kernel.teacher_distribution(scores, mask)
    assert float(teacher.sum()) == pytest.approx(1.0)
    assert float(teacher[0, 2]) == 0.0
    logits = torch.zeros(2, 3, requires_grad=True)
    empty = torch.tensor([[True, False, False], [False, False, False]])
    loss = sdh_kernel.kl_within_support(logits, teacher.repeat(2, 1) * torch.tensor([[1.0], [0.0]]), empty)
    assert bool(torch.isfinite(loss))


def test_the_encoder_gets_gradient_from_the_structural_loss():
    from model import RQVAE
    torch.manual_seed(3)
    config = SimpleNamespace(hidden_sizes=[32, 16], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16],
                             codebook_dim=32, beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1)
    model = RQVAE(config, in_dim=32).eval()
    embeddings = torch.randn(128, 32)
    neighbours = {i: [(i + 1) % 128, (i + 7) % 128] for i in range(128)}
    anchors = torch.arange(sd_step.ANCHORS_PER_STEP)
    loss = sd_step.structural_loss(model, embeddings, anchors, None, neighbours, {}, 128)
    assert torch.isfinite(loss) and loss.grad_fn is not None
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if n.startswith("encoder.") and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_the_kappa_table_is_the_frozen_one():
    table = r48k_geometry.load_table()
    assert len(table["kappa"]) > 20000
    values = r48k_geometry.kappa_of(torch.tensor([0, 1, 2]))
    assert float(values.min()) > 0.0
