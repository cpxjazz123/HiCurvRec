"""Issue #194 acceptance 2, 4 and 5: the triangle count, the same-code support, the curvature kernel.

Every case is stated from the graph or from the definition, never from the implementation's own output.  The triangle cases run against both teachers --
this tree's (fixed) and ``TIGER_af3_structdistill_rq``'s (shipped, which the audit measured as wrong) -- so the red-to-green record lives with the code that
had to change.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
REPO_ROOT = TREE.parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import af3_teacher as fixed_teacher, sdh_kernel  # noqa: E402


def _load_shipped():
    path = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_af3_structdistill_rq" / "model" / "af3_teacher.py"
    spec = importlib.util.spec_from_file_location("af3_shipped", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHIPPED = _load_shipped()

CYCLE4 = [(0, 1), (1, 2), (2, 3), (3, 0)]                # no triangle at all
TRIANGLE = [(0, 1), (1, 2), (2, 0)]                      # one triangle
K4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]  # every edge in two triangles
PATH_CHORD = [(0, 1), (1, 2), (2, 3), (1, 3)]            # one triangle, one edge outside it


def degrees(edges):
    counts: dict[int, int] = {}
    for i, j in edges:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    return counts


# --- acceptance 2: the graph cases, hand computed ----------------------------------------------------------------


@pytest.mark.parametrize("edges,expected", [
    (CYCLE4, {(0, 1): 0.0, (1, 2): 0.0, (2, 3): 0.0, (3, 0): 0.0}),      # 4 - 2 - 2 + 0
    (TRIANGLE, {(0, 1): 3.0, (1, 2): 3.0, (2, 0): 3.0}),                # 4 - 2 - 2 + 3*1
    (K4, {(i, j): 4.0 for i, j in K4}),                                 # 4 - 3 - 3 + 3*2
])
def test_triangle_count_is_the_true_common_neighbour_count(edges, expected):
    values = fixed_teacher.af3_values(edges, degrees(edges))
    assert values == pytest.approx(expected)


def test_path_with_a_chord_matches_the_hand_computed_values():
    values = fixed_teacher.af3_values(PATH_CHORD, degrees(PATH_CHORD))
    assert values[(0, 1)] == pytest.approx(0.0)
    assert values[(1, 2)] == pytest.approx(2.0)
    assert values[(1, 3)] == pytest.approx(2.0)
    assert values[(2, 3)] == pytest.approx(3.0)


def test_the_shipped_teacher_is_wrong_on_the_triangle_free_cycle():
    """The red case, kept as an executable statement of what the audit found: constant 3 where the truth is 0."""
    shipped = SHIPPED.af3_values(CYCLE4, degrees(CYCLE4))
    assert all(abs(value - 3.0) < 1e-12 for value in shipped.values())
    fixed = fixed_teacher.af3_values(CYCLE4, degrees(CYCLE4))
    assert all(abs(value) < 1e-12 for value in fixed.values())


def test_triangles_change_the_weights_where_the_degree_sum_alone_cannot():
    # the four-cycle and the triangle have identical degree sequences; only the triangle count separates them
    cycle = fixed_teacher.weights(CYCLE4, 4, *fixed_teacher.quantiles(fixed_teacher.af3_values(CYCLE4, degrees(CYCLE4)))).to_dense()
    chorded = fixed_teacher.weights(PATH_CHORD, 4, *fixed_teacher.quantiles(fixed_teacher.af3_values(PATH_CHORD, degrees(PATH_CHORD)))).to_dense()
    assert float(cycle[0, 1]) == pytest.approx(1.0)          # the only value, so u = 0
    assert float(chorded[1, 2]) > 1.0                        # a triangle-carrying edge rises above the floor
    assert float(chorded[0, 1]) == pytest.approx(1.0)        # a triangle-free edge does not


# --- acceptance 4: the support restriction ----------------------------------------------------------------


def test_pairs_sharing_no_code_level_are_outside_the_support():
    anchor = torch.tensor([[5, 9, 3]])
    candidates = torch.tensor([[[5, 9, 3], [5, 0, 1], [7, 8, 2]]])   # 3 shared, 1 shared, 0 shared
    mask = sdh_kernel.support_mask(anchor, candidates)
    assert mask.tolist() == [[True, True, False]]


def test_padding_slots_are_never_in_the_support_even_when_the_codes_coincide():
    anchor = torch.tensor([[1, 2, 3]])
    codes = torch.tensor([[[1, 2, 3], [-1, -1, -1]]])
    ids = torch.tensor([[0, -1]])
    mask = sdh_kernel.support_mask(anchor, codes, ids)
    assert mask.tolist() == [[True, False]]


def test_a_pair_without_shared_codes_carries_zero_teacher_weight():
    anchor = torch.tensor([[4, 4, 4]])
    candidates = torch.tensor([[[4, 4, 4], [9, 9, 9]]])
    mask = sdh_kernel.support_mask(anchor, candidates)
    teacher = sdh_kernel.teacher_distribution(torch.tensor([[0.7, 0.9]]), mask)
    assert float(teacher[0, 1]) == 0.0
    assert float(teacher[0, 0]) == pytest.approx(1.0)     # the only supported pair takes the whole mass


def test_support_off_opens_the_support_to_every_candidate():
    anchor = torch.tensor([[4, 4, 4]])
    candidates = torch.tensor([[[9, 9, 9]]])
    saved = sdh_kernel.SUPPORT_OFF
    try:
        sdh_kernel.SUPPORT_OFF = True
        open_mask = sdh_kernel.support_mask(anchor, candidates)
    finally:
        sdh_kernel.SUPPORT_OFF = saved
    assert open_mask.tolist() == [[True]]
    assert sdh_kernel.support_mask(anchor, candidates).tolist() == [[False]]


# --- acceptance 5: the curvature kernel ----------------------------------------------------------------


def test_affinity_is_one_for_a_coincident_pair_and_the_distance_definition_otherwise():
    from model import r48k_geometry

    embeddings = torch.randn(3, 6, dtype=torch.float32)
    anchor = embeddings[:1]
    cands = embeddings.unsqueeze(0)                       # [1, 3, 6]
    kappa_a = torch.tensor([1.0])
    kappa_c = torch.tensor([1.0, 1.0, 1.0])
    got = sdh_kernel.affinity(anchor, cands, kappa_a, kappa_c)

    k = r48k_geometry.pair_kappa(kappa_a, kappa_c).to(torch.float64)
    p = r48k_geometry.project_to_ball(anchor.to(torch.float64), k.unsqueeze(-1).expand(1, 3, 1))
    q = r48k_geometry.project_to_ball(cands.to(torch.float64), k.unsqueeze(-1))
    expected = 1.0 / (1.0 + r48k_geometry.distance(p, q, k.unsqueeze(-1)))

    assert got.dtype == torch.float32
    # identical points give 1 up to the ball's own epsilon: project_to_ball keeps points strictly inside ((1 - 1e-6) on the radius), so the
    # distance of a coincident pair is ~1.4e-6 rather than exactly zero and the affinity is 1 - 1.4e-6
    assert got[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert got.to(torch.float64) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_affinity_is_strictly_decreasing_in_the_distance_and_carries_no_free_constant():
    """Two candidates at the same curvature: the farther one must score lower, and the value is pinned by the formula, not by a parameter."""
    anchor = torch.zeros(1, 4)
    near = torch.full((1, 1, 4), 0.1)
    far = torch.full((1, 1, 4), 0.9)
    k_a = torch.tensor([1.0])
    k_c = torch.tensor([1.0])
    near_value = float(sdh_kernel.affinity(anchor, near, k_a, k_c))
    far_value = float(sdh_kernel.affinity(anchor, far, k_a, k_c))
    assert 0.0 < far_value < near_value <= 1.0


def test_affinity_is_detached_and_the_student_side_gradient_is_finite():
    anchor = torch.randn(2, 5)
    cands = torch.randn(2, 4, 5)
    k_a = torch.tensor([0.7, 1.3])
    k_c = torch.tensor([0.5, 0.9, 1.1, 1.7])
    teacher = sdh_kernel.affinity(anchor, cands, k_a, k_c)
    assert not teacher.requires_grad                         # a frozen teacher cannot leak gradient into the geometry

    logits = torch.randn(2, 4, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.bool)
    teacher_dist = sdh_kernel.teacher_distribution(teacher, mask)
    loss = sdh_kernel.kl_within_support(logits, teacher_dist, mask)
    loss.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert float(loss.detach()) >= -1e-6


def test_kl_is_zero_when_the_student_already_matches_the_teacher():
    teacher = torch.tensor([[0.25, 0.75]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    logits = torch.log(teacher).clone().requires_grad_(True)
    assert float(sdh_kernel.kl_within_support(logits, teacher, mask)) == pytest.approx(0.0, abs=1e-6)


def test_gradcheck_on_the_student_side_of_the_kl():
    torch.manual_seed(0)
    teacher = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)

    def f(logits):
        return sdh_kernel.kl_within_support(logits, teacher, mask)

    logits = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (logits,), eps=1e-6, atol=1e-4)


def test_an_anchor_with_no_supported_pair_contributes_nothing_instead_of_a_nan():
    teacher = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = sdh_kernel.kl_within_support(logits, teacher, mask)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())
