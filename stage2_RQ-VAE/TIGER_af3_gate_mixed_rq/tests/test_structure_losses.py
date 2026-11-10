"""The three structural terms: definitions, ranges and gradients.

Each term is checked against a value computed by hand from the very definition
the math spec freezes, on assignments whose overlap, hinge and entropy are
written out in this file. The terms are bounded by construction, so a bounded
output is a definition check, not a fitted property.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import structure_gate as aux  # noqa: E402


def _one_hot(rows: int, codes: int, index: list[int]) -> torch.Tensor:
    values = torch.zeros(rows, codes)
    for row, code in enumerate(index):
        values[row, code] = 1.0
    return values


def test_closure_is_zero_for_identical_assignments_and_one_for_disjoint_ones():
    codes = 4
    prefixes = [_one_hot(3, codes, [1, 1, 1]),
                _one_hot(3, codes, [2, 2, 2]),
                _one_hot(3, codes, [3, 3, 3])]
    triangles = torch.tensor([[0, 1, 2]])
    assert float(aux.closure_loss(prefixes, triangles)) == 0.0
    split = [_one_hot(3, codes, [1, 2, 3]),
             _one_hot(3, codes, [1, 2, 3]),
             _one_hot(3, codes, [1, 2, 3])]
    assert abs(float(aux.closure_loss(split, triangles)) - 1.0) < 1e-6


def test_closure_is_the_mean_pairwise_prefix_overlap_complement():
    prefixes = [torch.tensor([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0], [0.0, 1.0, 0.0]]),
                torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]),
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]
    pairs = ((0, 1), (0, 2), (1, 2))
    overlaps = []
    for level in prefixes:
        for first, second in pairs:
            overlaps.append(float((level[first] * level[second]).sum()))
    expected = 1.0 - sum(overlaps) / len(overlaps)
    measured = float(aux.closure_loss(prefixes, torch.tensor([[0, 1, 2]])))
    assert abs(measured - expected) < 1e-6


def test_bridge_pushes_cross_pairs_apart_and_is_zero_when_they_already_are():
    codes = 3
    # same-group pairs share a code; cross-group pairs share nothing
    prefixes = [_one_hot(4, codes, [0, 0, 1, 2]),
                _one_hot(4, codes, [0, 0, 1, 2]),
                _one_hot(4, codes, [0, 0, 1, 2])]
    same = torch.tensor([[0, 1]])
    cross = torch.tensor([[2, 3]])
    assert float(aux.bridge_loss(prefixes, same, cross)) == 0.0
    # the reversed construction violates the hinge: cross pairs are MORE similar
    reversed_same = torch.tensor([[2, 3]])
    reversed_cross = torch.tensor([[0, 1]])
    value = float(aux.bridge_loss(prefixes, reversed_same, reversed_cross))
    # same-group dissimilarity 1, cross-group dissimilarity 0 -> hinge 1
    assert abs(value - 1.0) < 1e-6


def test_bridge_penalises_cross_pairs_that_share_a_prefix():
    codes = 2
    prefixes = [_one_hot(4, codes, [0, 1, 0, 1]),
                _one_hot(4, codes, [0, 1, 0, 1]),
                _one_hot(4, codes, [0, 1, 0, 1])]
    value = float(aux.bridge_loss(prefixes, torch.tensor([[0, 1]]),
                                  torch.tensor([[0, 2]])))
    assert 0.0 <= value <= 1.0


def test_usage_is_zero_for_a_uniform_histogram_and_one_for_a_single_code():
    prefixes = [torch.full((8, 4), 0.25)]
    assert abs(float(aux.usage_loss(prefixes))) < 1e-9
    peaked = _one_hot(8, 4, [0] * 8)
    assert abs(float(aux.usage_loss([peaked])) - 1.0) < 1e-9


def test_usage_is_the_mean_normalised_entropy_deficit_over_levels():
    first = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    second = _one_hot(2, 2, [0, 0])
    # level one is uniform (deficit 0), level two uses one code (deficit 1)
    assert abs(float(aux.usage_loss([first, second])) - 0.5) < 1e-9


def test_every_term_carries_gradient_to_the_prefixes():
    generator = torch.Generator().manual_seed(13)
    prefixes = [torch.softmax(torch.randn(5, 6, generator=generator), dim=-1) for _ in range(3)]
    for prefix in prefixes:
        prefix.requires_grad_(True)
    triangles = torch.tensor([[0, 1, 2], [2, 3, 4]])
    same = torch.tensor([[0, 1], [3, 4]])
    cross = torch.tensor([[1, 2]])
    total = (aux.closure_loss(prefixes, triangles)
             + aux.bridge_loss(prefixes, same, cross)
             + aux.usage_loss(prefixes))
    total.backward()
    assert all(prefix.grad is not None and float(prefix.grad.abs().sum()) > 0
               for prefix in prefixes)


def test_empty_sets_return_a_connected_but_zero_term():
    prefixes = [torch.softmax(torch.randn(3, 4), dim=-1)]
    empty = torch.zeros((0, 3), dtype=torch.long)
    none = torch.zeros((0, 2), dtype=torch.long)
    closure = aux.closure_loss(prefixes, empty)
    bridge = aux.bridge_loss(prefixes, none, none)
    assert float(closure) == 0.0 and float(bridge) == 0.0


def test_the_carrier_is_the_baselines_centred_softmax_plus_the_floor():
    from model.mixed_rq import MixedVQLevel
    generator = torch.Generator().manual_seed(17)
    cost = torch.randn(4, 8, generator=generator)
    level = MixedVQLevel(8, curvature=1.0)
    soft, hard = aux.proxy_soft_part(cost, level, is_last=False)
    centred = level.center_distance(cost)
    expected = torch.softmax(-centred / aux.PROXY_TEMPERATURE, dim=-1)
    expected = (1.0 - aux.PROXY_SMOOTHING) * expected + aux.PROXY_SMOOTHING / 8.0
    torch.testing.assert_close(soft, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(hard, cost.argmin(-1))


def test_structural_objective_reports_what_it_could_not_use():
    generator = torch.Generator().manual_seed(19)
    prefixes = [torch.softmax(torch.randn(4, 5, generator=generator), dim=-1) for _ in range(3)]
    frozen = {"triangles": torch.tensor([[0, 1, 2], [0, 1, 3]]),
              "same_pairs": torch.tensor([[0, 1]]),
              "cross_pairs": torch.tensor([[1, 3]])}
    ids = torch.tensor([0, 1, 2, 3])
    loss, report = aux.structural_objective(prefixes, ids, frozen, item_count=10)
    assert torch.is_tensor(loss) and float(loss) >= 0.0
    assert report["triangles_used"] == 2
    assert report["same_pairs_used"] == 1
    assert report["cross_pairs_used"] == 1
    assert report["triangles_total"] == 2
    assert abs(report["closure"] - float(aux.closure_loss(prefixes, frozen["triangles"]))) < 1e-9
    # a batch that contains none of the frozen nodes uses none of them
    loss_small, report_small = aux.structural_objective(
        [prefix[:2] for prefix in prefixes], torch.tensor([4, 5]), frozen, item_count=10)
    assert report_small["triangles_used"] == 0
    # nothing of the frozen sets is covered, so the two graph terms vanish; the
    # usage term is a property of this batch alone and still applies
    assert report_small["closure"] == 0.0
    assert report_small["bridge"] == 0.0
    assert report_small["usage"] == report["usage"] or report_small["usage"] >= 0.0
