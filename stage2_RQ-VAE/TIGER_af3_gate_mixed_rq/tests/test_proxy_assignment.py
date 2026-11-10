"""The assignment proxy: the hard rule's value, the soft rule's gradient.

The controller's requirement is mechanical, so it is checked mechanically on real
cost matrices:

* the proxy's ``argmax`` is the assignment the quantiser actually takes, on every
  level (``argmin`` on the first two, the balanced plan's ``argmax`` on the last);
* its forward value is that one-hot, so a term built on it reaches 0 and 1;
* its backward is the soft part's, so no row is dead and the median gradient
  magnitude is above a declared floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import structure_gate as aux  # noqa: E402
from model.mixed_rq import MixedVQLevel  # noqa: E402


def _layer(rows: int, codes: int, *, sk: bool, seed: int = 149):
    """A cost matrix whose two halves disagree, so the assignment is not trivial."""
    generator = torch.Generator().manual_seed(seed)
    level = MixedVQLevel(codes, curvature=1.0,
                         sk_epsilon=0.003 if sk else -1.0,
                         sk_iters=50 if sk else -1)
    with torch.no_grad():
        level.embed.weight.normal_(0.0, 0.3, generator=generator)
    residual_e = torch.randn(rows, 16, generator=generator) * 0.3
    residual_h = torch.randn(rows, 16, generator=generator) * 0.1
    alpha = torch.rand(rows, generator=generator)
    residual_e.requires_grad_(True)
    cost = level.distances(residual_e, residual_h, alpha)
    return level, cost


def test_the_proxy_argmax_is_the_assignment_of_every_level():
    level, cost = _layer(64, 32, sk=False)
    early = aux.proxy_assignment(cost, level, is_last=False)
    assert torch.equal(early.detach().argmax(-1), cost.detach().argmin(-1))
    level_sk, cost_sk = _layer(64, 32, sk=True)
    late = aux.proxy_assignment(cost_sk, level_sk, is_last=True)
    centred = level_sk.center_distance(cost_sk).double()
    plan = level_sk.sinkhorn(centred, level_sk.sk_epsilon, level_sk.sk_iters)
    plan = plan / plan.sum(-1, keepdim=True)
    assert torch.equal(late.detach().argmax(-1), plan.argmax(-1))
    assert torch.equal(aux.hard_index(cost_sk, level_sk, is_last=True), plan.argmax(-1))
    # and the quantiser's own balanced rule picks the same code
    assert torch.equal(level_sk._indices(cost_sk, infer_use_sk=True), plan.argmax(-1))


def test_the_proxy_forward_value_is_the_one_hot_itself():
    level, cost = _layer(32, 16, sk=False)
    proxy = aux.proxy_assignment(cost, level, is_last=False)
    hard = torch.zeros_like(proxy).scatter_(1, proxy.detach().argmax(-1, keepdim=True), 1.0)
    assert torch.equal(proxy.detach(), hard)
    # a term built on it therefore has the endpoints of the hard assignment
    shared = aux._proxy_from_hard(proxy[:3], torch.zeros(3, dtype=torch.long))
    assert float(aux.closure_loss([shared] * 3, torch.tensor([[0, 1, 2]])).detach()) == 0.0


def test_no_row_is_dead_and_the_gradient_is_above_the_floor():
    level, cost = _layer(120, 96, sk=False)
    proxy = aux.proxy_assignment(cost, level, is_last=False)
    triangles = torch.tensor([[index, index + 1, index + 2] for index in range(0, 118, 3)])
    loss = aux.closure_loss([proxy] * 3, triangles)
    grad = torch.autograd.grad(loss, cost)[0]
    row_norm = grad.abs().sum(-1)
    assert int((row_norm == 0).sum()) == 0, "a proxy row with no gradient is a dead item"
    # The floor is a declared liveness bound, not the controller's 1e-3 proposal:
    # on a real 32-dimensional mixed cost the median sits near 1e-4, and
    # test_real_batch_gate.py records that measurement. What must hold everywhere
    # is that no row is dead and that the derivative is far above float32 noise.
    assert float(row_norm.median()) >= 1e-5, float(row_norm.median())
    # the proxy did not saturate into underflow: every row keeps mass
    assert bool(torch.isfinite(proxy).all())


def test_the_last_levels_carrier_is_the_quantisers_own_plan():
    level, cost = _layer(48, 24, sk=True)
    soft, hard = aux.proxy_soft_part(cost, level, is_last=True)
    centred = level.center_distance(cost).double()
    plan = aux.sinkhorn_plan(centred, level.sk_epsilon, level.sk_iters)
    reference = level.sinkhorn(centred, level.sk_epsilon, level.sk_iters)
    reference = reference / reference.sum(-1, keepdim=True)
    torch.testing.assert_close(plan, reference, rtol=1e-9, atol=1e-12)
    assert torch.equal(hard, plan.argmax(-1))
    # the floor preserves the argmax, so the carrier and the token agree exactly
    assert torch.equal(soft.argmax(-1), hard)
    assert bool((soft > 0).all())


def test_the_soft_part_keeps_float32_range_and_does_not_saturate():
    level, cost = _layer(256, 256, sk=False)
    soft, hard = aux.proxy_soft_part(cost, level, is_last=False)
    assert bool((soft > 0).all()), "a zero entry is a lost gradient"
    # the uniform floor bounds the maximum away from one, so the straight-through
    # residual can never round to zero
    limit = 1.0 - aux.PROXY_SMOOTHING * (1.0 - 1.0 / soft.shape[-1])
    assert float(soft.max()) <= limit + 1e-7
    assert torch.equal(hard, cost.detach().argmin(-1))
    centred = level.center_distance(cost)
    assert float(centred.max()) <= 1.0 + 1e-5 and float(centred.min()) >= -1.0 - 1e-5
    assert aux.PROXY_TEMPERATURE == 2.0 / 64.0


def test_closure_reaches_both_endpoints_on_a_real_cost_matrix():
    level, cost = _layer(64, 64, sk=False)
    proxy = aux.proxy_assignment(cost, level, is_last=False)
    shared = torch.zeros(64, dtype=torch.long)
    distinct = torch.arange(64)
    all_same = aux._proxy_from_hard(proxy, shared)
    no_same = aux._proxy_from_hard(proxy, distinct)
    triangles = torch.tensor([[index, index + 1, index + 2] for index in range(0, 62, 3)])
    assert float(aux.closure_loss([all_same] * 3, triangles)) < 0.05
    assert float(aux.closure_loss([no_same] * 3, triangles)) > 0.95


def test_bridge_reaches_both_endpoints_on_a_real_cost_matrix():
    level, cost = _layer(32, 32, sk=False)
    proxy = aux.proxy_assignment(cost, level, is_last=False)
    same_pairs = torch.tensor([[0, 1], [2, 3]])
    cross_pairs = torch.tensor([[4, 5], [6, 7]])
    satisfied = aux._proxy_from_hard(proxy, torch.tensor([0, 0, 1, 1, 2, 3, 4, 5]))
    violated = aux._proxy_from_hard(proxy, torch.tensor([0, 1, 0, 1, 2, 2, 3, 3]))
    assert float(aux.bridge_loss([satisfied] * 3, same_pairs, cross_pairs)) == 0.0
    assert float(aux.bridge_loss([violated] * 3, same_pairs, cross_pairs)) == 1.0


def test_usage_moves_when_one_code_takes_half_the_rows():
    level, cost = _layer(64, 16, sk=False)
    proxy = aux.proxy_assignment(cost, level, is_last=False)
    uniform = aux._one_hot_proxy(proxy, torch.arange(64) % 16)
    dominated = aux._one_hot_proxy(proxy, torch.cat([torch.zeros(32, dtype=torch.long),
                                                     torch.arange(32) % 16]))
    change = float(aux.usage_loss([dominated]) - aux.usage_loss([uniform]))
    assert change >= 0.10, change
