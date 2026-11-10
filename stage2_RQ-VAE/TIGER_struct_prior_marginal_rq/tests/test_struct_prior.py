"""The structure-stratified target marginal: the one internal quantity this arm changes.

The oracles here are the definitions, written out independently: a column vector
that sums to one and is constant inside a bucket, a bucket partition of equal
sizes, and the baseline's own Sinkhorn as the uniform-marginal special case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import struct_prior as prior  # noqa: E402
from model.mixed_rq import MixedVQLevel  # noqa: E402
from model.poincare_contract import squared_distance  # noqa: E402


def _cost(rows: int, codes: int, seed: int = 150):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, codes, generator=generator, dtype=torch.float64)


def test_uniform_marginal_reproduces_the_baseline_sinkhorn():
    cost = _cost(64, 32)
    level = MixedVQLevel(32, curvature=1.0, sk_epsilon=0.003, sk_iters=50)
    centred = level.center_distance(cost).double()
    marginal = torch.full((32,), 1.0 / 32.0, dtype=torch.float64)
    mine = prior.structured_sinkhorn(centred, 0.003, 50, marginal)
    reference = level.sinkhorn(centred, 0.003, 50)
    reference = reference / reference.sum(-1, keepdim=True)
    torch.testing.assert_close(mine / mine.sum(-1, keepdim=True), reference,
                               rtol=1e-9, atol=1e-11)


def test_the_declared_marginal_is_reached_when_the_iteration_converges():
    """In the smooth regime the plan really carries ``batch * pi`` as its columns."""
    codes, rows = 32, 96
    cost = _cost(rows, codes)
    level = MixedVQLevel(codes, curvature=1.0)
    centred = level.center_distance(cost).double()
    rho = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    partition = prior.equal_partition(codes, 4)
    marginal = prior.target_marginal(partition, rho)
    plan = prior.structured_sinkhorn(centred, 2.0, 400, marginal)
    torch.testing.assert_close(plan.sum(-1), torch.ones(rows, dtype=torch.float64),
                               rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(plan.sum(0), rows * marginal, rtol=0, atol=1e-3)


def test_at_the_frozen_epsilon_the_bucket_mass_follows_the_structure_shares():
    """At the run's own epsilon the plan is sharp; the ordering is what survives."""
    codes, rows = 32, 512
    cost = _cost(rows, codes)
    level = MixedVQLevel(codes, curvature=1.0, sk_epsilon=0.003, sk_iters=50)
    centred = level.center_distance(cost).double()
    rho = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    partition = prior.equal_partition(codes, 4)
    marginal = prior.target_marginal(partition, rho)
    plan = prior.structured_sinkhorn(centred, 0.003, 50, marginal)
    bucket_mass = [float(plan[:, torch.as_tensor(bucket, dtype=torch.long)].sum())
                   for bucket in partition]
    target = [rows * float(share) for share in rho]
    assert bucket_mass == sorted(bucket_mass), bucket_mass
    for measured, wanted in zip(bucket_mass, target):
        assert abs(measured - wanted) <= 0.30 * wanted, (measured, wanted)
    assert float(marginal.sum()) == 1.0


def test_buckets_are_equal_sized_and_cover_every_code():
    partition = prior.equal_partition(256, 4)
    sizes = [len(bucket) for bucket in partition]
    assert sizes == [64, 64, 64, 64]
    flat = sorted(code for bucket in partition for code in bucket)
    assert flat == list(range(256))


def test_usage_driven_partition_gives_each_layer_the_codes_it_uses():
    codes, layers = 8, 2
    usage = torch.zeros(layers, codes)
    usage[0, :4] = torch.tensor([9.0, 8.0, 7.0, 6.0])
    usage[1, :] = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 2.0])
    partition = prior.usage_partition(usage, layers, bucket_size=codes // layers)
    assert sorted(partition[0]) == [0, 1, 2, 3]
    assert sorted(partition[1]) == [4, 5, 6, 7]
    # a cross-layer collision is resolved by layer order, never by reuse
    usage[0] = 1.0
    partition = prior.usage_partition(usage, layers, bucket_size=codes // layers)
    assert sorted(partition[0]) == [0, 1, 2, 3]
    assert sorted(partition[1]) == [4, 5, 6, 7]


def test_behavior_off_flattens_the_marginal(monkeypatch):
    partition = prior.equal_partition(8, 2)
    rho = torch.tensor([0.9, 0.1], dtype=torch.float64)
    skewed = prior.target_marginal(partition, rho)
    assert float(skewed[0]) > float(skewed[-1])
    monkeypatch.setattr(prior, "BEHAVIOR_OFF", True)
    flat = prior.target_marginal(partition, rho)
    torch.testing.assert_close(flat, torch.full((8,), 1.0 / 8.0, dtype=torch.float64))


def test_the_three_terms_are_bounded_and_carry_gradient():
    codes = 8
    rows = 6
    generator = torch.Generator().manual_seed(7)
    proxies = [torch.softmax(torch.randn(rows, codes, generator=generator), dim=-1)
               for _ in range(3)]
    for proxy in proxies:
        proxy.requires_grad_(True)
    triangles = torch.tensor([[0, 1, 2]])
    within = torch.tensor([[0, 1], [2, 3]])
    cross = torch.tensor([[1, 4]])
    partition = prior.equal_partition(codes, 2)
    rho = torch.tensor([0.5, 0.5], dtype=torch.float64)
    marginal = prior.target_marginal(partition, rho)
    total = (prior.triangle_loss(proxies, triangles)
             + prior.fork_loss(proxies, within, cross)
             + prior.prior_kl(proxies, marginal))
    assert 0.0 <= float(prior.triangle_loss(proxies, triangles).detach()) <= 1.0
    assert 0.0 <= float(prior.fork_loss(proxies, within, cross).detach()) <= 1.0
    assert 0.0 <= float(prior.prior_kl(proxies, marginal).detach()) <= 1.0
    total.backward()
    assert all(proxy.grad is not None and float(proxy.grad.abs().sum()) > 0
               for proxy in proxies)


def test_prior_kl_is_zero_when_the_usage_already_matches_the_marginal():
    codes = 8
    plan = torch.zeros(4, codes)
    marginal = torch.full((codes,), 1.0 / codes)
    plan[:] = marginal
    value = prior._kl(plan.sum(0) / plan.shape[0], marginal)
    assert abs(float(value)) < 1e-9


def test_stratification_uses_tau_quartiles_and_frozen_proportions():
    tau = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    layer, rho = prior.stratify(tau, groups=4)
    assert layer.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]
    assert abs(float(rho.sum()) - 1.0) < 1e-12
    assert rho.tolist() == [0.3, 0.2, 0.2, 0.3]


def test_geometry_off_has_a_euclidean_hyperbolic_half():
    generator = torch.Generator().manual_seed(11)
    first = 0.1 * torch.randn(8, 16, generator=generator, dtype=torch.float64)
    second = 0.1 * torch.randn(8, 16, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(squared_distance(first, second, 0.0) / 4.0,
                               (first - second).pow(2).sum(-1), rtol=0, atol=1e-6)


def test_switch_constants_exist():
    assert prior.BEHAVIOR_OFF is False
    assert prior.GEOMETRY_OFF is False
    assert prior.GROUPS == 4
    assert prior.LAMBDA_TRIANGLE == 1.0
    assert prior.LAMBDA_FORK == 1.0
    assert prior.LAMBDA_PRIOR == 1.0
