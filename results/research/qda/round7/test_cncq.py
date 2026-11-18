"""The three guarantees CNCQ claims, each with the case that would break it.

Deliberately included: the adversarial encoder whose outputs are all identical. Layer 1
still satisfies its capacity floor (it can always move items around), which is exactly why
layer 3 has to exist -- the gate must reject that model rather than report a pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cncq import (InfeasibleCapacity, Policy, bounded_codes, certify, certify_separation,
                  hard_capacity_assignment, mixed_min_distance, separation_bound)


def test_capacity_guarantee_holds_for_random_costs():
    generator = torch.Generator().manual_seed(3)
    for _ in range(5):
        cost = torch.rand(500, 20, generator=generator)
        assignment, report = hard_capacity_assignment(cost, 12)
        counts = torch.bincount(assignment, minlength=20)
        assert int(counts.min()) >= 12, report
        assert report["satisfied"] is True


def test_capacity_is_deterministic():
    generator = torch.Generator().manual_seed(4)
    cost = torch.rand(300, 10, generator=generator)
    first, _ = hard_capacity_assignment(cost, 7)
    second, _ = hard_capacity_assignment(cost, 7)
    assert torch.equal(first, second)


def test_infeasible_catalogue_raises_instead_of_partially_satisfying():
    cost = torch.rand(20, 10)
    try:
        hard_capacity_assignment(cost, 3)     # needs 30 items
    except InfeasibleCapacity:
        return
    raise AssertionError("an infeasible floor must raise, not return a short assignment")


def test_geometry_bound_survives_an_adversarial_offset():
    """Every offset is 1000x the prototype spacing; the clip still keeps them apart."""
    prototypes = torch.eye(16)
    bound = certify_separation(prototypes, rho=0.1)
    assert bound["ok"] and bound["margin"] > 0

    offsets = torch.randn(16, 16) * 1e3
    codes = bounded_codes(prototypes, offsets, 0.1)
    pairwise = torch.cdist(codes, codes)
    pairwise.fill_diagonal_(float("inf"))       # 0 * inf would be nan
    assert float(pairwise.min()) >= bound["delta"] - 2 * 0.1 - 1e-9


def test_geometry_reports_failure_when_rho_exceeds_the_half_spacing():
    prototypes = torch.eye(8)
    # the prototypes sit sqrt(2) apart, so anything above 0.707 breaks the bound
    assert certify_separation(prototypes, rho=0.8)["ok"] is False
    assert certify_separation(prototypes, rho=0.5)["ok"] is True


def test_hyperbolic_half_is_measured_not_assumed():
    """The mixed metric has no free triangle inequality, so the module measures it."""
    codes = np.concatenate([torch.eye(12).numpy(), np.zeros((12, 16))], axis=1)
    assert mixed_min_distance(codes) > 0


def test_a_degenerate_encoder_passes_capacity_and_is_rejected_by_the_gate():
    """The case the safety module exists for: all items encode to one vector."""
    items, codewords = 1000, 8
    cost = torch.zeros(items, codewords)                 # every codeword equally cheap
    assignment, report = hard_capacity_assignment(cost, 50)
    assert report["min_count"] >= 50                     # layer 1 is satisfied

    metrics = {
        "counts": torch.bincount(assignment, minlength=codewords).tolist(),
        "geometry": certify_separation(torch.eye(codewords), rho=0.1),
        "effective_rank": 1.0,                           # the collapsed representation
        "fidelity": 0.5,                                 # chance
    }
    reference = {"fidelity": 0.6592, "strata": {}}
    verdict = certify(metrics, reference)
    assert verdict["certified"] is False
    assert "representation" in verdict["failed"]
    assert "fidelity" in verdict["failed"]
    assert "capacity" not in verdict["failed"]


def test_a_healthy_model_passes_every_layer():
    items, codewords = 1000, 8
    generator = torch.Generator().manual_seed(5)
    cost = torch.rand(items, codewords, generator=generator)
    assignment, _ = hard_capacity_assignment(cost, 50)
    metrics = {
        "counts": torch.bincount(assignment, minlength=codewords).tolist(),
        "geometry": certify_separation(torch.eye(codewords), rho=0.1),
        "effective_rank": 12.0,
        "fidelity": 0.6600,
        "strata": {"bridge": 0.6600, "intra_community": 0.6550},
    }
    reference = {"fidelity": 0.6592,
                 "strata": {"bridge": 0.6595, "intra_community": 0.6545}}
    verdict = certify(metrics, reference)
    assert verdict["certified"] is True, verdict["failed"]


def test_the_stratum_gate_fires_on_a_bridge_regression():
    reference = {"fidelity": 0.6592, "strata": {"bridge": 0.6595}}
    metrics = {"counts": [100] * 8, "geometry": {"ok": True, "margin": 1.0, "delta": 1.0,
                                                "two_rho": 0.0},
               "effective_rank": 12.0, "fidelity": 0.6590, "strata": {"bridge": 0.6400}}
    verdict = certify(metrics, reference)
    assert verdict["certified"] is False
    assert "stratum:bridge" in verdict["failed"]
