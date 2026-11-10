"""The contract binds the 64-scalar 16-dimension mixed_sq on both phi branches."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

import mixed_geometry as MG
from tools.research.stage2.reference import build_reference

E = 16


def calls():
    contract = json.loads((Path(__file__).resolve().parents[1] / "reference_contract.json").read_text())
    return [c for c in contract["calls"] if c.get("callable") == "mixed_geometry:mixed_sq"]


def ball(n, seed, radius=0.9):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, E, generator=g, dtype=torch.float64)
    return h / h.norm(dim=-1, keepdim=True) * (radius * torch.rand(n, 1, generator=g, dtype=torch.float64))


def pairs(seed, left_ball, right_ball, share_h=False):
    g = torch.Generator().manual_seed(seed)
    left = torch.randn(32, 32, generator=g, dtype=torch.float64)
    right = torch.randn(32, 32, generator=g, dtype=torch.float64)
    left[:, E:] = ball(32, left_ball)
    right[:, E:] = left[:, E:] + 1e-6 / (E ** 0.5) if share_h else ball(32, right_ball)
    if share_h:
        right[:, :E] += 1e-3
    return left, right


def test_both_branches_match_mixed_sq_in_value_and_gradient():
    found = calls()
    assert len(found) == 2, "the contract must bind both phi branches"
    series = [c for c in found if "t < " in (c.get("domain") or "")][0]
    asym = [c for c in found if "t >= " in (c.get("domain") or "")][0]
    for call, (left, right), tol_v, tol_g in [
            (series, pairs(7, 11, 12, share_h=True), 1e-12, 1e-10),
            (asym, pairs(8, 21, 22), 1e-12, 1e-10)]:
        reference, _ = build_reference(call)
        a = left.clone().requires_grad_(True)
        b = right.clone().requires_grad_(True)
        expected = reference(left=a, right=b)
        got = MG.mixed_sq(left, right)
        assert torch.allclose(expected.detach(), got, rtol=0.0, atol=tol_v)
        g_ref = torch.autograd.grad(expected.sum(), (a, b), retain_graph=True)
        c = left.clone().requires_grad_(True)
        d = right.clone().requires_grad_(True)
        g_torch = torch.autograd.grad(MG.mixed_sq(c, d).sum(), (c, d), retain_graph=True)
        for one, two in zip(g_ref, g_torch):
            assert torch.allclose(one, two, rtol=0.0, atol=tol_g)


def test_the_euclidean_branch_is_the_same_metric_without_geometry():
    left, right = pairs(9, 31, 32)
    euclid = ((left[:, :E] - right[:, :E]) ** 2).sum(-1) + ((left[:, E:] - right[:, E:]) ** 2).sum(-1)
    assert torch.allclose(MG.mixed_sq(left, right, False), euclid, rtol=0.0, atol=1e-12)
    assert not torch.allclose(MG.mixed_sq(left, right, True), euclid, rtol=0.0, atol=1e-12)


def test_the_binding_is_sixteen_dimensions_not_a_reduced_surface():
    for call in calls():
        widths = {spec["name"]: spec.get("width") for spec in call["args"]}
        assert widths == {"left": 32, "right": 32}
        assert len(call["symbols"]) == 64
