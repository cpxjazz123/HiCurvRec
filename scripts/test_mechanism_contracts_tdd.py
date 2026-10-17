"""RED/GREEN TDD contracts for six continuous hyperbolic RQ-VAE mechanisms.

Run independently: python scripts/test_mechanism_contracts_tdd.py
Tests are intentionally synthetic: no dataset, checkpoint or GPU required.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / "stage2_RQ-VAE"
NAMES = (
    "semantic_bounded_residual",
    "relational_structure_distill",
    "uncertainty_behavioral",
    "multi_intent_disentangled",
    "invertible_hyperbolic_flow",
    "multiscale_behavioral_spectrum",
)


def load(name: str, file: str):
    folder = ROOT / ("TIGER_" + name)
    # The mechanism module imports the neighbouring hyperbolic.py file.
    sys.path.insert(0, str(folder))
    previous = sys.modules.pop("hyperbolic", None)
    try:
        spec = importlib.util.spec_from_file_location("contract_" + name + "_" + file, folder / file)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)
        sys.modules.pop("hyperbolic", None)
        if previous is not None:
            sys.modules["hyperbolic"] = previous


def test_all_six_same_item_independent_of_other_batch_members():
    torch.manual_seed(7)
    item = torch.randn(1, 12) * 0.3
    background_small = torch.randn(5, 12) * 0.01
    background_large = torch.randn(5, 12) * 7
    for name in NAMES:
        g = load(name, "hyperbolic.py")
        z_alone = g.encode_ball(item, 1.0, 0.5)
        z_small = g.encode_ball(torch.cat([item, background_small]), 1.0, 0.5)[:1]
        z_large = g.encode_ball(torch.cat([item, background_large]), 1.0, 0.5)[:1]
        assert torch.allclose(z_alone, z_small, atol=1e-7), name
        assert torch.allclose(z_alone, z_large, atol=1e-7), name
        assert g.encode_ball(item * 2, 1.0, 0.5).norm() > z_alone.norm(), name


def test_all_six_behaviour_recovery_updates_real_decoder_not_codebooks():
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(6, 4)
            self.decoder = nn.Linear(4, 6)

    for name in NAMES:
        m = load(name, name + ".py")
        torch.manual_seed(11)
        net = MiniModel()
        aux = nn.Linear(4, 4)
        codebook = nn.Parameter(torch.randn(3, 4))
        inputs = torch.randn(3, 6)
        latent = net.encoder(inputs)
        # The unchanged quantizer's STE: codebook contributes forward values but
        # is detached on the encoder/decoder behaviour gradient route.
        codewords = codebook.detach()
        q = latent + (codewords - latent).detach()
        rebuilt_latent = m.decoder_reencoded_latent(net, q)
        behavior_loss = aux(rebuilt_latent).square().mean()
        behavior_loss.backward()
        assert net.encoder.weight.grad is not None and net.encoder.weight.grad.abs().sum() > 0, name
        assert net.decoder.weight.grad is not None and net.decoder.weight.grad.abs().sum() > 0, name
        assert codebook.grad is None, name


def test_idea1_safety_budget_is_detached_and_nn_preserving():
    m = load(NAMES[0], NAMES[0] + ".py")
    torch.manual_seed(77)
    anchors = torch.randn(30, 8, requires_grad=True) * 0.05
    cb = nn.Parameter(torch.randn(11, 8) * 0.06)
    budget = m._perturbation_budget(anchors, cb)
    assert not budget.requires_grad
    corrections = torch.randn_like(anchors)
    transformed = m._expmap_x(anchors, m._bound_to_budget(corrections, anchors, budget))
    assert torch.equal(torch.cdist(anchors, cb).argmin(-1),
                       torch.cdist(transformed, cb).argmin(-1))


def test_idea2_teacher_preserves_valid_triangle_and_encoder_geometry():
    m = load(NAMES[1], NAMES[1] + ".py")
    histories = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
                 [0, 2, 3, 4, 5, 6, 7, 9, 10]]
    teacher = m.build_relational_teacher(histories, 11)
    a = teacher["delta_strong"]
    b = teacher["delta_weak"]
    # Angle targets must arise from valid Hilbert-space chord triangles.
    assert np.isfinite(teacher["angle_target"]).all()
    assert np.all(teacher["angle_target"] >= 0)
    assert np.all(teacher["angle_target"] <= np.pi)
    assert (a > 0).all() and (b > 0).all()


def test_idea3_uncertainty_grows_when_reliability_falls():
    m = load(NAMES[2], NAMES[2] + ".py")
    reliability = torch.tensor([0.0, 0.25, 0.75, 1.0])
    target = m.sigma_target(reliability)
    assert (target[:-1] > target[1:]).all()
    points = torch.randn(4, 6) * 0.03
    samples = m.wrapped_normal_sample(points, target, torch.Generator().manual_seed(81))
    assert torch.isfinite(samples).all()


def test_idea4_intent_latents_are_low_dimensional_and_fuse_to_original_dim():
    m = load(NAMES[3], NAMES[3] + ".py")
    gate = m.MultiIntentGate(12, 20, intents=3)
    assert all(branch.out_features < 12 for branch in gate.branches)
    latent = torch.randn(5, 12)
    text = torch.randn(5, 20)
    mixed, weights, tangents = gate(latent, text)
    assert mixed.shape == (5, 12)
    assert all(t.shape[-1] < 12 for t in tangents)
    points = gate.intent_points(tangents)
    fused, recovered = m.fuse_intents(points, weights, projections=gate.projections)
    assert fused.shape == (5, 12)
    assert len(recovered) == 3


def test_idea5_flow_changes_both_coordinate_halves_and_inverts():
    m = load(NAMES[4], NAMES[4] + ".py")
    flow = m.HyperbolicCouplingFlow(8, 4)
    for layer in flow.layers:
        with torch.no_grad():
            layer.shift[-1].bias.fill_(0.07)
    x = torch.randn(12, 8) * 0.04
    out = flow(x)
    assert not torch.allclose(out[:, :4], x[:, :4], atol=1e-8)
    assert not torch.allclose(out[:, 4:], x[:, 4:], atol=1e-8)
    assert torch.allclose(flow.inverse(out), x, atol=1e-6)


def test_idea6_each_depth_supervised_once():
    m = load(NAMES[5], NAMES[5] + ".py")
    depth_losses = torch.tensor([0.3, 0.7, 1.2], requires_grad=True)
    behavior, recovery = m.compose_spectrum_supervision(depth_losses, 0.8, 0.5)
    assert torch.allclose(behavior + recovery, 0.8 * 0.5 * depth_losses.mean())
    assert torch.allclose(behavior, torch.zeros_like(behavior))
    (behavior + recovery).backward()
    assert depth_losses.grad is not None


if __name__ == "__main__":
    checks = [test_all_six_same_item_independent_of_other_batch_members,
              test_all_six_behaviour_recovery_updates_real_decoder_not_codebooks,
              test_idea1_safety_budget_is_detached_and_nn_preserving,
              test_idea2_teacher_preserves_valid_triangle_and_encoder_geometry,
              test_idea3_uncertainty_grows_when_reliability_falls,
              test_idea4_intent_latents_are_low_dimensional_and_fuse_to_original_dim,
              test_idea5_flow_changes_both_coordinate_halves_and_inverts,
              test_idea6_each_depth_supervised_once]
    failures = []
    for check in checks:
        try:
            check()
            print("PASS", check.__name__, flush=True)
        except Exception as exc:
            print("FAIL", check.__name__, repr(exc), flush=True)
            failures.append(check.__name__)
    if failures:
        raise SystemExit("RED: " + ", ".join(failures))
