"""Small CPU regression checks for research Idea mechanisms #22--#27.

Run from the repository root: python scripts/check_research_ideas_22_27.py
These are synthetic mechanism checks, not Stage2 or Stage3 retraining.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "stage2_RQ-VAE"
NAMES = (
    "semantic_bounded_residual",
    "relational_structure_distill",
    "uncertainty_behavioral",
    "multi_intent_disentangled",
    "invertible_hyperbolic_flow",
    "multiscale_behavioral_spectrum",
)


def module(name):
    folder = ROOT / ("TIGER_" + name)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            "_audit_" + name, folder / (name + ".py")
        )
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.pop(0)


def test_bounded_first_layer():
    m = module(NAMES[0])
    torch.manual_seed(41)
    anchor = torch.randn(24, 8) * 0.06
    codebook = torch.randn(15, 8) * 0.08
    before = torch.cdist(anchor, codebook).argmin(dim=1)
    budget = m._perturbation_budget(anchor, codebook)
    corrections = torch.randn_like(anchor) * 1.5
    after_point = m._expmap_x(anchor, m._bound_to_budget(corrections, anchor, budget))
    after = torch.cdist(after_point, codebook).argmin(dim=1)
    assert torch.equal(before, after), "first-layer Euclidean nearest code changed"
    assert torch.isfinite(after_point).all()


def test_relational_teacher():
    m = module(NAMES[1])
    histories = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        [0, 1, 2, 3, 4, 5, 6, 9, 10],
        [0, 2, 3, 4, 5, 6, 7, 9, 10],
    ]
    teacher = m.build_relational_teacher(histories, 12)
    assert teacher["anchor"].size
    assert np.isfinite(teacher["angle_target"]).all()
    assert np.isfinite(teacher["ratio_target"]).all()
    triples = m.teacher_triples(teacher, 32, torch.Generator().manual_seed(9),
                               torch.device("cpu"))
    assert (triples["negative"] != triples["anchor"]).all()
    assert (triples["negative"] != triples["strong"]).all()


def test_wrapped_noise_transport():
    m = module(NAMES[2])
    torch.manual_seed(3)
    base = torch.randn(8, 6) * 0.05
    v = torch.randn(8, 6) * 0.2
    moved = m._expmap_x(base, v)
    dist = m._geodesic(base, moved)
    assert torch.allclose(dist, v.norm(dim=-1), atol=2e-4), (dist, v.norm(dim=-1))
    samples = m.wrapped_normal_sample(
        base, torch.full((8,), 0.3), torch.Generator().manual_seed(8)
    )
    assert torch.isfinite(samples).all()
    assert (samples.norm(dim=-1) < 1.0).all()


def test_intent_pair_sampling():
    m = module(NAMES[3])
    pairs = np.array([[7, 2], [3, 8], [7, 5], [3, 9], [7, 1]])
    view = m.build_intent_views({"transition": pairs}, 15)["transition"]
    a, p, n = m.view_triples(view, 80, torch.Generator().manual_seed(8),
                             torch.device("cpu"))
    valid = set(map(tuple, pairs))
    assert all((int(x), int(y)) in valid for x, y in zip(a, p))
    assert (a != n).all()
    flow_points = [m._expmap0(torch.randn(3, 8) * 0.1) for _ in range(3)]
    weights = torch.softmax(torch.randn(3, 3), dim=-1)
    fused, _ = m.fuse_intents(flow_points, weights)
    assert torch.isfinite(fused).all()


def test_flow_adjacency_and_inverse():
    m = module(NAMES[4])
    pairs = np.array([[8, 1], [2, 4], [8, 6], [2, 7]])
    view = m.build_neighbour_pairs(pairs, 12)
    a, p = m.neighbour_batch(view, 60, torch.Generator().manual_seed(8),
                             torch.device("cpu"))
    valid = set(map(tuple, pairs))
    assert all((int(x), int(y)) in valid for x, y in zip(a, p))
    model = m.HyperbolicCouplingFlow(8, 4)
    x = torch.randn(12, 8) * 0.06
    result = m.hyperbolic_flow_inverse(m.hyperbolic_flow_forward(m._expmap0(x), model), model)
    assert torch.allclose(result, m._expmap0(x), atol=1e-5)


def test_graph_spectrum():
    m = module(NAMES[5])
    histories = [[0, 1, 2], [0, 1], [2, 3], [4, 5],
                 [3, 4], [5, 6], [6, 7], [1, 7], [2, 6]]
    spectrum = m.build_behaviour_spectrum(histories, 8, 8)
    data = spectrum["cumulative"]
    assert data.shape == (3, 8, 8)
    assert np.isfinite(data).all()
    assert np.linalg.norm(data[2, 0] - data[2, 5]) > 1e-4
    assert np.linalg.norm(data[0] - data[1]) > 1e-5
    target = torch.from_numpy(data[2])
    predicted = m._expmap0(torch.randn(8, 8, requires_grad=True) * 0.1)
    loss = m.band_relation_loss(predicted, target)
    assert torch.isfinite(loss)
    loss.backward()


if __name__ == "__main__":
    tests = (
        test_bounded_first_layer,
        test_relational_teacher,
        test_wrapped_noise_transport,
        test_intent_pair_sampling,
        test_flow_adjacency_and_inverse,
        test_graph_spectrum,
    )
    for test in tests:
        test()
        print("PASS", test.__name__)
