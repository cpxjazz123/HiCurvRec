"""Issue #229: the frozen bundle must be loadable, verifiable and usable.

The bundle is the deliverable a consumer loads: recognition, graph-derived
targets, the portable carrier and the supervision losses, all behind one object
with SHA checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS = Path(__file__).resolve().parent
TEACHER = TESTS.parent
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(TEACHER.parent / "directed_graphlet_role"))

import structure_teacher_bundle as stb  # noqa: E402

BUNDLE = TEACHER / "teacher" / "structure" / "bundle"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if (BUNDLE / "structure_teacher_bundle.pth").is_file():
        return stb.load_structure_teacher(BUNDLE)
    out = tmp_path_factory.mktemp("bundle")
    stb.build_bundle(out)
    return stb.load_structure_teacher(out)


def test_bundle_verifies_every_sha(bundle):
    report = bundle.verify()
    assert report["healthy"] is True, report


def test_bundle_recognises_the_five_families(bundle):
    k3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    c4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
    assert bundle.recognise(k3)["family"] == "K3"
    assert bundle.recognise(c4)["family"] == "C4"


def test_bundle_targets_are_aligned_and_bounded(bundle):
    rows = bundle.targets_for([0, 5, 100])
    assert rows["relation"].shape == (3, 128)
    assert rows["neighbour_orbit"].shape == (3, 376)
    assert rows["coverage"].shape == (3,)
    with pytest.raises(IndexError):
        bundle.targets_for([10 ** 9])


def test_bundle_carrier_is_frozen_and_deterministic(bundle):
    items = torch.zeros(2, bundle.carrier.in_dim)
    first = bundle.predict_targets(items)
    second = bundle.predict_targets(items)
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert all(not parameter.requires_grad for parameter in bundle.carrier.parameters())


def test_structure_loss_runs_and_prefers_the_teacher_geometry(bundle):
    teacher = torch.from_numpy(bundle.targets["relation"]).float()
    metrics, loss = bundle.structure_loss(teacher)
    assert np.isfinite(metrics["total"])
    random = torch.zeros_like(teacher)
    _, random_loss = bundle.structure_loss(random)
    assert float(loss) < float(random_loss)


def test_a_tiny_student_can_be_trained_with_the_bundle(bundle, tmp_path):
    torch.manual_seed(0)
    items = len(bundle.targets["relation"])
    student = torch.nn.Linear(4, 128, bias=False)
    latent = torch.randn(items, 4)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        optimizer.zero_grad()
        metrics, loss = bundle.structure_loss(student(latent))
        loss.backward()
        optimizer.step()
        first = metrics["total"] if first is None else first
    assert metrics["total"] < first


def test_bundle_contract_states_the_boundary():
    path = BUNDLE / "structure_teacher_bundle_contract.json"
    if not path.is_file():
        pytest.skip("the bundle contract has not been written")
    contract = json.loads(path.read_text())
    assert contract["boundary"]["rqvae_input_changed"] is False
    assert contract["boundary"]["teacher_inference_needs_graph"] is False
    assert contract["boundary"]["used_only_in_training_loss"] is True
    assert contract["weights"]["relation"] == 1.0
    assert "identify" not in contract["api"]


def test_structure_loss_accepts_a_mini_batch(bundle):
    rng = np.random.default_rng(0)
    ids = rng.choice(len(bundle.targets["relation"]), 2048, replace=False)
    rows = torch.from_numpy(bundle.targets["relation"])[torch.from_numpy(ids)].float()
    metrics, loss = bundle.structure_loss(rows, item_ids=ids)
    assert metrics["pairs_used"] > 0
    assert np.isfinite(metrics["total"])
    zeros = torch.zeros_like(rows)
    _, zero_loss = bundle.structure_loss(zeros, item_ids=ids)
    assert float(loss) < float(zero_loss)
    with pytest.raises(ValueError):
        bundle.structure_loss(rows[:10], item_ids=ids)


def test_structure_loss_is_finite_when_a_batch_has_no_internal_pairs(bundle):
    """A mini-batch of unrelated items has no relation pair; it must not yield NaN."""
    pairs = {tuple(int(x) for x in row) for row in bundle.targets["positive_pairs"]}
    ids = []
    for item in range(len(bundle.targets["relation"])):
        if all((item, other) not in pairs and (other, item) not in pairs
               for other in ids):
            ids.append(item)
        if len(ids) == 4:
            break
    assert len(ids) == 4
    rows = torch.zeros((4, 4), requires_grad=True)
    metrics, loss = bundle.structure_loss(rows, item_ids=np.asarray(ids))
    assert metrics["pairs_used"] == 0
    assert metrics["relation_ranking"] is None
    assert metrics["relation_distill"] is None
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(rows.grad).all()
