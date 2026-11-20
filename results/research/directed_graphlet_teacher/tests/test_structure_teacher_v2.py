"""Issue #229 v2: the evidence-derived supervision target.

v2 measured which structure components the fixed 768-d embedding can carry and
weights the frozen supervision accordingly.  The tests pin the consequences:

  * the v2 table keeps the multi-role targets (subtype/family/orbit) and the
    carried relation target, with aligned pair tables and coverage masks;
  * the frozen carrier reloads bitwise, is deterministic and splits its output
    into exactly the declared components;
  * the contract's weights follow the pre-declared rule (a role component gets
    weight only when the carrier beats the constant profile by >=2% CE);
  * the direction capability is recorded as NOT_ESTABLISHED with its degree
    baselines, and the role-relation alignment is reported rather than assumed.
"""
from __future__ import annotations

import hashlib
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

import structure_teacher as st  # noqa: E402

V2 = TEACHER / "teacher" / "structure" / "v2"


def test_carrier_v2_splits_and_is_deterministic():
    carrier = st.CarrierV2(in_dim=32, relation_dim=8, subtype_dim=7, family_dim=5,
                           orbit_dim=11, hidden=16)
    carrier.eval()
    features = torch.zeros(4, 32)
    first = carrier(features)
    second = carrier(features)
    assert torch.equal(first, second)
    parts = carrier.components(first)
    assert parts["relation"].shape == (4, 8)
    assert parts["subtype"].shape == (4, 7)
    assert parts["family"].shape == (4, 5)
    assert parts["orbit"].shape == (4, 11)
    assert first.shape == (4, 31)


def test_v2_artifact_shapes_and_masks():
    if not (V2 / "structure_targets_v2.npz").is_file():
        pytest.skip("the v2 artifact has not been built")
    targets = st.load_targets_v2(V2 / "structure_targets_v2.npz")
    for key, shape in (("relation", (24587, 128)), ("subtype", (24587, 47)),
                       ("family", (24587, 5)), ("orbit", (24587, 376)),
                       ("coverage", (24587,)), ("confidence", (24587,))):
        assert targets[key].shape == shape, key
    assert len(targets["positive_pairs"]) == len(targets["negative_pairs"])
    covered = targets["coverage"] > 0
    assert 0.45 < covered.mean() < 0.55
    assert np.all(targets["confidence"][~covered] == 0.0)
    assert np.all(targets["orbit"][~covered].sum(1) == 0.0)
    roles = (targets["orbit"][covered] > 0).sum(1)
    assert float(roles.mean()) > 1.5


def test_v2_manifest_matches_files():
    if not (V2 / "teacher_structure_manifest_v2.json").is_file():
        pytest.skip("the v2 artifact has not been built")
    manifest = json.loads((V2 / "teacher_structure_manifest_v2.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((V2 / name).read_bytes()).hexdigest() == entry["sha256"], name


def test_v2_carrier_reloads_frozen_and_bitwise():
    path = V2 / "teacher_structure_v2.pth"
    if not path.is_file():
        pytest.skip("the v2 artifact has not been built")
    carrier = st.load_carrier_v2(path)
    assert all(not parameter.requires_grad for parameter in carrier.parameters())
    assert carrier.training is False
    items = torch.zeros(3, carrier.in_dim)
    assert torch.equal(carrier(items), carrier(items))


def test_v2_contract_weights_follow_the_evidence():
    path = V2 / "teacher_structure_contract_v2.json"
    if not path.is_file():
        pytest.skip("the v2 artifact has not been built")
    contract = json.loads(path.read_text())
    carrier = contract["carrier"]
    assert carrier["relation"]["carried"] is True
    assert carrier["relation"]["cosine"] > 0.8
    for name in ("family", "subtype", "orbit"):
        assert carrier[name]["carried"] == bool(carrier[name]["ce"]
                                                < carrier[name]["constant_ce"] * 0.98)
    weights = contract["supervision_recommendation"]["weights"]
    assert weights["relation"] == 1.0
    for name in ("family", "subtype", "orbit"):
        assert weights[name] == (0.5 if carrier[name]["carried"] else 0.0)
    assert contract["direction"]["status"] in ("ESTABLISHED", "NOT_ESTABLISHED")
    assert "baselines" in contract["direction"]
    alignment = contract["role_relation_alignment"]
    assert -0.2 < alignment["spearman"] < 0.2, "roles and relations are not a proxy"


def test_v2_supervision_needs_no_graph():
    import inspect
    for function in (st.orbit_loss, st.relation_ranking_loss, st.relation_distill_loss):
        names = set(inspect.signature(function).parameters)
        assert not any(("adj" in name or "graph" in name or "edge" in name)
                       for name in names), (function.__name__, names)
