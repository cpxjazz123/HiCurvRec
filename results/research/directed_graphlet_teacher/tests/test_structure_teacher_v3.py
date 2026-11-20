"""Issue #229 v3: the teacher whose multi-role target is actually carried.

v2 showed an item's **own** orbit histogram is not carried by its 768-d
embedding.  v3 changes the target to the item's **neighbourhood** structure,
which is smoother and is the natural multi-role context: the mean orbit/family
profile and the mean relation vector of the observed neighbours.  The tests pin
the claims that survived measurement:

  * `neighbour_orbit` is carried (CE below the constant profile, ranking
    correlation well above zero, enough minority-role items to be non-trivial);
  * `neighbour_family` is carried in CE but degenerate (one dominant family),
    so the non-degeneracy gate must keep its weight at zero;
  * `own_orbit` / `own_subtype` are fidelity records and must not be sold as
    carried (own_orbit CE is worse than the constant);
  * relation and neighbour_relation are carried by cosine;
  * recognition is callable deterministically from an induced adjacency;
  * the frozen carrier reloads bitwise and the supervision path takes no graph.
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

V3 = TEACHER / "teacher" / "structure" / "v3"


def test_recognise_subgraph_on_the_five_families():
    k3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    p3 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    c4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
    p4 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=bool)
    star = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    for matrix, family in ((k3, "K3"), (p3, "P3"), (c4, "C4"), (p4, "P4"),
                           (star, "K1_3")):
        result = st.recognise_subgraph(matrix)
        assert result["family"] == family, (result, family)
        assert result["key"] is not None
        assert result["orbits"] is not None


def test_v3_artifact_shapes_and_masks():
    if not (V3 / "structure_targets_v3.npz").is_file():
        pytest.skip("the v3 artifact has not been built")
    targets = st.load_targets_v3(V3 / "structure_targets_v3.npz")
    for key, shape in (("relation", (24587, 128)), ("neighbour_relation", (24587, 128)),
                       ("neighbour_family", (24587, 5)),
                       ("neighbour_orbit", (24587, 376)), ("own_orbit", (24587, 376)),
                       ("own_subtype", (24587, 47)), ("coverage", (24587,)),
                       ("coverage_neighbour", (24587,)),
                       ("coverage_family", (24587,)), ("confidence", (24587,))):
        assert targets[key].shape == shape, key
    assert len(targets["positive_pairs"]) == len(targets["negative_pairs"])
    assert 0.95 < float((targets["coverage_neighbour"] > 0).mean()) < 1.0
    assert 0.45 < float((targets["coverage"] > 0).mean()) < 0.55
    assert np.all(targets["confidence"][targets["coverage_neighbour"] == 0] == 0.0)


def test_v3_manifest_matches_files():
    if not (V3 / "teacher_structure_manifest_v3.json").is_file():
        pytest.skip("the v3 artifact has not been built")
    manifest = json.loads((V3 / "teacher_structure_manifest_v3.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((V3 / name).read_bytes()).hexdigest() == entry["sha256"], name


def test_v3_carrier_frozen_and_deterministic():
    path = V3 / "teacher_structure_v3.pth"
    if not path.is_file():
        pytest.skip("the v3 artifact has not been built")
    carrier = st.load_carrier_v3(path)
    assert all(not parameter.requires_grad for parameter in carrier.parameters())
    assert carrier.training is False
    items = torch.zeros(2, carrier.in_dim)
    assert torch.equal(carrier(items), carrier(items))


def test_v3_weights_follow_the_rule_and_the_non_degeneracy_gate():
    path = V3 / "teacher_structure_contract_v3.json"
    if not path.is_file():
        pytest.skip("the v3 artifact has not been built")
    contract = json.loads(path.read_text())
    carrier = contract["carrier"]
    weights = contract["supervision_recommendation"]["weights"]
    assert carrier["relation"]["carried"] is True
    assert carrier["neighbour_relation"]["carried"] is True
    assert carrier["neighbour_orbit"]["carried"] is True
    assert carrier["neighbour_orbit"]["minority_items"] >= 50
    assert carrier["neighbour_orbit"]["role_rank_correlation"] > 0.3
    assert carrier["own_orbit"]["carried"] is False
    assert carrier["own_subtype"]["carried"] is False
    assert weights["relation"] == 1.0
    assert weights["neighbour_relation"] == 0.5
    assert weights["neighbour_orbit"] == 0.5
    assert weights["neighbour_family"] == 0.0, "single dominant class is not multi-role"
    assert weights["own_orbit"] == 0.0
    assert weights["own_subtype"] == 0.0
    assert contract["direction"]["status"] in ("ESTABLISHED", "NOT_ESTABLISHED")


def test_v3_supervision_needs_no_graph():
    import inspect
    for function in (st.orbit_loss, st.relation_ranking_loss, st.relation_distill_loss):
        names = set(inspect.signature(function).parameters)
        assert not any(("adj" in name or "graph" in name or "edge" in name)
                       for name in names), (function.__name__, names)
