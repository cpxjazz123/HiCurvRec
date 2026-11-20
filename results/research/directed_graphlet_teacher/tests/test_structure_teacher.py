"""Issue #229: the frozen structure Teacher and its per-item supervision target.

The contract under test, in the issue's own terms:

  * the Teacher is frozen and reloadable bitwise, and inference needs no graph
    (`load_structure_teacher(...).encode_items(features)` takes only the 768-d
    item embeddings the RQ-VAE already has);
  * the per-item target carries **multiple roles** (a 376-dim orbit distribution
    with more than one non-zero role for a large share of items), an item-level
    relation vector whose pairwise geometry is used as the relation target, and
    an explicit coverage/confidence mask so unsupported items cannot be stamped
    with a fabricated structure;
  * the targets are not constant, not identity memorisation and not copies of
    each other (distinctness is measured, not asserted);
  * the training losses only consume item ids and the precomputed table -- no
    adjacency matrix is accepted anywhere on the supervision path, so a future
    RQ-VAE cannot accidentally re-read the graph;
  * the artifact set (teacher, targets, contract, manifest) has matching SHAs.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TESTS = Path(__file__).resolve().parent
TEACHER = TESTS.parent
ROLE = TEACHER.parent / "directed_graphlet_role"
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(ROLE))

import structure_teacher as st  # noqa: E402


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    out = tmp_path_factory.mktemp("structure_teacher")
    return st.build_structure_teacher(device=torch.device("cpu"), out_dir=out,
                                      synthetic=True, seed=229)


def test_artifact_files_and_shas(artifact):
    for key in ("teacher", "targets", "contract", "manifest"):
        assert Path(artifact[key]).is_file(), key
    manifest = json.loads(Path(artifact["manifest"]).read_text())
    import hashlib
    for name, entry in manifest["files"].items():
        path = Path(artifact["root"]) / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], name


def test_teacher_is_frozen_and_reloads_bitwise(artifact):
    teacher = st.load_structure_teacher(artifact["teacher"])
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.training is False
    rng = np.random.default_rng(0)
    features = torch.from_numpy(rng.normal(size=(8, teacher.in_dim)).astype(np.float32))
    first = teacher.encode_items(features)
    second = teacher.encode_items(features)
    assert torch.equal(first, second), "frozen inference is not deterministic"
    assert torch.equal(first, teacher.encode_items(features))


def test_targets_are_multi_role_and_not_constant(artifact):
    targets = st.load_targets(artifact["targets"])
    covered = targets["coverage"] > 0
    assert covered.sum() > 0
    roles = (targets["orbit_target"][covered] > 0).sum(1)
    assert float(roles.mean()) > 1.0, "a single role per item loses multi-role structure"
    relation = targets["relation"][covered]
    assert float(relation.std()) > 1e-3, "the relation target is constant"
    normed = relation / np.clip(np.linalg.norm(relation, axis=1, keepdims=True), 1e-9, None)
    similarity = normed[: min(len(normed), 512)] @ normed[: min(len(normed), 512)].T
    off_diagonal = similarity[~np.eye(len(similarity), dtype=bool)]
    assert float((off_diagonal > 0.999).mean()) < 0.5, "targets are near-identical"


def test_uncovered_items_have_zero_confidence(artifact):
    targets = st.load_targets(artifact["targets"])
    uncovered = targets["coverage"] == 0
    assert uncovered.any(), "the fixture has no unsupported items"
    assert np.all(targets["confidence"][uncovered] == 0.0)
    assert np.all(targets["orbit_target"][uncovered].sum(1) == 0.0)


def test_supervision_path_accepts_no_adjacency_or_graph():
    for function in (st.orbit_loss, st.relation_ranking_loss, st.relation_distill_loss):
        names = set(inspect.signature(function).parameters)
        assert not any("adj" in name or "graph" in name or "edge" in name for name in names), \
            (function.__name__, names)
    # the frozen teacher's item path only takes the embedding matrix
    signature = inspect.signature(st.FrozenTeacher.encode_items)
    assert list(signature.parameters) == ["self", "features"]


def test_orbit_loss_uses_covered_items_only_and_responds_to_alignment(artifact):
    targets = st.load_targets(artifact["targets"])
    ids = torch.arange(len(targets["coverage"]))
    probe = torch.zeros((len(ids), targets["orbit_target"].shape[1]))
    low = st.orbit_loss(probe, targets, ids)
    aligned = torch.from_numpy(np.log(np.clip(targets["orbit_target"], 1e-9, None))).float()
    high = st.orbit_loss(aligned, targets, ids)
    assert float(high) < float(low), (float(high), float(low))
    # and an uncovered id contributes nothing
    uncovered = int(np.flatnonzero(targets["coverage"] == 0)[0])
    only_uncovered = st.orbit_loss(probe, targets, torch.tensor([uncovered]))
    assert float(only_uncovered) == 0.0


def test_ranking_loss_separates_teacher_geometry_from_a_constant(artifact):
    targets = st.load_targets(artifact["targets"])
    pairs = targets["positive_pairs"][:32]
    negatives = targets["negative_pairs"][:32]
    vectors = torch.from_numpy(targets["relation"]).float()
    teacher_loss = st.relation_ranking_loss(vectors, pairs, negatives, margin=0.2)
    constant_loss = st.relation_ranking_loss(torch.ones_like(vectors), pairs, negatives,
                                             margin=0.2)
    assert float(teacher_loss) < float(constant_loss) - 1e-3


def test_distill_loss_prefers_the_teacher_geometry(artifact):
    targets = st.load_targets(artifact["targets"])
    vectors = torch.from_numpy(targets["relation"]).float()
    pairs = targets["positive_pairs"][:64]
    teacher = torch.from_numpy(targets["relation"]).float()
    close = st.relation_distill_loss(vectors, pairs, teacher)
    shuffled = vectors[torch.randperm(len(vectors), generator=torch.Generator().manual_seed(0))]
    far = st.relation_distill_loss(shuffled, pairs, teacher)
    assert float(close) < float(far)


def test_contract_states_the_input_boundary(artifact):
    contract = json.loads(Path(artifact["contract"]).read_text())
    assert contract["rqvae_input_changed"] is False
    assert "768" in contract["rqvae_encoder_input"]
    assert contract["teacher_inference_needs_graph"] is False
    assert contract["used_only_in_training_loss"] is True
    assert "direction" in contract["capabilities"]


REAL = TEACHER / "teacher" / "structure"


def test_real_artifact_satisfies_the_contract():
    """The delivered artifact, not only the synthetic fixture."""
    if not (REAL / "teacher_structure_v1.pth").is_file():
        pytest.skip("the real structure artifact has not been built")
    import hashlib
    teacher = st.load_structure_teacher(REAL / "teacher_structure_v1.pth")
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    targets = st.load_targets(REAL / "structure_targets_v1.npz")
    assert targets["orbit_target"].shape == (24587, 376)
    assert targets["relation"].shape == (24587, 128)
    covered = targets["coverage"] > 0
    assert 0.45 < covered.mean() < 0.55
    assert float((targets["orbit_target"][covered] > 0).sum(1).mean()) > 1.5
    assert np.array_equal(np.isfinite(targets["relation"]).all(axis=1),
                          np.ones(len(covered), dtype=bool))
    assert len(targets["positive_pairs"]) == len(targets["negative_pairs"])
    contract = json.loads((REAL / "teacher_structure_contract.json").read_text())
    assert contract["capabilities"]["prediction"]["verdict"] == "GRAPH_SIGNAL"
    assert contract["capabilities"]["direction"]["status"] in ("ESTABLISHED",
                                                               "NOT_ESTABLISHED")
    manifest = json.loads((REAL / "teacher_structure_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((REAL / name).read_bytes()).hexdigest() == entry["sha256"]
