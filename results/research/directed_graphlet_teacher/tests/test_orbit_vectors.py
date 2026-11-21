"""Phase 1 orbit vectors: vocabulary, counts, split audit, reliability, learnability.

The tests re-derive the counts from the audited instance pool instead of trusting
the artifact, so a silent change in either the pool, the vocabulary or the
aggregation fails here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TEACHER = Path(__file__).resolve().parent.parent
VECTORS = TEACHER / "orbit_vectors"
sys.path.insert(0, str(VECTORS))
sys.path.insert(0, str(TEACHER.parent / "directed_graphlet_role"))


def load(name):
    path = VECTORS / name
    if not path.is_file():
        pytest.skip(f"{name} has not been built")
    return json.loads(path.read_text())


def test_vocabulary_is_fixed_and_every_dimension_is_named():
    vocab = load("orbit_vocab.json")
    assert vocab["k"] == 376
    dims = vocab["dimensions"]
    assert len(dims) == 376
    assert len({entry["index"] for entry in dims}) == 376
    assert len({entry["name"] for entry in dims}) == 376
    assert {entry["family"] for entry in dims} == {"K3", "P3", "C4", "P4", "K1_3"}
    for entry in dims:
        assert 0 <= entry["orbit"] <= 7
        assert entry["name"] == f"{entry['family']}:{entry['pattern_key']}:orbit{entry['orbit']}"


def test_counts_log_and_distribution_agree_and_reproduce():
    import build_orbit_vectors as bov
    payload = np.load(VECTORS / "orbit_vectors.npz")
    counts = payload["counts"].astype(np.int64)
    instances = payload["instances"].astype(np.int64)
    assert counts.shape == (24587, 376)
    assert (counts >= 0).all()
    assert np.array_equal(counts.sum(1), instances)
    assert np.allclose(payload["log1p"], np.log1p(counts), atol=1e-5)
    distribution = payload["distribution"]
    covered = instances > 0
    assert np.allclose(distribution[covered].sum(1), 1.0, atol=1e-5)
    assert np.allclose(distribution[~covered], 0.0)
    assert payload["reliable"].sum() == int((instances >= bov.MIN_RELIABLE).sum())

    raw = json.loads((bov.ROLE / "subtypes.json").read_text())
    dims, table = bov.build_vocabulary(raw)
    recomputed, _ = bov.count_roles(0, table, len(counts), len(dims))
    assert np.array_equal(recomputed, counts)


def test_no_heldout_edge_or_instance_enters_the_counts():
    audit = load("split_audit.json")
    assert audit["rebuild"]["set_equal"] is True
    assert audit["rebuild"]["weight_equal"] is True
    assert audit["overlap_with_heldout"]["graph_edges_not_in_train"] == 0
    assert audit["instance_pool"]["heldout_instances_used_in_counts"] == 0
    assert audit["stored_roles_spotcheck"]["checked"] > 0
    assert audit["stored_roles_spotcheck"]["mismatched"] == 0
    assert audit["graph"]["is_symmetric"] is False
    assert "TRAIN" in audit["graph"]["policy"]


def test_sampling_stability_is_reported_at_both_granularities():
    reliability = load("reliability.json")
    assert reliability["items_compared"] > 10000
    assert 0.0 <= reliability["cosine"]["mean"] <= 1.0
    assert reliability["family_level_cosine"]["median"] > 0.9
    assert "independent" in reliability["sampling_scheme"]


def test_learnability_flag_follows_its_own_rule():
    learn = load("learnability.json")
    payload = np.load(VECTORS / "orbit_vectors.npz")
    counts = payload["counts"].astype(np.int64)
    used = int((counts.sum(0) > 0).sum())
    assert 0 < learn["active_dimensions"] <= used
    fraction = learn["r2"]["fraction_above_null_p95"]
    expected = fraction >= 0.5 and learn["r2"]["mean"] >= 0.05
    assert learn["learnable"] is expected
    assert learn["trace_only"] is (fraction >= 0.5 and learn["r2"]["mean"] < 0.05)
