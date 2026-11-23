"""Asset-level checks on the frozen data (issue #245 §2/§5/§7/§25).

These run against the real artefacts, not synthetic graphs: coverage, completeness, index
consistency and the leak rules that decide whether a pair may be used as supervision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets as asset_io  # noqa: E402
from tools.graph_infra import graph_distance  # noqa: E402

ASSETS = REPO / "results" / "research" / "infra245" / "assets"
MANIFEST = ASSETS / "manifest.json"


def load():
    forward = asset_io.load_sparse(ASSETS / "graph" / "forward_full.npz")
    visible = asset_io.load_sparse(ASSETS / "graph" / "visible_forward.npz")
    evaluation = asset_io.load_sparse(ASSETS / "graph" / "eval_forward.npz")
    stats = json.loads((ASSETS / "graph" / "stats.json").read_text())
    return forward, visible, evaluation, stats


def test_manifest_verifies_every_asset():
    result = asset_io.verify_manifest(MANIFEST)
    assert result["verified"], result
    assert len(result["verified"]) == len(json.loads(MANIFEST.read_text())["assets"])


def test_graphs_have_no_self_loops_and_nest():
    forward, visible, evaluation, stats = load()
    for name, csr in (("forward", forward), ("visible", visible), ("eval", evaluation)):
        rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
        assert not np.any(rows == csr.indices), f"{name} has a self loop"
    assert visible.nnz <= forward.nnz and evaluation.nnz <= forward.nnz
    assert stats["full"]["directed_edges"] == forward.nnz
    assert stats["eval_edges"] == evaluation.nnz


def test_stats_counts_match_the_csr():
    forward, visible, _, stats = load()
    assert stats["full"]["nodes"] == forward.shape[0]
    assert stats["full"]["directed_edges"] == forward.nnz
    assert stats["visible"]["nodes"] == visible.shape[0]
    both = forward.multiply(forward.T)
    assert stats["full"]["reciprocal_pairs"] == int(both.nnz) // 2


def test_visible_and_eval_users_are_disjoint_by_count():
    _, _, _, stats = load()
    assert stats["visible_users"] + stats["eval_users"] == stats["users"]
    assert 0 < stats["eval_users"] < stats["users"]


def test_direct_bank_is_really_an_edge():
    _, visible, _, _ = load()
    for name in ("direct",):
        produced = np.load(ASSETS / "pairs" / f"bank_{name}.npy")
        assert len(produced), name
        for left, right in produced.tolist():
            assert visible[left, right] > 0, (name, left, right)


def test_two_hop_bank_is_not_a_direct_edge_and_has_a_common_neighbour():
    _, visible, _, _ = load()
    produced = np.load(ASSETS / "pairs" / "bank_two_hop.npy")
    undirected = graph_distance.undirected_projection(visible)
    for left, right in produced.tolist():
        assert visible[left, right] == 0 and visible[right, left] == 0
        neighbours = undirected[left].indices
        assert len(set(neighbours.tolist()) & set(undirected[right].indices.tolist())) > 0


def test_degree_matched_bank_has_no_edge_in_either_direction():
    _, visible, _, _ = load()
    produced = np.load(ASSETS / "pairs" / "bank_degree_matched.npy")
    for left, right in produced.tolist():
        assert visible[left, right] == 0 and visible[right, left] == 0


def test_no_bank_contains_self_or_duplicate_pairs():
    for path in sorted((ASSETS / "pairs").glob("bank_*.npy")):
        produced = np.load(path)
        assert not np.any(produced[:, 0] == produced[:, 1]), path.name
        assert len({tuple(p) for p in produced.tolist()}) == len(produced), path.name


def test_eval_banks_mark_eval_use_in_the_manifest():
    manifest = json.loads(MANIFEST.read_text())
    names = {entry["name"]: entry for entry in manifest["assets"]}
    for name in ("bank_direct_eval", "bank_two_hop_eval"):
        assert "never supervision" in names[name]["purpose"], names[name]["purpose"]


def test_ego_offsets_roots_and_coverage():
    manifest = json.loads(MANIFEST.read_text())
    coverage = {entry["name"]: entry["coverage"] for entry in manifest["assets"]
                if entry["name"].startswith("ego_")}
    for label in ("ego_one_hop", "ego_two_hop"):
        data = np.load(ASSETS / "ego" / f"{label}.npz")
        offsets = data["offsets"]
        assert offsets[0] == 0 and offsets[-1] == len(data["nodes"])
        assert len(offsets) == data["roots"].size + 1
        assert np.all(np.diff(offsets) >= 1)
        for index in (0, 1, len(offsets) - 2):
            start, end = offsets[index], offsets[index + 1]
            block = data["nodes"][start:end]
            assert len(set(block.tolist())) == len(block)
            local = data["root_local"][index]
            assert 0 <= local < len(block)
            assert block[local] == data["roots"][index]
            hops = data["hop"][start:end]
            assert hops[local] == 0, "the centre must carry hop 0"
            assert int((hops == 0).sum()) == 1, "exactly one centre per block"
            assert hops.min() >= 0 and hops.max() <= 2
        assert coverage[label] > 0.99
        assert 0.0 <= float(data["empty"].mean()) < 1.0


def test_motif_labels_match_their_report():
    report = json.loads((ASSETS / "motifs" / "motif_report.json").read_text())
    data = np.load(ASSETS / "motifs" / "motif_labels.npz")
    names = data["names"].tolist()
    assert names == ["chain", "fork", "merge", "triangle", "reciprocal"]
    labels = data["labels"]
    assert labels.shape == (report["n_items"], len(names))
    no_motif = int((~labels.any(axis=1)).sum())
    assert no_motif == report["no_motif"]
    for index, name in enumerate(names):
        assert int(labels[:, index].sum()) == report["per_type"][name]["items"]


def test_ppr_asset_offsets_and_mass():
    data = np.load(ASSETS / "distance" / "ppr_visible_sources.npz")
    offsets = data["offsets"]
    assert offsets[0] == 0 and offsets[-1] == len(data["nodes"])
    sources = data["sources"]
    for index in range(min(5, len(sources))):
        start, end = offsets[index], offsets[index + 1]
        block = data["nodes"][start:end]
        scores = data["scores"][start:end]
        assert len(set(block.tolist())) == len(block)
        assert scores.sum() <= 1.0 + 1e-9
        assert np.all(scores[:-1] >= scores[1:]), "scores must be sorted"


def test_bank_and_eval_graph_do_not_share_a_held_out_only_edge():
    """An edge that exists only in the eval graph must never appear inside a training bank."""
    _, visible, evaluation, _ = load()
    eval_only = evaluation - evaluation.multiply(visible)
    eval_only = ((eval_only + eval_only.T) > 0).astype(np.int8).tocsr()
    pairs_to_check = set()
    for path in sorted((ASSETS / "pairs").glob("bank_*.npy")):
        if path.stem.endswith("_eval"):
            continue
        for left, right in np.load(path).tolist():
            pairs_to_check.add((int(left), int(right)))
    leaked = [(left, right) for left, right in pairs_to_check if eval_only[left, right] > 0]
    assert not leaked, f"held-out-only edges inside a training bank: {leaked[:5]}"


def test_full_graph_matches_the_canonical_behavior_graph():
    """Independent cross-check: the frozen canonical graph must have the same counts."""
    metadata_path = REPO / "results" / "research" / "behavior_graph" / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text())
    forward = asset_io.load_sparse(ASSETS / "graph" / "forward_full.npz")
    undirected = asset_io.load_sparse(ASSETS / "graph" / "undirected_full.npz")
    assert forward.nnz == metadata["counts"]["directed_transitions_distinct"], (
        forward.nnz, metadata["counts"]["directed_transitions_distinct"])
    assert undirected.nnz // 2 == metadata["counts"]["edges"], (
        undirected.nnz // 2, metadata["counts"]["edges"])
    for source, digest in metadata["sources"].items():
        target = REPO / source
        if target.exists():
            assert asset_io.sha256(target) == digest, f"source drift: {source}"


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
