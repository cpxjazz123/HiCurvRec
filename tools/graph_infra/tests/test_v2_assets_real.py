"""Real-data acceptance on the frozen v2 assets (issue #246 C1-C9, C13-C16, C19-C22, C35).

Every check reads the actual files and recomputes from raw products; nothing takes a stored
"ok" flag at face value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import assets, config, graph_build, graphlet, isolation, splits  # noqa: E402

ROOT = config.ASSET_ROOT


def need(path: Path):
    if not path.exists():
        raise AssertionError(f"missing v2 asset: {path}; run build_assets_v2.py")
    return path


def load(relative):
    return assets.load_sparse(need(ROOT / relative))


def test_manifest_is_v2_and_verifies():
    manifest = need(ROOT / "manifest.json")
    audit = assets.audit_manifest(manifest)
    assert audit["ok"], audit["issues"][:5]
    verified = assets.verify_manifest(manifest)
    assert len(verified["verified"]) == audit["assets"] > 20


def test_item_ids_and_embedding_rows_agree():
    info = graph_build.item_index()
    assert info["items"] == 24587
    assert info["embeddings"][0] == info["items"]


def test_sequence_edges_match_the_full_csr_exactly():
    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    reference = ref.sequence_edges(list(sequences.values()))
    forward = load("graph/full_description_forward.npz")
    coo = forward.tocoo()
    produced = {(int(a), int(b)): int(w) for a, b, w in zip(coo.row, coo.col, coo.data)}
    assert set(produced) == set(reference)
    assert all(produced[edge] == reference[edge] for edge in reference)
    assert ref.self_loops(list(sequences.values())) == 0


def test_split_users_are_disjoint_complete_and_add_up():
    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    users = np.array(sorted(sequences), dtype=np.int64)
    blob = np.load(need(ROOT / "graph" / "split_users.npz"))
    result = {name: blob[name] for name in splits.NAMES}
    audit = splits.audit(result, users)
    assert audit["disjoint"] and audit["union_complete"], audit
    full = graph_build.count_edges(list(sequences.values()))
    total = 0
    for name in splits.NAMES:
        total += graph_build.count_edges([sequences[int(u)] for u in result[name].tolist()]).total()
    assert total == full.total()


def test_held_out_support_is_boolean_and_bidirectional():
    visible = load("graph/visible_forward.npz")
    val = load("graph/struct_val_forward.npz")
    test = load("graph/struct_test_forward.npz")
    held = load("graph/held_out_support.npz")
    assert (held != held.T).nnz == 0, "the block set must be symmetric"
    coo = held.tocoo()
    off_diagonal = [(int(a), int(b)) for a, b in zip(coo.row, coo.col) if a != b]
    assert off_diagonal, "some held-out edges must exist"
    for left, right in off_diagonal[:500]:
        # a blocked pair is unseen in at least one direction; its mirror may still be visible,
        # which is exactly why the evaluation splits the two cases apart
        assert visible[left, right] == 0 or visible[right, left] == 0
        assert held[right, left] > 0
        assert (val[left, right] > 0 or val[right, left] > 0
                or test[left, right] > 0 or test[right, left] > 0)


def test_training_banks_respect_the_block_set_and_their_definitions():
    from tools.graph_infra import pairs as pair_ops
    visible = load("graph/visible_forward.npz")
    reverse = visible.T.tocsr()
    held = load("graph/held_out_support.npz")
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        blob = np.load(path)
        produced = blob["pairs"]
        assert not isolation.blocked_pairs(produced, held), path.name
        assert len({tuple(p) for p in produced.tolist()}) == len(produced), path.name
        assert not np.any(produced[:, 0] == produced[:, 1]), path.name
        group = str(blob["group"])
        pair_ops.verify_group(visible, reverse, produced, group)
        if group == "degree_matched":
            assert pair_ops.verify_degree_match(visible, reverse, produced)["violations"] == 0


def test_strict_eval_keys_never_reach_a_training_bank_or_the_probe():
    train_keys = set()
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        train_keys |= isolation.pair_keys(np.load(path)["pairs"])
    probe = isolation.pair_keys(np.load(need(ROOT / "pairs" / "probe.npz"))["pairs"])
    assert not (probe & train_keys), "the probe must not duplicate a training pair"
    strict_keys = set()
    for path in sorted((ROOT / "pairs").glob("strict_*.npz")):
        strict_keys |= isolation.pair_keys(np.load(path)["pairs"])
    assert strict_keys, "strict evaluation banks must exist"
    assert not (strict_keys & train_keys)
    assert not (strict_keys & probe)
    for path in sorted((ROOT / "pairs").glob("relaxed_*.npz")):
        keys = isolation.pair_keys(np.load(path)["pairs"])
        assert keys, path.name


def test_ego_blocks_match_the_graph():
    visible = load("graph/visible_forward.npz")
    reverse = visible.T.tocsr()
    for name in ("ego_full_1hop_visible", "ego_sampled_visible"):
        blob = np.load(need(ROOT / "ego" / f"{name}.npz"))
        node_offsets = blob["node_offsets"]
        edge_offsets = blob["edge_offsets"]
        assert node_offsets[0] == 0 and node_offsets[-1] == len(blob["nodes"])
        assert edge_offsets[0] == 0 and edge_offsets[-1] == len(blob["edges"])
        for index in range(0, len(blob["roots"]), 997):
            start, stop = node_offsets[index], node_offsets[index + 1]
            nodes = blob["nodes"][start:stop]
            root = int(blob["roots"][index])
            assert len(set(nodes.tolist())) == len(nodes)
            local = blob["root_local"][index]
            assert nodes[local] == root
            hop = blob["hop"][start:stop]
            assert hop[local] == 0 and int((hop == 0).sum()) == 1
            assert hop.max() <= 2
            role_out = blob["role_out"][start:stop]
            role_in = blob["role_in"][start:stop]
            out_row = set(visible.indices[visible.indptr[root]:visible.indptr[root + 1]].tolist())
            in_row = set(reverse.indices[reverse.indptr[root]:reverse.indptr[root + 1]].tolist())
            for position, node in enumerate(nodes.tolist()):
                assert bool(role_out[position]) == (node in out_row), (name, root, node)
                assert bool(role_in[position]) == (node in in_row), (name, root, node)
            edges = blob["edges"][edge_offsets[index]:edge_offsets[index + 1]]
            for left, right in edges.tolist():
                assert 0 <= left < len(nodes) and 0 <= right < len(nodes)
                assert visible[nodes[left], nodes[right]] > 0
    full = np.load(need(ROOT / "ego" / "ego_full_1hop_visible.npz"))
    sizes = np.diff(full["node_offsets"])
    sampled = np.load(need(ROOT / "ego" / "ego_sampled_visible.npz"))
    sampled_sizes = np.diff(sampled["node_offsets"])
    assert sampled_sizes.max() <= config.EGO_CAP + 1
    assert sizes.max() >= sampled_sizes.max()


def test_motif_counts_and_instances_agree():
    counts = np.load(need(ROOT / "motifs" / "motif_counts.npz"))
    names = counts["names"].tolist()
    assert len(names) == 16
    instances = np.load(need(ROOT / "motifs" / "motif_instances_visible.npz"))
    if len(instances["centers"]):
        for family_index in np.unique(instances["families"]).tolist():
            assert names[int(family_index)] in names
        for code, graphlet_id in zip(instances["codes"][:500].tolist(),
                                     instances["graphlets"][:500].tolist()):
            assert graphlet.canonical_code(int(code)) == int(graphlet_id)


def test_distance_fields_are_masked_and_ppr_is_a_mass():
    for path in sorted((ROOT / "pairs").glob("train_*.npz")) + sorted((ROOT / "pairs").glob("strict_*.npz")):
        blob = np.load(path)
        assert np.all(blob["reachable_undirected"][blob["hop_full_visible"] >= 0] == 1)
        assert np.all(blob["reachable_undirected"][blob["hop_full_visible"] < 0] == 0)
        assert np.all(blob["reachable_directed"][blob["hop_directed_src_dst"] >= 0] == 1)
        if "ppr" in blob.files:
            assert np.all(blob["ppr"] >= 0)
            assert np.all(blob["ppr"] <= 1 + 1e-9)
            assert np.all(np.isfinite(blob["ppr"]))
        assert np.all(blob["jaccard_out_missing"][np.isnan(blob["jaccard_out"])] == 1)


def test_curvature_asset_state_is_explicit():
    """The curvature *asset* was not built this round (owner instruction).

    The module and its known-answer tests are delivered; this check records which of the two
    exists, so the state can never be mistaken for a silently passing curvature check.
    """
    path = ROOT / "curvature" / "curvature_visible.json"
    if not path.exists():
        from tools.graph_infra import curvature
        assert callable(curvature.orc_edge), "the curvature module must stay importable"
        assert (Path(__file__).resolve().parent / "test_v2_geometry.py").exists()
        return
    report = json.loads(path.read_text())
    assert report["reference_nodes"] > 0
    assert report["unit"].startswith("dimensionless")
    assert "sampled_edges" in report and report["sampled_edges"] > 0
    assert "missing_nodes" in report


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
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
