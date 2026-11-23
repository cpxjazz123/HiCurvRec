"""Build and freeze the issue #245 assets.

Everything here is model-independent: the directed graph, its statistics, the per-item induced
ego subgraphs, the typed pair banks, a distance cache and the manifest that binds them.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets, graph_distance, graphlet, pairs, topology  # noqa: E402

TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
FEATURES = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = REPO / "results" / "research" / "infra245" / "assets"
USER_SPLIT_SEED = 245001
EGO_SEED = 245002
EGO_CAP = 32
DISTANCE_ROOTS = 2000


def sequences() -> dict[int, list[int]]:
    frame = pd.read_parquet(TRAIN)
    longest: dict[int, list[int]] = {}
    for user, history in zip(frame["user"].to_numpy(), frame["history"]):
        key = int(user)
        if key not in longest or len(history) > len(longest[key]):
            longest[key] = list(history)
    by_user = {}
    for user, history, target in zip(frame["user"].to_numpy(), frame["history"], frame["target"]):
        key = int(user)
        if len(history) == len(longest[key]):
            by_user[key] = [int(x) for x in history] + [int(target)]
    return by_user


def build_graph(seqs, users) -> Counter:
    counts: Counter = Counter()
    for user in users:
        sequence = seqs[user]
        for left, right in zip(sequence[:-1], sequence[1:]):
            if left != right:
                counts[(left, right)] += 1
    return counts


def to_csr(counts: Counter, n: int):
    keys = np.array(sorted(counts), dtype=np.int64)
    weights = np.array([counts[tuple(k)] for k in keys], dtype=np.int64)
    csr = sp.coo_matrix((weights, (keys[:, 0], keys[:, 1])), shape=(n, n)).tocsr()
    csr.sort_indices()
    return csr


def ego_blocks(roots, out_csr, in_csr, rng, radius, cap):
    """Store every ego subgraph in offset form: nodes, induced edges, hop tags, root row."""
    node_rows, edge_rows, hop_rows, offsets, root_rows, empty = [], [], [], [0], [], []
    for root in roots:
        sample = topology.ego_induced(int(root), out_csr, in_csr, rng, radius=radius, cap=cap)
        node_rows.append(sample["nodes"])
        hop_rows.append(np.asarray(sample["role"], dtype=np.int8))
        root_rows.append(sample["nodes"].tolist().index(int(root)))
        if sample["edges"].size:
            edge_rows.append(sample["edges"] + offsets[-1])
        offsets.append(offsets[-1] + len(sample["nodes"]))
        empty.append(sample["empty_neighbourhood"])
    return {
        "nodes": (np.concatenate(node_rows) if node_rows else np.zeros(0, dtype=np.int64)),
        "hop": (np.concatenate(hop_rows) if hop_rows else np.zeros(0, dtype=np.int8)),
        "edges": (np.concatenate(edge_rows) if edge_rows else np.zeros((0, 2), dtype=np.int64)),
        "offsets": np.array(offsets, dtype=np.int64),
        "root_local": np.array(root_rows, dtype=np.int64),
        "empty": np.array(empty, dtype=bool),
        "roots": np.array(roots, dtype=np.int64),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = "python3 tools/graph_infra/build_assets.py"
    entries = []

    seqs = sequences()
    users = np.array(sorted(seqs), dtype=np.int64)
    rng = np.random.default_rng(USER_SPLIT_SEED)
    order = rng.permutation(len(users))
    cut = int(round(len(users) * 0.9))
    visible_users = users[np.sort(order[:cut])]
    eval_users = users[np.sort(order[cut:])]

    full_counts = build_graph(seqs, users)
    visible_counts = build_graph(seqs, visible_users)
    eval_counts = build_graph(seqs, eval_users)
    n_items = int(max(max(s) for s in seqs.values()) + 1)

    full_csr = to_csr(full_counts, n_items)
    visible_csr = to_csr(visible_counts, n_items)
    eval_csr = to_csr(eval_counts, n_items)

    # ---- graph assets ---------------------------------------------------
    graph_dir = OUT / "graph"
    assets.save_sparse(graph_dir / "forward_full.npz", full_csr)
    assets.save_sparse(graph_dir / "reverse_full.npz", full_csr.T.tocsr())
    full_undirected = graph_distance.undirected_projection(full_csr)
    assets.save_sparse(graph_dir / "undirected_full.npz", full_undirected)
    assets.save_sparse(graph_dir / "visible_forward.npz", visible_csr)
    assets.save_sparse(graph_dir / "visible_undirected.npz", graph_distance.undirected_projection(visible_csr))
    assets.save_sparse(graph_dir / "eval_forward.npz", eval_csr)

    full_stats = assets.graph_stats(full_csr)
    visible_stats = assets.graph_stats(visible_csr)
    stats = {
        "items": n_items,
        "users": int(len(users)),
        "visible_users": int(len(visible_users)),
        "eval_users": int(len(eval_users)),
        "self_loops_dropped": 0,
        "full": full_stats,
        "visible": visible_stats,
        "eval_edges": int(eval_csr.nnz),
        "note": "edge weight is the transition count; AF3 is NOT a graph distance and is not stored here",
    }
    (graph_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for name, path, purpose in (
        ("graph_forward_full", graph_dir / "forward_full.npz", "directed transitions, weight = count"),
        ("graph_reverse_full", graph_dir / "reverse_full.npz", "in-edges of the same graph"),
        ("graph_undirected_full", graph_dir / "undirected_full.npz", "simple undirected projection"),
        ("graph_visible_forward", graph_dir / "visible_forward.npz", "visible graph (90% users)"),
        ("graph_visible_undirected", graph_dir / "visible_undirected.npz", "undirected visible projection"),
        ("graph_eval_forward", graph_dir / "eval_forward.npz", "held-out users' transitions"),
        ("graph_stats", graph_dir / "stats.json", "counts and degree statistics"),
    ):
        entries.append(assets.entry(name, path, schema="npz.csr" if path.suffix == ".npz" else "json",
                                    purpose=purpose, command=command,
                                    dependencies=["results/stage0_build_parquet/train.parquet"]))

    # ---- ego assets -----------------------------------------------------
    ego_dir = OUT / "ego"
    ego_rng = np.random.default_rng(EGO_SEED)
    for label, radius in (("one_hop", 1), ("two_hop", 2)):
        blocks = ego_blocks(np.arange(n_items), full_csr, full_csr.T.tocsr(), ego_rng, radius, EGO_CAP)
        path = ego_dir / f"ego_{label}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **blocks)
        coverage = float(len(np.unique(blocks["nodes"])) / n_items)
        entries.append(assets.entry(f"ego_{label}", path, schema="ego.offset.v1",
                                    purpose=f"deterministic {radius}-hop induced ego subgraphs, cap {EGO_CAP}",
                                    command=command,
                                    dependencies=["graph_forward_full"],
                                    extra={"coverage": coverage, "cap": EGO_CAP,
                                           "empty_fraction": float(blocks["empty"].mean())}))

    # ---- pair banks -----------------------------------------------------
    pair_dir = OUT / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    bank_rng = np.random.default_rng(EGO_SEED + 1)
    # a pair may exist in the visible graph and still be held out through its mirror image;
    # both directions of every held-out edge are blocked from the training banks
    eval_only = (eval_csr - eval_csr.multiply(visible_csr)).tocsr()
    eval_only = ((eval_only + eval_only.T) > 0).astype(np.int8).tocsr()
    over = pairs.PER_GROUP * 2
    groups = {
        "direct": pairs.direct_pairs(visible_csr, bank_rng, over),
        "shared_out": pairs.shared_successor_pairs(visible_csr, n_items, bank_rng, over),
        "shared_in": pairs.shared_predecessor_pairs(visible_csr.T.tocsr(), n_items, bank_rng, over),
        "two_hop": pairs.two_hop_pairs(visible_csr, n_items, bank_rng, over),
        "degree_matched": pairs.degree_matched_pairs(visible_csr, visible_csr.T.tocsr(), n_items,
                                                     bank_rng, over),
    }
    report = {"_held_out_blocking": int(eval_only.nnz)}
    for name, produced in groups.items():
        before = len(produced)
        kept = pairs.exclude_undirected(produced, eval_only)[: pairs.PER_GROUP]
        report[name] = pairs.group_stats(kept)
        report[name]["dropped_by_held_out_blocking"] = int(before - len(pairs.exclude_undirected(produced, eval_only)))
        if len(kept):
            pairs.verify_group(visible_csr, visible_csr.T.tocsr(), kept, name)
        path = pair_dir / f"bank_{name}.npy"
        np.save(path, kept)
        entries.append(assets.entry(f"bank_{name}", path, schema="pairs.int64.Nx2",
                                    purpose=f"{name} candidate pairs on the visible graph, "
                                            "mirror-held-out pairs removed",
                                    command=command, dependencies=["graph_visible_forward"],
                                    extra={"pairs": int(len(kept)),
                                           "dropped_by_held_out_blocking": report[name]["dropped_by_held_out_blocking"]}))
    (pair_dir / "types.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entries.append(assets.entry("pair_types", pair_dir / "types.json", schema="json",
                                purpose="per-group pair counts, duplicates and self pairs",
                                command=command, dependencies=["bank_*"]))

    # ---- distance cache -------------------------------------------------
    dist_dir = OUT / "distance"
    dist_dir.mkdir(parents=True, exist_ok=True)
    roots = np.sort(rng.choice(np.arange(n_items), size=min(DISTANCE_ROOTS, n_items), replace=False))
    visible_undirected = graph_distance.undirected_projection(visible_csr)
    cache = graph_distance.bfs_distances(visible_undirected, roots)
    np.savez_compressed(dist_dir / "bfs_visible_root_cache.npz", roots=roots, distance=cache)
    entries.append(assets.entry("distance_visible_root_cache", dist_dir / "bfs_visible_root_cache.npz",
                                schema="npz.bfs.v1",
                                purpose="unweighted visible-graph distances from fixed reference roots",
                                command=command, dependencies=["graph_visible_undirected"],
                                extra={"roots": int(len(roots)), "unreachable_fraction":
                                       float(np.isinf(cache).mean())}))

    # ---- curvature (reused, with its caliber) ---------------------------
    curvature_csv = REPO / "results" / "research" / "issue236" / "node_curvature_exact_geodesic.csv"
    if curvature_csv.exists():
        entries.append(assets.entry(
            "curvature_proxy_236", curvature_csv, schema="csv.node_curvature.v1",
            purpose="per-item curvature proxy from #236",
            command="produced by issue #236; reused here without recomputation",
            dependencies=["results/research/behavior_graph/graph.npz"],
            extra={"caliber": "FULL train undirected graph (315,391 edges), not the visible-only "
                              "split; undirected only, do NOT read as a directed curvature",
                   "rows": int(sum(1 for _ in curvature_csv.open())) - 1}))

    # ---- typed motifs ---------------------------------------------------
    motif_dir = OUT / "motifs"
    motif_dir.mkdir(parents=True, exist_ok=True)
    labels = graphlet.label_nodes(full_csr, full_csr.T.tocsr(), full_undirected, range(n_items))
    matrix = np.zeros((n_items, len(graphlet.MOTIFS)), dtype=bool)
    for index, name in enumerate(graphlet.MOTIFS):
        matrix[:, index] = [labels["rows"][item][name] for item in range(n_items)]
    motif_path = motif_dir / "motif_labels.npz"
    np.savez_compressed(motif_path, names=np.array(graphlet.MOTIFS), labels=matrix)
    report = {
        "per_type": labels["per_type"],
        "overlap": labels["overlap"],
        "no_motif": labels["no_motif"],
        "n_items": labels["n_items"],
        "degree_stratified": graphlet.degree_stratified(labels),
        "note": "types are predicates on the full train undirected graph; a type is never a second "
                "drawing of the same ego",
    }
    (motif_dir / "motif_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entries.append(assets.entry("motif_labels", motif_path, schema="npz.motif.bool",
                                purpose="per-item chain/fork/merge/triangle/reciprocal labels",
                                command=command, dependencies=["graph_forward_full"],
                                extra={"types": list(graphlet.MOTIFS)}))
    entries.append(assets.entry("motif_report", motif_dir / "motif_report.json", schema="json",
                                purpose="coverage, overlap and degree-matched control for the motifs",
                                command=command, dependencies=["motif_labels"]))

    # ---- probe + eval-side banks ---------------------------------------
    probe = pairs.probe_bank(visible_csr, points=64)
    probe_path = pair_dir / "probe.npz"
    np.savez_compressed(probe_path, pairs=probe["pairs"], graph_hops=probe["graph_hops"],
                        seed=np.array(probe["seed"]))
    entries.append(assets.entry("gradient_probe", probe_path, schema="npz.probe.v1",
                                purpose="frozen gradient-calibration probe; no model values are stored",
                                command=command, dependencies=["graph_visible_forward"]))
    eval_banks = pairs.split_banks(eval_csr, eval_csr.T.tocsr(), n_items,
                                   {"eval": eval_csr}, np.random.default_rng(EGO_SEED + 2))
    for name, produced in eval_banks["eval"].items():
        path = pair_dir / f"bank_{name}_eval.npy"
        np.save(path, produced)
        entries.append(assets.entry(f"bank_{name}_eval", path, schema="pairs.int64.Nx2",
                                    purpose=f"{name} pairs on the EVAL graph only; evaluation use, "
                                            "never supervision",
                                    command=command, dependencies=["graph_eval_forward"],
                                    extra={"pairs": int(len(produced))}))

    # ---- PPR top-k on fixed sources -------------------------------------
    dist_dir_ppr = OUT / "distance"
    sources = np.sort(rng.choice(np.flatnonzero(np.diff(visible_csr.indptr) > 0), size=256, replace=False))
    rows, cols, vals, offsets = [], [], [], [0]
    for source in sources.tolist():
        result = graph_distance.ppr(visible_csr, source, restart=0.15, tolerance=1e-8, top_k=32)
        rows.append(result["nodes"]); vals.append(result["scores"])
        offsets.append(offsets[-1] + len(result["nodes"]))
    ppr_path = dist_dir_ppr / "ppr_visible_sources.npz"
    np.savez_compressed(ppr_path, sources=sources,
                        nodes=np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
                        scores=np.concatenate(vals) if vals else np.zeros(0),
                        offsets=np.array(offsets, dtype=np.int64))
    entries.append(assets.entry("ppr_visible_sources", ppr_path, schema="npz.ppr.v1",
                                purpose="PPR top-32 from 256 fixed visible-graph sources, restart 0.15, "
                                        "tolerance 1e-8; a propagation score, not a distance",
                                command=command, dependencies=["graph_visible_forward"],
                                extra={"sources": int(len(sources)), "top_k": 32}))

    # ---- manifest -------------------------------------------------------
    manifest = OUT / "manifest.json"
    assets.write_manifest(manifest, entries, issue=245,
                          notes="All assets are model-independent. Rebuild with "
                                "`python3 tools/graph_infra/build_assets.py`; verify with "
                                "`python3 tools/graph_infra/verify_assets.py`.")
    verification = assets.verify_manifest(manifest)
    print(json.dumps({"assets": len(entries), "verified": len(verification["verified"]),
                      "seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
