"""Step 1: build the real item purchase-transition graph and report its statistics.

Reads train.parquet only. Writes results/behavior_graph_geometry/:
  behavior_graph_statistics.json   graph statistics (the power-law block is added by step 2)
  graph_arrays.npz                 the raw distributions the later steps analyse
  lcc_edges.npz                    the undirected largest connected component

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/behavior_graph_geometry/step1_build_graph.py
"""
from __future__ import annotations

import json
import platform
import time
import subprocess
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import graph_lib as L  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frame = pd.read_parquet(C.TRAIN_PARQUET)
    print(f"[t={time.time()-t0:.1f}s] loaded parquet {frame.shape}", flush=True)
    sequences, audit = L.load_user_sequences(frame)
    directed, self_loops = L.transitions_from_sequences(sequences)

    print(f"[t={time.time()-t0:.1f}s] sequences {len(sequences)} users, "
          f"{len(directed)} directed transitions, {sum(self_loops.values())} self loops", flush=True)
    dg = L.directed_graph(directed)
    ug = L.undirected_graph(directed)

    print(f"[t={time.time()-t0:.1f}s] graphs built", flush=True)
    # interaction volume per item: how often it appears anywhere in a sequence
    appear: Counter = Counter()
    for _u, seq in sequences.items():
        appear.update(seq)
    interactions = np.array([appear.get(i, 0) for i in range(C.N_ITEMS)], dtype=np.int64)

    out_deg = np.zeros(C.N_ITEMS, dtype=np.int64)
    in_deg = np.zeros(C.N_ITEMS, dtype=np.int64)
    for (a, b), w in directed.items():
        out_deg[a] += w
        in_deg[b] += w
    out_neigh = np.zeros(C.N_ITEMS, dtype=np.int64)
    in_neigh = np.zeros(C.N_ITEMS, dtype=np.int64)
    for a, b in directed:
        out_neigh[a] += 1
        in_neigh[b] += 1

    weights = np.array([w for w in directed.values()], dtype=np.int64)
    strength = np.zeros(C.N_ITEMS, dtype=np.int64)
    for (a, b), w in directed.items():
        strength[a] += w
        strength[b] += w

    # direction asymmetry: of the unordered pairs with an edge, how many are seen in
    # both directions, and how far apart the two counts are
    pairs_both, pairs_one = 0, 0
    asym_ratio = []
    for a, b, d in ug.edges(data=True):
        f, r = d["fwd"], d["rev"]
        if f > 0 and r > 0:
            pairs_both += 1
            asym_ratio.append(abs(f - r) / (f + r))
        else:
            pairs_one += 1
    asym_ratio = np.asarray(asym_ratio, dtype=np.float64) if asym_ratio else np.zeros(0)

    print(f"[t={time.time()-t0:.1f}s] per-item distributions ready", flush=True)
    stats = {
        "study": "behavior_graph_geometry",
        "step": 1,
        "seed": C.SEED,
        "provenance": {
            "git_sha": git_sha(),
            "train_parquet": str(C.TRAIN_PARQUET),
            "train_parquet_bytes": int(C.TRAIN_PARQUET.stat().st_size),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "pandas": pd.__version__,
            "test_or_valid_used": False,
        },
        "data_audit": audit,
        "transitions": {
            "directed_distinct_pairs": int(len(directed)),
            "directed_total_count": int(sum(directed.values())),
            "self_loops_distinct_items": int(len(self_loops)),
            "self_loops_total_count": int(sum(self_loops.values())),
            "weight_min": int(weights.min()) if weights.size else 0,
            "weight_max": int(weights.max()) if weights.size else 0,
            "weight_mean": float(weights.mean()) if weights.size else 0.0,
            "weight_median": float(np.median(weights)) if weights.size else 0.0,
            "share_of_weight_one": float((weights == 1).mean()) if weights.size else 0.0,
        },
        "directed_graph": {
            "nodes": int(dg.number_of_nodes()),
            "edges": int(dg.number_of_edges()),
            "weakly_connected_components": int(nx.number_weakly_connected_components(dg)),
            "isolated_nodes": int(C.N_ITEMS - dg.number_of_nodes()),
        },
        "undirected_graph": L.graph_stats(ug),
        "direction_asymmetry": {
            "unordered_pairs_with_any_direction": int(ug.number_of_edges()),
            "pairs_seen_in_both_directions": int(pairs_both),
            "pairs_seen_in_one_direction_only": int(pairs_one),
            "share_seen_in_both_directions": float(pairs_both / max(ug.number_of_edges(), 1)),
            "mean_relative_count_gap_on_both_direction_pairs": (
                float(asym_ratio.mean()) if asym_ratio.size else 0.0),
            "median_relative_count_gap_on_both_direction_pairs": (
                float(np.median(asym_ratio)) if asym_ratio.size else 0.0),
        },
        "item_universe": {
            "n_items_declared": int(C.N_ITEMS),
            "items_appearing_in_train": int((interactions > 0).sum()),
            "items_with_an_outgoing_transition": int((out_neigh > 0).sum()),
            "items_with_an_incoming_transition": int((in_neigh > 0).sum()),
            "items_with_an_undirected_edge": int(ug.number_of_nodes()),
            "items_with_no_transition_at_all": int((interactions > 0).sum() - ug.number_of_nodes()),
        },
    }

    print(f"[t={time.time()-t0:.1f}s] graph stats done", flush=True)
    lcc_nodes = max(nx.connected_components(ug), key=len)
    lcc = ug.subgraph(lcc_nodes).copy()
    stats["undirected_graph"]["lcc"] = L.graph_stats(lcc)
    print(f"[t={time.time()-t0:.1f}s] LCC stats done", flush=True)

    (C.OUT_DIR / "behavior_graph_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # raw distributions for the later steps
    np.savez_compressed(
        C.OUT_DIR / "graph_arrays.npz",
        interactions=interactions,
        out_strength=out_deg,
        in_strength=in_deg,
        out_neighbours=out_neigh,
        in_neighbours=in_neigh,
        strength=strength,
        weights=weights,
        self_loop_counts=np.array([self_loops.get(i, 0) for i in range(C.N_ITEMS)],
                                  dtype=np.int64),
    )
    nodes = np.array(sorted(lcc_nodes), dtype=np.int64)
    index = {int(v): i for i, v in enumerate(nodes)}
    ei = np.array([[index[int(a)], index[int(b)]] for a, b in lcc.edges()], dtype=np.int64)
    np.savez_compressed(C.OUT_DIR / "lcc_edges.npz", nodes=nodes, edges=ei)

    print(json.dumps(stats, indent=2))
    print(f"\nwritten: {C.OUT_DIR/'behavior_graph_statistics.json'}")
    print(f"written: {C.OUT_DIR/'graph_arrays.npz'}")
    print(f"written: {C.OUT_DIR/'lcc_edges.npz'}  (LCC {len(nodes)} nodes, {len(ei)} edges)")
    print(f"[t={time.time()-t0:.1f}s] total", flush=True)


if __name__ == "__main__":
    main()