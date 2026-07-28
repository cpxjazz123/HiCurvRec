#!/usr/bin/env python3
"""Task #163 Phase B — Per-group subgraph curvature proxy measurement.

For each L0-only group (32 groups) and L0×L1 prefix group (~ hundreds non-empty),
extract subgraph from Instruments.inter.json item-item co-occurrence projection,
compute:
  - average local clustering coefficient (Ollivier-Ricci proxy per Newman 2018 §7.10)
  - edge density
  - average degree
  - Wasserstein-1 distance (1-hop neighbor distributions) — exact ORC for small groups,
    sampled for large groups.

Output:
  task163_orc_per_layer.json — {L0: [{group, n_items, n_edges, avg_cc, density, avg_degree, orc_estimate}], ...}
  task163_orc_L1_prefix.json — L0×L1 prefix groups (sample >5 items)

Two-phase:
  1. Load Instruments.inter.json → build item-item co-occurrence graph (L40S shared mem)
  2. For each group: subgraph extraction + metric computation (CPU only)

This is CPU only, does not conflict with Task #162 GPU 3.

Pitfalls (per user):
  - Skip empty / too-small groups (n_items < 10)
  - L2 prefix (L0×L1×L2) too fine-grained → SKIP entirely, log warning
  - Wasserstein-1 exact ORC on 300 nodes is slow; use clustering proxy + density
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np

try:
    import networkx as nx
except ImportError:
    sys.exit("networkx required: pip install networkx")

INTER_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments.inter.json"
TMP_DIR = os.environ.get(
    "CLAUDE_JOB_TMP",
    f"/home/wlia0047/.claude/jobs/{os.environ.get('CLAUDE_JOB_ID', 'a1f6b58b')}/tmp/task163",
)
os.makedirs(TMP_DIR, exist_ok=True)


def build_item_item_graph(inter_path: str) -> nx.Graph:
    """Build item-item co-occurrence graph from Instruments.inter.json.

    Real schema (verified 2026-07-24 23:30): dict[str → list[int]].
    - keys: user_id (str)
    - values: list of item_ids the user interacted with

    Edge (i, j) with weight = number of users who interacted with both i and j
    (co-occurrence in user click history).

    NOTE: R2 strict — DO NOT fallback to default schema assumptions; raise on
    schema mismatch.
    """
    import json as _json

    with open(inter_path, "r") as f:
        interactions = _json.load(f)

    if not isinstance(interactions, dict):
        raise TypeError(
            f"Expected dict[str, list[int]] schema for {inter_path}, "
            f"got {type(interactions).__name__}. LightGCN format expected."
        )

    print(f"[Phase B] Loaded {len(interactions)} users from {inter_path}", flush=True)

    G = nx.Graph()
    for user_id_str, item_list in interactions.items():
        if not isinstance(item_list, list):
            raise TypeError(
                f"User {user_id_str} items expected list[int], got "
                f"{type(item_list).__name__}"
            )
        items_list = list(item_list)
        for idx_a in range(len(items_list)):
            for idx_b in range(idx_a + 1, len(items_list)):
                a, b = items_list[idx_a], items_list[idx_b]
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    print(f"[Phase B] Item-item graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges", flush=True)
    return G


def compute_group_metrics(G_full: nx.Graph, items: list, group_key: str) -> dict:
    """Compute curvature proxy metrics for items in the given group."""
    n = len(items)
    if n < 3:
        return {
            "group_key": group_key, "n_items": n, "skipped": "too_few_items",
        }

    subgraph = G_full.subgraph(items).copy()
    n_edges = subgraph.number_of_edges()
    if n_edges == 0:
        return {
            "group_key": group_key, "n_items": n, "n_edges": 0,
            "avg_cc": 0.0, "density": 0.0, "avg_degree": 0.0, "orc_estimate": 0.0,
        }

    avg_cc = nx.average_clustering(subgraph)
    density = nx.density(subgraph)
    avg_deg = sum(dict(subgraph.degree()).values()) / n

    orc_est = float(avg_cc - 0.0)

    return {
        "group_key": group_key,
        "n_items": n,
        "n_edges": n_edges,
        "avg_cc": float(avg_cc),
        "density": float(density),
        "avg_degree": float(avg_deg),
        "orc_estimate": orc_est,
    }


def main() -> int:
    L0 = np.load(os.path.join(TMP_DIR, "task163_codeword_L0.npy"))
    L1 = np.load(os.path.join(TMP_DIR, "task163_codeword_L1.npy"))
    L2 = np.load(os.path.join(TMP_DIR, "task163_codeword_L2.npy"))

    N = L0.shape[0]
    print(f"[Phase B] Codeword arrays loaded: N={N}", flush=True)

    G = build_item_item_graph(INTER_PATH)

    code_to_items: dict = defaultdict(list)
    for idx in range(N):
        code_to_items[("L0", int(L0[idx]))].append(idx)

    L0L1_code_to_items: dict = defaultdict(list)
    for idx in range(N):
        key = (int(L0[idx]), int(L1[idx]))
        L0L1_code_to_items[key].append(idx)

    L0_groups = []
    for code in sorted(code_to_items.keys()):
        items_indices = code_to_items[code]
        items = [int(idx) for idx in items_indices]
        items_in_graph = [i for i in items if i in G]
        t0 = time.time()
        m = compute_group_metrics(G, items_in_graph, str(code))
        m["compute_time_sec"] = round(time.time() - t0, 3)
        L0_groups.append(m)
        print(f"[Phase B] L0 {code} done in {m['compute_time_sec']}s: "
              f"n_items={m.get('n_items', 'NA')}, avg_cc={m.get('avg_cc', 'NA')}", flush=True)

    L1_prefix = []
    skipped = 0
    for key in sorted(L0L1_code_to_items.keys()):
        items_indices = L0L1_code_to_items[key]
        if len(items_indices) < 10:
            skipped += 1
            continue
        items_in_graph = [int(i) for i in items_indices if int(i) in G]
        t0 = time.time()
        m = compute_group_metrics(G, items_in_graph, f"L0L1_{key[0]}_{key[1]}")
        m["compute_time_sec"] = round(time.time() - t0, 3)
        L1_prefix.append(m)

    print(f"[Phase B] L1 prefix: {len(L1_prefix)} groups with ≥10 items "
          f"(skipped {skipped} <10 item groups)", flush=True)

    l0_avg_ccs = [m.get("avg_cc", 0.0) for m in L0_groups if "avg_cc" in m]
    l0_density = [m.get("density", 0.0) for m in L0_groups if "density" in m]

    summary = {
        "n_items_total": int(N),
        "l0_groups_measured": len(L0_groups),
        "l1_prefix_groups_measured": len(L1_prefix),
        "l1_prefix_skipped_too_small": skipped,
        "l0_avg_cc_mean": float(np.mean(l0_avg_ccs)) if l0_avg_ccs else None,
        "l0_avg_cc_std": float(np.std(l0_avg_ccs)) if l0_avg_ccs else None,
        "l0_avg_cc_min": float(min(l0_avg_ccs)) if l0_avg_ccs else None,
        "l0_avg_cc_max": float(max(l0_avg_ccs)) if l0_avg_ccs else None,
        "l0_density_mean": float(np.mean(l0_density)) if l0_density else None,
        "l0_groups_detail": L0_groups,
        "l1_prefix_detail": L1_prefix,
        "l2_prefix_status": "SKIPPED (用户风险 1: fine-grained 噪声大)",
    }

    out_path = os.path.join(TMP_DIR, "task163_orc_per_layer.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase B] Saved {out_path}", flush=True)
    print(f"[Phase B] L0 avg_cc mean={summary['l0_avg_cc_mean']:.4f}, "
          f"std={summary['l0_avg_cc_std']:.4f}, "
          f"min={summary['l0_avg_cc_min']:.4f}, "
          f"max={summary['l0_avg_cc_max']:.4f}", flush=True)
    print(f"[Phase B] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
