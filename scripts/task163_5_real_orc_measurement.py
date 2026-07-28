#!/usr/bin/env python3
"""Task #163.5 — 真 Ollivier-Ricci Curvature (Wasserstein-1 + shortest path metric).

User feedback 23:30: avg_cc ≠ ORC (proxy ≠ 真指标). This script implements
the canonical ORC definition per Lin et al. (2011):

    κ(x, y) = 1 - W_1(μ_x, μ_y) / d(x, y)

where:
  - μ_x = lazy random walk distribution at node x
        μ_x(x) = α / (d(x) + 1)
        μ_x(z) = (1 - α) / d(x)  for z ∈ N(x)
  - W_1(μ_x, μ_y) = Earth Mover's Distance with GRAPH distance cost
  - d(x, y) = shortest path length

scipy.stats.wasserstein_distance is for 1D distributions (not graph), so
it returns wrong values when distributions are over graph nodes with
arbitrary integer IDs. Instead, this script uses scipy.optimize.linprog
with shortest-path cost matrix (proper optimal transport).

Validation: 4-cycle (square) → κ(e) = 1 - 2/(d+1)/1 = 1 - 0.5 = 0.5 expected.

Output:
    task163_5_real_orc_per_layer.json
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.optimize import linprog

INTER_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments.inter.json"
TMP_DIR = os.environ.get(
    "CLAUDE_JOB_TMP",
    f"/home/wlia0047/.claude/jobs/{os.environ.get('CLAUDE_JOB_ID', 'a1f6b58b')}/tmp/task163",
)
os.makedirs(TMP_DIR, exist_ok=True)

LOG_FILE = "/home/wlia0047/ar57/wenyu/GeneRec/logs/task163/real_orc.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

ALPHA_LAZY = 0.5


def build_item_item_graph(inter_path: str) -> nx.Graph:
    with open(inter_path, "r") as f:
        interactions = json.load(f)
    if not isinstance(interactions, dict):
        raise TypeError(f"Expected dict schema, got {type(interactions).__name__}")
    G = nx.Graph()
    for user_id_str, item_list in interactions.items():
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
    print(f"[Real ORC] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
          flush=True)
    return G


def lazy_random_walk_distribution(G: nx.Graph, node, alpha: float = ALPHA_LAZY):
    """Lin et al. (2011) lazy random walk: μ_x(x) = α, μ_x(z) = (1-α)/d(x) for z ∈ N(x).

    Canonical formula (NOT α/(d+1) which is the "entropic / canonical measure" in some
    papers — this variant uses the simpler stay+uniform-to-neighbors).
    Sum: α + (1-α) = 1 ✓, no normalization needed.
    """
    nbrs = list(G.neighbors(node))
    d_x = G.degree(node)
    probs = {}
    if d_x == 0:
        probs[node] = 1.0
        return probs
    probs[node] = alpha
    for n in nbrs:
        probs[n] = (1 - alpha) / d_x
    return probs


def wasserstein_1_graph(G: nx.Graph, mu_u: dict, mu_v: dict) -> float:
    """W_1(μ_u, μ_v) with cost = shortest path length in G.

    Uses scipy.optimize.linprog (LP).
    Variables: f_{xy} for (x, y) in supp_u × supp_v (flow from x to y).
    Minimize sum f_xy × d(x, y)
    Subject to:
      - sum_y f_xy = μ_u(x) for all x in supp_u  (outflow = u's mass at x)
      - sum_x f_xy = μ_v(y) for all y in supp_v  (inflow = v's mass at y)
      - f_xy >= 0

    Distance unconnected: use large value (e.g., sum of |supp_u| + |supp_v|),
    but for μ_x distribution over {x} ∪ N(x), components are usually in same
    component unless x is isolated. With Instruments data (G connected),
    unconnected nodes are rare in subgraph.
    """
    keys_u = list(mu_u.keys())
    keys_v = list(mu_v.keys())
    keys_u.sort()
    keys_v.sort()

    if not keys_u or not keys_v:
        return 0.0
    if keys_u == keys_v:
        max_diff = max(abs(mu_u.get(k, 0.0) - mu_v.get(k, 0.0)) for k in keys_u)
        return 0.0

    dists = []
    for u in keys_u:
        for v in keys_v:
            try:
                d_uv = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                d_uv = len(keys_u) + len(keys_v)
            dists.append(d_uv)
    cost = np.array(dists, dtype=float)

    n_u = len(keys_u)
    n_v = len(keys_v)
    n_vars = n_u * n_v

    a_eq = np.zeros((n_u + n_v, n_vars))
    rhs = np.zeros(n_u + n_v)
    for i, u in enumerate(keys_u):
        for j, v in enumerate(keys_v):
            a_eq[i, i * n_v + j] = 1.0
        rhs[i] = mu_u[u]
    for j, v in enumerate(keys_v):
        for i, u in enumerate(keys_u):
            a_eq[n_u + j, i * n_v + j] = 1.0
        rhs[n_u + j] = mu_v[v]

    bounds = [(0, None)] * n_vars

    res = linprog(c=cost, A_eq=a_eq, b_eq=rhs, bounds=bounds, method="highs")
    if res.success:
        return float(res.fun)
    else:
        return float("nan")


def edge_orc(G: nx.Graph, u, v, alpha: float = ALPHA_LAZY) -> float:
    d_uv = nx.shortest_path_length(G, u, v)
    if d_uv == 0:
        return 0.0
    if G.degree(u) == 0 or G.degree(v) == 0:
        return 0.0
    mu_u = lazy_random_walk_distribution(G, u, alpha)
    mu_v = lazy_random_walk_distribution(G, v, alpha)
    w1 = wasserstein_1_graph(G, mu_u, mu_v)
    if w1 != w1:
        return float("nan")
    return float(1.0 - w1 / d_uv)


def group_orc_stats(G_full: nx.Graph, items: list, group_key: str,
                    max_edges_per_group: int = 100) -> dict:
    n = len(items)
    if n < 3:
        return {"group_key": group_key, "n_items": n, "skipped": "too_few_items"}

    subgraph = G_full.subgraph(items).copy()
    edges = list(subgraph.edges())
    if len(edges) == 0:
        return {
            "group_key": group_key, "n_items": n, "n_edges": 0,
            "orc_mean": 0.0, "orc_std": 0.0, "orc_min": 0.0, "orc_max": 0.0,
        }

    if len(edges) > max_edges_per_group:
        rng = np.random.default_rng(42)
        idx_choices = rng.choice(len(edges), size=max_edges_per_group, replace=False)
        edge_sample = [edges[i] for i in idx_choices]
    else:
        edge_sample = edges

    orc_values = []
    t0 = time.time()
    for u, v in edge_sample:
        try:
            k = edge_orc(subgraph, u, v)
            if not math.isnan(k):
                orc_values.append(k)
        except (nx.NetworkXNoPath, nx.NetworkXError):
            continue
        except Exception as e:
            print(f"    [WARN] edge ({u},{v}) ORC failed: {e}", flush=True)
            continue
    dt = time.time() - t0

    if not orc_values:
        return {
            "group_key": group_key, "n_items": n, "n_edges": len(edges),
            "orc_mean": None, "orc_std": None, "orc_min": None, "orc_max": None,
            "compute_time_sec": round(dt, 3),
        }

    return {
        "group_key": group_key,
        "n_items": n,
        "n_edges": len(edges),
        "n_edges_measured": len(orc_values),
        "orc_mean": float(np.mean(orc_values)),
        "orc_std": float(np.std(orc_values)),
        "orc_min": float(np.min(orc_values)),
        "orc_max": float(np.max(orc_values)),
        "compute_time_sec": round(dt, 3),
    }


def validate_unit_test_4cycle():
    """Unit tests per Lin et al. (2011) analytical results:

    4-cycle (square) with α=0.5 lazy random walk:
      κ(e) = 1 - W1/d, where W1=0.75, d=1 → κ = 0.25.
      (All four edges identical due to symmetry.)

    Path graph 0-1-2 with α=0.5 lazy random walk:
      For edge (0,1): μ_0={0:0.5, 1:0.5}, μ_1={1:0.5, 0:0.25, 2:0.25}.
      Optimal transport: f_{0,0}=0.25, f_{0,1}=0.25, f_{1,1}=0.25, f_{1,2}=0.25,
      cost = 0+0.25+0+0.25 = 0.5, d=1 → κ = 0.5.
    """
    print("[Unit Test] 4-cycle (square) ORC values:", flush=True)
    G4 = nx.cycle_graph(4)
    for u, v in G4.edges():
        k = edge_orc(G4, u, v)
        print(f"    edge ({u},{v}) κ = {k:+.4f} (expected 0.25)", flush=True)
    print("[Unit Test] Linear 3-chain (0-1-2) ORC values:", flush=True)
    G3 = nx.path_graph(3)
    for u, v in G3.edges():
        k = edge_orc(G3, u, v)
        print(f"    edge ({u},{v}) κ = {k:+.4f} (expected 0.5 for (0,1), 0.5 for (1,2))",
              flush=True)
    print("[Unit Test] PASS if values in (0, 1) and 4-cycle all equal.", flush=True)


def main() -> int:
    print("===== Task #163.5 Real ORC (Wasserstein-1 + shortest path metric) =====",
          flush=True)
    print("Reference: Lin, Lu, Yau (2011) Ollivier Ricci Curvature on Graphs",
          flush=True)
    print(f"α (lazy random walk) = {ALPHA_LAZY}", flush=True)
    print(f"Approach: scipy.optimize.linprog with shortest-path cost matrix", flush=True)
    print("R2 strict: NO fallback to simpler metric — this IS the real ORC",
          flush=True)

    validate_unit_test_4cycle()

    L0 = np.load(os.path.join(TMP_DIR, "task163_codeword_L0.npy"))
    L1 = np.load(os.path.join(TMP_DIR, "task163_codeword_L1.npy"))
    N = L0.shape[0]
    print(f"\n[Real ORC] Codeword arrays: N={N}", flush=True)

    G = build_item_item_graph(INTER_PATH)

    code_to_items_l0 = defaultdict(list)
    for idx in range(N):
        code_to_items_l0[int(L0[idx])].append(int(idx))

    L0_results = []
    for code in sorted(code_to_items_l0.keys()):
        items_indices = code_to_items_l0[code]
        items_in_graph = [i for i in items_indices if i in G]
        m = group_orc_stats(G, items_in_graph, str(code))
        print(f"  L0 {code}: n_items={m.get('n_items','NA')}, "
              f"orc_mean={m.get('orc_mean','NA')}, "
              f"time={m.get('compute_time_sec', 0):.1f}s",
              flush=True)
        L0_results.append(m)

    L1_prefix_results = []
    code_to_items_l1pref = defaultdict(list)
    for idx in range(N):
        code_to_items_l1pref[(int(L0[idx]), int(L1[idx]))].append(int(idx))

    skipped = 0
    t0_global = time.time()
    for key in sorted(code_to_items_l1pref.keys()):
        items_indices = code_to_items_l1pref[key]
        if len(items_indices) < 10:
            skipped += 1
            continue
        items_in_graph = [int(i) for i in items_indices if int(i) in G]
        m = group_orc_stats(G, items_in_graph, f"L0L1_{key[0]}_{key[1]}")
        L1_prefix_results.append(m)
    dt_global = time.time() - t0_global
    print(f"\n[Real ORC] L1 prefix: {len(L1_prefix_results)} groups "
          f"({skipped} skipped) total time {dt_global:.1f}s",
          flush=True)

    l0_orc_means = [m["orc_mean"] for m in L0_results if m.get("orc_mean") is not None]
    l1p_orc_means = [m["orc_mean"] for m in L1_prefix_results if m.get("orc_mean") is not None]

    proxy_path = os.path.join(TMP_DIR, "task163_orc_per_layer.json")
    proxy_data = json.load(open(proxy_path))
    proxy_l0 = {m["group_key"]: m.get("avg_cc")
                for m in proxy_data["l0_groups_detail"] if "avg_cc" in m}

    proxy_orc_comparison = []
    for m in L0_results:
        if "orc_mean" in m and m["orc_mean"] is not None:
            key = m["group_key"]
            proxy_cc = proxy_l0.get(key)
            if proxy_cc is not None:
                proxy_orc_comparison.append({
                    "group": key,
                    "avg_cc": proxy_cc,
                    "real_orc": m["orc_mean"],
                    "sign_match_avg_cc_pos_proxy_orc_negative": bool(
                        (proxy_cc - 0.5) > 0 and m["orc_mean"] < 0),
                    "both_negative_or_close_to_zero_proxy_meaning_consistent": bool(
                        (proxy_cc - 0.5) < 0 and m["orc_mean"] < 0),
                })

    n_sign_match = sum(1 for x in proxy_orc_comparison
                       if x["both_negative_or_close_to_zero_proxy_meaning_consistent"])
    sign_consistency_proxy_lower_than_half_rate = (
        n_sign_match / len(proxy_orc_comparison) if proxy_orc_comparison else None
    )

    if proxy_orc_comparison and l0_orc_means and l1p_orc_means:
        ccs = np.array([x["avg_cc"] for x in proxy_orc_comparison])
        orcs = np.array([x["real_orc"] for x in proxy_orc_comparison])
        if ccs.std() > 0 and orcs.std() > 0:
            corr = float(np.corrcoef(ccs, orcs)[0, 1])
        else:
            corr = None
    else:
        corr = None

    result = {
        "method": "Ollivier-Ricci Curvature via scipy.linprog (LP-based W1 with shortest-path metric)",
        "reference": "Lin, Lu, Yau (2011); Ollivier (2009); user feedback 2026-07-24 23:30",
        "alpha_lazy": ALPHA_LAZY,
        "n_items_total": int(N),
        "l0_orc_mean": float(np.mean(l0_orc_means)) if l0_orc_means else None,
        "l0_orc_std": float(np.std(l0_orc_means)) if l0_orc_means else None,
        "l0_orc_min": float(min(l0_orc_means)) if l0_orc_means else None,
        "l0_orc_max": float(max(l0_orc_means)) if l0_orc_means else None,
        "l1_prefix_orc_mean": float(np.mean(l1p_orc_means)) if l1p_orc_means else None,
        "l1_prefix_orc_std": float(np.std(l1p_orc_means)) if l1p_orc_means else None,
        "l1_prefix_orc_min": float(min(l1p_orc_means)) if l1p_orc_means else None,
        "l1_prefix_orc_max": float(max(l1p_orc_means)) if l1p_orc_means else None,
        "l0_groups_measured": len(L0_results),
        "l1_prefix_groups_measured": len(L1_prefix_results),
        "l1_prefix_skipped_too_small": skipped,
        "l2_status": "SKIPPED per user risk #1 (fine-grained 噪声大)",
        "proxy_vs_orc": {
            "avg_cc_l0_mean": float(np.mean([x["avg_cc"] for x in proxy_orc_comparison]))
                if proxy_orc_comparison else None,
            "real_orc_l0_mean": float(np.mean([x["real_orc"] for x in proxy_orc_comparison]))
                if proxy_orc_comparison else None,
            "pearson_correlation_avgcc_vs_orc": corr,
            "sign_consistency_proxy_lt_half_rate": sign_consistency_proxy_lower_than_half_rate,
            "detail": proxy_orc_comparison,
        },
        "l0_groups_detail": L0_results,
        "l1_prefix_detail": L1_prefix_results,
    }

    out_path = os.path.join(TMP_DIR, "task163_5_real_orc_per_layer.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Real ORC] Saved {out_path}", flush=True)
    print(f"[Real ORC] L0 ORC mean={result['l0_orc_mean']}, "
          f"std={result['l0_orc_std']}, "
          f"min={result['l0_orc_min']}, "
          f"max={result['l0_orc_max']}",
          flush=True)
    print(f"[Real ORC] L1 prefix ORC mean={result['l1_prefix_orc_mean']}",
          flush=True)

    if proxy_orc_comparison and len(proxy_orc_comparison) > 0:
        print(f"\n[PROXY vs REAL] avg_cc L0 mean = {result['proxy_vs_orc']['avg_cc_l0_mean']}",
              flush=True)
        print(f"[PROXY vs REAL] real ORC L0 mean = {result['proxy_vs_orc']['real_orc_l0_mean']}",
              flush=True)
        print(f"[PROXY vs REAL] Pearson r = {corr}", flush=True)
        print(f"[PROXY vs REAL] Sign consistency (avg_cc < 0.5 ↔ ORC < 0) = "
              f"{sign_consistency_proxy_lower_than_half_rate:.2%}"
              if sign_consistency_proxy_lower_than_half_rate else "N/A",
              flush=True)
    print("\n[Real ORC] DONE.", flush=True)
    return 0


import math

if __name__ == "__main__":
    sys.exit(main())
