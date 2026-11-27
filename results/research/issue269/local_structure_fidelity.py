"""Does the SID separate closed triangles from open wedges, and does AF3 matter?

Two measurements on the canonical behaviour graph
(`results/research/behavior_graph/graph.npz`), both degree-matched:

1. **Closure test.**  A wedge is a centre with two neighbours (A-C-B).  It is a
   *triangle* when the two leaves are also adjacent and an *open wedge* when they
   are not.  Both have three items and two behaviour edges, so the only thing that
   differs is closure.  If the SID only learned "items with behaviour links share
   codes", triangles and open wedges look alike; if it learned local topology, the
   leaves of a triangle share a code more often than the leaves of an open wedge.
   Centres and leaves are matched on degree bucket.

2. **AF3 strata.**  ``kappa_ij = 4 - d_i - d_j + 3 T_ij`` is dominated by degree,
   so edges are stratified by the larger endpoint's degree bucket and by their own
   triangle count (0 / 1-2 / >=3).  Within a cell the degree is held roughly fixed
   and the triangle count varies, which is the closing part of kappa.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        analysis_sid_recall/local_structure_fidelity.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
OUT = REPO / "results/research/issue269/analysis/LOCAL_STRUCTURE.json"
ARMS = {
    "TIGER_calibrated_kappa_rq": "results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/item_sids.json",
    "TIGER_RQ-VAE": "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
    "TIGER_sinkhorn_all_levels_rq": "results/stage2_RQ-VAE/TIGER_sinkhorn_all_levels_rq/item_sids.json",
    "resample (recall .0634)": "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
    "mixed_rq_F0 (recall .0573)": "results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json",
}
SEED = 42
SAMPLES = 20_000


def bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


def load_graph():
    with np.load(GRAPH, allow_pickle=False) as payload:
        rows = np.asarray(payload["rows"], dtype=np.int64)
        cols = np.asarray(payload["cols"], dtype=np.int64)
        af3 = np.asarray(payload["af3"], dtype=np.float64)
        triangles = np.asarray(payload["triangles"], dtype=np.int64)
    n_nodes = int(max(rows.max(), cols.max()) + 1)
    adjacency = [set() for _ in range(n_nodes)]
    for left, right in zip(rows.tolist(), cols.tolist()):
        adjacency[left].add(right)
        adjacency[right].add(left)
    degree = np.array([len(values) for values in adjacency], dtype=np.int64)
    return rows, cols, af3, triangles, adjacency, degree, n_nodes


def sample_wedges(adjacency, degree, triangle_edges, rng):
    """Triangles and degree-matched open wedges, both as (centre, leaf, leaf)."""
    items = [node for node in range(len(adjacency)) if degree[node] > 0]
    by_degree = defaultdict(list)
    for node in items:
        by_degree[bucket(int(degree[node]))].append(node)
    closed, open_ = [], []
    order = list(range(len(triangle_edges)))
    rng.shuffle(order)
    for index in order:
        if len(closed) >= SAMPLES:
            break
        left, right, count = triangle_edges[index]
        common = adjacency[left] & adjacency[right]
        if not common:
            continue
        third = rng.choice(sorted(common))
        members = [left, right, third]
        centre = max(members, key=lambda node: degree[node])
        leaves = [node for node in members if node != centre]
        closed.append((int(centre), int(leaves[0]), int(leaves[1])))
        # Open control: same degree bucket for the centre, two of its own
        # neighbours that are not adjacent to each other and whose degrees fall
        # in the same buckets as the triangle's leaves.
        for _ in range(60):
            pool = by_degree.get(bucket(int(degree[centre])))
            if not pool:
                break
            control_centre = rng.choice(pool)
            neighbours = adjacency[control_centre]
            if len(neighbours) < 2:
                continue
            first, second = rng.sample(sorted(neighbours), 2)
            if second in adjacency[first]:
                continue
            wanted = [bucket(int(degree[node])) for node in members if node != centre]
            if [bucket(int(degree[first])), bucket(int(degree[second]))] != wanted:
                continue
            open_.append((control_centre, first, second))
            break
    return closed, open_


def share_rate(sid, wedges, depth_all, depth_leaves):
    if not wedges:
        return None
    all_share = 0
    leaf_share = 0
    for centre, left, right in wedges:
        values = [sid.get(centre), sid.get(left), sid.get(right)]
        if any(value is None for value in values):
            continue
        if values[0][:depth_all] == values[1][:depth_all] == values[2][:depth_all]:
            all_share += 1
        if values[1][:depth_leaves] == values[2][:depth_leaves]:
            leaf_share += 1
    total = len(wedges)
    return {"all_three": all_share / total, "leaves": leaf_share / total}


def main() -> None:
    rng = random.Random(SEED)
    rows, cols, af3, triangles, adjacency, degree, n_nodes = load_graph()
    print(f"graph {n_nodes} nodes, {len(rows)} edges, {int(triangles.sum()) // 3} triangles", flush=True)
    triangle_edges = [(int(r), int(c), int(t)) for r, c, t in zip(rows, cols, triangles)]
    closed, open_ = sample_wedges(adjacency, degree, triangle_edges, rng)
    print(f"closed wedges {len(closed)}  open wedges {len(open_)}", flush=True)

    report = {"closure": {"closed": len(closed), "open": len(open_), "arms": {}},
              "af3_strata": {"rows": [], "arms": {}}}
    for name, path in ARMS.items():
        sid_path = REPO / path
        if not sid_path.is_file():
            print("skip", name, flush=True)
            continue
        payload = json.loads(sid_path.read_text())
        sid = {int(key): tuple(value[:3]) for key, value in payload.items()}
        entry = {}
        for depth, label in ((1, "L1"), (2, "L1+L2"), (3, "full SID")):
            entry[label] = {"triangle": share_rate(sid, closed, depth, depth),
                            "open_wedge": share_rate(sid, open_, depth, depth)}
        report["closure"]["arms"][name] = entry
        print(f"\n{name}")
        for label in ("L1", "L1+L2", "full SID"):
            t = entry[label]["triangle"]; o = entry[label]["open_wedge"]
            ratio = (t["leaves"] / o["leaves"]) if o["leaves"] else float("inf")
            print(f"  {label:9s} all-three  tri {t['all_three']:.4f} vs open {o['all_three']:.4f}"
                  f"   leaves  tri {t['leaves']:.4f} vs open {o['leaves']:.4f}  (x{ratio:.2f})",
                  flush=True)

    # AF3 strata: degree bucket of the larger endpoint x triangle count bucket.
    degree_bucket = np.array([bucket(int(value)) for value in
                              np.maximum(degree[rows], degree[cols])])
    triangle_bucket = np.where(triangles == 0, 0, np.where(triangles <= 2, 1, 2))
    cells = sorted(set(zip(degree_bucket.tolist(), triangle_bucket.tolist())))
    for arm_name, path in ARMS.items():
        sid_path = REPO / path
        if not sid_path.is_file():
            continue
        payload = json.loads(sid_path.read_text())
        sid = {int(key): tuple(value[:3]) for key, value in payload.items()}
        rows_out = []
        for deg_cell, tri_cell in cells:
            mask = (degree_bucket == deg_cell) & (triangle_bucket == tri_cell)
            if mask.sum() < 200:
                continue
            left = [sid.get(int(item)) for item in rows[mask]]
            right = [sid.get(int(item)) for item in cols[mask]]
            keep = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
            if not keep:
                continue
            share = np.mean([a[:1] == b[:1] for a, b in keep])
            share2 = np.mean([a[:2] == b[:2] for a, b in keep])
            rows_out.append({"arm": arm_name, "degree_bucket": int(deg_cell),
                             "triangle_bucket": int(tri_cell), "edges": int(mask.sum()),
                             "mean_af3": float(af3[mask].mean()),
                             "share_L1": float(share), "share_L1L2": float(share2)})
        report["af3_strata"]["arms"][arm_name] = rows_out
        print(f"\n{arm_name}  (degree bucket x triangle bucket -> share L1)")
        for row in rows_out:
            print(f"  deg<={row['degree_bucket']:5d} T={row['triangle_bucket']}  "
                  f"edges {row['edges']:6d}  af3 {row['mean_af3']:9.2f}  "
                  f"L1 {row['share_L1']:.4f}  L1+L2 {row['share_L1L2']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
