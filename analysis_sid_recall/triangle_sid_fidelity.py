"""Do the three SID levels keep the behaviour graph's triangles?

Measurement only.  The graph is the project's canonical one
(`results/research/behavior_graph/graph.npz`): edges are **adjacent transitions**
of the training sequences (A-B-C-D gives A-B, B-C, C-D), undirected and
deduplicated.  A triangle is three items that are pairwise adjacent -- the
tightest local structure the split offers.

For each arm the exported three-code SID is read and two questions are answered,
at the pair level and at the triangle level:

* pair  : P(same L1), P(same L1+L2), P(same L1+L2+L3) for behaviour edges and,
          for comparison, for degree-matched non-edges;
* triple: P(triangle items all share L1), all share L1+L2, all share one SID,
          again against degree-matched non-triangles.

Degree matching matters: a hub is adjacent to many items, so an unmatched
comparison would report "triangles share SIDs" for the trivial reason that hubs
share SIDs with everybody.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        analysis_sid_recall/triangle_sid_fidelity.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
OUT = REPO / "results" / "research" / "sid_utilization" / "triangle_fidelity.json"
ARMS = {
    "TIGER_RQ-VAE": REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
    "TIGER_sinkhorn_all_levels_rq": REPO / "results/stage2_RQ-VAE/TIGER_sinkhorn_all_levels_rq/item_sids.json",
    "resample (recall .0634)": REPO / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
    "mixed_rq_F0 (recall .0573)": REPO / "results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json",
}
SEED = 42
EDGE_SAMPLES = 200_000
TRIANGLE_SAMPLES = 20_000
MAX_DEGREE_FOR_TRIANGLES = 2000


def behaviour_graph() -> dict:
    """Adjacency from the canonical adjacent-transition graph."""
    with np.load(GRAPH, allow_pickle=False) as payload:
        rows = np.asarray(payload["rows"], dtype=np.int64)
        cols = np.asarray(payload["cols"], dtype=np.int64)
        n_nodes = int(payload["nodes"])
    adjacency = defaultdict(set)
    for left, right in zip(rows.tolist(), cols.tolist()):
        adjacency[left].add(right)
        adjacency[right].add(left)
    return {"adjacency": adjacency, "n_nodes": n_nodes, "edges": int(len(rows))}


def degree(adjacency: dict) -> dict:
    return {item: len(neighbours) for item, neighbours in adjacency.items()}


def degree_bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


def by_bucket(degrees: dict) -> dict:
    buckets = defaultdict(list)
    for item, value in degrees.items():
        buckets[degree_bucket(value)].append(item)
    return buckets


def sample_pairs(adjacency: dict, degrees: dict, rng: random.Random):
    """Behaviour edges and degree-matched non-edges."""
    items = [item for item in adjacency if degrees[item] <= MAX_DEGREE_FOR_TRIANGLES]
    buckets = by_bucket({item: degrees[item] for item in items})
    edges = []
    for left in items:
        for right in adjacency[left]:
            if left < right and degrees[right] <= MAX_DEGREE_FOR_TRIANGLES:
                edges.append((left, right))
    rng.shuffle(edges)
    edges = edges[:EDGE_SAMPLES]
    controls = []
    for left, right in edges:
        for _ in range(20):
            other_left = rng.choice(buckets[degree_bucket(degrees[left])])
            other_right = rng.choice(buckets[degree_bucket(degrees[right])])
            if other_left == other_right:
                continue
            if other_right in adjacency.get(other_left, ()):
                continue
            controls.append((other_left, other_right))
            break
    return edges, controls


def sample_triangles(adjacency: dict, degrees: dict, rng: random.Random):
    """Triangles and degree-matched non-triangles."""
    items = [item for item in adjacency if degrees[item] <= MAX_DEGREE_FOR_TRIANGLES]
    buckets = by_bucket(degrees)
    triangles, controls = [], []
    seeds = [(left, right) for left in items for right in adjacency[left] if left < right]
    rng.shuffle(seeds)
    for left, right in seeds:
        if len(triangles) >= TRIANGLE_SAMPLES:
            break
        common = adjacency[left] & adjacency[right]
        if not common:
            continue
        third = rng.choice(sorted(common))
        if degrees[third] > MAX_DEGREE_FOR_TRIANGLES:
            continue
        triangles.append((left, right, third))
        for _ in range(30):
            pools = [buckets.get(degree_bucket(degrees[item])) for item in (left, right, third)]
            if any(not pool for pool in pools):
                break
            triple = tuple(rng.choice(pool) for pool in pools)
            if len(set(triple)) != 3:
                continue
            if (triple[1] in adjacency.get(triple[0], ())
                    or triple[2] in adjacency.get(triple[0], ())
                    or triple[2] in adjacency.get(triple[1], ())):
                continue
            controls.append(triple)
            break
    return triangles, controls


def load_sid(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return {int(key): tuple(value[:3]) for key, value in payload.items()}


def rates(sid: dict, pairs, triples) -> dict:
    def same(item_left, item_right, depth):
        left, right = sid.get(item_left), sid.get(item_right)
        if left is None or right is None:
            return None
        return left[:depth] == right[:depth]

    def any_pair(triple, depth):
        values = [sid.get(item) for item in triple]
        if any(value is None for value in values):
            return None
        return (values[0][:depth] == values[1][:depth]
                or values[0][:depth] == values[2][:depth]
                or values[1][:depth] == values[2][:depth])

    def all_three(triple, depth):
        values = [sid.get(item) for item in triple]
        if any(value is None for value in values):
            return None
        return values[0][:depth] == values[1][:depth] == values[2][:depth]

    result = {}
    for depth, name in ((1, "L1"), (2, "L1+L2"), (3, "full SID")):
        pair_share = [same(a, b, depth) for a, b in pairs]
        pair_share = [value for value in pair_share if value is not None]
        any_share = [any_pair(t, depth) for t in triples]
        any_share = [value for value in any_share if value is not None]
        all_share = [all_three(t, depth) for t in triples]
        all_share = [value for value in all_share if value is not None]
        result[name] = {
            "pair_rate": float(np.mean(pair_share)) if pair_share else None,
            "triple_any_pair_rate": float(np.mean(any_share)) if any_share else None,
            "triple_all_rate": float(np.mean(all_share)) if all_share else None,
        }
    return result


def main() -> None:
    rng = random.Random(SEED)
    graph = behaviour_graph()
    adjacency = graph["adjacency"]
    degrees = degree(adjacency)
    print(f"graph: {graph['n_nodes']} nodes, {graph['edges']} edges "
          f"({GRAPH.relative_to(REPO)})", flush=True)
    edges, non_edges = sample_pairs(adjacency, degrees, rng)
    triangles, non_triangles = sample_triangles(adjacency, degrees, rng)
    print(f"edges {len(edges)}  matched non-edges {len(non_edges)}  "
          f"triangles {len(triangles)}  matched non-triangles {len(non_triangles)}", flush=True)
    report = {"graph": {"source": str(GRAPH.relative_to(REPO)), "nodes": graph["n_nodes"],
                        "edges": graph["edges"], "edges_sampled": len(edges),
                        "triangles_sampled": len(triangles),
                        "degrees": {"median": float(np.median(list(degrees.values()))),
                                    "max": int(max(degrees.values()))}},
              "arms": {}}
    for name, path in ARMS.items():
        if not path.is_file():
            print("skip", name, "(no SID)", flush=True)
            continue
        sid = load_sid(path)
        edge_rates = rates(sid, edges, triangles)
        control_rates = rates(sid, non_edges, non_triangles)
        report["arms"][name] = {"edges_and_triangles": edge_rates,
                                "matched_controls": control_rates}
        print(f"\n{name}")
        for level in ("L1", "L1+L2", "full SID"):
            e = edge_rates[level]; c = control_rates[level]
            print(f"  {level:9s} pair edge {e['pair_rate']:.4f} vs control {c['pair_rate']:.4f}"
                  f" (lift {e['pair_rate']/max(c['pair_rate'],1e-9):.2f})   "
                  f"triangle all-three {e['triple_all_rate']:.4f} vs {c['triple_all_rate']:.4f}",
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
