"""Issue #217 secondary structure fidelity for the five arms and the frozen baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
OUT = REPO / "results" / "research" / "graphlet_roles" / "secondary.json"
ARMS = ("E", "T", "R", "Q", "QR", "baseline")
TRIANGLES = 20000


def structure():
    payload = np.load(GRAPH, allow_pickle=False)
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    adjacency = [set() for _ in range(int(payload["nodes"]))]
    for a, b in zip(rows.tolist(), cols.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    return rows, cols, adjacency


def load_sid(arm: str) -> np.ndarray:
    directory = (REPO / "results" / "stage2_RQ-VAE" / f"TIGER_graphlet_role_{arm}"
                 if arm != "baseline" else REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE")
    payload = json.loads((directory / "item_sids.json").read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def share(codes: np.ndarray, left: np.ndarray, right: np.ndarray, depth: int) -> float:
    return float((codes[left, :depth] == codes[right, :depth]).all(axis=1).mean())


def main() -> None:
    rows, cols, adjacency = structure()
    rng = np.random.default_rng(217)
    triangles = []
    for a in range(len(adjacency)):
        neighbours = sorted(adjacency[a])
        for index, b in enumerate(neighbours):
            for c in neighbours[index + 1:]:
                if c in adjacency[b]:
                    triangles.append((a, b, c))
        if len(triangles) >= TRIANGLES * 4:
            break
    triangles = np.array(triangles, dtype=np.int64)
    triangles = triangles[rng.choice(len(triangles), TRIANGLES, replace=False)]
    heldout = np.load(REPO / "results" / "research" / "g4_closure" / "pairs_heldout.npz")
    non_edges = heldout["triples"][:, [1, 2]].astype(np.int64)
    results = {}
    for arm in ARMS:
        try:
            codes = load_sid(arm)
        except FileNotFoundError:
            continue
        l1 = np.bincount(codes[:, 0], minlength=256)
        entry = {
            "L1_used": int((l1 > 0).sum()),
            "L1_max_bucket": int(l1.max()),
            "L2_prefixes": int(len({tuple(row[:2]) for row in codes.tolist()})),
            "collision": float(1 - len({tuple(row[:3]) for row in codes.tolist()}) / len(codes)),
            "edge_L1": share(codes, rows, cols, 1),
            "edge_L1L2": share(codes, rows, cols, 2),
            "matched_non_edge_L1": share(codes, non_edges[:, 0], non_edges[:, 1], 1),
            "triangle_L1": share(codes, triangles[:, 0], triangles[:, 1], 1),
        }
        results[arm] = entry
        print(f"{arm:9s} edge {entry['edge_L1']:.4f} non-edge {entry['matched_non_edge_L1']:.4f} "
              f"tri {entry['triangle_L1']:.4f} coll {entry['collision']:.4f} "
              f"L1max {entry['L1_max_bucket']:5d} L2pfx {entry['L2_prefixes']}")
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
