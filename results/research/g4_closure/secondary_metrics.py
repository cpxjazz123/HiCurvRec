"""Issue #216 secondary diagnostics: code usage, structure fidelity, reconstruction.

Everything here is measured on the frozen artifacts (SID files + checkpoints) so the
four arms are compared with one piece of code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
G4 = REPO / "results" / "research" / "g4_closure"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
ARMS = ["A", "B", "C", "D", "baseline"]
DIRS = {arm: (REPO / f"results/stage2_RQ-VAE/TIGER_g4_closure_{arm}" if arm != "baseline"
              else REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE") for arm in ARMS}
OUT = G4 / "secondary.json"
TRIANGLE_SAMPLE = 20000


def load_sid(arm: str) -> np.ndarray:
    payload = json.loads((DIRS[arm] / "item_sids.json").read_text())
    return np.array([payload[str(i)] for i in range(len(payload))], dtype=np.int64)


def structure() -> tuple:
    payload = np.load(GRAPH, allow_pickle=False)
    rows, cols = payload["rows"].astype(np.int64), payload["cols"].astype(np.int64)
    degree = np.bincount(np.concatenate([rows, cols]), minlength=int(payload["nodes"]))
    adjacency: list[set] = [set() for _ in range(int(payload["nodes"]))]
    for a, b in zip(rows.tolist(), cols.tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    return rows, cols, degree, adjacency


def share(codes: np.ndarray, left: np.ndarray, right: np.ndarray, depth: int) -> float:
    first = codes[left, :depth]
    second = codes[right, :depth]
    return float((first == second).all(axis=1).mean())


def main() -> None:
    rows, cols, degree, adjacency = structure()
    rng = np.random.default_rng(216)
    triangles = []
    checked = 0
    for a in range(len(adjacency)):
        neighbours = sorted(adjacency[a])
        for index, b in enumerate(neighbours):
            for c in neighbours[index + 1:]:
                if c in adjacency[b]:
                    triangles.append((a, b, c))
                    checked += 1
        if len(triangles) >= TRIANGLE_SAMPLE * 4:
            break
    triangles = np.array(triangles, dtype=np.int64)
    if len(triangles) > TRIANGLE_SAMPLE:
        triangles = triangles[rng.choice(len(triangles), TRIANGLE_SAMPLE, replace=False)]
    heldout = np.load(G4 / "pairs_heldout.npz")
    matched_non_edges = heldout["triples"][:, [1, 2]].astype(np.int64)
    results = {}
    for arm in ARMS:
        codes = load_sid(arm)
        l1 = codes[:, 0]
        counts = np.bincount(l1, minlength=256)
        entry = {
            "L1_used": int((counts > 0).sum()),
            "L1_max_bucket": int(counts.max()),
            "L1_max_share": float(counts.max() / len(codes)),
            "L2_prefixes": int(len({tuple(row[:2]) for row in codes.tolist()})),
            "L3_triples": int(len({tuple(row[:3]) for row in codes.tolist()})),
            "collision": float(1 - len({tuple(row[:3]) for row in codes.tolist()}) / len(codes)),
            "edge_L1": share(codes, rows, cols, 1),
            "edge_L1L2": share(codes, rows, cols, 2),
            "matched_non_edge_L1": share(codes, matched_non_edges[:, 0], matched_non_edges[:, 1], 1),
            "matched_non_edge_L1L2": share(codes, matched_non_edges[:, 0], matched_non_edges[:, 1], 2),
            "triangle_L1": share(codes, triangles[:, 0], triangles[:, 1], 1),
            "triangle_L1L2": share(codes, triangles[:, 0], triangles[:, 1], 2),
            "triangles_sampled": int(len(triangles)),
        }
        results[arm] = entry
        print(f"{arm:9s} L1 used {entry['L1_used']:3d} max {entry['L1_max_bucket']:5d} "
              f"edge L1 {entry['edge_L1']:.4f} non-edge L1 {entry['matched_non_edge_L1']:.4f} "
              f"edge L1L2 {entry['edge_L1L2']:.4f} non-edge L1L2 {entry['matched_non_edge_L1L2']:.4f} "
              f"tri L1 {entry['triangle_L1']:.4f}")
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
