"""The frozen AF3 edge evidence of issue #154 (TRAIN interactions only).

AF3(e) = 4 - deg(u) - deg(v) + 3*m_e on the training behaviour graph, m_e the number of
triangles containing the edge. The quartile cuts q1/q3 split edges into the high set H
(>= q3) and the low set B (<= q1). The table, cuts and coverage counts are written once to
results/research/issue154/af3_edges.json with a content SHA.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "research" / "issue150" / "structure_prior.npz"
OUT = ROOT / "results" / "research" / "issue154" / "af3_edges.json"


def pair_keys(tail: np.ndarray, head: np.ndarray, nodes: int) -> np.ndarray:
    low, high = np.minimum(tail, head), np.maximum(tail, head)
    return low * np.int64(nodes) + high


def main() -> int:
    raw = np.load(SOURCE, allow_pickle=True)
    edges = raw["edges"].astype(np.int64)
    triangles = raw["triangles"].astype(np.int64)
    nodes = int(max(edges.max(initial=0), triangles.max(initial=0))) + 1
    degree = np.bincount(edges.reshape(-1), minlength=nodes)

    edge_keys = pair_keys(edges[:, 0], edges[:, 1], nodes)
    order = np.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    tri_keys = np.concatenate([pair_keys(triangles[:, 0], triangles[:, 1], nodes),
                               pair_keys(triangles[:, 1], triangles[:, 2], nodes),
                               pair_keys(triangles[:, 0], triangles[:, 2], nodes)])
    unique_tri, triangle_counts = np.unique(tri_keys, return_counts=True)
    position = np.searchsorted(sorted_keys, unique_tri)
    valid = position < sorted_keys.size
    valid[valid] = sorted_keys[position[valid]] == unique_tri[valid]
    multiplicity = np.zeros(edge_keys.size, dtype=np.int64)
    multiplicity[order[position[valid]]] = triangle_counts[valid]

    af3 = 4.0 - degree[edges[:, 0]] - degree[edges[:, 1]] + 3.0 * multiplicity
    q1, q3 = (float(value) for value in np.quantile(af3, [0.25, 0.75]))
    high = af3 >= q3
    low = af3 <= q1

    pitch = np.int64(nodes)
    lookup = {int(key): int(index) for index, key in enumerate(sorted_keys.tolist())}
    closed = 0
    for a, b, c in triangles.tolist():
        members = [int(a) * int(pitch) + int(b) if a <= b else int(b) * int(pitch) + int(a),
                   int(b) * int(pitch) + int(c) if b <= c else int(c) * int(pitch) + int(b),
                   int(a) * int(pitch) + int(c) if a <= c else int(c) * int(pitch) + int(a)]
        indices = [lookup.get(key) for key in members]
        if all(index is not None and bool(high[index]) for index in indices):
            closed += 1

    payload = {
        "issue": 154,
        "source": "results/research/issue150/structure_prior.npz",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rule": "AF3(e) = 4 - deg(u) - deg(v) + 3*m_e, TRAIN interactions only",
        "q1": q1,
        "q3": q3,
        "counts": {"edges": int(edge_keys.size), "triangles": int(triangles.shape[0]),
                   "high_edges": int(high.sum()), "low_edges": int(low.sum()),
                   "triangles_all_high": int(closed)},
        "af3_quantiles": {str(q): float(np.quantile(af3, q))
                          for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "blind_spot": {"items_without_structure": 31, "nodes": nodes},
        "high_edges": [[int(u), int(v)] for u, v in edges[high]],
        "low_edges": [[int(u), int(v)] for u, v in edges[low]],
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in
                      ("q1", "q3", "counts", "af3_quantiles", "blind_spot")}, indent=2))
    print("sha256", hashlib.sha256(text.encode()).hexdigest(), "bytes", len(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
