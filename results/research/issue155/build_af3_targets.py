"""The frozen per-edge AF3 targets of issue #155 (TRAIN interactions only).

AF3(e) = 4 - deg(u) - deg(v) + 3*m_e, monotone-normalised with the q05/q95 cuts into
u_e = clip((AF3 - q05) / (q95 - q05), 0, 1). u == 1 is the high set, u == 0 the low set; the
clipped extremes are exactly those cuts, so no extra threshold is invented.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "research" / "issue150" / "structure_prior.npz"
OUT = ROOT / "results" / "research" / "issue155" / "af3_edges.json"


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
    unique_tri, counts = np.unique(tri_keys, return_counts=True)
    position = np.searchsorted(sorted_keys, unique_tri)
    valid = position < sorted_keys.size
    valid[valid] = sorted_keys[position[valid]] == unique_tri[valid]
    multiplicity = np.zeros(edge_keys.size, dtype=np.int64)
    multiplicity[order[position[valid]]] = counts[valid]
    af3 = 4.0 - degree[edges[:, 0]] - degree[edges[:, 1]] + 3.0 * multiplicity

    q05, q95 = (float(value) for value in np.quantile(af3, [0.05, 0.95]))
    span = q95 - q05
    u = np.clip((af3 - q05) / span, 0.0, 1.0) if span > 0 else np.zeros_like(af3)
    high = u >= 1.0
    low = u <= 0.0

    lookup = {int(key): int(index) for index, key in enumerate(sorted_keys.tolist())}
    pitch = int(nodes)
    closed = 0
    for a, b, c in triangles.tolist():
        members = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]
        indices = [lookup.get(tail * pitch + head) for tail, head in members]
        if all(index is not None and bool(high[index]) for index in indices):
            closed += 1

    payload = {
        "issue": 155,
        "source": "results/research/issue150/structure_prior.npz",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rule": "AF3(e) = 4 - deg(u) - deg(v) + 3*m_e; u = clip((AF3 - q05)/(q95 - q05), 0, 1)",
        "q05": q05,
        "q95": q95,
        "counts": {"edges": int(edge_keys.size), "triangles": int(triangles.shape[0]),
                   "high_edges": int(high.sum()), "low_edges": int(low.sum()),
                   "triangles_all_high": int(closed)},
        "blind_spot": {"items_without_structure": 31, "nodes": nodes},
        "edges": [[int(tail), int(head)] for tail, head in edges],
        "u": [round(float(value), 4) for value in u],
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("q05", "q95", "counts", "blind_spot")},
                     indent=2))
    print("sha256", hashlib.sha256(text.encode()).hexdigest(), "bytes", len(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
