"""Relate local structural roles to per-item curvature.

Issue #235 C20 asks for the relation between chain/fork/merge/closed structures and
item-level curvature, stratified by degree -- and for an explicit statement that a
curvature sign does not identify a directed type or a graphlet.

The frozen Graphlet banks hold supervision *samples* (1,795 / 1,772 / 1,894 triples),
not a per-node role table, so the structural variables used here are computed on the
same canonical graph and are therefore exact for every node:

* ``triangles``      -- closed structure: distinct triangles through the node;
* ``clustering``     -- local clustering coefficient (0 = tree-like, 1 = clique-like);
* ``open_wedges``    -- forks: neighbour pairs not joined to each other, i.e. exactly
                        the triples Eq. A.1 averages over, so curvature is a function
                        of these and nothing else;
* ``degree`` and its strata.

Everything is reported as a correlation and a stratified table; nothing here is
evidence about direction.

Usage (no arguments, per CLAUDE.md 9-10):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue235/run_structure.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import run_analysis as ra

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue235"
BANKS = REPO / "results" / "research" / "issue231" / "banks" / "banks_global.json"


def local_structure(adjacency: sp.csr_matrix) -> dict:
    """Degree, triangle count, clustering and open-wedge count per node."""
    adjacency = adjacency.tocsr().astype(np.int32)
    degree = np.diff(adjacency.indptr).astype(np.int64)
    # triangles through v = (A^3)_vv / 2; A has no self loops
    cubed = (adjacency @ adjacency).multiply(adjacency)
    triangles = np.asarray(cubed.sum(axis=1)).ravel() // 2
    wedge_pairs = degree * (degree - 1) // 2
    closed_pairs = triangles * 2  # each triangle closes 2 of the node's neighbour pairs
    open_wedges = wedge_pairs - closed_pairs
    with np.errstate(invalid="ignore", divide="ignore"):
        clustering = np.where(wedge_pairs > 0, closed_pairs / wedge_pairs, 0.0)
    return {"degree": degree, "triangles": triangles.astype(np.int64),
            "open_wedges": open_wedges.astype(np.int64), "clustering": clustering}


def main() -> int:
    started = time.time()
    adjacency, metadata = ra.load_graph()
    structure = local_structure(adjacency)
    print(f"structure computed in {time.time() - started:.0f}s", flush=True)

    rows = {}
    for row in _read_csv(OUT / "node_curvature.csv"):
        rows[int(row["item"])] = row

    # the frozen graphlet banks: sample counts per type, for the report only
    bank_info = json.loads(BANKS.read_text(encoding="utf-8")) if BANKS.is_file() else {}

    items = []
    for node, row in rows.items():
        if int(row["n_triples"]) == 0:
            continue
        items.append({
            "item": node,
            "degree": structure["degree"][node],
            "triangles": structure["triangles"][node],
            "clustering": float(structure["clustering"][node]),
            "open_wedges": structure["open_wedges"][node],
            "mean_kappa": float(row["mean"]),
            "mixed": row["mixed_sign"] == "True",
        })
    print(f"{len(items)} items with curvature", flush=True)

    degree = np.array([item["degree"] for item in items], dtype=float)
    kappa = np.array([item["mean_kappa"] for item in items])
    clustering = np.array([item["clustering"] for item in items])
    triangles = np.array([item["triangles"] for item in items], dtype=float)

    report = {
        "items": len(items),
        "spearman": {
            "degree_vs_kappa": _spearman(degree, kappa),
            "clustering_vs_kappa": _spearman(clustering, kappa),
            "triangles_vs_kappa": _spearman(triangles, kappa),
        },
        "closure_strata": _closure_strata(items),
        "degree_strata": _degree_strata(items),
        "graphlet_banks": {
            "path": str(BANKS.relative_to(REPO)) if BANKS.is_file() else None,
            "counts": bank_info.get("counts"),
            "note": ("the banks hold supervision samples, not a per-node role table; the "
                     "structural variables above are computed exactly for every node"),
        },
        "caveats": [
            "curvature is a function of the open wedge count and the distances inside each "
            "wedge, so any correlation with degree is partly algebraic",
            "the graph is an undirected projection: nothing here identifies a directed "
            "chain/fork/merge type, and curvature cannot uniquely identify a graphlet",
        ],
        "elapsed_s": round(time.time() - started, 1),
    }
    path = OUT / "structure_summary.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"-> {path}")
    return 0


def _read_csv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(values.size, dtype=float)
        return ranks
    a, b = rank(left), rank(right)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else float("nan")


def _closure_strata(items: list[dict]) -> list[dict]:
    """Open (no triangles) vs closed (has triangles) structure."""
    out = []
    for label, predicate in (("open (0 triangles)", lambda it: it["triangles"] == 0),
                             ("closed (>=1 triangle)", lambda it: it["triangles"] > 0)):
        block = [item for item in items if predicate(item)]
        if not block:
            continue
        values = np.array([item["mean_kappa"] for item in block])
        out.append({
            "structure": label,
            "items": len(block),
            "mean_kappa": float(values.mean()),
            "median_kappa": float(np.median(values)),
            "negative_fraction": float((values < -ra.NEAR_ZERO).mean()),
            "mixed_fraction": float(np.mean([item["mixed"] for item in block])),
            "median_degree": float(np.median([item["degree"] for item in block])),
            "median_clustering": float(np.median([item["clustering"] for item in block])),
        })
    return out


def _degree_strata(items: list[dict]) -> list[dict]:
    edges = [1, 5, 13, 50, 200, 10 ** 9]
    labels = ["1-4", "5-12", "13-49", "50-199", "200+"]
    out = []
    for low, high, label in zip(edges[:-1], edges[1:], labels):
        block = [item for item in items if low <= item["degree"] < high]
        if not block:
            continue
        values = np.array([item["mean_kappa"] for item in block])
        out.append({
            "degree_range": label,
            "items": len(block),
            "mean_kappa": float(values.mean()),
            "median_triangles": float(np.median([item["triangles"] for item in block])),
            "median_clustering": float(np.median([item["clustering"] for item in block])),
            "mixed_fraction": float(np.mean([item["mixed"] for item in block])),
        })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
