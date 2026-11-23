"""Frozen-asset plumbing (issue #245 C3/C5/C18/C19/C22).

Every frozen file carries a manifest entry: name, path, schema, shape, dtype, sha256, the
command that produced it, its legal uses and its dependencies.  ``verify`` recomputes the
hashes and fails closed on any drift.

``join_scores`` is the only supported way to attach a future model's per-pair numbers to the
frozen bank: missing pairs, duplicates and misaligned ids raise instead of silently producing
a shifted vector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_stats(csr: sp.csr_matrix) -> dict:
    """Node/edge/degree/reciprocal/isolated/component statistics for one directed graph."""
    n = csr.shape[0]
    out_degree = np.diff(csr.indptr)
    in_csr = csr.T.tocsr()
    in_degree = np.diff(in_csr.indptr)
    coo = csr.tocoo()
    edges = set(zip(coo.row.tolist(), coo.col.tolist()))
    reciprocal = sum(1 for a, b in edges if (b, a) in edges) // 2
    undirected = (csr + csr.T).tocsr()
    undirected.data[:] = 1
    from scipy.sparse import csgraph
    _, labels = csgraph.connected_components(undirected, directed=False)
    return {
        "nodes": int(n),
        "directed_edges": int(len(edges)),
        "self_loops": int(sum(1 for a, b in edges if a == b)),
        "reciprocal_pairs": int(reciprocal),
        "isolated_nodes": int(((out_degree == 0) & (in_degree == 0)).sum()),
        "nodes_with_out_edge": int((out_degree > 0).sum()),
        "nodes_with_in_edge": int((in_degree > 0).sum()),
        "out_degree_max": int(out_degree.max()) if n else 0,
        "in_degree_max": int(in_degree.max()) if n else 0,
        "out_degree_median": float(np.median(out_degree)) if n else 0.0,
        "components": int(labels.max() + 1) if n else 0,
        "largest_component": int(np.bincount(labels).max()) if n else 0,
    }


def save_sparse(path: Path, matrix: sp.spmatrix) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indptr=matrix.indptr, indices=matrix.indices,
                        data=matrix.data, shape=np.array(matrix.shape, dtype=np.int64))


def load_sparse(path: Path) -> sp.csr_matrix:
    blob = np.load(path)
    return sp.csr_matrix((blob["data"], blob["indices"], blob["indptr"]),
                         shape=tuple(int(x) for x in blob["shape"]))


def entry(name: str, path: Path, *, schema: str, purpose: str, command: str,
          dependencies=None, extra: dict | None = None) -> dict:
    payload = {
        "name": name,
        "path": str(Path(path)),
        "schema": schema,
        "purpose": purpose,
        "command": command,
        "sha256": sha256(path),
        "bytes": Path(path).stat().st_size,
        "dependencies": dependencies or [],
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: Path, entries: list[dict], *, issue: int, notes: str = "") -> None:
    payload = {"issue": issue, "notes": notes, "assets": entries}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_manifest(path: Path) -> dict:
    """Recompute every hash; raise on drift, return the verified names."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    problems, verified = [], []
    for asset in payload.get("assets", []):
        target = Path(asset["path"])
        if not target.exists():
            problems.append(f"missing: {asset['path']}")
            continue
        if sha256(target) != asset["sha256"]:
            problems.append(f"hash drift: {asset['path']}")
            continue
        verified.append(asset["name"])
    if problems:
        raise ValueError("asset manifest verification failed: " + "; ".join(problems))
    return {"issue": payload.get("issue"), "verified": verified}


def join_scores(bank_pairs: np.ndarray, model_pairs: np.ndarray, model_scores: np.ndarray) -> np.ndarray:
    """Align a model's per-pair scores onto the frozen bank; fail closed on any mismatch."""
    bank_pairs = np.asarray(bank_pairs, dtype=np.int64)
    model_pairs = np.asarray(model_pairs, dtype=np.int64)
    model_scores = np.asarray(model_scores, dtype=np.float64)
    if model_pairs.shape != bank_pairs.shape:
        raise ValueError(f"pair count mismatch: model {model_pairs.shape}, bank {bank_pairs.shape}")
    if model_scores.shape[0] != model_pairs.shape[0]:
        raise ValueError("score count does not match pair count")
    index = {}
    for row, (i, j) in enumerate(model_pairs.tolist()):
        key = (int(i), int(j))
        if key in index:
            raise ValueError(f"duplicate pair in model output: {key}")
        index[key] = row
    aligned = np.empty(len(bank_pairs), dtype=np.float64)
    missing = []
    for row, (i, j) in enumerate(bank_pairs.tolist()):
        key = (int(i), int(j))
        if key not in index:
            missing.append(key)
            continue
        aligned[row] = model_scores[index[key]]
    if missing:
        raise ValueError(f"{len(missing)} bank pairs missing from the model output, e.g. {missing[:3]}")
    if not np.isfinite(aligned).all():
        raise ValueError("non-finite scores after the join")
    return aligned
