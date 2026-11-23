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


# ---------------------------------------------------------------- v2 manifest and join
REQUIRED_V2 = ("name", "path", "schema", "shape", "dtype", "graph_version", "legal_use", "version",
               "purpose", "command", "source_sha256", "code_sha256", "dependencies", "sha256", "bytes")
LEGAL_USES = ("description", "generalisation", "evaluation", "reuse")
_USE_ORDER = {"description": 0, "reuse": 1, "evaluation": 2, "generalisation": 3}


def entry_v2(name: str, path: Path, *, repo: Path, schema: str, shape, dtype: str,
             graph_version: str, legal_use: str, version: str, purpose: str, command: str,
             source_sha256: str, code_sha256: str, dependencies, extra: dict | None = None) -> dict:
    """A v2 entry: repo-relative path, machine-checkable legal use, provenance hashes."""
    if legal_use not in LEGAL_USES:
        raise ValueError(f"legal_use must be one of {LEGAL_USES}, got {legal_use!r}")
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(Path(repo).resolve())
    except ValueError as error:
        raise ValueError(f"asset {name} is outside the repository: {path}") from error
    payload = {
        "name": name,
        "path": str(relative),
        "schema": schema,
        "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
        "dtype": dtype,
        "graph_version": graph_version,
        "legal_use": legal_use,
        "version": version,
        "purpose": purpose,
        "command": command,
        "source_sha256": source_sha256,
        "code_sha256": code_sha256,
        "dependencies": list(dependencies),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if extra:
        payload.update(extra)
    return payload


def audit_manifest(path: Path) -> dict:
    """Check the v2 schema; returns issues instead of raising so callers can report them."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    issues = []
    if payload.get("version") != "v2":
        issues.append(f"manifest version is {payload.get('version')!r}, expected 'v2'")
    for entry in payload.get("assets", []):
        missing = [key for key in REQUIRED_V2 if key not in entry]
        if missing:
            issues.append(f"{entry.get('name', '?')}: missing {missing}")
        if Path(str(entry.get("path", ""))).is_absolute():
            issues.append(f"{entry.get('name', '?')}: absolute path")
        if entry.get("legal_use") not in LEGAL_USES:
            issues.append(f"{entry.get('name', '?')}: illegal legal_use {entry.get('legal_use')!r}")
    return {"ok": not issues, "issues": issues, "assets": len(payload.get("assets", []))}


def write_manifest_v2(path: Path, entries: list, *, issue: int, notes: str = "") -> None:
    payload = {"issue": issue, "version": "v2", "notes": notes, "assets": entries}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_legal_use(manifest_path: Path, name: str, use: str) -> dict:
    """Refuse a reader whose request outranks the asset's declared legal use."""
    if use not in LEGAL_USES:
        raise ValueError(f"unknown use {use!r}")
    payload = load_manifest(manifest_path)
    entry = next((item for item in payload["assets"] if item["name"] == name), None)
    if entry is None:
        raise KeyError(f"{name} not in {manifest_path}")
    if _USE_ORDER[use] > _USE_ORDER[entry["legal_use"]]:
        raise ValueError(f"{name} is declared legal_use={entry['legal_use']} but was read as {use}; "
                         f"a {entry['legal_use']}-only asset cannot serve {use}")
    return entry


def join_scores_v2(bank_pairs: np.ndarray, model_pairs: np.ndarray, model_scores: np.ndarray,
                   *, expected_graph_version: str, bank_graph_version: str) -> np.ndarray:
    """Strict, id-keyed join; any shape, dtype, set or finiteness violation raises."""
    if not isinstance(expected_graph_version, str) or not expected_graph_version:
        raise ValueError("expected_graph_version is required")
    if not isinstance(bank_graph_version, str) or not bank_graph_version:
        raise ValueError("bank_graph_version is required")
    if bank_graph_version != expected_graph_version:
        raise ValueError(f"graph version mismatch: bank {bank_graph_version!r} vs expected "
                         f"{expected_graph_version!r}")
    bank_pairs = np.asarray(bank_pairs)
    model_pairs = np.asarray(model_pairs)
    model_scores = np.asarray(model_scores)
    if bank_pairs.ndim != 2 or bank_pairs.shape[1] != 2:
        raise ValueError(f"bank pairs must be (N,2), got {bank_pairs.shape}")
    if model_pairs.ndim != 2 or model_pairs.shape[1] != 2:
        raise ValueError(f"model pairs must be (N,2), got {model_pairs.shape}")
    if not np.issubdtype(bank_pairs.dtype, np.integer) or not np.issubdtype(model_pairs.dtype, np.integer):
        raise ValueError("pair ids must be integers")
    if model_scores.ndim != 1:
        raise ValueError(f"model scores must be 1-D, got {model_scores.shape}")
    if model_scores.shape[0] != model_pairs.shape[0]:
        raise ValueError("score count does not match pair count")
    if not np.issubdtype(model_scores.dtype, np.floating):
        model_scores = model_scores.astype(np.float64)
    if not np.all(np.isfinite(model_scores)):
        raise ValueError("model scores contain non-finite values")
    bank_index = {}
    for position, (left, right) in enumerate(bank_pairs.tolist()):
        key = (int(left), int(right))
        if key in bank_index:
            raise ValueError(f"duplicate pair in the bank: {key}")
        bank_index[key] = position
    model_index = {}
    for left, right, score in zip(model_pairs[:, 0].tolist(), model_pairs[:, 1].tolist(),
                                  model_scores.tolist()):
        key = (int(left), int(right))
        if key in model_index:
            raise ValueError(f"duplicate pair in the model output: {key}")
        model_index[key] = float(score)
    missing = sorted(set(bank_index) - set(model_index))
    extra = sorted(set(model_index) - set(bank_index))
    if missing:
        raise ValueError(f"{len(missing)} bank pairs missing from the model output, e.g. {missing[:3]}")
    if extra:
        raise ValueError(f"{len(extra)} model pairs absent from the bank, e.g. {extra[:3]}")
    aligned = np.empty(len(bank_pairs), dtype=np.float64)
    for key, position in bank_index.items():
        aligned[position] = model_index[key]
    return aligned
