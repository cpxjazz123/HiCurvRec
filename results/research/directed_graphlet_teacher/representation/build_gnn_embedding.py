#!/usr/bin/env python3
"""Freeze the trained behaviour GNN's item representation into a table RQ-VAE can read.

The GNN is only run **while building** the table, exactly like the relation target in
`teacher/structure/v3`.  After this script finishes, the graph is not needed again: the
table is a plain ``(n, 128)`` float32 matrix whose row ``i`` is item ``i`` of the Stage1
and Stage2 item universe.

Frozen choices (no tuning, no seeds to pick):

* checkpoint ``C_residual_edge_main_seed226.pth`` -- the seed that produced the existing
  relation target, verified against its own ``param_sha``;
* the message graph used during training (192528 directed rows from
  ``teacher/opt/data_contract.npz``), whose sha must equal the cached contract's
  ``graph_sha``; otherwise the run stops instead of silently using another graph;
* the Stage1 ``sentence_t5.npy`` the encoder was trained on; the shas are recorded.

Output: ``representation/gnn_item_embedding.npz`` holding ``embedding`` and ``item_ids``
plus ``representation/manifest.json`` with every input sha and the coverage counts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
sys.path.insert(0, str(TEACHER))

import graph_teacher3 as gt3  # noqa: E402

REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir()
            and (parent / "stage1_GeneEmbedding").is_dir()
            and (parent / "results").is_dir())
CHECKPOINT = TEACHER / "teacher" / "opt" / "C_residual_edge_main_seed226.pth"
CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
CONTRACT_META = TEACHER / "teacher" / "opt" / "data_contract.json"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUT = HERE / "gnn_item_embedding.npz"
MANIFEST = HERE / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def undirected_graph_sha(rows: np.ndarray, cols: np.ndarray) -> str:
    """The sha the training contract recorded: COO rows then COO cols of the CSR graph."""
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    return hashlib.sha256(
        np.stack([rows, cols]).astype(np.int64).tobytes()).hexdigest()


def training_graph():
    """Undirected edges and sha, recomputed exactly as graph_teacher3.prepare_data does."""
    sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_role"))
    from directed_graphlets import load_adjacency
    graph = load_adjacency()
    rows, cols = graph.nonzero()
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return rows, cols, undirected_graph_sha(rows, cols)


def build_adjacency(edges: np.ndarray, n_items: int) -> torch.Tensor:
    indices = torch.from_numpy(edges.T.astype(np.int64))
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n_items, n_items)).coalesce()


def validate_graph(edges: np.ndarray, meta: dict, n_items: int,
                   source_sha: str) -> dict:
    """The encoder was trained on a specific message graph; refuse a different one."""
    rows, cols = edges[:, 0], edges[:, 1]
    measured = undirected_graph_sha(*training_graph()[:2])
    if measured != meta["graph_sha"]:
        raise SystemExit(f"message graph changed: {measured} != {meta['graph_sha']}")
    if edges.shape[0] != meta["message_edges"]:
        raise SystemExit("edge count disagrees with the cached contract")
    if max(int(rows.max()), int(cols.max())) >= n_items or edges.min() < 0:
        raise SystemExit("edge endpoints fall outside the item universe")
    return {"source_sha": source_sha, "message_rows": int(edges.shape[0]),
            "directed": True, "self_loop_rows": int((rows == cols).sum())}


def coverage(edges: np.ndarray, n_items: int) -> dict:
    degree = np.bincount(edges.ravel(), minlength=n_items)
    return {"items": int(n_items), "isolated_items": int((degree == 0).sum()),
            "degree_median": float(np.median(degree)), "degree_max": int(degree.max()),
            "items_with_edge": int((degree > 0).sum())}


def encode(checkpoint: Path, features: torch.Tensor, adjacency: torch.Tensor,
           device: torch.device) -> np.ndarray:
    variant = gt3.build_variant("C_residual")
    payload = gt3.load_checkpoint(checkpoint, variant)   # verifies param_sha
    variant.to(device).eval()
    with torch.no_grad():
        state = variant.encode(features, adjacency).float()
    return state.cpu().numpy().astype(np.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_np = np.load(SEMANTICS)
    n_items = int(features_np.shape[0])
    contract = np.load(CONTRACT)
    meta = json.loads(CONTRACT_META.read_text())
    edges = contract["message_edges"].astype(np.int64)
    rows, cols, source_sha = training_graph()
    graph = validate_graph(edges, meta, n_items, source_sha)

    features = torch.from_numpy(features_np.astype(np.float32)).to(device)
    adjacency = build_adjacency(edges, n_items).to(device)
    embedding = encode(CHECKPOINT, features, adjacency, device)
    if not np.isfinite(embedding).all():
        raise SystemExit("encoder produced non-finite values")
    item_ids = np.arange(n_items, dtype=np.int64)
    np.savez_compressed(OUT, embedding=embedding, item_ids=item_ids)

    norms = np.linalg.norm(embedding, axis=1)
    manifest = {
        "version": "gnn-item-embedding-v1",
        "checkpoint": {"path": str(CHECKPOINT.relative_to(REPO)),
                       "sha256": sha256(CHECKPOINT)},
        "semantics": {"path": str(SEMANTICS.relative_to(REPO)),
                      "sha256": sha256(SEMANTICS), "shape": list(features_np.shape)},
        "graph": graph,
        "output": {"path": str(OUT.relative_to(REPO)), "sha256": sha256(OUT),
                   "shape": list(embedding.shape), "dtype": str(embedding.dtype),
                   "bytes": OUT.stat().st_size},
        "coverage": coverage(edges, n_items),
        "value_statistics": {"norm_mean": float(norms.mean()),
                             "norm_median": float(np.median(norms)),
                             "components_std_mean": float(embedding.std(axis=0).mean()),
                             "components_std_max": float(embedding.std(axis=0).max()),
                             "components_std_min": float(embedding.std(axis=0).min())},
        "graph_needed_after_this": False,
        "rqvae_encoder_input": "the 128-d table above (control arm only)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
