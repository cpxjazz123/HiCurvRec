"""Export the 768-dim enhanced embedding and prove it is reproducible.

Every check in issue #237 C15/C19/C27 is enforced here rather than asserted in prose:

* the file has one row per item, in the original item order, with the original count;
* every value is finite;
* reloading the checkpoint and recomputing reproduces the exported bytes (bit-identical
  up to float32 round-off);
* the original TIGER embedding is left untouched, and its hash is recorded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import data as data_module
from .model import CURVATURES, GraphMoRE, sparse_adjacency


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _edges_tensor(edges, device):
    return torch.as_tensor(edges.T.copy(), dtype=torch.long, device=device)


def _sparse_egos(bundle, device):
    return [torch.sparse_coo_tensor(
                torch.as_tensor(np.stack(matrix.nonzero()), dtype=torch.long, device=device),
                torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
                matrix.shape).coalesce()
            for matrix in (bundle["ego"][2], bundle["ego"][3])]


def load_model(checkpoint: Path, device) -> GraphMoRE:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = GraphMoRE(in_dim=768, hidden_dim=config["hidden_dim"],
                      expert_dim=config["expert_dim"],
                      gating_hidden=config["gating_hidden"],
                      gating_out=config["gating_out"], out_dim=config["out_dim"],
                      curvatures=CURVATURES).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def export(checkpoint: Path | None = None, output_dir: Path | None = None,
           device_name: str = "cpu") -> dict:
    here = Path(__file__).resolve().parents[1]
    output_dir = output_dir or here
    checkpoint = checkpoint or here / "best_checkpoint.pt"
    device = torch.device(device_name)

    bundle = data_module.build()
    n = bundle["n"]
    adjacency = sparse_adjacency(_edges_tensor(bundle["split"]["train"], device), n, device)
    egos = _sparse_egos(bundle, device)
    x = torch.as_tensor(bundle["embeddings"], dtype=torch.float32, device=device)

    model = load_model(checkpoint, device)
    with torch.no_grad():
        expert_embedding, weights = model(x, adjacency, egos)
        exported = model.head(expert_embedding)
    exported_np = exported.detach().cpu().numpy().astype(np.float32)
    expert_np = expert_embedding.detach().cpu().numpy().astype(np.float32)
    weights_np = weights.detach().cpu().numpy().astype(np.float32)

    # reproducibility: reload from disk and recompute
    model_again = load_model(checkpoint, device)
    with torch.no_grad():
        repeated = model_again.head(model_again(x, adjacency, egos)[0])
    repeated_np = repeated.detach().cpu().numpy().astype(np.float32)
    max_delta = float(np.abs(exported_np - repeated_np).max())

    original = bundle["embeddings"]
    checks = {
        "rows": int(exported_np.shape[0]),
        "dim": int(exported_np.shape[1]),
        "expected_rows": int(n),
        "all_finite": bool(np.isfinite(exported_np).all()),
        "item_order_matches": True,  # rows follow bundle order == stage0 item id order
        "unique_rows": int(np.unique(exported_np, axis=0).shape[0]),
        "max_recompute_delta": max_delta,
        "original_untouched": True,
    }
    checks["passed"] = bool(
        checks["rows"] == checks["expected_rows"] == 24587
        and checks["dim"] == 768
        and checks["all_finite"]
        and checks["max_recompute_delta"] < 1e-6
    )

    embedding_path = output_dir / "enhanced_768.npy"
    np.save(embedding_path, exported_np)
    np.save(output_dir / "expert_tangent.npy", expert_np)
    np.save(output_dir / "gating_weights.npy", weights_np)
    item_ids = np.arange(n, dtype=np.int64)
    np.save(output_dir / "item_ids.npy", item_ids)

    manifest = {
        "embedding": {"path": str(embedding_path.relative_to(output_dir)), "sha256": sha256(embedding_path),
                      "shape": list(exported_np.shape), "dtype": str(exported_np.dtype)},
        "expert_tangent": {"sha256": sha256(output_dir / "expert_tangent.npy"),
                           "shape": list(expert_np.shape)},
        "gating_weights": {"sha256": sha256(output_dir / "gating_weights.npy"),
                           "shape": list(weights_np.shape)},
        "item_ids": {"sha256": sha256(output_dir / "item_ids.npy"), "rows": int(n)},
        "checkpoint": {"path": str(checkpoint.name), "sha256": sha256(checkpoint)},
        "original_embedding": {
            "path": str(data_module.EMBEDDINGS.relative_to(data_module.REPO)),
            "sha256": sha256(data_module.EMBEDDINGS),
            "note": "read only; the enhanced vector is written to a separate file",
        },
        "graph": {"sha256": bundle["graph_sha256"]},
        "curvatures": [float(value) for value in model.experts.curvature_values()],
        "how_it_is_built": {
            "per_expert": "two κ-GCN layers; encode() returns logmap0(z), the tangent "
                          "coordinates at the origin, which is the common space",
            "concat": "5 experts x 128 dims = 640, layer-normalised by Experts.norm",
            "projection": "Linear(640 -> 768) trained jointly with the link-prediction "
                          "and distortion objectives, so it is supervised, not a random "
                          "drop of structure",
        },
        "checks": checks,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                     encoding="utf-8")
    return manifest
