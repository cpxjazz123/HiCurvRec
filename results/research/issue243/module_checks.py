"""Per-module verification on real data (issue #243, C12).

For every implemented module: the actual forward output, whether each parameter receives
a gradient (None vs true zero vs non-zero), and whether one AdamW step actually moves the
parameters.  A finite global gradient is not evidence that a particular module learned, so
each module is checked on its own, on 128 real item embeddings.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ISSUE = Path(__file__).resolve().parent
HERE = ISSUE
sys.path.insert(0, str(ISSUE))

import kappa_gcn  # noqa: E402
import manifold  # noqa: E402

REPO = ISSUE.parents[2]
FEATURES = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
GRAPH = REPO / "results" / "research" / "issue242" / "graph" / "ego_csr.npz"
NODES = 128
SEED = 243001


def graph_edges():
    import scipy.sparse as sp
    blob = np.load(GRAPH)
    csr = sp.csr_matrix((blob["data"], blob["indices"], blob["indptr"]), shape=tuple(blob["shape"]))
    sub = csr[:NODES, :NODES].tocoo()
    edges = np.stack([sub.row, sub.col], axis=1)
    return torch.tensor(edges, dtype=torch.long)


def parameter_report(module: torch.nn.Module, loss_value: torch.Tensor, step: bool) -> dict:
    loss_value.backward(retain_graph=True)
    rows = {}
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        if grad is None:
            rows[name] = {"grad": "None", "l2": None, "nonzero": None}
        else:
            rows[name] = {
                "grad": "present",
                "l2": float(grad.norm()),
                "nonzero": int((grad != 0).sum()) > 0,
                "finite": bool(torch.isfinite(grad).all()),
            }
    if step:
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
        before = {name: p.detach().clone() for name, p in module.named_parameters()}
        optimizer.step()
        for name, parameter in module.named_parameters():
            delta = float((parameter.detach() - before[name]).norm())
            rows[name]["adam_delta"] = delta
    return rows


def main() -> int:
    torch.manual_seed(SEED)
    embeddings = np.load(FEATURES)[:NODES].astype(np.float32)
    x = torch.tensor(embeddings)
    edges = graph_edges()
    report = {"nodes": NODES, "edges": int(edges.shape[0]), "modules": {}}

    # ---- manifold operators on real semantic vectors --------------------
    for k in (-3.0, -1.0, 0.0, 1.0, 3.0):
        point = manifold.expmap0(manifold.proju(x, k), k)
        other = manifold.expmap0(manifold.proju(torch.roll(x, 1, 0), k), k)
        distance = manifold.dist(point, other, k)
        zero_rows = int((point.norm(dim=-1) < 1e-9).sum())
        radius = None if k == 0 else 1.0 / math.sqrt(abs(k))
        report["modules"][f"manifold.k={k}"] = {
            "output_norm_mean": float(point.norm(dim=-1).mean()),
            "output_norm_max": float(point.norm(dim=-1).max()),
            "ball_radius": radius,
            "zero_vector_rate": zero_rows / NODES,
            "distance_mean": float(distance.mean()),
            "distance_std": float(distance.std()),
            "distance_finite": bool(torch.isfinite(distance).all()),
        }

    # ---- kappa layer and experts ---------------------------------------
    layer = kappa_gcn.KappaGCNLayer(768, 16, -1.0)
    h = manifold.expmap0(manifold.proju(x, -1.0), -1.0)
    out = layer(h, edges, edges)
    report["modules"]["kappa_gcn_layer"] = {
        "output_shape": list(out.shape),
        "output_norm_mean": float(out.norm(dim=-1).mean()),
        "params": parameter_report(layer, out.pow(2).mean(), step=True),
    }

    for k in (-3.0, -1.0, 0.0, 1.0, 3.0):
        expert = kappa_gcn.KappaExpert(768, 32, 16, k)
        z = expert(x, edges, edges)
        report["modules"][f"expert.k={k}"] = {
            "output_norm_mean": float(z.norm(dim=-1).mean()),
            "output_norm_max": float(z.norm(dim=-1).max()),
            "zero_vector_rate": float((z.norm(dim=-1) < 1e-9).float().mean()),
            "finite": bool(torch.isfinite(z).all()),
            "params": parameter_report(expert, z.pow(2).mean(), step=True),
        }

    (HERE / "MODULE_CHECKS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, entry in report["modules"].items():
        if "params" in entry:
            grads = [row["grad"] for row in entry["params"].values()]
            print(f"{name}: params={len(grads)} none={grads.count('None')} "
                  f"zero={(sum(1 for r in entry['params'].values() if r.get('nonzero') is False))}")
        else:
            print(f"{name}: zero_rate={entry['zero_vector_rate']:.4f} "
                  f"dist_mean={entry['distance_mean']:.4f} std={entry['distance_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
