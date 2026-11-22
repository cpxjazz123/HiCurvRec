"""Per-item geometry, set by the behaviour graph's own curvature.

CLAUDE.md 19h requires a training-only behaviour signal **and** an explicit curve operation
coupled on one connected path, and it names "replacing the Euclidean distance with a
hyperbolic one" as out of scope on its own.  This metric is the coupling:

    w_i   = rank-normalised mean |AF3_ij| over i's neighbours in the canonical behaviour
            graph (CLAUDE.md 89b), computed from train-visible interactions only
    cost_i = (1 - w_i) * |z_i - c|^2  +  w_i * d_{-1}(z_i, c)^2

So item i's geometry is a mixture whose weight is the curvature of its own neighbourhood:
chain-like neighbourhoods (large |AF3|) push the item towards the hyperbolic ball, cohesive
triangle neighbourhoods (small |AF3|) leave it Euclidean.  Removing the behaviour signal
(w_i := const) or removing the curve operation (w_i := 0) both change the assignment, which
is what 19h asks the mechanism to satisfy.

The per-item weights arrive through `set_item_ids`, which the residual stack calls with the
batch's item ids.  The `_distances`/`_indices` call sites stay byte-identical, so the
anti-collapse floor keeps every anchor it had.
"""
from __future__ import annotations

import numpy as np
import torch

NAME = "graphlet"
DESCRIPTION = ("per-item Euclidean/hyperbolic mixture weighted by the item's behaviour-graph "
               "AF3 curvature (issue 238)")

#: Measured curvature of the item graph (#236): the ball is curvature -1.
KAPPA = -1.0
RADIUS_CAP = 0.9
EPS = 1e-6

_KAPPA_TABLE: torch.Tensor | None = None
_ITEM_IDS: torch.Tensor | None = None


def build_weights(edges_path, graph_path, n_items: int = 24587):
    """Rebuild the per-item mixing weights from the canonical behaviour graph (89b).

    The table is a derived artifact -- `stage2_RQ-VAE/**/*.npy` is gitignored -- so the tree
    regenerates it instead of shipping a binary.  w_i is the rank-normalised mean |AF3_ij| over
    i's neighbours: large values mean a chain-like, strongly curved neighbourhood.
    """
    import pandas as pd

    edges = pd.read_parquet(edges_path)
    graph = np.load(graph_path, allow_pickle=True)
    af3 = np.abs(np.asarray(graph["af3"], dtype=np.float64))
    if af3.shape[0] != len(edges):
        raise ValueError("the graph's AF3 array does not align with its edge table")
    src = edges.iloc[:, 0].to_numpy()
    dst = edges.iloc[:, 1].to_numpy()
    total = np.zeros(n_items)
    count = np.zeros(n_items)
    np.add.at(total, src, af3)
    np.add.at(total, dst, af3)
    np.add.at(count, src, 1.0)
    np.add.at(count, dst, 1.0)
    mean_abs = np.divide(total, np.maximum(count, 1.0), out=np.zeros(n_items), where=count > 0)
    order = np.argsort(np.argsort(mean_abs))
    return (order / (n_items - 1.0)).astype(np.float32)


def set_kappa_table(weights) -> None:
    """Install the per-item mixing weights (a 1-D array/tensor over item ids)."""
    global _KAPPA_TABLE
    if weights is None:
        _KAPPA_TABLE = None
        return
    _KAPPA_TABLE = torch.as_tensor(np.asarray(weights), dtype=torch.float32)


def set_item_ids(item_ids) -> None:
    """The ids of the rows in the next `distances` call; `None` means one row per id."""
    global _ITEM_IDS
    if item_ids is None:
        _ITEM_IDS = None
        return
    _ITEM_IDS = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)


def _weights(rows: int, device) -> torch.Tensor:
    if _KAPPA_TABLE is None:
        return torch.zeros(rows, device=device)
    table = _KAPPA_TABLE.to(device)
    if _ITEM_IDS is None:
        if rows == table.shape[0]:
            return table
        return torch.zeros(rows, device=device)
    ids = _ITEM_IDS.to(device)
    if ids.shape[0] != rows:
        return torch.zeros(rows, device=device)
    return table[ids]


def _to_ball(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return RADIUS_CAP * torch.tanh(norm / tau) * x / norm


def euclidean(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return (
        latent.pow(2).sum(1, keepdim=True)
        - 2 * latent @ codebook.t()
        + codebook.pow(2).sum(1, keepdim=True).t()
    )


def hyperbolic(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Squared geodesic distance on the curvature -1 ball."""
    tau = codebook.norm(dim=-1).mean().clamp_min(EPS)
    u = _to_ball(latent, tau)
    v = _to_ball(codebook, tau)
    root = abs(KAPPA) ** 0.5
    u2 = (u * u).sum(-1, keepdim=True).clamp(max=RADIUS_CAP ** 2)
    v2 = (v * v).sum(-1).clamp(max=RADIUS_CAP ** 2)
    diff2 = (u.pow(2).sum(-1, keepdim=True) - 2 * u @ v.t()
             + v.pow(2).sum(-1, keepdim=True).t()).clamp_min(0.0)
    inner = 1.0 + 2.0 * diff2 / ((1.0 - u2) * (1.0 - v2).unsqueeze(0))
    geodesic = torch.acosh(inner.clamp_min(1.0 + EPS)) / root
    return geodesic.pow(2)


def distances(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    weight = _weights(latent.shape[0], latent.device).unsqueeze(1)
    return (1.0 - weight) * euclidean(latent, codebook) + weight * hyperbolic(latent, codebook)
