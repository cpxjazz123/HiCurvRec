"""Representation-level curvature for issue #183.

Instead of changing which distance a call site uses, this arm changes the representation itself: every item's encoder output is
pushed onto its own curvature's ball by ``exp_0^k``, and quantisation, residuals and the contrast then all work on that mapped
vector. Three states:

* ``GEOMETRY_OFF`` -- no map and Euclidean distances, which must reproduce R48 bit for bit;
* ``EMBED_ONLY`` -- the map only, distances stay Euclidean (a cheap wiring check);
* default (BOTH_ON) -- the map and the curvature distance.
"""
from __future__ import annotations

import torch

GEOMETRY_OFF = False
EMBED_ONLY = False


def expmap0(z: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``tanh(sqrt(k)||z||)/(sqrt(k)||z||) * z``, the zero vector taking the limit 1."""
    if GEOMETRY_OFF:
        return z
    root = torch.sqrt(kappa)
    norm = z.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > 0,
                        torch.tanh(root * norm) * STRICT_SHRINK / (root * norm).clamp_min(1e-30),
                        torch.ones_like(norm))
    return z * scale


#: The map shrinks by this factor so that ``kappa|p|^2`` stays strictly below one; the inverse divides it back out, which is what
#: makes the round trip exact rather than merely close.
STRICT_SHRINK = 1.0 - 1e-6


def logmap0(p: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The exact inverse of :func:`expmap0`.

    ``expmap0`` sends ``z`` to ``z * tanh(a) * shrink / a`` with ``a = sqrt(k)||z||``, so ``||p|| = tanh(a) * shrink / sqrt(k)``
    and ``a = atanh(sqrt(k)||p|| / shrink)``. Returning ``p * a / (sqrt(k)||p||)`` undoes it exactly.
    """
    root = torch.sqrt(kappa)
    norm = p.norm(dim=-1, keepdim=True)
    arg = (root * norm / STRICT_SHRINK).clamp(max=1.0 - 1e-12)
    a = torch.atanh(arg)
    return p * (a / (root * norm).clamp_min(1e-30))


def distance(p: torch.Tensor, q: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The asinh form of the Poincare distance, finite at coincident points."""
    kp = (p.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - 1e-7)
    kq = (q.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - 1e-7)
    diff2 = (p - q).pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(p.dtype).eps * torch.maximum(
        torch.maximum(p.pow(2).sum(-1, keepdim=True), q.pow(2).sum(-1, keepdim=True)), torch.ones_like(diff2))
    denominator = torch.sqrt((1.0 - kp).clamp_min(eps) * (1.0 - kq).clamp_min(eps))
    return ((2.0 / torch.sqrt(kappa)) * torch.asinh(torch.sqrt(kappa) * torch.sqrt(diff2 + eps) / denominator)).squeeze(-1)


def pair_kappa(k_i: torch.Tensor, k_j: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((k_i.reshape(-1, 1) * k_j.reshape(1, -1)).clamp_min(0.0)).clamp_min(1e-3)


def behaviour_distances(source, candidates, source_kappa, candidate_kappa) -> torch.Tensor:
    """The contrast's distance matrix: map each side, then measure at the pair's own curvature."""
    if GEOMETRY_OFF or EMBED_ONLY:
        return torch.cdist(source, candidates, p=2)
    dtype, device = source.dtype, source.device
    kappa = pair_kappa(source_kappa.to(device, torch.float64), candidate_kappa.to(device, torch.float64))
    rows, cols, dim = source.shape[0], candidates.shape[0], source.shape[-1]
    # per-row curvature for the map: a row's own kappa against each candidate's, which is the pair curvature
    p = expmap0(source.to(torch.float64).unsqueeze(1).expand(rows, cols, dim), kappa.unsqueeze(-1))
    q = expmap0(candidates.to(torch.float64).unsqueeze(0).expand(rows, cols, dim), kappa.unsqueeze(-1))
    return distance(p, q, kappa.unsqueeze(-1)).to(dtype)


_TABLE = None
from pathlib import Path as _Path
TABLE_RELPATH = _Path("results") / "research" / "issue167" / "kappa.json"


def repo_root(start=None) -> _Path:
    here = (start or _Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / m).is_dir() for m in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return candidate
    raise FileNotFoundError("no repository root above this module")


def load_table() -> dict:
    global _TABLE
    if _TABLE is None:
        import json
        _TABLE = json.loads((repo_root() / TABLE_RELPATH).read_text())
    return _TABLE


def kappa_of(item_ids: torch.Tensor, table: dict | None = None) -> torch.Tensor:
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(i)), mean_frozen)) for i in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float32).clamp_min(1e-3)
