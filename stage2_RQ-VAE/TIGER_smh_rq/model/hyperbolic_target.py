"""Curvature-conditioned target distance for struct_mixed's triple term, issue #190.

struct_mixed's structure term measures each triple pair with ``mixed_sq``; this arm changes only that measurement, to the pair-curvature
Poincare distance

    d_k(a, b) = (2/sqrt(k)) asinh( sqrt(k) |a-b|_eps / sqrt((1 - k|a|^2)(1 - k|b|^2)) ),   k = sqrt(k_a k_b)

with ``k`` frozen (detached) and taken from the per-item table. With ``GEOMETRY_OFF`` the caller keeps ``mixed_sq`` and this module is not
consulted, which is what makes that state bitwise identical to struct_mixed.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

#: Switch: off means the caller keeps struct_mixed's mixed_sq, so the term is bitwise that arm's.
GEOMETRY_OFF = False
BOUNDARY = 1.0 - 1e-7


def pair_kappa(k_a: torch.Tensor, k_b: torch.Tensor) -> torch.Tensor:
    """The geometric mean of the two items' curvatures, floored."""
    return torch.sqrt((k_a * k_b).clamp_min(0.0)).clamp_min(1e-3)


def distance(a: torch.Tensor, b: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The asinh form of the Poincare distance, finite at coincident points."""
    k = kappa.reshape(-1) if kappa.dim() > 1 else kappa
    k = k.to(a.dtype)
    a2 = (a.pow(2).sum(-1) * k).clamp(max=BOUNDARY)
    b2 = (b.pow(2).sum(-1) * k).clamp(max=BOUNDARY)
    diff2 = (a - b).pow(2).sum(-1)
    eps = torch.finfo(a.dtype).eps * torch.maximum(a.pow(2).sum(-1), b.pow(2).sum(-1)).clamp_min(1.0)
    denominator = torch.sqrt((1.0 - a2).clamp_min(eps) * (1.0 - b2).clamp_min(eps))
    return (2.0 / torch.sqrt(k)) * torch.asinh(torch.sqrt(k) * torch.sqrt(diff2 + eps) / denominator)


def squared(a: torch.Tensor, b: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``d_kappa(a, b) ** 2``, the same squared role mixed_sq plays in the stress."""
    return distance(a, b, kappa).pow(2)


_TABLE = None
TABLE_RELPATH = Path("results") / "research" / "issue167" / "kappa.json"


def repo_root(start=None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / m).is_dir() for m in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return candidate
    raise FileNotFoundError("no repository root above this module")


def load_table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.loads((repo_root() / TABLE_RELPATH).read_text())
    return _TABLE


def kappa_of(item_ids: torch.Tensor, table: dict | None = None) -> torch.Tensor:
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(i)), mean_frozen)) for i in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float64).clamp_min(1e-3)
