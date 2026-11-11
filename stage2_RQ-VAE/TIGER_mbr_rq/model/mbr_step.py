"""Moebius residual steps at the item's own curvature, issue #186.

``struct_mixed`` picks the nearest code with the hyperbolic metric but then subtracts the chosen codeword as if the space were flat, so
levels two and three keep splitting in the wrong geometry. This module replaces only the subtraction:

    x (-)_k y = [(1 + 2k<x,y> + k||y||^2) y - x] / [1 + 2k<x,y> + k^2 ||y||^2 ||x||^2]

with ``k = 0`` giving ``y - x`` exactly, which is the baseline's flat step.
"""
from __future__ import annotations

import torch

#: Switch states: STEP_OFF makes the subtraction flat (kappa = 0), METRIC_OFF leaves the level-1 metric at one, BOTH_ON is the arm.
STEP_OFF = False
METRIC_OFF = False

SHRINK = 1.0 - 1e-6


def mobius_sub(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``x (-)_kappa y``, i.e. ``mobius_add(-x, y, kappa)``, with an epsilon floor and a pull-back inside the ball.

    The frozen contract writes ``mobius_add(a, b, c) = [(1 + 2c<a,b> + c|b|^2) a + (1 - c|a|^2) b] / [1 + 2c<a,b> + c^2|a|^2|b|^2]``, so the
    subtraction carries two numerator terms, not one: with ``a = -x`` the first term flips sign and the second supplies ``y``. At
    ``kappa = 0`` that collapses to ``y - x``.
    """
    if STEP_OFF:
        return y - x
    k = kappa.to(x.device, x.dtype).reshape(-1, 1) if kappa.dim() == 1 else kappa.to(x.device, x.dtype)
    xy = (x * y).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    x2 = x.pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(x.dtype).eps * torch.maximum(torch.maximum(x2, y2), torch.ones_like(x2))
    cross = -xy                                          # <a, b> with a = -x
    numerator = -(1.0 + 2.0 * k * cross + k * y2) * x + (1.0 - k * x2) * y
    denominator = (1.0 + 2.0 * k * cross + k.pow(2) * y2 * x2).clamp_min(eps)
    out = numerator / denominator
    limit = (k * out.pow(2).sum(-1, keepdim=True)).clamp_min(0.0)
    over = limit >= 1.0
    if bool(over.any()):
        scale = torch.sqrt((1.0 - 1e-6) / limit.clamp_min(1e-30))
        out = torch.where(over, out * scale, out)
    return out


def shrink_events(x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor) -> int:
    """How many rows the pull-back had to touch, for the smoke's record."""
    if STEP_OFF:
        return 0
    k = kappa.reshape(-1, 1) if kappa.dim() == 1 else kappa
    xy = (x * y).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    x2 = x.pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(x.dtype).eps * torch.maximum(torch.maximum(x2, y2), torch.ones_like(x2))
    numerator = (1.0 + 2.0 * k * xy + k * y2) * y - x
    denominator = (1.0 + 2.0 * k * xy + k.pow(2) * y2 * x2).clamp_min(eps)
    out = numerator / denominator
    return int(((k * out.pow(2).sum(-1, keepdim=True)) >= 1.0).sum())


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
    """The frozen per-item curvature, mean-filled for the blind spot and floored."""
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(i)), mean_frozen)) for i in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float64).clamp_min(1e-3)
