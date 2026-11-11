"""Pair-curvature Poincare distances for the R48 contrast, issue #178.

The contrast step of TIGER_BEHAVIOR_R48 measures ``cdist(p=2)`` between residual anchors and candidates. This module replaces
only that measurement: each pair (i, j) gets its own curvature ``kappa_ij = sqrt(kappa_i * kappa_j)`` read from the frozen
table, both points are projected into that curvature's ball, and the distance is the Poincare metric at that curvature. With
``GEOMETRY_OFF`` the Euclidean cdist is returned untouched, which is what must reproduce R48's behaviour loss bit for bit.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

#: Switch: off returns the Euclidean cdist of R48, bitwise.
GEOMETRY_OFF = False
#: Switch: on reshuffles the behaviour pairs within their own distribution (pairs never trained this way).
BEHAVIOUR_OFF = False

KAPPA_FLOOR = 1e-3
BALL_EPS = 1e-7


def project_to_ball(u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``tanh(sqrt(k)||u||)/(sqrt(k)||u||) * u``, with the zero vector taking the limit 1."""
    root = torch.sqrt(kappa)
    norm = u.norm(dim=-1, keepdim=True)
    # the (1 - eps) keeps the projection strictly inside: tanh saturates to exactly 1 in float64 for a large norm
    scale = torch.where(norm > 0,
                        torch.tanh(root * norm) * (1.0 - 1e-6) / (root * norm).clamp_min(1e-30),
                        torch.ones_like(norm))
    return u * scale


def distance(p: torch.Tensor, q: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``(1/sqrt(k)) arccosh(1 + 2k||p-q||^2 / ((1-k||p||^2)(1-k||q||^2)))`` for broadcastable ``p``, ``q``, ``kappa``."""
    p2 = (p.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - BALL_EPS)
    q2 = (q.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - BALL_EPS)
    diff2 = (p - q).pow(2).sum(-1, keepdim=True)
    argument = 1.0 + 2.0 * kappa * diff2 / ((1.0 - p2) * (1.0 - q2))
    return (torch.acosh(argument.clamp_min(1.0)) / torch.sqrt(kappa)).squeeze(-1)


def pair_kappa(source_kappa: torch.Tensor, candidate_kappa: torch.Tensor) -> torch.Tensor:
    """``sqrt(kappa_i kappa_j)`` with the numerical floor applied and counted."""
    product = source_kappa.reshape(-1, 1) * candidate_kappa.reshape(1, -1)
    kappa = torch.sqrt(product.clamp_min(0.0))
    return kappa.clamp_min(KAPPA_FLOOR)


def behaviour_distances(source: torch.Tensor, candidates: torch.Tensor,
                        source_kappa: torch.Tensor, candidate_kappa: torch.Tensor) -> torch.Tensor:
    """The ``[B, B]`` distance matrix the contrast consumes."""
    if GEOMETRY_OFF:
        return torch.cdist(source, candidates, p=2)
    # the lookup returns a CPU float32 tensor; the residuals may be on a GPU in another dtype
    device, dtype = source.device, source.dtype
    kappa = pair_kappa(source_kappa.to(device=device, dtype=dtype),
                       candidate_kappa.to(device=device, dtype=dtype))
    rows, cols, dim = source.shape[0], candidates.shape[0], source.shape[-1]
    p = project_to_ball(source.unsqueeze(1).expand(rows, cols, dim), kappa.unsqueeze(-1))
    q = project_to_ball(candidates.unsqueeze(0).expand(rows, cols, dim), kappa.unsqueeze(-1))
    return distance(p, q, kappa.unsqueeze(-1))


_TABLE: dict | None = None
TABLE_RELPATH = Path("results") / "research" / "issue167" / "kappa.json"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).is_dir() for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return candidate
    raise FileNotFoundError("No repository root with stage2_RQ-VAE, results and stage1_GeneEmbedding above this module.")


def load_table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.loads((repo_root() / TABLE_RELPATH).read_text())
    return _TABLE


def kappa_of(item_ids: torch.Tensor, table: dict | None = None) -> torch.Tensor:
    """Per-item curvature, the frozen mean for the blind spot, floored and counted by the caller."""
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(item)), mean_frozen)) for item in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float32).clamp_min(KAPPA_FLOOR)
