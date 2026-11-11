"""Per-code curvature for the level-1 assignment, issue #184.

The baseline's product geometry uses one constant curvature, and the observed per-item table has never reached the quantiser. This
module makes the curvature a property of the **code** and refreshes it from what the assignment itself decided:

    kappa_k(0) = 1.0,   kappa_k(t+1) = mean{ kappa_i : item i was assigned to code k at epoch t }

with no gradient, codes that received nothing keeping their previous value, and a below-floor value replaced by 1.0 and counted.
"""
from __future__ import annotations

import torch

#: Switch states of the issue: FROZEN keeps kappa_k == 1 (the baseline), BEHAVIOUR_OFF permutes the table, LEARNED is the arm.
KAPPA_FROZEN = False
BEHAVIOUR_OFF = False
FLOOR = 1e-3


def initial(n_codes: int, dtype=torch.float64) -> torch.Tensor:
    return torch.ones(n_codes, dtype=dtype)


def update(kappa_codes: torch.Tensor, assigned_item_kappa: torch.Tensor, assigned_codes: torch.Tensor,
           counts: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One epoch's refresh: each code takes the mean curvature of the items it received.

    ``assigned_item_kappa`` and ``assigned_codes`` are per-assigned-item; codes with no assignments keep their old value. Returns the
    new table and a record of what happened (how many codes moved, how many stayed empty, how many hit the floor).
    """
    if KAPPA_FROZEN:
        return kappa_codes.clone(), {"frozen": True, "moved": 0, "empty": int(kappa_codes.numel()), "floored": 0}
    n_codes = kappa_codes.numel()
    sums = torch.zeros(n_codes, dtype=torch.float64)
    seen = torch.zeros(n_codes, dtype=torch.float64)
    if assigned_codes.numel():
        codes = assigned_codes.to(torch.long)
        sums.index_add_(0, codes, assigned_item_kappa.to(torch.float64))
        seen.index_add_(0, codes, torch.ones_like(codes, dtype=torch.float64))
    moved = seen > 0
    # the refresh is a statistic of the assignment, never a differentiable path: detach here so no gradient can reach the encoder
    new_table = kappa_codes.clone()
    with torch.no_grad():
        new_table[moved] = (sums[moved] / seen[moved]).detach()
    new_table = new_table.detach()
    if BEHAVIOUR_OFF and n_codes > 1:
        # the same distribution, assigned to the wrong codes: the arm's negative control
        permutation = torch.randperm(n_codes, generator=torch.Generator().manual_seed(0))
        new_table = new_table[permutation]
    floored = int(((new_table < FLOOR)).sum())
    new_table = torch.where(new_table < FLOOR, torch.ones_like(new_table), new_table)
    record = {"frozen": False, "moved": int(moved.sum()), "empty": int((~moved).sum()), "floored": floored}
    return new_table, record


def to_ball(u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``tanh(sqrt(k)||u||) * (1 - 1e-6) / (sqrt(k)||u||) * u``, strictly inside every ball, zero taking the limit 1."""
    root = torch.sqrt(kappa)
    norm = u.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > 0, torch.tanh(root * norm) * (1.0 - 1e-6) / (root * norm).clamp_min(1e-30),
                        torch.ones_like(norm))
    return u * scale


def distance(p: torch.Tensor, q: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """The asinh form of the Poincare distance, finite at coincident points."""
    kp = (p.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - 1e-7)
    kq = (q.pow(2).sum(-1, keepdim=True) * kappa).clamp(max=1.0 - 1e-7)
    diff2 = (p - q).pow(2).sum(-1, keepdim=True)
    eps = torch.finfo(p.dtype).eps * torch.maximum(
        torch.maximum(p.pow(2).sum(-1, keepdim=True), q.pow(2).sum(-1, keepdim=True)), torch.ones_like(diff2))
    denominator = torch.sqrt((1.0 - kp).clamp_min(eps) * (1.0 - kq).clamp_min(eps))
    return ((2.0 / torch.sqrt(kappa)) * torch.asinh(torch.sqrt(kappa) * torch.sqrt(diff2 + eps) / denominator)).squeeze(-1)


def level1_cost(latent: torch.Tensor, codes: torch.Tensor, kappa_codes: torch.Tensor) -> torch.Tensor:
    """``C[i, k] = d_{kappa_k}(to_ball(r_i; kappa_k), to_ball(c_k; kappa_k))`` -- the code's curvature, not the item's."""
    if KAPPA_FROZEN:
        return (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t())
    rows, cols, dim = latent.shape[0], codes.shape[0], latent.shape[-1]
    kappa = kappa_codes.to(latent.device, torch.float64).reshape(1, cols, 1)
    p = to_ball(latent.to(torch.float64).unsqueeze(1).expand(rows, cols, dim), kappa)
    q = to_ball(codes.to(torch.float64).unsqueeze(0).expand(rows, cols, dim), kappa)
    return distance(p, q, kappa).to(latent.dtype)


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
    """The frozen per-item curvature, the fill for the blind spot, floored at the issue's 1e-3."""
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    values = [float(entries.get(str(int(i)), mean_frozen)) for i in item_ids.detach().cpu().tolist()]
    return torch.tensor(values, dtype=torch.float64).clamp_min(FLOOR)
