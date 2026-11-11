"""Relative curvature for the level-1 assignment, issue #185.

The frozen baseline's cost uses a constant curvature, the measured per-item table gains about three percent when used directly, and the
learned per-code table from #184 gains almost nothing. This module measures the item against the code it would join:

    rho_ik = clip(kappa_i / kappa_k, 1e-3, 1e3)

with kappa_i from the frozen table and kappa_k the mean curvature of the items the code received last epoch (detached, refreshed once per
epoch, empty codes keeping their value, a below-floor value becoming 1.0 and counted).

Four states, all wired, none of them trained as a separate arm:

* ``RATIO_OFF``        -- rho = 1, which must reproduce the frozen baseline bit for bit;
* ``NUMERATOR_ONLY``   -- rho = kappa_i, which must reproduce TIGER_struct_mixed_rq bit for bit;
* ``DENOMINATOR_ONLY`` -- rho = 1 / kappa_k;
* default (BOTH_ON)    -- the ratio itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

RATIO_OFF = False
NUMERATOR_ONLY = False
DENOMINATOR_ONLY = False

FLOOR = 1e-3
CLIP = (1e-3, 1e3)


def _ones(text: str) -> bool:
    return text.strip() == "1"


def rho_matrix(item_kappa: torch.Tensor, code_kappa: torch.Tensor) -> torch.Tensor:
    """``clip(kappa_i / kappa_k, 1e-3, 1e3)`` as a ``[items, codes]`` matrix, per the four state switches."""
    items = item_kappa.reshape(-1, 1).to(torch.float64)
    codes = code_kappa.reshape(1, -1).to(torch.float64)
    if RATIO_OFF:
        return torch.ones((items.shape[0], codes.shape[1]), dtype=torch.float64)
    if NUMERATOR_ONLY:
        return items.expand(-1, codes.shape[1]).contiguous()
    if DENOMINATOR_ONLY:
        return (1.0 / codes).expand(items.shape[0], -1).contiguous()
    return (items / codes).clamp(*CLIP)


def to_ball(u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``tanh(sqrt(k)||u||) * (1 - 1e-6) / (sqrt(k)||u||) * u``, strictly inside, zero taking the limit 1."""
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


def level1_cost(latent: torch.Tensor, codes: torch.Tensor, item_kappa: torch.Tensor,
                code_kappa: torch.Tensor) -> torch.Tensor:
    """``C[i, k] = d_{rho_ik}(to_ball(r_i; rho_ik), to_ball(c_k; rho_ik))``, the Euclidean expression in the RATIO_OFF state."""
    if RATIO_OFF:
        return (latent.pow(2).sum(1, keepdim=True) - 2 * latent @ codes.t() + codes.pow(2).sum(1, keepdim=True).t())
    rows, cols, dim = latent.shape[0], codes.shape[0], latent.shape[-1]
    rho = rho_matrix(item_kappa, code_kappa).unsqueeze(-1).to(latent.device)
    p = to_ball(latent.to(torch.float64).unsqueeze(1).expand(rows, cols, dim), rho)
    q = to_ball(codes.to(torch.float64).unsqueeze(0).expand(rows, cols, dim), rho)
    return distance(p, q, rho).to(latent.dtype)


def refresh_codes(codes_kappa: torch.Tensor, item_kappa: torch.Tensor, assigned_codes: torch.Tensor):
    """One epoch's update of ``kappa_k``: the mean curvature of the items each code received, detached."""
    n_codes = codes_kappa.numel()
    sums = torch.zeros(n_codes, dtype=torch.float64)
    seen = torch.zeros(n_codes, dtype=torch.float64)
    if assigned_codes.numel():
        codes = assigned_codes.to(torch.long)
        sums.index_add_(0, codes, item_kappa.to(torch.float64))
        seen.index_add_(0, codes, torch.ones_like(codes, dtype=torch.float64))
    moved = seen > 0
    new_table = codes_kappa.clone()
    with torch.no_grad():
        new_table[moved] = (sums[moved] / seen[moved]).detach()
    new_table = new_table.detach()
    floored = int((new_table < FLOOR).sum())
    new_table = torch.where(new_table < FLOOR, torch.ones_like(new_table), new_table)
    return new_table, {"moved": int(moved.sum()), "empty": int((~moved).sum()), "floored": floored}


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
    return torch.tensor(values, dtype=torch.float64).clamp_min(FLOOR)
