from __future__ import annotations

import json
from pathlib import Path

import torch

from . import poincare_contract

#: Every level takes the baseline curvature: the state that has to reproduce the frozen start.
BEHAVIOUR_OFF = False
#: Put the curvature on the first two levels too, for the wiring self-check only.  Never trained.
LEVEL12_ON = False

#: The frozen per-item AF3 curvature, the same table #180/#181 froze (mean 1.0, median 0.9147, max 8.2671).
TABLE_RELPATH = Path("results") / "research" / "issue167" / "kappa.json"
BASELINE_CURVATURE = 1.0
KAPPA_FLOOR = 1e-3
ARTANH_EPS = 1e-6
DENOMINATOR_FLOOR = 1e-3

_TABLE: dict | None = None


def repo_root() -> Path:
    """The repository root, found by the marker directories rather than by counting parents."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "stage2_RQ-VAE").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError("could not locate the repository root from the model package")


def load_table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.loads((repo_root() / TABLE_RELPATH).read_text())
    return _TABLE


def kappa_rows(item_ids, table: dict | None = None) -> torch.Tensor:
    """``kappa_i`` for every row, read straight from the frozen table.

    An item the table does not cover takes ``mean_frozen`` -- the table's own blind-spot value -- and the result is floored at the value #180 uses for this
    same table, because the table contains zero-curvature entries and ``2/sqrt(0)`` is infinite.  ``BEHAVIOUR_OFF`` returns the baseline curvature for
    everything, which is the state that must be bit-identical to the frozen start.
    """
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    rows = [BASELINE_CURVATURE if BEHAVIOUR_OFF else float(entries.get(str(int(item)), mean_frozen))
            for item in torch.as_tensor(item_ids).reshape(-1).tolist()]
    return torch.tensor(rows, dtype=torch.float32).clamp_min(KAPPA_FLOOR)


def level_uses_kappa(level: int) -> bool:
    """Which level the geometry acts on: the third, plus the first two when ``LEVEL12_ON`` is set for the self-check."""
    if BEHAVIOUR_OFF:
        return False
    return level == 3 or (LEVEL12_ON and level in (1, 2))


def switches() -> dict:
    return {"behaviour_off": BEHAVIOUR_OFF, "level12_on": LEVEL12_ON}


# --- the level-1 cost under a per-item curvature -------------------------------------------------------------------


def _mobius_add_rows(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a (+) b`` under a curvature that varies per row.

    ``poincare_contract.mobius_add`` takes a scalar curvature (it calls ``float(c)``), and this mechanism needs one curvature per item, so the same
    expression is written here with broadcasting: with ``<.,.>`` the inner product on the last axis,

        a (+) b = ((1 + 2c<a,b> + c|b|^2) a + (1 - c|a|^2) b) / (1 + 2c<a,b> + c^2 |a|^2 |b|^2)
    """
    inner = (a * b).sum(-1, keepdim=True)
    a2 = a.pow(2).sum(-1, keepdim=True)
    b2 = b.pow(2).sum(-1, keepdim=True)
    c = c.unsqueeze(-1) if c.dim() == a.dim() - 1 else c
    numerator = (1.0 + 2.0 * c * inner + c * b2) * a + (1.0 - c * a2) * b
    denominator = 1.0 + 2.0 * c * inner + c.pow(2) * a2 * b2
    # the denominator vanishes only in the boundary limit; clamping it at a fixed floor keeps the quotient finite instead of letting 1/1e-30 blow the
    # whole step up, which is what turned the first smoke's epoch-1 loss into a non-finite value
    return numerator / denominator.clamp_min(DENOMINATOR_FLOOR)


def _into_row_balls(points: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Scale a row down when its curvature demands a smaller ball: ``sqrt(c)|x| <= 1 - eps``."""
    c = c.reshape(-1, 1)
    norm = points.norm(dim=-1, keepdim=True)
    limit = (1.0 - ARTANH_EPS) / torch.sqrt(c)
    scale = torch.where(norm > limit, limit / norm.clamp_min(1e-30), torch.ones_like(norm))
    return points * scale


def curved_squared(residual_h: torch.Tensor, code_balls: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``d_kappa(p, q)^2`` for every (item, code) pair, ``kappa`` one value per row.

    This builds the full ``[B, K]`` matrix over codes.  For the commitment and codebook terms, where each row has its own picked point, use
    :func:`curved_squared_paired` -- feeding the picked points here would build a ``[B, B]`` matrix instead.

    ``residual_h`` is ``[B, D]`` in the ball, ``code_balls`` is ``[K, D]`` in the ball; the returned matrix is ``[B, K]``.  The expression is the frozen
    ``(2/sqrt(c)) artanh(sqrt(c)|(-p) (+) q|)`` with the curvature varying per row.  A row whose curvature shrinks the ball is first scaled into it --
    the same projection the frozen contract applies, applied per row -- and the pair is evaluated by broadcasting, so the 24k-item export never builds a
    ``[B, K, D]`` tensor.
    """
    c = torch.as_tensor(kappa, dtype=residual_h.dtype, device=residual_h.device).reshape(-1, 1)
    p = _into_row_balls(residual_h, c).unsqueeze(1)
    q = code_balls.unsqueeze(0)
    difference = _mobius_add_rows(-p, q, c)
    norm = difference.norm(dim=-1)
    root = torch.sqrt(c)
    argument = (root * norm).clamp(max=1.0 - ARTANH_EPS)
    return _finite_or_raise(((2.0 / root) * torch.atanh(argument)).pow(2), kappa)


def _finite_or_raise(values: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """A non-finite cost must name itself here rather than surface as a dead epoch."""
    if not bool(torch.isfinite(values).all()):
        bad = int((~torch.isfinite(values)).sum().item())
        kappa_range = (float(torch.min(kappa)), float(torch.max(kappa)))
        raise RuntimeError(f"curved cost is not finite on {bad} pairs (kappa range {kappa_range})")
    return values


def curved_squared_paired(points: torch.Tensor, picked: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """``d_kappa^2`` between each row and its own picked point, ``[B]`` -- no cross matrix.

    The commitment and codebook terms compare every item with the single code it chose, so building the full ``[B, B]`` matrix there is both wrong and
    fatal: the codebook initialisation does it on the whole training set, and 24556 squared rows do not fit in any GPU.
    """
    c = torch.as_tensor(kappa, dtype=points.dtype, device=points.device).reshape(-1, 1)
    p = _into_row_balls(points, c)
    q = _into_row_balls(picked, c)
    difference = _mobius_add_rows(-p, q, c)
    norm = difference.norm(dim=-1)
    root = torch.sqrt(c).reshape(-1)
    argument = (root * norm).clamp(max=1.0 - ARTANH_EPS)
    return _finite_or_raise(((2.0 / root) * torch.atanh(argument)).pow(2), kappa)
