"""Issue #209: one radial recalibration of the hyperbolic half before the codes are chosen.

The quantiser's geometry, its codebooks, its Sinkhorn and its readout are F0's.  What changes is where an item sits in the ball at the moment of selection: its
hyperbolic half is pushed out or pulled in by ``r_i = kappa_i^{-1/2}``, and the picked code is mapped back before it reaches the readout, so the decoder keeps the
F0 scale and only the selection sees the structure.

``phi_r(p) = tanh(r * atanh(min(|p|, 1 - 1e-6))) * p / |p|`` keeps the direction and moves the radius.  ``r = 1`` is the identity, which is what ``BEHAVIOUR_OFF``
sets for every item.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

#: Every item keeps ``r = 1`` -- the state that has to be bit-identical to F0.
BEHAVIOUR_OFF = False
#: Apply the transform on the way in but not the inverse on the way out, for the wiring self-check only.  Never trained.
INVERSE_SKIPPED = False

#: The frozen per-item AF3 curvature, the same table #180/#181/#199/#207 froze (mean 1.0, median 0.9147, max 8.2671).
TABLE_RELPATH = Path("results") / "research" / "issue167" / "kappa.json"
KAPPA_FLOOR = 1e-3
#: Keeps ``atanh`` and the division by ``|p|`` away from their singular points; it is a representation guard, not a tuned quantity.
RADIUS_EPS = 1e-6

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
    """``kappa_i`` from the frozen table, the blind-spot mean where it is silent, floored as #180 floors it."""
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    rows = [float(entries.get(str(int(item)), mean_frozen)) for item in torch.as_tensor(item_ids).reshape(-1).tolist()]
    return torch.tensor(rows, dtype=torch.float32).clamp_min(KAPPA_FLOOR)


def radius_rows(item_ids, table: dict | None = None) -> torch.Tensor:
    """``r_i = kappa_i^{-1/2}``; ``1`` for every row when ``BEHAVIOUR_OFF``, which is the identity map."""
    if BEHAVIOUR_OFF:
        return torch.ones(torch.as_tensor(item_ids).reshape(-1).numel(), dtype=torch.float32)
    return kappa_rows(item_ids, table).rsqrt()


def phi(radius: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``tanh(r * atanh(clamp(|p|))) * p / |p|``: same direction, radius moved, differentiable through the clamped ``atanh``.

    The radius is transformed in double precision and cast back: in float32 the composition of ``atanh`` and ``tanh`` loses enough precision at moderate radii to
    tilt a direction by 3e-4 radians, and the map is supposed to leave the direction exactly alone.  The clamp is applied to the argument rather than to the
    radius, so ``phi_{1/r}`` composed with ``phi_r`` is the identity whenever the clamp is not active.
    """
    dtype = points.dtype
    r = radius.reshape(-1, 1).to(dtype=torch.float64, device=points.device)
    p64 = points.to(torch.float64)
    norm = p64.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(1e-30)
    argument = norm.clamp(min=0.0, max=1.0 - RADIUS_EPS)
    scaled = torch.tanh(r * torch.atanh(argument))
    return (p64 / safe * scaled).to(dtype)


def phi_inverse(radius: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``phi`` with ``1/r``: puts a transformed point back on F0's scale."""
    return phi(1.0 / radius.reshape(-1).clamp_min(1e-6), points)


def switches() -> dict:
    return {"behaviour_off": BEHAVIOUR_OFF, "inverse_skipped": INVERSE_SKIPPED}
