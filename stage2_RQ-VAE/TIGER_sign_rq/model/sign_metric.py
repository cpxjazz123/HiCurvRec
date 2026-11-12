"""Issue #197: the *sign* of the in-group minus out-of-group AF3 weight decides the level-1 metric, twice over.

Every structural quantity this line has tried so far entered the pipeline as a continuous magnitude -- a learned curvature per code (#184), a radius
dispersion (#187), a Moebius residual step (#186), a ratio (#185), an affinity kernel (#194).  The quantity the main line named and never used as a
*decision* is the sign of ``Delta_i = w_in(i) - w_out(i)``: how much heavier a item's ties are inside its triangle-closed group than across groups.

The mechanism is two regimes and nothing else:

    Delta_i > 0  =>  kappa_i = the item's frozen AF3 curvature
    Delta_i <= 0 =>  kappa_i = 1  (the frozen baseline's own value)

so structure itself decides whether it is allowed to rewrite the geometry, and no threshold, scale, clip or temperature is introduced: the group is a
property of the graph (a connected component of the triangle-closed subgraph, an integer object) and the comparison is a sign.

``SIGN_OFF`` sets every item to ``kappa = 1``, which is the state that has to reproduce the baseline; ``SIGN_FLIP`` swaps the two regimes and exists only
so the wiring can be checked without training a closed arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import poincare_contract

#: Every item keeps curvature 1 -- the frozen baseline's geometry.
SIGN_OFF = False
#: The two regimes swapped, for the wiring self-check only.  Never trained.
SIGN_FLIP = False

#: The frozen per-item AF3 curvature, the same table #180/#181 froze (mean 1.0, median 0.9147, max 8.2671).
TABLE_RELPATH = Path("results") / "research" / "issue167" / "kappa.json"
BASELINE_CURVATURE = 1.0
ARTANH_EPS = 1e-6
#: Floor on the Moebius denominator.  The exact expression reaches zero only on the boundary of the ball, and a boundary pair is already outside the
#: metric's domain; a fixed floor keeps the arithmetic finite without moving any interior value (it is never active there).
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


# --- the group and the two means ----------------------------------------------------------------------------------


def common_neighbours(edges) -> dict:
    """``m_ij`` per observed edge: the number of shared neighbours, the triangle count through that edge."""
    adjacency: dict[int, set[int]] = {}
    for i, j in edges:
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))
    return {(int(i), int(j)): len(adjacency.get(int(i), set()) & adjacency.get(int(j), set())) for i, j in edges}


def closed_subgraph(edges) -> list:
    """The edges that lie in at least one triangle -- ``m_ij >= 1``, an integer property, not a threshold."""
    shared = common_neighbours(edges)
    return [(i, j) for (i, j) in edges if shared[(int(i), int(j))] >= 1]


def components(edges) -> dict[int, int]:
    """Connected components of the triangle-closed subgraph, as ``node -> component id`` (smallest node first)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {node: find(node) for node in list(parent)}


def af3_weights(edges) -> dict:
    """``W_ij = 1 + clip((AF3 - q05)/(q95 - q05), 0, 1)`` with the corrected triangle count ``3 m_ij``.

    The clipping quantiles are the edge set's own, exactly as the frozen teacher computes them; the expression is #193's repair, and it is applied here
    because a weight built on the constant-3 defect would put the wrong mass on the in-group and out-group means alike.
    """
    edges = [(int(i), int(j)) for i, j in edges]
    degrees: dict[int, int] = {}
    for i, j in edges:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1
    shared = common_neighbours(edges)
    values = {(i, j): 4.0 - float(degrees[i]) - float(degrees[j]) + 3.0 * shared[(i, j)] for i, j in edges}
    if not values:
        return {}
    ordered = sorted(values.values())
    q05 = ordered[int(0.05 * (len(ordered) - 1))]
    q95 = ordered[int(0.95 * (len(ordered) - 1))]
    span = max(q95 - q05, 1e-12)
    return {edge: 1.0 + min(max((value - q05) / span, 0.0), 1.0) for edge, value in values.items()}


def delta_of(edges, weights: dict, membership: dict[int, int]) -> dict[int, float | None]:
    """``Delta_i = w_in(i) - w_out(i)``; ``None`` when a side is empty (``DATA_UNAVAILABLE``, excluded from the decision)."""
    inside: dict[int, list[float]] = {}
    outside: dict[int, list[float]] = {}
    for (i, j), w in weights.items():
        group_i = membership.get(i)
        same = group_i is not None and membership.get(j) == group_i
        (inside if same else outside).setdefault(i, []).append(w)
        (inside if same else outside).setdefault(j, []).append(w)
    out: dict[int, float | None] = {}
    for node in set(inside) | set(outside):
        a = inside.get(node, [])
        b = outside.get(node, [])
        out[node] = None if not a or not b else float(sum(a) / len(a) - sum(b) / len(b))
    return out


def regime_kappa(item_ids, delta: dict[int, float | None], table: dict | None = None) -> torch.Tensor:
    """``kappa_i``: the frozen AF3 value where ``Delta_i > 0``, and 1 -- the baseline's own value -- otherwise."""
    table = load_table() if table is None else table
    entries, mean_frozen = table["kappa"], float(table["mean_frozen"])
    rows = []
    for item in torch.as_tensor(item_ids).reshape(-1).tolist():
        if SIGN_OFF:
            rows.append(BASELINE_CURVATURE)
            continue
        value = delta.get(int(item))
        enabled = bool(value is not None and value > 0.0)
        if SIGN_FLIP:
            enabled = not enabled
        rows.append(float(entries.get(str(int(item)), mean_frozen)) if enabled else BASELINE_CURVATURE)
    return torch.tensor(rows, dtype=torch.float32)


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
