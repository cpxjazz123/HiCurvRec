"""Track E (issue #84): keep L1 winner identities by constraining the deep update.

The margin of item ``i`` against its nearest competitor is

    m_ij(theta, c) = d(z_i(theta), c_j) - d(z_i(theta), c_a),   a = argmin_k d(z_i, c_k)

and a first-order step from ``theta`` to ``theta + dtheta`` moves it to

    m_ij + <grad_theta m_ij, dtheta> + <grad_c m_ij, dc> + O(||dtheta||^2).

The candidate projects the deep-behaviour update onto the half-spaces
``<grad_theta m_ij, dtheta> >= rhs_ij`` for the items that would otherwise cross, where
``rhs_ij = delta_i - m_ij - <grad_c m_ij, dc>``.

Three things this module refuses to do, because each would turn the guarantee into a slogan:

1. it never claims a constraint on the raw gradient is a constraint on the parameter update --
   AdamW applies moments, weight decay and clipping, so the realised step is measured separately;
2. it never trusts the linearisation -- after every update it recomputes the true margins through
   the quantizer and reports the violations, so a step that crosses a boundary anyway is visible;
3. it never substitutes a smaller item subset for the declared one and calls it the full
   guarantee: the guarded set is reported next to the total.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_replay_engine as G  # noqa: E402

DEFAULT_DELTA = 0.0


# --------------------------------------------------------------------------------------
# geometry: the exact margins
# --------------------------------------------------------------------------------------
def exact_margins(model, embeddings: torch.Tensor) -> dict[str, Any]:
    """The quantizer's own margins, winners and runner-ups. No linearisation here."""
    was_training = model.training
    model.eval()
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        trace = model.encoder(embeddings)
        if isinstance(trace, tuple):
            trace = trace[0]
        distance = layer._distances(trace.reshape(-1, layer.dim))
        ordered, index = distance.sort(dim=1)
        margin = ordered[:, 1] - ordered[:, 0]
        winner, runner_up = index[:, 0], index[:, 1]
    model.train(was_training)
    return {"margin": margin, "winner": winner, "runner_up": runner_up,
            "winner_distance": ordered[:, 0], "items": int(margin.numel())}


def margin_gradients(model, embeddings: torch.Tensor, positions: torch.Tensor
                     ) -> dict[str, Any]:
    """grad_theta of each selected margin, in the parameters the deep update actually touches.

    The margin depends on the codebook as well as the encoder, so the codebook contribution is
    separated out rather than folded into the encoder half-space.
    """
    layer = model.rq.vq_layers[0]
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.startswith("encoder.")]
    encoder_parameters = [parameter for _, parameter in named]
    trace = model.encoder(embeddings)
    if isinstance(trace, tuple):
        trace = trace[0]
    distance = layer._distances(trace.reshape(-1, layer.dim))
    ordered, index = distance.sort(dim=1)
    winner, runner_up = index[:, 0], index[:, 1]
    # d(margin)/d(theta) = -(d(winner)/d(theta) - d(runner_up)/d(theta)); each margin is
    # differentiated one at a time so the result is a per-item vector, not a sum.
    gradients = []
    for position in positions.tolist():
        selected = ordered[position, 0] - ordered[position, 1]
        gradients.append(torch.autograd.grad(selected, encoder_parameters, retain_graph=True,
                                              allow_unused=True))
    codebook = layer.get_code_embs()
    code_terms = []
    for position in positions.tolist():
        selected = ordered[position, 0] - ordered[position, 1]
        code_terms.append(float(torch.autograd.grad(selected, [codebook], retain_graph=True,
                                                    allow_unused=True)[0].norm()))
    return {
        "encoder_names": [name for name, _ in named],
        "gradients": gradients,
        "code_terms": code_terms,
        "margin": ordered[positions, 1] - ordered[positions, 0],
        "winner": winner[positions],
        "runner_up": runner_up[positions],
    }


# --------------------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------------------
def project_halfspace(vector: torch.Tensor, gradient: torch.Tensor,
                      bound: float) -> torch.Tensor:
    """Exact projection of one vector onto ``<g, x> >= bound``.

    This closed form is the independent oracle the iterative solver is checked against: any
    disagreement is a bug in the solver, not in the reference.
    """
    squared = float((gradient * gradient).sum())
    if squared <= 0.0:
        return vector
    # The constraint is <g, x> >= bound, so a violation is `bound - <g, x>` when that is positive.
    # Moving the point ALONG +g by t changes <g, x> by t*||g||^2, so t = violation / ||g||^2 puts
    # it exactly on the boundary -- which is the nearest feasible point.
    violation = bound - float((gradient * vector).sum())
    if violation <= 0.0:
        return vector
    return vector + (violation / squared) * gradient


def dykstra_project(vector: torch.Tensor, constraints: list[tuple[torch.Tensor, float]],
                    *, iterations: int = 500, tolerance: float = 1e-12
                    ) -> tuple[torch.Tensor, int]:
    """Project onto an intersection of half-spaces by Dykstra-corrected cyclic projection.

    Dykstra rather than plain alternation: plain iteration lands somewhere in the intersection but
    not necessarily at the point nearest the original vector, which would make the "minimises
    deviation" half of the claim false.
    """
    corrections = [torch.zeros_like(vector) for _ in constraints]
    current = vector.clone()
    used = 0
    for used in range(1, iterations + 1):
        for index, (gradient, bound) in enumerate(constraints):
            candidate = current + corrections[index]
            projected = project_halfspace(candidate, gradient, bound)
            corrections[index] = candidate - projected
            current = projected
        if all(float((gradient * current).sum()) >= bound - tolerance
               for gradient, bound in constraints):
            break
    return current, used


def active_set(margins: torch.Tensor, delta: float) -> torch.Tensor:
    """Items whose margin is already inside the band the guarantee has to cover."""
    return torch.nonzero(margins <= delta, as_tuple=False).reshape(-1)


def build_constraints(gradients: list[torch.Tensor], margins: torch.Tensor,
                      code_terms: list[float], delta: float) -> list[tuple[torch.Tensor, float]]:
    """One half-space per at-risk item: ``<g, dtheta> >= delta - m - <g_c, dc>``."""
    return [(gradient, float(delta - float(margin) - code_term))
            for gradient, margin, code_term in zip(gradients, margins, code_terms)]


# --------------------------------------------------------------------------------------
# the check the linearisation cannot supply
# --------------------------------------------------------------------------------------
def realised_violations(before: dict[str, Any], after: dict[str, Any],
                        positions: torch.Tensor, delta: float) -> dict[str, Any]:
    """Recompute the true margins after the update and count the ones that still crossed."""
    if positions.numel() == 0:
        return {"items_guarded": 0, "violated": 0, "violation_rate": 0.0,
                "definition": "no item was inside the guarded band at this step"}
    before_margin = before["margin"][positions]
    after_margin = after["margin"][positions]
    violated = after_margin < delta
    return {
        "items_guarded": int(positions.numel()),
        "violated": int(violated.sum()),
        "violation_rate": float(violated.float().mean()),
        "margin_before_mean": float(before_margin.mean()),
        "margin_after_mean": float(after_margin.mean()),
        "margin_after_min": float(after_margin.min()),
        "crossed_any": int((after_margin < before_margin).sum()),
        "definition": "a violation is an item whose margin is below delta AFTER the real update, "
                      "measured through the quantizer rather than through the linearisation",
    }


def relative_deviation(proposed: torch.Tensor, projected: torch.Tensor) -> float:
    """How much the projected update deviates from the update the arm wanted to apply."""
    squared = float(((proposed - projected) ** 2).sum())
    total = float((proposed ** 2).sum())
    return squared / total if total > 0 else 0.0
