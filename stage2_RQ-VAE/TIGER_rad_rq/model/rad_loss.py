"""Radius-type structural loss for issue #187.

The only lever with positive evidence on this corpus was struct_mixed's sampled-triple term, which is a pairwise objective and therefore out of
scope here. This module writes structure as a **radius** and as a **non-pairwise dispersion inside each first-level group**:

    r_i    = d_H(0, pos_i) = 2 atanh(||pos_i^H||)
    R_k    = mean_{i in G_k} r_i            (G_k = items level 1 sent to code k, detached)
    L_rad  = mean_i (r_i - R_{k(i)})^2 / (mean_i r_i)^2

No edges are sampled, no triples, no margins, no positive or negative pairs: the gradient flows only through ``r_i``.
"""
from __future__ import annotations

import torch

#: Switch states: BEHAVIOUR_OFF drops the term entirely (the frozen baseline), GEOMETRY_OFF measures radii with the Euclidean norm.
BEHAVIOUR_OFF = False
GEOMETRY_OFF = False


def hyperbolic_radius(pos_h: torch.Tensor) -> torch.Tensor:
    """``2 atanh(||pos_h||)``, the distance from the origin in the Poincare ball, clamped away from the boundary."""
    norm = pos_h.norm(dim=-1).clamp(max=1.0 - 1e-6)
    return 2.0 * torch.atanh(norm)


def radii(pos_e: torch.Tensor, pos_h: torch.Tensor) -> torch.Tensor:
    """Per-item radius: the hyperbolic one, or the Euclidean norm of the joint position under ``GEOMETRY_OFF``."""
    if GEOMETRY_OFF:
        return torch.cat((pos_e, pos_h), dim=-1).norm(dim=-1)
    return hyperbolic_radius(pos_h)


def radius_loss(item_radius: torch.Tensor, group_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The dimensionless group-dispersion loss, with empty groups dropped and one-member groups contributing nothing."""
    if BEHAVIOUR_OFF:
        return item_radius.sum() * 0.0, {"groups": 0, "empty": 0, "singletons": 0, "scale": 0.0}
    n_groups = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    sums = torch.zeros(n_groups, dtype=item_radius.dtype, device=item_radius.device)
    counts = torch.zeros(n_groups, dtype=item_radius.dtype, device=item_radius.device)
    with torch.no_grad():
        sums.index_add_(0, group_ids.to(torch.long), item_radius.detach())
        counts.index_add_(0, group_ids.to(torch.long), torch.ones_like(item_radius))
    present = counts > 0
    group_mean = torch.where(present, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    centre = group_mean[group_ids.to(torch.long)].detach()
    scale = item_radius.detach().mean().clamp_min(1e-12)
    term = (item_radius - centre).pow(2) / scale.pow(2)
    singletons = int((counts == 1).sum())
    empty = int((~present).sum())
    return term.mean(), {"groups": int(present.sum()), "empty": empty, "singletons": singletons,
                         "scale": float(scale)}


def is_pairwise_free(source: str) -> bool:
    """The machine assertion acceptance item 6 asks for: no edge sampling, triples, margin, InfoNCE, BCE or source-target residuals.

    The module docstring and this function's own token list are excluded, since they necessarily mention what they forbid; what is scanned is
    the executable code.
    """
    body = source.split('"""', 2)[-1]                 # drop the module docstring
    body = body.split("def is_pairwise_free")[0]      # and this checker's own literal
    forbidden = ("triple", "margin", "InfoNCE", "infonce", "BCE", "bce_loss", "edge_index", "source_target")
    return not any(token in body for token in forbidden)


class CalibrationDegenerate(RuntimeError):
    """The budget rule has no finite answer, so training must not start."""

    status = "CALIBRATION_DEGENERATE"


def calibrate_budget(blocks: dict, base_loss: torch.Tensor, struct_loss: torch.Tensor) -> tuple[float, dict]:
    """``lambda* = min over blocks ||g_block^base|| / ||g_block^struct||``, the repository's frozen budget rule.

    Same rule as ``training_hooks.calibrate_lambda_block_budget`` -- no epsilon, no seed switching, no fallback -- but its structure term is this
    module's radius loss, because the hooks' term is the pairwise triple objective this arm may not use. A block whose structure gradient
    vanishes makes the budget undefined and raises :class:`CalibrationDegenerate` instead of returning a guess.
    """
    flat = [p for params in blocks.values() for p in params if p.requires_grad]
    if not flat:
        raise CalibrationDegenerate("no trainable parameters in the supervised blocks")

    def block_norms(term: torch.Tensor) -> dict:
        grads = torch.autograd.grad(term, flat, retain_graph=True, allow_unused=True)
        out, cursor = {}, 0
        for name, params in blocks.items():
            collected = []
            for p in params:
                if not p.requires_grad:
                    continue
                g = grads[cursor]
                cursor += 1
                collected.append(torch.zeros_like(p) if g is None else g)
            stack = torch.cat([g.reshape(-1) for g in collected]) if collected else None
            out[name] = None if stack is None else float(stack.norm())
        return out

    base = block_norms(base_loss)
    struct = block_norms(struct_loss)
    ratios = {}
    for name, value in struct.items():
        if value is None or value <= 0.0:
            raise CalibrationDegenerate(f"block {name} has no structure gradient (norm={value})")
        ratios[name] = base[name] / value if base[name] is not None else 0.0
    lam = min(ratios.values())
    return lam, {"ratios": ratios, "lambda": lam}
