"""The independent reference implementation of the masked directed codebook margin.

This file is the oracle the M1 and M2 gates compare the production mechanism against. Two
rules make it an oracle rather than a second copy:

* it never imports the production module -- it is written from the specification's
  equation, in a different code path (explicit loops over rows and codebook entries
  instead of the production code's vectorised broadcasting), so a mistake in the
  production code does not travel here;
* it exposes its intermediates, so the gate compares not only the scalar loss but the
  quantities the loss is built from.

The gradient check uses central finite differences rather than a hand-written VJP: the
directional derivative it compares against autograd is then independent of both the
production implementation and of any analytic derivation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MARGIN = 0.5
DEFAULT_STRUCTURAL_WEIGHT = 0.1


def _l1(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(sum(abs(float(left[d]) - float(right[d])) for d in range(left.shape[0])))


def _distance_matrix(rows: torch.Tensor, codebook: torch.Tensor) -> List[List[float]]:
    """The L1 distance from every row to every codebook entry."""
    return [[_l1(rows[b], codebook[c]) for c in range(codebook.shape[0])] for b in range(rows.shape[0])]


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    scaled = [-value / temperature for value in values]
    largest = max(scaled)
    exps = [pow(2.718281828459045, value - largest) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def reference_intermediates(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    margin: float = DEFAULT_MARGIN,
    structural_weight: float = DEFAULT_STRUCTURAL_WEIGHT,
) -> Dict[str, Any]:
    """Every quantity the loss is built from, computed one entry at a time."""
    batch = int(z_src.shape[0])
    entries = int(codebook.shape[0])

    src_dist = _distance_matrix(z_src, codebook)
    tgt_dist = _distance_matrix(z_tgt, codebook)

    src_index: List[int] = []
    tgt_index: List[int] = []
    src_nearest: List[float] = []
    tgt_nearest: List[float] = []
    margin_term: List[float] = []
    structural: List[float] = []

    geometry = [[_l1(codebook[i], codebook[j]) for j in range(entries)] for i in range(entries)]

    for b in range(batch):
        row_src = src_dist[b]
        row_tgt = tgt_dist[b]
        # The lowest index wins a tie -- the same rule the production argmin follows.
        best_src = 0
        for c in range(1, entries):
            if row_src[c] < row_src[best_src]:
                best_src = c
        best_tgt = 0
        for c in range(1, entries):
            if row_tgt[c] < row_tgt[best_tgt]:
                best_tgt = c
        src_index.append(best_src)
        tgt_index.append(best_tgt)
        src_nearest.append(row_src[best_src])
        tgt_nearest.append(row_tgt[best_tgt])
        margin_term.append(max(0.0, margin + row_tgt[best_tgt] - row_src[best_src]))

        assign_src = _softmax(row_src, temperature)
        assign_tgt = _softmax(row_tgt, temperature)
        total = 0.0
        for i in range(entries):
            for j in range(entries):
                total += assign_src[i] * assign_tgt[j] * geometry[i][j]
        structural.append(total)

    per_pair = [margin_term[b] + structural_weight * structural[b] for b in range(batch)]
    weights = [float(mask[b]) for b in range(batch)]
    valid = sum(weights)
    if valid <= 0.0:
        loss = 0.0
    else:
        loss = sum(weights[b] * per_pair[b] for b in range(batch)) / valid

    return {
        "src_dist": src_dist,
        "tgt_dist": tgt_dist,
        "src_index": src_index,
        "tgt_index": tgt_index,
        "src_nearest": src_nearest,
        "tgt_nearest": tgt_nearest,
        "margin_term": margin_term,
        "structural": structural,
        "per_pair": per_pair,
        "weights": weights,
        "valid": valid,
        "loss": loss,
    }


def reference_loss(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    codebook: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    margin: float = DEFAULT_MARGIN,
    structural_weight: float = DEFAULT_STRUCTURAL_WEIGHT,
) -> float:
    """The loss value, computed independently of the production implementation."""
    return float(
        reference_intermediates(
            z_src,
            z_tgt,
            codebook,
            mask,
            temperature=temperature,
            margin=margin,
            structural_weight=structural_weight,
        )["loss"]
    )


def finite_difference_gradient(
    loss_fn: Callable[[Sequence[torch.Tensor]], torch.Tensor],
    tensors: Sequence[torch.Tensor],
    *,
    step: float = 1e-6,
) -> List[torch.Tensor]:
    """Central differences of ``loss_fn`` with respect to each tensor.

    ``loss_fn`` takes the tensors as a sequence and returns a scalar tensor. The gradient
    is returned in the same order and shape as the inputs. The inputs are never perturbed
    in place: a leaf that requires grad cannot be written to, and the perturbation has to
    be a different tensor anyway.
    """
    gradients: List[torch.Tensor] = []
    for index, tensor in enumerate(tensors):
        grad = torch.zeros_like(tensor)
        flat = grad.reshape(-1)
        base = tensor.detach().clone().reshape(-1)
        for position in range(base.numel()):
            original = float(base[position])
            plus_tensors = [t.detach().clone() for t in tensors]
            minus_tensors = [t.detach().clone() for t in tensors]
            plus_tensors[index].reshape(-1)[position] = original + step
            minus_tensors[index].reshape(-1)[position] = original - step
            plus = float(loss_fn(plus_tensors).detach())
            minus = float(loss_fn(minus_tensors).detach())
            flat[position] = (plus - minus) / (2.0 * step)
        gradients.append(grad)
    return gradients


def directional_derivative(
    loss_fn: Callable[[Sequence[torch.Tensor]], torch.Tensor],
    tensors: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    *,
    step: float = 1e-6,
) -> float:
    """The central-difference directional derivative along ``direction``."""
    plus = [t.detach() + step * d for t, d in zip(tensors, direction)]
    minus = [t.detach() - step * d for t, d in zip(tensors, direction)]
    return (float(loss_fn(plus).detach()) - float(loss_fn(minus).detach())) / (2.0 * step)


def as_float_list(values: Any) -> List[float]:
    """Flatten a tensor into plain floats."""
    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.reshape(-1).tolist()]
    return [float(v) for v in values]


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_STRUCTURAL_WEIGHT",
    "DEFAULT_TEMPERATURE",
    "as_float_list",
    "directional_derivative",
    "finite_difference_gradient",
    "reference_intermediates",
    "reference_loss",
]
