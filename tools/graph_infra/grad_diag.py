"""Gradient and reconstruction diagnostics (issue #249, CONTRACT.md §5 and the issue's step 5).

The diagnostics are deliberately read-only: they report what the gradients look like, they never
change the model. Two families:

* ``curvature_diagnostics`` — for one batch of triples: the statistic, the gradient of the
  statistic with respect to the quantized vectors, its norm, the ratio between the raw and the
  weighted reconstruction, and the cosine between the raw and the weighted gradient.
* ``reconstruction_diagnostics`` — for a reconstruction error: the gradient of the error with
  respect to the vectors, before and after an optional gradient clip.

Nothing here claims that a differentiable distance implies the real SID can change; the report
states that explicitly.
"""
from __future__ import annotations
import torch

from tools.graph_infra.diff_sectional import sectional_terms


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = float((first * second).sum())
    denominator = float(first.norm() * second.norm())
    if denominator == 0.0:
        return float('nan')
    return numerator / denominator


def curvature_diagnostics(vectors: torch.Tensor, centre: torch.Tensor, b: torch.Tensor,
                           c: torch.Tensor, references: torch.Tensor, metric,
                           weights: torch.Tensor | None = None,
                           mask: torch.Tensor | None = None) -> dict:
    """Statistics and gradients of the A.1 statistic for one batch of triples.

    ``vectors`` must carry ``requires_grad`` if gradients are wanted; the returned gradient is
    None otherwise. The report also states, in words, that a differentiable distance is not evidence
    that the discrete SID can move.
    """
    reference_vectors = vectors[references]
    centre_vector = vectors[centre]
    b_vector = vectors[b]
    c_vector = vectors[c]
    distances = {
        'd_am': metric(reference_vectors, centre_vector.unsqueeze(-2)),
        'd_ab': metric(reference_vectors, b_vector.unsqueeze(-2)),
        'd_ac': metric(reference_vectors, c_vector.unsqueeze(-2)),
        'd_bc': metric(b_vector, c_vector),
    }
    weighted, mass = sectional_terms(distances, weights=weights, mask=mask)
    statistic = weighted.sum(dim=-1) / mass
    report = {
        'statistic': statistic.detach().tolist() if statistic.numel() > 1 else float(statistic.detach()),
        'reference_mass': mass.detach().tolist() if mass.numel() > 1 else float(mass.detach()),
        'weights_are_caller_specified': weights is not None,
        'caveat': ('a differentiable distance is not evidence that the discrete SID can change; '
                   'the SID codes are discrete and a hard assignment has no gradient'),
    }
    if vectors.requires_grad:
        statistic.sum().backward()
        gradient = vectors.grad
        if gradient is None:
            report['gradient'] = None
            report['gradient_norm'] = 0.0
            report['gradient_is_finite'] = True
        else:
            report['gradient'] = gradient.detach().tolist()
            report['gradient_norm'] = float(gradient.norm())
            report['gradient_is_finite'] = bool(torch.isfinite(gradient).all())
            report['gradient_is_nonzero'] = bool(gradient.abs().sum() > 0)
    else:
        report['gradient'] = None
        report['gradient_norm'] = 0.0
        report['gradient_is_finite'] = True
        report['gradient_is_nonzero'] = False
    return report


def weighted_vs_raw(vectors: torch.Tensor, centre: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                    references: torch.Tensor, metric, weights: torch.Tensor) -> dict:
    """Compare the weighted and the unweighted statistic and their gradient directions."""
    reference_vectors = vectors[references]
    centre_vector = vectors[centre]
    b_vector = vectors[b]
    c_vector = vectors[c]
    distances = {
        'd_am': metric(reference_vectors, centre_vector.unsqueeze(-2)),
        'd_ab': metric(reference_vectors, b_vector.unsqueeze(-2)),
        'd_ac': metric(reference_vectors, c_vector.unsqueeze(-2)),
        'd_bc': metric(b_vector, c_vector),
    }
    raw_weighted, raw_mass = sectional_terms(distances)
    weighted_terms, weighted_mass = sectional_terms(distances, weights=weights)
    raw_statistic = (raw_weighted.sum(dim=-1) / raw_mass).detach()
    weighted_statistic = (weighted_terms.sum(dim=-1) / weighted_mass).detach()
    report = {
        'raw_statistic': raw_statistic.tolist(),
        'weighted_statistic': weighted_statistic.tolist(),
        'ratio': (weighted_statistic / raw_statistic).tolist(),
        'mass_ratio': (weighted_mass / raw_mass).tolist(),
    }
    if vectors.requires_grad:
        vectors.grad = None
        raw_statistic.sum().backward(retain_graph=True)
        raw_gradient = vectors.grad.clone()
        vectors.grad = None
        weighted_statistic.sum().backward()
        weighted_gradient = vectors.grad.clone()
        report['raw_gradient_norm'] = float(raw_gradient.norm())
        report['weighted_gradient_norm'] = float(weighted_gradient.norm())
        report['gradient_cosine'] = _cosine(raw_gradient, weighted_gradient)
        report['gradient_norm_ratio'] = (float(weighted_gradient.norm() / raw_gradient.norm())
                                         if float(raw_gradient.norm()) > 0 else float('nan'))
    return report


def reconstruction_diagnostics(target: torch.Tensor, prediction: torch.Tensor,
                                vectors: torch.Tensor | None = None,
                                clip: float | None = None) -> dict:
    """Gradients of the reconstruction error, before and after an optional clip.

    ``vectors`` is optional: when it is given, the gradient of the error with respect to the
    vectors is reported.
    """
    error = (prediction - target).pow(2).sum()
    report = {'reconstruction_error': float(error.detach()),
              'shape': list(target.shape)}
    if vectors is None or not vectors.requires_grad:
        report['gradient'] = None
        report['gradient_norm'] = 0.0
        report['gradient_is_finite'] = True
        return report
    error.backward()
    gradient = vectors.grad.clone()
    report['gradient_norm'] = float(gradient.norm())
    report['gradient_is_finite'] = bool(torch.isfinite(gradient).all())
    report['gradient_is_nonzero'] = bool(gradient.abs().sum() > 0)
    report['gradient_before_clip'] = gradient.detach().tolist()
    if clip is not None:
        report['clip'] = clip
        report['gradient_after_clip'] = gradient.clamp(-clip, clip).detach().tolist()
        report['clipped_fraction'] = float((gradient.abs() > clip).float().mean())
        report['norm_before_clip'] = float(gradient.norm())
        report['norm_after_clip'] = float(gradient.clamp(-clip, clip).norm())
    return report
