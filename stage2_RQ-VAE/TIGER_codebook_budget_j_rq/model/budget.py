"""Issue #272 round 3: codebook-side gradient budget.

The local curvature term and the global J term act on the encoder and the
codebook.  The codebook-side share of those two terms is scaled by ``beta``
after the total backward; the encoder, the decoder and the reconstruction keep
their full gradients.  The codebook gradient becomes
``g_recon + beta * (g_local + g_constraint)``.

Owner 2026-11-27 iteration contract: this is the allowed "gradient budget /
conflict handling" intervention; no objective, architecture, selection,
Sinkhorn or reconstruction change.
"""
from __future__ import annotations

import torch


def scale_codebook_terms(local_term, constraint_term, codebook_parameters, beta):
    """Scale the local/global codebook gradient after ``loss.backward()``.

    The total loss must have been backpropagated with ``retain_graph=True``.
    The helper subtracts ``(1 - beta)`` times the local+constraint codebook
    gradient from the already accumulated ``.grad``; it never touches other
    parameters.  Returns a small JSON-safe record for the metrics stream.
    """
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    params = list(codebook_parameters)
    if not params:
        raise ValueError("codebook_parameters must not be empty")
    combined = local_term + constraint_term
    grads = torch.autograd.grad(combined, params, allow_unused=True)
    scale = 1.0 - beta
    touched = 0
    for parameter, grad in zip(params, grads):
        if parameter.grad is None or grad is None:
            continue
        parameter.grad.sub_(grad, alpha=scale)
        touched += 1
    return {"beta": beta, "parameters": len(params), "touched": touched}
