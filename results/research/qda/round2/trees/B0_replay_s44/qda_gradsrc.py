"""Which loss moves the encoder, and how far the move carries the representation.

The round-2 telemetry established *that* the encoder output travels 40-100 times
faster than the L1 codebook during the first epochs. This module answers *why*: it
splits the encoder's gradient by loss term, compares the terms' directions, and
converts each term into the output displacement it would produce on its own.

The three terms are the ones the frozen trainer actually optimises:

    L = L_recon + mean_level[ codebook_l + beta * commitment_l ] / TOTAL_DIM

`quant_parts` reproduces `MixedVQLevel.forward` exactly (same alpha lookup, same
assignment rule, same residual update) and returns the two quantisation terms
separately, because the frozen forward only ever returns their sum.

A term's displacement is a *virtual first-order step*: the encoder parameters are
moved by `-lr * g_term`, the probe is re-encoded, and the parameters are restored.
That is not AdamW's update; it is the honest one-step linearisation and is labelled
as such wherever it is reported.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

TOTAL_DIM = 32


def quant_parts(model, batch: torch.Tensor, curvature: float = 1.0):
    """The two quantisation terms of one batch, separate and still attached to the graph."""
    from model.mixed_rq import expmap0, mobius_add, project, split

    encoded = model.encoder(batch)
    z_e, z_t = split(encoded)
    residual_e = z_e
    residual_h = expmap0(z_t, curvature) if curvature > 0 else z_t
    tokens: list[torch.Tensor] = []
    codebook_total = 0.0
    commitment_total = 0.0
    for level, layer in enumerate(model.rq.levels, start=1):
        alpha = model.rq.alpha_table.lookups(level, tokens).to(encoded.device)
        cost = layer.distances(residual_e, residual_h, alpha)
        index = layer._indices(cost, False)
        codes = layer.get_code_embs()
        codes_e, codes_h_tangent = split(codes)
        picked_e = codes_e[index]
        picked_h_tangent = codes_h_tangent[index]
        picked_h = (expmap0(picked_h_tangent, curvature)
                    if curvature > 0 else picked_h_tangent)
        commitment = layer._cost_of(residual_e, residual_h,
                                    picked_e.detach(), picked_h.detach(), alpha)
        codebook = layer._cost_of(residual_e.detach(), residual_h.detach(),
                                  picked_e, picked_h, alpha)
        codebook_total = codebook_total + codebook.mean() / float(TOTAL_DIM)
        commitment_total = commitment_total + commitment.mean() / float(TOTAL_DIM)
        with torch.no_grad():
            residual_e = residual_e - picked_e.detach()
            residual_h = (project(mobius_add(-picked_h.detach(), residual_h, curvature),
                                  curvature)
                          if curvature > 0 else residual_h - picked_h.detach())
        tokens.append(index)
    levels = float(len(model.rq.levels))
    return codebook_total / levels, commitment_total / levels


@torch.no_grad()
def _encode(model, probe: torch.Tensor) -> torch.Tensor:
    return model.encoder(probe).detach().clone()


def _grads(model) -> list[torch.Tensor]:
    return [p.grad.detach().clone() if p.grad is not None else None
            for p in model.encoder.parameters()]


def _flat(grads) -> torch.Tensor:
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


@torch.no_grad()
def _virtual_step(model, grads, lr: float, probe: torch.Tensor,
                  baseline: torch.Tensor) -> float:
    """Mean ||dz|| on the probe after one first-order step with `grads` alone."""
    parameters = list(model.encoder.parameters())
    saved = [p.detach().clone() for p in parameters]
    for parameter, grad in zip(parameters, grads):
        if grad is not None:
            parameter.add_(grad, alpha=-lr)
    moved = float((model.encoder(probe) - baseline).norm(dim=1).mean())
    for parameter, value in zip(parameters, saved):
        parameter.copy_(value)
    return moved


def measure(model, probe: torch.Tensor, beta: float, lr: float,
            curvature: float = 1.0) -> dict:
    """Gradient attribution for one epoch on a fixed probe batch."""
    baseline = _encode(model, probe)

    model.zero_grad(set_to_none=True)
    reconstructed, _, _, _ = model(probe)
    recon = F.mse_loss(reconstructed, probe)
    recon.backward()
    grad_recon = _grads(model)

    model.zero_grad(set_to_none=True)
    codebook, _ = quant_parts(model, probe, curvature)
    codebook.backward()
    grad_codebook = _grads(model)

    model.zero_grad(set_to_none=True)
    _, commitment = quant_parts(model, probe, curvature)
    (beta * commitment).backward()
    grad_commit = _grads(model)

    model.zero_grad(set_to_none=True)

    flat = {"recon": _flat(grad_recon), "codebook": _flat(grad_codebook),
            "commitment": _flat(grad_commit)}
    cosines = {}
    for a, b in (("recon", "codebook"), ("recon", "commitment"),
                 ("codebook", "commitment")):
        denom = flat[a].norm() * flat[b].norm()
        cosines[f"cos_{a}_{b}"] = float(flat[a] @ flat[b] / denom) if float(denom) > 0 else 0.0

    displacement = {name: _virtual_step(model, grad, lr, probe, baseline)
                    for name, grad in (("recon", grad_recon),
                                       ("codebook", grad_codebook),
                                       ("commitment", grad_commit))}

    layers = []
    for (name, parameter), grad in zip(model.encoder.named_parameters(), grad_recon):
        if grad is None:
            continue
        layers.append({"name": name, "shape": list(parameter.shape),
                       "param_norm": float(parameter.norm()),
                       "grad_norm": float(grad.norm())})

    return {
        "probe_items": int(probe.shape[0]),
        "grad_norm": {key: float(value.norm()) for key, value in flat.items()},
        "displacement": displacement,
        "z_norm": float(baseline.norm(dim=1).mean()),
        "cosines": cosines,
        "layers": layers,
        "loss": {"recon": float(recon), "codebook": float(codebook),
                 "commitment": float(commitment)},
    }
