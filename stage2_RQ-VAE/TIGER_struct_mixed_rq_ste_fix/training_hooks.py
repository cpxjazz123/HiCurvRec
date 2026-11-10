"""Glue between the trainer and the structure term, kept small and testable."""
from __future__ import annotations

import random

import torch

import structure as S


def _positions(model, embeddings: torch.Tensor, item_ids: list[int]) -> torch.Tensor:
    """The three cumulative readbacks q_1, q_2, q_3 from the same forward.

    forward_with_codes already returns every level as reconstruct(codes up to
    that level) plus the tangent straight-through, so reading all three adds no
    second codebook path: the codebook sees the native composition Jacobian and
    the encoder sees the STE of x.
    """
    gathered = embeddings[torch.tensor(item_ids, dtype=torch.long)]
    _, _, _, _, levels = model.forward_with_codes(gathered)
    return torch.stack(levels, dim=0)


def structure_term(model, embeddings: torch.Tensor, graph, rng: random.Random, *,
                   scale: float, size: int, behaviour_on: bool = True,
                   geometry_on: bool | None = None,
                   triples=None, capture=None) -> torch.Tensor:
    """(1/3) mean pairwise term over sampled triples, on quantized positions."""
    if not behaviour_on:
        return torch.zeros((), device=embeddings.device)
    if triples is None:
        triples = S.sample_triples(graph, size, rng)
    else:
        triples = list(triples)
    if not triples:
        return torch.zeros((), device=embeddings.device)
    item_ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    positions_all = _positions(model, embeddings, item_ids).reshape(3, len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions_all.device)
    if capture is not None:
        capture.append({"items": list(item_ids),
                        "triples": [tuple(t) for t in triples],
                        "targets": h.detach().tolist(),
                        "position_h_norm_max": [float(positions_all[l][..., 16:].norm(dim=-1).max())
                                                if positions_all.shape[-1] > 16 else 0.0
                                                for l in range(3)],
                        "codeword_h_norm_max": [float(layer.get_code_embs()[..., 16:].norm(dim=-1).max().detach())
                                                for layer in model.rq.vq_layers]})
    losses = [S.structure_loss(positions_all[l], h, scale=scale, geometry_on=geometry_on)
              for l in range(3)]
    return torch.stack(losses).mean()


def calibrate_structure_scale(model, embeddings: torch.Tensor, graph, rng: random.Random, *,
                              size: int, geometry_on: bool | None = None) -> float:
    """Median target-side distance at initialisation; deterministic for a given rng."""
    triples = S.sample_triples(graph, size, rng)
    if not triples:
        return 1.0
    item_ids = [item for centre, a, b, _ in triples for item in (centre, a, b)]
    positions = _positions(model, embeddings, item_ids)[-1].reshape(len(triples), 3, -1)
    h = S.triple_targets([kind for *_, kind in triples], device=positions.device)
    return S.calibrate_scale(positions, h)


class CalibrationDegenerate(RuntimeError):
    """The frozen calibration cannot define a weight; the run must not train."""

    def __init__(self, detail: dict) -> None:
        super().__init__("CALIBRATION_DEGENERATE: %s" % detail)
        self.detail = detail
        self.status = "CALIBRATION_DEGENERATE"


def calibrate_lambda_block_budget(model, embeddings: torch.Tensor, graph,
                                  rng: random.Random, *, scale: float, size: int,
                                  base_loss: torch.Tensor, geometry_on: bool | None = None,
                                  epsilon: float = 0.0) -> tuple[float, dict]:
    """lambda* = min over parameter blocks of ||g_block^base|| / ||g_block^struct||.

    The budget is pre-registered: the structure term may not exceed the base
    term's gradient norm on ANY of the jointly supervised blocks at calibration
    time.  The decoder is deliberately not a block: a large decoder gradient must
    not raise the structure weight.  No epsilon, no seed switching, no fallback.
    """
    blocks = {
        "encoder": list(model.encoder.parameters()),
        "codebooks": [p for layer in model.rq.vq_layers for p in layer.parameters()],
    }
    structure = structure_term(model, embeddings, graph, rng, scale=scale, size=size,
                               behaviour_on=True, geometry_on=geometry_on)

    def block_gradients(term: torch.Tensor) -> dict[str, torch.Tensor | None]:
        flat = [p for params in blocks.values() for p in params if p.requires_grad]
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
            out[name] = torch.cat([g.reshape(-1) for g in collected]) if collected else None
        return out

    base_grads = block_gradients(base_loss)
    struct_grads = block_gradients(structure)
    detail = {"blocks": {}, "epsilon": epsilon}
    candidates = []
    for name in blocks:
        gb, gs = base_grads[name], struct_grads[name]
        nb = float(gb.norm()) if gb is not None else None
        ns = float(gs.norm()) if gs is not None else None
        cosine = None
        if gb is not None and gs is not None and nb > 0 and ns > 0:
            cosine = float(torch.dot(gb, gs) / (gb.norm() * gs.norm()))
        detail["blocks"][name] = {"base_norm": nb, "structure_norm": ns, "cosine": cosine}
        if (gb is None or gs is None or nb is None or ns is None
                or not (torch.isfinite(torch.tensor(nb)) and torch.isfinite(torch.tensor(ns)))
                or nb == 0 or ns == 0):
            detail["degenerate_block"] = name
            detail["status"] = "CALIBRATION_DEGENERATE"
            detail["lambda"] = None
            raise CalibrationDegenerate(detail)
        candidates.append(nb / ns)
    lam = min(candidates)
    detail["lambda"] = lam
    return lam, detail
