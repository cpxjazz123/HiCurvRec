"""QDA: record the quantization trajectory at fixed epochs without touching training.

The observer answers one question per observation point: what did the encoder
produce, what did each quantization level do with it, and how much structure
survived.  It is strictly read-only -- ``eval()`` mode, ``no_grad``, no RNG, no
optimizer call -- and it restores the training mode before returning, so a run
with the monitor on and a run with it off must produce identical weights.  That
pair is the control the experiment reports.

Recorded per observation point, for every item in the frozen id order:

* ``z``                    encoder output, 32 dims;
* ``readout_1/2/3``        the cumulative quantized vector phi(q) after each level;
* ``residual_e/h_1..3``    the residual each level started from;
* ``codes``                the argmin code of each level (evaluation rule);
* ``codes_sinkhorn``       the last level's balanced assignment (training rule);
* ``margin_1..3``          second-nearest minus nearest cost, per item;
* ``error_1..3``           the item's quantized cost in the mixed metric;
* ``usage_1..3``           the per-codeword assignment histogram.

The two code sets are kept apart on purpose: the trainer assigns the last level
with Sinkhorn and exports with argmin, and mixing them would make "code changed"
mean two different things.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

#: The observation points the experiment froze before the first run.
OBSERVATIONS = (0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000)


def _codebook_stats(codes: torch.Tensor, size: int) -> np.ndarray:
    counts = torch.bincount(codes, minlength=size).float()
    return counts.cpu().numpy()


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _effective_rank(matrix: np.ndarray) -> float:
    """exp(entropy of the singular-value spectrum), a scale-free rank proxy."""
    values = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    p = values / values.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def observe(model, embeddings: torch.Tensor, epoch: int, out_dir: Path,
            curvature: float = 1.0, light: bool = False) -> dict:
    """One read-only pass; writes ``epoch_<NNNN>.npz`` and returns its summary."""
    from model.mixed_rq import (expmap0, logmap0, mobius_add, project,
                                quantized_readout, split)

    was_training = model.training
    model.eval()
    try:
        encoded = model.encoder(embeddings)
        z_e, z_t = split(encoded)
        residual_e = z_e
        residual_h = expmap0(z_t, curvature) if curvature > 0 else z_t
        tokens: list[torch.Tensor] = []
        picked_e: list[torch.Tensor] = []
        picked_h: list[torch.Tensor] = []
        record: dict[str, np.ndarray] = {"z": encoded.cpu().numpy().astype(np.float16)}

        for level, layer in enumerate(model.rq.levels, start=1):
            alpha = model.rq.alpha_table.lookups(level, tokens).to(encoded.device)
            cost = layer.distances(residual_e, residual_h, alpha)
            smallest, index_of_smallest = torch.topk(cost, 2, dim=1, largest=False)
            index = index_of_smallest[:, 0]
            margin = (smallest[:, 1] - smallest[:, 0])
            codes = layer.get_code_embs()
            codes_e, codes_h_tangent = split(codes)
            chosen_e = codes_e[index]
            chosen_h = (expmap0(codes_h_tangent[index], curvature)
                        if curvature > 0 else codes_h_tangent[index])
            # The item's own cost under its chosen code, in the same convention
            # the loss uses (2 * [a * dE^2 + (1-a) * dH^2 / 4]).
            position = index.view(-1, 1)
            chosen_cost = cost.gather(1, position).squeeze(1)
            record[f"residual_e_{level}"] = residual_e.cpu().numpy().astype(np.float16)
            record[f"residual_h_{level}"] = residual_h.cpu().numpy().astype(np.float16)
            # float32 for the scalars: the margins are ~1e-4 and float16 rounds
            # most of them to zero, which silently turns the margin axis into noise.
            record[f"margin_{level}"] = margin.cpu().numpy().astype(np.float32)
            record[f"error_{level}"] = chosen_cost.cpu().numpy().astype(np.float32)
            record[f"usage_{level}"] = _codebook_stats(index, layer.n_embed)
            if level == len(model.rq.levels):
                sinkhorn_index = layer._indices(cost, infer_use_sk=True)
                record["codes_sinkhorn"] = sinkhorn_index.cpu().numpy().astype(np.int16)
            picked_e.append(chosen_e)
            picked_h.append(chosen_h)
            tokens.append(index)
            record[f"readout_{level}"] = quantized_readout(
                picked_e, picked_h, curvature).cpu().numpy().astype(np.float16)
            residual_e = residual_e - chosen_e
            residual_h = (project(mobius_add(-chosen_h, residual_h, curvature), curvature)
                          if curvature > 0 else residual_h - chosen_h)
        record["codes"] = torch.stack(tokens, dim=1).cpu().numpy().astype(np.int16)

        summary = {
            "epoch": int(epoch),
            "items": int(encoded.shape[0]),
            "entropy": [_entropy(record[f"usage_{level}"])
                        for level in range(1, len(model.rq.levels) + 1)],
            "used_codes": [int((record[f"usage_{level}"] > 0).sum())
                           for level in range(1, len(model.rq.levels) + 1)],
            "mean_error": [float(np.mean(record[f"error_{level}"].astype(np.float32)))
                           for level in range(1, len(model.rq.levels) + 1)],
            "mean_margin": [float(np.mean(record[f"margin_{level}"].astype(np.float32)))
                            for level in range(1, len(model.rq.levels) + 1)],
            "effective_rank": {
                "z": _effective_rank(record["z"].astype(np.float32)),
                **{f"readout_{level}": _effective_rank(
                    record[f"readout_{level}"].astype(np.float32))
                   for level in range(1, len(model.rq.levels) + 1)},
                **{f"residual_{level}": _effective_rank(
                    record[f"residual_e_{level}"].astype(np.float32))
                   for level in range(1, len(model.rq.levels) + 1)},
            },
        }
    finally:
        if was_training:
            model.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    if light:
        # The dense points of the fine window need the summary only; a 9 MB snapshot
        # per epoch would be 300 MB per arm for numbers nothing reads.
        (out_dir / ("epoch_%04d.json" % epoch)).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    np.savez_compressed(out_dir / ("epoch_%04d.npz" % epoch), **record)
    (out_dir / ("epoch_%04d.json" % epoch)).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def gradient_snapshot(model) -> dict:
    """Per-group gradient and update sizes, taken around one optimizer step.

    Called by the trainer right after ``backward`` (gradients) and right after
    ``step`` (update), so the ratio compares what the optimizer actually changed
    with the norm of the weights it changed.
    """
    out = {}
    encoder = [p.grad.detach() for p in model.encoder.parameters() if p.grad is not None]
    out["encoder_grad_norm"] = float(torch.sqrt(sum(g.pow(2).sum() for g in encoder))) \
        if encoder else 0.0
    for level, layer in enumerate(model.rq.levels, start=1):
        grads = [p.grad.detach() for p in layer.parameters() if p.grad is not None]
        out[f"codebook_{level}_grad_norm"] = float(torch.sqrt(sum(g.pow(2).sum() for g in grads))) \
            if grads else 0.0
    return out


def update_snapshot(before: dict, model) -> dict:
    """||delta w|| / ||w|| for the encoder and every codebook."""
    out = {}
    groups = {"encoder": list(model.encoder.parameters())}
    for level, layer in enumerate(model.rq.levels, start=1):
        groups[f"codebook_{level}"] = list(layer.parameters())
    for name, params in groups.items():
        delta = 0.0
        weight = 0.0
        for index, param in enumerate(params):
            key = f"{name}:{index}"
            if key not in before:
                continue
            delta += float((param.detach() - before[key]).pow(2).sum())
            weight += float(param.detach().pow(2).sum())
        out[f"{name}_update_to_weight"] = (delta ** 0.5) / (weight ** 0.5) if weight > 0 else 0.0
    return out


def weights_snapshot(model) -> dict:
    return {f"{name}:{index}": param.detach().clone()
            for name, params in (("encoder", list(model.encoder.parameters())),
                                 *[(f"codebook_{level}", list(layer.parameters()))
                                   for level, layer in enumerate(model.rq.levels, start=1)])
            for index, param in enumerate(params)}
