"""Decompose one real quantize call into its measured parts (diagnostic)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build

HERE = Path(__file__).resolve().parent


def main():
    context = build()
    model = context['model']
    loader = context['loader']
    optimizer = context['optimizer']
    catalogue = context['catalogue']
    sampler = context['sampler']
    device = context['device']
    cand = context['cand']

    captured = {}
    original = model.quantize

    def capture(residual, weights, *, balanced):
        captured.setdefault('residual', residual.detach().clone())
        captured.setdefault('weights', weights.detach().clone())
        captured.setdefault('balanced', balanced)
        return original(residual, weights, balanced=balanced)

    model.quantize = capture
    model.train()
    sampler.begin_epoch()
    for step, (batch, item_ids) in enumerate(loader):
        if step >= 2:
            break
        batch = batch.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        sample = sampler.draw(step, 0, 1)
        reconstructed, quant_parts, _, _, curvature_loss = model(
            batch, item_ids, catalogue_embeddings=catalogue, structural_sample=sample)
        original_loss, components = cand.CRITERION(reconstructed, batch, quant_parts)
        loss = model.combined_loss(original_loss, curvature_loss)
        loss.backward()
        optimizer.step()
    model.quantize = original

    import assignment_floor as floor_module
    from model import selection

    residual = captured['residual']
    weights = captured['weights']
    layer = model.rq.vq_layers[0]
    report = {}

    def timed(name, function, repeats=3):
        values = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                function()
            except Exception as error:  # a raised guard is part of the measurement
                values.append(float('nan'))
                report[name + '_error'] = type(error).__name__
                continue
            torch.cuda.synchronize()
            values.append(time.perf_counter() - start)
        report[name] = {'first': values[0], 'median': float(np.median(values)),
                        'all': [round(v, 6) for v in values]}

    timed('sinkhorn_probabilities', lambda: selection.probabilities(
        residual, balanced=True, epsilon=cand.SK_EPSILON, iterations=cand.SK_ITERS, training=True))
    cost = model.assignment_cost(residual, layer.embed.weight, weights)
    timed('assignment_cost', lambda: model.assignment_cost(residual, layer.embed.weight, weights))
    timed('constrained_assignment', lambda: floor_module.constrained_assignment(cost, 1, counts_key=id(layer)))
    timed('verify_codebook', lambda: floor_module.verify_codebook(layer.embed.weight, distance=False))
    timed('verify_separation', lambda: floor_module.verify_separation(layer.embed.weight, cost, key=id(layer)))
    timed('verify_margins', lambda: floor_module.verify_margins(cost))
    hard_ids, _ = floor_module.constrained_assignment(cost, 1, counts_key=id(layer))
    timed('verify_assignment', lambda: floor_module.verify_assignment(hard_ids, layer.n_embed, 1))
    timed('select_whole', lambda: selection.select(
        cost, layer.embed.weight, balanced=layer.use_sk, epsilon=layer.sk_epsilon,
        iterations=layer.sk_iters, training=True, hard_ids=hard_ids))
    timed('quantize_whole', lambda: model.quantize(residual, weights, balanced=True))
    timed('losses', lambda: (model.commitment_loss(residual, layer.embed.weight[hard_ids]),
                              model.codebook_loss(residual, layer.embed.weight[hard_ids])))
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
