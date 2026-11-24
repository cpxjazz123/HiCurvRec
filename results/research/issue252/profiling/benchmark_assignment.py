"""Time the assignment components on real captured cost matrices (diagnostic)."""
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

    captured = []
    original = model.quantize

    def capture(residual, weights, *, balanced):
        costs = [model.assignment_cost(residual, layer.embed.weight, weights)
                 for layer in model.rq.vq_layers]
        captured.append([cost.detach().clone() for cost in costs])
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

    costs = captured[-1]
    np.save(HERE / 'captured_costs.npy', np.stack([cost.cpu().numpy() for cost in costs]))

    import assignment_floor as floor_module
    from model import selection

    report = {}
    layer = model.rq.vq_layers[0]
    cost = costs[0]
    for name, function in (
            ('constrained_assignment', lambda: floor_module.constrained_assignment(cost, 1, counts_key=id(layer))),
            ('argmin_only', lambda: torch.argmin(cost.detach().float(), dim=-1)),
            ('compiled_sinkhorn', lambda: selection.probabilities(
                cost, balanced=True, epsilon=cand.SK_EPSILON, iterations=cand.SK_ITERS, training=True)),
            ('verify_codebook', lambda: floor_module.verify_codebook(layer.embed.weight, distance=False)),
            ('verify_separation', lambda: floor_module.verify_separation(layer.embed.weight, cost, key=id(layer))),
            ('verify_margins', lambda: floor_module.verify_margins(cost)),
    ):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        report[name] = time.perf_counter() - start
        if isinstance(result, tuple):
            report[name + '_report'] = str(result[1])
    # The CPU repair path alone.
    assignment = torch.argmin(cost.detach().float(), dim=-1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    repaired = floor_module._strict_cpu_repair(cost.detach().float().cpu().numpy(),
                                               assignment.cpu().numpy(), 1)
    report['cpu_repair'] = time.perf_counter() - start
    report['cpu_repair_result'] = 'None' if repaired is None else 'assignment'
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
