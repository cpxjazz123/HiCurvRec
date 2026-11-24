"""Measure the floor-repair path distribution on real captured cost matrices."""
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
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for step, (batch, item_ids) in enumerate(loader):
        if step >= steps:
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

    rows = []
    for index, group in enumerate(captured):
        for layer_index, cost in enumerate(group):
            costs = cost.detach().float()
            assignment = torch.argmin(costs, dim=-1)
            counts = torch.bincount(assignment, minlength=costs.shape[1])
            torch.cuda.synchronize()
            start = time.perf_counter()
            repaired = floor_module._strict_cpu_repair(costs.cpu().numpy(), assignment.cpu().numpy(), 1)
            cpu_seconds = time.perf_counter() - start
            empty = int((counts < 1).sum())
            rows.append({'matrix': index, 'layer': layer_index, 'empty_codewords': empty,
                         'cpu_repair_seconds': cpu_seconds,
                         'cpu_repair': 'none' if repaired is None else 'assignment'})
    directory = HERE / 'costs'
    directory.mkdir(exist_ok=True)
    for index, group in enumerate(captured):
        for layer_index, cost in enumerate(group):
            np.save(directory / ('m%02d_l%d.npy' % (index, layer_index)), cost.cpu().numpy())
    print(json.dumps({'matrices': len(rows),
                      'cpu_repair_none': sum(1 for row in rows if row['cpu_repair'] == 'none'),
                      'empty_codewords_mean': float(np.mean([row['empty_codewords'] for row in rows])),
                      'cpu_repair_seconds_median': float(np.median([row['cpu_repair_seconds'] for row in rows])),
                      'cpu_repair_seconds_max': float(np.max([row['cpu_repair_seconds'] for row in rows])),
                      'rows': rows[:12]}, indent=2))


if __name__ == '__main__':
    main()
