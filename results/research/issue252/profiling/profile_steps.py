"""Per-phase attribution of one real training step (diagnostic only)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build


def timed(name, module, attribute, totals):
    original = getattr(module, attribute)

    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        totals[name] = totals.get(name, 0.0) + time.perf_counter() - start
        return result

    setattr(module, attribute, wrapper)
    return original


def main():
    context = build()
    model = context['model']
    loader = context['loader']
    optimizer = context['optimizer']
    catalogue = context['catalogue']
    sampler = context['sampler']
    device = context['device']
    cand = context['cand']

    totals = {}
    timed('gates_for_items', model.structure, 'gates_for_items', totals)
    timed('sampled_curvature', model, 'sampled_curvature', totals)
    timed('encoder', model.encoder, 'forward', totals)
    timed('quantize', model, 'quantize', totals)
    timed('decoder', model.decoder, 'forward', totals)

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model.train()
    sampler.begin_epoch()
    step_seconds = []
    cpu_seconds = []
    for step, (batch, item_ids) in enumerate(loader):
        if step >= warmup + steps:
            break
        if step == warmup:
            totals.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        draw_start = time.perf_counter()
        sample = sampler.draw(step, 0, 1)
        totals['sampler_draw'] = totals.get('sampler_draw', 0.0) + time.perf_counter() - draw_start
        optimizer.zero_grad(set_to_none=True)
        reconstructed, quant_parts, _, _, curvature_loss = model(
            batch, item_ids, catalogue_embeddings=catalogue, structural_sample=sample)
        original_loss, components = cand.CRITERION(reconstructed, batch, quant_parts)
        loss = model.combined_loss(original_loss, curvature_loss)
        loss.backward()
        optimizer.step()
        cpu_elapsed = time.perf_counter() - start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if step >= warmup:
            step_seconds.append(time.perf_counter() - start)
            cpu_seconds.append(cpu_elapsed)

    report = {'steps': len(step_seconds), 'batch_size_per_rank': cand.BATCH_SIZE_PER_RANK,
              'train_items': int(len(context['train_ids'])),
              'mean_step_seconds': float(np.mean(step_seconds)),
              'mean_cpu_launch_seconds': float(np.mean(cpu_seconds)),
              'phases_seconds': {k: float(v) for k, v in sorted(totals.items())}}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
