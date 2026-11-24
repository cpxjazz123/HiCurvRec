"""Compare quantize forward with and without autograd on real captured tensors."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build


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

    report = {}

    def timed(name, function, repeats=3):
        values = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            function()
            torch.cuda.synchronize()
            values.append(time.perf_counter() - start)
        report[name] = {'median': float(np.median(values)), 'all': [round(v, 6) for v in values]}

    detached = captured['residual']
    live = captured['residual'].clone().requires_grad_(True)
    live_weights = captured['weights'].clone().requires_grad_(True)

    timed('quantize_no_grad_input', lambda: model.quantize(detached, captured['weights'], balanced=True))
    timed('quantize_grad_input', lambda: model.quantize(live, live_weights, balanced=True))
    timed('quantize_grad_input_balanced_false',
          lambda: model.quantize(live, live_weights, balanced=False))

    def forward_backward():
        out = model.quantize(live, live_weights, balanced=True)
        loss = out[0].square().sum()
        torch.autograd.grad(loss, live, retain_graph=False)

    timed('quantize_grad_forward_backward', forward_backward)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
