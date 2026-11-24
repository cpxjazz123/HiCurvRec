"""Count and time the framework assignment guards inside one training step."""
import collections
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build

COUNTS = collections.Counter()
SECONDS = collections.Counter()


def wrap(module, name):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        COUNTS[name] += 1
        SECONDS[name] += time.perf_counter() - start
        return result

    setattr(module, name, wrapper)
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

    import assignment_floor as floor_module
    for name in ('constrained_assignment', 'verify_codebook', 'verify_separation',
                 'verify_margins', 'verify_assignment', '_strict_cpu_repair'):
        wrap(floor_module, name)

    model.train()
    sampler.begin_epoch()

    def one_step(step):
        batch, item_ids = next(iter(loader))
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

    one_step(0)
    torch.cuda.synchronize()
    for key in COUNTS:
        COUNTS[key] = 0
        SECONDS[key] = 0.0
    one_step(1)
    torch.cuda.synchronize()
    for label, count in COUNTS.most_common():
        print('%5d calls  %8.4f s  %s' % (count, SECONDS[label], label))


if __name__ == '__main__':
    main()
