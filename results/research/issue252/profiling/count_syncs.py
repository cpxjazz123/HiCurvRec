"""Count device-to-host synchronizations in one real training step (diagnostic)."""
import sys
import warnings
from pathlib import Path

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
    torch.cuda.set_sync_debug_mode('warn')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        one_step(1)
    torch.cuda.set_sync_debug_mode('default')
    torch.cuda.synchronize()
    messages = [str(item.message) for item in caught]
    print('python_warnings:', len(messages))
    counts = {}
    for message in messages:
        key = message.split('\n')[0][:200]
        counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items(), key=lambda item: -item[1]):
        print('%4d  %s' % (count, key))


if __name__ == '__main__':
    main()
