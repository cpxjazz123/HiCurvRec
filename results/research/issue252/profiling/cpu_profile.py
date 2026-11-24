"""CPU-side profile of one real training step (diagnostic)."""
import cProfile
import pstats
import sys
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
    profiler = cProfile.Profile()
    profiler.enable()
    one_step(1)
    profiler.disable()
    torch.cuda.synchronize()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime').print_stats(28)


if __name__ == '__main__':
    main()
