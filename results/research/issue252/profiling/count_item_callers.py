"""Attribute device synchronizations to their Python callers (diagnostic)."""
import collections
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build

CALLERS = collections.Counter()
ORIGINAL_ITEM = torch.Tensor.item
ORIGINAL_NONZERO = torch.Tensor.nonzero
ORIGINAL_BINCOUNT = torch.bincount


def site():
    stack = traceback.extract_stack()[:-1]
    return ' <- '.join('%s:%d' % (Path(frame.filename).name, frame.lineno)
                        for frame in reversed(stack[-3:]))


def install():
    def item(self, *args, **kwargs):
        CALLERS['item ' + site()] += 1
        return ORIGINAL_ITEM(self, *args, **kwargs)

    def nonzero(self, *args, **kwargs):
        CALLERS['nonzero ' + site()] += 1
        return ORIGINAL_NONZERO(self, *args, **kwargs)

    def bincount(self, *args, **kwargs):
        CALLERS['bincount ' + site()] += 1
        return ORIGINAL_BINCOUNT(self, *args, **kwargs)

    torch.Tensor.item = item
    torch.Tensor.nonzero = nonzero
    torch.bincount = bincount


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
    install()
    one_step(1)
    torch.cuda.synchronize()
    for label, count in CALLERS.most_common(25):
        print('%5d  %s' % (count, label))


if __name__ == '__main__':
    main()
