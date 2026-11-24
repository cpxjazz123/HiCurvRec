"""Count the Python-level operations that synchronize with the device (diagnostic)."""
import collections
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_context import build

COUNTS = collections.Counter()


def wrap(owner, name, label):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        COUNTS[label] += 1
        return original(*args, **kwargs)

    setattr(owner, name, wrapper)


def install():
    wrap(torch.Tensor, 'item', 'Tensor.item')
    wrap(torch.Tensor, 'tolist', 'Tensor.tolist')
    wrap(torch.Tensor, 'cpu', 'Tensor.cpu')
    wrap(torch.Tensor, 'numpy', 'Tensor.numpy')
    wrap(torch.Tensor, '__bool__', 'Tensor.__bool__')
    wrap(torch.Tensor, '__float__', 'Tensor.__float__')
    wrap(torch.Tensor, '__int__', 'Tensor.__int__')
    wrap(torch.Tensor, 'nonzero', 'Tensor.nonzero')
    wrap(torch.Tensor, 'unique', 'Tensor.unique')
    wrap(torch, 'bincount', 'torch.bincount')
    wrap(torch, 'repeat_interleave', 'torch.repeat_interleave')
    wrap(torch, 'arange', 'torch.arange')
    wrap(torch, 'sparse_csr_tensor', 'torch.sparse_csr_tensor')
    wrap(torch.sparse, 'mm', 'torch.sparse.mm')
    wrap(torch, 'argsort', 'torch.argsort')
    wrap(torch, 'sort', 'torch.sort')
    wrap(torch, 'topk', 'torch.topk')
    wrap(torch, 'where', 'torch.where')


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
    for label, count in COUNTS.most_common():
        print('%5d  %s' % (count, label))


if __name__ == '__main__':
    main()
