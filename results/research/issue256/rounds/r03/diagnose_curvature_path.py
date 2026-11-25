"""Find which encoder parameter lost its curvature-gradient path under R03."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TREE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_role_conditioned_l1_rq')
sys.path.insert(0, str(TREE))

import train_rqvae as training


def main() -> int:
    torch.set_num_threads(1)
    training.set_seed(training.SEED)
    catalogue = torch.from_numpy(np.load(training.EMBEDDING_FILE).astype(np.float32))
    train_ids = np.unique(pd.read_parquet(training.TRAIN_FILE)['target'].to_numpy(dtype=np.int64))
    ids = torch.from_numpy(train_ids[:256])
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                            distance=training.DISTANCE).train()
    initial = torch.load('/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                         map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(initial, strict=False)
    model.freeze_tangent_scale(catalogue)
    sample = model.structural_sampler().draw(0, 0, 4, track=False)
    batch = catalogue[ids]
    reconstructed, parts, _, _, curvature = model(batch, ids, catalogue_embeddings=catalogue,
                                                  structural_sample=sample)
    original, components = training.CRITERION(reconstructed, batch, parts)
    names = [name for name, _ in model.encoder.named_parameters()]
    parameters = tuple(model.encoder.parameters())
    for label, value in (('original', original), ('reconstruction', components['reconstruction']),
                         ('curvature', curvature)):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        unused = [name for name, gradient in zip(names, gradients) if gradient is None]
        print(label, 'unused encoder parameters:', unused)
    print('curvature requires grad:', curvature.requires_grad,
          'grad_fn:', type(curvature.grad_fn).__name__)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
