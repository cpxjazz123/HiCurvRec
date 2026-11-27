"""Calibration peeks the actual first loader batch without perturbing its RNG."""
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))


@pytest.mark.parametrize('mode', ['random', 'distributed', 'explicit_generator'])
def test_first_loader_batch_and_rng_survive_calibration_peek(mode):
    import train_rqvae as training
    torch.manual_seed(42)
    ids = torch.arange(8192)
    features = ids.float().reshape(-1, 1)
    dataset = training._BatchedTensorDataset(features, ids)
    sampler = (DistributedSampler(dataset, num_replicas=4, rank=0, seed=42)
               if mode == 'distributed' else None)
    generator = torch.Generator().manual_seed(42) if mode == 'explicit_generator' else None
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler,
                        shuffle=sampler is None, generator=generator,
                        collate_fn=training._batch_passthrough)
    state = torch.random.get_rng_state()
    generator_state = generator.get_state() if generator is not None else None
    with torch.random.fork_rng(devices=[]):
        expected_batch, expected_ids = next(iter(loader))
    if generator is not None:
        generator.set_state(generator_state)
    actual_batch, actual_ids = training._calibration_batch(loader, 'cpu')
    assert torch.equal(actual_ids, expected_ids), 'calibration must use the actual first loader population'
    assert torch.equal(actual_batch, expected_batch), 'calibration features must be the actual first batch'
    assert torch.equal(torch.random.get_rng_state(), state), 'calibration must restore global training RNG'
    if generator is not None:
        assert torch.equal(generator.get_state(), generator_state), 'calibration must restore explicit loader RNG'
    next_batch, next_ids = next(iter(loader))
    assert torch.equal(next_ids, expected_ids), 'calibration peek must not change the first training population'
    assert torch.equal(next_batch, expected_batch)
