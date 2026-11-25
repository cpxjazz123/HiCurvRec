"""The actual calibrator must constrain every coupled real-loss gradient."""
from itertools import islice
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DistributedSampler

TREE = Path(__file__).resolve().parent.parent
ROOT = next(p for p in TREE.parents if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(TREE))


def check_actual_calibration(device, batch_items):
    import train_rqvae as training
    from validation_entry import statistics

    torch.set_num_threads(1)
    training.set_seed(training.SEED)
    catalogue = torch.from_numpy(np.load(training.EMBEDDING_FILE).astype(np.float32)).to(device)
    train_ids = np.unique(pd.read_parquet(training.TRAIN_FILE)['target'].to_numpy(dtype=np.int64))
    sampler = DistributedSampler(train_ids, num_replicas=4, rank=0, shuffle=True, seed=42)
    positions = np.fromiter(islice(iter(sampler), batch_items), dtype=np.int64)
    ids = torch.from_numpy(train_ids[positions]).to(device)
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).to(device).train()
    # Existing frozen weights are only a fast deterministic CPU diagnostic
    # fixture, never an initialization for this candidate's formal training.
    state = torch.load(ROOT / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                       map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=False)
    model.freeze_tangent_scale(catalogue)
    sample = model.structural_sampler().draw(0, 0, 4, track=False)
    before = [p.detach().clone() for p in model.parameters()]
    measured = model.calibrate_curvature_weight(catalogue[ids], ids, catalogue,
                                                structural_sample=sample)
    reconstructed, parts, _, _, curvature = model(
        catalogue[ids], ids, catalogue_embeddings=catalogue, structural_sample=sample)
    original, _ = training.CRITERION(reconstructed, catalogue[ids], parts)
    parameters = tuple(model.parameters())
    original_g = torch.autograd.grad(original, parameters, allow_unused=True, retain_graph=True)
    curvature_g = torch.autograd.grad(curvature, parameters, allow_unused=True)
    original_stats = statistics(model, dict(zip(map(id, parameters), original_g)))
    curvature_stats = statistics(model, dict(zip(map(id, parameters), curvature_g)))
    ratios = {}
    for name, original_stat in original_stats.items():
        if name == 'decoder':
            assert curvature_stats[name]['norm'] == 0
            continue
        a, b = original_stat['norm'], curvature_stats[name]['norm']
        assert a > 0 and b > 0, 'every coupled module must have both actual-loss VJPs'
        ratios[name] = measured['lambda'] * b / a
    # R03: the learned gate owns two levels, so one query row less than the
    # three-level lineage; every remaining coupled module must still be bounded.
    assert len(ratios) == 16
    assert max(ratios.values()) <= 1 + 2e-6, ('module budget exceeded', ratios)
    assert abs(max(ratios.values()) - 1) < 2e-6, 'maximal budget must have an active bottleneck'
    assert set(measured['module_budgets']) == set(ratios)
    for parameter, prior in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), prior), 'calibration must not update parameters'


def test_actual_calibration_bounds_all_modules_not_only_encoder():
    # Full1,024-item/four-card checks run through module_budget_entry.py and the
    # actual Trainer observer. CPU256 is the smallest nonrotating-codebook
    # population and keeps this deterministic full-suite check CPU-safe.
    check_actual_calibration('cpu', 256)
