"""Calibrate against the actual full original objective, not reconstruction only."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
ROOT = next(p for p in TREE.parents if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(TREE))


def test_lambda_balances_full_original_encoder_gradient_on_real_training_data():
    import train_rqvae as training
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        training.set_seed(training.SEED)
        catalogue = torch.from_numpy(np.load(training.EMBEDDING_FILE).astype(np.float32))
        train_ids = np.unique(pd.read_parquet(training.TRAIN_FILE)['target'].to_numpy(dtype=np.int64))
        ids = training._calibration_ids(train_ids, 'cpu')
        model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                                distance=training.DISTANCE).train()
        initial = torch.load(ROOT / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                             map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(initial, strict=False)
        model.freeze_tangent_scale(catalogue)
        sample = model.structural_sampler().draw(0, 0, 4, track=False)
        # This regression isolates the calibration numerator with positive
        # frozen unit scales. Full catalogue scale estimation belongs to the
        # real-entry/four-rank diagnostic, not this real-batch VJP check.
        batch = catalogue[ids]
        reconstructed, parts, _, _ = model(batch, ids)
        original, components = training.CRITERION(reconstructed, batch, parts)
        parameters = tuple(model.encoder.parameters())
        original_g = torch.autograd.grad(original, parameters, retain_graph=True)
        reconstruction_g = torch.autograd.grad(components['reconstruction'], parameters)
        curvature = model.sampled_curvature(catalogue, sample)
        curvature_g = torch.autograd.grad(curvature, parameters)

        def norm(gradients):
            return float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())

        original_norm = norm(original_g)
        reconstruction_norm = norm(reconstruction_g)
        curvature_norm = norm(curvature_g)
        # This real batch distinguishes the two possible numerators.
        assert abs(original_norm - reconstruction_norm) > 0.01 * original_norm
        before = [p.detach().clone() for p in model.parameters()]
        measured = model.calibrate_curvature_weight(batch, ids, catalogue,
                                                    structural_sample=sample)
        assert measured['lambda'] == pytest.approx(original_norm / curvature_norm, rel=1.e-6)
        assert measured['original_gradient'] == pytest.approx(original_norm, rel=1.e-6)
        assert measured['reconstruction_gradient'] == pytest.approx(reconstruction_norm, rel=1.e-6)
        assert measured['curvature_gradient'] == pytest.approx(curvature_norm, rel=1.e-6)
        for parameter, prior in zip(model.parameters(), before):
            assert torch.equal(parameter.detach(), prior), 'calibration must not update parameters'
    finally:
        torch.set_num_threads(previous_threads)
