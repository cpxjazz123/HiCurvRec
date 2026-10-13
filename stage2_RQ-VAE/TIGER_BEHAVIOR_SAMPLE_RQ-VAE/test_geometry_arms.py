"""Trainer-level regression tests for the fixed-capacity geometry arms."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


STAGE2 = Path(__file__).resolve().parent
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))

import train_rqvae as trainer
import geom_losses
from arm_geometry import ARM_CONFIGS
from model import RQVAE, VQLayer


ARMS = ("E0", "H1", "E1")


class GeometryTrainerArmTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        embeddings = np.load(root / "stage1_GeneEmbedding/output/sentence_t5.npy")
        with np.load(root / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz") as pairs:
            item_ids = np.concatenate((
                pairs["anchors"][:4],
                pairs["positives"][:4],
                pairs["negatives"][:4, 0],
            )).astype(np.int64)
        self.item_pool = torch.from_numpy(embeddings[item_ids].copy())
        self.pairs = self.item_pool[:8]
        self.source_ids = torch.arange(4)
        self.target_ids = torch.arange(4, 8)
        self.negative_ids = torch.arange(8, 12)
        self.model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        checkpoint = (
            root / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/"
            "resample/rqvae_best.pth"
        )
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "rqvae_best.pth"
            shutil.copy2(checkpoint, staged)
            saved = torch.load(staged, map_location="cpu")
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()

    def test_t1_arms_resolve_identical_capacity_training_and_behavior(self):
        shared = (
            trainer.CODEBOOK_DIM, trainer.CODEBOOK_SIZE, trainer.HIDDEN_SIZES,
            trainer.EPOCHS, trainer.LR, trainer.OPTIMIZER,
            trainer.BATCH_SIZE_PER_RANK, trainer.BETA, trainer.BEHAVIOR_WEIGHT_MAX,
            trainer.BEHAVIOR_TEMPERATURE, trainer.RAMP_START_EPOCH,
            trainer.PAIR_BATCH_SIZE_PER_RANK, trainer.RESAMPLE_PAIRS,
            trainer.FALSE_NEGATIVE_MASK, trainer.SEED,
        )
        geometries = []
        for arm in ARMS:
            resolved = trainer._resolve_geometry_arm(arm)
            self.assertEqual(resolved["shared"], shared)
            geometries.append(resolved["geometry"])
        self.assertEqual(geometries, [ARM_CONFIGS[arm] for arm in ARMS])
        self.assertNotEqual(geometries[0], geometries[1])
        self.assertNotEqual(geometries[1], geometries[2])

    def test_t2_zero_geometry_weight_preserves_behavior_bit_identically(self):
        residuals = trainer._residuals_with_trace(self.model, self.pairs[:8])
        behavior = trainer.behavior_contrastive_loss(
            residuals, self.source_ids, self.target_ids
        )
        geometry_pool = self.item_pool
        latent = self.model.encoder(geometry_pool)
        codebook = self.model.rq.vq_layers[0].get_code_embs()
        for arm in ARMS:
            term = trainer._geometry_loss_for_pairs(
                arm, latent, self.source_ids, codebook, weight=0.0,
            )
            self.assertTrue(torch.equal(trainer._compose_geometry_loss(behavior, term), behavior))

    def test_t3_geometry_uses_same_indices_and_reports_encoder_codebook_norms(self):
        negative_pools = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14],
                                       [1, 3, 5, 7, 9, 11, 13, 15]])
        item_buckets = torch.tensor([0, 1] * 8)
        source_ids = torch.arange(8)
        target_ids = torch.arange(8, 16)
        torch.manual_seed(42)
        sampled = trainer._matched_geometry_negative_ids(
            source_ids, target_ids, negative_pools, item_buckets,
        )
        self.assertTrue(torch.equal(item_buckets[sampled], item_buckets[source_ids]))
        self.assertTrue(bool((sampled != source_ids).all()))
        self.assertTrue(bool((sampled != target_ids).all()))
        codebook = self.model.rq.vq_layers[0].get_code_embs()
        latent = self.model.encoder(self.item_pool)
        calls = []
        original = trainer._geometry_loss

        latent_norms = torch.linalg.vector_norm(latent.detach(), dim=-1)
        print("T3 real latent_norm_mean={:.8g} min={:.8g} max={:.8g}".format(
            float(latent_norms.mean()), float(latent_norms.min()),
            float(latent_norms.max()),
        ))
        self.assertGreater(float(latent_norms.mean()), 0.3)
        self.assertLess(float(latent_norms.mean()), 0.6)
        def capture(arm, z, book, anchors, positives, negatives, **kwargs):
            calls.append((anchors.clone(), positives.clone(), negatives.clone()))
            return original(arm, z, book, anchors, positives, negatives, **kwargs)

        trainer._geometry_loss = capture
        try:
            for arm in ARMS:
                term = trainer._geometry_loss_for_pairs(
                    arm, latent, self.source_ids,
                    codebook, weight=ARM_CONFIGS[arm]["weight"],
                )
                gradients = torch.autograd.grad(
                    term, tuple(self.model.encoder.parameters()) + (codebook,),
                    retain_graph=True, allow_unused=True,
                )
                encoder_gradients, codebook_gradient = gradients[:-1], gradients[-1]
                encoder_norm = float(torch.sqrt(sum(
                    gradient.detach().square().sum()
                    for gradient in encoder_gradients if gradient is not None
                )))
                codebook_norm = 0.0 if codebook_gradient is None else float(
                    torch.linalg.vector_norm(codebook_gradient)
                )
                reported = trainer._geometry_gradient_norms(term, self.model)
                print("T3 {} encoder_grad_norm={:.8g} codebook_grad_norm={:.8g}".format(
                    arm, encoder_norm, codebook_norm
                ))
                self.assertGreater(encoder_norm, 0.0)
                self.assertAlmostEqual(reported["encoder"], encoder_norm, delta=1e-15)
                self.assertEqual(codebook_norm, 0.0)
                self.assertAlmostEqual(reported["codebook"], codebook_norm, delta=1e-15)
        finally:
            trainer._geometry_loss = original
        self.assertEqual(len(calls), 3)
        for indices in zip(*calls):
            self.assertTrue(torch.equal(indices[0], indices[1]))
            self.assertTrue(torch.equal(indices[0], indices[2]))
        self.assertTrue(torch.equal(calls[0][0], torch.arange(4)))
        self.assertTrue(torch.equal(calls[0][1], torch.arange(4, 8)))
        self.assertTrue(torch.equal(calls[0][2], torch.arange(8, 12)))

    def test_t4_geometry_loss_does_not_change_vq_forward_outputs(self):
        layer = VQLayer(codebook_size=16, codebook_dim=32)
        latent = torch.randn(8, 32) * 0.08
        codebook = layer.get_code_embs()
        with torch.no_grad():
            before = layer(latent)
        for arm in ARMS:
            geometry = geom_losses.geometry_loss(
                arm, torch.cat((latent[:4], latent[4:], latent[:4])), codebook,
                torch.arange(4), torch.arange(4, 8), torch.arange(8, 12),
                curvature=1.0, weight=ARM_CONFIGS[arm]["weight"],
            )
            self.assertTrue(torch.isfinite(geometry))
            with torch.no_grad():
                after = layer(latent)
            for before_value, after_value in zip(before, after):
                if isinstance(before_value, torch.Tensor):
                    self.assertTrue(torch.equal(before_value, after_value))
                else:
                    self.assertEqual(before_value, after_value)

    def test_t5_each_arm_has_a_distinct_output_directory(self):
        output_dirs = [trainer._resolve_geometry_arm(arm)["output_dir"] for arm in ARMS]
        self.assertEqual(len(set(output_dirs)), len(ARMS))

    def test_t6_geometry_arms_are_deterministic_at_seed_42(self):
        for arm in ARMS:
            def run():
                torch.manual_seed(42)
                z = torch.randn(8, 32) * 0.08
                book = torch.randn(16, 32) * 0.08
                return geom_losses.geometry_loss(
                    arm, z, book, torch.arange(4), torch.arange(4, 8),
                    torch.arange(4), curvature=1.0, weight=ARM_CONFIGS[arm]["weight"],
                )

            self.assertTrue(torch.equal(run(), run()), arm)


if __name__ == "__main__":
    unittest.main()
