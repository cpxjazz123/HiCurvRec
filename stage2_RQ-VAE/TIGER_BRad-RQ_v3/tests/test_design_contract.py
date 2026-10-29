"""BRad-RQ v3: the rules of the frozen definition, checked on the real functions.

The v2 definition optimised a reconstruction of the model's own latent, and that
turned out to have a degenerate minimum: the encoder compressed every item into a
few directions, the codebook collapsed to 12 / 5 / 3 active codes, and the loss
still fell to 1.6e-05. Version 3 changes four things and this file binds each of
them to the arithmetic that actually trains:

1. the reconstruction target is the ORIGINAL Stage1 embedding, not a vector the
   encoder produces (the anti-degeneracy test at the end is the one that catches
   a silent return to the old target);
2. the codebook is initialised on the behaviour-scaled tangent the quantizer
   scores, after the radii are bound, with the real assignment rule per level;
3. the assignment is the nearest codeword in every phase -- no Sinkhorn;
4. codes that stay unused for the fixed patience window are re-seeded from the
   worst-quantised items, deterministically.

The contract is results/research/pipeline_v2/design/BRad-RQ-v3.json.
"""
import unittest
from types import SimpleNamespace

import torch

from model.behaviour_radius import rank_radii
from model.model import RQVAE


def config(use_curvature=True, radial=True, patience=2, codes=16):
    return SimpleNamespace(hidden_sizes=(32,), codebook_num=3,
                           codebook_size=(codes, codes, codes),
                           codebook_dim=8, dropout=0.0, beta=0.25, vq_type="vq",
                           ema_decay=0.99, fix_code_embs=False,
                           sk_epsilon=0.003, sk_iters=50, frechet_codebook=False,
                           use_behaviour=True, use_curvature=use_curvature,
                           frechet_eta=1.0, radial=radial, dead_code_patience=patience)


def build(use_curvature=True, items=32, dim=16, seed=0, codes=16):
    torch.manual_seed(seed)
    model = RQVAE(config(use_curvature, codes=codes), in_dim=dim)
    embeddings = torch.randn(items, dim)
    radius = torch.linspace(0.05, 1.2, items)
    model.set_radii(radius)
    with torch.no_grad():
        model.init_codebook(embeddings, item_ids=torch.arange(items))
    return model, embeddings, torch.arange(items), radius


class OriginalReconstructionTests(unittest.TestCase):
    """Rule 1: the decoder reconstructs the Stage1 embedding, an external target."""

    def test_the_reference_is_the_original_embedding(self):
        model, embeddings, ids, radius = build()
        with torch.no_grad():
            _, _, _, _, reference = model(embeddings, ids, radius=radius[ids])
        self.assertTrue(torch.equal(reference, embeddings))

    def test_the_reconstruction_has_the_input_width(self):
        model, embeddings, ids, radius = build()
        with torch.no_grad():
            recon, _, _, _, _ = model(embeddings, ids, radius=radius[ids])
        self.assertEqual(tuple(recon.shape), tuple(embeddings.shape))

    def test_the_decoder_takes_a_gradient(self):
        model, embeddings, ids, radius = build()
        recon, quant_loss, _, _, reference = model(embeddings, ids, radius=radius[ids])
        loss, _ = model.compute_loss(reference, recon, quant_loss)
        loss.backward()
        grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0 for g in grads),
                        "the decoder receives no gradient: the loss bypasses it")

    def test_the_encoder_takes_a_gradient_too(self):
        model, embeddings, ids, radius = build()
        recon, quant_loss, _, _, reference = model(embeddings, ids, radius=radius[ids])
        loss, _ = model.compute_loss(reference, recon, quant_loss)
        loss.backward()
        grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0 for g in grads))


class DegenerateTargetTests(unittest.TestCase):
    """The test that catches a quiet return to v2's target.

    Two different items that receive the same quantised representation cannot both
    be reconstructed with zero error when the target is the original embedding.
    If the target is the model's own latent, the pair can.
    """

    def test_two_items_with_one_quantised_point_cannot_both_be_reconstructed(self):
        model, embeddings, ids, radius = build()
        # force both items onto the same code at every level
        with torch.no_grad():
            for layer in model.rq.vq_layers:
                layer.embed.weight[1:] = layer.embed.weight[0:1].expand_as(layer.embed.weight[1:])
            recon, _, _, tokens, reference = model(embeddings[:2], ids[:2],
                                                   radius=radius[:2])
            self.assertEqual(len(set(map(tuple, tokens.tolist()))), 1)
            loss = torch.nn.functional.mse_loss(recon, reference)
        self.assertGreater(float(loss), 1e-6,
                           "one quantised point reconstructs two different targets: "
                           "the reconstruction target is not the original embedding")


class InitialisationTests(unittest.TestCase):
    """Rule 2: centres live where the training quantises, under the real rule."""

    def test_the_codebook_is_initialised_on_the_scored_tangent(self):
        torch.manual_seed(0)
        model = RQVAE(config(), in_dim=16)
        embeddings = torch.randn(32, 16)
        radius = torch.linspace(0.05, 1.2, 32)
        model.set_radii(radius)
        with torch.no_grad():
            model.init_codebook(embeddings, item_ids=torch.arange(32))
            encoded = model.encoder(embeddings)
            tangent = model.scale_by_radius(encoded, radius)
            first = model.rq.vq_layers[0].get_code_embs()
            distance_to_tangent = torch.cdist(first, tangent).min(dim=1).values.mean()
            distance_to_encoded = torch.cdist(first, encoded).min(dim=1).values.mean()
        self.assertLess(float(distance_to_tangent), float(distance_to_encoded))

    def test_initialisation_refuses_to_run_before_the_radii_are_bound(self):
        model = RQVAE(config(), in_dim=16)
        with self.assertRaises(RuntimeError):
            model.init_codebook(torch.randn(8, 16))

    def test_the_initial_residual_follows_the_real_assignment(self):
        model, embeddings, ids, radius = build()
        with torch.no_grad():
            tangent = model.scale_by_radius(model.encoder(embeddings), radius)
            layer = model.rq.vq_layers[0]
            expected = torch.argmin(layer.mapped_cost(layer.point_of(tangent)), dim=-1)
            chosen = layer.embed_code(expected)
        stats = model.rq.assignment_stats(tangent)
        self.assertTrue(torch.equal(stats[0]["index"], expected))
        self.assertTrue(torch.allclose(stats[0]["input"], tangent))


class AssignmentTests(unittest.TestCase):
    """Rule 3: nearest codeword in every phase; training and export agree."""

    def test_training_uses_the_nearest_codeword(self):
        model, embeddings, ids, radius = build()
        model.train()
        with torch.no_grad():
            tokens = model(embeddings, ids, radius=radius[ids])[3]
            layer = model.rq.vq_layers[0]
            target = model.scale_by_radius(model.encoder(embeddings), radius[ids])
            expected = torch.argmin(
                layer.mapped_cost(layer.point_of(target.view(-1, layer.dim))), dim=-1)
        self.assertTrue(torch.equal(tokens[:, 0], expected))

    def test_the_export_writes_the_code_the_training_chose(self):
        model, embeddings, ids, radius = build()
        model.train()
        with torch.no_grad():
            trained = model(embeddings, ids, radius=radius[ids])[3]
        model.eval()
        with torch.no_grad():
            exported = model.get_indices(embeddings, item_ids=ids)
        self.assertTrue(torch.equal(trained, exported))


class DeadCodeRecoveryTests(unittest.TestCase):
    """Rule 4: an unused code is re-seeded from the worst-quantised items."""

    def test_a_dead_code_is_replaced_and_no_live_code_is_touched(self):
        # more codes than items can occupy, so some code provably receives nothing
        model, embeddings, ids, radius = build(items=64, codes=256)
        with torch.no_grad():
            tangent = model.scale_by_radius(model.encoder(embeddings), radius)
            stats = model.rq.assignment_stats(tangent)
            layer = model.rq.vq_layers[0]
            before = layer.get_code_embs().clone()
            used = stats[0]["used"] > 0
            dead = ~used
            self.assertTrue(bool(dead.any()), "the fixture left no dead code to replace")
            count = layer.reset_dead_codes(stats[0]["input"], stats[0]["error"], dead)
            after = layer.get_code_embs()
        # at most one code per available item can be re-seeded in one pass
        self.assertEqual(count, min(int(dead.sum()), int(stats[0]["error"].numel())))
        self.assertTrue(torch.allclose(after[used], before[used]),
                        "a live code was moved by the recovery")

    def test_the_recovery_respects_the_patience_window(self):
        model, embeddings, ids, radius = build(items=64, codes=256)
        streak = [torch.zeros(layer.n_embed, dtype=torch.long) for layer in model.rq.vq_layers]
        with torch.no_grad():
            tangent = model.scale_by_radius(model.encoder(embeddings), radius)
            stats = model.rq.assignment_stats(tangent)
            first = model.recover_dead_codes(stats, streak)
            second = model.recover_dead_codes(stats, streak)
            third = model.recover_dead_codes(stats, streak)
        self.assertTrue(all(entry["codes"] == 0 for entry in first),
                        "a code reset before the patience window elapsed")
        self.assertTrue(any(entry["codes"] > 0 for entry in second) or
                        any(entry["codes"] > 0 for entry in third),
                        "a code unused for the whole window was never re-seeded")


class RadiusTests(unittest.TestCase):
    """The behaviour radius is the input the quantizer sees, and only measured items rank."""

    def test_the_scored_point_sits_at_the_behaviour_radius(self):
        model, embeddings, ids, radius = build()
        tangent = model.scale_by_radius(model.encoder(embeddings), radius[ids])
        self.assertTrue(torch.allclose(tangent.norm(dim=-1), radius[ids], atol=1e-5))

    def test_only_measured_items_are_ranked(self):
        radius = rank_radii(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 5.0, 0.0]))
        self.assertAlmostEqual(float(radius[2]), float(radius[:2].median()), places=6)

    def test_a_radial_model_refuses_to_run_without_a_radius(self):
        model, embeddings, ids, _ = build()
        with self.assertRaises(ValueError):
            model(embeddings, ids)


if __name__ == "__main__":
    unittest.main()
