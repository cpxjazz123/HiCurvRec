"""Interface tests: the residual stack and `RQVAE` carry the mechanism end to end.

These are the checks that would catch a mechanism written but never wired:

* every level gets its OWN metric view (`g` and `lambda_l` are per level), and the
  assignment rule is untouched -- `argmin` on the inner levels, Sinkhorn on the last;
* the inner-level assignment really uses `C = d + lambda_l H`, not `d`;
* `init_codebook` runs `codeword_potential_init` + `lambda_calibration` on every level;
* the structural payload carries `s_i`, `token_l(i)` and the module that owns `g`, so
  `L_struct` can be computed and its gradient lands on `g` and nowhere near the encoder;
* a model built without a potential is the frozen baseline: no payload, baseline cost.
"""

from __future__ import annotations

import pytest
import torch

from distances import build_distance
from distances.af3_offset import Af3Offset
from losses.structural import structural_loss
from model.codeword_potential import CodewordPotential
from model.model import RQVAE

IN_DIM = 6
NUM_ITEMS = 24


def _build(config, potential, in_dim: int = IN_DIM):
    metric = build_distance(None, "af3_offset", base=build_distance(), potential=potential)
    model = RQVAE(config, in_dim=in_dim, distance=metric, potential=potential)
    return model, metric


def _batch(num_items: int = NUM_ITEMS, in_dim: int = IN_DIM):
    generator = torch.Generator().manual_seed(11)
    embeddings = torch.randn(num_items, in_dim, generator=generator)
    ids = torch.arange(num_items)
    return embeddings, ids


def test_every_level_gets_its_own_metric_view(tiny_config, tiny_potential):
    model, metric = _build(tiny_config, tiny_potential)
    views = model.rq.level_distances
    assert len(views) == tiny_config.codebook_num
    assert all(isinstance(view, type(metric.at_level(0))) for view in views)
    assert views[0] is not views[1]                       # no shared "current level"
    assert [view.level for view in views] == [0, 1]
    assert all(view.potential is tiny_potential for view in views)


def test_only_the_last_level_uses_sinkhorn(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    layers = model.rq.vq_layers
    assert layers[-1].use_sk is True
    assert all(layer.use_sk is False for layer in layers[:-1])
    # the metric type moved, the assignment rule did not
    assert model.rq.codebook_num == tiny_config.codebook_num


def test_init_codebook_calibrates_every_level(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    assert bool(tiny_potential.initialised.all())
    assert torch.isfinite(tiny_potential.lambdas).all()
    metrics = model.structural_metrics()
    for level in range(tiny_config.codebook_num):
        assert "den_level%d" % (level + 1) in metrics
        assert "lambda_level%d" % (level + 1) in metrics


def test_inner_level_assignment_uses_the_af3_cost_not_the_plain_distance(tiny_config, tiny_potential):
    model, metric = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    model.train()
    with torch.no_grad():
        encoded = model.encoder(embeddings)
        _, _, _, tokens = model.rq(encoded, item_ids=ids)
        codes = model.rq.vq_layers[0].get_code_embs()
        cost = metric.at_level(0)(encoded, codes, ids)
        plain = build_distance()(encoded, codes)
        assert torch.equal(tokens[:, 0], torch.argmin(cost, dim=-1))
        # and the offset is not a no-op here, so the equality above is not vacuous
        assert not torch.equal(cost, plain)


def test_forward_with_structural_returns_the_payload(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    reconstructed, quant_parts, unused, tokens, payload = model.forward_with_structural(embeddings, ids)
    assert reconstructed.shape == embeddings.shape
    assert len(quant_parts) == tiny_config.codebook_num
    assert tokens.shape == (NUM_ITEMS, tiny_config.codebook_num)
    assert payload["potentials"].shape == (NUM_ITEMS,)
    assert torch.equal(payload["potentials"], tiny_potential.potentials_for(ids))
    assert payload["codeword"] is tiny_potential
    assert torch.equal(payload["tokens"], tokens)


def test_forward_keeps_the_baseline_signature(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    out = model(embeddings, ids)
    assert len(out) == 4
    reconstructed, quant_parts, unused, tokens = out
    assert isinstance(unused, int)
    with pytest.raises(ValueError):
        # without ids the cost cannot be indexed by item, so the model refuses
        model.forward_with_structural(embeddings)


def test_structural_loss_trains_g_and_not_the_encoder(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    model.train()
    _, quant_parts, _, _, payload = model.forward_with_structural(embeddings, ids)
    loss = structural_loss(payload["potentials"], payload["tokens"], payload["codeword"])
    loss.backward()
    assert tiny_potential.g[0].grad is not None
    assert tiny_potential.g[0].grad.abs().sum() > 0
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert all(grad is None or torch.count_nonzero(grad) == 0 for grad in encoder_grads)
    assert tiny_potential.s.grad is None


def test_the_quantisation_term_is_unchanged_by_the_mechanism(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    model.train()
    _, quant_parts, _, _, _ = model.forward_with_structural(embeddings, ids)
    # one (commitment, codebook) pair per level, both finite scalars (the shapes the
    # baseline's `losses/quantization.py` expects)
    for commitment, codebook_term in quant_parts:
        assert commitment.numel() == 1 and torch.isfinite(commitment)
        if codebook_term is not None:
            assert codebook_term.numel() == 1 and torch.isfinite(codebook_term)


def test_a_model_without_a_potential_is_the_frozen_baseline(tiny_config):
    model = RQVAE(tiny_config, in_dim=IN_DIM, distance=build_distance())
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    reconstructed, quant_parts, _, tokens, payload = model.forward_with_structural(embeddings, ids)
    assert payload is None
    assert model.structural_metrics() == {}
    assert tokens.shape == (NUM_ITEMS, tiny_config.codebook_num)
    assert reconstructed.shape == embeddings.shape
    assert len(quant_parts) == tiny_config.codebook_num


def test_g_is_a_module_parameter_so_ddp_synchronises_it(tiny_config, tiny_potential):
    model, _ = _build(tiny_config, tiny_potential)
    names = dict(model.named_parameters())
    assert any(name.startswith("potential.g.") for name in names)
    modules = dict(model.named_modules())
    assert "potential" in modules and modules["potential"] is tiny_potential
    # the calibration numbers are buffers, not parameters: nothing trains them
    assert not any("lambdas" in name for name in names)


def test_indices_helper_accepts_the_item_ids(tiny_config, tiny_potential):
    model, metric = _build(tiny_config, tiny_potential)
    embeddings, ids = _batch()
    model.init_codebook(embeddings, ids)
    model.eval()
    indices = model.get_indices(embeddings, item_ids=ids)
    assert indices.shape == (NUM_ITEMS, tiny_config.codebook_num)
    assert isinstance(metric, Af3Offset)
