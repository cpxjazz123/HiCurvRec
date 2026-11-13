"""Injection: does the metric reach the quantiser, and does `item_ids` reach the metric?

The read-out and the two quantisation errors must be the ones the metric defines; the
assignment rule must stay the baseline's (`argmin` on every level except the last,
Sinkhorn on the last); and the geometry must add **no** parameter, buffer or state-dict
key, otherwise the DDP sync and the checkpoint of the frozen baseline would change.

The spies below are the point of the file: a two-argument metric proves the layer does
not pass rows when the metric does not ask for them, and a recording metric proves it
does pass them when it does -- with the same rows on every level.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import CODEBOOK_DIM


class RecordingMetric:
    """A metric that declares `needs_rows` and records what the layer handed it."""

    needs_rows = True

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, latent, codebook, rows):
        self.calls.append(None if rows is None else rows.clone())
        return self.inner(latent, codebook, rows)

    def quant_parts(self, latent, assigned, rows):
        return self.inner.quant_parts(latent, assigned, rows)


def two_argument_metric(latent, codebook):  # no `rows`, deliberately
    return (latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def test_the_metric_reaches_every_level_with_the_same_rows(tiny_config, tiny_metric):
    from model.layers import RQLayer

    metric = RecordingMetric(tiny_metric)
    layer = RQLayer(tiny_config, distance=metric)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(6, CODEBOOK_DIM, generator=generator)
    rows = torch.tensor([3, 0, 7, 1, 6, 2])
    layer(x, item_ids=rows)
    assert len(metric.calls) == tiny_config.codebook_num == 2
    for seen in metric.calls:
        assert torch.equal(seen, rows), "every level must read the item's own row"


def test_a_metric_without_needs_rows_is_called_with_two_arguments(tiny_config):
    from model.layers import RQLayer

    layer = RQLayer(tiny_config, distance=two_argument_metric)
    x = torch.randn(5, CODEBOOK_DIM, generator=torch.Generator().manual_seed(6))
    tokens = layer(x, item_ids=torch.arange(5))[3]
    assert tokens.shape == (5, tiny_config.codebook_num)


def test_sinkhorn_stays_on_the_last_level_only(tiny_config, tiny_metric):
    from model.layers import RQLayer

    layer = RQLayer(tiny_config, distance=tiny_metric)
    assert [vq.use_sk for vq in layer.vq_layers] == [False, True]
    assert [vq.sk_epsilon for vq in layer.vq_layers] == [-1.0, tiny_config.sk_epsilon]
    tokens = layer(torch.zeros(3, CODEBOOK_DIM), item_ids=torch.arange(3))[3]
    assert tokens.shape == (3, 2)
    assert int(tokens.max()) < max(tiny_config.codebook_size)


def test_the_quant_parts_come_from_the_metric_not_from_an_mse(tiny_config, tiny_metric,
                                                              curvature_table):
    from model.layers import RQLayer
    from model.per_item_curvature_geometry import PerItemCurvatureGeometry

    layer = RQLayer(tiny_config, distance=tiny_metric)
    x = torch.randn(4, CODEBOOK_DIM, generator=torch.Generator().manual_seed(7)) * 0.2
    rows = torch.arange(4)
    tokens = layer(x, item_ids=rows)[3]

    # rebuild the assignment by hand and compare with the geometric errors
    geometry = PerItemCurvatureGeometry(curvature_table)
    residual = x
    for level, vq in enumerate(layer.vq_layers):
        indices = tokens[:, level]
        assigned = vq.embed_code(indices)
        if level == 0:
            _, parts, _, _ = layer(residual, item_ids=rows)
            want = geometry.quant_parts(residual, assigned, rows)
            torch.testing.assert_close(parts[0][0], want[0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(parts[0][1], want[1], rtol=1e-5, atol=1e-6)
        residual = residual - assigned


def test_a_two_argument_metric_keeps_the_baseline_mse_pair(tiny_config):
    from losses.codebook import codebook_loss
    from losses.commitment import commitment_loss
    from model.layers import VQLayer

    layer = VQLayer(5, CODEBOOK_DIM, beta=0.25, distance=two_argument_metric)
    x = torch.randn(4, CODEBOOK_DIM, generator=torch.Generator().manual_seed(8))
    quantized, parts, _, indices = layer(x)
    assigned = F.embedding(indices, layer.get_code_embs())
    torch.testing.assert_close(parts[0], commitment_loss(x, assigned), rtol=0.0, atol=0.0)
    torch.testing.assert_close(parts[1], codebook_loss(x, assigned), rtol=0.0, atol=0.0)


def test_a_metric_that_needs_rows_but_gets_none_stops(tiny_config, tiny_metric):
    from model.layers import RQLayer

    layer = RQLayer(tiny_config, distance=tiny_metric)
    with pytest.raises(ValueError, match="rows"):
        layer(torch.zeros(4, CODEBOOK_DIM))


@pytest.mark.parametrize("vq_type", ["ema", "simvq"])
def test_the_other_vq_branches_are_refused(tiny_config, tiny_metric, vq_type):
    from model.layers import RQLayer

    tiny_config.vq_type = vq_type
    with pytest.raises(ValueError, match="vq_type"):
        RQLayer(tiny_config, distance=tiny_metric)


def test_init_codebook_threads_the_rows_and_moves_the_entries(tiny_config, tiny_metric):
    from model.layers import RQLayer

    metric = RecordingMetric(tiny_metric)
    layer = RQLayer(tiny_config, distance=metric)
    x = torch.randn(12, CODEBOOK_DIM, generator=torch.Generator().manual_seed(9))
    rows = torch.arange(12)
    before = [vq.get_code_embs().detach().clone() for vq in layer.vq_layers]
    residual = layer.init_codebook(x, torch.device("cpu"), rows)
    assert residual.shape == x.shape
    assert len(metric.calls) == tiny_config.codebook_num
    for seen in metric.calls:
        assert torch.equal(seen, rows)
    for old, vq in zip(before, layer.vq_layers):
        assert not torch.equal(old, vq.get_code_embs().detach())


def test_the_quantized_side_is_reported_per_level_not_summed(tiny_config, tiny_metric):
    from losses.quantization import quantization_total

    layer = __import__("model.layers", fromlist=["RQLayer"]).RQLayer(tiny_config,
                                                                    distance=tiny_metric)
    x = torch.randn(4, CODEBOOK_DIM, generator=torch.Generator().manual_seed(10))
    quantized, parts, unused, tokens = layer(x, item_ids=torch.arange(4))
    assert len(parts) == tiny_config.codebook_num
    assert quantized.shape == x.shape
    assert isinstance(unused, (int, float))
    total = quantization_total(parts, tiny_config.beta)
    assert total.shape == ()
    assert torch.isfinite(total)


def test_the_model_adds_no_parameter_and_no_state_dict_key(tiny_config, tiny_metric):
    from distances import build_distance
    from model.model import RQVAE

    geometric = RQVAE(tiny_config, in_dim=CODEBOOK_DIM, distance=tiny_metric)
    baseline = RQVAE(tiny_config, in_dim=CODEBOOK_DIM, distance=build_distance())
    assert [name for name, _ in geometric.named_parameters()] == \
           [name for name, _ in baseline.named_parameters()]
    assert sum(p.numel() for p in geometric.parameters()) == \
           sum(p.numel() for p in baseline.parameters())
    assert sorted(geometric.state_dict()) == sorted(baseline.state_dict())


def test_the_model_carries_item_ids_through_forward_and_get_indices(tiny_config, tiny_metric):
    from model.model import RQVAE

    model = RQVAE(tiny_config, in_dim=CODEBOOK_DIM, distance=tiny_metric)
    embeddings = torch.randn(7, CODEBOOK_DIM, generator=torch.Generator().manual_seed(12))
    rows = torch.tensor([6, 0, 5, 1, 4, 2, 3])
    reconstructed, parts, _, tokens = model(embeddings, item_ids=rows)
    assert reconstructed.shape == embeddings.shape
    assert len(parts) == tiny_config.codebook_num
    torch.testing.assert_close(model.get_indices(embeddings, item_ids=rows), tokens)
    with pytest.raises(ValueError, match="rows"):
        model.get_indices(embeddings)


def test_one_backward_reaches_the_encoder_the_codebook_and_the_decoder(tiny_config,
                                                                       tiny_metric):
    from losses.reconstruction import reconstruction_loss
    from losses.quantization import quantization_total
    from model.model import RQVAE

    model = RQVAE(tiny_config, in_dim=CODEBOOK_DIM, distance=tiny_metric)
    embeddings = torch.randn(8, CODEBOOK_DIM, generator=torch.Generator().manual_seed(13))
    rows = torch.arange(8)
    reconstructed, parts, _, _ = model(embeddings, item_ids=rows)
    loss = reconstruction_loss(reconstructed, embeddings) + quantization_total(
        parts, tiny_config.beta)
    loss.backward()
    encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters())
    decoder_grad = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters())
    codebook_grad = sum(float(vq.get_code_embs().grad.abs().sum())
                        for vq in model.rq.vq_layers)
    assert encoder_grad > 0, "the reconstruction term must reach the encoder"
    assert decoder_grad > 0, "the reconstruction term must reach the decoder"
    assert codebook_grad > 0, "the codebook term must reach the entries"
    for vq in model.rq.vq_layers:
        assert vq.embed.weight.grad is not None
        assert torch.isfinite(vq.embed.weight.grad).all()
