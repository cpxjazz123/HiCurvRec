"""The quantiser wiring: which assignment runs, where `p` enters, which gradient moves.

Two groups again:

* statements that already hold in the copied tree and must keep holding -- the last
  level is the only Sinkhorn level, Sinkhorn is NOT replaced by a plain argmin, and the
  mechanism adds no trainable parameter;
* the interface stated by `4_method.modified_files` (share + item ids through
  `RQLayer`/`RQVAE`, `vq_type == 'vq'` asserted at construction). Those fail until that
  wiring is written: the failing test is the specification, and a green run without it
  would mean the mechanism never reached the quantiser.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

import factor_share_geometry as geometry
from distances import factor_share as adapter
from model.layers import RQLayer, VQLayer
from model.model import RQVAE


def tokenizer_config(**overrides):
    values = dict(
        hidden_sizes=[8, 4],
        codebook_num=2,
        codebook_size=(6, 6),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=50,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------------------
# the baseline assignment contract, which the mechanism must not change
# --------------------------------------------------------------------------------------

def test_only_the_last_level_carries_a_sinkhorn_temperature():
    layer = RQLayer(tokenizer_config(codebook_num=3, codebook_size=(4, 4, 4)),
                    distance=adapter.distances)
    assert [vq.use_sk for vq in layer.vq_layers] == [False, False, True]
    assert layer.vq_layers[-1].sk_epsilon == 0.003
    assert layer.vq_layers[-1].sk_iters == 50


def test_sinkhorn_is_not_replaced_by_a_plain_argmin():
    """On a distance matrix where the balanced assignment differs from argmin, the layer
    must follow Sinkhorn. A rewrite that scores rows with argmin fails here."""
    generator = torch.Generator().manual_seed(7)
    found = None
    for _ in range(500):
        distances = torch.randint(0, 4, (4, 3), generator=generator).double()
        centered = VQLayer.center_distance(distances).double()
        # `sinkhorn` is an instance method; calling it off the class would bind
        # `centered` to `self` and pass `1.0` as the distance matrix (TypeError).
        probe = VQLayer(codebook_size=3, codebook_dim=2, beta=0.25,
                        sk_epsilon=1.0, sk_iters=20)
        assignment = probe.sinkhorn(centered, 1.0, 20)
        if not torch.isfinite(assignment).all():
            continue
        sinkhorn_argmax = assignment.argmax(dim=-1)
        if not torch.equal(sinkhorn_argmax, distances.argmin(dim=-1)):
            found = (distances, sinkhorn_argmax)
            break
    assert found is not None, "no discriminating distance matrix was found"

    layer = VQLayer(codebook_size=3, codebook_dim=2, beta=0.25, sk_epsilon=1.0, sk_iters=20)
    layer.train()
    tokens = layer._indices(found[0], False)
    assert torch.equal(tokens, found[1])
    assert not torch.equal(tokens, found[0].argmin(dim=-1))


def test_the_mechanism_adds_no_trainable_parameter():
    torch.manual_seed(0)
    geometry_model = RQVAE(tokenizer_config(), in_dim=8, distance=adapter.distances)
    torch.manual_seed(0)
    euclidean_model = RQVAE(tokenizer_config(), in_dim=8, distance=None)
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(geometry_model) == count(euclidean_model)
    assert not any("share" in name for name, _ in geometry_model.named_parameters())


# --------------------------------------------------------------------------------------
# the modified-file interface (implemented: `4_method.modified_files`, `model/layers.py`)
# --------------------------------------------------------------------------------------

def test_rq_layer_forward_takes_the_per_item_share():
    parameters = list(inspect.signature(RQLayer.forward).parameters)
    assert "p" in parameters
    parameters = list(inspect.signature(VQLayer.forward).parameters)
    assert "p" in parameters


def test_the_share_changes_the_assignment():
    """If `p` were ignored, the two token sequences below would be identical."""
    torch.manual_seed(0)
    config = tokenizer_config(sk_epsilon=-1.0, sk_iters=-1)  # argmin everywhere
    layer = RQLayer(config, distance=adapter.distances)
    layer.eval()
    latent = torch.randn(8, geometry.D_TOTAL) * 0.4
    low = torch.full((8,), 0.05)
    high = torch.full((8,), 0.95)
    with torch.no_grad():
        _, _, _, tokens_low = layer(latent, p=low)
        _, _, _, tokens_high = layer(latent, p=high)
    assert not torch.equal(tokens_low, tokens_high)


def test_rq_layer_refuses_a_missing_share():
    layer = RQLayer(tokenizer_config(sk_epsilon=-1.0, sk_iters=-1),
                    distance=adapter.distances)
    layer.eval()
    with pytest.raises(ValueError):
        layer(torch.randn(4, geometry.D_TOTAL))


def test_model_entry_points_require_item_ids():
    model = RQVAE(tokenizer_config(sk_epsilon=-1.0, sk_iters=-1), in_dim=8,
                  distance=adapter.distances)
    embeddings = torch.randn(16, 8)
    with pytest.raises(ValueError):
        model(embeddings)
    with pytest.raises(ValueError):
        model.get_indices(embeddings)
    with pytest.raises(ValueError):
        model.init_codebook(embeddings)


def test_rq_layer_asserts_the_vq_type():
    """EMA / SimVQ entries do not take the geometric losses; construction must fail."""
    for vq_type in ("ema", "simvq"):
        with pytest.raises(ValueError):
            RQLayer(tokenizer_config(vq_type=vq_type), distance=adapter.distances)


def test_geometric_parts_reach_the_codebook_parameters():
    torch.manual_seed(0)
    config = tokenizer_config(codebook_num=1, codebook_size=(6,), sk_epsilon=-1.0, sk_iters=-1)
    layer = RQLayer(config, distance=adapter.distances)
    latent = (torch.randn(8, geometry.D_TOTAL) * 0.3).requires_grad_(True)
    p = torch.linspace(0.1, 0.9, 8)
    quantized, parts, _, _ = layer(latent, p=p)
    commitment = sum(pair[0] for pair in parts)
    codebook_term = sum(pair[1] for pair in parts)
    (commitment + codebook_term).backward()
    assert latent.grad is not None and float(latent.grad.abs().sum()) > 0.0
    weight = layer.vq_layers[0].embed.weight
    assert weight.grad is not None and float(weight.grad.abs().sum()) > 0.0


def test_the_layer_reports_the_geometric_parts_and_not_the_baseline_mse():
    """`4_method`/`geometric_quantization`: the pair the layer hands to the loss is
    `factor_share_geometry.quant_parts` on the SAME `D`, not `F.mse_loss` on the
    32-vector. Without this assertion a silent fallback to the baseline terms would
    leave every other layer test green."""
    torch.manual_seed(0)
    config = tokenizer_config(codebook_num=1, codebook_size=(6,), sk_epsilon=-1.0, sk_iters=-1)
    layer = RQLayer(config, distance=adapter.distances)
    layer.eval()
    latent = torch.randn(5, geometry.D_TOTAL) * 0.3
    p = torch.tensor([0.12, 0.37, 0.50, 0.68, 0.91])
    with torch.no_grad():
        quantized, parts, _, tokens = layer(latent, p=p)
    codebook = layer.vq_layers[0].get_code_embs()
    values = geometry.share(p, "struct")
    v_hi, v_lo = geometry.codeword_chart(codebook, values)
    rE, _, _ = geometry.split_tangent(latent)
    x_hi, x_lo = geometry.item_chart_head(latent)
    expected = geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo,
                                    codebook[:, :geometry.D_E], tokens[:, 0])
    assert torch.allclose(parts[0][0], expected[0], rtol=0, atol=1e-6)
    assert torch.allclose(parts[0][1], expected[1], rtol=0, atol=1e-6)
    # ... and that pair is NOT the baseline MSE on the same 32-vector
    baseline = (latent - codebook[tokens[:, 0]]).pow(2).mean()
    assert abs(float(parts[0][0]) - float(baseline)) > 1e-7
    assert abs(float(parts[0][1]) - float(baseline)) > 1e-7
