"""The production RQLayer reads back the native composition, not a plain sum.

Every case runs the real layer; nothing is mocked.  The reconstruction-only loss
must reach the encoder, and the codebook must receive its own gradient from the
same readback.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import native_product
from model.layers import RQLayer


def _cfg():
    return SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=8,
                           vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1)


def _layer(seed=0):
    torch.manual_seed(seed)
    return RQLayer(_cfg()).train()


def _codes(layer, tokens):
    return [vq.embed_code(tokens[:, level])
            for level, vq in enumerate(layer.vq_layers)]


def test_forward_is_the_native_readback():
    layer = _layer()
    x = torch.empty(6, 32).normal_(0, 0.3)
    with torch.no_grad():
        quantized, _, _, tokens, levels = layer(x, return_levels=True)
        expected = native_product.reconstruct(_codes(layer, tokens))
    assert torch.allclose(quantized, expected, atol=1e-5)
    assert torch.allclose(levels[-1], expected, atol=1e-5)


def test_reconstruction_gradient_reaches_the_encoder_and_the_codebooks():
    layer = _layer()
    x = torch.empty(6, 32).normal_(0, 0.3).requires_grad_(True)
    quantized, _, _, _, _ = layer(x, return_levels=True)
    quantized.pow(2).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert len(grads) == layer.codebook_num
    assert all(float(g.abs().sum()) > 0 for g in grads)


def test_geometry_off_still_takes_the_tangent_path():
    layer = _layer()
    x = torch.empty(6, 32).normal_(0, 0.3)
    old = G.GEOMETRY_ON
    try:
        G.GEOMETRY_ON = False
        with torch.no_grad():
            quantized, _, _, tokens = layer(x)
            tangent = native_product.tangent_reconstruct(x, _codes(layer, tokens))
        assert torch.allclose(quantized, tangent, atol=1e-6)
    finally:
        G.GEOMETRY_ON = old


if __name__ == "__main__":
    test_forward_is_the_native_readback()
    test_reconstruction_gradient_reaches_the_encoder_and_the_codebooks()
    test_geometry_off_still_takes_the_tangent_path()
    print("ok")
