"""One geometry switch: off must make both the assignment and the readback tangent."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import native_product
from model.layers import RQLayer


def _layer(seed=0):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=8,
                          vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1)
    return RQLayer(cfg).eval()


def test_the_switch_drives_the_assignment_and_the_readback_together():
    layer = _layer()
    x = torch.empty(6, 32).normal_(0, 2.0)
    with torch.no_grad():
        xq_on, _, _, tok_on, lv_on = layer(x, return_levels=True)
        codes = [vq.embed_code(tok_on[:, i]) for i, vq in enumerate(layer.vq_layers)]
        native = native_product.reconstruct(codes)
        assert torch.allclose(lv_on[-1], native, atol=1e-4)
        G.GEOMETRY_ON = False
        try:
            xq_off, _, _, tok_off = layer(x)
            level = layer.vq_layers[0]
            euclid = ((x.unsqueeze(1) - level.get_code_embs().unsqueeze(0)) ** 2).sum(-1)
            assert torch.allclose(level._distances(x), euclid, atol=1e-4), (
                "geometry off did not make the assignment Euclidean")
            tangent = native_product.tangent_reconstruct(
                x, [vq.embed_code(tok_off[:, i]) for i, vq in enumerate(layer.vq_layers)])
            assert torch.allclose(xq_off, tangent, atol=1e-4)
        finally:
            G.GEOMETRY_ON = True


if __name__ == "__main__":
    test_the_switch_drives_the_assignment_and_the_readback_together()
    print("ok")
