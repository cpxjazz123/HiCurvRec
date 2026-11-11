"""Analytic E16 factor of the STE proxy, measured on the real RQLayer.

Discrete selection is fixed by monkeypatching _indices, so the only thing measured
is the differentiable path through the selected codewords.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed_geometry as G
from model.layers import RQLayer

torch.manual_seed(0)
G.GEOMETRY_ON = True


def build():
    cfg = SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=256,
                          vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1)
    layer = RQLayer(cfg).double()
    with torch.no_grad():
        for level, vq in enumerate(layer.vq_layers):
            embs = vq.get_code_embs()
            embs.normal_(0, 50.0)          # every other code is far away
            embs[0] = 0.0
            embs[0, 0] = 0.1 * (level + 1)   # E first coordinate 0.1/0.2/0.3
            embs[0, 16] = 0.0
    return layer


def measure():
    layer = build()
    z = torch.zeros(1, 32, dtype=torch.float64)
    z[0, 0] = 0.8
    z.requires_grad_(True)
    # fixed discrete choice: code 0 at every level
    for vq in layer.vq_layers:
        vq._indices = lambda distances, infer: torch.zeros(distances.shape[0], dtype=torch.long)
    out = layer(z)
    q = out[0][0]
    print("selected tokens", out[3].tolist() if len(out) > 3 else out[0].shape)
    q_e0 = q[0]
    grads = torch.autograd.grad(q_e0, [z] + [vq.get_code_embs() for vq in layer.vq_layers],
                                allow_unused=True)
    dz = grads[0][0, 0]
    codes = [g[0, 0] if g is not None else torch.tensor(float("nan"), dtype=torch.float64)
             for g in grads[1:]]
    return float(q_e0.detach()), float(dz), [float(c) for c in codes]


if __name__ == "__main__":
    value, dz, codes = measure()
    print("forward q_E0", value)
    print("d q_E0 / d z_E0", dz)
    print("d q_E0 / d cE0 per level", codes)
    print("current coefficients", (dz, *codes))
    print("declared target     ", (1.0, 1.0, 1.0, 1.0))
