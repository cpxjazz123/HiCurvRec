"""Target proxy on the real RQLayer: encoder I, each selected codebook I.

Fails on the pre-fix graph, which gives (3, -1, 0, +1).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
from model.layers import RQLayer


def build():
    torch.manual_seed(0)
    cfg = SimpleNamespace(codebook_num=3, codebook_dim=32, codebook_size=256,
                          vq_type="vq", beta=0.25, sk_epsilon=-1.0, sk_iters=-1)
    layer = RQLayer(cfg).double()
    with torch.no_grad():
        for level, vq in enumerate(layer.vq_layers):
            embs = vq.get_code_embs()
            embs.normal_(0, 50.0)
            embs[0] = 0.0
            embs[0, 0] = 0.1 * (level + 1)
            embs[0, 16] = 0.0
    for vq in layer.vq_layers:
        vq._indices = lambda distances, infer: torch.zeros(distances.shape[0], dtype=torch.long)
    return layer


def jacobian(which_level):
    layer = build()
    z = torch.zeros(1, 32, dtype=torch.float64)
    z[0, 0] = 0.8
    z.requires_grad_(True)
    out = layer(z, return_levels=True)
    q = out[4][which_level][0]
    target = q[0]
    grads = torch.autograd.grad(target, [z] + [vq.get_code_embs() for vq in layer.vq_layers],
                                allow_unused=True)
    codes = [0.0 if g is None else float(g[0, 0]) for g in grads[1:]]
    return (float(q[0].detach()), float(grads[0][0, 0]), codes)


def expected(level):
    forward = sum(0.1 * (i + 1) for i in range(level + 1))
    dz = 1.0
    codes = [1.0 if i <= level else 0.0 for i in range(3)]
    return forward, dz, codes


if __name__ == "__main__":
    for level in range(3):
        got = jacobian(level)
        want = expected(level)
        print(f"level {level + 1}: forward {got[0]:.6f} dz {got[1]:.6f} codes {got[2]} | target {want}")
        assert abs(got[0] - want[0]) < 1e-9, (level, "forward", got[0], want[0])
        assert abs(got[1] - want[1]) < 1e-9, (level, "encoder coefficient", got[1], want[1])
        for i, (g, w) in enumerate(zip(got[2], want[2])):
            assert abs(g - w) < 1e-9, (level, f"codebook {i+1}", g, w)
    print("ok")
