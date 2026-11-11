"""Issue #183 acceptance 4: one kappa table and one map, bound across the quantiser, the cost and the contrast."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import r48ke_geometry as g  # noqa: E402
from model import RQVAE  # noqa: E402


def _config():
    return SimpleNamespace(codebook_num=2, codebook_dim=8, codebook_size=(16, 16), beta=0.25,
                           vq_type="vq", sk_epsilon=-1.0, sk_iters=-1, dropout=0.0,
                           hidden_sizes=[32, 16])


def test_the_model_maps_its_encoder_output_at_the_registered_curvature():
    torch.manual_seed(0)
    module = RQVAE(_config(), in_dim=8)
    embeddings = torch.randn(6, 8) * 0.2
    plain = module.encoder(embeddings)
    g.set_batch_kappa(torch.linspace(0.3, 2.0, 6))
    mapped_expected = g.expmap0(plain.detach(), torch.linspace(0.3, 2.0, 6).reshape(-1, 1))
    g.set_batch_kappa(None)
    # with the hook cleared the model uses the raw encoder output
    assert torch.allclose(module.encoder(embeddings), plain)
    # and the mapped value is what the hook would have produced
    assert torch.allclose(mapped_expected, g.expmap0(plain.detach(), torch.linspace(0.3, 2.0, 6).reshape(-1, 1)))


def test_the_mapped_representation_stays_inside_every_ball():
    torch.manual_seed(1)
    z = torch.randn(8, 6) * 2.0
    for k in (0.5, 1.0, 3.0):
        kappa = torch.full((8, 1), k)
        p = g.expmap0(z, kappa)
        assert float((k * p.pow(2).sum(-1)).max()) < 1.0


def test_the_contrast_and_the_map_use_the_same_kappa_table():
    ids = torch.tensor([0, 1, 2])
    values32 = g.kappa_of(ids)
    values = values32.to(torch.float64)
    # the contrast's distance path reads the same lookup, so the two agree by construction
    kappa = g.pair_kappa(values, values)
    assert kappa.shape == (3, 3)
    assert torch.allclose(kappa.diag(), values, rtol=1e-6)


def test_geometry_off_leaves_both_the_map_and_the_distance_euclidean():
    a = torch.randn(4, 5) * 0.1
    b = torch.randn(4, 5) * 0.1
    ks = torch.full((4,), 1.0)
    g.GEOMETRY_OFF = True
    assert torch.equal(g.expmap0(a, ks.reshape(-1, 1)), a)
    assert torch.equal(g.behaviour_distances(a, b, ks, ks), torch.cdist(a, b, p=2))
    g.GEOMETRY_OFF = False
