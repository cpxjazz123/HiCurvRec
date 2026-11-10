"""The assignment cost contains geometry and a TRAIN-only behaviour offset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import model.mixed_rq as M


def _setup():
    torch.manual_seed(0)
    rq = M.MixedRQ([8, 8, 8], M.AlphaTable(levels=3))
    z = torch.randn(4, 32)
    beh = torch.randn(4, 32)
    return rq, z, beh


def _offsets(rq, beh):
    b = F.normalize(beh[:, :16], dim=-1)
    return [F.normalize(l.get_code_embs()[:, :16], dim=-1).matmul(b.t()).t() for l in rq.levels]


def test_behaviour_offset_changes_the_tokens():
    rq, z, beh = _setup()
    base = M.encode_tokens(z, rq)
    shifted = M.encode_tokens(z, rq, offsets=_offsets(rq, beh * 50))
    assert not torch.equal(base, shifted)


def test_geometry_switch_changes_the_cost():
    rq, z, _ = _setup()
    layer = rq.levels[0]
    ze, zh = M.split(z[:, :32])
    zh = M.expmap0(zh, rq.curvature)
    a = torch.full((4,), 0.5)
    on = layer.distances(ze, zh, a)
    M.GEOMETRY_ON = False
    try:
        off = layer.distances(ze, zh, a)
    finally:
        M.GEOMETRY_ON = True
    assert not torch.allclose(on, off)


def test_assignment_cost_keeps_sinkhorn_balancing():
    rq, z, beh = _setup()
    layer = rq.levels[-1]
    layer.sk_epsilon, layer.sk_iters, layer.use_sk = 0.05, 5, True
    cost = torch.rand(16, 8)
    layer._indices(cost, False)
    assert layer.use_sk
