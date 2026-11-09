"""Per-block gradient measurement, and the branch check on the real layer.

The old probe concatenated every parameter into one vector and labelled the angle
"encoder"; these cases make that mistake impossible to repeat.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_blocks as B


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4, bias=False)
        self.codebook = torch.nn.Parameter(torch.eye(4))


def test_per_block_cosine_separates_same_and_opposite_blocks():
    model = Toy()
    g = torch.ones(16)
    # block A follows g, block B opposes it: the global cosine is ~0 while the two
    # block cosines are +1 and -1
    blocks = {"encoder": [model.encoder.weight], "codebook": [model.codebook]}
    grads = {"encoder": [g[:16].reshape(4, 4)],
             "codebook": [-g[:16].reshape(4, 4)]}
    out = B.block_cosines(g, blocks, grads=grads)
    assert abs(out["encoder"]["cosine"] - 1.0) < 1e-6
    assert abs(out["codebook"]["cosine"] + 1.0) < 1e-6


def test_none_and_zero_are_distinguished():
    w = torch.nn.Parameter(torch.zeros(3))
    blocks = {"unused": [w]}
    zero_grads = {"unused": [torch.zeros(3)]}
    none_grads = {"unused": [None]}
    a = B.describe_blocks(blocks, zero_grads)
    b = B.describe_blocks(blocks, none_grads)
    assert a["unused"]["zero_tensors"] == 1 and a["unused"]["none_tensors"] == 0
    assert b["unused"]["none_tensors"] == 1 and b["unused"]["zero_tensors"] == 0


def test_undefined_cosine_when_a_block_has_no_norm():
    g = torch.ones(3)
    blocks = {"empty": [torch.nn.Parameter(torch.zeros(3))]}
    out = B.block_cosines(g, blocks, grads={"empty": [torch.zeros(3)]})
    assert out["empty"]["cosine"] == "UNDEFINED"


def test_the_branch_check_counts_the_real_sinkhorn(tmp_path=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mixed_geometry as G
    from model.layers import VQLayer

    torch.manual_seed(0)
    layer = VQLayer(8, 32, 0.25, 0.003, 2)
    with torch.no_grad():
        layer.get_code_embs().normal_(0, 0.5)
    x = torch.empty(16, 32).normal_(0, 1.0)
    calls = {"n": 0}
    original = layer.sinkhorn

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    layer.sinkhorn = counted
    G.GEOMETRY_ON = True
    try:
        B.branch_tokens_real(layer, x, mode="train")
        train_calls = calls["n"]
        B.branch_tokens_real(layer, x, mode="export")
        export_calls = calls["n"] - train_calls
    finally:
        G.GEOMETRY_ON = True
    assert train_calls == 1, "train mode must take the Sinkhorn branch"
    assert export_calls == 0, "eval + infer_use_sk=False must take argmin"


if __name__ == "__main__":
    test_per_block_cosine_separates_same_and_opposite_blocks()
    test_none_and_zero_are_distinguished()
    test_undefined_cosine_when_a_block_has_no_norm()
    test_the_branch_check_counts_the_real_sinkhorn()
    print("ok")
