"""None vs zero gradients, and a checkpoint that must actually restore."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe_factors as F
from model.model import RQVAE
import train_rqvae as T

CKPT = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")


def test_none_and_zero_gradients_are_counted_separately():
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(3))
    out = F.block_status([a, b], [None, torch.zeros(3)])
    assert out["none_tensors"] == 1 and out["zero_tensors"] == 1
    assert out["nonzero_tensors"] == 0


def test_the_checkpoint_restores_weights_and_rejects_a_truncated_state():
    model = RQVAE(T._tokenizer_config(), in_dim=768)
    state = torch.load(CKPT, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    reference = next(iter(state.values()))
    got = next(iter(model.state_dict().values()))
    assert torch.equal(got, reference), "weights were not restored"
    truncated = {k: v for i, (k, v) in enumerate(state.items()) if i > 3}
    try:
        model.load_state_dict(truncated, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a truncated state must be rejected")


if __name__ == "__main__":
    test_none_and_zero_gradients_are_counted_separately()
    test_the_checkpoint_restores_weights_and_rejects_a_truncated_state()
    print("ok")
