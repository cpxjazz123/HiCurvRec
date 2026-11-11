"""The frozen assignment protocol, pinned per level from the tree's own config."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_rqvae as T
from model import RQVAE


def build():
    torch.manual_seed(0)
    return RQVAE(T._tokenizer_config(), in_dim=32)


def test_only_the_last_level_uses_sinkhorn():
    # The frozen baseline sets Sinkhorn on the LAST level only; the first two
    # levels are argmin even in training.  This test exists because an earlier
    # note in this tree claimed "every level", which the source contradicts.
    model = build()
    flags = [layer.use_sk for layer in model.rq.vq_layers]
    assert len(flags) == T.CODEBOOK_NUM == 3
    assert flags == [False, False, True]
    last = model.rq.vq_layers[-1]
    assert last.sk_iters == T.SK_ITERS == 50 and last.sk_epsilon == T.SK_EPSILON


def test_training_takes_sinkhorn_on_the_last_level_and_eval_takes_argmin():
    model = build()
    x = torch.randn(4, 32)
    last = model.rq.vq_layers[-1]
    distances = last._distances(x)
    model.train()
    plan = last.sinkhorn(last.center_distance(distances).double(), T.SK_EPSILON, T.SK_ITERS)
    assert torch.equal(last._indices(distances, False), torch.argmax(plan, dim=-1))
    model.eval()
    assert torch.equal(last._indices(distances, False), torch.argmin(distances, dim=-1))
    first = model.rq.vq_layers[0]
    assert torch.equal(first._indices(distances, False), torch.argmin(distances, dim=-1))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
