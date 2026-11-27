"""The candidate has no gate and reads its frozen curvature cache."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sys

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
from model import RQVAE
from train_config import load_train_config


def build():
    config = load_train_config(TREE / "config/train_config.json")
    sizes = list(config.CODEBOOK_SIZE)
    sizes = sizes * config.CODEBOOK_NUM if len(sizes) == 1 else sizes
    tokenizer = SimpleNamespace(
        hidden_sizes=config.HIDDEN_SIZES, codebook_num=config.CODEBOOK_NUM,
        codebook_size=tuple(sizes), codebook_dim=config.CODEBOOK_DIM,
        dropout=0.0, beta=config.BETA, vq_type=config.VQ_TYPE,
        ema_decay=config.EMA_DECAY, sk_epsilon=config.SK_EPSILON,
        sk_iters=config.SK_ITERS, kmeans_init=False, kmeans_iters=0)
    return RQVAE(tokenizer, in_dim=768)


def test_no_gate_path_and_frozen_cache_identity():
    model = build()
    assert not hasattr(model, "structure")
    assert not any(name.startswith(("structure.", "gate.")) for name, _ in model.named_parameters())
    assert model.kappa.rows == 28_391_948
    groups = {name for name in ("encoder",) if True}
    names = {name.split(".")[0] for name, _ in model.named_parameters()}
    assert "encoder" in names and "rq" in names and "decoder" in names
    assert groups


def test_quantize_has_no_gate_argument():
    import inspect
    signature = inspect.signature(RQVAE.quantize)
    assert "weights" not in signature.parameters
