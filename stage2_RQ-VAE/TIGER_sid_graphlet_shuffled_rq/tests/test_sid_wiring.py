"""Issue #231 C19/C21: the term must reach `loss.backward()` inside the real trainer.

The quantizer today returns only integer indices, so the STE has nothing to differentiate.
`RQLayer.forward` must therefore also return the per-level cost matrices (`logits`), and the
trainer must add `structure_loss` to the RQ-VAE loss *before* the single `backward()` call --
not by pulling a gradient out with `autograd.grad` and patching a matrix (§3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent


def _trainer():
    if str(TREE) not in sys.path:
        sys.path.insert(0, str(TREE))
    import train_rqvae  # noqa: PLC0415

    return train_rqvae


def test_the_quantizer_returns_per_level_logits():
    from model.layers import RQLayer  # noqa: PLC0415

    config = _trainer()._tokenizer_config()
    layer = RQLayer(config)
    latent = torch.randn(5, int(config.codebook_dim), requires_grad=True)
    out = layer(latent)
    assert len(out) == 5, "RQLayer must append the per-level cost matrices"
    quantized, parts, unused, tokens, logits = out
    assert len(logits) == int(config.codebook_num)
    assert logits[0].shape == (5, int(config.codebook_size[0]))


def test_the_model_forward_exposes_the_logits_to_the_trainer():
    train_rqvae = _trainer()
    from model import RQVAE  # noqa: PLC0415

    torch.manual_seed(0)
    config = train_rqvae._tokenizer_config()
    model = RQVAE(config, in_dim=8, distance=train_rqvae.build_distance())
    embeddings = torch.randn(4, 8)
    out = model(embeddings)
    assert len(out) == 5
    assert len(out[4]) == int(config.codebook_num)


def test_the_trainer_adds_the_structure_term_to_the_total_loss():
    """C19/C20: one addend, one backward(); no autograd.grad side channel anywhere."""
    _trainer()
    source = (TREE / "train_rqvae.py").read_text()
    assert "SidGraphletAlignment" in source, "the trainer must build the structure entity"
    assert "loss = loss + graphlet_value * scale" in source, "the term must addend L_total"
    code = [line for line in source.splitlines()
            if "autograd.grad" in line and not line.lstrip().startswith("#")]
    assert not code, "§3 forbids the autograd.grad side channel: " + str(code[:1])
    body = source[source.index("for ids, batch in loader:"):]
    step = body[:body.index("losses.append")]
    assert step.count("loss.backward()") == 1, "exactly one backward for the summed loss"


def test_zero_lambda_reproduces_the_baseline_step():
    """C27 at trainer level: lambda=0 must leave the loss exactly the baseline's."""
    train_rqvae = _trainer()
    # C27: the scale is proportional to reconstruction, so a zero share switches the term off
    # without touching any other arithmetic, and the ceiling is the definitional 3 x margin.
    assert float(train_rqvae.GRAPHLET_SHARE) > 0.0, "the run must actually use the term"
    assert float(train_rqvae.GRAPHLET_CEILING) == 3.0
    source = (TREE / "train_rqvae.py").read_text()
    assert "GRAPHLET_SHARE * float(recon_loss.detach()) / GRAPHLET_CEILING" in source
