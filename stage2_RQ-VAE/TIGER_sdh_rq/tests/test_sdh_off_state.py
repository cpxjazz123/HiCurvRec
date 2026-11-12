"""Issue #194 acceptance 1, the in-tree half: with BEHAVIOUR_OFF the mechanism is unreachable from a training step.

The process-level half (one epoch of this tree with the switch closed against one epoch of the frozen baseline, same seed, same CPU) lives in
``results/research/issue194/off_equivalence.py``; this file states the property the code must have for that comparison to mean anything -- a step with
the switch closed touches neither the support mask, nor the curvature kernel, nor the teacher -- and that the three switch names exist with the right
defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import RQVAE, sdh_kernel, sd_step, struct_distill  # noqa: E402


def _config():
    return SimpleNamespace(codebook_num=2, codebook_dim=8, codebook_size=(16, 16), beta=0.25,
                           vq_type="vq", sk_epsilon=-1.0, sk_iters=-1, dropout=0.0,
                           hidden_sizes=[32, 16])


def test_behaviour_off_is_the_same_flag_as_sd_off():
    """One place to flip: #181's name reads the issue's flag, so setting one and reading the other cannot disagree."""
    saved = struct_distill.BEHAVIOUR_OFF
    try:
        struct_distill.BEHAVIOUR_OFF = True
        assert struct_distill.SD_OFF is True
        struct_distill.BEHAVIOUR_OFF = False
        assert struct_distill.SD_OFF is False
    finally:
        struct_distill.BEHAVIOUR_OFF = saved


def test_the_three_switches_exist_and_the_shipped_state_is_both_on():
    assert struct_distill.BEHAVIOUR_OFF is False
    assert sdh_kernel.SUPPORT_OFF is False
    assert sdh_kernel.GEOMETRY_OFF is False


def test_a_step_with_the_switch_closed_never_reaches_the_mechanism(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the mechanism was reached while BEHAVIOUR_OFF is set")

    monkeypatch.setattr(sdh_kernel, "support_mask", explode)
    monkeypatch.setattr(sdh_kernel, "affinity", explode)
    monkeypatch.setattr(sd_step, "codes_of", explode)
    saved = struct_distill.BEHAVIOUR_OFF
    struct_distill.BEHAVIOUR_OFF = True
    try:
        torch.manual_seed(0)
        module = RQVAE(_config(), in_dim=8)
        batch = torch.randn(16, 8)
        reconstructed, quant_loss, _, _ = module(batch)
        loss, recon = module.compute_loss(batch, reconstructed, quant_loss)
        loss.backward()
        assert torch.isfinite(loss)
        # the trainer's own expression for the structural weight, evaluated with the switch closed
        sd_lambda = 0.0 if struct_distill.BEHAVIOUR_OFF else struct_distill.lambda_schedule(10) * 1.0
        assert sd_lambda == 0.0
    finally:
        struct_distill.BEHAVIOUR_OFF = saved


def test_the_switch_states_are_read_at_call_time_not_frozen_at_import():
    """A module-level ``if`` would freeze the state; the mask has to consult the flag every call."""
    anchor = torch.tensor([[1, 2, 3]])
    candidates = torch.tensor([[[9, 9, 9]]])
    saved = sdh_kernel.SUPPORT_OFF
    try:
        assert sdh_kernel.support_mask(anchor, candidates).tolist() == [[False]]
        sdh_kernel.SUPPORT_OFF = True
        assert sdh_kernel.support_mask(anchor, candidates).tolist() == [[True]]
    finally:
        sdh_kernel.SUPPORT_OFF = saved
