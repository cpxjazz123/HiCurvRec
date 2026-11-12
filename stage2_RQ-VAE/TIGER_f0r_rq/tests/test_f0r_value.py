"""Issue #206 acceptance 1, 2, 5: the value ratio, both lambdas recorded, and the two gate predicates.

The numbers are the ones the real entry point produced in `results/research/issue206/one_epoch_run`, so the cases check the arithmetic that decides whether this
arm trains, not a paraphrase of it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]


def _trainer():
    spec = importlib.util.spec_from_file_location("f0r_trainer_under_test", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


t = _trainer()

#: measured on the first batch of the real entry point, issue #206
LAMBDA_VALUE = 2.529077259702385
LAMBDA_BLOCK = 247.97704236198626
L_SD = 0.749666690826416
L_RECON = 1.8959649801254272
COLLISION = 0.5121812339854395


def test_the_value_ratio_is_the_ratio_of_the_two_values():
    assert LAMBDA_VALUE == pytest.approx(L_RECON / L_SD, rel=1e-12)


def test_the_identity_holds_to_rtol_and_fails_when_the_weight_is_off():
    assert t.value_identity_holds(LAMBDA_VALUE, L_SD, L_RECON)
    assert not t.value_identity_holds(LAMBDA_VALUE * 1.001, L_SD, L_RECON)


def test_the_block_budget_is_recorded_and_would_have_refused():
    assert LAMBDA_BLOCK == pytest.approx(247.977, rel=1e-5)
    assert LAMBDA_BLOCK * L_SD == pytest.approx(185.9, abs=0.5)     # against 10 * 1.896 = 18.96
    assert LAMBDA_BLOCK * L_SD > 10.0 * L_RECON
    assert 1e-6 <= LAMBDA_BLOCK <= 1e3                              # the range test alone would have passed


def test_the_collapse_gate_fires_on_the_observed_collision_and_not_below_it():
    assert t.collapse_gate_fires(COLLISION)
    assert t.collapse_gate_fires(0.9)
    assert not t.collapse_gate_fires(0.3999)
    assert not t.collapse_gate_fires(0.252369)


def test_the_scale_gate_predicate_still_rejects_the_block_ratio():
    # the block gate is kept as an assertion about the block lambda; the value lambda passes it by construction
    assert t.value_identity_holds(LAMBDA_VALUE, L_SD, L_RECON)
    assert not t.value_identity_holds(LAMBDA_BLOCK, L_SD, L_RECON)
