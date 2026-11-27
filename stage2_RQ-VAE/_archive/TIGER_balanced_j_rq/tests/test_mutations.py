"""Mutations of the distance formula must be caught by named assertions.

Each case edits the production source text and executes the mutated module.  A
crash is not a catch: the designated check must return False with the mutated
module loaded.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch

import sys

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
SOURCE = (TREE / "model/stereographic.py").read_text()


def reference_squared(x, y, k):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = ((x - y) ** 2).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    if k == 0:
        return 4.0 * xy
    if k < 0:
        a = -k
        argument = 1.0 + 2.0 * a * xy / ((1.0 - a * x2) * (1.0 - a * y2))
        return np.arccosh(argument) ** 2 / a
    radius2 = 1.0 / k
    dot = (x * y).sum(-1)
    numerator = 4.0 * radius2 * dot + (x2 - radius2) * (y2 - radius2)
    denominator = (radius2 + x2) * (radius2 + y2)
    return radius2 * np.arccos(np.clip(numerator / denominator, -1.0, 1.0)) ** 2


def load(source):
    module = types.ModuleType("mutated_stereographic")
    exec(compile(source, "mutated_stereographic.py", "exec"), module.__dict__)
    return module


def check_zero_exact(module):
    torch.manual_seed(1)
    x = torch.randn(16, 8, dtype=torch.float64)
    y = torch.randn(16, 8, dtype=torch.float64)
    return bool(torch.equal(module.squared(x, y, 0.0),
                            4.0 * (x - y).square().sum(-1)))


def check_reference(module):
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.2, size=(32, 8))
    y = rng.normal(scale=0.2, size=(32, 8))
    good = True
    for k in (-0.125, -0.05, 0.05, 0.5, 1.0):
        got = module.squared(torch.from_numpy(x), torch.from_numpy(y), k).numpy()
        good = good and np.allclose(got, reference_squared(x, y, k), rtol=1e-8, atol=1e-8)
    return good


MUTATIONS = {
    "scale_4_to_2": ("return 4.0 * values", "return 2.0 * values", "zero_exact"),
    "negative_sign_flip": ("denominator = (1.0 + k * x2) * (1.0 + k * y2)",
                           "denominator = (1.0 - k * x2) * (1.0 - k * y2)", "reference"),
    "sphere_asinh": ("sphere = torch.asin(", "sphere = torch.asinh(", "reference"),
    "unsigned_curvature": ("magnitude = k.abs().clamp_min(torch.finfo(x.dtype).tiny)",
                           "magnitude = k.clamp_min(torch.finfo(x.dtype).tiny)", "reference"),
    "single_branch": ("values = torch.where(k > 0, sphere, torch.where(k < 0, hyperbolic, xy))",
                      "values = sphere", "reference"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_is_caught_by_the_named_check(name):
    needle, replacement, check_name = MUTATIONS[name]
    assert needle in SOURCE, name
    mutated = load(SOURCE.replace(needle, replacement))
    caught = (check_zero_exact(mutated) if check_name == "zero_exact"
              else check_reference(mutated))
    assert caught is False, name
