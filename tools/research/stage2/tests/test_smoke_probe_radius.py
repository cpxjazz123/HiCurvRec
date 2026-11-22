"""The gradient smoke must not hand a tree a keyword its forward never declared.

The fault this refuses: the checkpoint probe called every candidate's forward as
``model(batch, batch_ids, radius=radius)``. A tree that carries no radius table
has ``forward(x, item_ids)``, so the call raised ``TypeError: forward() got an
unexpected keyword argument 'radius'`` -- and the smoke reported a crash that
belonged to the probe. The first dispatch of such a tree never noticed, because
a fresh result directory has no checkpoint and the probe is skipped; the second
dispatch always failed.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def _source() -> str:
    return Path(__file__).resolve().parents[3].joinpath(
        "genrec-mcp", "server.py").read_text(encoding="utf-8")


def test_the_probe_guards_the_radius_keyword():
    source = _source()
    assert "if radius is None:" in source, "the probe no longer guards the keyword"
    guarded = source.split("if radius is None:", 1)[1].split("\n\n", 1)[0]
    assert "probe_model(batch, batch_ids)" in guarded
    assert "radius=radius" in guarded


def test_a_forward_without_a_radius_parameter_still_works():
    """The call the probe makes for such a tree is the two-argument one."""
    class Model:
        def forward(self, x, item_ids):
            return x

    model = Model()
    assert list(inspect.signature(model.forward).parameters) == ["x", "item_ids"]
    model.forward(1, 2)


def test_passing_the_keyword_unconditionally_is_the_defect():
    class Model:
        def forward(self, x, item_ids):
            return x

    with pytest.raises(TypeError):
        Model().forward(1, 2, radius=None)


def test_the_probe_accepts_a_four_tuple_forward():
    """The frozen baseline returns four values; the probe must not demand five."""
    source = _source()
    assert "if len(outputs) >= 5:" in source
    assert "reference = batch" in source


def test_a_four_tuple_forward_still_scores_the_reconstruction():
    batch = "the batch"
    outputs = ("reconstructed", "quant", "unused", "tokens")
    if len(outputs) >= 5:
        reconstructed, quant_loss, _u, _t, reference = outputs[:5]
    else:
        reconstructed, quant_loss, _u, _t = outputs[:4]
        reference = batch
    assert reference == batch
    assert reconstructed == "reconstructed"
