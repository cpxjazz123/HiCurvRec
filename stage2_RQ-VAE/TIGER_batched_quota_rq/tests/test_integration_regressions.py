"""Regression tests for the Issue-139 experiment protocol, not a training run.

The fixtures use a tiny synthetic graph and temporary calibration files. They
exercise the shared-readout producer/consumer contract without reading private
training data or launching any formal Stage2/Stage3 job.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model.af3_structure import calibrate, readout_delta  # noqa: E402


def _resolve(*args, **kwargs):
    # Import inside the test: the old tree's missing lifecycle is a red test,
    # but must not prevent the independent numerical tests from collecting.
    module = importlib.import_module("model.af3_runtime")
    return module.resolve_shared_readout(*args, **kwargs)


def _identity():
    return {
        "graph_sha256": "a" * 64,
        "support_sha256": "b" * 64,
        "embedding_sha256": "c" * 64,
        "item_ids_sha256": "d" * 64,
        "code_sha256": "e" * 64,
    }


def _calibration():
    return {"tau": 0.2, "intercept": 0.4,
            "density": 0.25, "mean_probability": 0.25,
            "snapshot_sha256": "0" * 64}


def _must_not_calibrate():
    raise AssertionError("a consumer/existing producer must never refit the readout")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calibration_finds_root_outside_fixed_sixty_interval(dtype):
    # Orthonormal items have delta=1 on every distinct pair. IQR=0, tau=1e-6,
    # so the exact intercept is 1e6 + logit(1/3), not a value in [-60, 60].
    phi = torch.eye(4, dtype=dtype)
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    support = {"pairs": pairs, "edges": 2}
    result = calibrate(phi, support)
    assert result["tau"] == pytest.approx(1e-6, rel=0.0, abs=1e-15)
    expected = 1e6 + math.log(0.5)
    assert result["intercept"] == pytest.approx(expected, rel=0.0, abs=1e-5)
    # Independent probability calculation in double precision, not the
    # calibration function's self-reported mean_probability.
    delta = readout_delta(phi.double(), pairs)
    mean = torch.sigmoid(result["intercept"] - delta / result["tau"]).mean().item()
    assert abs(mean - 1.0 / 3.0) <= 1e-8
    assert abs(result["mean_probability"] - mean) <= 1e-8


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_calibration_rejects_non_finite_readout(bad):
    phi = torch.eye(3, dtype=torch.float64)
    phi[0, 0] = bad
    support = {"pairs": np.array([(0, 1), (0, 2), (1, 2)]), "edges": 1}
    with pytest.raises((ValueError, RuntimeError)):
        calibrate(phi, support)


@pytest.mark.parametrize("edges", [0, 3])
def test_calibration_rejects_density_without_finite_intercept(edges):
    phi = torch.eye(3, dtype=torch.float64)
    support = {"pairs": np.array([(0, 1), (0, 2), (1, 2)]), "edges": edges}
    # An all-zero/all-one target has no finite optimum for an intercept-only
    # logistic calibration; returning the search boundary would be false success.
    with pytest.raises((ValueError, RuntimeError)):
        calibrate(phi, support)


def test_f0_creates_shared_readout_only_at_epoch_1500(tmp_path):
    path = tmp_path / "readout.json"
    for epoch in (0, 1499):
        assert _resolve("F0", epoch, path, _must_not_calibrate, _identity()) is None
        assert not path.exists()
    calls = []

    def measured():
        calls.append("called")
        return _calibration()

    record = _resolve("F0", 1500, path, measured, _identity())
    assert calls == ["called"]
    assert path.is_file()
    assert record["source_arm"] == "F0"
    assert record["source_epoch"] == 1500
    assert record["identity"] == _identity()
    assert record["tau"] == 0.2
    assert record["intercept"] == 0.4
    assert record["schema"]
    assert json.loads(path.read_text(encoding="utf-8")) == record


@pytest.mark.parametrize("arm", ["G", "L", "E", "A"])
def test_consumers_fail_before_training_if_f0_readout_is_missing(tmp_path, arm):
    path = tmp_path / "readout.json"
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        _resolve(arm, 0, path, _must_not_calibrate, _identity())
    assert not path.exists()


def test_shared_readout_is_reused_without_recalibrating_or_overwriting(tmp_path):
    path = tmp_path / "readout.json"
    record = _resolve("F0", 1500, path, _calibration, _identity())
    before = path.read_bytes()
    for arm in ("F0", "G", "L", "E", "A"):
        loaded = _resolve(arm, 1500, path, _must_not_calibrate, _identity())
        assert loaded == record
        assert path.read_bytes() == before


def test_consumer_rejects_shared_readout_with_wrong_input_identity(tmp_path):
    path = tmp_path / "readout.json"
    _resolve("F0", 1500, path, _calibration, _identity())
    other_identity = _identity()
    other_identity["item_ids_sha256"] = "f" * 64
    with pytest.raises((ValueError, RuntimeError)):
        _resolve("L", 0, path, _must_not_calibrate, other_identity)


@pytest.mark.parametrize("field,value", [
    ("source_arm", "L"), ("source_epoch", 2000),
    ("tau", float("nan")), ("tau", -0.1),
    ("intercept", float("inf")),
    ("mean_probability", 0.5),
])
def test_corrupt_or_unshared_readout_is_not_silently_accepted(tmp_path, field, value):
    path = tmp_path / "readout.json"
    record = _resolve("F0", 1500, path, _calibration, _identity())
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        _resolve("E", 0, path, _must_not_calibrate, _identity())
    assert path.read_bytes() == original


def test_existing_invalid_f0_file_is_not_overwritten(tmp_path):
    path = tmp_path / "readout.json"
    path.write_text('{"this_is_not_a_calibration": true}', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        _resolve("F0", 1500, path, _must_not_calibrate, _identity())
    assert path.read_bytes() == original


def test_auxiliary_budget_cannot_add_gradient_to_zero_base_group():
    auxiliary = importlib.import_module("model.af3_hierarchy")
    budget = auxiliary.GradientBudget(share=0.10)
    scale = budget.observe(base_encoder=0.0, new_encoder=2.0,
                           base_codebook=10.0, new_codebook=1.0)
    assert scale == 0.0
    assert budget.encoder_scale == 0.0


def test_auxiliary_budget_uses_the_stricter_parameter_group():
    auxiliary = importlib.import_module("model.af3_hierarchy")
    budget = auxiliary.GradientBudget(share=0.10)
    scale = budget.observe(base_encoder=2.0, new_encoder=1.0,
                           base_codebook=5.0, new_codebook=10.0)
    assert scale == pytest.approx(0.05)
    assert scale * 1.0 <= 0.10 * 2.0
    assert scale * 10.0 <= 0.10 * 5.0


def test_auxiliary_gradient_measurement_is_read_only_and_includes_all_parameters():
    auxiliary = importlib.import_module("model.af3_hierarchy")
    x = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
    y = torch.nn.Parameter(torch.tensor([12.0], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([999.0], dtype=torch.float64))
    loss = 0.5 * (x.square().sum() + y.square().sum())
    norm = auxiliary.gradient_norm(loss, [x, y, unused])
    assert norm == pytest.approx(13.0)
    assert x.grad is None and y.grad is None and unused.grad is None
    loss.backward()
    torch.testing.assert_close(x.grad, x.detach())
    torch.testing.assert_close(y.grad, y.detach())


def _trainer_module():
    spec = importlib.util.spec_from_file_location(
        "issue139_protocol_test_trainer", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model():
    from model.model import MixedRQVAE
    config = SimpleNamespace(hidden_sizes=(16,), codebook_num=3,
                             codebook_size=(4, 4, 4), codebook_dim=32,
                             dropout=0.0, beta=0.25, sk_epsilon=0.003,
                             sk_iters=50)
    with torch.random.fork_rng():
        torch.manual_seed(827)
        model = MixedRQVAE(config, in_dim=8)
        with torch.no_grad():
            for layer in model.rq.levels:
                layer.embed.weight.mul_(0.02)
    return model


def _outer_fixture(monkeypatch):
    from model import af3_selector
    trainer = _trainer_module()
    monkeypatch.setattr(trainer, "ARM_VARIANT", "L")
    model = _tiny_model().eval()
    all_embeddings = torch.tensor(np.random.default_rng(43).normal(size=(12, 8)),
                                  dtype=torch.float32)
    graph = {"n_nodes": 3, "rows": np.array([0, 1]),
             "cols": np.array([1, 2]), "item_ids": np.array([2, 5, 9])}
    pairs = np.array([(0, 1), (0, 2), (1, 2)], dtype=np.int64)
    support = {"pairs": pairs, "is_edge": np.array([True, False, True]),
               "edges": 2, "unobserved": 1}
    targets = np.array([1.0, 0.0, 1.0])
    chosen = []

    def no_change(*args, **kwargs):
        chosen.append((kwargs["level"], kwargs["prefix"]))
        return {"accepted": False, "attempts": []}

    monkeypatch.setattr(af3_selector, "choose_weight", no_change)
    return trainer, model, all_embeddings, (graph, support, targets), chosen


def test_outer_loop_aligns_embeddings_with_graph_item_ids_not_train_target_order(monkeypatch):
    trainer, model, embeddings, bundle, _ = _outer_fixture(monkeypatch)
    seen = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().cpu().clone()))
    try:
        trainer.run_outer_loop(
            1500, model, embeddings[[0, 1, 2]], embeddings, torch.device("cpu"),
            bundle, {"tau": 0.25, "intercept": 0.0}, lambda *a, **kw: None)
    finally:
        handle.remove()
    assert seen
    torch.testing.assert_close(seen[0], embeddings[[2, 5, 9]])


def test_outer_loop_never_silently_skips_search_to_meet_wall_clock_budget(monkeypatch):
    trainer, model, embeddings, bundle, chosen = _outer_fixture(monkeypatch)
    clock = iter(float(i * 1000000) for i in range(100))
    monkeypatch.setattr(trainer.time, "time", lambda: next(clock))
    record = trainer.run_outer_loop(
        1500, model, embeddings[bundle[0]["item_ids"]], embeddings,
        torch.device("cpu"), bundle, {"tau": 0.25, "intercept": 0.0},
        lambda *a, **kw: None)
    # This tiny graph has exactly one eligible region: the L1 root. The next
    # levels fail the declared 32-item minimum; time is not a second threshold.
    assert chosen == [(1, ())]
    assert not record.get("deferred", [])


def test_a_arm_is_connected_to_the_actual_training_loss():
    tree = ast.parse((TREE / "train_rqvae.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert any(isinstance(node, ast.Name) and node.id == "AUX_AF3"
               and isinstance(node.ctx, ast.Load) for node in ast.walk(main)), \
        "the A-only switch must be read by the training entry point"
    # This is a deliberately narrow wiring check. The active-step numerical
    # test below, not this AST check, establishes the gradient budget behavior.
    assert any(isinstance(node, ast.Call)
               and ((isinstance(node.func, ast.Name) and node.func.id == "augment_loss")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "augment_loss"))
               for node in ast.walk(main)), "the computed auxiliary loss is never used"


def test_inactive_auxiliary_step_is_an_identity_without_sampling():
    auxiliary = importlib.import_module("model.af3_auxiliary")

    class NoSampling:
        def __getattr__(self, name):
            raise AssertionError("inactive A arm accessed its sampler")

    base = torch.tensor(2.0, requires_grad=True)
    for epoch, state in ((1, {}), (1500, {}), (3000, None)):
        loss, _ = auxiliary.augment_loss(base, None, state, epoch, NoSampling())
        assert loss is base


def test_auxiliary_sampler_uses_train_item_mapping_and_true_non_edges():
    auxiliary = importlib.import_module("model.af3_auxiliary")
    item_ids = np.array([2, 5, 9, 11, 13, 16], dtype=np.int64)
    graph = {"n_nodes": 6, "item_ids": item_ids,
             "rows": np.arange(5), "cols": np.arange(1, 6),
             "weights": np.array([1.0, 5.0, 3.0, 7.0, 0.0]),
             "af3_median": 3.0, "af3_p90": 7.0}
    embeddings = torch.arange(20 * 8, dtype=torch.float32).reshape(20, 8) / 100.0
    state = auxiliary.prepare_state(graph, embeddings, torch.device("cpu"))
    first = auxiliary.sample_hierarchical_pairs(state, np.random.default_rng(42))
    second = auxiliary.sample_hierarchical_pairs(state, np.random.default_rng(42))
    observed = {tuple(sorted((int(item_ids[a]), int(item_ids[b]))))
                for a, b in zip(graph["rows"], graph["cols"])}
    allowed = set(item_ids.tolist())
    expected_counts = {"af3": 192, "positive": 192 * 4,
                       "negative": 192 * 2, "unrelated": 192 * 2}
    for key, count in expected_counts.items():
        torch.testing.assert_close(first[key], second[key])
        assert first[key].shape == (count, 2)
        assert set(first[key].cpu().numpy().reshape(-1).tolist()) <= allowed
        pairs = [tuple(sorted(pair)) for pair in first[key].cpu().tolist()]
        assert all(a != b for a, b in pairs)
        if key in ("af3", "positive"):
            assert all(pair in observed for pair in pairs)
        else:
            assert all(pair not in observed for pair in pairs)
    assert torch.all(first["af3_weight"] > 0)
    assert torch.all(first["af3_weight"] <= 1)


def test_active_auxiliary_step_changes_gradients_within_each_group_budget(monkeypatch):
    auxiliary = importlib.import_module("model.af3_auxiliary")
    model = _tiny_model().eval()
    embeddings = torch.tensor(np.random.default_rng(13).normal(size=(6, 8)),
                              dtype=torch.float32)
    base_reconstruction, quant_loss, _, _ = model(embeddings)
    base, _ = model.compute_loss(embeddings, base_reconstruction, quant_loss)
    picked = {
        "af3": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "af3_weight": torch.tensor([0.5, 1.0]),
        "positive": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "negative": torch.tensor([[0, 3], [2, 5]], dtype=torch.long),
        "unrelated": torch.tensor([[1, 4], [3, 5]], dtype=torch.long),
    }
    calls = []

    def fixed_sample(state, rng):
        calls.append(True)
        return picked

    monkeypatch.setattr(auxiliary, "sample_hierarchical_pairs", fixed_sample)
    before = model.rq.alpha_table.to_dict()
    augmented, _ = auxiliary.augment_loss(
        base, model, {"all_embeddings": embeddings}, 2250,
        np.random.default_rng(42))
    assert calls == [True]
    assert torch.isfinite(augmented)
    assert float((augmented - base).detach()) > 0
    assert model.rq.alpha_table.to_dict() == before

    def norm(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True,
                                        allow_unused=True)
        vectors = [g.reshape(-1).double() for g in gradients if g is not None]
        return float(torch.cat(vectors).norm()) if vectors else 0.0

    groups = [list(model.encoder.parameters()),
              [layer.embed.weight for layer in model.rq.levels]]
    added = []
    for parameters in groups:
        base_norm = norm(base, parameters)
        extra_norm = norm(augmented - base, parameters)
        assert base_norm > 0
        # Epoch 2250 is exactly halfway through the frozen 1500..3000 ramp.
        assert extra_norm <= 0.10 * 0.5 * base_norm * (1.0 + 1e-5) + 1e-8
        added.append(extra_norm)
    assert min(added) > 1e-8, "A arm must actually supervise both encoder and codebook"


