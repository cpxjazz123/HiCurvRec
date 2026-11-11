"""Tiny CPU entry-point fixtures, never a formal Stage2 or Stage3 run.

Only these tests override the trainer's constants. They use eight synthetic
embedding rows, temporary outputs, and one actual optimisation epoch. No real
TRAIN/VALID/TEST input, GPU launcher, socket, or experiment directory is used.
The active-phase fixture similarly executes one selected epoch, not 1501 fits.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "issue139_synthetic_cpu_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def _configure(monkeypatch, tmp_path, *, arm="F0", epoch=1):
    trainer = _load_trainer()
    # This is a local unit-test fixture. The source constants and the formal
    # launcher are unchanged, and monkeypatch restores every setting afterwards.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constants = {
        "ARM_VARIANT": arm, "CURVATURE": 1.0,
        "OUTER_LOOP": arm in ("G", "L", "E"),
        "AUX_AF3": arm == "A", "EPOCHS": epoch,
        "BATCH_SIZE_PER_RANK": 4, "HIDDEN_SIZES": (8,),
        "CODEBOOK_SIZE": (4, 4, 4), "NUM_WORKERS": 0,
        "EVAL_INTERVAL": 1, "WARMUP_EPOCHS": 0,
        "RESULT_DIR": tmp_path / arm,
        "CHECKPOINT": tmp_path / arm / "snapshot.pth",
        "OUTPUT_SID": tmp_path / arm / "sids.npy",
        "OUTPUT_JSON": tmp_path / arm / "sids.json",
        "LOG_DIR": tmp_path / arm / "logs",
        "METRICS_PATH": tmp_path / arm / "logs" / "metrics.jsonl",
        "READOUT_FILE": tmp_path / "readout.json",
    }
    for name, value in constants.items():
        monkeypatch.setattr(trainer, name, value)
    embeddings = np.random.default_rng(203).normal(size=(8, 8)).astype(np.float32) * 0.1
    frame = pd.DataFrame({"target": np.arange(8),
                          "history": [[(i + 1) % 8] for i in range(8)]})
    monkeypatch.setattr(trainer, "load_embeddings", lambda path: embeddings.copy())
    monkeypatch.setattr(trainer.pd, "read_parquet", lambda *args, **kwargs: frame.copy())
    return trainer, embeddings


def _assert_export_matches_loaded_checkpoint(trainer, embeddings, expected_epoch):
    from model.model import MixedRQVAE
    checkpoint = torch.load(trainer.CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == expected_epoch
    assert checkpoint["selection"] == "final_epoch"
    assert "rq._extra_state" in checkpoint["state_dict"], \
        "a geometry table absent from the checkpoint cannot reconstruct its SIDs"
    restored = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                          curvature=trainer.CURVATURE).eval()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.no_grad():
        expected = restored.get_indices(torch.from_numpy(embeddings)).cpu().numpy()
    sid = np.load(trainer.OUTPUT_SID)
    assert sid.shape == (len(embeddings), 4)
    np.testing.assert_array_equal(sid[:, :3], expected)
    assert len(np.unique(sid, axis=0)) == len(embeddings)
    payload = json.loads(trainer.OUTPUT_JSON.read_text(encoding="utf-8"))
    assert payload == {str(i): row.tolist() for i, row in enumerate(sid)}
    events = [json.loads(line) for line in trainer.METRICS_PATH.read_text(
        encoding="utf-8").splitlines()]
    assert any(event["event"] == "ckpt_saved" for event in events)
    assert events[-1]["event"] == "train_end"
    return checkpoint, events


def test_f0_real_entry_point_saves_alpha_state_and_restores_export(monkeypatch, tmp_path):
    trainer, embeddings = _configure(monkeypatch, tmp_path)

    def no_graph_before_calibration():
        raise AssertionError("F0 before epoch 1500 must not need a behaviour graph")

    monkeypatch.setattr(trainer, "_frozen_structure", no_graph_before_calibration)
    trainer.main()
    _assert_export_matches_loaded_checkpoint(trainer, embeddings, 1)
    assert not trainer.READOUT_FILE.exists()


def _inject_toy_readout_inputs(trainer, monkeypatch):
    graph = {"n_nodes": 4, "item_ids": np.array([0, 2, 5, 7]),
             "rows": np.array([0, 0, 1, 2]), "cols": np.array([1, 2, 2, 3]),
             "weights": np.array([3.0, 2.0, 2.0, 0.0]),
             "af3_median": 2.0, "af3_p90": 2.7}
    pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    support = {"pairs": pairs, "is_edge": np.array([True, True, False, True, False, True]),
               "edges": 4, "unobserved": 2}
    targets = np.array([3.0, 2.0, 0.0, 2.0, 0.0, 0.0])
    identity = {"graph_sha256": "1" * 64, "support_sha256": "2" * 64,
                "embedding_sha256": "3" * 64, "train_sha256": "4" * 64,
                "item_ids_sha256": "5" * 64, "code_sha256": "6" * 64}

    def prepare(raw_module):
        raw_module._structure_bundle = (graph, support, targets)
        return graph, support, identity

    monkeypatch.setattr(trainer, "_prepare_readout_inputs", prepare)
    return graph, support, identity


def _only_selected_epoch(trainer, monkeypatch, epoch):
    # Change only this module's top-level epoch iterator, not Python's builtin
    # range or any assignment/quantizer loop. This is an active-phase fixture.
    def selected_range(*args):
        return [epoch] if args == (1, epoch + 1) else range(*args)

    monkeypatch.setattr(trainer, "range", selected_range, raising=False)


def _freeze_fixture_readout(trainer, support, identity):
    from model.af3_runtime import resolve_shared_readout
    from model.af3_structure import calibrate

    def synthetic_f0_snapshot():
        record = calibrate(torch.eye(4, dtype=torch.float64), support)
        record["snapshot_sha256"] = "0" * 64
        return record

    return resolve_shared_readout("F0", 1500, trainer.READOUT_FILE,
                                  synthetic_f0_snapshot, identity)


def test_active_a_phase_runs_in_actual_main_and_exports_checkpoint(monkeypatch, tmp_path):
    from model import af3_auxiliary
    trainer, embeddings = _configure(monkeypatch, tmp_path, arm="A", epoch=1501)
    _, support, identity = _inject_toy_readout_inputs(trainer, monkeypatch)
    _freeze_fixture_readout(trainer, support, identity)
    before_readout = trainer.READOUT_FILE.read_bytes()
    _only_selected_epoch(trainer, monkeypatch, 1501)
    original_augment = af3_auxiliary.augment_loss
    additions = []
    applied_losses = []

    def observe(base_loss, raw_module, state, epoch, rng):
        loss, log = original_augment(base_loss, raw_module, state, epoch, rng)
        additions.append((float((loss - base_loss).detach()), epoch, dict(log)))
        applied_losses.append(float(loss.detach()))
        return loss, log

    monkeypatch.setattr(af3_auxiliary, "augment_loss", observe)
    trainer.main()
    _, events = _assert_export_matches_loaded_checkpoint(trainer, embeddings, 1501)
    assert len(additions) == 2, "each of the two actual training batches must use A"
    assert all(epoch == 1501 and log["active"] for _, epoch, log in additions)
    assert all(log["ramp"] == pytest.approx(1.0 / 1500.0)
               for _, _, log in additions)
    assert any(value > 0.0 for value, _, _ in additions), "A must change the real objective"
    assert trainer.READOUT_FILE.read_bytes() == before_readout
    auxiliary_events = [row for row in events if row["event"] == "auxiliary"]
    assert len(auxiliary_events) == 1 and auxiliary_events[0]["active_steps"] == 2
    training_events = [row for row in events if row["event"] == "train"]
    assert len(training_events) == 1 and training_events[0]["epoch"] == 1501
    # Calling the helper and then discarding its returned objective must fail:
    # the actual trainer's batch-loss aggregation must contain the augmentation.
    expected_mean = float(torch.tensor(sum(applied_losses) / len(applied_losses)))
    assert training_events[0]["loss"] == pytest.approx(expected_mean, rel=0, abs=1e-12)


def test_f0_epoch1500_main_creates_snapshot_bound_shared_calibration(monkeypatch, tmp_path):
    from model.af3_runtime import snapshot_sha256
    from model.model import MixedRQVAE
    trainer, embeddings = _configure(monkeypatch, tmp_path, arm="F0", epoch=1500)
    _, _, identity = _inject_toy_readout_inputs(trainer, monkeypatch)
    _only_selected_epoch(trainer, monkeypatch, 1500)
    trainer.main()
    checkpoint, events = _assert_export_matches_loaded_checkpoint(trainer, embeddings, 1500)
    readout = json.loads(trainer.READOUT_FILE.read_text(encoding="utf-8"))
    assert readout["source_arm"] == "F0" and readout["source_epoch"] == 1500
    assert readout["identity"] == identity
    restored = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                          curvature=trainer.CURVATURE)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert readout["snapshot_sha256"] == snapshot_sha256(restored)
    assert any(row["event"] == "shared_readout_frozen" for row in events)
    assert abs(readout["mean_probability"] - readout["density"]) <= 1e-8


def test_global_arm_actual_main_uses_shared_readout_for_scheduled_round(monkeypatch, tmp_path):
    trainer, embeddings = _configure(monkeypatch, tmp_path, arm="G", epoch=1500)
    _, support, identity = _inject_toy_readout_inputs(trainer, monkeypatch)
    _freeze_fixture_readout(trainer, support, identity)
    before = trainer.READOUT_FILE.read_bytes()
    _only_selected_epoch(trainer, monkeypatch, 1500)
    trainer.main()
    _, events = _assert_export_matches_loaded_checkpoint(trainer, embeddings, 1500)
    rounds = [row for row in events if row["event"] == "outer_loop"]
    assert len(rounds) == 1 and rounds[0]["epoch"] == 1500
    assert trainer.READOUT_FILE.read_bytes() == before


@pytest.mark.parametrize("arm", ["G", "A"])
def test_actual_main_fails_before_optimizer_when_shared_readout_missing(
        monkeypatch, tmp_path, arm):
    trainer, _ = _configure(monkeypatch, tmp_path, arm=arm, epoch=1)

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("consumer reached the optimiser before validating F0 provenance")

    monkeypatch.setattr(trainer.torch.optim, "AdamW", forbidden_optimizer)
    with pytest.raises((FileNotFoundError, RuntimeError), match="F0"):
        trainer.main()
    assert not trainer.CHECKPOINT.exists()
    assert not trainer.OUTPUT_JSON.exists()

