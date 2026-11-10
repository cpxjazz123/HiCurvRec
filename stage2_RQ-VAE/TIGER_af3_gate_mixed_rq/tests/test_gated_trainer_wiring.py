"""The gated arm through the real entry point, on synthetic data only.

This is the wiring check the mechanism needs before a GPU minute is spent: the
trainer's own ``main`` must read the frozen table, route the batch's item ids
through the gate, add the three structural terms to the same scalar, export the
same codes the gated assignment produced, and save a checkpoint that restores
into a gated model. The data are eight synthetic rows; nothing here is a Stage2
run.
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

from model import structure_table  # noqa: E402

ROWS = 8
DIM = 8


def _load_trainer():
    spec = importlib.util.spec_from_file_location("issue149_synthetic_gated_trainer",
                                                  TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def _write_table(path: Path, item_count: int = ROWS) -> Path:
    """A tiny frozen table with the same schema and a manifest that pins it."""
    generator = np.random.default_rng(149)
    values = generator.normal(size=(item_count, structure_table.DESCRIPTOR_DIMS)).astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    observed[item_count - 1, 6] = False
    values[item_count - 1, 6] = 0.0
    metadata = {"n_items": item_count, "issue": 149, "descriptor_names":
                list(structure_table.DESCRIPTOR_NAMES), "items_without_structure": 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, values=values, raw=values.copy(), observed=observed,
        mean=np.zeros(structure_table.DESCRIPTOR_DIMS), std=np.ones(structure_table.DESCRIPTOR_DIMS),
        item_ids=np.arange(item_count, dtype=np.int64),
        probe_items=np.arange(item_count, dtype=np.int32),
        community=np.zeros(item_count, dtype=np.int64),
        triangles=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        cross_pairs=np.array([[2, 4], [3, 5]], dtype=np.int32),
        same_pairs=np.array([[0, 1], [4, 5]], dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    manifest = {"table": str(path), "table_sha256": structure_table.sha256_of(path),
                "metadata": metadata}
    structure_table.manifest_path(path).write_text(json.dumps(manifest, sort_keys=True),
                                                   encoding="utf-8")
    return path


def _configure(monkeypatch, tmp_path):
    trainer = _load_trainer()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constants = {
        "ARM_VARIANT": "AF3GATE", "CURVATURE": 1.0, "OUTER_LOOP": False, "AUX_AF3": False,
        "EPOCHS": 1, "BATCH_SIZE_PER_RANK": 4, "HIDDEN_SIZES": (8,),
        "CODEBOOK_SIZE": (4, 4, 4), "NUM_WORKERS": 0, "EVAL_INTERVAL": 1, "WARMUP_EPOCHS": 0,
        "RESULT_DIR": tmp_path / "arm", "CHECKPOINT": tmp_path / "arm" / "snapshot.pth",
        "OUTPUT_SID": tmp_path / "arm" / "sids.npy",
        "OUTPUT_JSON": tmp_path / "arm" / "sids.json",
        "LOG_DIR": tmp_path / "arm" / "logs",
        "METRICS_PATH": tmp_path / "arm" / "logs" / "metrics.jsonl",
        "STRUCTURE_TABLE": _write_table(tmp_path / "structure_table.npz"),
    }
    for name, value in constants.items():
        monkeypatch.setattr(trainer, name, value)
    embeddings = np.random.default_rng(203).normal(size=(ROWS, DIM)).astype(np.float32) * 0.1
    frame = pd.DataFrame({"target": np.arange(ROWS),
                          "history": [[(i + 1) % ROWS] for i in range(ROWS)]})
    monkeypatch.setattr(trainer, "load_embeddings", lambda path: embeddings.copy())
    monkeypatch.setattr(trainer.pd, "read_parquet", lambda *args, **kwargs: frame.copy())
    return trainer, embeddings


def test_gated_entry_point_exports_what_the_gate_produced(monkeypatch, tmp_path):
    trainer, embeddings = _configure(monkeypatch, tmp_path)
    trainer.main()
    assert trainer.OUTPUT_JSON.is_file() and trainer.CHECKPOINT.is_file()
    events = [json.loads(line) for line in trainer.METRICS_PATH.read_text(
        encoding="utf-8").splitlines()]
    start = next(event for event in events if event["event"] == "train_start")
    assert start["arm_variant"] == "AF3GATE"
    assert start["structure_switches"] == {"behavior_off": False, "geometry_off": False,
                                           "both_on": True}
    assert start["structure_table_sha256"] == structure_table.sha256_of(trainer.STRUCTURE_TABLE)
    assert start["structure_lambdas"] == {"closure": 1.0, "bridge": 1.0, "usage": 1.0}
    structure_events = [event for event in events if event["event"] == "structure"]
    assert structure_events, "the trainer must record the three terms it stepped on"
    last = structure_events[-1]
    for name in ("closure", "bridge", "usage", "total"):
        assert 0.0 <= last[name] <= 1.0 + 1e-6, (name, last[name])
    # the probe subgraph covers its own induced sets completely, every epoch
    assert last["triangles_used"] == last["triangles_total"] * last["steps"]
    assert last["same_pairs_used"] == last["same_pairs_total"] * last["steps"]
    assert last["cross_pairs_used"] == last["cross_pairs_total"] * last["steps"]
    assert last["triangles_total"] == 2
    assert last["probe_items"] >= 1
    assert [row["level"] for row in last["levels"]] == [1, 2, 3]
    for row in last["levels"]:
        assert 0.0 < row["alpha_mean"] < 1.0
        assert 0.0 < row["alpha_std"]
        assert row["codes_used"] >= 1
        assert 0.0 < row["max_bucket_share"] <= 1.0
        assert 0.0 <= row["usage_entropy_normalised"] <= 1.0
        assert "euclidean_over_curved" in row
        assert -1.0 <= row["max_abs_spearman_alpha_vs_descriptor"] <= 1.0

    # the export is the gated assignment, restored from the checkpoint
    checkpoint = torch.load(trainer.CHECKPOINT, map_location="cpu", weights_only=True)
    from model.model import MixedRQVAE
    restored = MixedRQVAE.from_checkpoint(trainer._tokenizer_config(),
                                          checkpoint["state_dict"], in_dim=DIM,
                                          structure_table=trainer.STRUCTURE_TABLE).eval()
    with torch.no_grad():
        expected = restored.get_indices(torch.from_numpy(embeddings)).cpu().numpy()
    sid = np.load(trainer.OUTPUT_SID)
    np.testing.assert_array_equal(sid[:, :3], expected)
    assert len(np.unique(sid, axis=0)) == ROWS


def test_gated_entry_point_rejects_a_table_that_covers_too_few_items(monkeypatch, tmp_path):
    trainer, _ = _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer, "STRUCTURE_TABLE",
                        _write_table(tmp_path / "short_table.npz", item_count=ROWS - 3))
    with pytest.raises(ValueError, match="does not cover the embedding rows"):
        trainer.main()


def test_behaviour_off_makes_the_export_item_independent(monkeypatch, tmp_path):
    trainer, embeddings = _configure(monkeypatch, tmp_path)
    from model import structure_gate
    monkeypatch.setattr(structure_gate, "BEHAVIOR_OFF", True)
    trainer.main()
    events = [json.loads(line) for line in trainer.METRICS_PATH.read_text(
        encoding="utf-8").splitlines()]
    last = [event for event in events if event["event"] == "structure"][-1]
    # with the item-independent descriptor the gate is one constant per level, so
    # a batch of items shows zero spread on every level
    for row in last["levels"]:
        # the GEMM's accumulation order leaves ~1e-8, not a per-item value
        assert row["alpha_std"] < 1e-6, row
