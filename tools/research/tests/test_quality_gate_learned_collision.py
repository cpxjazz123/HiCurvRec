"""The collision gate must be able to read the run's own metrics stream.

Two faults of the gate are reproduced here, because a fix without a test that
fails on the old behaviour is a fix that can be undone:

* it looked for the metrics ONLY in the result directory's logs, while the
  current trainer writes them into the TREE's logs (the arm's source directory
  under stage2_RQ-VAE/).  Every arm trained with the current trainer therefore
  failed the gate closed with "the learned collision of this SID could not be
  read from the run's metrics";
* it compared the trainer's collision (the share of distinct triplets) with the
  gate's own item-weighted collision, so an arm whose SID was never touched was
  refused for a disagreement between two different measures.

The gate is read from its file, the way the server reads it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _gate():
    spec = importlib.util.spec_from_file_location(
        "genrec_quality_gate", REPO / "tools" / "genrec-mcp" / "stage2_quality_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(directory: Path, collision: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"event": "train", "epoch": 10, "collision": collision},
            {"event": "train_end", "final_collision": collision}]
    (directory / "training_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_the_learned_collision_is_read_from_the_result_directory(tmp_path):
    gate = _gate()
    result = tmp_path / "TIGER_something"
    _metrics(result / "logs", 0.25)
    assert gate.learned_collision(result) == 0.25


def test_the_learned_collision_is_read_from_the_trees_logs(tmp_path, monkeypatch):
    """The layout the current trainer writes: the arm's own source tree."""
    gate = _gate()
    repository = tmp_path / "repo"
    tree = repository / "stage2_RQ-VAE" / "TIGER_something" / "logs"
    _metrics(tree, 0.0123)
    monkeypatch.setattr(gate, "REPOSITORY", repository)
    result = repository / "results" / "stage2_RQ-VAE" / "TIGER_something"
    result.mkdir(parents=True, exist_ok=True)
    assert gate.learned_collision(result) == 0.0123


def test_the_gate_compares_the_trainers_own_measure():
    """The trainer's collision and the exported table's must be like with like."""
    gate = _gate()
    sid = {"0": [1, 2, 3, 0], "1": [1, 2, 3, 1], "2": [4, 5, 6, 2]}
    raw = gate.exported_raw_collision(sid)
    # one distinct triplet out of three items
    assert abs(raw - (1.0 - 2.0 / 3.0)) < 1e-12
    learned = 1.0 - 2.0 / 3.0
    # the candidate's own collision is under the gate and agrees with the log
    assert gate.collision_gate({"l3_collision": 0.2}, learned, raw) is None
    # a SID that was post-processed after training disagrees with the log
    assert gate.collision_gate({"l3_collision": 0.2}, learned, raw + 0.2) is not None
