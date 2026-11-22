"""Regression test: a stale duplicate must not decide the learned collision.

On 2026-11-08 a duplicate run of the same arm kept appending to the shared
``training_metrics.jsonl`` after the delivered run had finished. The gate's
learned collision is the LAST ``train`` row that carries a ``collision`` field,
so the stale duplicate's number was read instead of the delivered run's and the
arm was refused for "export-time post-processing" it never did.

The recovered single-run stream (``run_metrics.jsonl``) must win. No torch
needed; run with the system interpreter:

    python3 -m tools.research.stage2.tests.test_learned_collision_split
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

GATE_PATH = Path(__file__).resolve().parents[4] / "tools" / "genrec-mcp" / "stage2_quality_gate.py"


def _gate():
    spec = importlib.util.spec_from_file_location("stage2_quality_gate_test", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_the_delivered_run_beats_a_stale_duplicate() -> str:
    """The last row of the shared file belongs to a duplicate; the split file wins."""
    delivered = 0.208077
    stale = 0.172490
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = root / "results" / "stage2_RQ-VAE" / "TIGER_X"
        tree = root / "stage2_RQ-VAE" / "TIGER_X" / "logs"
        (result / "logs").mkdir(parents=True)
        tree.mkdir(parents=True)
        _write(tree / "training_metrics.jsonl",
               [{"event": "train", "epoch": 3000, "collision": delivered},
                {"event": "train", "epoch": 900, "collision": stale}])
        _write(tree / "run_metrics.jsonl",
               [{"event": "train", "epoch": 3000, "collision": delivered}])
        gate = _gate()
        original = gate.REPOSITORY
        gate.REPOSITORY = root
        try:
            value = gate.learned_collision(result)
        finally:
            gate.REPOSITORY = original
        assert value == delivered, \
            "the stale duplicate's collision %r decided the gate, not %r" % (value, delivered)
    return "the recovered single-run stream decides the learned collision"


def main() -> int:
    try:
        print("PASS test_the_delivered_run_beats_a_stale_duplicate:",
              test_the_delivered_run_beats_a_stale_duplicate(), flush=True)
    except AssertionError as exc:
        print("FAIL test_the_delivered_run_beats_a_stale_duplicate:", exc, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
