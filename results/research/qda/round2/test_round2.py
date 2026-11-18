"""The three engineering faults of round 2, each turned into a test.

1. `revive_l1` was appended to the trainer, i.e. defined *after* `main()` ran, and the
   arm died at the first revival epoch with NameError.  A built arm must therefore have
   the function before the entry point.
2. The observation schedule was rewritten in the trainer, but the gate reads the
   observer's `OBSERVATIONS`; the arm silently ran without its fine points.  A built
   arm's schedule must be the one in its manifest.
3. The lifecycle classifier is the only real logic here and it decides what the report
   calls never-used / died / revived, so it is pinned against a synthetic history.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TREES = HERE / "trees"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_revival_function_precedes_the_entry_point():
    source = (TREES / "B4_revival" / "train_rqvae.py").read_text(encoding="utf-8")
    assert "def revive_l1(" in source
    assert source.index("def revive_l1(") < source.index('if __name__ == "__main__":')


def test_every_arm_observes_its_declared_schedule():
    manifest = json.loads((HERE / "arm_manifest.json").read_text(encoding="utf-8"))
    arms = manifest["arms"]
    assert arms, "manifest describes no arms"
    for name, spec in arms.items():
        observer = (TREES / name / "qda_observer.py").read_text(encoding="utf-8")
        expected = "OBSERVATIONS = " + repr(tuple(spec["observations"]))
        assert expected in observer, f"{name}: observer schedule is not the manifest's"
        trainer = (TREES / name / "train_rqvae.py").read_text(encoding="utf-8")
        assert "from qda_observer import OBSERVATIONS as QDA_OBSERVATIONS" in trainer, (
            f"{name}: the trainer does not gate on the observer's schedule")


def test_lifecycle_separates_never_died_and_revived():
    """Four codes, four histories, one classifier."""
    telemetry = _load("qda_telemetry", TREES / "B0_replay" / "qda_telemetry.py")

    class Level:
        n_embed = 4

        def __init__(self):
            self.embed = torch.nn.Embedding(4, 2)

        def get_code_embs(self):
            return self.embed.weight

    class RQ:
        def __init__(self):
            self.levels = [Level(), Level(), Level()]

    class Model:
        def __init__(self):
            self.rq = RQ()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.training = True

        def train(self, mode=True):
            self.training = mode
            return self

        def eval(self):
            return self.train(False)

    model = Model()
    probe = torch.zeros(3, 2)
    tele = telemetry.EpochTelemetry(model, probe)
    # code 0: used every epoch; code 1: used, abandoned, used, dead at the end;
    # code 2: used, abandoned, used, still alive; code 3: never used at all.
    # The classes overlap on purpose: 1 is both `died` (not alive in the last epoch)
    # and `revived` (it had a gap), and the report's table relies on that.
    history = [[0, 1, 2], [0, 2], [0, 1], [0, 2]]
    for epoch, alive in enumerate(history, start=1):
        tokens = torch.zeros(len(alive), 3, dtype=torch.long)
        for row, code in enumerate(alive):
            tokens[row] = code
        tele.step(tokens, [None, None, None])
        tele.step_z(torch.zeros(len(alive), 2))
        tele.finish(epoch)

    report = tele.close(len(history))["level_1"]
    assert report["never_used"] == [3]
    assert report["died"] == [1]
    assert report["revived"] == [1, 2]
    assert report["active_epochs"][0] == len(history)
    assert report["first_used"][3] == -1 and report["last_used"][3] == -1
