"""Build the round-6 codebook-following arms from the frozen F0 tree.

The encoder keeps its ordinary dynamics and the quantiser keeps argmin; the only change
is that the codewords are moved by the encoder's own distribution shift, measured from the
optimizer step itself.

    F1_dead_shift   the codewords that no item selected this step receive the mean move
    F2_rigid_shift  every codeword receives it, on top of its gradient

No constant is introduced: the displacement is whatever the assigned codewords did in
that step, so there is nothing to calibrate and nothing to tune.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "round2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = _load("qda_build", ROUND2 / "build_arms.py")
SCREEN = build.SCREEN
OUT = HERE / "trees"

IMPORT = (
    """from qda_gradsrc import measure""",
    """from qda_gradsrc import measure
from qda_shift import apply_shift, assigned_mean_move, snapshot as code_snapshot""",
)

CONSTANTS = '''ARMS_PATCHED_MARKER = True
#: Round 6: whether the rigid shift also moves the codewords the step already reached.
SHIFT_ASSIGNED = {assigned!r}
'''

STEP = (
    """            optimizer.step()""",
    """            # Round 6: the codewords follow the encoder's own displacement this step.
            _codes_before = code_snapshot(raw_module.rq.levels)
            optimizer.step()
            _moves = assigned_mean_move(
                _codes_before,
                [layer.get_code_embs().detach() for layer in raw_module.rq.levels],
                tokens, len(raw_module.rq.levels))
            apply_shift(raw_module.rq.levels, tokens, _moves, SHIFT_ASSIGNED)""",
)

ARMS = {
    "F1_dead_shift": {"assigned": False},
    "F2_rigid_shift": {"assigned": True},
}


def main() -> None:
    manifest = {"source": str(build.SOURCE), "arms": {}}
    for name, spec in ARMS.items():
        target = OUT / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build.SOURCE, target,
                        ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))

        pairs = [p for p in build.COMMON]
        pairs[0] = (pairs[0][0], pairs[0][1].replace("{full!r}", repr(SCREEN))
                    .replace("{gradsrc!r}", "False"))
        pairs[1] = (pairs[1][0], pairs[1][1].replace('"round2"', '"round6"'))
        pairs += [
            ('ARM_NAME = "QDA_F0_seed42"', f'ARM_NAME = "{name}"'),
            ('EPOCHS              = 3000', 'EPOCHS              = 300'),
            ("""OBSERVATIONS   = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                  2250, 2500, 2750, 3000)""",
             f"OBSERVATIONS   = {SCREEN!r}"),
            ("QDA_GRADSRC_EPOCHS = 60",
             "QDA_GRADSRC_EPOCHS = 60\n" + CONSTANTS.replace("{assigned!r}",
                                                             repr(spec["assigned"]))),
            IMPORT,
            STEP,
        ]

        patches = {"train_rqvae.py": build.rewrite(target / "train_rqvae.py", pairs, name)}
        patches["qda_observer.py"] = build.rewrite(
            target / "qda_observer.py", [build.schedule_patch(SCREEN)] + build.OBSERVER, name)
        shutil.copy2(ROUND2 / "telemetry_src.py", target / "qda_telemetry.py")
        shutil.copy2(HERE.parent / "round3" / "gradsrc_src.py", target / "qda_gradsrc.py")
        shutil.copy2(HERE / "codebook_shift_src.py", target / "qda_shift.py")
        patches["qda_shift.py"] = ["installed from round6/codebook_shift_src.py"]

        manifest["arms"][name] = {"epochs": 300, "observations": list(SCREEN),
                                  "patches": patches, "digests": build.digest_tree(target)}
        print(f"built {name}")

    (HERE / "arm_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                            encoding="utf-8")
    print("wrote arm_manifest.json")


if __name__ == "__main__":
    main()
