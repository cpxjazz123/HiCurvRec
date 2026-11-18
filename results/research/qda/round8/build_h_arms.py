"""Build the two arms that carry the shared anti-collapse assignment.

Both arms are the frozen F0 tree plus exactly one idea: the assignment step enforces a
floor of one item per codeword. H1 keeps the baseline's Sinkhorn on the last level (it only
replaces the argmin path); H2 turns Sinkhorn off and uses the constrained rule everywhere,
so training and export share one rule.

Mechanism-agnostic by construction: the layer consumes the cost matrix every mechanism
already produces, and touches nothing else. Any candidate built on `MixedVQLevel` inherits
it by copying these two edits.
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

CONSTANTS = '''ARMS_PATCHED_MARKER = True
#: Round 8: every codeword must receive at least this many items at every assignment.
#: 1 is the smallest floor that makes the absorbing state unreachable; the batch (4096)
#: can afford it for 256 codewords at every level.
ASSIGNMENT_FLOOR = 1
#: When true the Sinkhorn path is disabled as well, so one rule serves training and export.
FORCE_CONSTRAINED_ALL = {force!r}
'''

MIXED_CONSTANTS = (
    """CURVATURE = 1.0""",
    """#: Round 8: the shared anti-collapse floor. Every assignment -- training and export --
#: must give each codeword at least this many items. 1 makes the absorbing state
#: unreachable; a batch of 4096 can afford it for 256 codewords at every level.
ASSIGNMENT_FLOOR = 1
#: When true the Sinkhorn path is disabled too, so training and export share one rule.
FORCE_CONSTRAINED_ALL = {force!r}

CURVATURE = 1.0""",
)

IMPORT = (
    """from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance""",
    """from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance
from qda_floor import constrained_assignment""",
)

USE_SK = (
    """        self.use_sk = sk_epsilon > 0 and sk_iters > 0""",
    """        self.use_sk = (sk_epsilon > 0 and sk_iters > 0) and not FORCE_CONSTRAINED_ALL""",
)

INDICES = (
    """        return torch.argmin(distances, dim=-1)""",
    """        # Round 8: the cheapest assignment that also keeps every codeword alive.
        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR)
        return index.to(distances.device)""",
)

EXPORT = (
    """            index = torch.argmin(cost, dim=-1)""",
    """            # The export must use the same rule as training, otherwise the floor is
            # a training-only fiction and the SID is unconstrained again.
            index, _ = constrained_assignment(cost, ASSIGNMENT_FLOOR)""",
)

ARMS = {"H1_floor_argmin": {"force": False}, "H2_floor_all": {"force": True}}


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
        pairs[1] = (pairs[1][0], pairs[1][1].replace('"round2"', '"round8"'))
        pairs += [
            ('ARM_NAME = "QDA_F0_seed42"', f'ARM_NAME = "{name}"'),
            ('EPOCHS              = 3000', 'EPOCHS              = 300'),
            ("""OBSERVATIONS   = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                  2250, 2500, 2750, 3000)""",
             f"OBSERVATIONS   = {SCREEN!r}"),
            ("QDA_GRADSRC_EPOCHS = 60",
             "QDA_GRADSRC_EPOCHS = 60"),
        ]

        patches = {"train_rqvae.py": build.rewrite(target / "train_rqvae.py", pairs, name)}
        patches["qda_observer.py"] = build.rewrite(
            target / "qda_observer.py", [build.schedule_patch(SCREEN)] + build.OBSERVER, name)
        mixed = target / "model" / "mixed_rq.py"
        mixed_pairs = [MIXED_CONSTANTS, IMPORT, INDICES, EXPORT]
        mixed_pairs[0] = (mixed_pairs[0][0],
                          mixed_pairs[0][1].replace("{force!r}", repr(spec["force"])))
        if spec["force"]:
            mixed_pairs.append(USE_SK)
        patches["model/mixed_rq.py"] = build.rewrite(mixed, mixed_pairs, name)
        shutil.copy2(ROUND2 / "telemetry_src.py", target / "qda_telemetry.py")
        shutil.copy2(HERE.parent / "round3" / "gradsrc_src.py", target / "qda_gradsrc.py")
        shutil.copy2(HERE / "floor_src.py", target / "qda_floor.py")
        patches["qda_floor.py"] = ["installed from round8/floor_src.py"]

        manifest["arms"][name] = {"epochs": 300, "observations": list(SCREEN),
                                  "patches": patches, "digests": build.digest_tree(target)}
        print(f"built {name}")

    (HERE / "arm_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                            encoding="utf-8")
    print("wrote arm_manifest.json")


if __name__ == "__main__":
    main()
