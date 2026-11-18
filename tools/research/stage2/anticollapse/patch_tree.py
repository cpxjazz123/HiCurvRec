"""Install the assignment floor into a Stage2 candidate tree, then prove it works.

The candidate owns its mechanism; this patcher owns only the assignment step. It knows the
two quantiser families that exist in this repository:

    mixed    model/mixed_rq.py    MixedVQLevel._indices + encode_tokens   (the F0 family)
    euclid   model/layers.py      VQLayer._indices + the export argmin    (the frozen baseline)

Applying is idempotent and fails closed: a family whose file exists but whose anchors do not
match is reported, never silently skipped. Verification imports the *patched* tree in a fresh
interpreter and feeds it a cost matrix where eight codewords are cheap and every other one is
unreachable -- the shape that collapsed every arm in this investigation.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR_MODULE = HERE / "floor.py"

#: The installed file name inside a candidate tree.
TARGET_NAME = "assignment_floor.py"

CONSTANTS = '''#: The shared anti-collapse floor. Every assignment -- training and export -- must give
#: each codeword at least this many items. 1 is the smallest value that makes the
#: absorbing state unreachable; a batch of 4096 affords it for 256 codewords.
ASSIGNMENT_FLOOR = 1
#: True disables the Sinkhorn path as well, so training and export share one rule.
FORCE_CONSTRAINED_ALL = {force!r}
'''


@dataclass
class Family:
    """One quantiser implementation and the edits that give it the floor."""

    name: str
    module: str
    import_anchor: tuple[str, str]
    edits: list[tuple[str, str]] = field(default_factory=list)
    force_edit: tuple[str, str] | None = None
    #: The construction used by the verifier: a fresh layer of the patched class.
    layer_expr: str = ""


MIXED = Family(
    name="mixed",
    module="model/mixed_rq.py",
    import_anchor=(
        "from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance",
        "from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance\n"
        f"from {TARGET_NAME[:-3]} import (constrained_assignment, verify_assignment,\n"
        f"                                     verify_codebook)",
    ),
    edits=[
        # training assignment
        ("        return torch.argmin(distances, dim=-1)",
         "        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR)\n"
         "        return index.to(distances.device)"),
        # the call site: whatever `_indices` returned is checked, so a mechanism that
        # assigns somewhere else (e.g. re-enables the Sinkhorn branch) is caught here
        ("        index = self._indices(cost, infer_use_sk)\n"
         "        codes = self.get_code_embs()",
         "        index = self._indices(cost, infer_use_sk)\n"
         "        verify_assignment(index, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        codes = self.get_code_embs()\n"
         "        if not self.training:\n"
         "            verify_codebook(codes)"),
        # export assignment: the floor, then the geometric check on the table it just used
        ("            index = torch.argmin(cost, dim=-1)",
         "            index, _ = constrained_assignment(cost, ASSIGNMENT_FLOOR)\n"
         "            verify_codebook(layer.get_code_embs())"),
    ],
    force_edit=("        self.use_sk = sk_epsilon > 0 and sk_iters > 0",
                "        self.use_sk = (sk_epsilon > 0 and sk_iters > 0) and not FORCE_CONSTRAINED_ALL"),
    layer_expr="MixedVQLevel(8)",
)

EUCLID = Family(
    name="euclid",
    module="model/layers.py",
    import_anchor=(
        "from distances import build_distance",
        "from distances import build_distance\n"
        f"from {TARGET_NAME[:-3]} import (constrained_assignment, verify_assignment,\n"
        f"                                     verify_codebook)",
    ),
    edits=[
        ("        return torch.argmin(distances, dim=-1)",
         "        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR)\n"
         "        return index.to(distances.device)"),
        ("        embed_ind = self._indices(self._distances(latent), infer_use_sk)\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)",
         "        embed_ind = self._indices(self._distances(latent), infer_use_sk)\n"
         "        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        if not self.training:\n"
         "            verify_codebook(self.get_code_embs())\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)"),
        ("        embed_ind = torch.argmin(self._distances(x), dim=-1).view(*x.shape[:-1])",
         "        embed_ind, _ = constrained_assignment(self._distances(x), ASSIGNMENT_FLOOR)\n"
         "        embed_ind = embed_ind.view(*x.shape[:-1])"),
    ],
    force_edit=("        self.use_sk = sk_epsilon > 0 and sk_iters > 0",
                "        self.use_sk = (sk_epsilon > 0 and sk_iters > 0) and not FORCE_CONSTRAINED_ALL"),
    layer_expr="VQLayer(8, 4)",
)

FAMILIES = {family.name: family for family in (MIXED, EUCLID)}


class UnknownTree(RuntimeError):
    """The tree's quantiser is not one this patcher knows; it was not protected."""


class ReadOnlyBaseline(RuntimeError):
    """The frozen baseline is the copy source, never a target: it must stay byte-identical."""


def _refuse_frozen_baseline(tree: Path) -> None:
    """A baseline tree is read-only (CLAUDE.md 19b); patching it would destroy provenance."""
    parts = Path(tree).resolve().parts
    for index in range(len(parts) - 1):
        if parts[index] == "stage2_RQ-VAE" and parts[index + 1] == "baseline":
            raise ReadOnlyBaseline(
                "refusing to modify the frozen baseline: " + str(tree))


def _already_patched(text: str) -> bool:
    return "constrained_assignment" in text


def apply_floor(tree: Path, family: str | None = None, force_constrained: bool = True,
                floor: int = 1) -> dict:
    """Patch `tree` in place; returns what happened, per file.

    Idempotent: a tree that already imports the floor is left alone. `family` is inferred
    from which quantiser file exists when it is not given.
    """
    tree = Path(tree).resolve()
    _refuse_frozen_baseline(tree)
    if family is None:
        for name, candidate in FAMILIES.items():
            if (tree / candidate.module).is_file():
                family = name
                break
        else:
            raise UnknownTree(f"no known quantiser under {tree}")
    spec = FAMILIES[family]
    module = tree / spec.module
    if not module.is_file():
        raise UnknownTree(f"{spec.module} is missing under {tree}")

    text = module.read_text(encoding="utf-8")
    if _already_patched(text):
        return {"family": family, "module": spec.module, "changed": False,
                "reason": "already patched"}

    target = tree / TARGET_NAME
    shutil.copy2(FLOOR_MODULE, target)

    # constants first: the edits below reference them
    constants = CONSTANTS.replace("{force!r}", repr(bool(force_constrained)))
    constants = constants.replace("ASSIGNMENT_FLOOR = 1", f"ASSIGNMENT_FLOOR = {int(floor)}")
    anchor = "CURVATURE = 1.0" if family == "mixed" else "class VQLayer"
    if anchor not in text:
        raise UnknownTree(f"cannot place the constants in {spec.module}")
    text = text.replace(anchor, constants + "\n" + anchor, 1)

    edits = list(spec.edits)
    if force_constrained and spec.force_edit is not None:
        edits.append(spec.force_edit)
    edits.append(spec.import_anchor)

    applied = []
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise UnknownTree(
                f"{spec.module}: anchor appears {count} times, expected 1:\n{old[:100]}")
        text = text.replace(old, new)
        applied.append(old.splitlines()[0][:80])
    module.write_text(text, encoding="utf-8")
    return {"family": family, "module": spec.module, "changed": True,
            "edits": applied, "installed": TARGET_NAME}


VERIFY_TEMPLATE = """import json, sys, torch
from {module} import {cls}
layer = {expr}
cost = torch.rand(4096, 8) + 10.0
cost[:, :3] = torch.rand(4096, 3)          # only three codewords are cheap
index = layer._indices(cost, False)
print(json.dumps({{"used": int(torch.unique(index).numel()), "codewords": 8}}))
"""


def verify_tree(tree: Path, family: str | None = None, python: str | None = None) -> dict:
    """Run the patched quantiser on a collapsed cost matrix in a fresh interpreter.

    A fresh process is the point: it proves the patched module imports and behaves the way
    training will see it, not the way this test process happens to have cached it.
    """
    tree = Path(tree).resolve()
    _refuse_frozen_baseline(tree)
    if family is None:
        for name, candidate in FAMILIES.items():
            if (tree / candidate.module).is_file():
                family = name
                break
        else:
            raise UnknownTree(f"no known quantiser under {tree}")
    spec = FAMILIES[family]
    module = spec.module[len("model/"):-len(".py")]
    cls = spec.layer_expr.split("(")[0]
    script = VERIFY_TEMPLATE.format(module=f"model.{module}", cls=cls,
                                    expr=spec.layer_expr)
    result = subprocess.run([python or sys.executable, "-c", script], cwd=str(tree),
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise UnknownTree(f"verification failed in {tree}:\n{result.stderr[-2000:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["ok"] = payload["used"] == payload["codewords"]
    payload["tree"] = str(tree)
    payload["family"] = family
    return payload
