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
#: The baseline's balancing rule is never switched off (CLAUDE.md 19h): the floor is
#: applied to the assignment the Sinkhorn produces, not to a replaced rule.
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
    #: Every other quantiser class in the same module; a candidate may train through it
    #: (VQ_TYPE="ema"), so the probe must cover it too rather than assume the default.
    extra_layer_exprs: list[str] = field(default_factory=list)


MIXED = Family(
    name="mixed",
    module="model/mixed_rq.py",
    import_anchor=(
        "from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance",
        "from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance\n"
        f"from {TARGET_NAME[:-3]} import (constrained_assignment, verify_assignment,\n"
        f"                                     verify_codebook, verify_margins,\n"
        f"                                     verify_separation)",
    ),
    edits=[
        # training assignment
        ("        return torch.argmin(distances, dim=-1)",
         "        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR, "
         "counts_key=self)\n"
         "        return index.to(distances.device)"),
        ("            return torch.argmax(q, dim=-1)",
         "            index, _ = constrained_assignment(-q, ASSIGNMENT_FLOOR, "
             "counts_key=self)\n"
         "            return index.to(q.device)"),
        # the call site: whatever `_indices` returned is checked, so a mechanism that
        # assigns somewhere else (e.g. re-enables the Sinkhorn branch) is caught here
        # The call site is anchored on the single line that consumes the cost, so trees
        # with a different forward body still match: whatever `_indices` returns is
        # checked here, and the cost it came from is checked for ties.
        ("        index = self._indices(cost, infer_use_sk)",
         "        verify_margins(cost, rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        index = self._indices(cost, infer_use_sk)\n"
         "        verify_assignment(index, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        verify_codebook(self.get_code_embs(), distance=False)\n"
         "        verify_separation(self.get_code_embs(), cost, key=id(self))"),
        # export assignment: the floor, then the geometric check on the table it just used
        # optional: #232-style export assigns from a hoisted `cost`; the frozen baseline
        # exports through `embed_ind = torch.argmin(self._distances(x), ...)` below
        ("            index = torch.argmin(cost, dim=-1)",
         "            verify_margins(cost, rank_local=True)\n"   # export runs on rank 0 only
         "            index, _ = constrained_assignment(cost, ASSIGNMENT_FLOOR)\n"
         "            verify_codebook(layer.get_code_embs())", True),
    ],
    layer_expr="MixedVQLevel(8)",
)

EUCLID = Family(
    name="euclid",
    module="model/layers.py",
    import_anchor=(
        "from distances import build_distance",
        "from distances import build_distance\n"
        f"from {TARGET_NAME[:-3]} import (constrained_assignment, verify_assignment,\n"
        f"                                     verify_codebook, verify_margins,\n"
        f"                                     verify_separation)",
    ),
    edits=[
        ("        return torch.argmin(distances, dim=-1)",
         "        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR, "
         "counts_key=self)\n"
         "        return index.to(distances.device)"),
        ("            return torch.argmax(q, dim=-1)",
         "            index, _ = constrained_assignment(-q, ASSIGNMENT_FLOOR, "
             "counts_key=self)\n"
         "            return index.to(q.device)"),
        # `layers.py` reaches this line from two places, so the anchor carries the line
        # that follows it in `forward`; both must stay in step.
        # the anchor reads whatever the tree already computed as its cost, so a candidate that
        # hoists the metric into `cost` (issue #232: the supervision branch needs it) matches too
        # optional: only a tree that hoists the metric into `cost` (issue #232) has this
        # anchor. The frozen baseline calls `self._distances(latent)` inline instead, which
        # the next edit covers; `unguarded_sites` is what proves neither was missed.
        ("        embed_ind = self._indices(cost, infer_use_sk)\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)",
         "        verify_codebook(self.get_code_embs(), distance=False)\n"
         "        verify_separation(self.get_code_embs(), cost, key=id(self), "
         "rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        verify_margins(cost, rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        embed_ind = self._indices(cost, infer_use_sk)\n"
         "        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)", True),
        # the frozen baseline's VQLayer.forward: inline metric, `onehot` as the next line
        ("        embed_ind = self._indices(self._distances(latent), infer_use_sk)\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)",
         "        cost = self._distances(latent)\n"
         "        verify_codebook(self.get_code_embs(), distance=False)\n"
         "        verify_separation(self.get_code_embs(), cost, key=id(self), "
         "rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        verify_margins(cost, rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        embed_ind = self._indices(cost, infer_use_sk)\n"
         "        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        onehot = F.one_hot(embed_ind, self.n_embed)"),
        # `EMAVQLayer.forward` reaches the same line with a different next line, so it
        # needs its own anchor: a candidate that sets VQ_TYPE="ema" trains through this one
        ("        embed_ind = self._indices(self._distances(latent), infer_use_sk)\n"
         "        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)",
         "        cost = self._distances(latent)\n"
         "        verify_codebook(self.get_code_embs(), distance=False)\n"
         "        verify_separation(self.get_code_embs(), cost, key=id(self), "
         "rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        verify_margins(cost, rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"
         "        embed_ind = self._indices(cost, infer_use_sk)\n"
         "        verify_assignment(embed_ind, self.n_embed, ASSIGNMENT_FLOOR)\n"
         "        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)"),
        ("        embed_ind = torch.argmin(self._distances(x), dim=-1).view(*x.shape[:-1])",
         "        embed_ind, _ = constrained_assignment(self._distances(x), ASSIGNMENT_FLOOR)\n"
         "        embed_ind = embed_ind.view(*x.shape[:-1])"),
    ],
    layer_expr="VQLayer(8, 4)",
    extra_layer_exprs=["EMAVQLayer(8, 4)"],
)

FAMILIES = {family.name: family for family in (MIXED, EUCLID)}

#: A mechanism may keep a `argmin`/`argmax` that is *not* an assignment, but it has to be
#: declared here with a reason: `{"model/xxx.py": {"the line to allow": "why"}}`.
ALLOWED_SITES: dict[str, dict[str, str]] = {}

#: The spellings that pick a codeword. `argmin` is the one the baseline uses; the others
#: reach the same result and would otherwise slip past the scan.
ASSIGNMENT_PATTERNS = ("argmin(", "argmax(", "torch.topk(", ".topk(", ".min(dim=", ".max(dim=")

#: The files a mechanism can assign codewords in *through the path this family patches*:
#: the quantiser module itself, the model wrapper, and the trainer. Other quantiser modules
#: may exist as dead code in a derived tree (`model/layers.py` sits next to a mixed arm) and
#: are deliberately not scanned -- a false refusal there would block a legitimate run.
SCANNED = ("model/model.py", "train_rqvae.py")


def floor_constant(tree: Path, family: str) -> dict:
    """The floor is a number, and a zero floor proves nothing: check it, do not read it.

    Setting `ASSIGNMENT_FLOOR = 0` leaves every guard call in place -- `verify_assignment`
    passes trivially, the probe sees all codewords -- while the assignment silently goes
    back to a bare `argmin`. That is a guard removal spelled as an integer.
    """
    import re

    module = Path(tree) / FAMILIES[family].module
    if not module.is_file():
        raise UnknownTree("no quantiser module at " + str(module))
    values = re.findall(r"^\s*ASSIGNMENT_FLOOR\s*=\s*(\d+)", module.read_text(encoding="utf-8"),
                        flags=re.MULTILINE)
    if not values:
        raise UnknownTree("the floor constant is missing from " + str(module))
    value = min(int(entry) for entry in values)
    if value < 1:
        raise UnknownTree("the floor was disabled (ASSIGNMENT_FLOOR = " + str(value)
                          + ") in " + str(module))
    return {"ASSIGNMENT_FLOOR": value}


def unguarded_sites(tree: Path, family: str | None = None) -> list[dict]:
    """Every `argmin`/`argmax` left in the paths that decide a codeword.

    The floor works by replacing the assignment, so a patched tree contains no raw
    `argmin`/`argmax` over a cost matrix at all. Any that appears afterwards is either a
    new assignment the guard does not see -- exactly how `EMAVQLayer` escaped for one
    revision -- or a use that must be declared in `ALLOWED_SITES` with a reason.
    """
    tree = Path(tree)
    found: list[dict] = []
    scanned = list(SCANNED)
    if family in FAMILIES:
        scanned.insert(0, FAMILIES[family].module)
    else:
        scanned = [spec.module for spec in FAMILIES.values()] + scanned
    for relative in scanned:
        path = tree / relative
        if not path.is_file():
            continue
        allowed = ALLOWED_SITES.get(relative, {})
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not any(pattern in line for pattern in ASSIGNMENT_PATTERNS):
                continue
            if any(pattern in line for pattern in allowed):
                continue
            found.append({"file": relative, "line": number, "text": line.strip()[:120]})
    return found


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
    """Both halves must be there: the repair *and* the check on what it returned."""
    return "constrained_assignment(" in text and "verify_assignment(" in text


#: Call-site forms that a later revision changed. A tree patched by an older guard keeps
#: working only if these are applied, so they run before the "already patched" return.
MIGRATIONS = (
    # The export anchor (12 spaces) is a plain function without `self`; it must be fixed
    # before the 8-space rule, and both are anchored on a newline so the shorter pattern
    # cannot match inside the longer one.
    ("\n            verify_margins(cost)\n", "\n            verify_margins(cost, rank_local=True)\n"),
    ("\n        verify_margins(cost)\n",
     '\n        verify_margins(cost, rank_local=getattr(self, "_skip_ddp_reduce", False))\n'),
    # the spacing check itself was added after the first installs: give it both the import
    # and its call site, or an already-patched tree never receives the new check
    ("verify_codebook, verify_margins)",
     "verify_codebook, verify_margins,\n                                     verify_separation)"),
    # the spacing check gained the rank-local flag after the first installs
    ("\n        verify_separation(self.get_code_embs(), cost, key=id(self))\n",
     "\n        verify_separation(self.get_code_embs(), cost, key=id(self), "
     "rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n"),
    # repair for trees migrated by the first (buggy) ordering: export has no `self`
    ('\n            verify_margins(cost, rank_local=getattr(self, "_skip_ddp_reduce", False))\n',
     "\n            verify_margins(cost, rank_local=True)\n"),
)


CB_ANCHOR = "\n        verify_codebook(self.get_code_embs(), distance=False)\n"
SEP_CALL = ("\n        verify_separation(self.get_code_embs(), cost, key=id(self), "
            "rank_local=getattr(self, \"_skip_ddp_reduce\", False))\n")


def _migrate(text: str) -> str:
    # Layer objects own weak usage history; recycled integer ids cannot own it.
    text = text.replace("counts_key=id(self)", "counts_key=self")
    if "counts_key=self" not in text:
        text = text.replace("constrained_assignment(distances, ASSIGNMENT_FLOOR)",
                            "constrained_assignment(distances, ASSIGNMENT_FLOOR, "
                            "counts_key=self)")
        text = text.replace("constrained_assignment(-q, ASSIGNMENT_FLOOR)",
                            "constrained_assignment(-q, ASSIGNMENT_FLOOR, "
                            "counts_key=self)")
    for old, new in MIGRATIONS:
        text = text.replace(old, new)
    # A check added after the first installs has to reach them too -- once. The guard is
    # the call itself: a second pass must not stack another check on the same line.
    if "verify_separation(self" not in text and CB_ANCHOR in text:
        text = text.replace(CB_ANCHOR, CB_ANCHOR + SEP_CALL, 1)
    return text


def _guard_was_removed(text: str) -> bool:
    """Only a *patched* file that lost its check counts as tampering.

    A mechanism may legitimately call `constrained_assignment` in its own quantiser before
    the guard is installed -- that is the documented way to protect a custom assignment --
    so the marker for "this file was patched" is the guard's own verification call, not the
    helper it shares.
    """
    return ("constrained_assignment(" in text and "verify_codebook(" in text
            and "verify_assignment(" not in text)


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
    migrated = _migrate(text)
    if migrated != text:
        module.write_text(migrated, encoding="utf-8")
        text = migrated
    if _guard_was_removed(text):
        # Someone kept the floor call and deleted the check that proves it worked. That is
        # indistinguishable from tampering, and it must not be re-applied silently.
        raise UnknownTree(
            "the assignment check was removed from " + str(module)
            + "; restore the tree from its baseline copy")
    if _already_patched(text):
        # A tree that is already patched must still pass the integrity checks: the guards
        # can be weakened afterwards (a zeroed constant, a second assignment site), and a
        # tree is installed once but trained many times.
        floor_constant(tree, family)
        sites = unguarded_sites(tree, family)
        if sites:
            raise UnknownTree(
                "the tree still assigns without the guard: " + "; ".join(
                    f"{site['file']}:{site['line']} {site['text']}" for site in sites[:4]))
        # The quantiser edits are idempotent, but the installed module is a copy: a tree
        # patched before the algorithm changed would keep running the old code forever.
        import hashlib

        installed = tree / TARGET_NAME
        stale = (not installed.is_file()
                 or hashlib.sha256(installed.read_bytes()).hexdigest()
                 != hashlib.sha256(FLOOR_MODULE.read_bytes()).hexdigest())
        if stale:
            shutil.copy2(FLOOR_MODULE, installed)
            return {"family": family, "module": spec.module, "changed": "module",
                    "reason": "the installed floor was older than the source"}
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
    edits.append(spec.import_anchor)

    applied = []
    for edit in edits:
        old, new = edit[0], edit[1]
        optional = len(edit) > 2 and bool(edit[2])
        count = text.count(old)
        if count == 0 and optional:
            # this anchor belongs to another revision of the same quantiser; the
            # unguarded-site scan below still runs, so nothing is silently left open
            continue
        if count != 1:
            raise UnknownTree(
                f"{spec.module}: anchor appears {count} times, expected 1:\n{old[:100]}")
        text = text.replace(old, new)
        applied.append(old.splitlines()[0][:80])
    module.write_text(text, encoding="utf-8")
    floor_constant(tree, family)
    sites = unguarded_sites(tree, family)
    if sites:
        raise UnknownTree(
            "the tree still assigns without the guard: "
            + "; ".join(f"{site['file']}:{site['line']} {site['text']}" for site in sites[:4]))
    return {"family": family, "module": spec.module, "changed": True,
            "edits": applied, "installed": TARGET_NAME}


VERIFY_TEMPLATE = """import json, sys, torch
from {module} import {imports}

def unique(index):
    return int(torch.unique(index.reshape(-1)).numel())

def probe(layer):
    # 1. `_indices` on a collapsed cost: only three codewords are cheap.
    cost = torch.rand(4096, 8) + 10.0
    cost[:, :3] = torch.rand(4096, 3)
    used_indices = unique(layer._indices(cost, False))
    # 2. the baseline balances with Sinkhorn (CLAUDE.md 19h): the floor must hold on the
    #    assignment that balancing produces, not only on a bare argmin.
    layer.use_sk = True
    layer.sk_epsilon = 0.003
    layer.sk_iters = 5
    used_balanced = unique(layer._indices(cost, False))
    # 3. `forward` on a collapsed *codebook*: five of eight codewords are unreachable, but
    #    distinct from each other. A mechanism that replaced the assignment inside forward
    #    is caught here, because the floor lives in the path it no longer takes.
    codes = layer.get_code_embs()
    with torch.no_grad():
        codes[3:] = 1000.0 + torch.arange(codes.shape[0] - 3,
                                          dtype=codes.dtype).unsqueeze(1)
{forward_call}
    return {{"used": used_indices, "used_balanced": used_balanced,
            "used_forward": unique(index), "codewords": 8}}

report = {{}}
for name, layer in [{layers}]:
    # (each family class is probed below)
    try:
        report[name] = probe(layer)
    except Exception as error:
        print("GUARD " + name + ": " + type(error).__name__ + ": " + str(error)[:200])
        raise
# 4. the *wired* model: a mechanism may add a quantiser module and connect it through
#    model.py. Nothing static can see that, but the constructed model can.
wiring = {{}}
try:
    import train_rqvae as trainer
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=32, distance=trainer.DISTANCE)
    levels = [module for module in model.rq.modules() if hasattr(module, "_indices")]
    wiring = {{"levels": len(levels),
               "modules": sorted({{type(module).__module__ for module in levels}}),
               "classes": sorted({{type(module).__name__ for module in levels}})}}
except Exception as error:                     # noqa: BLE001
    wiring = {{"error": type(error).__name__ + ": " + str(error)[:120]}}
print(json.dumps({{"layers": report, "codewords": 8, "wiring": wiring}}))
"""

#: How each family drives `forward` in the probe: (setup, call).
FORWARD_PROBE = {
    "mixed": ("e = torch.randn(4096, 16) * 0.01\nh = torch.randn(4096, 16) * 0.01\n"
              "a = torch.full((4096,), 0.5)\nout = layer(e, h, a)",
              "index = [value for value in out if torch.is_tensor(value)"
              " and value.dtype in (torch.int64, torch.int32) and value.numel() == 4096][-1]"),
    "euclid": ("dim = layer.get_code_embs().shape[1]\n"
               "x = torch.randn(4096, dim) * 0.01\n"
               "out = layer(x)",
               "index = [value for value in out if torch.is_tensor(value)"
               " and value.dtype in (torch.int64, torch.int32) and value.numel() == 4096][-1]"),
}


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
    try:
        floor_constant(tree, family)
    except UnknownTree as error:
        return {"ok": False, "used": 0, "used_forward": 0, "layers": {}, "codewords": 8,
                "tree": str(tree), "family": family, "reason": str(error)}
    sites = unguarded_sites(tree, family)
    if sites:
        return {"ok": False, "used": 0, "used_forward": 0, "layers": {},
                "codewords": 8, "tree": str(tree), "family": family,
                "reason": "an assignment site is not guarded: " + "; ".join(
                    f"{site['file']}:{site['line']}" for site in sites[:4])}
    module = spec.module[len("model/"):-len(".py")]
    setup, pick = FORWARD_PROBE[family]
    expressions = [spec.layer_expr] + list(spec.extra_layer_exprs)
    names = [expression.split("(")[0] for expression in expressions]
    script = VERIFY_TEMPLATE.format(
        module=f"model.{module}", imports=", ".join(names),
        layers=", ".join(f'("{name}", {expression})'
                         for name, expression in zip(names, expressions)),
        forward_call="\n".join("    " + line
                               for line in (setup + "\n" + pick).splitlines()))
    result = subprocess.run([python or sys.executable, "-c", script], cwd=str(tree),
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        # A guard refusal is a verdict, not a broken tree: the patched call site noticed
        # that the assignment it was handed violates the floor and stopped the step.
        if "CodebookCollapse" in result.stderr:
            reason = [line for line in result.stderr.splitlines()
                      if "CodebookCollapse" in line][-1]
            return {"ok": False, "used": 0, "used_forward": 0, "layers": {},
                    "codewords": 8, "reason": reason.strip(), "tree": str(tree),
                    "family": family}
        raise UnknownTree(f"verification failed in {tree}:\n{result.stderr[-2000:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    wiring = payload.get("wiring", {})
    expected = "model." + spec.module[len("model/"):-len(".py")]
    # Best effort, reported not enforced: a tree whose model class cannot be built from
    # the config alone (the mixed family builds `MixedRQVAE` with a curvature argument) is
    # not evidence of a hole, and refusing every such tree would be wrong. What the check
    # does catch -- a quantiser from an unpatched module wired into a buildable model --
    # is reported as a warning the caller can act on.
    if "error" in wiring:
        payload["wiring_warning"] = ("not inspected: " + str(wiring["error"]))
    else:
        foreign = [name for name in wiring.get("modules", []) if name != expected]
        if foreign:
            payload["wiring_warning"] = ("a quantiser from another module is wired in: "
                                         + ", ".join(foreign))
    payload["ok"] = all(entry[key] == payload["codewords"]
                        for entry in payload["layers"].values()
                        for key in ("used", "used_balanced", "used_forward"))
    if not payload["ok"]:
        short = [name + " (" + str(entry.get("used_forward")) + "/"
                 + str(payload["codewords"]) + " through forward)"
                 for name, entry in payload["layers"].items()
                 if entry.get("used_forward") != payload["codewords"]]
        payload["reason"] = ("the probe did not cover every codeword: "
                             + "; ".join(short or [str(payload["layers"])]))
    # The worst layer is the tree's answer: a per-class view stays available in `layers`.
    for key in ("used", "used_balanced", "used_forward"):
        payload[key] = min(entry[key] for entry in payload["layers"].values())
    payload["tree"] = str(tree)
    payload["family"] = family
    return payload
