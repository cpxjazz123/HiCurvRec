"""The negative control: break the mechanism on purpose, and require the test to see it.

A template that still passes a deliberately inert mechanism measures nothing. The
contract names the edit that breaks it and the template that must fail; this module
applies the edit to a throwaway copy and checks that the template failed.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from . import contracts


#: The edits a contract may name, as text substitutions on the candidate's source.
EDITS = {
    # 1 the behaviour signal stops reaching the geometry
    "constant radius": ("model/behaviour_radius.py",
                        "    values = torch.as_tensor(scale, dtype=torch.float64).flatten()",
                        "    return torch.full((24,), 0.5)\n"
                        "    values = torch.as_tensor(scale, dtype=torch.float64).flatten()"),
    # 2 the curved score becomes the Euclidean one, in the path the assignment
    # actually uses (the layer's own cost), not in a helper nobody calls
    "euclidean kernel": ("model/layers.py",
                         "        e = project(self.point_of(self.get_code_embs()), 1.0)",
                         "        e = self.get_code_embs()"),
    # 3 the gradient path from the behaviour to the encoder is cut
    "cut gradient path": ("model/model.py",
                          "        return radius.unsqueeze(-1) * direction",
                          "        return radius.detach().unsqueeze(-1) * direction"),
}


def apply_edit(tree: Path, edit: str) -> dict:
    """Apply a named edit to a COPY of the tree and return where it lives."""
    key = str(edit).strip().lower()
    if key not in EDITS:
        raise ValueError("unknown negative-control edit: " + str(edit))
    relative, old, new = EDITS[key]
    scratch = Path(tempfile.mkdtemp(prefix="v2_negative_"))
    copy = scratch / Path(tree).name
    shutil.copytree(Path(tree), copy)
    target = copy / relative
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError("the negative-control edit did not match " + relative)
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return {"tree": copy, "edited": relative, "scratch": scratch}


def run(tree: Path, contract: dict) -> dict:
    """Break the mechanism on purpose and require the named template to fail.

    Every control of the contract is run: a template that still passes a
    deliberately inert mechanism is measuring nothing, and that is a result, not
    a remark.
    """
    controls = contract.get("negative_controls") or []
    if not controls:
        control = contract.get("negative_control") or {}
        controls = [control] if control else []
    if not controls:
        return {"status": "NOT_RUN", "detail": "the contract declares no negative control"}
    outcomes = []
    for control in controls:
        outcome = _run_one(tree, contract, control)
        outcomes.append(outcome)
    failed = [entry for entry in outcomes if entry.get("status") != "PASS"]
    return {"status": "PASS" if not failed else "FAIL",
            "detail": "; ".join("%s -> %s" % (entry.get("edit"), entry.get("status"))
                                for entry in outcomes),
            "controls": outcomes}


def _run_in_process(tree: Path, contract: dict) -> dict:
    """Run the contract's templates on a tree, in a fresh interpreter."""
    import json
    import subprocess
    import sys

    from . import deterministic

    script = (
        "import json, sys;"
        "sys.path.insert(0, " + repr(str(deterministic.REPO)) + ");"
        "from tools.research.pipeline_v2 import contracts as c;"
        "contract = c.load_contract(" + repr(str(contract.get('__path__') or '')) + ");"
        "print('RESULT:' + json.dumps(c.run_tests(" + repr(str(tree)) + ", contract)))")
    if not contract.get("__path__"):
        return contracts.run_tests(tree, contract)
    done = subprocess.run([str(deterministic.PY_STAGE2), "-c", script],
                          cwd=str(deterministic.REPO), capture_output=True, text=True,
                          timeout=900)
    marked = [line for line in (done.stdout or "").splitlines()
              if line.startswith("RESULT:")]
    if not marked:
        return {"tests": {}, "ran": False,
                "error": (done.stdout + done.stderr)[-600:]}
    return json.loads(marked[-1][len("RESULT:"):])


def _run_one(tree: Path, contract: dict, control: dict) -> dict:
    applied = apply_edit(tree, control["edit"])
    try:
        # The templates import the candidate, and Python caches the import: a run in
        # this process would test the ORIGINAL modules and not the broken copy.
        # The copy is therefore checked in its own interpreter.
        outcome = _run_in_process(applied["tree"], contract)
        name = str(control["must_fail"])
        entry = (outcome.get("tests") or {}).get(name) or {}
        if entry.get("status") == "FAIL":
            return {"edit": control["edit"], "must_fail": name, "status": "PASS",
                    "detail": name + " caught the deliberate break",
                    "edited": applied["edited"]}
        return {"edit": control["edit"], "must_fail": name, "status": "FAIL",
                "detail": name + " still passed with the mechanism broken (" +
                          str(entry.get("status")) + "): the template does not see it",
                "edited": applied["edited"]}
    except Exception as exc:                               # noqa: BLE001
        return {"edit": control.get("edit"), "status": "FAIL",
                "detail": type(exc).__name__ + ": " + str(exc)}
    finally:
        shutil.rmtree(applied["scratch"], ignore_errors=True)
