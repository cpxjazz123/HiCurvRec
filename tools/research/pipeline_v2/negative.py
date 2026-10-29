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
    "constant radius": ("model/behaviour_radius.py",
                        "    values = torch.as_tensor(scale, dtype=torch.float64).flatten()",
                        "    return torch.full((24,), 0.5)\n"
                        "    values = torch.as_tensor(scale, dtype=torch.float64).flatten()"),
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
    """Apply the contract's edit and require the named template to fail."""
    control = contract.get("negative_control") or {}
    if not control:
        return {"status": "NOT_RUN", "detail": "the contract declares no negative control"}
    applied = apply_edit(tree, control["edit"])
    try:
        outcome = contracts.run_tests(applied["tree"], contract)
        name = str(control["must_fail"])
        entry = (outcome.get("tests") or {}).get(name) or {}
        if entry.get("status") == "FAIL":
            return {"status": "PASS", "detail": name + " caught the deliberate break",
                    "edited": applied["edited"], "template_result": entry}
        return {"status": "FAIL",
                "detail": name + " still passed with the mechanism broken (" +
                          str(entry.get("status")) + "): the template does not see it",
                "edited": applied["edited"], "template_result": entry}
    finally:
        shutil.rmtree(applied["scratch"], ignore_errors=True)
