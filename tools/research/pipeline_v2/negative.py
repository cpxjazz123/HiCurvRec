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
    # 2 the curved score becomes the Euclidean one
    "euclidean kernel": ("model/behaviour_radius.py",
                         "    if not use_geometry:\n        return (a - e).pow(2).sum(dim=-1)",
                         "    return (a - e).pow(2).sum(dim=-1)"),
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


def _run_one(tree: Path, contract: dict, control: dict) -> dict:
    applied = apply_edit(tree, control["edit"])
    try:
        outcome = contracts.run_tests(applied["tree"], contract)
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
