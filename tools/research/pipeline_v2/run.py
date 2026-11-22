"""The V2 flow: register, run the programs, and say what they found.

``verify_candidate`` is the whole automatic path. It ends at VERIFIED when every
deterministic check passed, at TEST_FAILED with the failing check named, or at
INFRA_ERROR when the verification itself could not run. There is no review agent:
the mechanism was judged during ideation and scoring, and a second opinion on the
code is not what the project needs.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import deterministic, states
from .controller import Controller

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results" / "research" / "pipeline_v2"


def _fingerprints(tree: Path) -> dict:
    """The digests of the inputs the verification depends on."""
    import hashlib
    from . import deterministic

    name = Path(tree).name
    name = name[6:] if name.startswith("TIGER_") else name
    contract = (REPO / "results" / "research" / "pipeline_v2" / "contracts" /
                (name + ".json"))
    values = {"contract": (hashlib.sha256(contract.read_bytes()).hexdigest()
                           if contract.is_file() else ""),
              "checks": hashlib.sha256(
                  Path(deterministic.__file__).read_bytes()).hexdigest()}
    return values


def verify_candidate(tree: Path, *, checks=None, controller: Controller = None) -> dict:
    """Run the deterministic verification of one candidate."""
    tree = Path(tree)
    controller = controller or Controller(ROOT / tree.name)
    record = controller.register(tree, note="V2 verification")
    fingerprints = _fingerprints(tree)
    if record.get("fingerprints") and record["fingerprints"] != fingerprints:
        # the checks themselves changed: the earlier rounds measured something
        # else, so their repair count does not apply to this one
        record = controller.reset_budget("the verification inputs changed")
    if record.get("state") == states.REJECTED:
        return record
    # The checks run again even for a verified tree: the check SET grows (a new
    # template, a stricter contract), and a record that never re-runs would keep
    # blessing a tree under an older set. The source digest is what makes this
    # safe: an unchanged tree and an unchanged check set give the same answer.
    results = deterministic.run_all(tree, checks=checks)
    record["fingerprints"] = fingerprints
    # The run is attributable to a DEFINITION, not only to a source digest: the
    # design contract's version and hash travel with the record, so a later change
    # to what the mechanism is cannot be read as the same experiment.
    design = (results.get("checks") or {}).get("design") or {}
    if design.get("status") == "PASS":
        record["design_contract"] = {"version": design.get("version"),
                                     "sha256": design.get("sha256"),
                                     "path": design.get("path")}
    controller.directory.mkdir(parents=True, exist_ok=True)
    (controller.directory / "test_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if (record.get("state") == states.TEST_FAILED and results.get("ran")
            and results.get("passed")):
        # A failing check needs a change before it can pass again. The change may be
        # in the candidate or in the check itself (a wrong template is not a defect
        # of the mechanism); either way it is one repair and the history says so.
        record = controller.repair("re-run after a change to the candidate or the checks")
    record = controller.record_tests(results)
    # record_tests reloads the record from disk, so the definition is attached
    # after that write and saved through the controller.
    if design.get("status") == "PASS" and record.get("design_contract") != {
            "version": design.get("version"), "sha256": design.get("sha256"),
            "path": design.get("path")}:
        stored = controller.load()
        stored["design_contract"] = {"version": design.get("version"),
                                     "sha256": design.get("sha256"),
                                     "path": design.get("path")}
        controller.save(stored)
        record = stored
    return record
