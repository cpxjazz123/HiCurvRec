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


def verify_candidate(tree: Path, *, checks=None, controller: Controller = None) -> dict:
    """Run the deterministic verification of one candidate."""
    tree = Path(tree)
    controller = controller or Controller(ROOT / tree.name)
    record = controller.register(tree, note="V2 verification")
    if record.get("state") in (states.VERIFIED, states.TEST_FAILED, states.REJECTED):
        return record
    results = deterministic.run_all(tree, checks=checks)
    controller.directory.mkdir(parents=True, exist_ok=True)
    (controller.directory / "test_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return controller.record_tests(results)
