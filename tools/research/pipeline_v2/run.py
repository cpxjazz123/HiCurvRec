"""The V2 flow: register, run the programs, then ask one scientist.

``verify_candidate`` is the whole automatic path. It stops at VERIFIED, at
NEEDS_FIX with findings that carry a location, or at INFRA_ERROR when something in
the verification itself did not work; only the first two are statements about the
candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agents import runner
from ..agents import policy
from . import deterministic, review, states
from .controller import Controller

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "tools" / "research" / "contracts" / "HYPERBOLIC_GEOMETRY_CONTRACT.md"
ROOT = REPO / "results" / "research" / "pipeline_v2"


def verify_candidate(tree: Path, *, objective: str, run_review: bool = True,
                     checks=None, deadline_seconds: int = 300,
                     controller: Controller = None) -> dict:
    """Run the whole automatic verification of one candidate."""
    tree = Path(tree)
    controller = controller or Controller(ROOT / tree.name)
    record = controller.register(tree, note="V2 verification")
    if record.get("state") in (states.VERIFIED, states.TEST_FAILED, states.REJECTED):
        return record

    results = deterministic.run_all(tree, checks=checks)
    controller.directory.mkdir(parents=True, exist_ok=True)
    (controller.directory / "test_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if not results["ran"]:
        return controller.fail("a deterministic check could not run: " +
                               ", ".join(name + "=" + str(entry.get("status"))
                                         for name, entry in results["checks"].items()))
    if not results["passed"]:
        return controller.record_tests(results)          # -> TEST_FAILED
    record = controller.record_tests(results)             # -> REVIEW_REQUIRED
    if not run_review:
        return record

    prompt = review.review_prompt(tree, objective=objective, contract_path=CONTRACT,
                                  tests=results)
    try:
        runner.run_agent(
            "v2_review_" + tree.name, prompt, "review.txt", controller.directory,
            binary=None, model=None, timeout=deadline_seconds, role="verify",
            read_paths=[str(tree), str(CONTRACT),
                        str(controller.directory / "test_results.json")],
            allowed_roots=[str(tree), str(CONTRACT.parent), str(controller.directory)],
            deadline_seconds=deadline_seconds)
    except Exception as exc:                               # noqa: BLE001
        return controller.fail("the reviewer did not complete: " +
                               type(exc).__name__ + ": " + str(exc))
    answer = (controller.directory / "agents" / ("v2_review_" + tree.name) /
              "review.txt")
    if not answer.is_file():
        return controller.fail("the reviewer produced no answer")
    try:
        parsed = review.parse(answer.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        return controller.fail("the reviewer's answer is unreadable: " + str(exc))
    parsed["role"] = review.ROLE_ID
    return controller.record_review(parsed)
