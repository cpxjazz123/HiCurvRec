"""The two verification stages, and the one verdict they produce.

There is no separate CGV any more: its checks live in the two stages below, and
neither duplicates the other.

**Stage 1 - mathematical verification** (cheap, run at registration):

* the frozen specification exists, the file on disk is the one the record
  cites, and two Math Reviewers passed it;
* three independent implementations of that specification agree: the same
  forward values, the same gradients, the same gradient DIRECTION, and each
  against a reference rebuilt from its own statements with gradcheck on both
  sides;
* the candidate's contract states the frozen specification's statements
  verbatim.

**Stage 2 - training integration verification** (expensive, run at dispatch):

* the framework's conformance checks - the frozen constants and the launcher,
  the entry point running one epoch and exporting the identifier, the
  identifier itself, the four ranks finishing, the reference agreement;
* the ENCODER receives the reconstruction gradient in the candidate's own
  model: three correct implementations of a distance function prove nothing
  about the wiring, and an accidental ``detach`` leaves the real encoder with
  no gradient while every mathematical check still passes.

A check that did not run is NOT a pass: ``FAIL``, ``INCOMPARABLE``, ``NOT_RUN``,
``TIMEOUT`` and ``SKIPPED`` all block.
"""
from __future__ import annotations

import json
from pathlib import Path

SPEC_NAME = "math_spec.json"

#: The statuses that block. Anything that is not an executed pass blocks.
BLOCKING = ("FAIL", "INCOMPARABLE", "NOT_RUN", "TIMEOUT", "SKIPPED")


def _check(name: str, status: str, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if status == "PASS" else "BLOCKED",
            "detail": detail, "raw_status": status}


def mathematical_verdict(tree: Path, state: dict) -> dict:
    """Stage 1: the frozen formula and the three independent implementations."""
    from . import specification as specification_module

    tree = Path(tree)
    directory = Path(state.get("_directory") or tree.parent)
    checks = []

    frozen = state.get("math_spec") or {}
    if not frozen.get("sha256"):
        checks.append(_check("specification", "NOT_RUN",
                             "no frozen mathematical specification"))
    else:
        written = specification_module.matches_frozen(directory, frozen["sha256"])
        reviews = state.get("math_reviews") or []
        passed = [review for review in reviews
                  if str(review.get("verdict", "")).upper() == "PASS"]
        if not written.get("matches"):
            checks.append(_check("specification", "FAIL", str(written.get("reason"))))
        elif len(passed) < 2:
            checks.append(_check("specification", "FAIL",
                                 "the specification was not passed by two reviewers"))
        else:
            checks.append(_check("specification", "PASS",
                                 "frozen %s, %d reviews" % (frozen["sha256"][:12],
                                                            len(reviews))))

    coding = state.get("coding") or {}
    coding_verdict = coding.get("verdict") or {}
    implementations = coding.get("implementations") or []
    pairs = coding_verdict.get("pairs") or []
    if len(implementations) < 3:
        checks.append(_check("implementations", "NOT_RUN",
                             "%d of three implementations" % len(implementations)))
    elif coding_verdict.get("status") != "AGREE":
        checks.append(_check("implementations", "FAIL",
                             str(coding_verdict.get("status")) + ": " +
                             str(coding_verdict.get("reason"))[:200]))
    else:
        checks.append(_check("implementations", "PASS",
                             "three implementations agree"))
        worst = min([float(pair.get("gradient_sign_agreement", 1.0))
                     for pair in pairs] or [1.0])
        checks.append(_check("gradient_direction",
                             "PASS" if worst >= 0.99 else "FAIL",
                             "the worst pair agrees on the sign of %.3f of the "
                             "gradient entries" % worst))

    contract_path = tree / "reference_contract.json"
    if not contract_path.is_file():
        checks.append(_check("contract", "FAIL", "no reference_contract.json"))
    else:
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            stated = list((contract.get("calls") or [{}])[0].get("statements") or [])
        except ValueError as exc:
            checks.append(_check("contract", "FAIL", "unreadable: " + str(exc)))
            stated = None
        if stated is not None:
            expected = specification_module.statements(frozen.get("spec") or {})
            if not expected:
                checks.append(_check("contract", "NOT_RUN",
                                     "the specification states no surface"))
            elif stated != expected:
                checks.append(_check("contract", "FAIL",
                                     "the contract's statements are not the frozen "
                                     "specification's"))
            else:
                checks.append(_check("contract", "PASS",
                                     "the contract states the frozen statements"))
    blocked = [entry for entry in checks if entry["status"] != "PASS"]
    return {"stage": "mathematical", "status": "BLOCKED" if blocked else "PASS",
            "checks": checks, "blocked_by": [entry["name"] for entry in blocked]}


def training_verdict(tree: Path, state: dict) -> dict:
    """Stage 2: the candidate's real model, wired, trained, and four-ranked."""
    from ..stage2 import checks as stage2_checks

    tree = Path(tree)
    checks = []
    try:
        report = stage2_checks.run_all(tree)
    except Exception as exc:  # noqa: BLE001
        report = {"checks": [{"name": "conformance", "status": "FAIL",
                              "detail": "the checks could not run: " + str(exc)[:200]}]}
    for entry in report.get("checks") or []:
        raw = str(entry.get("status") or "NOT_RUN")
        checks.append(_check(str(entry.get("name")), raw,
                             str(entry.get("detail"))[:200]))
    blocked = [entry for entry in checks if entry["status"] != "PASS"]
    return {"stage": "training", "status": "BLOCKED" if blocked else "PASS",
            "checks": checks, "blocked_by": [entry["name"] for entry in blocked]}


def verdict(tree: Path, state: dict, *, deep: bool = False) -> dict:
    """The one verdict: stage one always, stage two when ``deep``.

    The registration reads the mathematical stage (it is cheap); the dispatch
    reads both, and the training stage is cached by the source digest there
    because it trains an epoch.
    """
    tree = Path(tree)
    stages = [mathematical_verdict(tree, state)]
    if deep:
        stages.append(training_verdict(tree, state))
    blocked = [stage for stage in stages if stage["status"] != "PASS"]
    return {"tree": str(tree), "deep": bool(deep),
            "status": "BLOCKED" if blocked else "PASS",
            "stages": stages,
            "blocked_by": [name for stage in blocked for name in stage["blocked_by"]]}
