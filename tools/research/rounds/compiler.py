"""The formula compiler's gate: every check, one verdict.

A candidate may train only when the whole chain has been executed and every
link of it passed:

1. the frozen specification exists, the file on disk is the one the record
   cites, and two Math Reviewers passed it;
2. three independent implementations of that specification agree - the
   differential verdict is AGREE, not "comparable" and not "not run";
3. the candidate's contract states the frozen specification's statements
   verbatim, so the code and the mathematics cannot drift apart;
4. the framework's conformance checks all pass - the frozen constants and the
   launcher, the real entry point running one epoch and exporting the
   identifier, the identifier itself, and the four ranks finishing;
5. the reference agreement passes on the implemented tree, with the coupling
   surface showing that the mechanism's own behaviour term moves the
   computation.

A check that did not run is NOT a pass: ``NOT_RUN``, ``NOT_APPLICABLE`` (where
the check applies), ``INCOMPARABLE`` and a skipped check all block.
"""
from __future__ import annotations

import json
from pathlib import Path

SPEC_NAME = "math_spec.json"

#: The statuses that block. Anything that is not an executed pass blocks.
BLOCKING = ("FAIL", "INCOMPARABLE", "NOT_RUN", "TIMEOUT", "SKIPPED")


def _check(name: str, status: str, detail: str = "") -> dict:
    passed = status == "PASS"
    return {"name": name, "status": "PASS" if passed else "BLOCKED",
            "detail": detail, "raw_status": status}


def verdict(tree: Path, state: dict, *, deep: bool = False) -> dict:
    """The one verdict the registration and the dispatch both read.

    ``deep`` adds the framework's conformance checks - the one-epoch run, the
    identifier, the four ranks - which are the expensive part and belong to the
    dispatch, where they are already cached by the source digest. The cheap
    verdict (the specification, the three implementations, the contract) is
    what the registration pays for.
    """
    from . import specification as specification_module
    from ..stage2 import checks as stage2_checks

    tree = Path(tree)
    directory = Path(state.get("_directory") or tree.parent)
    checks = []

    # 1. the frozen specification
    frozen = state.get("math_spec") or {}
    if not frozen.get("sha256"):
        checks.append(_check("specification", "NOT_RUN",
                             "no frozen mathematical specification"))
    else:
        written = specification_module.matches_frozen(directory, frozen["sha256"])
        reviews = state.get("math_reviews") or []
        reviewed = [review for review in reviews
                    if str(review.get("verdict", "")).upper() == "PASS"]
        if not written.get("matches"):
            checks.append(_check("specification", "FAIL", str(written.get("reason"))))
        elif len(reviewed) < 2:
            checks.append(_check("specification", "FAIL",
                                 "the specification was not passed by two reviewers"))
        else:
            checks.append(_check("specification", "PASS",
                                 "frozen %s, %d reviews" % (frozen["sha256"][:12],
                                                            len(reviews))))

    # 2. the three independent implementations
    coding = state.get("coding") or {}
    coding_verdict = coding.get("verdict") or {}
    implementations = coding.get("implementations") or []
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

    # 3. the contract states the frozen specification
    contract_path = tree / "reference_contract.json"
    if not contract_path.is_file():
        checks.append(_check("contract", "NOT_RUN", "no reference_contract.json"))
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

    if not deep:
        blocked = [entry for entry in checks if entry["status"] != "PASS"]
        return {"tree": str(tree), "status": "BLOCKED" if blocked else "PASS",
                "deep": False, "checks": checks,
                "blocked_by": [entry["name"] for entry in blocked]}

    # 4. the framework's conformance checks
    try:
        report = stage2_checks.run_all(tree)
    except Exception as exc:  # noqa: BLE001
        report = {"checks": [{"name": "conformance", "status": "FAIL",
                              "detail": "the checks could not run: " + str(exc)[:200]}]}
    for entry in report.get("checks") or []:
        raw = str(entry.get("status") or "NOT_RUN")
        if entry.get("name") == "reference_agreement":
            checks.append(_check("reference_agreement",
                                 "PASS" if raw == "PASS" else raw,
                                 str(entry.get("detail"))[:200]))
            continue
        checks.append(_check(str(entry.get("name")), raw,
                             str(entry.get("detail"))[:160]))

    blocked = [entry for entry in checks if entry["status"] != "PASS"]
    return {"tree": str(tree), "status": "BLOCKED" if blocked else "PASS",
            "deep": True, "checks": checks,
            "blocked_by": [entry["name"] for entry in blocked]}
