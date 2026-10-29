"""One scientific reviewer, the whole tree, the tests already run.

The reviewer answers the only question a language model is better at than a
program: is this mechanism worth a training run? It sees the complete candidate
tree - that tree and nothing else - plus the geometry contract and the
deterministic results, and it may report a defect only with a location and a
reproduction. Its role is fixed when the task is created, not inferred later.
"""
from __future__ import annotations

import json
from pathlib import Path

ROLE_ID = "science-review"

PROMPT = """You are the scientific reviewer of ONE candidate mechanism for a
generative recommender's semantic IDs. A program has already checked that the code
runs: the geometry contract, the candidate's own tests, the export/assignment
agreement and a two-rank DDP run. Do not re-check those; judge the MECHANISM.

THE RESEARCH OBJECTIVE:
{objective}

THE GEOMETRY CONTRACT (normative; the candidate may compose it, never redefine it):
{contract_path}

WHAT THE PROGRAM FOUND (do not repeat these checks):
{test_summary}

THE CANDIDATE TREE (complete, read-only; this is the only tree you may read):
{tree}

Answer these three questions, each with a verdict of PASS or NEEDS_FIX:
  a) COUPLING: does the training-only behaviour signal and the curved operation
     act on ONE connected computation path that decides the SID assignment? Name
     the path (file:line for the signal, the geometry and the decision). Remove
     the behaviour and the geometry and say what changes; if either removal
     changes nothing, the coupling is not load-bearing and that is NEEDS_FIX.
  b) MATHEMATICS: is the geometry used consistently with the contract (curvature,
     distance, exp/log, projection), and is the formula that turns the behaviour
     statistic into the geometric quantity well defined over the range the data
     will actually produce? Name the range and the constants.
  c) CLAIM: is the stated hypothesis falsifiable by ONE Stage 3 run, and does the
     mechanism avoid validation/test leakage and per-user SIDs?

Answer in PLAIN TEXT, exactly:

VERDICT: PASS or NEEDS_FIX
CHECK: <coupling | mathematics | claim>
CHECK VERDICT: PASS or NEEDS_FIX
FINDING SEVERITY: <blocking | minor | none>
FINDING LOCATION: <file:line or n/a>
FINDING DETAIL: <what is wrong, with the evidence you read>
FINDING REPRODUCTION: <how a reader can see it>
FINDING CORRECTION: <the smallest correction>

A NEEDS_FIX must name a defect with a location; a NEEDS_FIX without one is not a
report. A minor finding does not block the training run. Print nothing else.
"""


def review_prompt(tree: Path, *, objective: str, contract_path: Path,
                  tests: dict) -> str:
    """The one prompt, with the tests already run and the tree to read."""
    summary = []
    for name, result in (tests.get("checks") or {}).items():
        summary.append("  - %s: %s" % (name, result.get("status")))
    return PROMPT.format(objective=str(objective).strip(),
                         contract_path=str(contract_path),
                         test_summary="\n".join(summary) or "  (no checks recorded)",
                         tree=str(Path(tree).resolve()))


def parse(text: str) -> dict:
    """Read the reviewer's answer into the record the controller stores."""
    verdict = ""
    checks = []
    current = None
    finding = None
    for line in str(text).splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.startswith("CHECK:"):
            current = {"name": line.split(":", 1)[1].strip(), "verdict": "",
                       "findings": []}
            checks.append(current)
            finding = None
        elif line.startswith("CHECK VERDICT:") and current is not None:
            current["verdict"] = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FINDING SEVERITY:") and current is not None:
            finding = {"severity": line.split(":", 1)[1].strip().lower()}
            current["findings"].append(finding)
        elif line.startswith("FINDING LOCATION:") and finding is not None:
            finding["location"] = line.split(":", 1)[1].strip()
        elif line.startswith("FINDING DETAIL:") and finding is not None:
            finding["detail"] = line.split(":", 1)[1].strip()
        elif line.startswith("FINDING REPRODUCTION:") and finding is not None:
            finding["reproduction"] = line.split(":", 1)[1].strip()
        elif line.startswith("FINDING CORRECTION:") and finding is not None:
            finding["correction"] = line.split(":", 1)[1].strip()
    if verdict not in ("PASS", "NEEDS_FIX"):
        raise ValueError("the review has no PASS or NEEDS_FIX verdict")
    findings = []
    for check in checks:
        for item in check["findings"]:
            severity = str(item.get("severity") or "")
            if severity in ("none", "n/a", ""):
                continue
            findings.append({"check": check["name"], "severity": severity,
                             "location": str(item.get("location") or ""),
                             "detail": str(item.get("detail") or ""),
                             "reproduction": str(item.get("reproduction") or ""),
                             "correction": str(item.get("correction") or "")})
    return {"verdict": verdict, "checks": checks, "findings": findings,
            "raw": str(text)}
