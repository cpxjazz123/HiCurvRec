"""Plain-text contracts for the round agents.

The agents are language models, not parsers. Asking them for JSON made a
single stray comma throw away a whole answer, and the framework then kept an
empty tree and an approval. They now write a labelled plain-text file, and
this module reads that file back into the records the round keeps.

The format is line oriented and forgiving:

    KEY: value
    continuation line of the value above
    KEY: another value

Keys are matched case insensitively, a value continues until the next known
key, unknown lines are ignored, and a document that says nothing usable raises
a :class:`PlainTextError` instead of returning a half-filled record.
"""
from __future__ import annotations

import re
from typing import Iterable

_KEY_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z _/-]{0,40}?)\s*:\s*(.*)$")


class PlainTextError(ValueError):
    """An agent's answer cannot be read as the document it promised."""


def _canonical(key: str, known: Iterable[str]) -> str | None:
    """The canonical name of a key, or None when the line is not a key line."""
    text = " ".join(str(key).strip().lower().split())
    for name in known:
        if text == name.lower():
            return name
    return None


def parse_fields(text: str, keys: Iterable[str]) -> list[tuple[str, str]]:
    """Read a document into ordered ``(key, value)`` pairs.

    A value spans every line after its key that is not itself a key line, so an
    agent can explain itself in prose without breaking the format.
    """
    known = list(keys)
    fields: list[tuple[str, str]] = []
    current: list[str] | None = None
    for raw in str(text or "").splitlines():
        match = _KEY_LINE.match(raw)
        name = _canonical(match.group(1), known) if match else None
        if name is not None:
            if current is not None:
                fields.append((current[0], "\n".join(current[1]).strip()))
            current = [name, [match.group(2).strip()]]
            continue
        if current is None:
            # Prose before the first key is kept: it is the agent's summary and
            # is worth having, but it is never a field.
            continue
        current[1].append(raw.rstrip())
    if current is not None:
        fields.append((current[0], "\n".join(current[1]).strip()))
    return fields


def _values(fields: list[tuple[str, str]], key: str) -> list[str]:
    return [value for name, value in fields if name == key]


def _first(fields: list[tuple[str, str]], key: str) -> str:
    values = _values(fields, key)
    return values[0] if values else ""


IDEA_KEYS = ("IDEA", "TITLE", "MOTIVATION", "NOVELTY", "CHANGE", "SPECIFICATION",
             "COST", "HYPOTHESIS", "RISKS",
             "BEHAVIOR SIGNAL", "BEHAVIOR COMPUTATION", "LEAKAGE BOUNDARY",
             "HYPERBOLIC OPERATOR", "HYPERBOLIC LOCATION",
             "COUPLING PATH", "COUPLING EQUATION", "OBSERVABLES",
             "ABLATION BEHAVIOR OFF", "ABLATION GEOMETRY OFF", "ABLATION BOTH ON",
             "SPEC EQUATIONS", "SPEC INPUTS OUTPUTS", "SAFEGUARDS",
             "END IDEA")

_IDEA_FIELDS = {
    "IDEA": "idea_id", "TITLE": "title", "MOTIVATION": "motivation",
    "NOVELTY": "novelty", "CHANGE": "key_change", "SPECIFICATION": "specification",
    "COST": "cost_estimate", "HYPOTHESIS": "falsifiable_hypothesis", "RISKS": "risks",
    "BEHAVIOR SIGNAL": "behavior_signal",
    "BEHAVIOR COMPUTATION": "behavior_computation",
    "LEAKAGE BOUNDARY": "leakage_boundary",
    "HYPERBOLIC OPERATOR": "hyperbolic_operator",
    "HYPERBOLIC LOCATION": "hyperbolic_location",
    "COUPLING PATH": "coupling_path",
    "COUPLING EQUATION": "coupling_equation",
    "OBSERVABLES": "observables",
    "ABLATION BEHAVIOR OFF": "ablation_behavior_off",
    "ABLATION GEOMETRY OFF": "ablation_geometry_off",
    "ABLATION BOTH ON": "ablation_both_on",
    "SPEC EQUATIONS": "spec_equations",
    "SPEC INPUTS OUTPUTS": "spec_inputs_outputs",
    "SAFEGUARDS": "safeguards",
}


def _idea_blocks(text: str) -> list[str]:
    """Split an ideation answer into its idea blocks.

    A block starts at ``IDEA:`` and ends at ``END IDEA``; the terminator is a
    line without a value, so it is handled here rather than as a key.
    """
    blocks: list[str] = []
    current: list[str] = []
    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        if stripped.upper() == "END IDEA":
            if current:
                blocks.append("\n".join(current))
            current = []
            continue
        if _KEY_LINE.match(raw) and _canonical(_KEY_LINE.match(raw).group(1), IDEA_KEYS) == "IDEA":
            if current:
                blocks.append("\n".join(current))
            current = [raw]
            continue
        if current:
            current.append(raw)
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_ideas(text: str) -> dict:
    """Read the ideas of one ideation agent.

    One idea per ``IDEA:`` block, ended by ``END IDEA``. A block that names no
    identifier is not an idea and is skipped.
    """
    ideas: list[dict] = []
    for block in _idea_blocks(text):
        fields = parse_fields(block, IDEA_KEYS)
        identifier = _first(fields, "IDEA").strip()
        if not identifier:
            continue
        idea = {"idea_id": identifier}
        for name, value in fields:
            target = _IDEA_FIELDS.get(name)
            if target and target not in idea:
                idea[target] = value
        ideas.append(idea)
    ideas = [idea for idea in ideas if idea.get("idea_id")]
    if not ideas:
        raise PlainTextError("the answer holds no idea block")
    return {"ideas": ideas}


RANK_KEYS = ("RANK",)


def parse_ranks(text: str) -> dict:
    """Read one reviewer's ranking of the proposals.

    ``RANK: <idea_id> | position=<n> | reason=<why>`` places a proposal. The
    positions must be a permutation of 1..N: a missing proposal, a repeated one
    or a tie is not a ranking.
    """
    fields = parse_fields(text, RANK_KEYS)
    ranking = []
    for name, value in fields:
        if name != "RANK":
            continue
        parts = [part.strip() for part in value.split("|")]
        identifier = parts[0].strip()
        if not identifier:
            continue
        position = None
        reason = ""
        for part in parts[1:]:
            key, _, raw = part.partition("=")
            key = key.strip().lower()
            if key == "position":
                try:
                    position = int(raw.strip())
                except ValueError:
                    position = None
            elif key == "reason":
                reason = raw.strip()
        ranking.append({"idea_id": identifier, "position": position,
                         "reason": reason})
    if not ranking:
        raise PlainTextError("the answer holds no rank line")
    ranking.sort(key=lambda entry: (entry["position"] is None, entry["position"] or 0,
                                    entry["idea_id"]))
    return {"ranking": ranking}


VERIFY_KEYS = ("VERDICT", "CHECK", "CHECK VERDICT", "EXERCISED PATH",
               "FINDING SEVERITY", "FINDING LOCATION", "FINDING DETAIL",
               "FINDING REPRODUCTION", "FINDING CORRECTION")


def parse_verification(text: str) -> dict:
    """Read one verifier's report.

    ``CHECK:`` opens a check, the finding lines belong to the check that is
    open, and ``VERDICT:`` at the top is the reviewer's overall verdict.
    """
    fields = parse_fields(text, VERIFY_KEYS)
    verdict = _first(fields, "VERDICT").strip().upper()
    checks: list[dict] = []
    current: dict | None = None
    finding: dict | None = None
    for name, value in fields:
        if name == "VERDICT":
            continue
        if name == "EXERCISED PATH":
            if current is None:
                continue
            parts = [part.strip() for part in value.split("|")]
            current["exercised_path"] = parts[0]
            current["exercised_how"] = parts[1] if len(parts) > 1 else ""
            continue
        if name == "CHECK":
            if current is not None:
                checks.append(current)
            current = {"name": value.strip(), "verdict": "", "findings": [],
                       "exercised_path": "", "exercised_how": ""}
            finding = None
            continue
        if current is None:
            continue
        if name == "CHECK VERDICT":
            current["verdict"] = value.strip().upper()
            continue
        if name == "FINDING SEVERITY":
            # Every severity line opens a finding: a check may report a blocking
            # one and a minor one, and folding the second into the first would
            # keep the minor severity and silently drop the blocker.
            finding = {"severity": value.strip()}
            current["findings"].append(finding)
            continue
        if finding is None:
            finding = {}
            current["findings"].append(finding)
        if name == "FINDING LOCATION":
            finding["location"] = value.strip()
        elif name == "FINDING DETAIL":
            finding["detail"] = value.strip()
        elif name == "FINDING REPRODUCTION":
            finding["reproduction"] = value.strip()
        elif name == "FINDING CORRECTION":
            finding["correction"] = value.strip()
    if current is not None:
        checks.append(current)
    if verdict not in ("PASS", "NEEDS_FIX"):
        raise PlainTextError("the answer has no PASS or NEEDS_FIX verdict")
    for check in checks:
        for entry in check["findings"]:
            entry.setdefault("severity", "blocking")
            entry.setdefault("location", "")
            entry.setdefault("detail", "")
            entry.setdefault("reproduction", "")
            entry.setdefault("correction", "")
    return {"verdict": verdict, "checks": checks}


IMPLEMENT_KEYS = ("TREE", "CHANGED FILE", "DEVIATION", "ASSUMPTION",
                  "SMOKE COMMAND", "SMOKE RESULT", "NOTES")


def parse_implementation(text: str) -> dict:
    """Read an implementation agent's report."""
    fields = parse_fields(text, IMPLEMENT_KEYS)
    deviations = []
    for value in _values(fields, "DEVIATION"):
        item, _, reason = value.partition("|")
        deviations.append({"item": item.strip(), "reason": reason.strip()})
    smoke_command = _first(fields, "SMOKE COMMAND")
    smoke_result = _first(fields, "SMOKE RESULT")
    return {
        "tree": _first(fields, "TREE").strip(),
        "changed_files": [value.strip() for value in _values(fields, "CHANGED FILE")
                          if value.strip()],
        "deviations": deviations,
        "assumptions": [value.strip() for value in _values(fields, "ASSUMPTION")
                        if value.strip()],
        "smoke": {"command": smoke_command, "result": smoke_result},
        "notes": _first(fields, "NOTES"),
    }


REPAIR_PLAN_KEYS = ("PLAN", "REMAINING ISSUE", "NOTES")


def parse_repair_plan(text: str) -> dict:
    """Read a repair agent's PLAN.

    The planner does not touch the code, so the answer carries the change and
    the check that proves it, not a smoke result: the operator applies the plan
    and then runs the check.

    A plan without a single PLAN line is refused rather than accepted as an
    empty plan: an empty plan would send the candidate back to the verifiers
    unchanged and burn another round on the same finding.
    """
    fields = parse_fields(text, REPAIR_PLAN_KEYS)
    plans = []
    for value in _values(fields, "PLAN"):
        parts = [part.strip() for part in value.split("|")]
        while len(parts) < 4:
            parts.append("")
        plans.append({"finding": parts[0], "location": parts[1],
                      "change": parts[2], "check": parts[3]})
    if not plans:
        raise PlainTextError(
            "the repair planner wrote no PLAN line: the answer must plan one change "
            "per finding as 'PLAN: <finding> | <file:line> | <change> | <check>'")
    return {
        "plans": plans,
        "remaining_issues": [value.strip() for value in _values(fields, "REMAINING ISSUE")
                             if value.strip()],
        "notes": _first(fields, "NOTES"),
    }


def render_idea(idea: dict) -> str:
    """Show one idea to an agent in the same plain-text shape it must answer in."""
    lines = ["IDEA: " + str(idea.get("idea_id", "")),
             "TITLE: " + str(idea.get("title", "")),
             "MOTIVATION: " + str(idea.get("motivation", "")),
             "NOVELTY: " + str(idea.get("novelty", "")),
             "CHANGE: " + str(idea.get("key_change", "")),
             "SPECIFICATION: " + str(idea.get("specification", "")),
             "COST: " + str(idea.get("cost_estimate", "")),
             "HYPOTHESIS: " + str(idea.get("falsifiable_hypothesis", "")),
             "RISKS: " + str(idea.get("risks", "")),
             "END IDEA"]
    return "\n".join(lines)


def render_ideas(ideas: list) -> str:
    """Show every idea to a scorer."""
    return "\n\n".join(render_idea(idea) for idea in ideas)


def render_findings(findings: list) -> str:
    """Show the blocking findings to a repair engineer."""
    blocks = []
    for finding in findings:
        blocks.append("\n".join([
            "FINDING SEVERITY: " + str(finding.get("severity", "")),
            "FINDING LOCATION: " + str(finding.get("location", "")),
            "FINDING DETAIL: " + str(finding.get("detail", "")),
            "FINDING REPRODUCTION: " + str(finding.get("reproduction", "")),
            "FINDING CORRECTION: " + str(finding.get("correction", "")),
        ]))
    return "\n\n".join(blocks)


PAIRWISE_KEYS = ("WINNER", "FIRST LISTED", "ARGUMENT", "FAILURE RISK")


def parse_rebuttal(text: str) -> dict:
    """Read the one bounded revision a proposal is allowed."""
    fields = parse_fields(text, REBUTTAL_KEYS)
    change, _, why = _first(fields, "CHANGE").partition("|")
    return {"change": change.strip(), "why": why.strip(),
            "unchanged": _first(fields, "UNCHANGED").strip(),
            "remaining": _first(fields, "REMAINING").strip()}


EVIDENCE_REVISION_KEYS = ("REVISION", "WITHDRAWN", "EVIDENCE NOTE")


def parse_evidence_revision(text: str) -> dict:
    """Read the one bounded, evidence-driven revision of a proposal."""
    fields = parse_fields(text, EVIDENCE_REVISION_KEYS)
    revision, _, gap = _first(fields, "REVISION").partition("|")
    return {"revision": revision.strip(), "answers": gap.strip(),
            "withdrawn": _first(fields, "WITHDRAWN").strip(),
            "evidence_note": _first(fields, "EVIDENCE NOTE").strip()}
