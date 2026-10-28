"""Auditable idea ranking and selection (Issue #122).

Three independent reviewers order the in-scope proposals, best first. The
aggregate is the total rank: the proposal the reviewers place first most often,
and in total least far down, is the round's mechanism. A ranking that is not a
permutation of the proposals is a schema error, never a partial average over the
ideas that happened to be ranked. The rule keeps the single best mechanism,
keeps the runners-up as ordered backups, and refuses to invent a winner when
nobody qualifies (``NO_QUALIFIED_IDEA``).

Nothing here looks at ER, prefix counts or any other Stage 2 diagnostic: those
are measurements, not selection criteria.
"""
from __future__ import annotations

from . import rubric as rubric_module
from . import scope as scope_module

#: The policy this code was written against. A round pins it by version and
#: digest; the order of the rule is a policy changed before a new round, never
#: midway.
RUBRIC = rubric_module.load_rubric()

#: The selection policy: the same names the prompt, the parser and the decision
#: record read.
DEFAULT_SELECTION = dict(RUBRIC["selection"])

#: Why a proposal did not become the round's mechanism. They are kept apart so
#: a scientific negative is never confused with a resource or evidence problem.
REASON_OUT_OF_SCOPE = "OUT_OF_SCOPE"
REASON_NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
REASON_RESOURCE = "RESOURCE_BLOCKED"
REASON_CAP = "NOT_SELECTED_CAP"
REASON_BELOW_RANK = "BELOW_MINIMUM_REVIEWERS"
STATUS_SELECTED = "SELECTED"
STATUS_NO_QUALIFIED = "NO_QUALIFIED_IDEA"


class ScoreError(ValueError):
    """A ranking report is not usable."""


def validate_report(report: dict, *, reviewer: str) -> dict:
    """Check one reviewer's ranking against the schema, strictly.

    Every proposal must be ranked exactly once and the ranks must be a
    permutation of 1..N: a missing proposal, a repeated one or a tied position
    is an error the caller retries, never a partial average.
    """
    if not isinstance(report, dict):
        raise ScoreError("ranking report must be an object")
    ranking = report.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ScoreError("reviewer " + str(reviewer) + " returned no ranking")
    seen = set()
    positions = []
    for entry in ranking:
        if not isinstance(entry, dict) or not entry.get("idea_id"):
            raise ScoreError("reviewer " + str(reviewer) +
                             " returned a rank without an idea_id")
        identifier = str(entry["idea_id"])
        if identifier in seen:
            raise ScoreError("reviewer " + str(reviewer) + " ranked " + identifier + " twice")
        seen.add(identifier)
        position = entry.get("position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 1:
            raise ScoreError("rank for " + identifier + " has no integer position")
        positions.append(position)
        if not str(entry.get("reason", "")).strip():
            raise ScoreError("rank for " + identifier + " has no reason: write the line "
                             "'RANK: " + identifier + " | position=" +
                             str(position) + " | reason=<why it is placed there>'")
    if sorted(positions) != list(range(1, len(positions) + 1)):
        raise ScoreError("the ranks of reviewer " + str(reviewer) +
                         " are not a permutation of 1.." + str(len(positions)) +
                         ": " + str(sorted(positions)))
    return report


def build_matrix(ideas: list, reports: list) -> dict:
    """The rank matrix: every reviewer on every idea, with the aggregate."""
    idea_ids = [idea["idea_id"] for idea in ideas]
    by_reviewer = {}
    for report in reports:
        reviewer = str(report.get("reviewer") or report.get("scorer") or "")
        if not reviewer:
            raise ScoreError("a ranking report has no reviewer identity")
        by_reviewer[reviewer] = {str(entry["idea_id"]): entry
                                 for entry in report["ranking"]}
    missing = [reviewer + ":" + identifier for reviewer, entries in by_reviewer.items()
               for identifier in idea_ids if identifier not in entries]
    if missing:
        raise ScoreError("every reviewer must rank every idea; missing " +
                         ", ".join(sorted(missing)))
    cost_by_idea = {}
    for idea in ideas:
        for key in ("cost_estimate_hours", "hours_for_frozen_budget"):
            value = idea.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cost_by_idea[idea["idea_id"]] = float(value)
                break
    rows = []
    for identifier in idea_ids:
        positions = {reviewer: int(entries[identifier]["position"])
                     for reviewer, entries in by_reviewer.items()}
        idea = next((entry for entry in ideas if entry["idea_id"] == identifier), {})
        verdict = scope_module.check_scope(idea)
        total = sum(positions.values())
        first_places = sum(1 for value in positions.values() if value == 1)
        row = {
            "idea_id": identifier,
            "per_reviewer_position": positions,
            "per_reviewer_reason": {reviewer: by_reviewer[reviewer][identifier].get("reason", "")
                                    for reviewer in sorted(by_reviewer)},
            "total_rank": total,
            "mean_rank": round(total / len(positions), 4) if positions else None,
            "first_places": first_places,
            "reviewers": len(positions),
            "scope_verdict": "IN_SCOPE" if verdict["in_scope"] else "OUT_OF_SCOPE",
            "scope_reasons": verdict["reasons"],
            "scope_decision_basis": "proposal content: behaviour signal, curved "
                                    "operation, coupling pathway, ablations",
            "evidence_gaps": [name for name, present in (verdict.get("checks") or {}).items()
                              if not present],
        }
        if identifier in cost_by_idea:
            row["declared_hours"] = cost_by_idea[identifier]
        rows.append(row)
    return {"rubric_version": RUBRIC["version"], "rubric_sha256": RUBRIC["sha256"],
            "rows": rows, "reviewers": sorted(by_reviewer)}


def eligible(row: dict, rule: dict) -> tuple:
    """The hard eligibility of one row, with the reasons it is not eligible."""
    reasons = []
    if row.get("scope_verdict") == "OUT_OF_SCOPE":
        reasons.append(REASON_OUT_OF_SCOPE + ": " +
                       ", ".join(row.get("scope_reasons", [])[:4]))
    if row.get("reviewers", 0) < rule.get("min_reviewers", 1):
        reasons.append(REASON_BELOW_RANK + ": only " + str(row.get("reviewers", 0)) +
                       " reviewer(s) ranked it")
    declared = row.get("declared_hours")
    if declared is not None and declared > rule.get("declared_hours_budget", 0.0):
        reasons.append(REASON_RESOURCE + ": declared " + str(declared) + " h")
    return (not reasons), reasons


def select(matrix: dict, *, rule: dict = None) -> dict:
    """Apply the explicit policy and return the decision with its reasons.

    The order is the one the policy documents: the in-scope proposals first,
    then the aggregate rank, then the number of first places, then the idea id.
    The best aggregate order wins, the runners-up stay as ordered backups, and
    there is no automatic fallback: when nobody qualifies the round says so.
    """
    rule = dict(DEFAULT_SELECTION if rule is None else rule)
    rows = matrix["rows"]
    qualified, rejected = [], []
    for row in rows:
        ok, reasons = eligible(row, rule)
        if ok:
            qualified.append(row)
        else:
            rejected.append({"idea_id": row["idea_id"], "reasons": reasons,
                             "status": (REASON_OUT_OF_SCOPE if any(
                                 reason.startswith(REASON_OUT_OF_SCOPE) for reason in reasons)
                                 else REASON_RESOURCE if any(
                                     reason.startswith(REASON_RESOURCE) for reason in reasons)
                                 else REASON_BELOW_RANK),
                             "total_rank": row.get("total_rank"),
                             "mean_rank": row.get("mean_rank")})
    qualified.sort(key=lambda row: (row.get("total_rank", 0),
                                    -row.get("first_places", 0),
                                    row["idea_id"]))
    if len(qualified) > rule["max_selected"]:
        for row in qualified[rule["max_selected"]:]:
            rejected.append({"idea_id": row["idea_id"], "status": REASON_CAP,
                             "reasons": [REASON_CAP + ": outside the one-mechanism cap"],
                             "total_rank": row.get("total_rank"),
                             "mean_rank": row.get("mean_rank")})
        qualified = qualified[:rule["max_selected"]]
    selected = [{"idea_id": row["idea_id"], "total_rank": row["total_rank"],
                 "mean_rank": row["mean_rank"], "first_places": row["first_places"],
                 "scope_verdict": row["scope_verdict"], "status": STATUS_SELECTED}
                for row in qualified]
    chosen = {entry["idea_id"] for entry in selected}
    backups = [{"idea_id": row["idea_id"], "total_rank": row["total_rank"],
                "mean_rank": row["mean_rank"], "first_places": row["first_places"],
                "status": "BACKUP"}
               for row in sorted((row for row in rows if row["idea_id"] not in chosen),
                                 key=lambda row: (row.get("total_rank", 0),
                                                  -row.get("first_places", 0),
                                                  row["idea_id"]))]
    status = STATUS_SELECTED if selected else STATUS_NO_QUALIFIED
    return {"rule": rule, "status": status, "selected": selected, "backups": backups,
            "rejected": rejected,
            "selection_is_multiple": len(selected) > 1,
            "tie_break_order": list(rule.get("tie_break_order", [])),
            "disagreement": {row["idea_id"]: (max(row["per_reviewer_position"].values())
                                                - min(row["per_reviewer_position"].values()))
                             for row in rows if row.get("per_reviewer_position")},
            "aggregate": {row["idea_id"]: {"total_rank": row["total_rank"],
                                           "mean_rank": row["mean_rank"],
                                           "first_places": row["first_places"]}
                          for row in rows}}


def evaluate(ideas: list, reports: list, rule: dict = None) -> dict:
    """The rank matrix and the selection, as one record."""
    for report in reports:
        validate_report(report, reviewer=str(report.get("reviewer")
                                            or report.get("scorer") or ""))
    matrix = build_matrix(ideas, reports)
    selection = select(matrix, rule=rule)
    return {"rubric_version": RUBRIC["version"], "rubric_sha256": RUBRIC["sha256"],
            "matrix": matrix, "selection": selection}
