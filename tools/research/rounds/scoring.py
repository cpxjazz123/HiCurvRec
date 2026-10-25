"""Weighted, auditable idea scoring and selection (Issue #122).

Three independent reviewers score every in-scope proposal on the seven
dimensions of the versioned rubric; the aggregate is the weighted median, not an
unweighted average, and an incomplete review is a schema error rather than a
number. The rule then keeps the single best qualified mechanism, keeps the
runners-up as ordered backups, and refuses to invent a winner when nobody
qualifies (``NO_QUALIFIED_IDEA``). Pairwise comparisons of the finalists break
near-ties deterministically and are persisted with the decision.

Nothing here looks at ER, prefix counts or any other Stage 2 diagnostic: those
are measurements, not selection criteria.
"""
from __future__ import annotations

import itertools
import statistics

from . import rubric as rubric_module
from . import scope as scope_module

#: The rubric this code was written against. A round pins it by version and
#: digest; the weights are a policy changed before a new round, never midway.
RUBRIC = rubric_module.load_rubric()

#: The rubric's dimensions, its scale and its selection policy. They are not
#: repeated here: the JSON policy is the single source, and the parsers, the
#: prompts and the matrix all read the same names.
DIMENSIONS = tuple(RUBRIC["dimension_names"])
SCALE = tuple(RUBRIC["scale"])
DEFAULT_SELECTION = dict(RUBRIC["selection"])

#: Why a proposal did not become the round's mechanism. They are kept apart so
#: a scientific negative is never confused with a resource or evidence problem.
REASON_OUT_OF_SCOPE = "OUT_OF_SCOPE"
REASON_NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
REASON_NEEDS_SPEC = "NEEDS_SPEC_REPAIR"
REASON_RESOURCE = "RESOURCE_BLOCKED"
REASON_CAP = "NOT_SELECTED_CAP"
REASON_BELOW_SCORE = "BELOW_MINIMUM_SCORE"
REASON_BELOW_DIMENSION = "BELOW_MINIMUM_DIMENSION"
REASON_TOO_FEW_REVIEWERS = "TOO_FEW_REVIEWERS_ABOVE_THRESHOLD"
STATUS_SELECTED = "SELECTED"
STATUS_NO_QUALIFIED = "NO_QUALIFIED_IDEA"

#: A subscore spread at or above this is a real disagreement, not noise.
DISAGREEMENT_SPREAD = 2.0


class ScoreError(ValueError):
    """A scoring report is not usable."""


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_report(report: dict, *, scorer: str) -> dict:
    """Check one reviewer's report against the rubric schema, strictly.

    Every proposal must be scored on exactly the rubric's dimensions: an unknown
    name, a renamed one or a missing one is an error the caller retries, never a
    partial average over the fields that happened to be present.
    """
    if not isinstance(report, dict):
        raise ScoreError("score report must be an object")
    scores = report.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ScoreError("scorer " + str(scorer) + " returned no scores")
    known = set(DIMENSIONS)
    seen = set()
    for entry in scores:
        if not isinstance(entry, dict) or not entry.get("idea_id"):
            raise ScoreError("scorer " + str(scorer) + " returned a score without an idea_id")
        identifier = str(entry["idea_id"])
        if identifier in seen:
            raise ScoreError("scorer " + str(scorer) + " scored " + identifier + " twice")
        seen.add(identifier)
        values = entry.get("scores")
        if not isinstance(values, dict):
            raise ScoreError("score for " + identifier + " has no dimension scores")
        unknown = sorted(set(values) - known)
        if unknown:
            raise ScoreError("score for " + identifier + " uses unknown dimensions: " +
                             ", ".join(unknown))
        missing = [dimension for dimension in DIMENSIONS if dimension not in values]
        if missing:
            raise ScoreError("score for " + identifier + " is missing dimensions: " +
                             ", ".join(missing))
        for dimension in DIMENSIONS:
            value = values[dimension]
            if not _number(value):
                raise ScoreError("score for " + identifier + " has a non-numeric " +
                                 dimension)
            if not SCALE[0] <= float(value) <= SCALE[1]:
                raise ScoreError("score for " + identifier + " is outside " +
                                 str(SCALE) + ": " + dimension)
        if not str(entry.get("rationale", "")).strip():
            raise ScoreError("score for " + identifier + " has no rationale")
        verdict = str(entry.get("scope_verdict", "")).upper()
        if verdict and verdict not in ("IN_SCOPE", "OUT_OF_SCOPE"):
            raise ScoreError("score for " + identifier + " has an unknown scope verdict: " +
                             verdict)
    return report


def build_matrix(ideas: list, reports: list) -> dict:
    """The weighted matrix: every reviewer on every idea, with the aggregate."""
    idea_ids = [idea["idea_id"] for idea in ideas]
    by_scorer = {}
    for report in reports:
        scorer = str(report.get("scorer", ""))
        if not scorer:
            raise ScoreError("a score report has no scorer identity")
        by_scorer[scorer] = {str(entry["idea_id"]): entry for entry in report["scores"]}
    missing = [scorer + ":" + identifier for scorer, entries in by_scorer.items()
               for identifier in idea_ids if identifier not in entries]
    if missing:
        raise ScoreError("every reviewer must judge every idea; missing " +
                         ", ".join(sorted(missing)))
    cost_by_idea = {}
    for idea in ideas:
        for key in ("cost_estimate_hours", "hours_for_frozen_budget"):
            value = idea.get(key)
            if _number(value):
                cost_by_idea[idea["idea_id"]] = float(value)
                break
    matrix = []
    for identifier in idea_ids:
        per_scorer = {scorer: {dimension: float(entries[identifier]["scores"][dimension])
                               for dimension in DIMENSIONS}
                      for scorer, entries in by_scorer.items()}
        medians = {dimension: statistics.median(
            [per_scorer[scorer][dimension] for scorer in sorted(per_scorer)])
            for dimension in DIMENSIONS}
        spreads = {dimension: max(per_scorer[scorer][dimension] for scorer in per_scorer)
                   - min(per_scorer[scorer][dimension] for scorer in per_scorer)
                   for dimension in DIMENSIONS}
        weighted = rubric_module.weighted_score(medians, RUBRIC)
        idea = next((entry for entry in ideas if entry["idea_id"] == identifier), {})
        verdict = scope_module.check_scope(idea)
        per_scorer_weighted = {scorer: rubric_module.weighted_score(per_scorer[scorer], RUBRIC)
                               for scorer in per_scorer}
        row = {
            "idea_id": identifier,
            "per_scorer": per_scorer,
            "per_scorer_weighted": per_scorer_weighted,
            "per_scorer_rationale": {scorer: by_scorer[scorer][identifier].get("rationale", "")
                                     for scorer in sorted(by_scorer)},
            "per_scorer_scope": {scorer: by_scorer[scorer][identifier].get("scope_verdict", "")
                                 for scorer in sorted(by_scorer)},
            "per_scorer_scope_reason": {
                scorer: by_scorer[scorer][identifier].get("scope_reason", "")
                for scorer in sorted(by_scorer)},
            "per_scorer_evidence": {scorer: by_scorer[scorer][identifier].get("evidence_ids", [])
                                    for scorer in sorted(by_scorer)},
            "per_scorer_objection": {scorer: by_scorer[scorer][identifier].get("objection", "")
                                     for scorer in sorted(by_scorer)},
            "medians": medians,
            "disagreement": spreads,
            "max_disagreement": max(spreads.values()) if spreads else 0.0,
            "disputed": sorted(dimension for dimension, spread in spreads.items()
                               if spread >= DISAGREEMENT_SPREAD),
            "weighted_score": weighted,
            "scorers_above_threshold": sum(
                1 for scorer, value in per_scorer_weighted.items()
                if value is not None and value >= DEFAULT_SELECTION["minimum_weighted_score"]),
            "scope_verdict": "IN_SCOPE" if verdict["in_scope"] else "OUT_OF_SCOPE",
            "scope_reasons": verdict["reasons"],
            "scope_decision_basis": "proposal content: behaviour signal, curved "
                                    "operation, coupling pathway, ablations",
            "evidence_gaps": [name for name, present in (verdict.get("checks") or {}).items()
                              if not present],
        }
        row["scope_disagreements"] = [
            {"scorer": scorer, "label": label,
             "reason": row["per_scorer_scope_reason"].get(scorer, "")}
            for scorer, label in row["per_scorer_scope"].items()
            if label and label != row["scope_verdict"]]
        if identifier in cost_by_idea:
            row["declared_hours"] = cost_by_idea[identifier]
        matrix.append(row)
    return {"rubric_version": RUBRIC["version"], "rubric_sha256": RUBRIC["sha256"],
            "dimensions": list(DIMENSIONS), "weights": dict(RUBRIC["weight_by_dimension"]),
            "scale": list(SCALE), "rows": matrix, "reviewers": sorted(by_scorer)}


def eligible(row: dict, rule: dict) -> tuple:
    """The hard eligibility of one row, with the reasons it is not eligible."""
    reasons = []
    if row.get("scope_verdict") == "OUT_OF_SCOPE":
        reasons.append(REASON_OUT_OF_SCOPE + ": " +
                       ", ".join(row.get("scope_reasons", [])[:4]))
    if row.get("weighted_score") is None:
        reasons.append(REASON_NEEDS_EVIDENCE +
                       ": an incomplete review has no weighted score")
    else:
        if row["weighted_score"] < rule["minimum_weighted_score"]:
            reasons.append(REASON_BELOW_SCORE + ": weighted " +
                           str(round(row["weighted_score"], 3)) + " below " +
                           str(rule["minimum_weighted_score"]))
        weak = [dimension for dimension in DIMENSIONS
                if row["medians"].get(dimension) is not None
                and row["medians"][dimension] < rule["minimum_dimension_median"]]
        if weak:
            reasons.append(REASON_BELOW_DIMENSION + ": " + ", ".join(weak))
        if row["scorers_above_threshold"] < rule["min_reviewers_above_threshold"]:
            reasons.append(REASON_TOO_FEW_REVIEWERS + ": " +
                           str(row["scorers_above_threshold"]))
    declared = row.get("declared_hours")
    if declared is not None and declared > rule["declared_hours_budget"]:
        reasons.append(REASON_RESOURCE + ": declared " + str(declared) + " h")
    return (not reasons), reasons


def pairwise(rows: list, comparisons: list) -> dict:
    """Aggregate order-balanced A-vs-B comparisons deterministically.

    ``comparisons`` are the reviewers' votes: each names the two ideas, the
    reviewer and the preferred idea, plus optionally which idea was printed
    first. The aggregate is a Copeland score (a majority win/loss over the
    reviewers who compared the pair). Cycles, ties and a position bias are
    recorded, never hidden.
    """
    pairs = {}
    position_bias = {"first_listed_wins": 0, "second_listed_wins": 0}
    for comparison in comparisons:
        first = str(comparison.get("first", ""))
        second = str(comparison.get("second", ""))
        winner = str(comparison.get("winner", ""))
        if not first or not second or winner not in (first, second):
            raise ScoreError("a pairwise comparison is malformed: " +
                             str(comparison)[:120])
        key = tuple(sorted((first, second)))
        record = pairs.setdefault(key, {"votes": {first: 0, second: 0},
                                        "reviewers": []})
        record["votes"][winner] += 1
        record["reviewers"].append(str(comparison.get("reviewer", "")))
        first_listed = str(comparison.get("first_listed", ""))
        if first_listed in (first, second):
            position_bias["first_listed_wins" if winner == first_listed
                          else "second_listed_wins"] += 1
    totals = {row["idea_id"]: 0 for row in rows}
    detail = []
    for (a, b), record in sorted(pairs.items()):
        wins_a, wins_b = record["votes"][a], record["votes"][b]
        if wins_a == wins_b:
            detail.append({"pair": [a, b], "winner": None, "votes": record["votes"],
                           "reason": "tie"})
            continue
        winner, loser = (a, b) if wins_a > wins_b else (b, a)
        totals[winner] = totals.get(winner, 0) + 1
        totals[loser] = totals.get(loser, 0) - 1
        detail.append({"pair": [a, b], "winner": winner, "votes": record["votes"],
                       "disagreement": wins_a > 0 and wins_b > 0})
    cycles = []
    for a, b, c in itertools.permutations(sorted(totals), 3):
        decided = {}
        complete = True
        for pair in ((a, b), (b, c), (c, a)):
            record = pairs.get(tuple(sorted(pair)))
            if not record:
                complete = False
                break
            left, right = pair
            decided[pair] = (left if record["votes"][left] > record["votes"][right]
                             else right if record["votes"][right] > record["votes"][left]
                             else None)
        if complete and decided.get((a, b)) == a and decided.get((b, c)) == b \
                and decided.get((c, a)) == c:
            canonical = min([a, b, c])
            cycle = [canonical]
            rest = sorted([member for member in (a, b, c) if member != canonical])
            cycle.extend(rest)
            if cycle not in cycles:
                cycles.append(cycle)
    return {"rubric_version": RUBRIC["version"], "rubric_sha256": RUBRIC["sha256"],
            "copeland": totals, "comparisons": detail, "cycles": cycles,
            "position_bias": position_bias,
            "ties": [entry["pair"] for entry in detail if entry["winner"] is None]}


def select(matrix: dict, *, rule: dict = None, pairwise_result: dict = None) -> dict:
    """Apply the explicit policy and return the decision with its reasons.

    The order is the one the rubric documents: scope, then evidence and
    feasibility, then the weighted score, then the pairwise outcome, then the
    idea id. No automatic fallback: when nobody qualifies the round says so.
    """
    rule = dict(DEFAULT_SELECTION if rule is None else rule)
    rows = matrix["rows"]
    copeland = (pairwise_result or {}).get("copeland", {})
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
                                 else REASON_NEEDS_EVIDENCE if any(
                                     reason.startswith(REASON_NEEDS_EVIDENCE)
                                     for reason in reasons)
                                 else REASON_BELOW_SCORE),
                             "weighted_score": row.get("weighted_score"),
                             "max_disagreement": row.get("max_disagreement")})
    qualified.sort(key=lambda row: (-(row["weighted_score"] or 0.0),
                                    -(copeland.get(row["idea_id"], 0)),
                                    row["idea_id"]))
    if len(qualified) > rule["max_selected"]:
        dropped = qualified[rule["max_selected"]:]
        for row in dropped:
            rejected.append({"idea_id": row["idea_id"], "status": REASON_CAP,
                             "reasons": [REASON_CAP + ": outside the one-mechanism cap"],
                             "weighted_score": row.get("weighted_score"),
                             "max_disagreement": row.get("max_disagreement")})
        qualified = qualified[:rule["max_selected"]]
    selected = [{"idea_id": row["idea_id"], "weighted_score": row["weighted_score"],
                 "medians": row["medians"], "scope_verdict": row["scope_verdict"],
                 "max_disagreement": row["max_disagreement"],
                 "copeland": copeland.get(row["idea_id"], 0),
                 "status": STATUS_SELECTED}
                for row in qualified]
    backups = [{"idea_id": row["idea_id"], "weighted_score": row["weighted_score"],
                "status": "BACKUP"}
               for row in sorted((row for row in rows
                                  if row["idea_id"] not in {entry["idea_id"]
                                                            for entry in selected}),
                                 key=lambda row: (-(row["weighted_score"] or 0.0),
                                                  row["idea_id"]))[:rule["pairwise_finalists"]]]
    status = STATUS_SELECTED if selected else STATUS_NO_QUALIFIED
    return {"rule": rule, "status": status, "selected": selected, "backups": backups,
            "rejected": rejected,
            "selection_is_multiple": len(selected) > 1,
            "tie_break_order": list(rule.get("tie_break_order", [])),
            "pairwise": pairwise_result,
            "disagreement": {row["idea_id"]: row["max_disagreement"] for row in rows},
            "disputed": {row["idea_id"]: row.get("disputed", []) for row in rows}}


def evaluate(ideas: list, reports: list, comparisons: list = None,
             rule: dict = None) -> dict:
    """Matrix, pairwise, selection and the rubric identity, as one record."""
    for report in reports:
        validate_report(report, scorer=str(report.get("scorer", "")))
    matrix = build_matrix(ideas, reports)
    pairwise_result = pairwise(matrix["rows"], comparisons or []) if comparisons else None
    selection = select(matrix, rule=rule, pairwise_result=pairwise_result)
    return {"rubric_version": RUBRIC["version"], "rubric_sha256": RUBRIC["sha256"],
            "matrix": matrix, "pairwise": pairwise_result, "selection": selection}
