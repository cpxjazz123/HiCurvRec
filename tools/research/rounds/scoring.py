"""Idea scoring: three independent scorers, a full matrix, and a multi-select.

The issue requires that every scorer judges EVERY idea, that the complete score
matrix is persisted, that disagreement is visible, that one agent's veto cannot
decide the round, and that the selection rule is explicit and normally selects
more than one candidate. Nothing here looks at ER, prefix counts or any other
Stage 2 diagnostic: selection is innovation-first.
"""
from __future__ import annotations

import statistics

from . import scope as scope_module

DIMENSIONS = ("novelty", "scientific_motivation", "feasibility",
              "downstream_impact", "experimental_distinctness")
SCALE = (0, 5)

#: The default rule keeps every idea whose median across the three scorers
#: reaches the threshold, and never returns fewer than one idea when any idea
#: qualifies. It is data, not code, so an operator can change it.
DEFAULT_SELECTION = {
    "median_threshold": 3.0,
    "mean_threshold": 3.0,
    "min_dimension_median": 2.0,
    "max_selected": 6,
    "require_min_scorers_above_threshold": 2,
    "fallback_to_best": True,
    # A proposal that does not couple behaviour and geometry is never selected,
    # not even as the fallback.
    "require_scope": True,
    # An idea that declares a per-step cost making the frozen 3000-epoch budget
    # infeasible cannot be implemented here, whatever its other scores.
    "max_declared_hours_for_the_frozen_budget": 72.0,
}


class ScoreError(ValueError):
    """A scoring report is not usable."""


def validate_report(report: dict, *, scorer: str) -> dict:
    """Check one scorer's report over all ideas.

    A missing scope verdict is recorded, not rejected: the scope decision is
    taken from the proposal's own content by :func:`scope.check_scope`, so one
    reviewer's label never decides an idea on its own.
    """
    if not isinstance(report, dict):
        raise ScoreError("score report must be an object")
    scores = report.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ScoreError("scorer " + str(scorer) + " returned no scores")
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
        missing = []
        for dimension in DIMENSIONS:
            value = values.get(dimension)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                missing.append(dimension)
                continue
            if not SCALE[0] <= float(value) <= SCALE[1]:
                raise ScoreError("score for " + identifier + " is outside 0-5: " + dimension)
        if missing:
            # One incomplete line does not cost the scorer its other ideas: the
            # gap is recorded and the matrix shows it.
            entry["missing_dimensions"] = missing
        if not str(entry.get("rationale", "")).strip():
            entry["rationale_missing"] = True
        verdict = str(entry.get("scope_verdict", "")).upper()
        if verdict and verdict not in ("IN_SCOPE", "OUT_OF_SCOPE"):
            raise ScoreError("score for " + identifier + " has an unknown scope verdict: "
                             + verdict)
        if verdict:
            entry["scope_verdict"] = verdict
        else:
            entry["scope_verdict_missing"] = True
    return report


def build_matrix(ideas: list, reports: list) -> dict:
    """The complete matrix: every scorer against every idea, with the aggregate."""
    idea_ids = [idea["idea_id"] for idea in ideas]
    by_scorer = {}
    for report in reports:
        scorer = str(report.get("scorer", ""))
        if not scorer:
            raise ScoreError("a score report has no scorer identity")
        by_scorer[scorer] = {str(entry["idea_id"]): entry
                             for entry in report["scores"]}
    missing = []
    for scorer, entries in by_scorer.items():
        for identifier in idea_ids:
            if identifier not in entries:
                missing.append(scorer + ":" + identifier)
    if missing:
        raise ScoreError("every scorer must judge every idea; missing " + ", ".join(missing))
    cost_by_idea = {}
    for idea in ideas:
        for key in ("cost_estimate_hours", "hours_for_frozen_budget"):
            value = idea.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cost_by_idea[idea["idea_id"]] = float(value)
                break
    matrix = []
    for identifier in idea_ids:
        per_scorer = {}
        incomplete = []
        for scorer, entries in by_scorer.items():
            values = entries[identifier]["scores"]
            per_scorer[scorer] = {dimension: float(values[dimension])
                                  for dimension in DIMENSIONS if dimension in values}
            if len(per_scorer[scorer]) != len(DIMENSIONS) or \
                    entries[identifier].get("rationale_missing"):
                incomplete.append(scorer)
        medians = {dimension: statistics.median(
            [per_scorer[scorer][dimension] for scorer in sorted(per_scorer)
             if dimension in per_scorer[scorer]])
            for dimension in DIMENSIONS}
        means = {dimension: (sum(per_scorer[scorer][dimension] for scorer in per_scorer
                                 if dimension in per_scorer[scorer])
                             / len([scorer for scorer in per_scorer
                                    if dimension in per_scorer[scorer]]))
                 for dimension in DIMENSIONS}
        spreads = {dimension: max(per_scorer[scorer][dimension] for scorer in per_scorer
                                  if dimension in per_scorer[scorer])
                   - min(per_scorer[scorer][dimension] for scorer in per_scorer
                         if dimension in per_scorer[scorer])
                   for dimension in DIMENSIONS}
        overall_median = statistics.median(
            [statistics.median([per_scorer[scorer][dimension] for scorer in per_scorer
                                 if dimension in per_scorer[scorer]])
             for dimension in DIMENSIONS])
        overall_mean = sum(means[dimension] for dimension in DIMENSIONS) / len(DIMENSIONS)
        row = {
            "idea_id": identifier,
            "per_scorer": per_scorer,
            "per_scorer_rationale": {scorer: by_scorer[scorer][identifier].get("rationale", "")
                                     for scorer in sorted(by_scorer)},
            "medians": medians,
            "means": means,
            "disagreement": spreads,
            "max_disagreement": max(spreads.values()),
            "overall_median": overall_median,
            "overall_mean": overall_mean,
            "scorers_above_median_threshold": sum(
                1 for scorer in per_scorer
                if statistics.median([per_scorer[scorer][dimension]
                                      for dimension in DIMENSIONS
                                      if dimension in per_scorer[scorer]])
                >= DEFAULT_SELECTION["median_threshold"]),
            "incomplete_scorers": sorted(incomplete),
            "per_scorer_scope": {scorer: by_scorer[scorer][identifier].get("scope_verdict", "")
                                 for scorer in sorted(by_scorer)},
            "per_scorer_scope_reason": {
                scorer: by_scorer[scorer][identifier].get("scope_reason", "")
                for scorer in sorted(by_scorer)},
        }
        # The scope decision is audited against the proposal's own content, so
        # it is not one reviewer's opinion: a reviewer that disagrees with the
        # content is recorded with its reason instead of deciding the idea.
        idea = next((entry for entry in ideas if entry["idea_id"] == identifier), {})
        verdict = scope_module.check_scope(idea)
        row["scope_verdict"] = "IN_SCOPE" if verdict["in_scope"] else "OUT_OF_SCOPE"
        row["scope_reasons"] = verdict["reasons"]
        row["scope_decision_basis"] = "proposal content: behaviour signal, curved " \
                                      "operation, coupling pathway, ablations"
        row["scope_disagreements"] = [
            {"scorer": scorer, "label": label,
             "reason": row["per_scorer_scope_reason"].get(scorer, "")}
            for scorer, label in row["per_scorer_scope"].items()
            if label and label != row["scope_verdict"]]
        if identifier in cost_by_idea:
            row["declared_hours"] = cost_by_idea[identifier]
        matrix.append(row)
    return {"dimensions": list(DIMENSIONS), "scale": list(SCALE), "rows": matrix,
            "scorers": sorted(by_scorer)}


def select(matrix: dict, *, rule: dict = None) -> dict:
    """Apply the explicit selection rule and return the selection with reasons."""
    rule = dict(DEFAULT_SELECTION if rule is None else rule)
    selected, rejected = [], []
    for row in matrix["rows"]:
        reasons = []
        # Scope is a gate, not a dimension: an out-of-scope proposal is not
        # compensated by its scores.
        if row.get("scope_verdict") == "OUT_OF_SCOPE":
            reasons.append("out of scope: " + ", ".join(row.get("scope_reasons", [])[:6]))
        weak_dimension = [dimension for dimension in DIMENSIONS
                          if row["medians"][dimension] < rule["min_dimension_median"]]
        if weak_dimension:
            reasons.append("median below the per-dimension floor on " +
                           ", ".join(weak_dimension))
        if row["overall_median"] < rule["median_threshold"]:
            reasons.append("overall median " + str(round(row["overall_median"], 3)) +
                           " below " + str(rule["median_threshold"]))
        if row["overall_mean"] < rule["mean_threshold"]:
            reasons.append("overall mean " + str(round(row["overall_mean"], 3)) +
                           " below " + str(rule["mean_threshold"]))
        if row["scorers_above_median_threshold"] < rule["require_min_scorers_above_threshold"]:
            reasons.append("only " + str(row["scorers_above_median_threshold"]) +
                           " scorer(s) reached the median threshold")
        declared = row.get("declared_hours")
        if declared is not None and declared > rule["max_declared_hours_for_the_frozen_budget"]:
            reasons.append("declared cost " + str(declared) + " h exceeds the " +
                           str(rule["max_declared_hours_for_the_frozen_budget"]) +
                           " h budget for the frozen 3000-epoch run")
        if reasons:
            rejected.append({"idea_id": row["idea_id"], "reasons": reasons,
                             "overall_median": row["overall_median"],
                             "max_disagreement": row["max_disagreement"]})
        else:
            selected.append({"idea_id": row["idea_id"],
                             "overall_median": row["overall_median"],
                             "overall_mean": row["overall_mean"],
                             "max_disagreement": row["max_disagreement"]})
    selected.sort(key=lambda entry: (-entry["overall_median"], -entry["overall_mean"],
                                     entry["idea_id"]))
    if len(selected) > rule["max_selected"]:
        dropped = selected[rule["max_selected"]:]
        for entry in dropped:
            rejected.append({"idea_id": entry["idea_id"],
                             "reasons": ["outside the configured selection cap"],
                             "overall_median": entry["overall_median"]})
        selected = selected[:rule["max_selected"]]
    eligible = [row for row in matrix["rows"]
                if not rule.get("require_scope") or row.get("scope_verdict") != "OUT_OF_SCOPE"]
    if not selected and rule["fallback_to_best"] and eligible:
        best = max(eligible, key=lambda row: (row["overall_median"], row["overall_mean"]))
        selected = [{"idea_id": best["idea_id"], "overall_median": best["overall_median"],
                     "overall_mean": best["overall_mean"],
                     "max_disagreement": best["max_disagreement"],
                     "selection_reason": "fallback: no idea met the rule, the strongest median is kept"}]
    return {"rule": rule, "selected": selected, "rejected": rejected,
            "selection_is_multiple": len(selected) > 1,
            "disagreement": {row["idea_id"]: row["max_disagreement"]
                             for row in matrix["rows"]}}
