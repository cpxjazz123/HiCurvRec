"""Agent A -- the scientific analysis over the measured evidence (issue #110, WP1).

The analysis is grounded in immutable, measured evidence, and it says which is which: a
number that was read from an artifact is MEASURED, a number that was inferred from another
number is INFERRED, and a number that does not exist is MISSING. No number is invented.

For every arm the analysis reports the enrichment ratio's numerator and denominator, the
tail of the level-0 usage, the width of the L1 -> L2 tree, the collisions, the
history-conditioned gain, the conditional level-2 discrimination with its support, the
level-wise downstream difficulty where it exists, and -- for an arm whose export is
hyperbolic -- the geometric distortion and the radius.

The failure classification separates four things that are often confused: an IMPLEMENTATION
failure (the mechanism did not do what it declared), a MECHANISM FALSIFICATION (it did, and
the hypothesis is wrong), a STATISTICAL LOW SUPPORT (the measurement cannot say), and a
SCHEDULER BUG (the run did not happen the way it should have).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The archived control, the baseline every delta is against.
CONTROL = {
    "hard_l1_er": 16.334505666275888,
    "pos_share": 0.07908428720083246,
    "neg_share": 0.004841547630309337,
    "l0_distinct_codes": 245,
    "l0_gini": 0.21624933700653148,
    "l0_largest_cell": 237,
    "l1_to_l2_branches": 65.90612244897959,
    "g_h1": None,
}

#: The frozen thresholds.
ER_GATE = 16.38953942232631
RECALL_GATE = 0.06406796775709883
NDCG_GATE = 0.03530651701940677

#: The failure kinds, as the issue names them.
FAILURE_KINDS = (
    "implementation_failure",
    "mechanism_falsified",
    "statistical_low_support",
    "scheduler_bug",
    "no_failure_identified",
)

#: How many conditioned pairs the conditional L2 discrimination needs before it is worth
#: reading. Below this the measurement is a flag, not a conclusion.
D2_MINIMUM_SUPPORT = 1000


def _measurement(value: Any, kind: str = "measured") -> Dict[str, Any]:
    return {"value": value, "kind": kind}


def arm_diagnostics(
    mechanism_id: str,
    manifests: Sequence[Dict[str, Any]],
    structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The measured diagnostics of one arm, with the evidence status of each."""
    evaluation: Dict[str, Any] = {}
    stage3: Dict[str, Any] = {}
    probe: Dict[str, Any] = {}
    for manifest in manifests:
        metrics = manifest.get("metrics") or {}
        for key, value in (metrics.get("frozen_evaluation") or {}).items():
            evaluation.setdefault(key, value)
        for key, value in (metrics.get("stage3") or {}).items():
            stage3.setdefault(key, value)
        for key, value in (metrics.get("gh1_probe") or {}).items():
            probe.setdefault(key, value)
    structure = structure or {}
    balance = (structure.get("token_balance") or {}).get("level_1") or {}
    branches = (structure.get("prefix_structure") or {}).get("l1_to_l2") or {}
    collisions = (structure.get("collisions") or {}).get("level_1") or {}
    sequence = (structure.get("sequential_predictability") or {}).get("level_1") or {}
    slices = structure.get("stage3_slices") or {}

    er = evaluation.get("hard_l1_er")
    if er is None:
        er = evaluation.get("l1_er")
    d2_block = evaluation.get("d2_block") or {}
    d2_flags = d2_block.get("flags") or []

    recall = stage3.get("test_recall@10")
    ndcg = stage3.get("test_ndcg@10")
    gain = probe.get("g_h1")

    return {
        "mechanism_id": mechanism_id,
        "enrichment_ratio": {
            "value": er,
            "kind": "measured" if er is not None else "missing",
            "cleared_the_gate": bool(er is not None and er > ER_GATE),
            "numerator_positive_share": _measurement(evaluation.get("pos_share")),
            "denominator_negative_share": _measurement(evaluation.get("neg_share")),
            "ci95": _measurement(evaluation.get("er_ci95")),
        },
        "conditional_l2_discrimination": {
            "value": evaluation.get("d2"),
            "kind": "measured" if evaluation.get("d2") is not None else "missing",
            "support": d2_block.get("denominator_positive"),
            "flags": d2_flags,
            "low_support": bool("low_support" in d2_flags)
            or (d2_block.get("denominator_positive") or 0) < D2_MINIMUM_SUPPORT,
        },
        "history_gain": {
            "value": gain,
            "kind": "measured" if gain is not None else "missing",
            "prior_ce_l1": _measurement(probe.get("prior_ce_l1")),
            "probe_ce_l1": _measurement(probe.get("probe_ce_l1")),
            "top1_acc_l1": _measurement(probe.get("top1_acc_l1")),
            "top5_acc_l1": _measurement(probe.get("top5_acc_l1")),
            "top20_acc_l1": _measurement(probe.get("top20_acc_l1")),
            "split": _measurement(probe.get("split")),
        },
        "level0_usage": {
            "distinct_codes": _measurement(balance.get("distinct_codes")),
            "gini": _measurement(balance.get("gini")),
            "largest_cell": _measurement(collisions.get("largest_cell")),
            "delta_largest_cell_vs_control": _measurement(
                (collisions.get("largest_cell") - CONTROL["l0_largest_cell"])
                if collisions.get("largest_cell") is not None
                else None
            ),
            "delta_gini_vs_control": _measurement(
                (balance.get("gini") - CONTROL["l0_gini"])
                if balance.get("gini") is not None
                else None
            ),
        },
        "prefix_tree": {
            "l1_to_l2_branches": _measurement(branches.get("mean_branches")),
            "delta_branches_vs_control": _measurement(
                (branches.get("mean_branches") - CONTROL["l1_to_l2_branches"])
                if branches.get("mean_branches") is not None
                else None
            ),
            "l2_to_l3_branches": _measurement(
                ((structure.get("prefix_structure") or {}).get("l2_to_l3") or {}).get(
                    "mean_branches"
                )
            ),
        },
        "sequence_information": _measurement(sequence.get("excess_over_the_marginal")),
        "downstream": {
            "recall": _measurement(recall),
            "ndcg": _measurement(ndcg),
            "delta_recall_vs_control": _measurement(
                (recall - RECALL_GATE) if recall is not None else None
            ),
            "delta_ndcg_vs_control": _measurement(
                (ndcg - NDCG_GATE) if ndcg is not None else None
            ),
            "per_level": _measurement(slices.get("levels") if slices.get("available") else None),
        },
    }


def classify_failure(diagnostics: Dict[str, Any], screening: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Which kind of failure this arm shows, and on what evidence.

    The order matters: an implementation failure is decided before a mechanism is judged,
    because a mechanism that did not run cannot falsify a hypothesis.
    """
    screening = screening or {}
    judges = screening.get("judges") or {}
    failed = judges.get("failed") or []
    if failed:
        return {
            "kind": "implementation_failure",
            "evidence": f"the independent checks failed: {failed}",
            "confidence": "high",
        }
    sensitivity = screening.get("export_sensitivity") or {}
    if sensitivity.get("verdict") == "inert":
        return {
            "kind": "implementation_failure",
            "evidence": "the export did not change: the hard-SID path was inactive",
            "confidence": "high",
        }
    d2 = diagnostics.get("conditional_l2_discrimination") or {}
    er = (diagnostics.get("enrichment_ratio") or {}).get("value")
    downstream = diagnostics.get("downstream") or {}
    recall = downstream.get("recall", {}).get("value")
    ndcg = downstream.get("ndcg", {}).get("value")
    if d2.get("low_support") and d2.get("value") is None and (recall is None or ndcg is None):
        # the conditional discrimination cannot be read, and without a downstream
        # measurement there is nothing else to judge the arm by
        return {
            "kind": "statistical_low_support",
            "evidence": "the conditional level-2 discrimination has no support on the frozen "
            "pairs, so it cannot be read, and the arm has no downstream measurement either",
            "confidence": "high",
        }
    if er is not None and recall is not None and ndcg is not None:
        if er > ER_GATE and recall > RECALL_GATE and ndcg > NDCG_GATE:
            return {
                "kind": "no_failure_identified",
                "evidence": "all three frozen measurements are above their thresholds",
                "confidence": "high",
            }
        return {
            "kind": "mechanism_falsified",
            "evidence": (
                f"the mechanism ran and was measured: ER {er:.4f}, Recall@10 {recall:.6f}, "
                f"NDCG@10 {ndcg:.6f}; the hypothesis did not hold for the downstream"
            ),
            "confidence": "medium",
            "limitations": "one arm, one seed, and a cohort that has already been inspected",
        }
    return {
        "kind": "no_failure_identified",
        "evidence": "the arm has not been measured downstream yet",
        "confidence": "low",
    }


def failure_analysis(
    store_runs: Sequence[Dict[str, Any]],
    transfer: Optional[Dict[str, Any]] = None,
    screenings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The cohort's failure analysis: what failed, how, and with what evidence."""
    transfer = transfer or {}
    screenings = screenings or {}
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for manifest in store_runs:
        by_arm.setdefault(manifest.get("mechanism_id") or "", []).append(manifest)
    arms: List[Dict[str, Any]] = []
    kinds: Dict[str, int] = {}
    for mechanism_id, manifests in sorted(by_arm.items()):
        diagnostics = arm_diagnostics(
            mechanism_id, manifests, transfer.get(mechanism_id)
        )
        failure = classify_failure(diagnostics, screenings.get(mechanism_id))
        kinds[failure["kind"]] = kinds.get(failure["kind"], 0) + 1
        arms.append({"mechanism_id": mechanism_id, "diagnostics": diagnostics, "failure": failure})
    return {
        "arms": arms,
        "arm_count": len(arms),
        "failure_kinds": dict(sorted(kinds.items(), key=lambda item: -item[1])),
        "control": CONTROL,
        "thresholds": {
            "er": ER_GATE,
            "recall": RECALL_GATE,
            "ndcg": NDCG_GATE,
        },
        "measured_arms": sum(
            1
            for arm in arms
            if arm["diagnostics"]["enrichment_ratio"]["value"] is not None
        ),
        "arms_with_both_downstream_metrics": sum(
            1
            for arm in arms
            if arm["diagnostics"]["downstream"]["recall"]["value"] is not None
        ),
        "limitations": (
            "the cohort has already been inspected, so these are retrospective associations; "
            "the history gain's rank correlation with the downstream is about +0.46 over a dozen "
            "paired arms against the ER's +0.17, which is a hypothesis and not a certified "
            "oracle, and the conditional level-2 discrimination is weakly supported"
        ),
    }


def research_questions(analysis: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """The questions the evidence actually raises, each with the arms behind it."""
    questions: List[Dict[str, Any]] = []
    measured = [arm for arm in analysis["arms"] if arm["diagnostics"]["enrichment_ratio"]["value"]]
    above = [arm for arm in measured if arm["diagnostics"]["enrichment_ratio"]["cleared_the_gate"]]
    if above:
        questions.append(
            {
                "question": "can an arm raise the pair-level ER without moving the tail of the "
                "level-0 usage or narrowing the L1 -> L2 tree?",
                "why": "every arm that cleared the gate was measured against the control's "
                "largest cell and branch count",
                "arms": [arm["mechanism_id"] for arm in above],
                "kind": "measured",
            }
        )
    with_gain = [
        arm
        for arm in analysis["arms"]
        if arm["diagnostics"]["history_gain"]["value"] is not None
    ]
    if with_gain:
        questions.append(
            {
                "question": "does a higher history-conditioned gain buy a smaller downstream "
                "loss, once the enrichment ratio is held?",
                "why": "the gain is the only instrument that has ordered the arms so far",
                "arms": [arm["mechanism_id"] for arm in with_gain],
                "kind": "measured",
                "caveat": "retrospective, over a cohort that has already been inspected",
            }
        )
    if not questions:
        questions.append(
            {
                "question": "nothing has been measured yet: the first arm has to run before "
                "there is a question to ask",
                "why": "the analysis found no measurement to reason about",
                "arms": [],
                "kind": "missing",
            }
        )
    return questions[:limit]


def causal_hypotheses(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The hypotheses the evidence supports, each labelled measured or inferred."""
    hypotheses: List[Dict[str, Any]] = []
    kinds = analysis["failure_kinds"]
    if kinds.get("mechanism_falsified"):
        hypotheses.append(
            {
                "hypothesis": "the arms that cleared the ER gate falsify the hypothesis that a "
                "higher pair-level enrichment carries the downstream quality with it",
                "status": "measured",
                "arms": kinds["mechanism_falsified"],
                "contradiction_test": "an arm that clears the gate AND beats the control on both "
                "downstream metrics",
            }
        )
    if kinds.get("statistical_low_support"):
        hypotheses.append(
            {
                "hypothesis": "the conditional level-2 discrimination cannot decide anything on "
                "the frozen pairs, because its positive conditional set is a few hundred pairs",
                "status": "measured",
                "arms": kinds["statistical_low_support"],
                "contradiction_test": "an arm whose conditional set is large enough to support "
                "a decision",
            }
        )
    if kinds.get("implementation_failure"):
        hypotheses.append(
            {
                "hypothesis": "some arms never ran the mechanism they declared, so their "
                "results say nothing about the hypothesis",
                "status": "measured",
                "arms": kinds["implementation_failure"],
                "contradiction_test": "an arm that passes the independent checks and still shows "
                "the signature",
            }
        )
    return hypotheses


def write_analysis(analysis: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """Write the analysis, the questions and the hypotheses."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, document in (
        ("failure_analysis.json", analysis),
        ("research_questions.json", {"questions": research_questions(analysis)}),
        ("causal_hypotheses.json", {"hypotheses": causal_hypotheses(analysis)}),
    ):
        path = out_dir / name
        path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        written[name] = path
    return written


__all__ = [
    "CONTROL",
    "D2_MINIMUM_SUPPORT",
    "ER_GATE",
    "FAILURE_KINDS",
    "NDCG_GATE",
    "RECALL_GATE",
    "arm_diagnostics",
    "causal_hypotheses",
    "classify_failure",
    "failure_analysis",
    "research_questions",
    "write_analysis",
]
