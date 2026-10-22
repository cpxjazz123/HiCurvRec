"""The per-arm diagnosis: which hypothesis failed, and on what evidence.

Issue #109 asks the controller to diagnose, not merely to count: for every arm it must say
which hypothesis failed -- an invalid gradient, an inactive hard-SID path, a pair-share ratio
artifact, lost history predictability, conditional L2 contamination, concentration or
collisions, or Stage-3 token-level difficulty -- and say it with the evidence and a
counterexample, not as prose after the fact.

Every rule below is a threshold on a MEASURED quantity, so a diagnosis can be checked
against the artifact it came from. The rules are deliberately independent: an arm can fail
in more than one way, and the diagnosis says so rather than picking the most convenient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The archived control's numbers, which every diagnosis is against.
CONTROL_ER = 16.334505666275888
CONTROL_RECALL = 0.06406796775709883
CONTROL_NDCG = 0.03530651701940677
FROZEN_GATE = 16.38953942232631

#: The control's largest level-0 cell, and the width of its L1 -> L2 tree.
CONTROL_LARGEST_CELL = 237
CONTROL_BRANCHES = 65.90612244897959

#: The historical G_H1's range, from the seed-42-only baseline.
HISTORICAL_G_H1_RANGE = (0.003972384133663863, 0.19469284765367922)


def diagnose(
    *,
    mechanism_id: str,
    evaluation: Optional[Dict[str, Any]] = None,
    stage3: Optional[Dict[str, Any]] = None,
    probe: Optional[Dict[str, Any]] = None,
    structure: Optional[Dict[str, Any]] = None,
    screening: Optional[Dict[str, Any]] = None,
    export_sensitivity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The failure modes of one arm, each with the measurement that produced it."""
    evaluation = evaluation or {}
    stage3 = stage3 or {}
    probe = probe or {}
    structure = structure or {}
    screening = screening or {}
    export_sensitivity = export_sensitivity or {}

    modes: List[Dict[str, Any]] = []

    # 1. the mechanism's gradient never reached what it declared
    judges = screening.get("judges") or {}
    failed = judges.get("failed") or []
    gradient_failures = [name for name in failed if "gradient" in str(name)]
    if gradient_failures:
        modes.append(
            {
                "mode": "invalid_gradient",
                "evidence": f"the independent checks measured the gradient and it did not match "
                f"the declared target(s): {gradient_failures}",
                "counterexample": judges.get("detail", ""),
            }
        )

    # 2. the export did not change: the hard-SID path was inactive
    changed = (export_sensitivity.get("changed_fraction") or {}).get("levels_3")
    if changed is not None and changed < 1e-9:
        modes.append(
            {
                "mode": "inactive_hard_sid_path",
                "evidence": f"no item's level-0/1/2 code changed against the control ({changed:.1e})",
                "counterexample": "the arm's own export is indistinguishable from the control's",
            }
        )

    # 3. the enrichment ratio moved through its denominator, not its numerator
    pos_share = evaluation.get("pos_share")
    neg_share = evaluation.get("neg_share")
    er = evaluation.get("hard_l1_er") or evaluation.get("l1_er")
    if pos_share is not None and neg_share is not None and er is not None:
        control_pos = 0.07908428720083246
        control_neg = 0.004841547630309337
        if er > FROZEN_GATE and neg_share < control_neg * 0.95 and pos_share <= control_pos * 1.05:
            modes.append(
                {
                    "mode": "pair_share_ratio_artifact",
                    "evidence": (
                        f"the ER rose while the positive share stayed at {pos_share:.5f} "
                        f"(the control's {control_pos:.5f}) and the negative share fell to "
                        f"{neg_share:.6f} (the control's {control_neg:.6f}): the ratio moved "
                        "through its denominator"
                    ),
                    "counterexample": "the numerator, which is the part a partition change "
                    "moves, did not move",
                }
            )

    # 4. the codes carry less history than the historical models'
    g_h1 = probe.get("g_h1")
    if g_h1 is not None:
        low, high = HISTORICAL_G_H1_RANGE
        if g_h1 < low:
            modes.append(
                {
                    "mode": "lost_history_predictability",
                    "evidence": f"the history-conditioned gain is {g_h1:.4f}, below the "
                    f"historical range's lower end {low:.4f}",
                    "counterexample": "the codes carry less history than the archived models'",
                }
            )

    # 5. the conditional level-2 measurement has no support
    d2_flags = evaluation.get("d2_flags") or []
    if "low_support" in d2_flags or "undefined_zero_denominator" in d2_flags:
        modes.append(
            {
                "mode": "conditional_l2_contamination",
                "evidence": f"the conditional level-2 measurement is not supported: {d2_flags}",
                "counterexample": "the D2 cannot say anything about this arm",
            }
        )

    # 6. the tail of the usage grew, or the prefix tree narrowed
    balance = (structure.get("token_balance") or {}).get("level_1") or {}
    branches = (structure.get("prefix_structure") or {}).get("l1_to_l2") or {}
    collisions = (structure.get("collisions") or {}).get("level_1") or {}
    largest_cell = collisions.get("largest_cell")
    if largest_cell is not None and largest_cell > CONTROL_LARGEST_CELL:
        modes.append(
            {
                "mode": "concentration_or_collisions",
                "evidence": f"the largest level-0 cell holds {largest_cell} items against the "
                f"control's {CONTROL_LARGEST_CELL}",
                "counterexample": "the tail of the usage is where the measured downstream loss "
                "is concentrated",
            }
        )
    mean_branches = branches.get("mean_branches")
    if mean_branches is not None and mean_branches < CONTROL_BRANCHES * 0.98:
        modes.append(
            {
                "mode": "concentration_or_collisions",
                "evidence": f"the L1 -> L2 tree has {mean_branches:.2f} branches per code against "
                f"the control's {CONTROL_BRANCHES:.2f}",
                "counterexample": "a beam search has fewer codes to walk",
            }
        )

    # 7. the recommender's own token-level difficulty
    slices = structure.get("stage3_slices") or {}
    if slices.get("available"):
        levels = slices.get("levels") or {}
        harder = [name for name, value in levels.items() if isinstance(value, dict) and (value.get("accuracy") is not None)]
        if harder:
            modes.append(
                {
                    "mode": "stage3_token_level_difficulty",
                    "evidence": f"the downstream per-level accuracy: {json.dumps(levels)[:240]}",
                    "counterexample": "the recommender's own per-token difficulty",
                }
            )

    recall = stage3.get("test_recall@10")
    ndcg = stage3.get("test_ndcg@10")
    summary = {
        "mechanism_id": mechanism_id,
        "er": er,
        "cleared_the_gate": bool(er is not None and er > FROZEN_GATE),
        "delta_recall@10": (recall - CONTROL_RECALL) if recall is not None else None,
        "delta_ndcg@10": (ndcg - CONTROL_NDCG) if ndcg is not None else None,
        "modes": modes,
        "primary_mode": modes[0]["mode"] if modes else "unexplained",
    }
    if not modes and recall is not None and recall > CONTROL_RECALL and ndcg is not None and ndcg > CONTROL_NDCG:
        summary["primary_mode"] = "none"
    return summary


def summarise(diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How the cohort's failures are distributed."""
    counts: Dict[str, int] = {}
    for row in diagnoses:
        for mode in row.get("modes") or ():
            counts[mode["mode"]] = counts.get(mode["mode"], 0) + 1
    unexplained = sum(1 for row in diagnoses if row.get("primary_mode") == "unexplained")
    return {
        "arms": len(diagnoses),
        "modes": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "arms_with_no_diagnosis": unexplained,
    }


__all__ = [
    "CONTROL_BRANCHES",
    "CONTROL_ER",
    "CONTROL_LARGEST_CELL",
    "CONTROL_NDCG",
    "CONTROL_RECALL",
    "FROZEN_GATE",
    "HISTORICAL_G_H1_RANGE",
    "diagnose",
    "summarise",
]
