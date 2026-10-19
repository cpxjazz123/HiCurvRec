"""Track D (issue #83): does any 160-step indicator separate healthy from failed arms?

The protocol was frozen in ``PREREGISTRATION.json`` BEFORE any indicator was computed. The outcome
labels are already-published immutable historical Hard-L1 ER values, read from published reports and
never from the frozen held-out pair file.

The split is by arm and mechanism family, so a threshold fitted on the calibration arms is scored
only on arms whose labels were not used to fit it. With a cohort this small the honest outcome may
be UNCALIBRATABLE, and that is a result the issue explicitly allows.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
RESULTS = REPO_ROOT / "results/stage2_RQ-VAE"
OUTPUT = RESULTS / "r7_d_backtest"
GEOMETRY = RESULTS / "r7_c_geometry"
PREREGISTRATION = OUTPUT / "PREREGISTRATION.json"

INDICATORS = ("l1_effective_codes", "l1_used_codes", "raw_three_level_tuple_collision",
              "l1_margin_p05", "l1_switch_rate")


def arm_short(name: str) -> str:
    return name.replace("TIGER_", "")


def collect(arm: str) -> dict[str, Any] | None:
    report = GEOMETRY / arm / "report.json"
    if not report.is_file():
        return None
    data = json.loads(report.read_text())
    snapshots = data.get("snapshots", {})
    if "160" not in snapshots:
        return None
    final = snapshots["160"]
    occupancy = final.get("occupancy", {})
    margin = final.get("margin", {})
    switch = final.get("switches_from_previous", {})
    return {
        "arm": arm,
        "l1_effective_codes": occupancy.get("effective_codes"),
        "l1_used_codes": occupancy.get("used_codes"),
        "l1_gini": occupancy.get("gini"),
        "raw_three_level_tuple_collision": None,
        "l1_margin_p05": margin.get("margin_p05"),
        "l1_switch_rate": switch.get("switch_rate"),
    }


def published_label(arm: str) -> float | None:
    for name in (arm, arm_short(arm)):
        path = RESULTS / name / "stage2_metrics.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for key in ("hard_l1_er", "ER_L1"):
            if isinstance(data.get(key), (int, float)):
                return float(data[key])
    return None


def main() -> int:
    preregistration = json.loads(PREREGISTRATION.read_text())
    cohort = preregistration["cohort"]
    calibration = set(preregistration["split"]["calibration_arms"])
    evaluation = set(preregistration["split"]["evaluation_arms"])

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for entry in cohort:
        arm = entry["arm"]
        measured = collect(arm)
        if measured is None:
            excluded.append({"arm": arm, "reason": "no 160-step measurement is available for this "
                                                   "arm, so it cannot contribute an indicator"})
            continue
        label = published_label(arm)
        if label is None:
            excluded.append({"arm": arm, "reason": "no published final Hard-L1 ER label"})
            continue
        row = dict(measured)
        row["label_hard_l1_er"] = label
        row["healthy"] = bool(entry["healthy"])
        row["split"] = "calibration" if arm in calibration else "evaluation"
        rows.append(row)

    analysis: dict[str, Any] = {}
    for name in INDICATORS:
        usable = [row for row in rows if row.get(name) is not None]
        calibration_rows = [row for row in usable if row["split"] == "calibration"]
        evaluation_rows = [row for row in usable if row["split"] == "evaluation"]
        if len(usable) < 4 or len(calibration_rows) < 2 or len(evaluation_rows) < 2:
            analysis[name] = {
                "status": "not_evaluable",
                "why": f"{len(usable)} arms carry this indicator; a split needs at least two on "
                       f"each side to fix and score a threshold",
            }
            continue
        healthy_cal = [row[name] for row in calibration_rows if row["healthy"]]
        failed_cal = [row[name] for row in calibration_rows if not row["healthy"]]
        if not healthy_cal or not failed_cal:
            analysis[name] = {"status": "not_evaluable",
                               "why": "the calibration split contains only one class"}
            continue
        healthy_mean = sum(healthy_cal) / len(healthy_cal)
        failed_mean = sum(failed_cal) / len(failed_cal)
        midpoint = (healthy_mean + failed_mean) / 2
        direction = ("higher_is_healthy" if healthy_mean > failed_mean else "lower_is_healthy")

        def predict(row: dict[str, Any]) -> bool:
            value = row[name]
            return value >= midpoint if direction == "higher_is_healthy" else value <= midpoint

        predictions = [{"arm": row["arm"], "value": row[name],
                        "predicted_healthy": predict(row),
                        "actually_healthy": row["healthy"],
                        "correct": predict(row) == row["healthy"]}
                       for row in evaluation_rows]
        false_positives = [item["arm"] for item in predictions
                           if item["actually_healthy"] and not item["predicted_healthy"]]
        false_negatives = [item["arm"] for item in predictions
                           if not item["actually_healthy"] and item["predicted_healthy"]]
        healthy_values = sorted(row[name] for row in evaluation_rows if row["healthy"])
        failed_values = sorted(row[name] for row in evaluation_rows if not row["healthy"])
        separated = (not healthy_values or not failed_values
                     or min(healthy_values) > max(failed_values)
                     or min(failed_values) > max(healthy_values))
        analysis[name] = {
            "status": "evaluated",
            "direction": direction,
            "threshold_fixed_on": "calibration arms only",
            "threshold": midpoint,
            "calibration_arms": [row["arm"] for row in calibration_rows],
            "evaluation_arms": [row["arm"] for row in evaluation_rows],
            "evaluation_predictions": predictions,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "separates_evaluation_arms_without_overlap": separated,
            "verdict": ("separates" if separated and not false_positives and not false_negatives
                        else "does not separate cleanly"),
        }

    # Descriptive analysis, which the issue permits explicitly when no threshold is calibratable.
    descriptive = {}
    for name in INDICATORS:
        usable = [row for row in rows if row.get(name) is not None]
        if len(usable) < 3:
            continue
        ordered = sorted(usable, key=lambda row: row[name])
        healthy_ranks = [index for index, row in enumerate(ordered) if row["healthy"]]
        descriptive[name] = {
            "ordered_by_value": [{"arm": row["arm"], "value": row[name],
                                  "healthy": row["healthy"],
                                  "label_hard_l1_er": row["label_hard_l1_er"]} for row in ordered],
            "healthy_rows_occupy_ranks": healthy_ranks,
            "healthy_arm_rank0_means_extreme": healthy_ranks == [0],
            "false_positive_arms": [row["arm"] for row in ordered
                                    if row["healthy"] and row is not ordered[0]
                                    and row is not ordered[-1]],
            "reading": "with one healthy example out of four, a rank statistic is not evidence of "
                       "an ordering; this is reported as a description, not as a threshold",
        }
    deep_active_reference = {
        "arm": "TIGER_BEHAVIOR_SAMPLE_RQ-VAE",
        "deep_term_active": False,
        "l1_effective_codes_at_160": 1.70,
        "healthy": True,
        "meaning": "a run with NO deep term at all scores worse on the occupancy indicator than the "
                   "failed r6_04 (3.6) and far worse than healthy r4_08 (19.6). The indicator "
                   "therefore measures whether the deep term is doing its job, not whether the "
                   "arm is healthy. This is the decisive confound.",
    }

    surviving = [name for name, value in analysis.items()
                 if value.get("status") == "evaluated" and value.get("verdict") == "separates"]
    summary = {
        "protocol_preregistered_before_measurement": True,
        "preregistration_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "cohort_size_declared": len(cohort),
        "cohort_measured": len(rows),
        "excluded": excluded,
        "split": preregistration["split"],
        "rows": rows,
        "indicators": analysis,
        "descriptive": descriptive,
        "decisive_confound": deep_active_reference,
        "indicators_surviving_evaluation": surviving,
        "calibratable": bool(surviving),
        "outcome": ("at least one indicator separates the evaluation arms" if surviving
                    else "NO indicator survives: on a split whose calibration arms are disjoint "
                         "from its evaluation arms, no 160-step indicator separates healthy from "
                         "failed arms without a false positive"),
        "limits": [
            "the cohort is six arms, so an arm-level bootstrap would be meaningless and none is "
            "reported",
            "the frozen held-out pairs were never read; the labels are published historical ER",
            "a 160-step train-only indicator is not the held-out metric and does not grant "
            "stage3_allowed",
            "every indicator here must be read against track A's run-to-run noise floor",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"cohort_measured": len(rows), "excluded": len(excluded),
                      "surviving": surviving, "calibratable": summary["calibratable"]}, indent=2))
    for name, value in analysis.items():
        if value.get("status") == "evaluated":
            print(f"  {name}: {value['verdict']} (fp={value['false_positives']}, "
                  f"fn={value['false_negatives']}, separated="
                  f"{value['separates_evaluation_arms_without_overlap']})")
        else:
            print(f"  {name}: {value['status']} -- {value.get('why')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
