#!/usr/bin/env python3
"""Judge a finished arm against the pre-registered criteria in ``verdict.json``.

The slope of the fork-on-chain relation is refitted from the frozen 31-arm survey and must
agree with the frozen value; a drift means the survey or the fit changed and the verdict
must not be trusted.  Nothing here selects numbers after the fact: the criteria were written
before the three-structure run finished.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SURVEY = HERE / "arm_alignment_survey.json"
RULES = HERE / "verdict.json"


def fit_relation() -> dict:
    rows = json.loads(SURVEY.read_text())
    rows = [row for row in rows if "chain" in row and "fork" in row]
    chain = np.array([row["chain"] for row in rows])
    fork = np.array([row["fork"] for row in rows])
    design = np.vstack([chain, np.ones_like(chain)]).T
    slope, intercept = np.linalg.lstsq(design, fork, rcond=None)[0]
    residual = fork - (slope * chain + intercept)
    return {"slope": float(slope), "intercept": float(intercept),
            "r": float(np.corrcoef(chain, fork)[0, 1]),
            "residual_sd": float(residual.std()), "n_arms": len(rows)}


def residual(entry: dict, relation: dict) -> float:
    return entry["fork"]["delta"] - (relation["slope"] * entry["chain"]["delta"]
                                     + relation["intercept"])


def judge(arm: str, report: dict, relation: dict, rules: dict) -> list[tuple[str, bool, str]]:
    entry = report[arm]
    base = report["baseline"]
    chain_gain = residual(entry, relation) - residual(base, relation)
    benchmarks = rules["benchmarks"]
    checks = [
        ("chain_not_below_baseline",
         entry["chain"]["delta"] >= base["chain"]["delta"],
         f"{entry['chain']['delta']:+.5f} vs baseline {base['chain']['delta']:+.5f}"),
        ("merge_not_below_baseline",
         entry["merge"]["delta"] >= base["merge"]["delta"],
         f"{entry['merge']['delta']:+.5f} vs baseline {base['merge']['delta']:+.5f}"),
        ("fork_residual_gain_beats_chain_arm",
         chain_gain >= benchmarks["chain_arm_residual_gain_checkpoint_bound"] - 1e-9,
         f"gain {chain_gain:+.5f} vs benchmark "
         f"{benchmarks['chain_arm_residual_gain_checkpoint_bound']:+.5f} (checkpoint-bound; the"
         f" pre-rewrite value {benchmarks['chain_arm_residual_gain']:+.5f} is kept as a diagnostic)"),
        ("collision_within_ceiling",
         entry["collision"] <= benchmarks["collision_ceiling"],
         f"{entry['collision']:.4f} vs ceiling {benchmarks['collision_ceiling']:.4f}"),
        ("semantic_r2_kept",
         entry["semantic_r2"] >= 0.95 * base["semantic_r2"],
         f"{entry['semantic_r2']:.4f} vs 95% of baseline {0.95 * base['semantic_r2']:.4f}"),
    ]
    return checks


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "TIGER_graphlet_align_rq"
    rules = json.loads(RULES.read_text())
    report_path = HERE / f"arm_alignment_{arm}.json"
    if not report_path.is_file():
        print(f"missing {report_path}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    relation = fit_relation()
    frozen = rules["relation"]
    for key in ("slope", "intercept"):
        if abs(relation[key] - frozen[key]) > 1e-4:
            print(f"relation drift on {key}: measured {relation[key]:.6f} "
                  f"frozen {frozen[key]:.6f}", file=sys.stderr)
            return 2
    checks = judge(arm, report, relation, rules)
    failures = 0
    for name, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        failures += 0 if passed else 1
        print(f"{mark}  {name:36s} {detail}")
    gain = residual(report[arm], relation) - residual(report["baseline"], relation)
    print(f"residual(arm) {residual(report[arm], relation):+.5f}  "
          f"residual(baseline) {residual(report['baseline'], relation):+.5f}  gain {gain:+.5f}")
    print("VERDICT", "PASS" if failures == 0 else f"FAIL ({failures} criteria)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
