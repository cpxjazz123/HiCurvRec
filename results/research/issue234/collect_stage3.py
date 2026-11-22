#!/usr/bin/env python3
"""Collect the Stage3 outcome of A1-A6 into one table.

Reads each arm's ``test_final.json`` (written by the frozen Stage3 protocol) and
the Stage2 technical numbers already measured for the same arm, so the Stage2
trade-off and the Stage3 recall sit on the same row.

Nothing is inferred: an arm without ``test_final.json`` is reported as missing
rather than guessed, per CLAUDE.md 63.

Usage (no arguments):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/issue234/collect_stage3.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ISSUE = REPO / "results" / "research" / "issue234"
ARMS = [f"TIGER_i234_A{n}" for n in range(1, 7)]
LAMBDAS = {"TIGER_i234_A1": 0.002, "TIGER_i234_A2": 0.005, "TIGER_i234_A3": 0.01,
           "TIGER_i234_A4": 0.02, "TIGER_i234_A5": 0.05, "TIGER_i234_A6": 0.10}
WAIVED = {"TIGER_i234_A6"}


def stage2_numbers(arm: str) -> dict:
    path = ISSUE / f"metrics_{arm}_validation.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"collision": data["technical"]["collision"],
            "quant_error": data["technical"].get("quant_error")}


def stage3_result(arm: str) -> dict | None:
    hits = sorted((REPO / "results" / "stage3_T5Train" / arm).glob("**/test_final.json"))
    if not hits:
        return None
    return json.loads(hits[-1].read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for arm in ARMS:
        s2 = stage2_numbers(arm)
        s3 = stage3_result(arm)
        rows.append({
            "arm": arm,
            "lambda_max": LAMBDAS[arm],
            "collision": s2.get("collision"),
            "quant_error": s2.get("quant_error"),
            "collision_gate_waived": arm in WAIVED,
            "test_recall@10": None if s3 is None else s3["test_recall@10"],
            "test_recall@5": None if s3 is None else s3["test_recall@5"],
            "test_ndcg@10": None if s3 is None else s3["test_ndcg@10"],
            "n_eval": None if s3 is None else s3["n_eval"],
            "missing": s3 is None,
        })

    print(f"{'arm':14} {'lambda':>7} {'coll':>7} {'R@10':>9} {'R@5':>9} {'NDCG@10':>9}  waived")
    for row in rows:
        r10 = "missing" if row["missing"] else f"{row['test_recall@10']:.6f}"
        r5 = "-" if row["missing"] else f"{row['test_recall@5']:.6f}"
        nd = "-" if row["missing"] else f"{row['test_ndcg@10']:.6f}"
        print(f"{row['arm']:14} {row['lambda_max']:>7} {row['collision']:>7.4f} "
              f"{r10:>9} {r5:>9} {nd:>9}  {'YES' if row['collision_gate_waived'] else ''}")

    out = ISSUE / "stage3_arm_table.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    missing = [row["arm"] for row in rows if row["missing"]]
    print(f"\nmissing: {missing or 'none'}")
    print(f"table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
