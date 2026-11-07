"""Collect what every arm of issue #139 produced, into one table and one JSON.

Nothing here recomputes a number: every value is read from what the run itself
wrote -- the trainer's metrics file (the epochs, the outer-loop rounds and the
weight table) and the layered SID record the framework produced from the
exported identifiers. The one thing computed here is the arithmetic the arms
have in common: the difference of a metric against the frozen reference arm.

Usage:
    python3 results/research/issue139/collect_arms.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = ("F0", "G", "L", "E", "A")
RESULTS = REPO / "results" / "stage2_RQ-VAE"
TREES = REPO / "stage2_RQ-VAE"
OUT = REPO / "results" / "research" / "issue139"

REFERENCE = {"valid_l1_hard_er": 16.38953942232631,
              "valid_l1_positive_hit_rate": 0.07944376123356352,
              "valid_l1_negative_hit_rate": 0.004847223536089301,
              "level2_distinct_prefixes": 16145}


def read_metrics(arm: str) -> dict:
    path = TREES / ("TIGER_mixed_rq_" + arm) / "logs" / "training_metrics.jsonl"
    if not path.is_file():
        return {}
    events = [json.loads(line) for line in path.open() if line.strip()]
    train = [e for e in events if e.get("event") == "train"]
    end = [e for e in events if e.get("event") == "train_end"]
    rounds = [e for e in events if e.get("event") == "outer_loop"]
    return {"epochs": len(train), "last_epoch": train[-1]["epoch"] if train else 0,
            "epoch_time_s": train[-1].get("epoch_time_s") if train else None,
            "wall_s": end[0].get("total_wall_time_s") if end else None,
            "collision": end[0].get("final_collision") if end else None,
            "raw_unique": end[0].get("final_raw_unique") if end else None,
            "rounds": [{"epoch": r.get("epoch"), "round": r.get("round"),
                        "distortion_start": r.get("distortion_start"),
                        "distortion_end": r.get("distortion_end"),
                        "accepted": len(r.get("accepted") or []),
                        "rejected": len(r.get("rejected") or []),
                        "skipped": len(r.get("skipped") or []),
                        "deferred": len(r.get("deferred") or []),
                        "seconds": r.get("seconds"),
                        "alpha_table": r.get("alpha_table")} for r in rounds]}


def read_sid_metrics(arm: str) -> dict:
    folder = RESULTS / ("TIGER_mixed_rq_" + arm) / "stage2_metrics"
    if not folder.is_dir():
        return {}
    files = sorted(folder.glob("*.json"))
    if not files:
        return {}
    record = json.loads(files[-1].read_text())
    metrics = record.get("metrics") or {}
    return {**{k: metrics.get(k) for k in (
        "valid_l1_hard_er", "valid_l1_positive_hit_rate", "valid_l1_negative_hit_rate",
        "level2_distinct_prefixes", "level3_distinct_triples", "l3_collision",
        "l3_refinement_gain_bits", "conditional_l3_gain")},
        "technically_valid_sid": record.get("technically_valid_sid"),
        "sid_sha256": (record.get("quality_source") or {}).get("sid")}


def main() -> int:
    arms = {}
    for arm in ARMS:
        run = read_metrics(arm)
        sid = read_sid_metrics(arm)
        arms[arm] = {"run": run, "sid": sid}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arms_summary.json").write_text(json.dumps(
        {"reference_resample": REFERENCE, "arms": arms}, indent=2, sort_keys=True))

    header = ("| arm | epochs | wall s | collision | L1 hard ER | L1 pos hit | L1 neg hit "
              "| L2 prefixes | L3 triples | L3 collision | valid SID | rounds (accepted) |")
    lines = [header, "|" + "---|" * 12]
    for arm in ARMS:
        run, sid = arms[arm]["run"], arms[arm]["sid"]
        rounds = run.get("rounds") or []
        summary = ", ".join("%d@%d:%d" % (r["epoch"], r["round"], r["accepted"]) for r in rounds) or "-"
        def value(key, digits=4):
            got = sid.get(key)
            return ("%." + str(digits) + "f") % got if isinstance(got, (int, float)) else "-"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, run.get("last_epoch") or "-",
            ("%.0f" % run["wall_s"]) if run.get("wall_s") else "-",
            value("collision", 4) if run.get("collision") is None else "%.4f" % run["collision"],
            value("valid_l1_hard_er", 3), value("valid_l1_positive_hit_rate", 4),
            value("valid_l1_negative_hit_rate", 4),
            sid.get("level2_distinct_prefixes") or "-",
            sid.get("level3_distinct_triples") or "-",
            value("l3_collision", 4),
            sid.get("technically_valid_sid"),
            summary))
    table = "\n".join(lines)
    (OUT / "arms_table.md").write_text(
        "# Issue #139: the five arms\n\n"
        "The reference row is the frozen resample baseline "
        f"(hard ER {REFERENCE['valid_l1_hard_er']:.2f}, positive hit "
        f"{REFERENCE['valid_l1_positive_hit_rate']:.4f}, "
        f"{REFERENCE['level2_distinct_prefixes']} second-level prefixes).\n\n" + table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
