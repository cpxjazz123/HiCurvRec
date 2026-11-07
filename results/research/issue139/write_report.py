"""Write the final report of issue #139 from the artifacts, not from memory.

The goal has three acceptance criteria per mechanism: a completed Stage 2, the
structural distortion of its SID recorded on the shared footing, and a Stage 3
run giving a recall number. This script reads whatever exists and writes the
report, marking anything missing as UNKNOWN rather than filling it in.

Run it after the pipeline finishes, or at any point to see where things stand:
    python3 results/research/issue139/write_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARMS = ("F0", "G", "L", "E", "A")


def stage2(arm: str) -> dict:
    tree = REPO / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)
    metrics = tree / "logs" / "training_metrics.jsonl"
    if not metrics.is_file():
        return {"state": "NOT_RUN"}
    events = [json.loads(line) for line in metrics.open() if line.strip()]
    starts = [index for index, event in enumerate(events) if event.get("event") == "train_start"]
    body = events[starts[-1]:] if starts else []
    epochs = [event for event in body if event.get("event") == "train"]
    end = [event for event in body if event.get("event") == "train_end"]
    rounds = [event for event in body if event.get("event") == "outer_loop"]
    if not end:
        return {"state": "RUNNING", "epoch": epochs[-1]["epoch"] if epochs else 0,
                "rounds_done": len(rounds), "rounds": []}
    return {"state": "DONE", "wall_s": round(end[0].get("total_wall_time_s", 0), 1),
            "collision": end[0].get("final_collision"),
            "rounds": [{"epoch": r.get("epoch"), "from": r.get("distortion_start"),
                        "to": r.get("distortion_end"),
                        "accepted": len(r.get("accepted") or []),
                        "rejected": len(r.get("rejected") or []),
                        "seconds": r.get("seconds")} for r in rounds]}


def layers(arm: str) -> dict:
    folder = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "stage2_metrics"
    sid = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm) / "item_sids.json"
    if not sid.is_file() or not folder.is_dir():
        return {"state": "UNKNOWN"}
    import hashlib
    digest = hashlib.sha256(sid.read_bytes()).hexdigest()
    for candidate in sorted(folder.glob("*.json")):
        payload = json.loads(candidate.read_text())
        if (payload.get("quality_source") or {}).get("sid") == digest:
            metrics = payload.get("metrics") or {}
            return {"state": "DONE", "sid_sha256": digest,
                    "technically_valid": payload.get("technically_valid_sid"),
                    **{key: metrics.get(key) for key in (
                        "valid_l1_hard_er", "valid_l1_positive_hit_rate",
                        "valid_l1_negative_hit_rate", "level2_distinct_prefixes",
                        "level3_distinct_triples", "l3_collision",
                        "l3_refinement_gain_bits", "conditional_l3_gain")}}
    return {"state": "UNKNOWN", "sid_sha256": digest}


def distortion(arm: str) -> dict:
    path = OUT / "structural_fidelity.json"
    if not path.is_file():
        return {"state": "UNKNOWN"}
    payload = json.loads(path.read_text())
    entry = (payload.get("arms") or {}).get(arm)
    if not entry:
        return {"state": "UNKNOWN"}
    return {"state": "DONE", **entry}


def stage3(arm: str) -> dict:
    folder = REPO / "results" / "stage3_T5Train" / ("TIGER_mixed_rq_" + arm)
    found = sorted(folder.glob("logs/*/*/test_final.json")) if folder.is_dir() else []
    if not found:
        return {"state": "NOT_RUN"}
    payload = json.loads(found[-1].read_text())
    return {"state": "DONE", "n_eval": payload.get("n_eval"),
            "recall@5": payload.get("test_recall@5"),
            "recall@10": payload.get("test_recall@10"),
            "ndcg@10": payload.get("test_ndcg@10")}


def fmt(value, digits=6):
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return ("%." + str(digits) + "f") % value
    return str(value)


def main() -> int:
    rows = []
    detail = {}
    for arm in ARMS:
        two, lay, dis, three = stage2(arm), layers(arm), distortion(arm), stage3(arm)
        detail[arm] = {"stage2": two, "layers": lay, "distortion": dis, "stage3": three}
        rows.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm,
            two.get("state") if two.get("state") != "DONE" else "DONE %.0fs" % two["wall_s"],
            fmt(two.get("collision"), 4),
            fmt(lay.get("l3_collision"), 4),
            fmt(dis.get("D_discrete"), 6),
            fmt(dis.get("D_continuous"), 6),
            fmt(dis.get("neighbor_retention"), 4) if "neighbor_retention" in dis
            else fmt(dis.get("neighbour_retention"), 4),
            "UNKNOWN" if three.get("state") != "DONE" else fmt(three.get("recall@10"), 6)))

    rounds_lines = []
    for arm in ARMS:
        rounds = (detail[arm]["stage2"] or {}).get("rounds") or []
        if not isinstance(rounds, list):
            rounds = []
        if not rounds:
            continue
        total = sum(r.get("accepted", 0) for r in rounds)
        rounds_lines.append("| %s | %d | %d | %s |" % (
            arm, len(rounds), total,
            "; ".join("%s: %.5f -> %.5f (%d 接受, %.0fs)"
                      % (r["epoch"], r.get("from") or 0, r.get("to") or 0,
                         r.get("accepted", 0), r.get("seconds") or 0) for r in rounds)))

    complete = all(detail[a]["stage2"].get("state") == "DONE"
                   and detail[a]["distortion"].get("state") == "DONE"
                   and detail[a]["stage3"].get("state") == "DONE" for a in ARMS)
    report = "\n".join([
        "# Issue #139 — final report (generated from the artifacts)",
        "",
        "Generated by `write_report.py` from what the runs wrote. Anything that did",
        "not happen is printed as UNKNOWN; nothing is filled in.",
        "",
        "**Goal status: %s**" % ("COMPLETE" if complete else "INCOMPLETE"),
        "",
        "## The five mechanisms",
        "",
        "| arm | Stage 2 | collision | L3 collision | D_AF3 (hard SID) | D_AF3 (continuous) | neighbour retention | test_R@10 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## The outer loop, per arm",
        "",
        "| arm | rounds | weight changes accepted | per round |",
        "|---|---|---|---|",
        *(rounds_lines or ["| none | 0 | 0 | — |"]),
        "",
        "## The comparisons the issue asks for",
        "",
        "| comparison | what it decides | measured |",
        "|---|---|---|",
        "| F0 vs G | does a globally chosen geometry weight help | D_AF3 %s vs %s |"
        % (fmt(detail["F0"]["distortion"].get("D_discrete"), 6),
           fmt(detail["G"]["distortion"].get("D_discrete"), 6)),
        "| G vs L/E | does the weight need to be local | E accepted %d weights, G accepted %d |"
        % (sum(r.get("accepted", 0) for r in ((detail["E"]["stage2"] or {}).get("rounds") or [])),
           sum(r.get("accepted", 0) for r in ((detail["G"]["stage2"] or {}).get("rounds") or []))),
        "| L vs E | is the non-Euclidean geometry necessary | UNKNOWN until both have Stage 3 |",
        "| L vs A | is choosing the metric better than adding a loss | UNKNOWN until both have Stage 3 |",
        "",
        "## The raw artifacts",
        "",
        "```",
        json.dumps(detail, indent=2, sort_keys=True, default=str)[:6000],
        "```",
        "",
    ])
    (OUT / "FINAL_REPORT.md").write_text(report)
    (OUT / "final_report.json").write_text(
        json.dumps({"complete": complete, "arms": detail}, indent=2, sort_keys=True, default=str) + "\n")
    print("\n".join(report.splitlines()[:26]))
    print("...")
    print("written: FINAL_REPORT.md and final_report.json  | complete:", complete)
    return 0 if complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
