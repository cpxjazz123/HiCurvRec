"""Collect the numbers the review asked for from one finished Stage2 run.

Writes results/research/issue149/stage2_report.json. The two collision figures
are kept apart on purpose: the entry-gate number is ``1 - distinct triples /
items`` over the first three codes of ``item_sids.json`` (the same formula the
Stage3 entry gate applies), and the §63 ``l3_collision`` is the item-weighted
repeat share from the layer-metrics record. They are reported side by side and
never substituted for one another.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARM = "TIGER_af3_gate_mixed_rq"
RESULT = REPO / "results" / "stage2_RQ-VAE" / ARM
METRICS = REPO / "stage2_RQ-VAE" / ARM / "logs" / "training_metrics.jsonl"
OUT = REPO / "results" / "research" / "issue149" / "stage2_report.json"


def _entry_gate_collision(sid_path: Path) -> dict:
    rows = json.loads(sid_path.read_text(encoding="utf-8"))
    triples = {tuple(row[:3]) for row in rows.values()}
    items = len(rows)
    return {"items": items, "distinct_triples": len(triples),
            "collision": 1.0 - len(triples) / items,
            "formula": "1 - distinct(first three codes) / items"}


def main() -> int:
    events = [json.loads(line) for line in METRICS.read_text(encoding="utf-8").splitlines()]
    train = [event for event in events if event["event"] == "train"]
    structure = [event for event in events if event["event"] == "structure"]
    evals = [event for event in train if event.get("raw_total")]
    start = next(event for event in events if event["event"] == "train_start")
    end = next(event for event in events if event["event"] == "train_end")
    checkpoint = next(event for event in reversed(events) if event["event"] == "ckpt_saved")
    last_structure = structure[-1]
    gradient_probes = [entry for entry in structure
                       if any(key.startswith("dl_dalpha_median_level_") for key in entry)]
    record_dir = RESULT / "stage2_metrics"
    layer_record = None
    if record_dir.is_dir():
        candidates = sorted(record_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if candidates:
            layer_record = json.loads(candidates[-1].read_text(encoding="utf-8"))
    report = {
        "arm": ARM,
        "epochs": int(checkpoint["epoch"]),
        "epoch_time_s_last": train[-1]["epoch_time_s"],
        "total_wall_time_s": end["total_wall_time_s"],
        "final_three_code_collision_trainer": end.get("final_collision"),
        "switches": start["structure_switches"],
        "structure_table_sha256": start["structure_table_sha256"],
        "probe": {"items": last_structure["probe_items"],
                  "fraction_of_trainable": last_structure["probe_fraction"],
                  "triangles": last_structure["triangles_total"],
                  "within_community_edges": last_structure["same_pairs_total"],
                  "cross_community_edges": last_structure["cross_pairs_total"],
                  "coverage_last_epoch": {
                      "triangles": [last_structure["triangles_used"], last_structure["triangles_total"]],
                      "within": [last_structure["same_pairs_used"], last_structure["same_pairs_total"]],
                      "cross": [last_structure["cross_pairs_used"], last_structure["cross_pairs_total"]],
                  }},
        "terms_last_epoch": {name: last_structure[name]
                             for name in ("closure", "bridge", "usage", "total")},
        "levels_last_epoch": last_structure["levels"],
        "dl_dalpha_median_by_eval": {
            str(entry["epoch"]): {key: entry[key] for key in entry
                                  if key.startswith("dl_dalpha_median_level_")}
            for entry in gradient_probes},
        "collision_series": [{"epoch": entry["epoch"], "collision": entry["collision"],
                              "distinct_triples": entry["raw_unique"]} for entry in evals],
        "final_three_code_collision": _entry_gate_collision(RESULT / "item_sids.json"),
        "layer_record": layer_record,
        "items_without_structure": end.get("structure_unobserved"),
        "artifacts": {name: str((RESULT / name).relative_to(REPO))
                      for name in ("item_sids.json", "rqvae_best.pth", "sids_for_hgrec.npy")
                      if (RESULT / name).is_file()},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT),
                      "final_collision": report["final_three_code_collision"],
                      "epochs": report["epochs"],
                      "l3_collision_63": (layer_record or {}).get("metrics", {}).get("l3_collision")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
