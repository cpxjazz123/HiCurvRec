"""Does the existing SID partition already follow the AF3 preference?

Pre-training diagnostic for the `curvature_prefix_ranking` arm: on the frozen
TRAIN graph, every anchor with a cohesive (high-AF3) and a bridge (low-AF3)
neighbour is classified by whether each neighbour shares the anchor's hard
prefix at L1/L2/L3.  If the existing SIDs already give the cohesive neighbour a
higher share rate, the new loss reinforces an existing ordering; if not, the loss
has real work to do.  No model, no training, no Stage3 outcome is read.

Outputs: `pressure_by_level.csv`, `pressure.json`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO),
             str(REPO / "stage2_RQ-VAE" / "TIGER_curvature-prefix-ranking"),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "full_census_geometry"),
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import curvature_prefix  # noqa: E402
import features_lw  # noqa: E402
import train_rqvae  # noqa: E402


def main() -> int:
    pairs = [pair for pair in features_lw.load_pairs() if pair["tier"] != "HASH_MISMATCH"]
    nodes, index, _, _ = features_lw.behaviour_graph()
    n_items = int(nodes.size if hasattr(nodes, "size") else len(nodes))
    frame = __import__("pandas").read_parquet(train_rqvae.TRAIN_FILE)
    table = curvature_prefix.build_edge_tables(
        train_rqvae._training_edges(frame, n_items), n_items)
    lookup = curvature_prefix.TripleLookup(table)
    triples = lookup.triples(np.arange(n_items), use_behaviour=True)
    print(json.dumps({"edges": int(table["af3"].size),
                      "median": table["median"], "scale": table["scale"],
                      "anchors_with_both": int(triples["anchor"].size)}), flush=True)
    rows = []
    for number, pair in enumerate(pairs):
        sids = json.loads(pair["payload"])
        prefix = {level: {} for level in (1, 2, 3)}
        for item, code in sids.items():
            for level in (1, 2, 3):
                prefix[level][int(item)] = tuple(int(token) for token in code[:level])
        for level in (1, 2, 3):
            sharing = {"high": 0, "low": 0, "both": 0}
            for anchor, high, low in zip(triples["anchor"], triples["high"],
                                         triples["low"]):
                anchor_prefix = prefix[level].get(int(anchor))
                high_prefix = prefix[level].get(int(high))
                low_prefix = prefix[level].get(int(low))
                if anchor_prefix is None or high_prefix is None or low_prefix is None:
                    continue
                sharing["high"] += int(anchor_prefix == high_prefix)
                sharing["low"] += int(anchor_prefix == low_prefix)
                sharing["both"] += 1
            if not sharing["both"]:
                rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                             "series": pair["series"], "level": level, "anchors": 0,
                             "high_share": None, "low_share": None, "difference": None})
                continue
            high_rate = sharing["high"] / sharing["both"]
            low_rate = sharing["low"] / sharing["both"]
            rows.append({"sid_sha256": pair["sid_sha256"], "variant": pair["variant"],
                         "series": pair["series"], "level": level,
                         "anchors": sharing["both"], "high_share": high_rate,
                         "low_share": low_rate, "difference": high_rate - low_rate})
        if (number + 1) % 20 == 0:
            print(f"  {number + 1}/{len(pairs)}", flush=True)
    fields = ["sid_sha256", "variant", "series", "level", "anchors", "high_share",
              "low_share", "difference"]
    with (HERE / "pressure_by_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = {}
    for level in (1, 2, 3):
        values = [row["difference"] for row in rows
                  if row["level"] == level and row["difference"] is not None]
        high = [row["high_share"] for row in rows
                if row["level"] == level and row["high_share"] is not None]
        low = [row["low_share"] for row in rows
               if row["level"] == level and row["low_share"] is not None]
        summary[f"L{level}"] = {
            "arms": len(values), "mean_high": float(np.mean(high)) if high else None,
            "mean_low": float(np.mean(low)) if low else None,
            "mean_difference": float(np.mean(values)) if values else None,
            "positive_fraction": float(np.mean([value > 0 for value in values]))
            if values else None}
    (HERE / "pressure.json").write_text(json.dumps(
        {"edges": int(table["af3"].size), "median": float(table["median"]),
         "scale": float(table["scale"]), "anchors": int(triples["anchor"].size),
         "summary": summary}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
