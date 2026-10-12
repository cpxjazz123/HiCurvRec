import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (ARMS, BOOTSTRAP_REPLICATES, OUT_DIR, PAIR_SHA256, PRIMARY_PAIRS,
                    SECONDARY_PAIRS, SEED, TRAIN_PARQUET, ROOT)
from distances import analyze_distances, load_pairs
from neighborhoods import analyze_neighborhoods
from residuals import extract, norm_statistics


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _behavior_modules():
    scripts = ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"
    sys.path.insert(0, str(scripts))
    import analyze_prefix_behavior as prefix_behavior
    import eval_behavior_metrics as behavior_metrics
    return prefix_behavior, behavior_metrics


def _train_statistics(n_items, prefix_behavior):
    frame = pd.read_parquet(TRAIN_PARQUET, columns=["seen_history", "target"])
    counts = frame["target"].value_counts()
    freq = np.zeros(n_items, dtype=np.int64)
    ids = counts.index.to_numpy(dtype=np.int64)
    keep = (ids >= 0) & (ids < n_items)
    freq[ids[keep]] = counts.to_numpy(dtype=np.int64)[keep]
    transitions = prefix_behavior.transitions(frame, n_items)
    successors = {}
    for anchor, positive in transitions:
        successors.setdefault(int(anchor), set()).add(int(positive))
    return freq, successors


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted, residual_arrays = {}, {}
    for model in ARMS:
        extracted[model] = extract(model)
        with np.load(extracted[model]["residual_file"]) as data:
            residual_arrays[model] = data["residuals"]
    norm_rows = []
    for model, values in residual_arrays.items():
        for row in norm_statistics(values):
            norm_rows.append({"model": model, **row})
    _write_csv(OUT_DIR / "residual_norms.csv", norm_rows)
    print("RESIDUAL NORM TABLE")
    print(pd.DataFrame(norm_rows).to_string(index=False))

    pair_specs = (("primary_valid", PRIMARY_PAIRS), ("secondary_train_replication", SECONDARY_PAIRS))
    distance_records, distance_csv = [], []
    neighborhood_records, neighborhood_csv = [], []
    pair_hashes = {}
    prefix_behavior, behavior_metrics = _behavior_modules()
    freq, train_graph = _train_statistics(next(iter(residual_arrays.values())).shape[0], prefix_behavior)
    sid_results = {}
    for data_label, pair_path in pair_specs:
        pair_hashes[data_label] = _sha(pair_path)
        expected = PAIR_SHA256["primary_valid" if data_label == "primary_valid" else "secondary_train"]
        if pair_hashes[data_label] != expected:
            raise ValueError(f"Pair file hash mismatch for {data_label}: {pair_hashes[data_label]}")
        pairs = load_pairs(pair_path)
        sid_results[data_label] = {}
        for model in ARMS:
            sid_levels, sid_load_info = prefix_behavior.load_levels(ARMS[model]["sid_json"])
            sid_results[data_label][model] = {
                "sid_load_info": sid_load_info,
                **behavior_metrics.metrics_for(sid_levels, pairs),
            }
        records, rows = analyze_distances(residual_arrays, pairs, ("euclidean", "rms_normalized"),
                                          SEED, BOOTSTRAP_REPLICATES)
        for record in records:
            record["pair_set"] = data_label
            distance_records.append(record)
        for row in rows:
            row["pair_set"] = data_label
            distance_csv.append(row)
        neighborhood, nrows = analyze_neighborhoods(residual_arrays, pairs, freq, SEED, train_graph)
        for record in neighborhood:
            record["pair_set"] = data_label
            neighborhood_records.append(record)
        for row in nrows:
            row["pair_set"] = data_label
            neighborhood_csv.append(row)

    _write_csv(OUT_DIR / "distances.csv", distance_csv)
    _write_csv(OUT_DIR / "neighborhoods.csv", neighborhood_csv)
    primary_dist = [r for r in distance_records if r["pair_set"] == "primary_valid"]
    print("DISTANCE TABLE (PRIMARY VALID PAIRS)")
    print(pd.DataFrame([{"model": r["model"], "level": r["level"], "variant": r["variant"],
                         "positive_mean": r["positive"]["mean"], "negative_mean": r["negative"]["mean"],
                         "difference_mean": r["difference_negative_minus_positive"]["mean"],
                         "fraction_negative_closer": r["fraction_negative_closer"],
                         "ordering_error_rate": r["ordering_error_rate"],
                         "histogram_overlap": r["histogram_bin_overlap_coefficient"],
                         "overlapping_coefficient": r["overlapping_coefficient"]} for r in primary_dist]).to_string(index=False))
    print("AUC TABLE (MODEL / LEVEL / VARIANT; PRIMARY VALID PAIRS)")
    print(pd.DataFrame([{"model": r["model"], "level": r["level"], "variant": r["variant"],
                         "auc_neg_farther_than_pos": r["auc_neg_farther_than_pos"],
                         "anchor_bootstrap_95ci": r["auc_neg_farther_than_pos_anchor_bootstrap_95ci"]}
                        for r in primary_dist]).to_string(index=False))
    primary_neighborhood = [r for r in neighborhood_records if r["pair_set"] == "primary_valid"]
    print("TOP-K COMPOSITION (PRIMARY VALID PAIRS)")
    composition_rows = []
    for r in primary_neighborhood:
        for k, metrics in r["topk_composition"].items():
            composition_rows.append({"model": r["model"], "level": r["level"], "K": k, **metrics})
    print(pd.DataFrame(composition_rows).to_string(index=False))

    report = {"interpretation": "A smaller residual norm is NOT by itself evidence of crowding.",
              "distance_metric_definitions": {
                  "auc_neg_farther_than_pos": "P(d_neg > d_pos) + 0.5 * P(d_neg == d_pos), within each matched positive and its 10 negatives; high is informative.",
                  "ordering_error_rate": "P(d_neg < d_pos) + 0.5 * P(d_neg == d_pos), exact complement of auc_neg_farther_than_pos; low is informative.",
                  "fraction_negative_closer": "Strict fraction P(d_neg < d_pos), with ties excluded."},
              "residual_extraction": extracted,
              "sid_behavior_metrics_reused": sid_results,
              "pair_inputs": {name: {"path": str(path), "sha256": pair_hashes[name],
                                     "role": "primary held-out valid set" if name == "primary_valid" else "secondary canonical train-derived replication"}
                              for name, path in pair_specs},
              "distance_metrics": distance_records, "neighborhood_metrics": neighborhood_records,
              "undefined_or_under_sampled": _undefined(distance_records, neighborhood_records)}
    json_path = OUT_DIR / "crowding_diagnostic.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _plot(norm_rows, distance_records, OUT_DIR / "crowding_diagnostic.svg")
    print(f"OUTPUTS {json_path} {OUT_DIR / 'residual_norms.csv'} {OUT_DIR / 'distances.csv'} {OUT_DIR / 'neighborhoods.csv'} {OUT_DIR / 'crowding_diagnostic.svg'}")
    print("INPUT SHA256")
    for label, arm in ARMS.items():
        print(f"checkpoint {label} {extracted[label]['checkpoint_sha256']}")
    for name, digest in pair_hashes.items():
        print(f"pairs {name} {digest}")


def _undefined(distances, neighborhoods):
    out = []
    for record in distances:
        if record["auc_neg_farther_than_pos_anchor_bootstrap_95ci"] is None:
            out.append({"model": record["model"], "level": record["level"], "variant": record["variant"],
                        "metric": "AUC bootstrap CI", "status": "undefined", "reason": "fewer than two distinct anchors"})
    for record in neighborhoods:
        if record["n_anchors"] == 0:
            out.append({"model": record["model"], "level": record["level"], "pair_set": record["pair_set"],
                        "metric": "neighborhood metrics", "status": "undefined", "reason": "no anchors with successors"})
        for metric in ("constraint_satisfaction_rate", "mean_margin", "hard_negative_intrusion_rate"):
            if record[metric] is None:
                out.append({"model": record["model"], "level": record["level"], "pair_set": record["pair_set"],
                            "metric": metric, "status": "undefined", "reason": "no eligible anchor observations"})
        for name, value in record["confusability_correlations"].items():
            if value is None:
                out.append({"model": record["model"], "level": record["level"], "pair_set": record["pair_set"],
                            "metric": f"confusability correlation with {name}", "status": "undefined",
                            "reason": "fewer than three finite items or one input is constant"})
        for k, values in record["neighborhood_overlap"].items():
            for metric, value in values.items():
                if value is None:
                    out.append({"model": record["model"], "level": record["level"], "pair_set": record["pair_set"],
                                "metric": f"neighborhood overlap K={k} {metric}", "status": "undefined",
                                "reason": "no eligible related-anchor pairs or popularity-matched null pairs"})
    return out


def _plot(norm_rows, distance_records, path):
    width, height = 1100, 480
    colors = {"Exp1a": "#1769aa", "TIGER": "#c44735"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="275" y="30" text-anchor="middle" font-size="18">Residual norm (descriptive only)</text>',
             '<text x="825" y="30" text-anchor="middle" font-size="18">Held-out valid-pair Euclidean AUC</text>']
    primary = [r for r in distance_records if r["pair_set"] == "primary_valid" and r["variant"] == "euclidean"]
    for panel_x, title in ((70, "norm"), (620, "auc")):
        for j in range(4):
            y = 400 - j * 100
            parts.append(f'<line x1="{panel_x}" y1="{y}" x2="{panel_x+430}" y2="{y}" stroke="#dddddd"/>')
    for model, color in colors.items():
        rows = sorted((r for r in norm_rows if r["model"] == model), key=lambda r: r["level"])
        pts = []
        max_norm = max(r["mean"] for r in norm_rows) or 1
        for i, row in enumerate(rows):
            x, y = 90 + i * 190, 400 - 330 * row["mean"] / max_norm
            pts.append(f"{x},{y}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, row in enumerate(rows):
            x, y = 90 + i * 190, 400 - 330 * row["mean"] / max_norm
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
        arows = sorted((r for r in primary if r["model"] == model), key=lambda r: r["level"])
        pts = []
        for i, row in enumerate(arows):
            x, y = 640 + i * 190, 400 - 330 * row["auc_neg_farther_than_pos"]
            pts.append(f"{x},{y}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, row in enumerate(arows):
            x, y = 640 + i * 190, 400 - 330 * row["auc_neg_farther_than_pos"]
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    for i, level in enumerate(("L1", "L2", "L3")):
        parts.append(f'<text x="{90+i*190}" y="430" text-anchor="middle">{level}</text>')
        parts.append(f'<text x="{640+i*190}" y="430" text-anchor="middle">{level}</text>')
    parts.extend(['<text x="550" y="465" text-anchor="middle" font-size="12">Smaller residual norm is NOT by itself evidence of crowding</text>', '</svg>'])
    path.write_text("\\n".join(parts) + "\\n")


if __name__ == "__main__":
    main()
