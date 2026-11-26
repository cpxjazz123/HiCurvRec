"""Issue #266: assemble the ten-arm frozen score and all ranking statistics."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import reference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MATRIX = ROOT / "results/research/issue262/MATRIX.json"
OUT = HERE
NEW = {108: "TIGER_frozen_topology_gnn_rq",
       109: "TIGER_euclid_local_curve_supervision_rq"}
HISTORICAL_INTERVAL = (0.48640629657478884, 0.5096899206435366)


def load_historical() -> list[dict]:
    matrix = json.loads(MATRIX.read_text())
    rows = []
    for row in matrix["rows"]:
        values = reference.score(row["grad_loss_curvature"], row["grad_curvature_codebook0"])
        rows.append({
            "index": int(row["index"]),
            "arm": row["arm"],
            "recall": float(row["test_recall@10"]),
            "C": float(row["grad_loss_curvature"]),
            "G1": float(row["grad_curvature_codebook0"]),
            "c_component": values["c_component"],
            "g_component": values["g_component"],
            "S": values["S"],
            "source": "results/research/issue262/MATRIX.json",
            "endpoint": "results/research/issue262/QUANT_GRAD_SWITCH_%d.json" % row["index"],
        })
    if len(rows) != 8 or sorted(row["index"] for row in rows) != list(range(100, 108)):
        raise RuntimeError("expected exactly the eight historical arms 100..107")
    return rows


def load_new(index: int) -> dict:
    path = OUT / ("ENDPOINT_%d.json" % index)
    if not path.is_file():
        raise FileNotFoundError(path)
    endpoint = json.loads(path.read_text())
    if endpoint.get("index") != index or endpoint.get("arm") != NEW[index]:
        raise ValueError("endpoint identity mismatch for %d" % index)
    measurement = endpoint["measurement"]
    c = float(measurement["C"])
    g1 = float(measurement["G1"])
    values = reference.score(c, g1)
    return {
        "index": index,
        "arm": NEW[index],
        "recall": float(endpoint["identity"]["recall_test_recall@10"]),
        "C": c,
        "G1": g1,
        "c_component": values["c_component"],
        "g_component": values["g_component"],
        "S": values["S"],
        "source": str(path.relative_to(ROOT)),
        "endpoint": str(path.relative_to(ROOT)),
        "endpoint_sha256": endpoint["identity"]["final_sha256"],
        "checkpoint_sha256": endpoint["identity"]["checkpoint_sha256"],
        "sid_sha256": endpoint["identity"]["sid_sha256"],
        "structural_sample": endpoint["structural_sample"],
        "fixed_batch": endpoint["fixed_batch"],
    }


def main() -> None:
    historical = load_historical()
    new = [load_new(108), load_new(109)]
    rows = sorted(historical + new, key=lambda row: row["index"])
    eight = [row for row in rows if row["index"] < 108]
    ten = rows
    stats8 = reference.summary(eight)
    stats10 = reference.summary(ten)
    by_index = {row["index"]: row for row in rows}
    ordered = reference.rank_order(rows)
    positions = {row["index"]: position + 1 for position, row in enumerate(ordered)}
    lo, hi = HISTORICAL_INTERVAL
    new_report = {
        "historical_interval": [lo, hi],
        "historical_interval_arms": {"lower": 105, "upper": 103},
        "S_108": by_index[108]["S"],
        "S_109": by_index[109]["S"],
        "both_inside_interval": bool(lo < by_index[108]["S"] < hi and lo < by_index[109]["S"] < hi),
        "strict_108_lt_109": bool(by_index[108]["S"] < by_index[109]["S"]),
        "required_chain": bool(lo < by_index[108]["S"] < by_index[109]["S"] < hi),
        "positions_by_recall": {str(index): positions[index] for index in (108, 109)},
        "S_outside_historical_range": {str(index): bool(row["S"] < 0 or row["S"] > 1)
                                       for index, row in ((108, by_index[108]), (109, by_index[109]))},
    }
    payload = {
        "schema": "genrec.issue266.scores/1",
        "issue": 266,
        "score_formula": {
            "S": "0.8*(C-loC)/(hiC-loC)+0.2*(G1-loG)/(hiG-loG)",
            "loC": reference.LO_C, "hiC": reference.HI_C,
            "loG": reference.LO_G, "hiG": reference.HI_G,
            "weight_C": reference.WEIGHT_C, "weight_G": reference.WEIGHT_G,
            "clipped": False, "refit": False,
        },
        "rows": rows,
        "original_eight": stats8,
        "extended_ten": stats10,
        "new_arms": new_report,
        "full_recall_order": [row["index"] for row in ordered],
        "all_rows_retained": True,
        "notes": [
            "Historical C/G1 are reused from issue262's MATRIX, not measured again.",
            "The eight historical arms already had observed test recall; n=8 and n=10 are descriptive.",
            "No arm is dropped and no score is clipped or refit.",
        ],
    }
    (OUT / "SCORES.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["index", "arm", "recall@10", "C", "G1", "c_component", "g_component", "S", "source"]
    with (OUT / "SCORES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({("recall@10" if key == "recall@10" else key):
                             (row["recall"] if key == "recall@10" else row[key])
                             for key in fields})
    print(json.dumps({"rows": len(rows), "S_108": new_report["S_108"],
                      "S_109": new_report["S_109"],
                      "original8": {key: stats8[key] for key in ("pearson", "spearman", "kendall_tau_b")},
                      "extended10": {key: stats10[key] for key in ("pearson", "spearman", "kendall_tau_b")},
                      "adjacent_8": stats8["adjacent_monotonic"]["all_strict"],
                      "adjacent_10": stats10["adjacent_monotonic"]["all_strict"],
                      "inversions_10": stats10["inversions"]["count"],
                      "required_chain": new_report["required_chain"]}, indent=2))


if __name__ == "__main__":
    main()
