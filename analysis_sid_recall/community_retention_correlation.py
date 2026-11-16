"""Community-internal vs cross-community L1 retention and recall.

Reads the historical `results/community_retention.json` produced by
`community_retention.py` (68 archived arms, 15 Louvain communities, L1 SID retention,
AF3 quartiles).  For each arm it aggregates Q1-Q4 by edge count and correlates:

* inside-community retention and its lift over a shuffled-L1 null;
* between-community retention and its lift;
* the inside-minus-between gap;
* the same quantities per AF3 quartile.

Correlations are Spearman and Pearson, with a permutation p-value, plus partial
correlations controlling the arm's L3 SID collision rate.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/community_retention_correlation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402

INPUT = HERE / "results" / "community_retention.json"
OUT_JSON = HERE / "results" / "community_retention_correlation.json"
OUT_MD = HERE / "results" / "community_retention_correlation.md"
MIRROR = HERE / "r2"
PERMUTATIONS = 5000
QUARTILES = ("Q1", "Q2", "Q3", "Q4")


def weighted(row: dict, group: str, field: str) -> float:
    total = 0
    value = 0.0
    for quartile in QUARTILES:
        cell = row[group][quartile]
        edges = float(cell["edges"])
        total += edges
        value += edges * float(cell[field])
    return value / total if total else float("nan")


def correlation(values: np.ndarray, recall: np.ndarray, collision: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(collision)
    first = values[usable]
    second = recall[usable]
    control = collision[usable]
    if len(first) < 5:
        return {"n": int(len(first)), "spearman": None, "pearson": None}
    observed = {"spearman": analysis.spearman(first, second),
                "pearson": analysis.pearson(first, second)}
    p_values = analysis.permutation_p(first, second, observed, PERMUTATIONS)
    return {
        "n": int(len(first)),
        "spearman": observed["spearman"],
        "pearson": observed["pearson"],
        "p_spearman": p_values["spearman"],
        "p_pearson": p_values["pearson"],
        "partial_spearman": analysis.partial(first, second, control),
        "partial_pearson": analysis.partial(first, second, control, rank=False),
        "mean": float(np.mean(first)),
        "sd": float(np.std(first)),
    }


def main() -> int:
    payload = json.loads(INPUT.read_text())
    rows = payload["arms"]
    collision = []
    for row in rows:
        path = MIRROR / row["number"] / "item_sids.json"
        tokens = analysis.tokens_of(path, 24587) if path.is_file() else None
        collision.append(analysis.collision_of(tokens) if tokens is not None else np.nan)
    collision = np.asarray(collision, dtype=float)
    recall = np.asarray([row["recall@10"] if row["recall@10"] is not None else np.nan
                         for row in rows], dtype=float)

    values = {}
    for group in ("inside", "between"):
        values[f"{group}_retention"] = np.asarray([weighted(row, group, "retention") for row in rows])
        values[f"{group}_null"] = np.asarray([weighted(row, group, "null") for row in rows])
        values[f"{group}_lift"] = values[f"{group}_retention"] - values[f"{group}_null"]
    values["gap_retention"] = values["inside_retention"] - values["between_retention"]
    values["gap_lift"] = values["inside_lift"] - values["between_lift"]

    overall = {name: correlation(value, recall, collision) for name, value in values.items()}

    quartiles = {}
    for group in ("inside", "between"):
        for quartile in QUARTILES:
            for field in ("retention", "lift"):
                key = f"{group}_{quartile}_{field}"
                values[key] = np.asarray([row[group][quartile][field] for row in rows])
                quartiles[key] = correlation(values[key], recall, collision)
    for quartile in QUARTILES:
        key = f"gap_{quartile}_lift"
        values[key] = (
            np.asarray([row["inside"][quartile]["lift"] for row in rows])
            - np.asarray([row["between"][quartile]["lift"] for row in rows])
        )
        quartiles[key] = correlation(values[key], recall, collision)

    result = {
        "input": str(INPUT.relative_to(HERE.parent)),
        "arms_in_file": len(rows),
        "arms_with_recall": int(np.isfinite(recall).sum()),
        "arms_with_collision": int(np.isfinite(collision).sum()),
        "permutations": PERMUTATIONS,
        "communities": payload["summary"]["communities"],
        "modularity": payload["summary"]["modularity"],
        "recall_min": float(np.nanmin(recall)),
        "recall_max": float(np.nanmax(recall)),
        "overall": overall,
        "quartiles": quartiles,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# Community-internal vs cross-community L1 retention and recall",
        "",
        "Source: `analysis_sid_recall/results/community_retention.json` (historical, 68 arms, "
        "15 Louvain communities, L1 SID code retention, AF3 quartiles). "
        "The one arm without `test_recall@10` is excluded; n=67.",
        "",
        "L1 retention is the share of real behaviour edges whose endpoints receive the same first SID "
        "code. `lift` subtracts the same-size shuffled-L1 null. Correlations are across arms; "
        "partial correlations control the arm's L3 SID collision rate.",
        "",
        "## Overall",
        "",
        "| quantity | mean | Spearman | p | Pearson | p | partial Spearman | partial Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in overall.items():
        if value.get("spearman") is None:
            continue
        lines.append(
            f"| {name} | {value['mean']:.4f} | {value['spearman']:+.4f} | {value['p_spearman']:.4f} | "
            f"{value['pearson']:+.4f} | {value['p_pearson']:.4f} | {value['partial_spearman']:+.4f} | "
            f"{value['partial_pearson']:+.4f} |"
        )
    lines += [
        "",
        "## AF3 quartiles: inside and between lift",
        "",
        "| quantity | Spearman | p | Pearson | p | partial Spearman | partial Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, value in quartiles.items():
        if value.get("spearman") is None:
            continue
        lines.append(
            f"| {key} | {value['spearman']:+.4f} | {value['p_spearman']:.4f} | {value['pearson']:+.4f} | "
            f"{value['p_pearson']:.4f} | {value['partial_spearman']:+.4f} | {value['partial_pearson']:+.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
