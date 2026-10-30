#!/usr/bin/env python3
"""The metric ledger: every proposal, what it measured, and what it refuted.

The metric rounds are cumulative only if the refuted constructions are written
down, otherwise the next round proposes them again. This tool builds and
refreshes the ledger from the artefacts on disk -- the proposals of every round
and the measurement of every implemented metric -- so that:

* the ledger lists each construction once, with its round, its measurement and
  its verdict;
* the prompt of the next round carries the list of what has already been
  refuted, and the agents are asked for something whose construction differs.

    python3 tools/research/science/metric_ledger.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROUND_ROOT = REPO / "results" / "research" / "issue126" / "metric_round"
LEDGER = ROUND_ROOT / "ledger.json"

#: The families the refuted constructions fall into. A proposal that lands in one
#: of them has to say what its construction does differently.
FAMILIES = {
    "partition_geometry": ("the table's own geometry: tightness, margin, dispersion, "
                            "coverage, reach, ranking of cells -- measured, and none of "
                            "it tracks the recall"),
    "rewarding_an_answer_in_hand": ("metrics that grow when the answer is already "
                                    "inside the cell: the path likelihood, the "
                                    "generator-side hit rate -- they are NEGATIVELY "
                                    "correlated"),
    "behaviour_contrast": ("the family that works: the table is contrasted against "
                            "the behaviour reference with explicit positives and "
                            "negatives, on the level-1 prefix centroid"),
    "structural_alignment": ("label-free structure comparisons: does the hyperbolic "
                              "shared depth of a pair agree with the length of their "
                              "shared SID prefix, does the prefix tree preserve the "
                              "reference's distances, is a cell's angular cone tighter "
                              "than a different set of the same size -- measured with "
                              "the corrected definitions, and none of them is eligible"),
}


def collect_proposals() -> list:
    proposals = []
    for directory in sorted(ROUND_ROOT.glob("round_*")):
        path = directory / "metric_ideas.json"
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        for metric in record.get("metrics", []):
            proposals.append({
                "metric_id": metric.get("metric_id"),
                "title": metric.get("title"),
                "round": directory.name,
                "proposed_by": metric.get("proposed_by"),
                "formula": (metric.get("formula") or "")[:600],
            })
    return proposals


def collect_measurements() -> dict:
    measurements = {}
    for path in sorted((REPO / "results" / "research" / "issue126").glob("metric_bench*.json")):
        try:
            report = json.loads(path.read_text())
        except ValueError:
            continue
        for result in report.get("results", []):
            name = result["metric"]
            entry = {
                "spearman": result["spearman"],
                "p": result["p"],
                "ci95": result["bootstrap_ci95"],
                "eligible": result.get("eligible"),
                "pairs": report.get("pairs"),
                "report": str(path.relative_to(REPO)),
            }
            # the largest, most recent measurement wins
            if name not in measurements or (entry["pairs"] or 0) >= (measurements[name]["pairs"] or 0):
                measurements[name] = entry
    return measurements


def family_of(metric_id: str) -> str:
    name = metric_id.lower()
    # the "answer already in hand" family is checked FIRST: its members are named
    # after the contrast family (hbaa_...) but they belong to the other one.
    if any(word in name for word in ("hsoftmax_path", "emission", "hit10")):
        return "rewarding_an_answer_in_hand"
    if any(word in name for word in ("hbaa", "q_h", "hyp_cell", "hyp_hist", "hgca",
                                     "hyp_erank", "horo", "hfsc", "hyp_xview")):
        return "behaviour_contrast"
    if any(word in name for word in ("prefix_alignment", "tree_distortion",
                                     "cone_consistency")):
        return "structural_alignment"
    if any(word in name for word in ("voronoi", "adj_r2", "cover_radius", "reach_lift",
                                     "mass_rank", "rank_lift", "levels", "weighted")):
        return "partition_geometry"
    return "behaviour_contrast"


def build() -> dict:
    proposals = collect_proposals()
    measurements = collect_measurements()
    rows = []
    for proposal in proposals:
        name = proposal["metric_id"]
        measurement = measurements.get(name)
        rows.append({
            **proposal,
            "measured": measurement is not None,
            "measurement": measurement,
            "family": family_of(name or ""),
        })
    for name, measurement in measurements.items():
        if not any(row["metric_id"] == name for row in rows):
            rows.append({"metric_id": name, "title": None, "round": "(measured only)",
                         "proposed_by": None, "formula": "", "measured": True,
                         "measurement": measurement, "family": family_of(name)})
    ledger = {
        "purpose": ("every hyperbolic SID metric the rounds proposed, what it "
                    "measured on the confirmed pairs, and what it refuted; the next "
                    "round's prompt carries the refuted list"),
        "families": FAMILIES,
        "proposals": len(rows),
        "measured": sum(1 for row in rows if row["measured"]),
        "eligible": sum(1 for row in rows if (row.get("measurement") or {}).get("eligible")),
        "rows": sorted(rows, key=lambda row: -((row.get("measurement") or {}).get("spearman")
                                                or -1.0)),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def refuted_summary(ledger: dict) -> str:
    lines = []
    for row in ledger["rows"]:
        measurement = row.get("measurement")
        if not measurement or measurement.get("eligible"):
            continue
        lines.append("  - %s (round %s, family %s): Spearman %+0.3f"
                     % (row["metric_id"], row["round"], row["family"],
                        measurement["spearman"]))
    return "\n".join(lines)


def main() -> int:
    ledger = build()
    print(f"proposals: {ledger['proposals']} | measured: {ledger['measured']} | "
          f"eligible: {ledger['eligible']}")
    print(f"\nrefuted constructions:\n{refuted_summary(ledger)}")
    print(f"\nwritten: {LEDGER.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
