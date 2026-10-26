"""Correlation between a SID table's *behavioural* metrics and its Stage 3 recall.

Only behaviour-defined metrics are used as predictors. Each one is computed on the
canonical held-out behaviour pairs (52,855 anchor -> positive transitions from
valid.parquet, each with ten fixed negatives drawn from items that are not adjacent
in training), so a value says something about how the SID partitions *behaviour*
and nothing about the size of a bucket:

``p_pos_l1`` / ``p_neg_l1``
    the share of pairs whose positive (negative) shares the anchor's first-level code.
``er_l1``
    ``p_pos / p_neg``: how much more often a behavioural neighbour shares the first
    code than an unrelated item does.
``gap_l1``
    ``p_pos - p_neg``, the same contrast in absolute terms.
``p_pos_l2`` / ``p_neg_l2`` / ``er_l2`` / ``gap_l2``, and the same at depth 3, use the
    two- and three-token prefixes.

The structural counts (how many codes each level uses) are reported in
``sid_structure_correlation.csv`` as controls; they are not behaviour metrics.

The correlation is Spearman (rank), because the metrics are heavy-tailed, with a
pair-level bootstrap interval. A ratio is undefined when no negative shares the
prefix; those rows are dropped from that metric and the count is reported.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap(x, y, *, rounds: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    draws = []
    for _ in range(rounds):
        index = rng.integers(0, len(x), len(x))
        value = spearman(x[index], y[index])
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def behaviour_metrics(sid) -> dict:
    """The behaviour-defined metrics of one SID table."""
    from tools.research.diagnostics import sid_generation_forensics as forensics

    er = forensics.canonical_er(sid)
    structure = forensics.sid_structure(sid)
    out = {}
    for depth in (1, 2, 3):
        level = er["L%d" % depth]
        positive = level.get("positive_hit_rate")
        negative = level.get("negative_hit_rate")
        out["p_pos_l%d" % depth] = positive
        out["p_neg_l%d" % depth] = negative
        out["gap_l%d" % depth] = (positive - negative
                                 if positive is not None and negative is not None else None)
        out["er_l%d" % depth] = level.get("expected_rank_ratio")
    out["L1_distinct"] = structure["levels"]["L1"]["distinct"]
    out["L2_distinct"] = structure["levels"]["L2"]["distinct"]
    out["L3_distinct"] = structure["levels"]["L3"]["distinct"]
    out["ext_distinct"] = structure["collision_extension"]["fourth_token_distinct"]
    return out


BEHAVIOUR = ["p_pos_l1", "p_neg_l1", "er_l1", "gap_l1",
             "p_pos_l2", "p_neg_l2", "er_l2", "gap_l2",
             "p_pos_l3", "p_neg_l3", "er_l3", "gap_l3"]
STRUCTURE = ["L1_distinct", "L2_distinct", "L3_distinct", "ext_distinct"]


def main() -> int:
    from tools.research.diagnostics import sid_generation_forensics as forensics

    pairs = list(csv.DictReader((HERE / "sid_recall.csv").open()))
    rows = []
    for pair in pairs:
        sid = forensics.load_sid(HERE / "sids" / pair["sid_file"])
        metrics = behaviour_metrics(sid)
        metrics.update({"variant": pair["variant"], "sid_sha256": pair["sid_sha256"],
                        "recall@5": float(pair["test_recall@5"]),
                        "recall@10": float(pair["test_recall@10"]),
                        "n_eval": pair["n_eval"],
                        "sid_resolution": pair["sid_resolution"]})
        rows.append(metrics)
    fields = ["variant", "sid_sha256", "recall@5", "recall@10", "n_eval", "sid_resolution",
              *BEHAVIOUR, *STRUCTURE]
    with (HERE / "sid_behaviour_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    recall = [row["recall@10"] for row in rows]
    summary = []
    for name, group in (("behaviour", BEHAVIOUR), ("structure (control)", STRUCTURE)):
        for metric in group:
            values = [row[metric] for row in rows]
            defined = [(value, value_r) for value, value_r in zip(values, recall)
                       if value is not None]
            x = [value for value, _ in defined]
            y = [value_r for _, value_r in defined]
            low, high = bootstrap(x, y) if len(x) >= 3 else (float("nan"), float("nan"))
            summary.append({"group": name, "metric": metric, "n": len(x),
                            "pearson": pearson(x, y), "spearman": spearman(x, y),
                            "spearman_ci95_low": low, "spearman_ci95_high": high})
    with (HERE / "behaviour_correlation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("%-8s %-14s %3s %8s %8s  %s" % ("group", "metric", "n", "pearson", "spearman", "CI95"))
    for row in summary:
        print("%-8s %-14s %3d %8.3f %8.3f  [%6.3f, %6.3f]" % (
            row["group"], row["metric"], row["n"], row["pearson"], row["spearman"],
            row["spearman_ci95_low"], row["spearman_ci95_high"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
