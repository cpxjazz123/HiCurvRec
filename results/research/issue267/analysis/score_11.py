"""Issue #267: append the new arm and test the frozen score's prospective ranking."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[4]
ISSUE = ROOT / "results/research/issue267"
OUT = ISSUE / "analysis"
ARM = "TIGER_relative_curvature_ranking_rq"


def main():
    pred = json.loads((ISSUE / "SCORE_PREDICTION.json").read_text())
    new_final = sorted((ROOT / "results/stage3_T5Train" / ARM).glob("**/test_final.json"))
    if len(new_final) != 1:
        raise RuntimeError("expected one new Stage3 test_final")
    final = json.loads(new_final[0].read_text())
    rows = []
    for item in pred["historical"]:
        rows.append({"index": item["index"], "arm": item["arm"],
                     "recall": item["recall@10"], "S": item["S"],
                     "source": "historical_10"})
    rows.append({"index": 110, "arm": ARM, "recall": float(final["test_recall@10"]),
                 "S": float(pred["prediction"]["S"]), "source": "new"})
    rows.sort(key=lambda row: row["recall"])
    recalls = np.asarray([row["recall"] for row in rows])
    scores = np.asarray([row["S"] for row in rows])
    adjacent = []
    for left, right in zip(rows, rows[1:]):
        adjacent.append({"left": left["index"], "right": right["index"],
                         "left_S": left["S"], "right_S": right["S"],
                         "strict": bool(left["S"] < right["S"])})
    inversions = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if recalls[i] < recalls[j] and scores[i] > scores[j]:
                inversions.append({"lower_recall": rows[i]["index"],
                                   "higher_recall": rows[j]["index"]})
    result = {
        "schema": "genrec.issue267.scores_11/1",
        "rows": rows,
        "predicted": pred["prediction"],
        "actual": {"recall@10": float(final["test_recall@10"]),
                   "recall@5": float(final["test_recall@5"]),
                   "ndcg@10": float(final["test_ndcg@10"]),
                   "rank_by_recall": next(i + 1 for i, row in enumerate(rows)
                                          if row["index"] == 110)},
        "history": {"pearson": float(pearsonr(recalls, scores).statistic),
                    "spearman": float(spearmanr(recalls, scores).statistic),
                    "kendall_tau_b": float(kendalltau(recalls, scores, variant="b").statistic)},
        "adjacent": adjacent,
        "strict_adjacent": all(pair["strict"] for pair in adjacent),
        "inversions": inversions,
        "inversion_count": len(inversions),
        "prediction_interval": {"lower": pred["prediction"]["lower_historical"],
                                 "upper": pred["prediction"]["upper_historical"]},
    }
    (OUT / "SCORES_11.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "SCORES_11.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "arm", "recall@10", "S", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"index": row["index"], "arm": row["arm"],
                             "recall@10": row["recall"], "S": row["S"], "source": row["source"]})
    print(json.dumps({key: result[key] for key in ("history", "strict_adjacent",
                                                   "inversion_count", "actual")}, indent=2))


if __name__ == "__main__":
    main()
