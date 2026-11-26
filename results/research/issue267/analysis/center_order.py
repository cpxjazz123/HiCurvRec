"""Issue #267: within-centre ordering diagnostic from the full 28M raw table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "results/research/issue267/raw"
OUT = ROOT / "results/research/issue267/analysis"


def main():
    import sys
    tags = sys.argv[1:] or ["109"]
    result = {"schema": "genrec.issue267.center_order/1", "tags": tags, "spaces": {}}
    for tag in tags:
      for space in ("q", "z"):
        with np.load(RAW / f"curvature_{tag}_{space}.npz", allow_pickle=False) as blob:
            target = blob["target"].astype(np.float64)
            predicted = blob["predicted"].astype(np.float64)
            centre = blob["centre"].astype(np.int64)
        order = np.argsort(centre, kind="stable")
        centre_s, target_s, pred_s = centre[order], target[order], predicted[order]
        # Group boundaries without rescanning the full array per centre.
        unique, starts, counts = np.unique(centre_s, return_index=True, return_counts=True)
        values = []
        weights = []
        for start, count in zip(starts, counts):
            if int(count) < 3:
                continue
            t = target_s[start:start + count]
            p = pred_s[start:start + count]
            if np.ptp(t) == 0 or np.ptp(p) == 0:
                continue
            rho = float(spearmanr(t, p).statistic)
            if np.isfinite(rho):
                values.append(rho)
                weights.append(int(count))
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        result["spaces"][f"{tag}:{space}"] = {
            "centres": int(len(values)), "rows": int(weights.sum()),
            "mean_spearman": float(values.mean()) if len(values) else None,
            "median_spearman": float(np.median(values)) if len(values) else None,
            "spearman_quantiles": {str(q): float(x) for q, x in zip(
                (0, .01, .05, .25, .5, .75, .95, .99, 1),
                np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1]))} if len(values) else None,
            "positive_fraction": float((values > 0).mean()) if len(values) else None,
            "negative_fraction": float((values < 0).mean()) if len(values) else None,
            "weighted_mean": float(np.average(values, weights=weights)) if len(values) else None,
        }
    name = "CENTER_ORDER_" + "_".join(tags) + ".json"
    (OUT / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
