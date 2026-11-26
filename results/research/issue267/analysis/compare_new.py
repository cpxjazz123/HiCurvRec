"""Compare the new arm's full q/z curvature metrics with the #265 parent."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AN = ROOT / "results/research/issue267/analysis"
TAG = "TIGER_relative_curvature_ranking_rq"


def load(tag, space):
    path = AN / f"CURVATURE_FULL_{tag}_{space.upper()}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def summary(data):
    if data is None:
        return None
    m = data["metrics"]
    return {key: m[key] for key in ("peer", "pearson", "spearman", "rmse", "mae",
                                    "target_mean", "target_variance", "prediction_mean",
                                    "prediction_variance", "sign") if key in m}


def main():
    result = {"schema": "genrec.issue267.full_comparison/1", "parent": "109", "new": TAG,
              "spaces": {}}
    for space in ("q", "z"):
        parent = load("109", space)
        new = load(TAG, space)
        result["spaces"][space] = {"parent": summary(parent), "new": summary(new)}
        if parent and new:
            result["spaces"][space]["delta"] = {
                key: new["metrics"][key] - parent["metrics"][key]
                for key in ("pearson", "spearman", "rmse", "mae",
                            "prediction_mean", "prediction_variance")}
    (AN / "FULL_COMPARISON_109_NEW.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
