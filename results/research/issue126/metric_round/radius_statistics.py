"""The radius statistics against the improved hierarchy reference.

The first version of this analysis used a behaviour hierarchy built over at most
4,000 items, left the rest at a default depth, and compared the radius with the
distance from ONE random point. This version uses the hierarchy that covers every
item (12,000 in the tree, the rest inheriting from their nearest tree item) and
TEN random reference points, and it reports the partial correlation of the radius
with the hierarchy depth for each of them, so the excess of the true origin over
an arbitrary centre can be read off directly.

Run from the group records written by ``hbaa_tightness_swap.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
GROUPS = HERE.parents[0] / "group_records_107.npz"
HIERARCHY = HERE.parents[0] / "behaviour_hierarchy.npz"
OUT = HERE.parents[0] / "radius_statistics_107.json"


def partial_correlation(x, y, controls):
    design = np.column_stack([np.ones(len(x))] + [controls[:, i] for i in range(controls.shape[1])])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    residual_x = x - design @ beta
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual_y = y - design @ beta
    if residual_x.std() < 1e-18 or residual_y.std() < 1e-18:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def main() -> int:
    records = np.load(GROUPS)
    hierarchy = np.load(HIERARCHY)
    depth = hierarchy["depth"]
    height = hierarchy["height"]
    popularity = hierarchy["popularity"]
    table = records["table"]
    code = records["code"]
    radius = records["radius"]
    tightness = records["tightness"]
    log_size = np.log(records["size"].astype(float))
    log_popularity = np.log1p(records["log_popularity"].astype(float))
    random_radii = np.concatenate(
        [records["random_radius_%d" % t] for t in np.unique(table)], axis=1)

    def demeaned(values):
        out = np.empty_like(values, dtype=float)
        for identifier in np.unique(table):
            mask = table == identifier
            out[mask] = values[mask] - values[mask].mean()
        return out

    def item_level(per_item):
        values = np.empty(len(code), dtype=float)
        for identifier in np.unique(table):
            mask = table == identifier
            values[mask] = per_item(code[mask])
        return values

    depth_group = item_level(lambda c: np.asarray([depth[int(x)] for x in c], dtype=float))
    height_group = item_level(lambda c: np.asarray([height[int(x)] for x in c], dtype=float))
    radius_d = demeaned(radius)
    tightness_d = demeaned(tightness)
    size_d = demeaned(log_size)
    popularity_d = demeaned(log_popularity)
    controls = np.column_stack([tightness_d, size_d, popularity_d])

    report = {"groups": int(len(code)), "tables": int(len(np.unique(table)))}
    for name, values in (("depth", depth_group), ("height", height_group)):
        depth_d = demeaned(values)
        entry = {"corr_radius": float(np.corrcoef(radius_d, depth_d)[0, 1]),
                 "partial_radius": partial_correlation(radius_d, depth_d, controls),
                 "partial_tightness": partial_correlation(tightness_d, depth_d, controls),
                 "partial_random_centres": []}
        for index in range(random_radii.shape[0]):
            random_d = demeaned(random_radii[index])
            entry["partial_random_centres"].append(
                partial_correlation(random_d, depth_d, controls))
        entry["partial_random_centres_mean"] = float(np.mean(entry["partial_random_centres"]))
        entry["partial_random_centres_std"] = float(np.std(entry["partial_random_centres"]))
        entry["origin_excess_over_random"] = float(
            entry["partial_radius"] - entry["partial_random_centres_mean"])
        report[name] = entry
    report["corr_radius_vs_tightness"] = float(np.corrcoef(radius_d, tightness_d)[0, 1])
    report["corr_radius_vs_log_size"] = float(np.corrcoef(radius_d, size_d)[0, 1])
    report["corr_radius_vs_log_popularity"] = float(np.corrcoef(radius_d, popularity_d)[0, 1])
    report["radius_vs_random_correlation_mean"] = float(np.mean([
        float(np.corrcoef(radius_d, demeaned(random_radii[index]))[0, 1])
        for index in range(random_radii.shape[0])]))
    report["hierarchy"] = {"tree_items": 12000,
                            "depth_median": float(np.median(depth)),
                            "depth_max": int(depth.max()),
                            "height_median": float(np.median(height))}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
