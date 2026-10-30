"""Is the radial information hierarchy, or is it group compactness?

HBAA reads the radius of a prefix centroid as depth. The frozen reference puts
almost every item on a thin shell (radius p10 0.9050, median 0.9115, p90
0.9176), so the radius of a group's centroid could be measuring how tight the
group is rather than how deep it sits.

For every level-1 prefix group C_k of every one of the 107 tables this records

* ``r``    the hyperbolic distance from the origin to the group's Frechet mean,
* ``D``    the mean squared distance of the members to that mean (the dispersion),
* the group size, the members' popularity in the training interactions,
* the members' depth in an INDEPENDENT hierarchy: the Amazon category path,
* the same ``r`` measured from a RANDOM point of the ball (the control).

The statistics are computed within table (the tables differ by an arbitrary
offset) and then pooled:

* the raw correlation of ``r`` with the dispersion, the size and the popularity;
* the PARTIAL correlation of ``r`` with the reference depth, controlling for the
  size, the popularity and the dispersion -- the number that decides whether the
  radius says anything about the hierarchy once the alternatives are excluded;
* the same partial correlation with the random centre, which must vanish if the
  origin carries the meaning.

A third reference is built from the training interactions themselves (an
agglomerative clustering of the item co-occurrence profiles), so the conclusion
does not rest on one taxonomy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "radius_vs_hierarchy_107.json"
PARQUET = REPO / "results" / "stage0_build_parquet"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def category_depth() -> tuple:
    """The depth of every item in the Amazon category path, and its popularity."""
    import pandas as pd

    items = pd.read_parquet(PARQUET / "items.parquet")
    depth = {}
    for item_id, categories in zip(items["item_id"].tolist(), items["categories"].tolist()):
        if isinstance(categories, (list, tuple)):
            depth[int(item_id)] = len(categories)
        elif isinstance(categories, str) and categories:
            depth[int(item_id)] = len(categories.split(","))
        else:
            depth[int(item_id)] = 0
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    counts = np.zeros(30000, dtype=np.int64)
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        for item in history:
            if 0 <= int(item) < len(counts):
                counts[int(item)] += 1
        if target is not None and 0 <= int(target) < len(counts):
            counts[int(target)] += 1
    return depth, counts


def cooccurrence_depth() -> dict:
    """A behaviour hierarchy: the depth of every item in a clustering of the
    item co-occurrence profiles built from the TRAINING interactions."""
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    n_items = 30000
    rows = []
    columns = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = list(history) + ([int(target)] if target is not None else [])
        for first in sequence:
            for second in sequence:
                if first != second:
                    rows.append(first)
                    columns.append(second)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = np.zeros((n_items, n_items), dtype=np.float32)
    np.add.at(matrix, (np.asarray(rows), np.asarray(columns)), data)
    np.fill_diagonal(matrix, 0.0)
    norms = np.sqrt(matrix.sum(axis=1))
    keep = np.flatnonzero(norms > 0)
    if len(keep) > 4000:
        step = len(keep) // 4000 + 1
        keep = keep[::step][:4000]
    profile = matrix[keep] / np.maximum(norms[keep], 1e-9)[:, None]
    similarity = profile @ profile.T
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    # the depth of a leaf in the TREE, and the height at which it joins: the
    # cluster LABEL is an unordered integer and is not a depth at all
    n_leaves = len(keep)
    children = {}
    for step, (left, right) in enumerate(tree[:, :2]):
        children[n_leaves + step] = (int(left), int(right))
    depths = {}
    heights = {}

    def walk(node: int, level: int) -> None:
        if node < n_leaves:
            depths[node] = level
            # the height was set by the parent, which knows the merge distance
            heights.setdefault(node, 0.0)
            return
        left, right = children[node]
        step = node - n_leaves
        if left < n_leaves:
            heights[left] = float(tree[step, 2])
        if right < n_leaves:
            heights[right] = float(tree[step, 2])
        walk(left, level + 1)
        walk(right, level + 1)

    walk(2 * n_leaves - 2, 0)
    depth = {}
    for position, item in enumerate(keep):
        depth[int(item)] = int(depths.get(position, 0))
    height = {}
    for position, item in enumerate(keep):
        height[int(item)] = float(heights.get(position, 0.0))
    return depth, height


def work(index: int):
    from tools.research.science import sid_metric_bench as bench

    module = _load("hbaa_source", HERE / "metrics_hbaa.py")
    context = bench.load_context()
    entry = context["pairs"][index]
    table = entry["table"]
    target = np.asarray(context["target"], dtype=np.float64)[np.asarray(table["item_ids"])]
    codes = np.asarray(table["rows"], dtype=np.int64)[:, 0]
    category, popularity = category_depth()
    if not any(category.values()):
        category = {}
    behaviour, behaviour_height = cooccurrence_depth()

    rng = np.random.default_rng(1000 + index)
    direction = rng.normal(size=target.shape[1])
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    random_centre = direction * float(np.median(np.linalg.norm(target, axis=1)))

    records = []
    for code in np.unique(codes):
        members = target[codes == code]
        if len(members) < 4:
            continue
        centroid = module._frechet(members, 1.0)
        radius = float(module._norm(centroid[None, :], 1.0)[0])
        distance = module._distance(members, centroid[None, :], 1.0)
        dispersion = float(np.mean(distance ** 2))
        random_radius = float(module._distance(centroid[None, :], random_centre[None, :], 1.0)[0])
        items = np.asarray(table["item_ids"], dtype=np.int64)[codes == code]
        depths = [category.get(int(item), 0) for item in items]
        behaviour_depths = [behaviour.get(int(item), 0) for item in items]
        behaviour_heights = [behaviour_height.get(int(item), 0.0) for item in items]
        popularity_values = [popularity[int(item)] if int(item) < len(popularity) else 0
                             for item in items]
        records.append({
            "table": index,
            "code": int(code),
            "radius": radius,
            "random_radius": random_radius,
            "dispersion": dispersion,
            "size": int(len(items)),
            "log_size": float(np.log(len(items))),
            "log_popularity": float(np.log1p(np.mean(popularity_values))),
            "category_depth": float(np.mean(depths)),
            "behaviour_depth": float(np.mean(behaviour_depths)),
            "behaviour_height": float(np.mean(behaviour_heights)),
        })
    return index, records


def partial_correlation(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """The correlation of x and y after both are regressed on the controls."""
    design = np.column_stack([np.ones(len(x))] + [controls[:, i] for i in range(controls.shape[1])])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    residual_x = x - design @ beta
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual_y = y - design @ beta
    if residual_x.std() < 1e-18 or residual_y.std() < 1e-18:
        return 0.0
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def main() -> int:
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for row in pool.map(work, range(107))]
    rows.sort()
    records = [record for _, group in rows for record in group]
    report = {"groups": len(records), "tables": len(rows), "statistics": {}}

    def column(name):
        return np.asarray([record[name] for record in records], dtype=float)

    def demeaned(name):
        values = column(name)
        table_ids = column("table")
        out = np.empty_like(values)
        for table_id in np.unique(table_ids):
            mask = table_ids == table_id
            out[mask] = values[mask] - values[mask].mean()
        return out

    radius = demeaned("radius")
    random_radius = demeaned("random_radius")
    dispersion = demeaned("dispersion")
    log_size = demeaned("log_size")
    log_popularity = demeaned("log_popularity")
    statistics = {}
    for reference in ("category_depth", "behaviour_depth", "behaviour_height"):
        depth = demeaned(reference)
        statistics[reference] = {
            "corr_radius_vs_depth": float(np.corrcoef(radius, depth)[0, 1]),
            "corr_dispersion_vs_depth": float(np.corrcoef(dispersion, depth)[0, 1]),
            "partial_corr_radius_depth": partial_correlation(
                radius, depth, np.column_stack([dispersion, log_size, log_popularity])),
            "partial_corr_random_radius_depth": partial_correlation(
                random_radius, depth, np.column_stack([dispersion, log_size, log_popularity])),
        }
    table_ids = column("table")
    unique_tables = np.unique(table_ids)
    groups = {int(t): np.flatnonzero(table_ids == t) for t in unique_tables}

    def statistics_for(index):
        rows = groups[int(unique_tables[index])]
        r = radius[rows]
        d = dispersion[rows]
        ls = log_size[rows]
        lp = log_popularity[rows]
        controls = np.column_stack([d, ls, lp])
        out = {}
        for reference in ("category_depth", "behaviour_depth", "behaviour_height"):
            depth = demeaned(reference)[rows]
            if np.std(depth) < 1e-12 or np.std(r) < 1e-12:
                out[reference] = (float("nan"), float("nan"))
                continue
            out[reference] = (partial_correlation(r, depth, controls),
                              partial_correlation(random_radius[rows], depth, controls))
        return out

    rng = np.random.default_rng(11)
    bootstrap = {"category_depth": [], "behaviour_depth": [], "behaviour_height": []}
    for _ in range(400):
        pick = rng.choice(len(unique_tables), len(unique_tables), replace=True)
        rows = np.concatenate([groups[int(unique_tables[i])] for i in pick])
        r = radius[rows]
        d = dispersion[rows]
        ls = log_size[rows]
        lp = log_popularity[rows]
        controls = np.column_stack([d, ls, lp])
        for reference in bootstrap:
            depth = demeaned(reference)[rows]
            if np.std(depth) < 1e-12 or np.std(r) < 1e-12:
                continue
            bootstrap[reference].append(partial_correlation(r, depth, controls))
    for reference, values in bootstrap.items():
        if values:
            array = np.asarray(values)
            statistics.setdefault("bootstrap_" + reference, {})["ci95"] = [
                float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]
            statistics["bootstrap_" + reference]["p_two_sided"] = float(
                2 * min((array <= 0).mean(), (array >= 0).mean()))
    statistics["corr_radius_vs_dispersion"] = float(np.corrcoef(radius, dispersion)[0, 1])
    statistics["corr_radius_vs_log_size"] = float(np.corrcoef(radius, log_size)[0, 1])
    statistics["corr_radius_vs_log_popularity"] = float(np.corrcoef(radius, log_popularity)[0, 1])
    statistics["corr_radius_vs_random_radius"] = float(np.corrcoef(radius, random_radius)[0, 1])
    statistics["radius_spread"] = {"p10": float(np.percentile(column("radius"), 10)),
                                   "median": float(np.median(column("radius"))),
                                   "p90": float(np.percentile(column("radius"), 90))}
    report["statistics"] = statistics
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(statistics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
