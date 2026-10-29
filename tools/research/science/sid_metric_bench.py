#!/usr/bin/env python3
"""Score candidate SID metrics against the 97 confirmed SID+recall pairs.

A metric is a function of ONE exported SID table, the frozen Issue #126 behaviour
reference and the TRAINING split -- never a validation or test interaction. The
harness computes it for every table in the confirmed collection and reports how
its ranking tracks ``test_recall@10``.

The selection rule is fixed here, before any candidate is measured, because
several metrics are compared against the same 97 numbers:

* a metric is ELIGIBLE only if its Spearman is positive AND the lower bound of
  its 95% bootstrap interval is above zero;
* among the eligible ones the highest Spearman wins;
* a tie is broken by the fewer free constants a metric needs (fewer is better),
  then by the metric identifier, so the choice never depends on the order in
  which the metrics were proposed.

Usage:
    sid_metric_bench.py --module <python file with METRICS = {name: fn}>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import os as _os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[3]
COLLECTION = REPO / "analysis_sid_recall"
OUTPUT = REPO / "results" / "research" / "issue126" / "metric_bench.json"


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _kendall(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    return float(stats.kendalltau(x, y)[0])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    return float(stats.pearsonr(x, y)[0])


def _bootstrap(x: np.ndarray, y: np.ndarray, draws: int = 4000, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        index = rng.integers(0, len(x), len(x))
        if len(np.unique(x[index])) < 3:
            continue
        values.append(_spearman(x[index], y[index])[0])
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _leave_one_out(x: np.ndarray, y: np.ndarray, draws: int = 2000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    lowest = None
    highest = None
    for index in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[index] = False
        value = _spearman(x[keep], y[keep])[0]
        lowest = value if lowest is None else min(lowest, value)
        highest = value if highest is None else max(highest, value)
    return {"min": float(lowest), "max": float(highest)}


def load_context():
    """The frozen reference, the training pairs and the confirmed pairs."""
    sys.path.insert(0, str(REPO))
    from tools.research.science import hyperbolic_behavior_reference as reference_module

    reference = reference_module.load()
    rows = list(__import__("csv").DictReader((COLLECTION / "sid_recall.csv").open()))
    tables = []
    for row in rows:
        path = COLLECTION / "sids" / row["sid_file"]
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        items = sorted(int(key) for key in payload)
        table = {"rows": np.array([payload[str(item)] for item in items], dtype=np.int64),
                 "item_ids": np.array(items, dtype=np.int64)}
        tables.append({"row": row, "sid_sha256": row["sid_sha256"],
                       "test_recall@10": float(row["test_recall@10"]),
                       "test_recall@5": float(row["test_recall@5"]),
                       "table": table})
    train = np.load(REPO / "results" / "stage3_T5Train" / "monotonicity" /
                    "behavior_pairs_s42.npz")
    probe = np.load(REPO / "results" / "stage3_T5Train" / "monotonicity" /
                    "behavior_pairs_valid_s42.npz")
    return {
        "reference": reference,
        "source": reference["source"],
        "target": reference["target"],
        "temperature": float(reference["record"]["temperature"]),
        "train_pairs": {"anchors": train["anchors"], "positives": train["positives"],
                        "negatives": train["negatives"]},
        # the held-out probe Issue #126 froze for Q_H, so a contrastive metric is
        # comparable with it on the same queries
        "probe": {"anchors": probe["anchors"], "positives": probe["positives"],
                  "negatives": probe["negatives"]},
        "items": int(reference["source"].shape[0]),
        "pairs": tables,
    }


def score_metric(name: str, function, context, *, note: str = "",
                 values: list = None) -> dict:
    if values is None:
        values = [float(function(entry["table"], context))
                  for entry in context["pairs"]]
    x = np.array(values)
    y = np.array([entry["test_recall@10"] for entry in context["pairs"]])
    spearman, p = _spearman(x, y)
    return {
        "metric": name,
        "note": note,
        "values": values,
        "range": [float(x.min()), float(x.max())],
        "spearman": spearman, "p": p,
        "kendall": _kendall(x, y),
        "pearson": _pearson(x, y),
        "bootstrap_ci95": _bootstrap(x, y),
        "leave_one_out": _leave_one_out(x, y),
        "eligible": bool(spearman > 0 and _bootstrap(x, y)[0] > 0),
    }




def _module_metrics(path: str):
    spec = importlib.util.spec_from_file_location("candidate_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_METRICS = {}
_CONTEXT = None


def _evaluate_one(job):
    name, index = job
    import torch
    torch.set_num_threads(1)
    entry = _CONTEXT["pairs"][index]
    return float(_METRICS[name](entry["table"], _CONTEXT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True,
                        help="a python file that defines METRICS = {name: fn}")
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--workers", default=0, type=int,
                        help="parallel workers; each one is pinned to one BLAS thread")
    parser.add_argument("--free-constants", default="",
                        help="comma separated name=count, used only to break ties")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("candidate_metrics",
                                                  Path(args.module))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = getattr(module, "METRICS", None)
    if not metrics:
        raise SystemExit("the module defines no METRICS dictionary")

    context = load_context()
    print(f"pairs: {len(context['pairs'])}")
    workers = int(args.workers or 0)
    global _METRICS, _CONTEXT
    _METRICS, _CONTEXT = metrics, context
    values_by_metric = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(name, index) for name in metrics for index in range(len(context["pairs"]))]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for (name, index), value in zip(jobs, pool.map(_evaluate_one, jobs)):
                values_by_metric.setdefault(name, [None] * len(context["pairs"]))[index] = value
    results = []
    for name, function in metrics.items():
        result = score_metric(name, function, context,
                              note=str(getattr(function, "note", "")),
                              values=values_by_metric.get(name))
        results.append(result)
        print(f"{name:32s} rho {result['spearman']:+.4f} (p={result['p']:.3g}) "
              f"CI [{result['bootstrap_ci95'][0]:+.3f}, "
              f"{result['bootstrap_ci95'][1]:+.3f}] "
              f"{'ELIGIBLE' if result['eligible'] else 'not eligible'}")

    constants = {}
    for item in filter(None, args.free_constants.split(",")):
        key, _, value = item.partition("=")
        constants[key.strip()] = int(value or 0)
    eligible = [r for r in results if r["eligible"]]
    eligible.sort(key=lambda r: (-r["spearman"], constants.get(r["metric"], 0),
                                 r["metric"]))
    chosen = eligible[0]["metric"] if eligible else None
    report = {"pairs": len(context["pairs"]),
              "selection_rule": "positive Spearman with bootstrap lower bound > 0; "
                                "highest Spearman; ties by fewer free constants, then name",
              "results": results, "chosen": chosen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nchosen: {chosen}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
