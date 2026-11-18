"""Paired per-user comparison of every Stage3 run that dumped its test ranks.

Every Stage3 run writes `_test_shards/rank<r>.json`; when PER_USER_DUMP was on,
the payload carries `per_user[i].target_rank` for the test row that the rank
evaluated.  The test split and its order are frozen (`test.parquet`, shuffle is
off), and the row identity is `rank + local_row_index * world_size`, so two arms
can be compared row by row on exactly the same users.

Nothing here trains or evaluates a model.  It reads the recorded ranks, rebuilds
the metrics from them, refuses any run whose rebuilt metric disagrees with its
own `test_final.json`, and then reports paired deltas with a McNemar exact test
and a user-level paired bootstrap.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random

import numpy as np
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STAGE3 = os.path.join(REPO, "results", "stage3_T5Train")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

K_LIST = (5, 10)
NDCG_K = 10
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


def gain(rank, k):
    """1 when the target is inside the top-k, else 0."""
    return 0.0 if rank is None or rank >= k else 1.0


def dcg(rank, k):
    return 0.0 if rank is None or rank >= k else 1.0 / math.log2(rank + 2.0)


def load_run(run_dir):
    """Rebuild the per-row ranks of one run, keyed by the frozen test row."""
    shards = sorted(glob.glob(os.path.join(run_dir, "_test_shards", "rank*.json")))
    if not shards:
        return None
    per_row = {}
    world = None
    for path in shards:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("per_user")
        if not rows:
            return None
        rank = int(payload["rank"])
        local = int(rows[-1]["local_row_index"]) + 1
        if world is None:
            world = len(shards)
        for entry in rows:
            index = rank + int(entry["local_row_index"]) * world
            per_row[index] = entry["target_rank"]
    return {"per_row": per_row, "world": world, "shards": len(shards)}


def rebuild_metrics(per_row):
    n = len(per_row)
    values = list(per_row.values())
    recall = {k: sum(gain(r, k) for r in values) / n for k in K_LIST}
    ndcg = {k: sum(dcg(r, k) for r in values) / n for k in K_LIST}
    return n, recall, ndcg


def recorded_metrics(run_dir):
    path = os.path.join(run_dir, "test_final.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def arm_of(run_dir):
    """`.../stage3_T5Train/<arm>/logs/<dataset>/<timestamp>` -> `<arm>`."""
    return run_dir.split(os.sep + "logs" + os.sep)[0].split(os.sep)[-1]


def collect():
    """Every arm-level run whose metric can be rebuilt and checked."""
    candidates = defaultdict(list)
    for run_dir in glob.glob(os.path.join(STAGE3, "*", "logs", "*", "*")):
        if not os.path.isdir(os.path.join(run_dir, "_test_shards")):
            continue
        candidates[arm_of(run_dir)].append(run_dir)

    runs = {}
    rejected = []
    all_runs = {}
    for arm, dirs in sorted(candidates.items()):
        accepted = []
        for run_dir in sorted(dirs):
            loaded = load_run(run_dir)
            if loaded is None:
                rejected.append({"arm": arm, "run": run_dir, "why": "unreadable per_user"})
                continue
            n, recall, ndcg = rebuild_metrics(loaded["per_row"])
            rec = recorded_metrics(run_dir)
            if rec is None:
                rejected.append({"arm": arm, "run": run_dir, "why": "no test_final.json"})
                continue
            mismatch = []
            for k in K_LIST:
                key = f"test_recall@{k}"
                if key in rec and abs(rec[key] - recall[k]) > 1e-9:
                    mismatch.append((key, rec[key], recall[k]))
                key = f"test_ndcg@{k}"
                if key in rec and abs(rec[key] - ndcg[k]) > 1e-9:
                    mismatch.append((key, rec[key], ndcg[k]))
            if mismatch:
                rejected.append({"arm": arm, "run": run_dir, "why": "rebuild mismatch",
                                 "detail": mismatch[:3]})
                continue
            if rec.get("n_eval") is not None and rec["n_eval"] != n:
                rejected.append({"arm": arm, "run": run_dir, "why": "n_eval mismatch"})
                continue
            accepted.append({"run": run_dir, "n": n, "recall": recall, "ndcg": ndcg,
                             "per_row": loaded["per_row"], "world": loaded["world"]})
        if accepted:
            # Several verified runs of one arm: the largest full-split run, newest on ties.
            runs[arm] = sorted(accepted, key=lambda c: (c["n"], c["run"]))[-1]
            all_runs[arm] = accepted
    return runs, rejected, all_runs


def paired(a, b):
    """Paired comparison of arm `a` against arm `b` on the shared test rows."""
    rows = sorted(set(a["per_row"]) & set(b["per_row"]))
    if len(rows) != a["n"] or len(rows) != b["n"]:
        return {"error": f"shared rows {len(rows)} != n {a['n']}/{b['n']}"}
    ra = [a["per_row"][i] for i in rows]
    rb = [b["per_row"][i] for i in rows]
    n = len(rows)
    out = {"n": n}
    for k in K_LIST:
        ha = [gain(r, k) for r in ra]
        hb = [gain(r, k) for r in rb]
        both = sum(1 for x, y in zip(ha, hb) if x and y)
        only_a = sum(1 for x, y in zip(ha, hb) if x and not y)
        only_b = sum(1 for x, y in zip(ha, hb) if y and not x)
        delta = (only_a - only_b) / n
        out[f"recall@{k}"] = {
            "a": sum(ha) / n,
            "b": sum(hb) / n,
            "delta": delta,
            "discordant_a_only": only_a,
            "discordant_b_only": only_b,
            "mcnemar_exact_p": mcnemar(only_a, only_b),
        }
        da = [dcg(r, k) for r in ra]
        db = [dcg(r, k) for r in rb]
        out[f"ndcg@{k}"] = {"a": sum(da) / n, "b": sum(db) / n,
                            "delta": sum(x - y for x, y in zip(da, db)) / n}
    out["bootstrap_recall@10"] = bootstrap(ra, rb)
    # Row-identity check: two arms scoring the *same* test rows agree far more
    # often than two arms whose rows are unrelated.  The permuted figure is the
    # chance level this sample would show if the row order were not shared.
    hit_a = [1 if gain(r, 10) else 0 for r in ra]
    hit_b = [1 if gain(r, 10) else 0 for r in rb]
    shuffler = random.Random(BOOTSTRAP_SEED + 7)
    shuffled = hit_b[:]
    shuffler.shuffle(shuffled)
    out["row_identity"] = {
        "agreement": sum(1 for x, y in zip(hit_a, hit_b) if x == y) / n,
        "permuted_agreement": sum(1 for x, y in zip(hit_a, shuffled) if x == y) / n,
    }
    return out


def mcnemar(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(0, min(b, c) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def bootstrap(ra, rb):
    """User-level paired bootstrap of the recall@10 difference.

    Poisson(1) weights on the per-user differences are the standard
    "Bayesian"/weighted form of the resample-with-replacement bootstrap: the
    weight vector is the count of times each user is drawn.  Chunked so the
    weight matrix stays small.
    """
    
    ha = np.array([1.0 if gain(r, 10) else 0.0 for r in ra])
    hb = np.array([1.0 if gain(r, 10) else 0.0 for r in rb])
    diff = ha - hb
    n = diff.size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP)
    chunk = 500
    for start in range(0, BOOTSTRAP, chunk):
        stop = min(start + chunk, BOOTSTRAP)
        weights = rng.poisson(1.0, size=(stop - start, n))
        estimates[start:stop] = (weights * diff).sum(axis=1) / n
    estimates.sort()
    lo = float(estimates[int(0.025 * BOOTSTRAP)])
    hi = float(estimates[int(0.975 * BOOTSTRAP) - 1])
    point = float(diff.mean())
    share = float((estimates > 0).mean())
    return {"delta": point, "ci95": [lo, hi], "share_positive": share,
            "method": "poisson-weighted paired bootstrap"}


PAIRS = [
    ("mixed-vs-euclidean", "TIGER_mixed_rq_F0", "TIGER_mixed_rq_E"),
    ("fixed-vs-global-alpha", "TIGER_mixed_rq_F0", "TIGER_mixed_rq_G"),
    ("fixed-vs-local-alpha", "TIGER_mixed_rq_F0", "TIGER_mixed_rq_L"),
    ("global-vs-local-alpha", "TIGER_mixed_rq_L", "TIGER_mixed_rq_G"),
    ("mixed-vs-af3-aux", "TIGER_mixed_rq_F0", "TIGER_mixed_rq_A"),
    ("role-vs-global", "TIGER_role_geom_rq", "TIGER_role_geom_global_rq"),
    ("role-vs-shuffled", "TIGER_role_geom_rq", "TIGER_role_geom_shuffled_rq"),
    ("global-vs-shuffled", "TIGER_role_geom_global_rq", "TIGER_role_geom_shuffled_rq"),
    ("role-vs-mixed-f0", "TIGER_role_geom_rq", "TIGER_mixed_rq_F0"),
    ("mixed-f0-vs-baseline", "TIGER_mixed_rq_F0", "TIGER_RQ-VAE"),
    ("mixed-f0-vs-resample", "TIGER_mixed_rq_F0", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE__resample"),
    ("role-vs-resample", "TIGER_role_geom_rq", "TIGER_BEHAVIOR_SAMPLE_RQ-VAE__resample"),
    ("codeword-vs-global", "TIGER_graphlet_codebook_rq", "TIGER_graphlet_codebook_global_rq"),
    ("codeword-vs-f0", "TIGER_graphlet_codebook_rq", "TIGER_mixed_rq_F0"),
    ("global-vs-f0", "TIGER_graphlet_codebook_global_rq", "TIGER_mixed_rq_F0"),
]


def main():
    runs, rejected, all_runs = collect()
    arms = sorted(runs)
    print(f"rebuilt {len(arms)} arms from recorded per-user ranks; "
          f"{len(rejected)} run(s) rejected")
    for entry in rejected:
        print("  REJECT", entry["arm"], entry["why"], entry.get("detail", ""))

    table = {"schema": "genrec.paired_rank_analysis.v1", "arms": {}, "pairs": {},
             "rejected": rejected}

    # The noise floor: arms that were trained and evaluated more than once, with the
    # same recipe, differ in recall by more than the mechanism effects this report
    # discusses.  The spread is the honest scale against which a delta must be read.
    spreads = []
    for arm, candidates_runs in sorted(all_runs.items()):
        values = sorted({round(run["recall"][10], 12) for run in candidates_runs
                         if run["n"] == max(r["n"] for r in candidates_runs)})
        if len(values) > 1:
            spreads.append({"arm": arm, "runs": len(values), "min": values[0],
                            "max": values[-1], "spread": values[-1] - values[0]})
    table["rerun_spread"] = {
        "arms_with_several_runs": len(spreads),
        "median_spread": float(np.median([s["spread"] for s in spreads])) if spreads else None,
        "max_spread": float(max(s["spread"] for s in spreads)) if spreads else None,
        "per_arm": spreads,
    }
    if spreads:
        print(f"same-recipe rerun spread over {len(spreads)} arms: "
              f"median={np.median([s['spread'] for s in spreads]):.5f} "
              f"max={max(s['spread'] for s in spreads):.5f}")
    for arm in arms:
        run = runs[arm]
        table["arms"][arm] = {
            "run": run["run"],
            "n": run["n"],
            "world_size": run["world"],
            "recall@5": run["recall"][5],
            "recall@10": run["recall"][10],
            "ndcg@10": run["ndcg"][10],
        }
        print(f"{run['recall'][10]:.6f}  r@5={run['recall'][5]:.6f}  n={run['n']:6d}  {arm}")

    print("\n== paired ==")
    for name, a, b in PAIRS:
        if a not in runs or b not in runs:
            missing = [x for x in (a, b) if x not in runs]
            print(f"{name:32s} SKIPPED (missing {missing})")
            table["pairs"][name] = {"error": f"missing {missing}"}
            continue
        result = paired(runs[a], runs[b])
        table["pairs"][name] = {"a": a, "b": b, **result}
        if "error" in result:
            print(f"{name:32s} {result['error']}")
            continue
        r10 = result["recall@10"]
        boot = result["bootstrap_recall@10"]
        print(f"{name:32s} dR@10={r10['delta']:+.6f} "
              f"[{boot['ci95'][0]:+.6f},{boot['ci95'][1]:+.6f}] "
              f"p_mcnemar={r10['mcnemar_exact_p']:.3g} "
              f"onlyA={r10['discordant_a_only']} onlyB={r10['discordant_b_only']}")

    with open(os.path.join(OUT_DIR, "paired_rank_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, sort_keys=True)
    print("\nwrote paired_rank_analysis.json")


if __name__ == "__main__":
    sys.exit(main())
