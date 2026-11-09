"""Unified SID and behaviour metrics for every model, on the FULL test relations.

What changes relative to analyze_prefix_behavior (corrected v2), and why.

The earlier analysis kept one behaviour positive per anchor:

    _, uniq = np.unique(anchors, return_index=True)

test.parquet holds 56,636 distinct history[-1] -> target relations over 16,828
anchors, so an anchor has 3.37 distinct successors on average and that line
discarded 39,808 real relations, 70.3% of the available behaviour signal. Every
relation is kept here. Only self-loops and out-of-range ids are dropped, and the
count dropped at each filter is recorded.

Negatives go from one per positive to ten, matched on the popularity decile and
the appearance-frequency decile, excluding the anchor, the positive itself and
every item with an observed relation to that anchor in EITHER train or test, so
a test positive can never become its own negative. Ten distinct draws are taken
without replacement where the candidate pool allows it; pools that cannot supply
ten are recorded. The one-negative setting is computed alongside as a control,
because moving from 1 to 10 negatives mechanically shrinks naive confidence
intervals and the analysis must show whether any conclusion depends on it.

Dependencies. One anchor now produces several positives and ten negatives each,
so the observations are not independent. Both aggregates are reported:
micro over pairs and macro over anchors, with anchor-level cluster bootstrap
intervals. The cluster is the anchor, which is the unit that actually repeats.

Relations are never mixed: the independent test relation is primary, the
common-successor relation is separate. Test edges decide what counts as related
and are never used by any model.

Enrichment with a zero negative rate is undefined, never infinity and never an
epsilon, so a model cannot look infinitely selective by sharing nothing.

Read-only: no model, SID, checkpoint, protocol or prior result is modified.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prefix_behavior as AP  # noqa: E402

ROOT = AP.ROOT
OUT_DIR = ROOT / "results/stage3_T5Train/monotonicity"
INVENTORY = OUT_DIR / "model_inventory.json"
OUT_JSON = OUT_DIR / "unified_metrics.json"
OUT_CSV = OUT_DIR / "unified_metrics.csv"

N_ITEMS = 24587
N_LEVELS = 3
MIN_CELL = AP.MIN_CELL
N_BOOT = 2000
N_NEG = 10
MAX_PAIRS = 60000


def load_levels(path: Path) -> np.ndarray:
    """(N, 3) codes; the fourth stored token is the collision extension token."""
    levels, _ = AP.load_levels(path)
    return levels


# ------------------------------------------------------------------ relations

def build_test_relations(n_items: int) -> dict:
    """Every distinct history[-1] -> target in test.parquet, with filter counts."""
    test = pd.read_parquet(AP.DATA / "test.parquet")
    pairs = AP.transitions(test, n_items)
    n_raw = len(pairs)
    a, b = pairs[:, 0], pairs[:, 1]
    bad_id = (a < 0) | (a >= n_items) | (b < 0) | (b >= n_items)
    keep = ~bad_id & (a != b)
    kept = pairs[keep]
    n_anchor = len(np.unique(kept[:, 0]))
    return {
        "pairs": kept,
        "filters": {
            "rows_in_test_parquet": int(len(test)),
            "distinct_transitions": n_raw,
            "dropped_invalid_item_id": int(bad_id.sum()),
            "dropped_self_loop": int(((a == b) & ~bad_id).sum()),
            "kept_pairs": int(len(kept)),
            "distinct_anchors": int(n_anchor),
            "dropped_by_old_one_per_anchor_rule": int(len(kept) - n_anchor),
            "old_one_per_anchor_pairs": int(n_anchor),
            "mean_successors_per_anchor": float(len(kept) / max(n_anchor, 1)),
        },
    }


def build_common_successor(n_items: int) -> dict:
    train = pd.read_parquet(AP.DATA / "train.parquet")
    succ_tr = AP.successors(train, n_items)
    preds_tr = AP.predecessors(succ_tr)
    pairs = AP.common_successor_pairs(succ_tr, preds_tr, cap_per_item=64)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (a != b) & (a >= 0) & (a < n_items) & (b >= 0) & (b < n_items)
    kept = pairs[keep]
    # Without the one-per-anchor rule this relation has ~3.1M pairs, and ten
    # negatives each would need 31M draws. It is a secondary relation, so it is
    # subsampled to the size of the test relation with a fixed seed and the
    # subsample size is recorded rather than silently applied.
    subsampled = 0
    if len(kept) > MAX_PAIRS:
        rng = np.random.default_rng(AP.SEED + 77)
        sel = rng.choice(len(kept), size=MAX_PAIRS, replace=False)
        kept = kept[np.sort(sel)]
        subsampled = MAX_PAIRS
    return {
        "pairs": kept,
        "subsampled_to": subsampled,
        "succ_tr": succ_tr, "preds_tr": preds_tr,
        "filters": {
            "distinct_pairs": int(len(pairs)),
            "dropped_self_loop": int((a == b).sum()),
            "kept_pairs": int(keep.sum()),
            "distinct_anchors": int(len(np.unique(kept[:, 0]))),
            "cap_per_item": 64,
            "subsampled_to": subsampled,
        },
    }


def sample_negatives(
    anchors, positives, related, pop_bin, freq_bin, k: int, rng
):
    """k matched negatives per positive, distinct within a positive where possible."""
    cells = defaultdict(set)
    for it in range(N_ITEMS):
        cells[(int(pop_bin[it]), int(freq_bin[it]))].add(it)

    out = np.full((len(anchors), k), -1, dtype=np.int64)
    fallbacks, short_pools, violations = 0, 0, 0
    for i, (a, b) in enumerate(zip(anchors, positives)):
        a, b = int(a), int(b)
        rel_i = set(related[i])  # this anchor's own observed relations
        key = (int(pop_bin[b]), int(freq_bin[b]))
        pool = None
        for step in range(0, 11):
            if step == 0:
                cand_cells = [key]
            else:
                cand_cells = [
                    c for c in (
                        (key[0] - step, key[1]),
                        (key[0], key[1] - step),
                        (min(key[0] + step, 9), key[1]),
                        (key[0], min(key[1] + step, 9)),
                    )
                    if 0 <= c[0] < 10 and 0 <= c[1] < 10
                ]
            for c in cand_cells:
                members = cells.get(c)
                if not members:
                    continue
                avail = members - rel_i - {a, b}
                if len(avail) >= k:
                    pool = avail
                    break
            if pool is not None:
                break
        if step > 0:
            fallbacks += 1
        if pool is None:
            # Fall back to the whole catalogue minus everything related.
            pool = set(range(N_ITEMS)) - rel_i - {a, b}
        if len(pool) < k:
            short_pools += 1
        take = min(k, len(pool))
        # Sets are not subscriptable; materialise once and index into the array.
        pool_arr = np.fromiter(pool, dtype=np.int64, count=len(pool))
        drawn = rng.choice(len(pool_arr), size=take, replace=False)
        chosen = pool_arr[drawn].tolist()
        # Top up by sampling the remainder with replacement, and record it.
        while len(chosen) < k:
            chosen.append(chosen[rng.integers(len(chosen))])
        for x in chosen:
            if x == b:
                violations += 1
            if x == a:
                violations += 1
            if x in rel_i:
                violations += 1
        out[i] = chosen
    stats = {
        "k_negatives": k,
        "positives": int(len(anchors)),
        "negative_draws": int(len(anchors) * k),
        "bin_fallbacks": fallbacks,
        "pools_shorter_than_k": short_pools,
        "invariant_violations": violations,
        "distinct_negatives": int(len(set(out.reshape(-1).tolist()))),
        "max_negatives_per_positive_distinct": k,
    }
    return out, stats


# ------------------------------------------------------------------- statistics

def anchor_cluster_bootstrap(values: np.ndarray, anchors: np.ndarray,
                             rng) -> dict:
    """Mean with a 95% CI resampling ANCHORS, not pairs.

    One anchor contributes several positives and ten negatives each, so pairs
    from one anchor are not independent. Resampling pairs would shrink the
    interval purely by raising k.
    """
    uniq, inv = np.unique(anchors, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(np.float64), minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_clusters": int(len(uniq)),
    }


def sharing(pre_ids, anchors, positives, negatives_flat, rng) -> dict:
    """Positive and negative sharing with micro and macro aggregates.

    ``negatives_flat`` is (n_positives, k). Each negative is compared against
    ITS OWN anchor, so the anchor index is repeated k times rather than reused.
    """
    k = negatives_flat.shape[1]
    ps = (pre_ids[positives] == pre_ids[anchors]).astype(np.float64)
    neg_flat = negatives_flat.reshape(-1)
    anchor_rep = np.repeat(anchors, k)
    ns = (pre_ids[neg_flat] == pre_ids[anchor_rep]).astype(np.float64)

    micro_pos, micro_neg = float(ps.mean()), float(ns.mean())
    uniq = np.unique(anchors)
    macro_pos = float(np.mean([ps[anchors == a].mean() for a in uniq]))
    macro_neg = float(np.mean([ns[anchor_rep == a].mean() for a in uniq]))
    gap = micro_pos - micro_neg
    er = (micro_pos / micro_neg) if micro_neg > 0 else None
    return {
        "p_share_pos": micro_pos,
        "p_share_neg": micro_neg,
        "gap": gap,
        "er": er,
        "er_undefined_reason": None if er is not None else "negative sharing rate is zero",
        "macro_p_share_pos": macro_pos,
        "macro_p_share_neg": macro_neg,
        "macro_gap": macro_pos - macro_neg,
        "n_positive_pairs": int(len(ps)),
        "n_negative_pairs": int(len(ns)),
        "n_anchors": int(len(uniq)),
        "pos_ci": anchor_cluster_bootstrap(ps, anchors, rng)["ci"],
        "neg_ci": anchor_cluster_bootstrap(ns, anchor_rep, rng)["ci"],
        "sampled_precision_given_share": (
            int(ps.sum()) / (int(ps.sum()) + int(ns.sum()))
            if (int(ps.sum()) + int(ns.sum())) > 0 else None
        ),
    }


def conditional_retention(pre, pos_anchors, positives, neg_anchors,
                          negatives_flat, k) -> dict:
    """Positives and negatives each bring their own anchor index: the negative
    side is ten rows per anchor, so its anchors are repeated k times."""
    prev, cur = k - 1, k
    res = {}
    for side, anchors_i, others in (
        ("pos", pos_anchors, positives),
        ("neg", neg_anchors, negatives_flat),
    ):
        cond = pre[prev][anchors_i] == pre[prev][others]
        n = int(cond.sum())
        res[f"n_effective_{side}"] = n
        res[f"p_retain_{side}"] = (
            float((pre[cur][anchors_i][cond] == pre[cur][others][cond]).mean())
            if n >= MIN_CELL else None
        )
        if n < MIN_CELL:
            res[f"insufficient_{side}"] = f"n={n} below min_cell={MIN_CELL}"
    if res.get("p_retain_pos") is not None and res.get("p_retain_neg") is not None:
        res["retention_diff"] = res["p_retain_pos"] - res["p_retain_neg"]
        res["retention_er"] = (
            res["p_retain_pos"] / res["p_retain_neg"]
            if res["p_retain_neg"] > 0 else None
        )
    return res


def model_settings(levels, anchors, positives, negs, rng) -> dict:
    pre = {k: AP.prefix_ids(levels, k) for k in (1, 2, 3)}
    out = {}
    for k, label in ((1, "L1"), (2, "L1_L2"), (3, "L1_L2_L3")):
        s = sharing(pre[k], anchors, positives, negs, rng)
        s.pop("gap_ci", None)
        out[label] = s
    k_neg = negs.shape[1] if negs.ndim == 2 else 1
    flat = negs.reshape(-1) if negs.ndim == 2 else negs
    anchor_rep = np.repeat(anchors, k_neg)
    for k in (2, 3):
        out[f"L{k}_given_L{k-1}"] = conditional_retention(
            pre, anchors, positives, anchor_rep, flat, k
        )
    st = AP.structure_stats(levels)
    out["collision_rate"] = st["full_sid"]["collision_rate"]
    out["distinct_triples"] = st["full_sid"]["distinct_triples"]
    out["largest_triple_group"] = st["full_sid"]["largest_group"]
    for k, label in ((1, "L1"), (2, "L1_L2"), (3, "L1_L2_L3")):
        s = st[f"level{k}"]
        out[f"{label}_prefixes"] = s["distinct_prefixes"]
        out[f"{label}_singleton_item_frac"] = s["items_in_singleton_groups_frac"]
        out[f"{label}_singleton_prefix_frac"] = s["singleton_prefix_group_frac"]
        out[f"{label}_max_group"] = s["max_group_size"]
        out[f"{label}_mean_group"] = s["mean_group_size"]
    sizes = AP.group_sizes(pre[1])[pre[1]]
    out["L1_gini"] = float(_gini(sizes.astype(np.float64)))
    out["L1_frac_pos_in_group_ge100"] = float((sizes[positives] >= 100).mean())
    return out


def _gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    inv = json.loads(INVENTORY.read_text())
    train = pd.read_parquet(AP.DATA / "train.parquet")
    succ_te = AP.successors(pd.read_parquet(AP.DATA / "test.parquet"), N_ITEMS)
    pop, freq = AP.popularity_and_frequency(train, N_ITEMS)
    pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)

    test_rel = build_test_relations(N_ITEMS)
    cs = build_common_successor(N_ITEMS)
    succ_tr, preds_tr = cs["succ_tr"], cs["preds_tr"]

    def related_for(kind, anchors):
        out = []
        for a in anchors:
            a = int(a)
            if kind == "test_transition":
                rel = set(succ_tr.get(a, ())) | set(succ_te.get(a, ()))
                rel.discard(a)
            else:
                rel = AP.common_successor_neighbours(a, succ_tr, preds_tr)
            out.append(rel)
        return out

    pair_sets = {}
    for key, kind, pairs in (
        ("test_transition", "test_transition", test_rel["pairs"]),
        ("common_succ", "common_succ", cs["pairs"]),
    ):
        anchors, positives = pairs[:, 0].copy(), pairs[:, 1].copy()
        rel = related_for(kind, anchors)
        rng10 = np.random.default_rng(AP.SEED + 1000 * (1 if key == "test_transition" else 2))
        neg10, st10 = sample_negatives(anchors, positives, rel, pop_bin, freq_bin,
                                       N_NEG, rng10)
        rng1 = np.random.default_rng(AP.SEED + 1000 * (1 if key == "test_transition" else 2))
        neg1, st1 = sample_negatives(anchors, positives, rel, pop_bin, freq_bin,
                                     1, rng1)
        pair_sets[key] = {
            "anchors": anchors, "positives": positives,
            "neg10": neg10, "neg1": neg1,
            "stats_neg10": st10, "stats_neg1": st1,
            "filters": test_rel["filters"] if key == "test_transition" else cs["filters"],
        }
        print(f"{key}: {len(anchors)} positives over "
              f"{len(np.unique(anchors))} anchors; "
              f"neg10 draws {st10['negative_draws']} "
              f"(violations {st10['invariant_violations']}, "
              f"short pools {st10['pools_shorter_than_k']})")

    models = {}
    for m in inv["models"]:
        levels = load_levels(ROOT / m["sid_path"])
        rng = np.random.default_rng(AP.SEED + 7)
        settings = {}
        for key, ps in pair_sets.items():
            for tag, negs in (("neg10", ps["neg10"]), ("neg1", ps["neg1"])):
                settings[f"{key}_{tag}"] = model_settings(
                    levels, ps["anchors"], ps["positives"], negs, rng
                )
        models[m["model"]] = {
            "model": m["model"], "family": m["family"],
            "recall@10": m["recall@10"], "ndcg@10": m["ndcg@10"],
            "sid_path": m["sid_path"], "sid_sha256": m["sid_sha256"],
            "stage3_runs": m["stage3_runs"],
            "settings": settings,
        }
        print(f"  scored {m['model']}")

    report = {
        "seed": AP.SEED, "min_cell": MIN_CELL, "n_boot": N_BOOT,
        "n_negatives": N_NEG, "n_models": len(models),
        "pair_sets": {
            k: {
                "n_positive_pairs": int(len(v["anchors"])),
                "n_anchors": int(len(np.unique(v["anchors"]))),
                "filters": v["filters"],
                "neg10": v["stats_neg10"], "neg1": v["stats_neg1"],
            } for k, v in pair_sets.items()
        },
        "models": models,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    _write_csv(report)
    print(f"written: {OUT_JSON}\nwritten: {OUT_CSV}")


def _write_csv(report: dict) -> None:
    cols = ["model", "family", "recall@10", "setting", "collision_rate"]
    for lv in ("L1", "L1_L2", "L1_L2_L3"):
        cols += [f"{lv}_pos", f"{lv}_neg", f"{lv}_gap", f"{lv}_er",
                 f"{lv}_macro_pos", f"{lv}_macro_neg"]
    cols += ["L2gL1_pos", "L2gL1_neg", "L2gL1_npos", "L3gL2_pos", "L3gL2_neg",
             "L3gL2_npos", "L1_prefixes", "L1_max_group", "L1_gini",
             "L2_prefixes", "L3_prefixes"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, r in sorted(report["models"].items(),
                              key=lambda kv: -kv[1]["recall@10"]):
            for setting, s in r["settings"].items():
                row = [r["model"], r["family"], r["recall@10"], setting,
                       s["collision_rate"]]
                for lv in ("L1", "L1_L2", "L1_L2_L3"):
                    m = s[lv]
                    row += [m["p_share_pos"], m["p_share_neg"], m["gap"], m["er"],
                            m["macro_p_share_pos"], m["macro_p_share_neg"]]
                c2, c3 = s["L2_given_L1"], s["L3_given_L2"]
                row += [c2.get("p_retain_pos"), c2.get("p_retain_neg"),
                        c2.get("n_effective_pos"),
                        c3.get("p_retain_pos"), c3.get("p_retain_neg"),
                        c3.get("n_effective_pos"),
                        s["L1_prefixes"], s["L1_max_group"], s["L1_gini"],
                        s["L1_L2_prefixes"], s["L1_L2_L3_prefixes"]]
                w.writerow(row)


if __name__ == "__main__":
    main()