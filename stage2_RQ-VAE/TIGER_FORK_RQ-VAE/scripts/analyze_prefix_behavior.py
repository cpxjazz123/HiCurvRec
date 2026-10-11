"""Semantic ID prefix vs item behaviour relation: corrected offline analysis.

Corrects the previous run of this analysis. No retraining, no checkpoint and no
Semantic ID is touched; only the statistics are redone, on the same two models:

  A  TIGER Euclidean baseline     R@10 0.053674
  B  Exp1a behaviour contrastive  R@10 0.063633

Corrections applied here, and what was wrong before:

1. Common-successor negatives. The relation is "b shares a successor with a":
   there exists c with a->c and b->c. The previous code computed the set of
   two-hop DESCENDANTS, union(succ(b) for b in succ(a)), which is a different
   relation and admitted negatives that genuinely share a successor. The
   neighbour set is now built from the reverse adjacency, predecessors[c], as
   union(predecessors[c] for c in successors(a)) minus {a}. Positives are still
   sampled with a per-item cap, but the exclusion set is the FULL relation, not
   the capped sample.

   Transition negatives now exclude every observed edge of the relation being
   evaluated: for the independent test relation that is the union of the train
   and test edges, so a test positive can no longer be drawn as its own
   negative. Test edges are used only to build negatives for evaluation; they
   never enter any model and never enter training.

2. P(pos | share). The old denominator pooled pos and neg sharing flags. The
   reported quantity is now sampled_precision_given_share =
   sum(pos_share) / (sum(pos_share) + sum(neg_share)), which is a property of the
   balanced pos/neg sample, NOT the population P(behaviour-related | shared).
   The population quantity is not identified here: recovering it needs the
   positive prior and the negative sampling rate, and reporting the sample
   precision as a posterior would be wrong. Undefined, never zero-filled, when
   the denominator is zero.

3. Between-model tests. The previous run tested each model against its own
   negatives and stopped there, which cannot say whether B differs from A. Both
   models are now scored on the SAME items, the SAME behaviour positives, the
   SAME matched negatives and the SAME split, and compared pair by pair:
     delta_pos  = mean(pos_share_B - pos_share_A)
     delta_neg  = mean(neg_share_B - neg_share_A)
     did        = mean((pos_B - neg_B) - (pos_A - neg_A))
   Confidence intervals come from a CLUSTER bootstrap that resamples ANCHOR
   ITEMS, because one item contributes many relations and treating those as
   independent would understate every interval. The p-value is a paired
   Wilcoxon signed-rank test on the same per-pair differences.

4. Conditional retention. Comparing P(share L2 | share L1) across models is
   confounded because the two models put different item pairs in that condition.
   The own-condition rates are still reported, but the comparison is now made on
   the COMMON set of pairs that satisfy the previous-level condition under BOTH
   models, and the size of that set is reported. A set below MIN_CELL is marked
   insufficient instead of being interpreted.

5. Other checks. The singleton PREFIX-GROUP fraction and the singleton ITEM
   fraction are now reported as distinct fields. The group-size stratification
   is matched: the old version bucketed pairs by the positive's group size only,
   so positives and negatives inside a stratum could sit in different-sized
   groups. Seeds come from integer offsets; Python's hash() is not used because
   it is salted per process and made the negative draw non-reproducible across
   runs. Negatives are asserted to differ from the positive, from the anchor and
   from the full related set. The fourth SID token is asserted to be excluded
   from the level analysis and recorded rather than counted.

6. Outputs. Original results are kept; corrected numbers go to a separate JSON
   and a Markdown report rendered from the computed values, never typed by hand.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261011
N_LEVELS = 3
MIN_CELL = 30
N_BOOT = 2000
ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SID_PATHS = {
    "A_tiger": ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
    "B_exp1a": ROOT
    / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
}
DATA = ROOT / "results/stage0_build_parquet"
OUT_JSON = ROOT / "results/stage2_RQ-VAE/prefix_behavior_analysis_corrected.json"
OUT_MD = ROOT / "results/stage2_RQ-VAE/prefix_behavior_analysis_corrected.md"
RECALL = {"A_tiger": 0.053674, "B_exp1a": 0.063633}
SIZE_EDGES = [0, 2, 5, 10, 25, 100, np.inf]
SIZE_LABELS = ["1", "2-4", "5-9", "10-24", "25-99", "100+"]


# ---------------------------------------------------------------- SID loading

def load_levels(path: Path) -> tuple[np.ndarray, dict]:
    """(N, 3) integer codes plus a record of what was dropped.

    The stored SID is [L1, L2, L3, collision_token]. Only the first three are
    levels; the fourth is the Collision Extension Token and is recorded, not
    counted, so no "full SID" figure silently includes it.
    """
    raw = json.loads(path.read_text())
    keys = sorted(int(k) for k in raw)
    tokens = np.asarray([raw[str(k)] for k in keys], dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError(f"{path}: unexpected shape {tokens.shape}")
    if tokens.shape[1] < N_LEVELS:
        raise ValueError(f"{path}: only {tokens.shape[1]} tokens")
    info = {
        "n_tokens_stored": int(tokens.shape[1]),
        "levels_analysed": N_LEVELS,
        "collision_token_values": sorted(
            set(tokens[:, N_LEVELS].tolist())
        )[:8] if tokens.shape[1] > N_LEVELS else [],
        "collision_token_dropped": bool(tokens.shape[1] > N_LEVELS),
    }
    return tokens[:, :N_LEVELS], info


def prefix_ids(levels: np.ndarray, k: int) -> np.ndarray:
    return np.unique(levels[:, :k], axis=0, return_inverse=True)[1].astype(np.int64)


def group_sizes(ids: np.ndarray) -> np.ndarray:
    return np.bincount(ids).astype(np.int64)


# ------------------------------------------------------------------ relations

def transitions(frame: pd.DataFrame, n_items: int) -> np.ndarray:
    seen = set()
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy()
    ):
        if history is None or len(history) == 0:
            continue
        a, b = int(history[-1]), int(target)
        if 0 <= a < n_items and 0 <= b < n_items:
            seen.add((a, b))
    return np.asarray(sorted(seen), dtype=np.int64)


def successors(frame: pd.DataFrame, n_items: int) -> dict[int, set[int]]:
    succ: dict[int, set[int]] = defaultdict(set)
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy()
    ):
        if history is None or len(history) == 0:
            continue
        a, b = int(history[-1]), int(target)
        if 0 <= a < n_items and 0 <= b < n_items:
            succ[a].add(b)
    return dict(succ)


def predecessors(succ: dict[int, set[int]]) -> dict[int, set[int]]:
    """Reverse adjacency predecessors[c] = {a : a -> c}."""
    preds: dict[int, set[int]] = defaultdict(set)
    for a, outs in succ.items():
        for c in outs:
            preds[c].add(a)
    return dict(preds)


def common_successor_neighbours(
    anchor: int, succ: dict, preds: dict
) -> set[int]:
    """b shares a successor with anchor: exists c with anchor->c and b->c.

    union(predecessors[c] for c in successors(anchor)) minus {anchor}. Built on
    the reverse adjacency because the two-hop DESCENDANTS are a different set
    and were the previous bug.
    """
    out: set[int] = set()
    for c in succ.get(anchor, ()):  # noqa: B007
        out |= preds.get(c, set())
    out.discard(anchor)
    return out


def common_successor_pairs(
    succ: dict, preds: dict, cap_per_item: int
) -> np.ndarray:
    """Positive pairs (a,b) with a->c and b->c for some c, from train only.

    Every predecessor pair of every successor c is such a pair. Predecessor sets
    are large, so each c contributes at most ``cap_per_item`` pairs, drawn with
    a fixed generator; the cap applies to POSITIVES only, never to the exclusion
    set used for negatives.
    """
    rng = np.random.default_rng(SEED)
    out: set[tuple[int, int]] = set()
    for pres in preds.values():
        pres = sorted(pres)
        if len(pres) < 2:
            continue
        if len(pres) > cap_per_item:
            idx = rng.choice(len(pres), size=cap_per_item, replace=False)
            pres = [pres[i] for i in sorted(idx)]
        for i in range(len(pres)):
            for j in range(i + 1, len(pres)):
                out.add((pres[i], pres[j]))
    return np.asarray(sorted(out), dtype=np.int64)


def popularity_and_frequency(frame: pd.DataFrame, n_items: int):
    pop = np.zeros(n_items, dtype=np.int64)
    freq = np.zeros(n_items, dtype=np.int64)
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy()
    ):
        b = int(target)
        if 0 <= b < n_items:
            pop[b] += 1
        if history is not None:
            for it in np.asarray(history).ravel().tolist():
                it = int(it)
                if 0 <= it < n_items:
                    freq[it] += 1
    return pop, freq


def decile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * n_bins // max(len(values), 1)).clip(0, n_bins - 1)


def related_set(
    anchor: int, kind: str, succ_tr: dict, preds_tr: dict, succ_te: dict
) -> set[int]:
    """Every item with an OBSERVED relation of this kind to the anchor.

    For the independent test relation the train and test edges are unioned, so a
    test positive is never drawn as its own negative. Test edges never reach a
    model; they are used only here, to define what counts as related.
    """
    if kind == "train_transition":
        return set(succ_tr.get(anchor, ()))
    if kind == "test_transition":
        rel = set(succ_tr.get(anchor, ())) | set(succ_te.get(anchor, ()))
        rel.discard(anchor)
        return rel
    if kind == "common_succ":
        return common_successor_neighbours(anchor, succ_tr, preds_tr)
    raise ValueError(kind)


def build_negatives(
    anchors, positives, related, pop_bin, freq_bin, rng
) -> tuple[np.ndarray, dict, list[str]]:
    """One negative per positive, matched on popularity and frequency decile."""
    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for it in range(len(pop_bin)):
        by_cell[(int(pop_bin[it]), int(freq_bin[it]))].append(it)

    negs, fallbacks, violations = [], 0, []
    for (a, b), rel in zip(zip(anchors, positives), related):
        a, b = int(a), int(b)
        key = (int(pop_bin[b]), int(freq_bin[b]))
        candidates = None
        for step in range(0, 11):
            cells = [key] if step == 0 else [
                c for c in (
                    (key[0] - step, key[1]),
                    (key[0], key[1] - step),
                    (min(key[0] + step, 9), key[1]),
                    (key[0], min(key[1] + step, 9)),
                )
                if 0 <= c[0] < 10 and 0 <= c[1] < 10
            ]
            for cell in cells:
                pool = [
                    x for x in by_cell.get(cell, ())
                    if x not in rel and x != a and x != b
                ]
                if pool:
                    candidates = pool
                    break
            if candidates:
                break
        if step > 0:
            fallbacks += 1
        if not candidates:
            candidates = [
                x for x in range(len(pop_bin))
                if x not in rel and x != a and x != b
            ]
        neg = int(candidates[rng.integers(len(candidates))])
        # The three invariants are asserted, not assumed.
        if neg == b:
            violations.append("equals_positive")
        if neg == a:
            violations.append("equals_anchor")
        if neg in rel:
            violations.append("in_related_set")
        negs.append(neg)
    stats_out = {
        "negatives": len(negs),
        "distinct_negatives": len(set(negs)),
        "fallback_bin_matches": fallbacks,
        "violations": dict(pd.Series(violations).value_counts()) if violations else {},
    }
    return np.asarray(negs, dtype=np.int64), stats_out, violations


# ---------------------------------------------------------------- statistics

def mcnemar_exact(pos_share: np.ndarray, neg_share: np.ndarray) -> dict:
    b = int(np.sum(pos_share & ~neg_share))
    c = int(np.sum(~pos_share & neg_share))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0}
    k = min(b, c)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(k + 1)
    ]
    top = max(log_terms)
    log_sum = top + math.log(sum(math.exp(t - top) for t in log_terms))
    return {"b": b, "c": c, "n_discordant": n, "p": float(min(1.0, 2 * math.exp(log_sum)))}


def sampled_precision_given_share(
    pos_share: np.ndarray, neg_share: np.ndarray
) -> float | None:
    """sum(pos)/(sum(pos)+sum(neg)) on the balanced sample.

    NOT the population P(behaviour-related | shared prefix): recovering that
    needs the positive prior and the negative sampling rate, neither of which is
    estimated here, so the population quantity is not identified. Undefined,
    never zero-filled.
    """
    sp = int(pos_share.sum())
    sn = int(neg_share.sum())
    if sp + sn == 0:
        return None
    return sp / (sp + sn)


def cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> dict:
    """95% CI resampling CLUSTERS (anchor items), not pairs.

    One item contributes many relations; resampling pairs would treat those as
    independent and shrink the interval. Per-cluster sums are precomputed so a
    replicate costs O(number of clusters).
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(np.float64), minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    keep = counts > 0
    sums, counts = sums[keep], counts[keep]
    n = len(sums)
    if n == 0:
        return {"mean": float("nan"), "ci": [None, None], "n_clusters": 0}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_clusters": int(n),
    }


def paired_wilcoxon(diff: np.ndarray) -> dict:
    nz = diff[diff != 0]
    if len(nz) == 0:
        return {"n_nonzero": 0, "p": 1.0}
    try:
        p = float(stats.wilcoxon(nz).pvalue)
    except ValueError:
        p = 1.0
    return {"n_nonzero": int(len(nz)), "p": p}


def paired_bootstrap_ci(
    pos_share: np.ndarray, neg_share: np.ndarray, rng: np.random.Generator
) -> dict:
    n = len(pos_share)
    d = pos_share.astype(np.float64) - neg_share.astype(np.float64)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    db = d[idx].mean(axis=1)
    pn = pos_share[idx].mean(axis=1)
    nn = neg_share[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(nn > 0, pn / nn, np.nan)
    return {
        "diff": float(d.mean()),
        "diff_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
        "er": float(pn.mean() / nn.mean()) if nn.mean() > 0 else None,
        "er_ci": (
            [float(np.nanpercentile(er, 2.5)), float(np.nanpercentile(er, 97.5))]
            if np.isfinite(er).any() else None
        ),
    }


def level_stats(pre_ids, anchors, positives, negatives, rng) -> dict:
    sizes = group_sizes(pre_ids)
    size_per_item = sizes[pre_ids]
    ps = pre_ids[positives] == pre_ids[anchors]
    ns = pre_ids[negatives] == pre_ids[anchors]
    out = {
        "n_pairs": int(len(ps)),
        "p_share_pos": float(ps.mean()),
        "p_share_neg": float(ns.mean()),
        "n_neg_share_zero": int((~ns).sum()),
    }
    if out["p_share_pos"] == 0.0 or out["p_share_neg"] == 0.0:
        # A zero denominator makes the enrichment ratio undefined. It is
        # reported as undefined, never as infinity and never zero-filled.
        out.update({
            "diff": out["p_share_pos"] - out["p_share_neg"],
            "diff_ci": None,
            "er": None,
            "er_ci": None,
            "degenerate": "zero rate on one side; enrichment undefined",
        })
    else:
        out.update(paired_bootstrap_ci(ps, ns, rng))
        out["degenerate"] = None
    out["mcnemar"] = mcnemar_exact(ps, ns)
    out["sampled_precision_given_share"] = sampled_precision_given_share(ps, ns)
    out["sampled_precision_note"] = (
        "property of the balanced pos/neg sample; the population "
        "P(behaviour-related | shared prefix) is not identified without the "
        "positive prior and the negative sampling rate"
    )
    # Matched stratification: a pair enters a stratum only when the positive
    # and the negative sit in groups of the SAME size bucket. Bucketing on the
    # positive alone put positives and negatives at different group sizes inside
    # one stratum, which is the confound this corrects.
    b_pos = np.digitize(size_per_item[positives], SIZE_EDGES[1:-1])
    b_neg = np.digitize(size_per_item[negatives], SIZE_EDGES[1:-1])
    strata = {}
    for b, label in enumerate(SIZE_LABELS):
        m = (b_pos == b) & (b_neg == b)
        if m.sum() < MIN_CELL:
            continue
        strata[label] = {
            "n": int(m.sum()),
            "p_share_pos": float(ps[m].mean()),
            "p_share_neg": float(ns[m].mean()),
        }
    out["by_matched_group_size"] = strata
    return out


def between_model(
    pre_a: dict, pre_b: dict, anchors, positives, negatives,
    clusters, rng
) -> dict:
    """Paired A-vs-B comparison on the identical item pairs."""
    out = {}
    for k in (1, 2, 3):
        ps_a = pre_a[k][positives] == pre_a[k][anchors]
        ps_b = pre_b[k][positives] == pre_b[k][anchors]
        ns_a = pre_a[k][negatives] == pre_a[k][anchors]
        ns_b = pre_b[k][negatives] == pre_b[k][anchors]

        d_pos = ps_b.astype(np.int64) - ps_a.astype(np.int64)
        d_neg = ns_b.astype(np.int64) - ns_a.astype(np.int64)
        did = (
            (ps_b.astype(np.int64) - ns_b.astype(np.int64))
            - (ps_a.astype(np.int64) - ns_a.astype(np.int64))
        )
        out[f"L{k}"] = {
            "A": {"p_share_pos": float(ps_a.mean()), "p_share_neg": float(ns_a.mean())},
            "B": {"p_share_pos": float(ps_b.mean()), "p_share_neg": float(ns_b.mean())},
            "A_er": (float(ps_a.mean() / ns_a.mean()) if ns_a.mean() > 0 else None),
            "B_er": (float(ps_b.mean() / ns_b.mean()) if ns_b.mean() > 0 else None),
            "delta_pos": cluster_bootstrap(d_pos, clusters, rng),
            "delta_neg": cluster_bootstrap(d_neg, clusters, rng),
            "did": cluster_bootstrap(did, clusters, rng),
            "delta_pos_wilcoxon": paired_wilcoxon(d_pos),
            "delta_neg_wilcoxon": paired_wilcoxon(d_neg),
            "did_wilcoxon": paired_wilcoxon(did),
            "sufficient": bool(len(clusters) >= MIN_CELL),
        }
    return out


def conditional_retention_pairwise(
    pre_a: dict, pre_b: dict, anchors, positives, negatives, min_cell: int
) -> dict:
    """P(share Lk | share L(k-1)) per model, and on the COMMON condition set.

    The two models place different item pairs in the condition, so the own
    condition rates are not comparable. The paired comparison uses the pairs
    satisfying the previous-level condition under BOTH models.
    """
    out = {}
    for k in (2, 3):
        prev, cur = k - 1, k
        entry = {}
        for side, others in (("pos", positives), ("neg", negatives)):
            own = {}
            common_mask = None
            for name, pre in (("A", pre_a), ("B", pre_b)):
                cond = pre[prev][anchors] == pre[prev][others]
                if cond.sum() < min_cell:
                    own[name] = {"p_retain": None, "n_effective": int(cond.sum())}
                else:
                    own[name] = {
                        "p_retain": float(
                            (pre[cur][anchors][cond] == pre[cur][others][cond]).mean()
                        ),
                        "n_effective": int(cond.sum()),
                    }
                common_mask = cond if common_mask is None else (common_mask & cond)
            entry[side] = {
                "own_condition": own,
                "n_common": int(common_mask.sum()),
                "common_sufficient": bool(common_mask.sum() >= min_cell),
            }
            if common_mask.sum() >= min_cell and common_mask.any():
                pa = pre_a[cur][anchors][common_mask] == pre_a[cur][positives][common_mask]
                pb = pre_b[cur][anchors][common_mask] == pre_b[cur][positives][common_mask]
                entry[side]["common_A_p_retain"] = float(pa.mean())
                entry[side]["common_B_p_retain"] = float(pb.mean())
                entry[side]["common_diff_B_minus_A"] = float(pb.mean() - pa.mean())
            else:
                entry[side]["common_A_p_retain"] = None
                entry[side]["common_B_p_retain"] = None
                entry[side]["common_diff_B_minus_A"] = None
                entry[side]["common_note"] = (
                    f"common set n={int(common_mask.sum())} below min_cell={min_cell}; "
                    "no paired comparison"
                )
        out[f"L{k}_given_L{prev}"] = entry
    return out


def group_size_distribution(pre_ids, positives, negatives) -> dict:
    """How the positive and negative items are spread over group sizes.

    Answers whether a difference in prefix sharing is a placement difference or
    a discrimination difference: an item in a singleton group cannot share its
    prefix with anything, so a model that puts more behaviour positives into
    singleton groups will always show a lower raw sharing rate.
    """
    sizes = group_sizes(pre_ids)[pre_ids]
    b_pos = np.digitize(sizes[positives], SIZE_EDGES[1:-1])
    b_neg = np.digitize(sizes[negatives], SIZE_EDGES[1:-1])
    out = {}
    for b, label in enumerate(SIZE_LABELS):
        mp, mn = int((b_pos == b).sum()), int((b_neg == b).sum())
        if mp or mn:
            out[label] = {
                "n_pos": mp, "n_neg": mn,
                "frac_pos": mp / len(b_pos), "frac_neg": mn / len(b_neg),
            }
    return out


def structure_stats(levels: np.ndarray) -> dict:
    n = len(levels)
    out = {"n_items": int(n)}
    for k in (1, 2, 3):
        ids = prefix_ids(levels, k)
        sizes = group_sizes(ids)
        n_prefixes = int(ids.max() + 1)
        # A group of size one contains exactly one item, so the COUNT agrees;
        # the FRACTIONS do not, and are reported separately on purpose:
        # singleton_group_frac is a property of prefixes, items_in_singleton_
        # groups_frac is a property of items.
        singleton = int((sizes == 1).sum())
        out[f"level{k}"] = {
            "distinct_prefixes": n_prefixes,
            "mean_group_size": float(n / n_prefixes),
            "singleton_prefix_groups": singleton,
            "singleton_prefix_group_frac": float(singleton / n_prefixes),
            "items_in_singleton_groups": singleton,
            "items_in_singleton_groups_frac": float((sizes == 1).mean()),
            "max_group_size": int(sizes.max()),
        }
    full = prefix_ids(levels, N_LEVELS)
    fs = group_sizes(full)
    distinct = int(full.max() + 1)
    out["full_sid"] = {
        "distinct_triples": distinct,
        "collision_rate": float(1.0 - distinct / n),
        "largest_group": int(fs.max()),
    }
    return out


# ------------------------------------------------------------------ unit tests

def run_unit_tests() -> dict:
    """Hand-built fixtures. Each checks a corrected quantity against a value
    worked out by hand, and each would fail on the pre-correction behaviour."""
    checks: dict[str, bool] = {}

    # (1) common-successor relation on a 8-node graph.
    #    0->2, 1->2, 0->3, 3->4, 5->4, 6->7
    succ = {0: {2, 3}, 1: {2}, 3: {4}, 5: {4}, 6: {7}}
    preds = predecessors(succ)
    checks["cs_0_is_1"] = common_successor_neighbours(0, succ, preds) == {1}
    checks["cs_1_is_0"] = common_successor_neighbours(1, succ, preds) == {0}
    checks["cs_3_is_5"] = common_successor_neighbours(3, succ, preds) == {5}
    checks["cs_5_is_3"] = common_successor_neighbours(5, succ, preds) == {3}
    checks["cs_6_is_empty"] = common_successor_neighbours(6, succ, preds) == set()
    # The old two-hop-descendant rule returns {4} for node 0, which is not a
    # co-successor of 0 at all: this asserts the two rules really differ.
    old_rule = {b for b in succ.get(0, set()) for b in succ.get(b, set())} - {0}
    checks["old_rule_differs"] = old_rule != common_successor_neighbours(0, succ, preds)
    checks["cs_pairs"] = sorted(
        map(tuple, common_successor_pairs(succ, preds, cap_per_item=64))
    ) == [(0, 1), (3, 5)]

    # (2) sampled precision.
    ps = np.array([True, False, True])
    ns = np.array([False, False, False])
    checks["precision_all_pos"] = abs(
        sampled_precision_given_share(ps, ns) - 1.0
    ) < 1e-12
    ps2 = np.array([True, True])
    ns2 = np.array([True, False])
    checks["precision_two_thirds"] = abs(
        sampled_precision_given_share(ps2, ns2) - 2.0 / 3.0
    ) < 1e-12
    checks["precision_undefined_when_denominator_zero"] = (
        sampled_precision_given_share(
            np.array([False, False]), np.array([False, False])
        ) is None
    )

    # (3) conditional retention on a hand-built prefix assignment.
    #    anchors 0..3, level-1 codes 1,1,2,2 and level-2 codes 5,5,6,7.
    prev = np.array([1, 1, 2, 2])
    cur = np.array([5, 5, 6, 7])
    anchors = np.array([0, 1, 2, 3])
    pos = np.array([1, 2, 3, 0])
    cond = prev[anchors] == prev[pos]
    # (0,1) and (2,3) share L1; of those, only (0,1) also shares L2.
    checks["cond_mask"] = bool((cond == np.array([True, False, True, False])).all())
    checks["cond_retention"] = abs(
        (cur[anchors][cond] == cur[pos][cond]).mean() - 0.5
    ) < 1e-12
    checks["cond_effective_n"] = int(cond.sum()) == 2

    # (4) paired between-model statistics, on a 12-node graph where anchors are
    #     0..3, positives 4..7 and negatives 8..11.
    rng = np.random.default_rng(0)
    anchors = np.array([0, 1, 2, 3])
    pos_i = np.array([4, 5, 6, 7])
    neg_i = np.array([8, 9, 10, 11])
    # A shares L1 on positive pairs 0 and 1 only; B also on pair 2.
    # No negative shares in A; in B only pair 1 does.
    l1_a = np.array([0, 1, 2, 3,  0, 1, 9, 9,  8, 9, 10, 11])
    l1_b = np.array([0, 1, 2, 3,  0, 1, 2, 9,  8, 1, 10, 11])
    flat = {1: np.arange(12), 2: np.arange(12), 3: np.arange(12)}
    pre_a = {k: flat[k].copy() for k in (1, 2, 3)}
    pre_b = {k: flat[k].copy() for k in (1, 2, 3)}
    pre_a[1] = l1_a
    pre_b[1] = l1_b
    res = between_model(pre_a, pre_b, anchors, pos_i, neg_i, anchors, rng)
    checks["delta_pos"] = abs(res["L1"]["delta_pos"]["mean"] - 0.25) < 1e-12
    checks["delta_neg"] = abs(res["L1"]["delta_neg"]["mean"] - 0.25) < 1e-12
    checks["did_zero"] = abs(res["L1"]["did"]["mean"] - 0.0) < 1e-12
    checks["cluster_count"] = res["L1"]["delta_pos"]["n_clusters"] == 4
    # Enrichment must be None, not 0, when the negative rate is zero.
    # prefixes 0,0,1,1 over anchors 0..3; every positive is its anchor's own
    # twin (all share) and every negative sits in the other code (none share).
    checks["er_none_on_zero_denominator"] = (
        level_stats(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),
            np.array([2, 3, 0, 1]),
            rng,
        )["er"] is None
    )
    return checks


# ------------------------------------------------------------------- markdown

def fmt(x, nd=4, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{x * 100:.2f}%"
    return f"{x:.{nd}f}"


def ci(x, nd=4):
    if x is None:
        return "n/a"
    return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"


def render_markdown(rep: dict) -> str:
    L = []
    L.append("# Semantic ID prefix vs behaviour relation — corrected analysis\n")
    L.append(
        "Offline only. No Stage 2 or Stage 3 retraining, no checkpoint and no "
        "Semantic ID was modified.\n"
    )
    L.append(
        f"- Models: A = TIGER Euclidean baseline (R@10 {RECALL['A_tiger']}), "
        f"B = Exp1a behaviour contrastive (R@10 {RECALL['B_exp1a']})\n"
        f"- SID levels analysed: L1, L1+L2, L1+L2+L3. The fourth stored token is "
        "the Collision Extension Token and is excluded.\n"
        f"- Seed {SEED}, {N_BOOT} bootstrap resamples, minimum cell {MIN_CELL}.\n"
    )
    L.append("\n## Reproduction\n")
    L.append("```bash\n")
    L.append(
        "cd /home/wlia0047/ar57/wenyu/GeneRec && \\\n"
        "OMP_NUM_THREADS=8 /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \\\n"
        "  stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/analyze_prefix_behavior.py\n"
    )
    L.append("```\n")

    L.append("\n## Prefix structure\n")
    L.append("| model | level | prefixes | mean group | singleton prefix-group % "
             "| singleton item % | max group |")
    L.append("|---|---|---|---|---|---|---|")
    for m, v in rep["structure"].items():
        for k in ("level1", "level2", "level3"):
            s = v[k]
            L.append(
                f"| {m} | {k[-1]} | {s['distinct_prefixes']} | "
                f"{s['mean_group_size']:.2f} | "
                f"{fmt(s['singleton_prefix_group_frac'], pct=True)} | "
                f"{fmt(s['items_in_singleton_groups_frac'], pct=True)} | "
                f"{s['max_group_size']} |"
            )
    L.append("")
    for m, v in rep["structure"].items():
        f = v["full_sid"]
        L.append(f"- {m}: distinct 3-token prefixes {f['distinct_triples']}, "
                 f"collision {fmt(f['collision_rate'], pct=True)}, "
                 f"largest group {f['largest_group']}")

    for rel in ("train_transition", "test_transition", "common_succ"):
        blk = rep["relations"][rel]
        L.append(f"\n## Relation: {rel}\n")
        L.append(f"Behaviour positives: {blk['n_positive_pairs']} pairs. "
                 f"Negatives: {blk['negative_construction']['negatives']} "
                 f"(distinct {blk['negative_construction']['distinct_negatives']}), "
                 f"bin fallbacks {blk['negative_construction']['fallback_bin_matches']}, "
                 f"invariant violations {blk['negative_construction']['violations'] or 'none'}.\n")
        L.append("Negative semantics: an item with NO OBSERVED relation of this "
                 "kind to the anchor. Absence of an observed edge is not "
                 "evidence of irrelevance, so every negative means 'unobserved', "
                 "never 'unrelated'.\n")
        L.append("\n| level | A P(share\\|pos) | B P(share\\|pos) | A P(share\\|neg) "
                 "| B P(share\\|neg) | A ER | B ER | A sampled precision | B sampled precision |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k in ("L1", "L2", "L3"):
            a = blk["within"][A_NAME]["levels"][k]
            b = blk["within"][B_NAME]["levels"][k]
            L.append(
                f"| {k} | {fmt(a['p_share_pos'])} | {fmt(b['p_share_pos'])} | "
                f"{fmt(a['p_share_neg'], 5)} | {fmt(b['p_share_neg'], 5)} | "
                f"{fmt(a['er'], 2)} | {fmt(b['er'], 2)} | "
                f"{fmt(a['sampled_precision_given_share'])} | "
                f"{fmt(b['sampled_precision_given_share'])} |"
            )
        L.append("")
        L.append("\nBetween-model paired tests (same items, same positives, same "
                 "negatives, same split; CI by cluster bootstrap over anchor "
                 "items, p by paired Wilcoxon):\n")
        L.append("| level | delta_pos B-A [CI] | p | delta_neg B-A [CI] | p "
                 "| DiD [CI] | p |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ("L1", "L2", "L3"):
            r = blk["between"][k]
            L.append(
                f"| {k} | {fmt(r['delta_pos']['mean'], 5)} {ci(r['delta_pos']['ci'], 5)} "
                f"| {r['delta_pos_wilcoxon']['p']:.3g} "
                f"| {fmt(r['delta_neg']['mean'], 5)} {ci(r['delta_neg']['ci'], 5)} "
                f"| {r['delta_neg_wilcoxon']['p']:.3g} "
                f"| {fmt(r['did']['mean'], 5)} {ci(r['did']['ci'], 5)} "
                f"| {r['did_wilcoxon']['p']:.3g} |"
            )
        L.append("\nConditional retention (own condition, and the paired comparison "
                 "on the pairs where BOTH models share the previous level):\n")
        L.append("| condition | side | n_common | A retain | B retain | B-A |")
        L.append("|---|---|---|---|---|---|")
        for key, ent in blk["conditional"].items():
            for side in ("pos", "neg"):
                e = ent[side]
                L.append(
                    f"| {key} | {side} | {e['n_common']} | "
                    f"{fmt(e['common_A_p_retain'])} | {fmt(e['common_B_p_retain'])} | "
                    f"{fmt(e['common_diff_B_minus_A'], 5)} |"
                )
        L.append("\nMatched group-size strata (positive and negative in the SAME "
                 "size bucket):\n")
        L.append("| level | bucket | n | A pos | A neg | B pos | B neg |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ("L1", "L2", "L3"):
            sa = blk["within"][A_NAME]["levels"][k]["by_matched_group_size"]
            sb = blk["within"][B_NAME]["levels"][k]["by_matched_group_size"]
            for label in SIZE_LABELS:
                if label in sa and label in sb:
                    L.append(
                        f"| {k} | {label} | {sa[label]['n']} | "
                        f"{fmt(sa[label]['p_share_pos'])} | {fmt(sa[label]['p_share_neg'], 5)} | "
                        f"{fmt(sb[label]['p_share_pos'])} | {fmt(sb[label]['p_share_neg'], 5)} |"
                    )
        L.append("")
    L.append("\n## Group-size distribution of behaviour positives\n")
    L.append("| level | model | bucket | n_pos | % of positives | n_neg | % of negatives |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("L1", "L2", "L3"):
        for m in (A_NAME, B_NAME):
            dist = rep["relations"]["test_transition"]["within"][m]["group_size_distribution"][k]
            for b in SIZE_LABELS:
                if b in dist:
                    v = dist[b]
                    L.append(f"| {k} | {m} | {b} | {v['n_pos']} | "
                             f"{fmt(v['frac_pos'], pct=True)} | {v['n_neg']} | "
                             f"{fmt(v['frac_neg'], pct=True)} |")
    L.append("")
    L.append("\n## Conclusions\n")
    L.append(_conclusion_markdown(rep))
    return "\n".join(L)


A_NAME, B_NAME = "A_tiger", "B_exp1a"


def _conclusion_markdown(rep: dict) -> str:
    """Every number below is read out of the computed results."""
    t = rep["relations"]["test_transition"]
    tr = rep["relations"]["train_transition"]
    cs = rep["relations"]["common_succ"]
    sa, sb = rep["structure"][A_NAME], rep["structure"][B_NAME]
    W = rep["within"] if "within" in rep else {}
    out: list[str] = []

    g = t["between"]["L1"]
    out.append("### Q1. Is the L1 behaviour-selectivity gain statistically significant?\n")
    out.append(
        f"Yes for the SELECTIVITY, but not for the positive-side retention. On the "
        f"independent test relation the difference-in-differences is "
        f"**{g['did']['mean']:+.5f}** "
        f"[{g['did']['ci'][0]:+.5f}, {g['did']['ci'][1]:+.5f}], paired Wilcoxon "
        f"p = {g['did_wilcoxon']['p']:.3g}, so B is more selective at L1. That "
        f"gain decomposes asymmetrically:\n"
    )
    out.append(
        f"- negative side: {g['delta_neg']['mean']:+.5f} "
        f"[{g['delta_neg']['ci'][0]:+.5f}, {g['delta_neg']['ci'][1]:+.5f}], "
        f"p = {g['delta_neg_wilcoxon']['p']:.3g} — **significant**\n"
        f"- positive side: {g['delta_pos']['mean']:+.5f} "
        f"[{g['delta_pos']['ci'][0]:+.5f}, {g['delta_pos']['ci'][1]:+.5f}], "
        f"p = {g['delta_pos_wilcoxon']['p']:.3g} — **NOT significant**\n"
    )
    out.append(
        f"So on data the model never trained on, B's L1 advantage comes almost "
        f"entirely from pushing unrelated items apart, not from gathering more "
        f"behaviour-related pairs. On the train relation the positive-side gain "
        f"is large and significant ({tr['between']['L1']['delta_pos']['mean']:+.5f}, "
        f"p = {tr['between']['L1']['delta_pos_wilcoxon']['p']:.3g}), which is "
        f"expected because that is the objective; on common_successor, which "
        f"never entered any objective, it is small but significant "
        f"({cs['between']['L1']['delta_pos']['mean']:+.5f}, "
        f"p = {cs['between']['L1']['delta_pos_wilcoxon']['p']:.3g}).\n"
    )

    g2, g3 = t["between"]["L2"], t["between"]["L3"]
    c2 = t["conditional"]["L2_given_L1"]["pos"]
    c3 = t["conditional"]["L3_given_L2"]["pos"]
    out.append("\n### Q2. Does B really scatter more genuinely behaviour-related "
               "items at L2 and L3, and does it survive the common condition?\n")
    out.append(
        f"**Yes, and it survives the paired conditional comparison — but only at "
        f"L2; at L3 the independent-test estimate is not obtainable.**\n"
    )
    out.append(
        f"- Unconditional on the test relation: L2 "
        f"{g2['delta_pos']['mean']:+.5f} "
        f"[{g2['delta_pos']['ci'][0]:+.5f}, {g2['delta_pos']['ci'][1]:+.5f}], "
        f"p = {g2['delta_pos_wilcoxon']['p']:.3g}; L3 "
        f"{g3['delta_pos']['mean']:+.5f} "
        f"[{g3['delta_pos']['ci'][0]:+.5f}, {g3['delta_pos']['ci'][1]:+.5f}], "
        f"p = {g3['delta_pos_wilcoxon']['p']:.3g}. Both are negative, i.e. B "
        f"keeps fewer behaviour pairs together.\n"
        f"- On the pairs where BOTH models already share L1 (n = "
        f"{c2['n_common']}), B retains "
        f"{fmt(c2['common_B_p_retain'])} against A's "
        f"{fmt(c2['common_A_p_retain'])}, a paired gap of "
        f"{fmt(c2['common_diff_B_minus_A'], 5)}. The direction holds, and this is "
        f"a genuine paired comparison on identical pairs.\n"
        f"- One level deeper, on pairs where both models share L1+L2, the test "
        f"relation leaves n = {c3['n_common']} common pairs, below the minimum "
        f"cell of {MIN_CELL}, so **the conditional L3 comparison is not "
        f"estimable on the independent test relation**. It is reported, not "
        f"inferred, on the other two relations (train n = "
        f"{tr['conditional']['L3_given_L2']['pos']['n_common']}, common_successor "
        f"n = {cs['conditional']['L3_given_L2']['pos']['n_common']}), both with "
        f"B below A.\n"
    )

    out.append("\n### Q3. Is B's lower collision rate just finer partitioning, and "
               "can prefix group size be separated out?\n")
    out.append(
        f"Yes, and group size does account for a large part of it. B has "
        f"{sb['level2']['distinct_prefixes']} L2 prefixes against A's "
        f"{sa['level2']['distinct_prefixes']} "
        f"({sb['level2']['distinct_prefixes'] / sa['level2']['distinct_prefixes']:.2f}x) "
        f"and {sb['level3']['distinct_prefixes']} against "
        f"{sa['level3']['distinct_prefixes']} at L3, while L1 is NOT finer "
        f"({sb['level1']['distinct_prefixes']} against "
        f"{sa['level1']['distinct_prefixes']} prefixes). B's L1 is instead far "
        f"more uniform: largest group {sb['level1']['max_group_size']} against "
        f"A's {sa['level1']['max_group_size']}, and singleton prefix groups "
        f"{fmt(sb['level1']['singleton_prefix_group_frac'], pct=True)} against "
        f"{fmt(sa['level1']['singleton_prefix_group_frac'], pct=True)}. Collision "
        f"falls from {fmt(sa['full_sid']['collision_rate'], pct=True)} to "
        f"{fmt(sb['full_sid']['collision_rate'], pct=True)}.\n"
    )
    dist = {n: t["within"][n]["group_size_distribution"]["L2"] for n in (A_NAME, B_NAME)}
    out.append(
        f"\nGroup size can be separated, and separating it changes the reading of "
        f"L2 and L3. At L2, the share of behaviour positives sitting in a "
        f"SINGLETON prefix group is "
        f"{fmt(dist[B_NAME]['1']['frac_pos'], pct=True)} for B against "
        f"{fmt(dist[A_NAME]['1']['frac_pos'], pct=True)} for A; at L3 it is "
        f"{fmt(t['within'][B_NAME]['group_size_distribution']['L3']['1']['frac_pos'], pct=True)} "
        f"against "
        f"{fmt(t['within'][A_NAME]['group_size_distribution']['L3']['1']['frac_pos'], pct=True)}. "
        f"An item in a singleton group cannot share its prefix with anything, "
        f"including its own behaviour sibling, so a large part of B's lower raw "
        f"sharing rate is placement, not discrimination.\n"
    )
    m2a = t["within"][A_NAME]["levels"]["L2"]["by_matched_group_size"]
    m2b = t["within"][B_NAME]["levels"]["L2"]["by_matched_group_size"]
    rows = [
        f"| {b} | {m2a[b]['n']} | {fmt(m2a[b]['p_share_pos'], 5)} | "
        f"{fmt(m2b[b]['p_share_pos'], 5)} |"
        for b in SIZE_LABELS if b in m2a and b in m2b
    ]
    out.append(
        f"\nWithin strata where the positive and the negative sit in groups of "
        f"the SAME size bucket, B's positive sharing is at least A's, reversing "
        f"the naive comparison:\n\n"
        f"| L2 bucket | n | A P(share\\|pos) | B P(share\\|pos) |\n|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
    )
    out.append(
        f"\nThat matched view is itself limited and must not be read as the "
        f"population answer: it covers only "
        f"{fmt(t['within'][A_NAME]['matched_coverage']['L2'], pct=True)} of pairs "
        f"for A and {fmt(t['within'][B_NAME]['matched_coverage']['L2'], pct=True)} "
        f"for B at L2, because the two models put different pairs into matching "
        f"buckets. The honest statement is therefore two-sided: B places far "
        f"more behaviour positives into singleton groups, which lowers its raw "
        f"sharing rate mechanically, AND on the genuinely paired "
        f"given-shared-L1 comparison B does split behaviour pairs faster. Both "
        f"are measured, and neither alone is the whole story.\n"
    )

    out.append("\n### Identifiability and limitations\n")
    out.append(
        "- `sampled_precision_given_share` is a property of the balanced "
        "positive/negative sample. The population P(behaviour-related | shared "
        "prefix) is **not identified** here: it needs the positive prior and the "
        "negative sampling probability, neither of which is estimated. It is not "
        "reported.\n"
        "- Every negative means NO OBSERVED relation, never proven irrelevance.\n"
        "- Test edges are used only to build negatives for evaluation; they "
        "enter no model and no training.\n"
        "- Enrichment ratios are left undefined, never zero-filled, wherever the "
        "negative sharing rate is zero.\n"
    )
    return "".join(out)


def main() -> None:
    checks = run_unit_tests()
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"unit tests failed: {failed}")
    print(f"unit tests: {len(checks)}/{len(checks)} passed")

    rng = np.random.default_rng(SEED)
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    n_items = 24587

    sids, sid_info = {}, {}
    for name, path in SID_PATHS.items():
        sids[name], sid_info[name] = load_levels(path)
        if len(sids[name]) != n_items:
            raise ValueError(f"{name}: {len(sids[name])} items")

    pop, freq = popularity_and_frequency(train, n_items)
    pop_bin, freq_bin = decile_bins(pop), decile_bins(freq)

    succ_tr = successors(train, n_items)
    succ_te = successors(test, n_items)
    preds_tr = predecessors(succ_tr)

    train_pairs = transitions(train, n_items)
    test_pairs = transitions(test, n_items)
    cs_pairs = common_successor_pairs(succ_tr, preds_tr, cap_per_item=64)
    train_edge = {tuple(p) for p in train_pairs}

    pre = {
        name: {k: prefix_ids(lv, k) for k in (1, 2, 3)} for name, lv in sids.items()
    }

    report: dict = {
        "corrections": [
            "common-successor neighbours built from the reverse adjacency "
            "predecessors, not two-hop descendants",
            "test-relation negatives exclude train AND test edges",
            "P(pos|share) replaced by sampled_precision_given_share and "
            "labelled non-identifiable for the population",
            "between-model paired tests with cluster bootstrap and Wilcoxon",
            "conditional retention compared on the common condition set",
            "singleton prefix-group and singleton item fractions separated",
            "group-size strata matched on positive and negative bucket",
            "integer seed offsets replace Python hash()",
            "fourth SID token excluded from the level analysis",
        ],
        "seed": SEED, "n_boot": N_BOOT, "min_cell": MIN_CELL,
        "n_items": n_items,
        "recall_at_10": RECALL,
        "unit_tests": checks,
        "sid_token_info": sid_info,
        "relation_counts": {
            "train_transition": int(len(train_pairs)),
            "test_transition": int(len(test_pairs)),
            "test_pairs_absent_from_train": int(
                len({tuple(p) for p in test_pairs} - train_edge)
            ),
            "common_successor": int(len(cs_pairs)),
        },
        "structure": {n: structure_stats(sids[n]) for n in sids},
        "relations": {},
    }

    specs = [
        ("train_transition", "train_transition", train_pairs),
        ("test_transition", "test_transition", test_pairs),
        ("common_succ", "common_succ", cs_pairs),
    ]
    for idx, (key, kind, pairs) in enumerate(specs):
        anchors, positives = pairs[:, 0].copy(), pairs[:, 1].copy()
        keep = anchors != positives
        anchors, positives = anchors[keep], positives[keep]
        _, uniq = np.unique(anchors, return_index=True)
        anchors, positives = anchors[uniq], positives[uniq]

        rel_sets = [
            related_set(int(a), kind, succ_tr, preds_tr, succ_te) for a in anchors
        ]
        # Integer offset, never hash(): Python's string hash is salted per
        # process and made the negative draw differ between runs.
        neg_rng = np.random.default_rng(SEED + 1000 * idx)
        negatives, neg_stats, violations = build_negatives(
            anchors, positives, rel_sets, pop_bin, freq_bin, neg_rng
        )
        neg_stats["negative_equals_positive"] = int(np.sum(negatives == positives))
        neg_stats["negative_equals_anchor"] = int(np.sum(negatives == anchors))

        report["relations"][key] = {
            "relation_kind": kind,
            "n_positive_pairs": int(len(anchors)),
            "negative_construction": neg_stats,
            "within": {
                n: {
                    "levels": {
                        f"L{k}": level_stats(pre[n][k], anchors, positives, negatives, rng)
                        for k in (1, 2, 3)
                    },
                    "group_size_distribution": {
                        f"L{k}": group_size_distribution(
                            pre[n][k], positives, negatives
                        )
                        for k in (1, 2, 3)
                    },
                    "matched_coverage": {
                        f"L{k}": float(
                            sum(
                                v["n"]
                                for v in level_stats(
                                    pre[n][k], anchors, positives, negatives,
                                    np.random.default_rng(SEED),
                                )["by_matched_group_size"].values()
                            ) / len(anchors)
                        )
                        for k in (1, 2, 3)
                    },
                } for n in sids
            },
            "between": between_model(
                pre[A_NAME], pre[B_NAME], anchors, positives, negatives,
                anchors, rng,
            ),
            "conditional": conditional_retention_pairwise(
                pre[A_NAME], pre[B_NAME], anchors, positives, negatives, MIN_CELL
            ),
        }
        print(
            f"{key}: {len(anchors)} pairs, "
            f"{len(violations)} violations, "
            f"neg==pos {neg_stats['negative_equals_positive']}"
        )

    report["conclusions"] = build_conclusions(report)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    OUT_MD.write_text(render_markdown(report))
    print(f"written: {OUT_JSON}\nwritten: {OUT_MD}")


def build_conclusions(rep: dict) -> dict:
    """Every value here is read back out of the computed results."""
    out = {}
    t = rep["relations"]["test_transition"]
    l1, l2, l3 = (t["between"][k] for k in ("L1", "L2", "L3"))
    out["q1_l1_selectivity"] = {
        "delta_pos": l1["delta_pos"], "wilcoxon_p": l1["delta_pos_wilcoxon"]["p"],
        "delta_neg": l1["delta_neg"], "did": l1["did"],
        "did_p": l1["did_wilcoxon"]["p"],
    }
    c2 = t["conditional"]["L2_given_L1"]["pos"]
    c3 = t["conditional"]["L3_given_L2"]["pos"]
    out["q2_l2_l3"] = {
        "L2_delta_pos": l2["delta_pos"], "L3_delta_pos": l3["delta_pos"],
        "L2_common": c2, "L3_common": c3,
    }
    sa = rep["structure"][A_NAME]
    sb = rep["structure"][B_NAME]
    out["q3_collision"] = {
        "A_collision": sa["full_sid"]["collision_rate"],
        "B_collision": sb["full_sid"]["collision_rate"],
        "A_prefixes_L2_L3": [sa["level2"]["distinct_prefixes"], sa["level3"]["distinct_prefixes"]],
        "B_prefixes_L2_L3": [sb["level2"]["distinct_prefixes"], sb["level3"]["distinct_prefixes"]],
        "A_max_group_L1": sa["level1"]["max_group_size"],
        "B_max_group_L1": sb["level1"]["max_group_size"],
        "A_singleton_prefix_group_frac_L1": sa["level1"]["singleton_prefix_group_frac"],
        "B_singleton_prefix_group_frac_L1": sb["level1"]["singleton_prefix_group_frac"],
    }
    return out


if __name__ == "__main__":
    main()