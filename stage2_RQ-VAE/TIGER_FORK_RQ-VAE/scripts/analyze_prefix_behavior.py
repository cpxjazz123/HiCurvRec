"""Offline analysis: do Semantic ID prefixes keep item behaviour relations?

Two existing models, compared on identical SID inputs, identical relation sets
and identical negatives. Nothing is retrained and no checkpoint is touched.

  A  TIGER Euclidean baseline        results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json
  B  Exp1a behaviour contrastive     results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json

Both SIDs are [L1, L2, L3, collision_token]. Only the first three levels are
analysed: the fourth entry is the Collision Extension Token, not a quantisation
level, and treating it as one would inflate every "full SID" count.

Relation types are kept strictly apart, because they answer different questions:

  transition   history[-1] -> target, from train.parquet. The supervision the
               contrastive model actually trained on.
  test_transition  history[-1] -> target, from test.parquet. Stage 2 never saw
               test.parquet, so this is an independent measurement and not a
               re-read of the training signal.
  common_succ  pairs (a,b) with some c such that a -> c and b -> c, built from
               train only. Reported separately and never pooled with the above.

A negative is an item with NO OBSERVED relation of the matching type to the
anchor. That is a sampling statement, not a semantic one: absence of an observed
edge is not evidence of irrelevance, only of not having been witnessed. Every
negative here is therefore "unobserved", never "unrelated", and the limitation is
carried into the conclusions rather than dropped.

Negatives are matched to their positive on two nuisance axes so that a difference
in prefix sharing is not just a popularity artefact: how often the item appears
as a target, and how often it appears anywhere in the training data. Both are
ranked into deciles and a negative is drawn from the same cell as its positive,
falling back to the nearest non-empty cell and counting the fallbacks.

Statistics. The positive and the negative of each anchor form a matched pair, so
the comparison is paired throughout: paired bootstrap over pairs for the
confidence intervals, and McNemar's exact test on the discordant pairs for the
p-value. Rates of exactly zero and cells below a minimum count are reported
explicitly rather than being silently folded into an aggregate.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261011
N_LEVELS = 3
MIN_CELL = 30
ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SID_PATHS = {
    "A_tiger": ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
    "B_exp1a": ROOT
    / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
}
DATA = ROOT / "results/stage0_build_parquet"


# ---------------------------------------------------------------- SID helpers

def load_levels(path: Path) -> np.ndarray:
    """(N, 3) integer codes, level 0 = L1. The 4th token is dropped."""
    raw = json.loads(path.read_text())
    keys = sorted(int(k) for k in raw)
    tokens = np.asarray([raw[str(k)] for k in keys], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] < N_LEVELS:
        raise ValueError(f"{path}: unexpected SID shape {tokens.shape}")
    return tokens[:, :N_LEVELS]


def prefix_ids(levels: np.ndarray, k: int) -> np.ndarray:
    """Dense id for the first k levels."""
    return np.unique(levels[:, :k], axis=0, return_inverse=True)[1].astype(np.int64)


def group_sizes(ids: np.ndarray) -> np.ndarray:
    return np.bincount(ids).astype(np.int64)


# ------------------------------------------------------------ relation build

def transitions(frame: pd.DataFrame, n_items: int) -> np.ndarray:
    """Unique (history[-1] -> target) pairs, both directions of the edge kept
    only as observed; the pair is what matters, not the multiplicity."""
    seen = set()
    for history, target in zip(
        frame["seen_history"].to_numpy(), frame["target"].to_numpy()
    ):
        if history is None or len(history) == 0:
            continue
        a = int(history[-1])
        b = int(target)
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
        a = int(history[-1])
        b = int(target)
        if 0 <= a < n_items and 0 <= b < n_items:
            succ[a].add(b)
    return dict(succ)


def common_successor_pairs(
    succ: dict[int, set[int]], n_items: int, cap_per_item: int
) -> np.ndarray:
    """Pairs (a, b) sharing at least one successor, from train only.

    Built by inverting: every item c contributes all unordered pairs of its
    predecessors. Predecessor sets are huge, so each c contributes at most
    ``cap_per_item`` pairs, drawn with a fixed generator, and the cap is
    reported rather than hidden.
    """
    rng = np.random.default_rng(SEED)
    preds: dict[int, set[int]] = defaultdict(set)
    for a, bs in succ.items():
        for b in bs:
            preds[b].add(a)
    out: set[tuple[int, int]] = set()
    for _, pres in preds.items():
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


def popularity_and_frequency(
    frame: pd.DataFrame, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
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
    """Rank-based bins so heavy ties do not collapse the control."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * n_bins // max(len(values), 1)).clip(0, n_bins - 1)


def related_set(anchor: int, relation: str, succ: dict, train_pairs: np.ndarray,
                common_succ: set) -> set[int]:
    if relation == "transition":
        return succ.get(anchor, set())
    if relation == "common_succ":
        out: set[int] = set()
        for b in succ.get(anchor, set()):
            out |= {c for c in succ.get(b, set())}
        out.discard(anchor)
        return out
    raise ValueError(relation)


def build_negatives(
    anchors: np.ndarray,
    positives: np.ndarray,
    related: list[set[int]],
    pop_bin: np.ndarray,
    freq_bin: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """One negative per positive, matched on popularity and frequency decile."""
    by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    for it in range(len(pop_bin)):
        by_cell[(int(pop_bin[it]), int(freq_bin[it]))].append(it)

    negs, fallbacks = [], 0
    for (a, b), rel in zip(zip(anchors, positives), related):
        a, b = int(a), int(b)
        key = (int(pop_bin[b]), int(freq_bin[b]))
        candidates = None
        for step in range(0, 11):
            if step == 0:
                cells = [key]
            else:
                # Widen to neighbouring deciles, closest first. Built as a flat
                # list of (cell, candidates) pairs: nesting a one-element tuple
                # inside the loop made the loop unpack the wrong object.
                cells = []
                for cb, fb in (
                    (key[0] - step, key[1]),
                    (key[0], key[1] - step),
                    (min(key[0] + step, 9), key[1]),
                    (key[0], min(key[1] + step, 9)),
                ):
                    if 0 <= cb < 10 and 0 <= fb < 10 and (cb, fb) not in cells:
                        cells.append((cb, fb))
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
        negs.append(int(candidates[rng.integers(len(candidates))]))
    stats = {
        "negatives": len(negs),
        "fallback_bin_matches": fallbacks,
        "distinct_negatives": len(set(negs)),
    }
    return np.asarray(negs, dtype=np.int64), stats


# ------------------------------------------------------------------- metrics

def mcnemar_exact(pos_share: np.ndarray, neg_share: np.ndarray) -> dict:
    """Exact binomial test on the discordant matched pairs.

    Summed in log space: with tens of thousands of discordant pairs the
    binomial coefficients overflow a float long before they become negligible.
    """
    b = int(np.sum(pos_share & ~neg_share))
    c = int(np.sum(~pos_share & neg_share))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0}
    k = min(b, c)
    # Per-term log probabilities of Binomial(n, 0.5) at 0..k, then a max-shifted
    # log-sum-exp. Summing the logs directly would accumulate the binomial
    # coefficients instead of the probabilities and overflow upward.
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(k + 1)
    ]
    top = max(log_terms)
    log_sum = top + math.log(sum(math.exp(t - top) for t in log_terms))
    p = min(1.0, 2.0 * math.exp(log_sum))
    return {"b": b, "c": c, "n_discordant": n, "p": float(p)}


def paired_bootstrap(
    pos_share: np.ndarray,
    neg_share: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """CI for the difference and for the enrichment ratio, resampling pairs."""
    n = len(pos_share)
    d = pos_share.astype(np.float64) - neg_share.astype(np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    db = d[idx].mean(axis=1)
    pn = pos_share[idx].mean(axis=1)
    nn = neg_share[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(nn > 0, pn / nn, np.nan)
    return {
        "diff": float(d.mean()),
        "diff_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
        "er": float(pn.mean() / nn.mean()) if nn.mean() > 0 else float("nan"),
        "er_ci": [
            float(np.nanpercentile(er, 2.5)),
            float(np.nanpercentile(er, 97.5)),
        ],
        "p_share_pos": float(pos_share.mean()),
        "p_share_neg": float(neg_share.mean()),
    }


def level_stats(
    pre_ids: np.ndarray, anchors: np.ndarray, pos: np.ndarray, neg: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    sizes = group_sizes(pre_ids)
    size_per_item = sizes[pre_ids]
    ps = pre_ids[pos] == pre_ids[anchors]
    ns = pre_ids[neg] == pre_ids[anchors]
    out = {
        "n_pairs": int(len(ps)),
        "p_share_pos": float(ps.mean()),
        "p_share_neg": float(ns.mean()),
        "n_pos_share_zero_cells": int(np.sum(ps & ~ns)),
    }
    if out["p_share_pos"] == 0.0 or out["p_share_neg"] == 0.0:
        out["er"] = None
        out["er_ci"] = None
        out["diff"] = out["p_share_pos"] - out["p_share_neg"]
        out["diff_ci"] = None
        out["mcnemar"] = mcnemar_exact(ps, ns)
        out["degenerate"] = "zero rate on one side"
    else:
        out.update(paired_bootstrap(ps, ns, n_boot, rng))
        out["mcnemar"] = mcnemar_exact(ps, ns)
        out["degenerate"] = None
    both = ps | ns
    out["p_pos_given_shared"] = (
        float(ps[both].mean()) if both.any() else None
    )
    # Stratify the sharing decision by the size of the prefix group the pair
    # lands in, to test whether any enrichment is only a group-size effect.
    strata = {}
    edges = [0, 2, 5, 10, 25, 100, np.inf]
    labels = ["1", "2-4", "5-9", "10-24", "25-99", "100+"]
    buckets = np.digitize(size_per_item[pos], edges[1:-1], right=False)
    for b, label in enumerate(labels):
        m = buckets == b
        if m.sum() < MIN_CELL:
            continue
        s_ps, s_ns = ps[m], ns[m]
        strata[f"size_{label}"] = {
            "n": int(m.sum()),
            "p_share_pos": float(s_ps.mean()),
            "p_share_neg": float(s_ns.mean()),
        }
    out["by_group_size"] = strata
    return out


def conditional_retention(
    pre_prev: np.ndarray, pre_cur: np.ndarray,
    anchors: np.ndarray, pos_items: np.ndarray, neg_items: np.ndarray,
) -> dict:
    """Retention of level k among pairs that ALREADY share level k-1.

    Without this, a level looks informative simply because the previous level
    kept fewer pairs together. Conditioning on the previous level removes that:
    only pairs sharing L1 are asked whether they still share L2, and only pairs
    sharing L1+L2 are asked whether they still share L3.

    ``pos_items`` / ``neg_items`` are arrays of item ids aligned with
    ``anchors``, not boolean masks.
    """
    out = {}
    for name, others in (("pos", pos_items), ("neg", neg_items)):
        same_prev = pre_prev[anchors] == pre_prev[others]
        a, b = anchors[same_prev], others[same_prev]
        if len(a) < MIN_CELL:
            out[f"p_retain_{name}"] = None
            out[f"n_effective_{name}"] = int(len(a))
            continue
        out[f"p_retain_{name}"] = float((pre_cur[a] == pre_cur[b]).mean())
        out[f"n_effective_{name}"] = int(len(a))
    if out.get("p_retain_pos") is not None and out.get("p_retain_neg") is not None:
        out["retention_diff"] = out["p_retain_pos"] - out["p_retain_neg"]
        out["retention_er"] = (
            out["p_retain_pos"] / out["p_retain_neg"]
            if out["p_retain_neg"] > 0 else None
        )
    else:
        out["retention_diff"] = None
        out["retention_er"] = None
    return out


def structure_stats(levels: np.ndarray) -> dict:
    n = len(levels)
    out = {"n_items": int(n)}
    for k in (1, 2, 3):
        ids = prefix_ids(levels, k)
        sizes = group_sizes(ids)
        n_prefixes = int(ids.max() + 1)
        # len(np.unique(sizes)) counts DISTINCT GROUP SIZES, not groups; the
        # number of singleton groups is the count of sizes equal to one.
        singleton = int((sizes == 1).sum())
        out[f"level{k}"] = {
            "distinct_prefixes": n_prefixes,
            "mean_group_size": float(n / n_prefixes),
            "singleton_groups": singleton,
            "singleton_group_frac": float(singleton / n_prefixes),
            "items_in_singletons": singleton,
            "items_in_singletons_frac": float((sizes == 1).mean()),
            "max_group_size": int(sizes.max()),
        }
    full = prefix_ids(levels, N_LEVELS)
    fs = group_sizes(full)
    distinct = int(full.max() + 1)
    out["full_sid"] = {
        "distinct_triples": distinct,
        "collision_rate": float(1.0 - distinct / n),
        "largest_group": int(fs.max()),
        "singleton_triples": int((fs == 1).sum()),
    }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_boot = 2000

    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    n_items = 24587

    sids = {name: load_levels(p) for name, p in SID_PATHS.items()}
    for name, lv in sids.items():
        if len(lv) != n_items:
            raise ValueError(f"{name}: {len(lv)} items, expected {n_items}")

    pop, freq = popularity_and_frequency(train, n_items)
    pop_bin, freq_bin = decile_bins(pop), decile_bins(freq)

    succ = successors(train, n_items)
    train_pairs = transitions(train, n_items)
    test_pairs = transitions(test, n_items)
    cs_pairs = common_successor_pairs(succ, n_items, cap_per_item=64)

    report: dict = {
        "seed": SEED,
        "n_items": n_items,
        "n_boot": n_boot,
        "min_cell": MIN_CELL,
        "relation_counts": {
            "train_transition": int(len(train_pairs)),
            "test_transition": int(len(test_pairs)),
            "test_pairs_absent_from_train": int(
                len({tuple(p) for p in test_pairs} - {tuple(p) for p in train_pairs})
            ),
            "common_successor": int(len(cs_pairs)),
        },
        "structure": {name: structure_stats(lv) for name, lv in sids.items()},
        "levels": {},
    }

    relation_sets = {
        "train_transition": ("transition", train_pairs),
        "test_transition": ("transition", test_pairs),
        "common_succ": ("common_succ", cs_pairs),
    }

    for rel_name, (kind, pairs) in relation_sets.items():
        anchors = pairs[:, 0].copy()
        positives = pairs[:, 1].copy()
        # Drop self pairs and keep one anchor to control the multiple-testing
        # blow-up on very popular anchors.
        keep = anchors != positives
        anchors, positives = anchors[keep], positives[keep]
        _, uniq = np.unique(anchors, return_index=True)
        anchors, positives = anchors[uniq], positives[uniq]

        rel_sets = [related_set(int(a), kind, succ, train_pairs, set()) for a in anchors]
        neg_rng = np.random.default_rng(SEED + hash(rel_name) % 100000)
        negatives, neg_stats = build_negatives(
            anchors, positives, rel_sets, pop_bin, freq_bin, neg_rng
        )
        overlap = int(np.sum(negatives == positives))
        neg_stats["negative_equals_positive"] = overlap

        key = rel_name
        report["levels"][key] = {
            "relation_kind": kind,
            "n_positive_pairs": int(len(anchors)),
            "negative_construction": neg_stats,
            "models": {},
        }

        for name, lv in sids.items():
            pre = [prefix_ids(lv, k) for k in (1, 2, 3)]
            per_level = {}
            for k in (1, 2, 3):
                per_level[f"L{k}"] = level_stats(
                    pre[k - 1], anchors, positives, negatives, n_boot, rng
                )
            cond = {}
            for k in (2, 3):
                cond[f"L{k}_given_L{k-1}"] = conditional_retention(
                    pre[k - 2], pre[k - 1], anchors, positives, negatives
                )
            split = {}
            for k in (1, 2):
                split[f"L{k}->L{k+1}"] = {
                    "pos_separated": float(
                        per_level[f"L{k}"]["p_share_pos"]
                        - per_level[f"L{k+1}"]["p_share_pos"]
                    ),
                    "neg_separated": float(
                        per_level[f"L{k}"]["p_share_neg"]
                        - per_level[f"L{k+1}"]["p_share_neg"]
                    ),
                    "pos_separated_ci": per_level[f"L{k}"].get("diff_ci"),
                }
            report["levels"][key]["models"][name] = {
                "per_level": per_level,
                "conditional": cond,
                "split": split,
            }

    out = ROOT / "results/stage2_RQ-VAE/prefix_behavior_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["relation_counts"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()