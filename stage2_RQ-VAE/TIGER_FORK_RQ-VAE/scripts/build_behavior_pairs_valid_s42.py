"""The INDEPENDENT validation pair set the programme's step 7 asks for.

Same sampling rule, same seed, same negative matching as the canonical
train-based set, but the POSITIVES are drawn from valid.parquet instead of
train.parquet, so the relations the mechanisms are scored on never entered
training (training reads train.parquet only, and every mechanism arm uses the
train-based pair file for its own supervision).

  positives   all distinct history[-1] -> target relations in valid.parquet,
              filtered only for self-loops and out-of-range ids
  negatives   ten per positive, matched on the popularity decile and the
              appearance-frequency decile, excluding the anchor, the positive,
              and every item with an observed relation to that anchor in EITHER
              train.parquet OR valid.parquet -- an unobserved negative must not be
              an observed positive of the same anchor in either split

Popularity and frequency come from train.parquet, exactly as in the canonical
set, so the negative distribution stays comparable and only the positives change.

Seed 42. Nothing here is drawn twice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402

SEED = 42
OUT_DIR = MM.OUT_DIR
OUT_NPZ = OUT_DIR / "behavior_pairs_valid_s42.npz"
OUT_META = OUT_DIR / "behavior_pairs_valid_s42.meta.json"


def build_valid_relations(n_items: int) -> dict:
    frame = pd.read_parquet(AP.DATA / "valid.parquet")
    pairs = AP.transitions(frame, n_items)
    a, b = pairs[:, 0], pairs[:, 1]
    bad = (a < 0) | (a >= n_items) | (b < 0) | (b >= n_items)
    selfloop = (a == b) & ~bad
    keep = ~bad & ~selfloop
    kept = pairs[keep]
    return {
        "pairs": kept,
        "filters": {
            "rows_in_valid_parquet": int(len(frame)),
            "distinct_transitions": int(len(pairs)),
            "dropped_invalid_item_id": int(bad.sum()),
            "dropped_self_loop": int(selfloop.sum()),
            "kept_pairs": int(len(kept)),
            "distinct_anchors": int(len(np.unique(kept[:, 0]))),
            "mean_positives_per_anchor": float(
                len(kept) / max(len(np.unique(kept[:, 0])), 1)
            ),
        },
    }


def main() -> None:
    train_frame = pd.read_parquet(AP.DATA / "train.parquet")
    rel = build_valid_relations(MM.N_ITEMS)
    pairs = rel["pairs"]
    # "Completely independent" means the SCORED relation is not one the model saw:
    # a valid transition that is also a train transition of the same anchor is
    # still a relation training read, so it is dropped. What survives is a
    # held-out interaction, and the anchors may still appear in train with other
    # targets -- that is the point.
    train_pairs = {tuple(p) for p in AP.transitions(train_frame, MM.N_ITEMS).tolist()}
    keep_mask = np.array([tuple(p) not in train_pairs for p in pairs.tolist()])
    rel["filters"]["dropped_also_a_train_transition"] = int((~keep_mask).sum())
    rel["filters"]["held_out_pairs"] = int(keep_mask.sum())
    pairs = pairs[keep_mask]
    rel["filters"]["held_out_anchors"] = int(len(np.unique(pairs[:, 0])))
    anchors, positives = pairs[:, 0].copy(), pairs[:, 1].copy()

    succ_tr = AP.successors(train_frame, MM.N_ITEMS)
    succ_va = AP.successors(pd.read_parquet(AP.DATA / "valid.parquet"), MM.N_ITEMS)
    related = [
        (set(succ_tr.get(int(a), ())) | set(succ_va.get(int(a), ()))) - {int(a)}
        for a in anchors
    ]
    pop, freq = AP.popularity_and_frequency(train_frame, MM.N_ITEMS)
    pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)

    rng = np.random.default_rng(SEED)
    negatives, stats = MM.sample_negatives(
        anchors, positives, related, pop_bin, freq_bin, MM.N_NEG, rng
    )

    checks = {
        "no_invariant_violations": stats["invariant_violations"] == 0,
        "no_negative_equals_positive": not bool((negatives == positives[:, None]).any()),
        "no_negative_equals_anchor": not bool((negatives == anchors[:, None]).any()),
        "no_negative_is_a_train_or_valid_successor_of_its_anchor": not bool(
            np.array([negatives[i, j] in related[i]
                      for i in range(0, len(anchors), 97)
                      for j in range(negatives.shape[1])]).any()
        ),
        "ten_distinct_per_row": bool(
            (np.sort(negatives, axis=1)[:, 1:] != np.sort(negatives, axis=1)[:, :-1]).all()
        ),
        "all_ids_in_range": bool(
            (anchors >= 0).all() and (anchors < MM.N_ITEMS).all()
            and (positives >= 0).all() and (positives < MM.N_ITEMS).all()
            and (negatives >= 0).all() and (negatives < MM.N_ITEMS).all()
        ),
        "positive_is_always_a_valid_successor": bool(
            np.array([positives[i] in set(succ_va.get(int(anchors[i]), ())) - {int(anchors[i])}
                      for i in range(0, len(anchors), 97)]).all()
        ),
        "no_positive_is_a_train_successor": not bool(
            np.array([positives[i] in set(succ_tr.get(int(anchors[i]), ())) - {int(anchors[i])}
                      for i in range(0, len(anchors), 97)]).any()
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"validation pair file failed its invariants: {failed}")

    np.savez_compressed(
        OUT_NPZ,
        anchors=anchors.astype(np.int32),
        positives=positives.astype(np.int32),
        negatives=negatives.astype(np.int32),
    )
    OUT_META.write_text(json.dumps({
        "purpose": (
            "independent validation behaviour pairs: the relations are drawn from "
            "valid.parquet, so they never entered training"
        ),
        "relation": "valid_transition: history[-1] -> target, from valid.parquet",
        "seed": SEED,
        "k_negatives": MM.N_NEG,
        "item_id_space": (
            "raw item ids as stored in stage0_build_parquet and used as the keys "
            "of every item_sids.json; no re-indexing"
        ),
        "dtype": "int32",
        "shapes": {
            "anchors": list(anchors.shape),
            "positives": list(positives.shape),
            "negatives": list(negatives.shape),
        },
        "filters": rel["filters"],
        "negative_construction": stats,
        "negative_semantics": (
            "unobserved in BOTH splits, not proven irrelevant: no observed train or "
            "valid relation to that anchor"
        ),
        "checks": checks,
    }, indent=2))
    print(json.dumps({"filters": rel["filters"], "checks": checks}, indent=2))
    print(f"anchors {anchors.shape} positives {positives.shape} negatives {negatives.shape}")
    print(f"written: {OUT_NPZ}\nwritten: {OUT_META}")


if __name__ == "__main__":
    main()
