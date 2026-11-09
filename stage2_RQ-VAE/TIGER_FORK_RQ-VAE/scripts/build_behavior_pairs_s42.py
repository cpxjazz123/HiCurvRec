"""The canonical behaviour pair set every experiment in the programme shares.

One file, one seed, one sampling rule, used unchanged by the baseline and by
every mechanism that follows. The programme explicitly forbids multi-seed
experiments, so nothing here draws more than once and nothing downstream may
redraw: a mechanism that changed the pairs would be measuring a different
sample rather than a different mechanism.

  positives   all distinct history[-1] -> target relations in train.parquet,
              filtered only for self-loops and out-of-range ids, with the count
              dropped at each filter recorded
  negatives   ten per positive, matched on the popularity decile (how often the
              item is a target) and the appearance-frequency decile (how often it
              appears anywhere), excluding the anchor, the positive itself and
              every item with an observed relation to that anchor in train

Seed 42 exactly as the programme specifies, not the 20261011 used by the earlier
test-relation export; the two files describe different relations and different
draws and are not interchangeable.

The one-negative control used in the ratio comparison is negatives[:, 0] of this
same file, so the nested sets are consistent by construction instead of being two
independent draws.

A negative means UNOBSERVED. Absence of an observed edge is not evidence of
irrelevance, and nothing in the programme may treat it as such.
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
OUT_NPZ = OUT_DIR / "behavior_pairs_s42.npz"
OUT_META = OUT_DIR / "behavior_pairs_s42.meta.json"


def build_train_relations(n_items: int) -> dict:
    frame = pd.read_parquet(AP.DATA / "train.parquet")
    pairs = AP.transitions(frame, n_items)
    a, b = pairs[:, 0], pairs[:, 1]
    bad = (a < 0) | (a >= n_items) | (b < 0) | (b >= n_items)
    selfloop = (a == b) & ~bad
    keep = ~bad & ~selfloop
    kept = pairs[keep]
    return {
        "pairs": kept,
        "filters": {
            "rows_in_train_parquet": int(len(frame)),
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
    frame = pd.read_parquet(AP.DATA / "train.parquet")
    rel = build_train_relations(MM.N_ITEMS)
    pairs = rel["pairs"]
    anchors, positives = pairs[:, 0].copy(), pairs[:, 1].copy()

    succ_tr = AP.successors(frame, MM.N_ITEMS)
    related = [set(succ_tr.get(int(a), ())) - {int(a)} for a in anchors]
    pop, freq = AP.popularity_and_frequency(frame, MM.N_ITEMS)
    pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)

    rng = np.random.default_rng(SEED)
    negatives, stats = MM.sample_negatives(
        anchors, positives, related, pop_bin, freq_bin, MM.N_NEG, rng
    )

    checks = {
        "no_invariant_violations": stats["invariant_violations"] == 0,
        "no_negative_equals_positive": not bool((negatives == positives[:, None]).any()),
        "no_negative_equals_anchor": not bool((negatives == anchors[:, None]).any()),
        "no_negative_is_a_train_successor_of_its_anchor": not bool(
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
        "positive_is_always_a_train_successor": bool(
            np.array([positives[i] in related[i]
                      for i in range(0, len(anchors), 97)]).all()
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"pair file failed its invariants: {failed}")

    np.savez_compressed(
        OUT_NPZ,
        anchors=anchors.astype(np.int32),
        positives=positives.astype(np.int32),
        negatives=negatives.astype(np.int32),
    )
    OUT_META.write_text(json.dumps({
        "purpose": "canonical behaviour pairs shared by every experiment in the programme",
        "relation": "train_transition: history[-1] -> target, from train.parquet",
        "seed": SEED,
        "k_negatives": MM.N_NEG,
        "one_negative_control": "negatives[:, 0] of this same file, not a second draw",
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
            "unobserved, not proven irrelevant: no observed train relation to that anchor"
        ),
        "checks": checks,
        "reproduce": (
            "cd /home/wlia0047/ar57/wenyu/GeneRec && OMP_NUM_THREADS=8 "
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 "
            "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/build_behavior_pairs_s42.py"
        ),
    }, indent=2))
    print(json.dumps({"filters": rel["filters"], "checks": checks}, indent=2))
    print(f"anchors {anchors.shape} positives {positives.shape} negatives {negatives.shape}")
    print(f"written: {OUT_NPZ}\nwritten: {OUT_META}")


if __name__ == "__main__":
    main()