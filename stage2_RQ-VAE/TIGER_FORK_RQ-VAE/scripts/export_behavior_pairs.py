"""Export the frozen positive/negative pair index used by the metric analysis.

Writes results/stage3_T5Train/monotonicity/behavior_pairs.npz with

    anchors    (56636,)      history[-1] item id of each relation
    positives  (56636,)      the observed target item id of that relation
    negatives  (56636, 10)   ten matched unobserved negatives per positive

Every array is in the SAME item id space: the raw ids as they appear in
stage0_build_parquet, which is also the key space of every item_sids.json. No
re-indexing happens anywhere, so row i of anchors pairs with row i of positives
and row i of negatives, and each id can be looked up directly in a SID file.

The point of freezing this file is that the sharing rates, gaps and enrichment
ratios in the corrected reports are a function of these pairs. Publishing the
pairs makes those numbers checkable without re-running the sampler and without
depending on the exact numpy version that happened to draw them.

This does NOT re-implement the sampling. It imports build_test_relations and
sample_negatives from monotonicity_metrics, so the file is produced by the same
code path and the same seed as the metrics run, and the script asserts that the
result matches the counts the metrics run reported before it writes anything.

Read-only with respect to every experiment: no model, SID, checkpoint or result
is touched, and no training happens.
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

OUT = MM.OUT_DIR / "behavior_pairs.npz"
META = MM.OUT_DIR / "behavior_pairs.meta.json"
METRICS = MM.OUT_DIR / "unified_metrics.json"
# The test relation is index 1 in monotonicity_metrics' spec list, so its
# negative draw uses SEED + 1000 * 1. Using a different offset would produce a
# different sample and silently invalidate every published sharing rate.
RELATION_OFFSET = 1


def main() -> None:
    train = pd.read_parquet(AP.DATA / "train.parquet")
    test = pd.read_parquet(AP.DATA / "test.parquet")

    rel = MM.build_test_relations(MM.N_ITEMS)
    pairs = rel["pairs"]
    anchors = pairs[:, 0].copy()
    positives = pairs[:, 1].copy()

    succ_tr = AP.successors(train, MM.N_ITEMS)
    succ_te = AP.successors(test, MM.N_ITEMS)
    related = [
        (set(succ_tr.get(int(a), ())) | set(succ_te.get(int(a), ()))) - {int(a)}
        for a in anchors
    ]
    pop, freq = AP.popularity_and_frequency(train, MM.N_ITEMS)
    pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)

    rng = np.random.default_rng(AP.SEED + 1000 * RELATION_OFFSET)
    negatives, stats = MM.sample_negatives(
        anchors, positives, related, pop_bin, freq_bin, MM.N_NEG, rng
    )

    # Cross-check against the metrics run before writing: if these disagree the
    # file would describe a different sample from the one the reports used.
    published = json.loads(METRICS.read_text())["pair_sets"]["test_transition"]
    checks = {
        "n_positives_matches_metrics": int(len(anchors)) == published["n_positive_pairs"],
        "n_anchors_matches_metrics": int(len(np.unique(anchors))) == published["n_anchors"],
        "negative_draws_matches_metrics": int(stats["negative_draws"]) == published["neg10"]["negative_draws"],
        "no_invariant_violations": stats["invariant_violations"] == 0,
        "no_negative_equals_positive": not bool((negatives == positives[:, None]).any()),
        "no_negative_equals_anchor": not bool((negatives == anchors[:, None]).any()),
        "negatives_are_ten_distinct_per_row": bool(
            (np.sort(negatives, axis=1)[:, 1:] != np.sort(negatives, axis=1)[:, :-1]).all()
        ),
        "all_ids_in_range": bool(
            (anchors >= 0).all() and (anchors < MM.N_ITEMS).all()
            and (positives >= 0).all() and (positives < MM.N_ITEMS).all()
            and (negatives >= 0).all() and (negatives < MM.N_ITEMS).all()
        ),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"refusing to write a mismatched pair file: {failed}")

    np.savez_compressed(
        OUT,
        anchors=anchors.astype(np.int32),
        positives=positives.astype(np.int32),
        negatives=negatives.astype(np.int32),
    )
    META.write_text(json.dumps({
        "relation": "test_transition: history[-1] -> target, from test.parquet",
        "source_rows": rel["filters"],
        "seed": AP.SEED,
        "relation_offset": RELATION_OFFSET,
        "rng_seed_used": AP.SEED + 1000 * RELATION_OFFSET,
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
        "negative_construction": stats,
        "checks": checks,
        "negative_semantics": (
            "unobserved, not proven irrelevant: each negative is an item with no "
            "observed relation to that anchor in either train or test"
        ),
        "reproduce": (
            "cd /home/wlia0047/ar57/wenyu/GeneRec && OMP_NUM_THREADS=8 "
            "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 "
            "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/export_behavior_pairs.py"
        ),
    }, indent=2))
    print(json.dumps(checks, indent=2))
    print(f"anchors {anchors.shape}, positives {positives.shape}, negatives {negatives.shape}")
    print(f"distinct anchors {len(np.unique(anchors))}")
    print(f"written: {OUT}\nwritten: {META}")


if __name__ == "__main__":
    main()