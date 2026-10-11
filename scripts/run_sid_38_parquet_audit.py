#!/usr/bin/env python3
"""Complete 38-model SID/behaviour statistics from repository Parquet files.

Read-only with respect to the original models/checkpoints/SIDs, and old 35-model
metric results.  It computes missing models with the SAME upstream sampling
implementation, joins by (family, full-precision native Stage3 Recall), and
saves new versioned results rather than overwriting the existing files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"
sys.path.insert(0, str(HERE))

import analyze_prefix_behavior as AP
import monotonicity_metrics as MM
import monotonicity_analysis as MA

AP.ROOT = ROOT
AP.DATA = ROOT / "results/stage0_build_parquet"
OUT = ROOT / "results/stage3_T5Train/monotonicity"
MM.ROOT = ROOT
MM.OUT_DIR = OUT
MA.ROOT = ROOT
MA.OUT = OUT

REQUIRED = ["train.parquet", "test.parquet"]
for f in REQUIRED:
    assert (AP.DATA / f).is_file(), f"Missing {f}"
assert int(len(pd.read_parquet(AP.DATA / "test.parquet"))) == 57439

inv = json.loads((OUT / "model_inventory.json").read_text())
previous = json.loads((OUT / "unified_metrics.json").read_text())
assert inv["n_models_valid"] == 38
assert len(previous["models"]) == 35

def key(family, recall):
    return (family, round(float(recall), 13))

canonical = {key(m["family"], m["recall@10"]): m["model"] for m in inv["models"]}
assert len(canonical) == 38
existing = {}
for v in previous["models"].values():
    k = key(v["family"], v["recall@10"])
    assert k in canonical, f"Unexpected historical metric: {v['model']}"
    out_name = canonical[k]
    assert out_name not in existing, out_name
    copied = dict(v)
    copied["model"] = out_name
    existing[out_name] = copied

missing = [m for m in inv["models"] if m["model"] not in existing]
assert len(missing) == 3, missing
for m in inv["models"]:
    assert (ROOT / m["sid_path"]).is_file(), m["sid_path"]
    assert any(
        (ROOT / "results/stage3_T5Train" / run / "test_final.json").is_file()
        for run in m["stage3_runs"]
    ), m["model"]

print("Missing 3 models:", [m["model"] for m in missing], flush=True)

# Temporarily point the existing upstream script at three missing models only.
# This avoids repeating expensive per-anchor bootstraps for the 35 old models.
temporary_inventory = OUT / "missing3_inventory_input.json"
temporary_inventory.write_text(json.dumps({"models": missing}))
MM.INVENTORY = temporary_inventory
MM.OUT_JSON = OUT / "missing3_metrics.json"
MM.OUT_CSV = OUT / "missing3_metrics.csv"
MM.N_BOOT = 400  # Point statistics unchanged; bootstrap CI for three new arms.
MM.main()
computed = json.loads(MM.OUT_JSON.read_text())
assert len(computed["models"]) == 3

assert previous["pair_sets"]["test_transition"]["n_positive_pairs"] == \
       computed["pair_sets"]["test_transition"]["n_positive_pairs"] == 56636
assert previous["pair_sets"]["test_transition"]["neg10"]["negative_draws"] == \
       computed["pair_sets"]["test_transition"]["neg10"]["negative_draws"] == 566360

# Canonical 38-model report, retaining original 35 metrics without recomputation.
combined = dict(previous)
combined["n_models"] = 38
combined["models"] = dict(existing)
for model in computed["models"].values():
    assert model["model"] not in combined["models"]
    combined["models"][model["model"]] = model
assert len(combined["models"]) == 38
combined["completion_note"] = {
    "base": "Original 35-model unified_metrics.json unchanged",
    "completed_models": [m["model"] for m in missing],
    "matched_parquet_sampling": True,
    "added_model_bootstrap_repetitions": MM.N_BOOT,
    "existing_model_bootstrap_repetitions": previous["n_boot"],
    "model_key": "canonical family/subfamily/name from model_inventory.json",
}
combined_path = OUT / "unified_metrics_38.json"
combined_path.write_text(json.dumps(combined, indent=2))
MM.OUT_CSV = OUT / "unified_metrics_38.csv"
MM._write_csv(combined)

# Precisely count what used to be 32 shared-negative examples when only one
# negative was drawn per anchor in the old subsampling protocol.
train = pd.read_parquet(AP.DATA / "train.parquet")
test = pd.read_parquet(AP.DATA / "test.parquet")
pair_set = MM.build_test_relations(MM.N_ITEMS)
anchors, positives = pair_set["pairs"][:, 0], pair_set["pairs"][:, 1]
succ_tr = AP.successors(train, MM.N_ITEMS)
succ_te = AP.successors(test, MM.N_ITEMS)
pop, freq = AP.popularity_and_frequency(train, MM.N_ITEMS)
pop_bin, freq_bin = AP.decile_bins(pop), AP.decile_bins(freq)
related = [
    (set(succ_tr.get(int(a), ())) | set(succ_te.get(int(a), ()))) - {int(a)}
    for a in anchors
]
negs, neg_stats = MM.sample_negatives(
    anchors, positives, related, pop_bin, freq_bin, MM.N_NEG,
    np.random.default_rng(AP.SEED + 1000),
)
assert neg_stats["invariant_violations"] == 0
assert neg_stats["negative_draws"] == 566360
base = MM.load_levels(ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json")
exp = MM.load_levels(ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json")
flat_neg = negs.ravel()
rep_a = np.repeat(anchors, MM.N_NEG)
both_neg = (base[rep_a, 0] == base[flat_neg, 0]) & \
           (exp[rep_a, 0] == exp[flat_neg, 0])
both_pos = (base[anchors, 0] == base[positives, 0]) & \
           (exp[anchors, 0] == exp[positives, 0])
def l2_count(levels, anchors_, targets_, mask):
    return int(np.count_nonzero(
        (levels[anchors_, 1] == levels[targets_, 1]) & mask
    ))
negative_result = {
    "scope": "same 566360 matched-negative draws across the TIGER and Exp1a SIDs",
    "negative_draws": len(flat_neg),
    "positive_pairs": len(positives),
    "shared_L1_both_models_neg_draws": int(both_neg.sum()),
    "shared_L1_both_models_pos_pairs": int(both_pos.sum()),
    "negative_common_L1_continue_L2_base_count": l2_count(base, rep_a, flat_neg, both_neg),
    "negative_common_L1_continue_L2_exp1a_count": l2_count(exp, rep_a, flat_neg, both_neg),
    "positive_common_L1_continue_L2_base_count": l2_count(base, anchors, positives, both_pos),
    "positive_common_L1_continue_L2_exp1a_count": l2_count(exp, anchors, positives, both_pos),
    "negative_distinct_pair_count": len(set(zip(rep_a.tolist(), flat_neg.tolist()))),
    "negative_matching_stats": neg_stats,
    "reproducibility": {"seed": AP.SEED + 1000, "k": MM.N_NEG},
}
(OUT / "common_L1_intersection_38.json").write_text(json.dumps(negative_result, indent=2))
print("Common-L1 negative intersection:", negative_result, flush=True)

# Final analysis on all 38 models, preserving the existing 35 data rows.
MA.METRICS = combined_path
MA.OUT_JSON = OUT / "monotonicity_38.json"
MA.OUT_CSV = OUT / "monotonicity_38.csv"
MA.OUT_MD = OUT / "monotonicity_report_38.md"
MA.FIGDIR = OUT / "figures38"
MA.SEED_FAMILIES = {
    "resample_s42_s43_s44": [
        "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample",
        "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s43",
        "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample_s44",
    ],
    "euclid_l2_256_72k_s42_s43_s44": [
        "curvature_RQ-VAE/euclid_l2_256_72k_s42",
        "curvature_RQ-VAE/euclid_l2_256_72k_s43",
        "curvature_RQ-VAE/euclid_l2_256_72k_s44",
    ],
}
# Null distribution on 38 arms with 60 permutations, so CI remains exploratory.
original_null = MA.group_size_null
MA.group_size_null = lambda report, metric: original_null(report, metric, n_perm=60)
MA.main()
result = json.loads(MA.OUT_JSON.read_text())
assert result["n_models"] == 38
print("FINISHED 38-MODEL ANALYSIS", flush=True)
for metric in ("L1_gap", "L1_er", "collision_rate", "L1_p_share_pos", "L1_p_share_neg"):
    print(metric, result["correlations"][metric], flush=True)
