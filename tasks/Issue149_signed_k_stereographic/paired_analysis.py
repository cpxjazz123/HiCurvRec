"""Issue #149 paired prediction analysis."""

import os, json
import numpy as np
import pandas as pd
from scipy import stats

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_A = os.path.join(TASK_DIR, "control", "stage2", "eval", "raw_predictions_stage4_full.parquet")
PRED_B = os.path.join(TASK_DIR, "treatment", "stage2", "eval", "raw_predictions_stage4_full.parquet")
OUT_DIR = os.path.join(TASK_DIR, "paired_prediction_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def paired_bootstrap_ci(hits_A, hits_B, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(hits_A)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = hits_B[idx].mean() - hits_A[idx].mean()
        diffs.append(d)
    diffs = np.array(diffs)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    print("=" * 60)
    print("Issue #149 Paired Analysis")
    print("=" * 60)
    df_A = pd.read_parquet(PRED_A)
    df_B = pd.read_parquet(PRED_B)
    print(f"A cols: {list(df_A.columns)[:10]}")
    print(f"B cols: {list(df_B.columns)[:10]}")
    hit_col = "hit_at_10"
    if hit_col not in df_A.columns:
        print(f"⚠ {hit_col} not in df, columns: {list(df_A.columns)}")
        return 1
    n = len(df_A)
    assert n == len(df_B)
    hits_A = df_A[hit_col].values.astype(np.int32)
    hits_B = df_B[hit_col].values.astype(np.int32)
    rank_col = "best_rank_top20"
    prefix_col = "prefix_match_length"
    fe_col = "first_error_position"
    rank_A = df_A[rank_col].values
    rank_B = df_B[rank_col].values
    prefix_A = df_A[prefix_col].values
    prefix_B = df_B[prefix_col].values
    fe_A = df_A[fe_col].values
    fe_B = df_B[fe_col].values
    R10_A = float(hits_A.mean())
    R10_B = float(hits_B.mean())
    delta = R10_B - R10_A
    ci_lo, ci_hi = paired_bootstrap_ci(hits_A, hits_B)
    only_A = int(((hits_A == 1) & (hits_B == 0)).sum())
    only_B = int(((hits_A == 0) & (hits_B == 1)).sum())
    n_discord = only_A + only_B
    p_mcn = float(stats.binomtest(min(only_A, only_B), n_discord, 0.5).pvalue) if n_discord > 0 else 1.0
    valid_A = rank_A[rank_A > 0]
    valid_B = rank_B[rank_B > 0]
    avg_rank_A = float(valid_A.mean()) if len(valid_A) > 0 else -1
    avg_rank_B = float(valid_B.mean()) if len(valid_B) > 0 else -1
    pm_A = float(prefix_A.mean())
    pm_B = float(prefix_B.mean())
    fe_pos1_A = float((fe_A == 1).mean())
    fe_pos1_B = float((fe_B == 1).mean())
    result = {
        "issue": "#148",
        "n_samples": n,
        "control_R@10": R10_A,
        "treatment_R@10": R10_B,
        "delta_R@10": delta,
        "paired_bootstrap_CI_95": [ci_lo, ci_hi],
        "ci_cross_zero": bool(ci_lo < 0 < ci_hi),
        "ci_direction_correct": bool((delta > 0 and ci_lo > 0) or (delta < 0 and ci_hi < 0)),
        "McNemar": {"only_A": only_A, "only_B": only_B, "p_value": p_mcn},
        "best_rank": {"control_mean": avg_rank_A, "treatment_mean": avg_rank_B},
        "prefix_match_length_mean": {"control": pm_A, "treatment": pm_B},
        "first_error_position": {"control_pos_1_ratio": fe_pos1_A, "treatment_pos_1_ratio": fe_pos1_B},
    }
    out = os.path.join(OUT_DIR, "paired_prediction_analysis.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\ncontrol R@10: {R10_A:.4f}")
    print(f"treatment R@10: {R10_B:.4f}")
    print(f"Δ R@10: {delta:+.4f}")
    print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] (cross_zero={result['ci_cross_zero']})")
    print(f"McNemar: only_A={only_A} only_B={only_B} p={p_mcn:.4f}")
    print(f"avg_best_rank: A={avg_rank_A:.2f} B={avg_rank_B:.2f}")
    print(f"prefix_match: A={pm_A:.4f} B={pm_B:.4f}")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())