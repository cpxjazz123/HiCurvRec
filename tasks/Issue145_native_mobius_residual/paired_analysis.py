"""Issue #145 paired prediction analysis — Stage4 raw predictions 对比.

Spec 要求的产物:
  - paired bootstrap ΔR@10 95% CI
  - McNemar only-A, only-B, p-value
  - 每个 SID token 位置的 first-error 分布
  - paired hit, best-rank, prefix-match 对比
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_A = os.path.join(TASK_DIR, "control", "stage4", "raw_predictions_stage4_full.parquet")
PRED_B = os.path.join(TASK_DIR, "treatment", "stage4", "raw_predictions_stage4_full.parquet")
OUT_DIR = os.path.join(TASK_DIR, "paired_prediction_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def load_preds(path):
    df = pd.read_parquet(path)
    return df


def get_topk_per_sample(df, k=20):
    """每个 sample 取 top-K item_id."""
    out = {}
    for _, row in df.iterrows():
        sample_id = row.get("sample_id", _)
        # 解析 predictions
        preds = row["predictions"] if "predictions" in row else row.get("pred", [])
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        out[sample_id] = preds[:k] if len(preds) >= k else preds
    return out


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


def mcnemar_test(hits_A, hits_B, target_mask):
    """对 target 集合内的 sample 做 McNemar 检验."""
    only_A = int(((hits_A[target_mask] == 1) & (hits_B[target_mask] == 0)).sum())
    only_B = int(((hits_A[target_mask] == 0) & (hits_B[target_mask] == 1)).sum())
    # 排除双 0/双 1, 只算 only_A 和 only_B
    n_discord = only_A + only_B
    if n_discord == 0:
        p_value = 1.0
    else:
        # 精确二项检验 (scipy.stats.binomtest, 双向)
        result_bt = stats.binomtest(min(only_A, only_B), n_discord, 0.5)
        p_value = float(result_bt.pvalue)
    return only_A, only_B, p_value


def compute_hit_at_k(predictions, target, k=10):
    return int(target in predictions[:k])


def compute_best_rank(predictions, target):
    try:
        return predictions.index(target) + 1
    except ValueError:
        return -1


def main():
    print("=" * 70)
    print("Issue #145 Paired Prediction Analysis")
    print("=" * 70)

    df_A = load_preds(PRED_A)
    df_B = load_preds(PRED_B)

    # 解析列结构
    print(f"control columns: {list(df_A.columns)}")
    print(f"treatment columns: {list(df_B.columns)}")

    # Issue #145 Stage4 raw predictions 直接含 hit_at_10 / best_rank_top20 / prefix_match_length
    hit_col = "hit_at_10"
    rank_col = "best_rank_top20"
    prefix_col = "prefix_match_length"
    fe_col = "first_error_position"

    if hit_col not in df_A.columns:
        print(f"⚠️ 缺少 {hit_col} 列")
        return 1

    # 配对分析
    n = len(df_A)
    assert n == len(df_B), f"Sample count mismatch: {n} vs {len(df_B)}"

    hits_A = df_A[hit_col].values.astype(np.int32)
    hits_B = df_B[hit_col].values.astype(np.int32)
    best_rank_A = df_A[rank_col].values
    best_rank_B = df_B[rank_col].values
    prefix_A = df_A[prefix_col].values
    prefix_B = df_B[prefix_col].values
    fe_A = df_A[fe_col].values
    fe_B = df_B[fe_col].values

    R10_A = float(hits_A.mean())
    R10_B = float(hits_B.mean())
    delta_R10 = R10_B - R10_A

    # Paired bootstrap CI
    ci_lo, ci_hi = paired_bootstrap_ci(hits_A, hits_B)

    # McNemar (在 R@10=1 的 sample 子集)
    target_mask = np.ones(n, dtype=bool)
    only_A, only_B, p_mcnemar = mcnemar_test(hits_A, hits_B, target_mask)

    # Best rank (mean over all samples, rank > 0 means hit, =21 means miss)
    valid_A = best_rank_A[best_rank_A > 0]
    valid_B = best_rank_B[best_rank_B > 0]
    avg_rank_A = float(valid_A.mean()) if len(valid_A) > 0 else -1
    avg_rank_B = float(valid_B.mean()) if len(valid_B) > 0 else -1

    # Prefix match (length 0-4, average = mean prefix_match_length)
    pm_A = float(prefix_A.mean())
    pm_B = float(prefix_B.mean())

    # First-error position (which SID token position first mismatched)
    fe_A_mean = float(fe_A.mean())
    fe_B_mean = float(fe_B.mean())
    fe_pos_1_A = float((fe_A == 1).mean())
    fe_pos_1_B = float((fe_B == 1).mean())

    result = {
        "issue": "#145",
        "n_samples": n,
        "control_R@10": R10_A,
        "treatment_R@10": R10_B,
        "delta_R@10": delta_R10,
        "paired_bootstrap_CI_95": [ci_lo, ci_hi],
        "ci_cross_zero": bool(ci_lo < 0 < ci_hi),
        "ci_direction_correct": bool((delta_R10 > 0 and ci_lo > 0) or (delta_R10 < 0 and ci_hi < 0)),
        "McNemar": {
            "only_A": only_A,
            "only_B": only_B,
            "p_value": p_mcnemar,
        },
        "best_rank": {
            "control_mean": avg_rank_A,
            "treatment_mean": avg_rank_B,
        },
        "prefix_match_length_mean": {
            "control": pm_A,
            "treatment": pm_B,
        },
        "first_error_position": {
            "control_mean": fe_A_mean,
            "treatment_mean": fe_B_mean,
            "control_pos_1_ratio": fe_pos_1_A,
            "treatment_pos_1_ratio": fe_pos_1_B,
        },
    }
    out_path = os.path.join(OUT_DIR, "paired_prediction_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Paired analysis:")
    print(f"  control R@10: {R10_A:.4f}")
    print(f"  treatment R@10: {R10_B:.4f}")
    print(f"  Δ R@10: {delta_R10:+.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] (cross_zero={result['ci_cross_zero']})")
    print(f"  McNemar: only_A={only_A} only_B={only_B} p={p_mcnemar:.4f}")
    print(f"  avg_best_rank (命中): A={avg_rank_A:.2f} B={avg_rank_B:.2f}")
    print(f"  prefix_match_length: A={pm_A:.4f} B={pm_B:.4f}")
    print(f"  first_error_pos_1_ratio: A={fe_pos_1_A:.4f} B={fe_pos_1_B:.4f}")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
