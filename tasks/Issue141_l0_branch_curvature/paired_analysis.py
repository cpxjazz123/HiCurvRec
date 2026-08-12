"""Issue #141 Gate 3: A/B paired 分析 (固定同一批 test 样本).

产物:
- stage4_metrics_ab.csv: A/B 六指标
- paired_prediction_analysis.csv/json: per-sample paired hit/rank + bootstrap CI + 固定桶
- issue141_verdict.md/json: 总 verdict
"""
import json
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue141_l0_branch_curvature")

# ── 1. stage4_metrics_ab.csv ──
eval_a = json.load(open(TASK / "control/stage2/eval/eval_test.json"))
eval_b = json.load(open(TASK / "treatment/stage2/eval/eval_test.json"))
metrics = ["R@5", "R@10", "R@20", "NDCG@5", "NDCG@10", "NDCG@20"]
rows = []
for m in metrics:
    rows.append({"metric": m, "A_shared": eval_a[m], "B_codeword": eval_b[m], "B_minus_A": eval_b[m] - eval_a[m]})
with open(TASK / "stage4_metrics_ab.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "A_shared", "B_codeword", "B_minus_A"])
    w.writeheader()
    w.writerows(rows)
print("stage4_metrics_ab.csv:")
for r in rows:
    print(f"  {r['metric']}: A={r['A_shared']:.4f} B={r['B_codeword']:.4f} Δ={r['B_minus_A']:+.4f}")

# ── 2. paired 分析 (固定同一批 test 样本) ──
raw_a = pd.read_parquet(TASK / "control/stage2/eval/raw_predictions_stage4_full.parquet")
raw_b = pd.read_parquet(TASK / "treatment/stage2/eval/raw_predictions_stage4_full.parquet")
assert len(raw_a) == len(raw_b) == 24772
# sample_idx 对齐 (同 test 顺序, 固定划分)
pa = raw_a.sort_values("sample_idx").reset_index(drop=True)
pb = raw_b.sort_values("sample_idx").reset_index(drop=True)
assert (pa["sample_idx"].values == pb["sample_idx"].values).all()
# 注意: A/B 的 target_sid 编码不同 (L0 分配不同是实验本质), 但 sample_idx 对齐 =
# 同一批固定 test 样本 (同一 user history + 同一 target item). paired 对比按 sample_idx 对齐.

# per-sample paired hit@10 / rank
hit_a = pa["hit_at_10"].to_numpy().astype(int)
hit_b = pb["hit_at_10"].to_numpy().astype(int)
rank_a = pa["best_rank_top20"].to_numpy().astype(int)
rank_b = pb["best_rank_top20"].to_numpy().astype(int)
first_err_a = pa["first_error_position"].to_numpy().astype(int)
first_err_b = pb["first_error_position"].to_numpy().astype(int)

n = len(hit_a)
n_both_hit = int((hit_a & hit_b).sum())
n_only_b = int((hit_b & ~hit_a).sum())
n_only_a = int((hit_a & ~hit_b).sum())
n_both_miss = int((~hit_a.astype(bool) & ~hit_b.astype(bool)).sum())
# McNemar (paired proportion test)
from scipy import stats as sp
b_val = n_only_b
c_val = n_only_a
mcnemar_chi2 = (abs(b_val - c_val) - 1) ** 2 / max(b_val + c_val, 1)
mcnemar_p = 1 - sp.chi2.cdf(mcnemar_chi2, 1) if b_val + c_val > 0 else 1.0

# bootstrap CI on paired ΔR@10 (10000 次, seed=42)
rng = np.random.RandomState(42)
deltas = hit_b - hit_a  # per-sample ±1/0
boots = []
for _ in range(10000):
    idx = rng.randint(0, n, size=n)
    boots.append(deltas[idx].mean())
boot_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
boot_p = float(np.mean(np.array(boots) <= 0))  # 单侧: H0 Δ≤0
mean_delta = float(deltas.mean())
print(f"\npaired ΔR@10 = {mean_delta:+.4f}, 95% CI = [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}], "
      f"P(H0: Δ≤0) = {boot_p:.4f}")
print(f"McNemar: only_B={n_only_b} only_A={n_only_a} p={mcnemar_p:.4f}")

# ── 固定 L0 频率桶 (同一组固定 item index: 基于 reference-0 Issue139 SID 计算, 禁止按 A/B 各自结果选桶) ──
import numpy as np
ref_sid = np.load("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue139_l3_dedup_sid/stage2/sid_output.npy")  # (9922,4), row i = item id i+1
ref_l0_freq = pd.Series(ref_sid[:, 0]).value_counts().sort_index()
bucket_lo, bucket_hi = 200, 1000  # Issue138/139 固定最差桶定义 (reference-0 编码)
test_df = pd.read_parquet("/home/wlia0047/ar57/wenyu/GeneRec/dataset/test.parquet")
target_items = test_df["target"].to_numpy()  # (24772,) item id (1-indexed)
ref_l0_of_item = ref_sid[target_items - 1, 0]
freq_of_sample = ref_l0_freq.reindex(ref_l0_of_item).to_numpy()
mask_bucket = (freq_of_sample >= bucket_lo) & (freq_of_sample < bucket_hi)
print(f"固定 L0 桶 [{bucket_lo},{bucket_hi}) (reference-0 编码, 固定 item index): n={mask_bucket.sum()}")
ra10_b = hit_b[mask_bucket].mean() if mask_bucket.sum() else float("nan")
ra10_a = hit_a[mask_bucket].mean() if mask_bucket.sum() else float("nan")
print(f"  桶内 R@10: A={ra10_a:.4f} B={ra10_b:.4f} Δ={ra10_b - ra10_a:+.4f}")

# first-error 改善
fe1_a = float((first_err_a == 1).mean())
fe1_b = float((first_err_b == 1).mean())
print(f"first_error=1 比例: A={fe1_a:.4f} B={fe1_b:.4f} Δ={fe1_b - fe1_a:+.4f}")

# per-sample 落盘
out_df = pd.DataFrame({
    "sample_idx": pa["sample_idx"],
    "target_item": target_items,
    "target_l0_ref": ref_l0_of_item,
    "l0_freq_ref": freq_of_sample,
    "fixed_bucket_200_1000": mask_bucket,
    "hit_at_10_A": hit_a, "hit_at_10_B": hit_b,
    "rank_A": rank_a, "rank_B": rank_b,
    "first_error_A": first_err_a, "first_error_B": first_err_b,
    "paired_delta_hit10": deltas,
})
out_df.to_csv(TASK / "paired_prediction_analysis.csv", index=False)

analysis = {
    "issue": "#141",
    "n_test": n,
    "R10_A": eval_a["R@10"], "R10_B": eval_b["R@10"],
    "R10_delta": eval_b["R@10"] - eval_a["R@10"],
    "paired_delta_R10": mean_delta,
    "paired_delta_CI95": boot_ci,
    "paired_delta_p_bootstrap": boot_p,
    "paired_delta_CI_crosses_zero": boot_ci[0] < 0 < boot_ci[1],
    "mcnemar_only_B": n_only_b, "mcnemar_only_A": n_only_a,
    "mcnemar_p": mcnemar_p,
    "both_hit": n_both_hit, "both_miss": n_both_miss,
    "first_error_pos1_A": fe1_a, "first_error_pos1_B": fe1_b,
    "fixed_bucket": [bucket_lo, bucket_hi],
    "fixed_bucket_n": int(mask_bucket.sum()),
    "fixed_bucket_R10_A": ra10_a if not math.isnan(ra10_a) else None,
    "fixed_bucket_R10_B": ra10_b if not math.isnan(ra10_b) else None,
    "fixed_bucket_delta": (ra10_b - ra10_a) if not math.isnan(ra10_a) else None,
    "metrics_ab": rows,
}
with open(TASK / "paired_prediction_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)
print(f"\n-> paired_prediction_analysis.csv/json")

# ── 3. verdict ──
gate3_ok = (eval_b["R@10"] > eval_a["R@10"]) and (boot_ci[0] > 0)
first_err_ok = fe1_b < fe1_a or ra10_b > ra10_a  # 至少一项改善
# 另一项不得显著恶化: first-error 若恶化 ≤0.005, 桶若恶化 ≤0.01
if fe1_b > fe1_a:
    fe_deg = fe1_b - fe1_a
    fe_ok = fe_deg <= 0.005 and ra10_b > ra10_a
else:
    fe_ok = True
if ra10_b < ra10_a:
    bucket_deg = ra10_a - ra10_b
    bucket_ok = bucket_deg <= 0.01 and fe1_b < fe1_a
else:
    bucket_ok = True
mechanism_ok = gate3_ok and fe_ok and bucket_ok
target_reached = eval_b["R@10"] > 0.1020
# scale-shortcut audit (treatment)
audit_b = json.load(open(TASK / "treatment/stage2/scale_shortcut_audit.json"))
no_shortcut = not audit_b["scale_shortcut_detected"]

decision = "机制有效 (MECHANISM_EFFECTIVE)" if (mechanism_ok and no_shortcut) else "无效 (INVALID)"
if target_reached and mechanism_ok:
    decision = "达到目标 (TARGET_REACHED)"

verdict = {
    "issue": "#141",
    "decision": decision,
    "gate1_pass": json.load(open(TASK / "ab_initial_equivalence.json"))["gate1_pass"],
    "gate3": {
        "B_R10_gt_A": bool(eval_b["R@10"] > eval_a["R@10"]),
        "paired_CI95_not_cross_zero": bool(boot_ci[0] > 0),
        "first_error_or_bucket_improved": first_err_ok,
        "other_not_degraded": fe_ok and bucket_ok,
        "scale_shortcut_B": audit_b["scale_shortcut_detected"],
        "target_reached_0_1020": target_reached,
    },
    "A_shared": eval_a,
    "B_codeword": eval_b,
    "paired": analysis,
}
with open(TASK / "issue141_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2, default=str)
md = f"""# Issue #141 Verdict: {decision}

## A/B 六指标 (Stage4, beam=20, 全量 24772)

| metric | A (共享曲率) | B (codeword-specific) | Δ |
|---|---|---|---|
"""
for r in rows:
    md += f"| {r['metric']} | {r['A_shared']:.4f} | {r['B_codeword']:.4f} | {r['B_minus_A']:+.4f} |\n"
md += f"""
## Paired 分析 (同一批 test 样本, bootstrap 10000 次)

- ΔR@10 (paired) = {mean_delta:+.4f}, 95% CI = [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]
- McNemar: only_B={n_only_b}, only_A={n_only_a}, p={mcnemar_p:.4f}
- first_error=1: A={fe1_a:.4f} → B={fe1_b:.4f}
- 固定 L0 桶 [{bucket_lo},{bucket_hi}) (n={mask_bucket.sum()}): A={ra10_a:.4f} → B={ra10_b:.4f}

## Gate 判定

- B R@10 > A: {bool(eval_b['R@10'] > eval_a['R@10'])}
- paired CI 不跨 0: {bool(boot_ci[0] > 0)}
- first-error/固定桶至少一项改善: {first_err_ok}
- 另一项不显著恶化: {fe_ok and bucket_ok}
- B 无尺度捷径: {not audit_b['scale_shortcut_detected']}
- TARGET (R@10 > 0.1020): {target_reached}
"""
open(TASK / "issue141_verdict.md", "w").write(md)
print(f"\n-> issue141_verdict.md/json: {decision}")
