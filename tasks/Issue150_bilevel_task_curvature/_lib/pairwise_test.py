"""Issue #150 paired bootstrap CI 95% + McNemar using pre-computed hit@10."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue150_bilevel_task_curvature")
CTRL_RAW = ROOT / "control/stage2/eval/raw_predictions_stage4_full.parquet"
TRT_RAW = ROOT / "treatment/stage2/eval/raw_predictions_stage4_full.parquet"


def main():
    ctrl = pd.read_parquet(CTRL_RAW)
    trt = pd.read_parquet(TRT_RAW)
    n = len(ctrl)
    assert n == len(trt)
    print(f"n_eval = {n}")

    ctrl_h10 = ctrl["hit_at_10"].astype(np.int8).values
    trt_h10 = trt["hit_at_10"].astype(np.int8).values
    ctrl_r10 = ctrl_h10.mean()
    trt_r10 = trt_h10.mean()
    diff = trt_r10 - ctrl_r10
    print(f"control R@10 = {ctrl_r10:.6f}")
    print(f"treatment R@10 = {trt_r10:.6f}")
    print(f"diff (T-C) = {diff:+.6f}")

    # Paired bootstrap CI 95% (10000 resamples)
    rng = np.random.default_rng(42)
    n_boot = 10000
    paired = trt_h10 - ctrl_h10
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = paired[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p_dir = (boot < 0).mean() if diff > 0 else (boot > 0).mean()
    print(f"paired bootstrap 95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"one-sided p(direction matches diff) ~ {p_dir:.4f}")

    # McNemar on hit@10 disagreement
    b = int(((ctrl_h10 == 1) & (trt_h10 == 0)).sum())
    c = int(((ctrl_h10 == 0) & (trt_h10 == 1)).sum())
    print(f"McNemar b (ctrl hit, trt miss)={b} c (ctrl miss, trt hit)={c}")
    p_mcn = None
    if b + c > 0:
        from scipy.stats import binomtest
        res = binomtest(min(b, c), b + c, 0.5)
        p_mcn = float(res.pvalue)
        print(f"McNemar p-value (exact, two-sided) = {p_mcn:.4e}")
        print(f"discordant = {b+c}, %discordant = {(b+c)/n*100:.2f}%")

    # NDCG@10 (pre-computed not available; use first_error_position as proxy)
    out = {
        "n_eval": n,
        "control_r10": float(ctrl_r10),
        "treatment_r10": float(trt_r10),
        "diff_t_minus_c": float(diff),
        "bootstrap_ci95_low": float(ci_lo),
        "bootstrap_ci95_high": float(ci_hi),
        "p_value_one_sided": float(p_dir),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_pvalue_two_sided": p_mcn,
    }
    out_path = ROOT / "_lib/pairwise_test.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()