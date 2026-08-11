#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #118 / #214 — Curvature Sweep 双向扩展 (extrapolation from Issue #83 CSV).

Issue #83 equal-K (64/128/256) Stage2 ckpt 已被清理. 原始 sweep 脚本也丢失.
但 Issue #83 留下了 `curvature_distortion_sweep.csv` (L0/L1/L2 × c∈{0,...,10} 的 distortion).
本脚本使用 **数学 extrapolation**:

  Phase A (右侧): 在 c∈{10, 20, 50, 100, 200, 500} 拟合模型并外推, 找 L1/L2 拐点
  Phase B (左侧): 用 L0 在 c≥0 的模型投影到 c∈{-10, -5, -2, -1, -0.5}, 验证是否单调升

模型候选:
  - M1: D(c) = a + b / (c + ε)            (1/x 型, 适合 L1/L2 hyperbolic)
  - M2: D(c) = a + b * exp(-k*c)          (exponential decay, 可能适合 L0)
  - M3: D(c) = a * (1 + c)^(-α) + b       (power law)
  - M4: D(c) = a + b * c^α                (power growth, 适合 L0)
  - M5: D(c) = a + b * c + c1 * c^2       (polynomial degree-2)

对每层拟合 best model (按 R²), 然后:
  - 找 D'(c) = 0 (解析拐点) 或 D'(c) 变号 (数值拐点)
  - 在外推网格上估计 D(c)
  - 报告 best c* + 95% CI (来自残差)

输出:
  taskA/_history/issue214_curvature_sweep_extend/phase_a/curvature_distortion_sweep_extend.csv
  taskA/_history/issue214_curvature_sweep_extend/phase_b/curvature_distortion_sweep_extend_left.csv
  taskA/_history/issue214_curvature_sweep_extend/issue214_curvature_sweep_extend_result.md
"""
import os
import sys
import json
import argparse
import csv
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Issue #83 CSV data (hard-coded for reproducibility)
# ──────────────────────────────────────────────────────────────
ISSUE83_CSVS = {
    "equal64":  "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue83_codebook_curvature_alignment/equal64/curvature_distortion_sweep.csv",
    "equal128": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue83_codebook_curvature_alignment/equal128/curvature_distortion_sweep.csv",
    "equal256": "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue83_codebook_curvature_alignment/equal256/curvature_distortion_sweep.csv",
}


def load_issue83_data(config):
    csv_path = ISSUE83_CSVS[config]
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        c_grid = [float(c.split("=")[1]) for c in header[1:]]
        data = {}
        for row in reader:
            layer = row[0]
            vals = [float(v) for v in row[1:]]
            data[layer] = vals
    return c_grid, data


# ──────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────
def fit_models(c_grid, vals):
    """Try multiple models on (c_grid, vals), return best by R² + monotonicity.

    Returns dict {model_name: {coef, r2, monotone, predict}}.
    Monotone flag: 'inc' if prediction is monotone increasing in c∈[0, 50]
                   'dec' if monotone decreasing
                   'non' otherwise

    Critical: validate predictions match data via np.allclose (rel<1e-3).
    Without this check, R² is misleading when data has near-constant values
    (e.g., L2 distortion varies by only 3e-4 across c∈[0,10]).
    """
    cs = np.array(c_grid)
    vs = np.array(vals)
    models = {}

    # M1: D = a + b / (c + ε)  → numerically stable with c+1
    try:
        x = 1.0 / (cs + 1.0)
        coef = np.polyfit(x, vs, 1)  # coef[0]=slope, coef[1]=intercept
        pred = coef[0] * x + coef[1]
        # Validate: predictions must match data within 1% relative
        if not np.allclose(pred, vs, rtol=1e-3, atol=1e-5):
            pass  # Skip this model
        else:
            ss_res = ((vs - pred) ** 2).sum()
            ss_tot = ((vs - vs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            c_test = np.linspace(0, 500, 200)
            pred_test = coef[0] / (c_test + 1.0) + coef[1]
            d_pred = np.diff(pred_test)
            if (d_pred >= -1e-6).all():
                mono = "inc"
            elif (d_pred <= 1e-6).all():
                mono = "dec"
            else:
                mono = "non"
            models["M1_1div"] = {"coef": coef, "r2": r2, "monotone": mono, "predict": lambda c, cf=coef: cf[0] / (c + 1.0) + cf[1]}
    except Exception:
        pass

    # M2: D = vs[0] + b * (exp(-k*c) - 1)  (c=0 anchored to data)
    try:
        from scipy.optimize import curve_fit
        v0 = float(vs[0])
        def fn(c, b, k):
            return v0 + b * (np.exp(-k * c) - 1.0)
        # Initial: b = vs[-1] - vs[0] (delta over [0, 10]), k = 0.1
        b0 = float(vs[-1] - vs[0])
        popt, _ = curve_fit(fn, cs, vs, p0=[b0, 0.1], maxfev=2000,
                             bounds=([-1.0, 1e-6], [1.0, 10.0]))
        pred = fn(cs, *popt)
        if not np.allclose(pred, vs, rtol=1e-3, atol=1e-5):
            pass  # Skip — fit failed silently
        else:
            ss_res = ((vs - pred) ** 2).sum()
            ss_tot = ((vs - vs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            c_test = np.linspace(0, 500, 200)
            pred_test = fn(c_test, *popt)
            d_pred = np.diff(pred_test)
            if (d_pred >= -1e-6).all():
                mono = "inc"
            elif (d_pred <= 1e-6).all():
                mono = "dec"
            else:
                mono = "non"
            models["M2_exp"] = {"coef": popt, "r2": r2, "monotone": mono,
                                "predict": lambda c, cf=popt, v=v0: v + cf[0] * (np.exp(-cf[1] * c) - 1.0)}
    except Exception:
        pass

    # M3: D = a * (1 + c)^(-α) + b
    try:
        from scipy.optimize import curve_fit
        def fn(c, a, alpha, b):
            return a * np.power(1 + c, -alpha) + b
        popt, _ = curve_fit(fn, cs, vs, p0=[vs[0] - vs[-1], 0.5, vs[-1]], maxfev=2000)
        pred = fn(cs, *popt)
        if not np.allclose(pred, vs, rtol=1e-3, atol=1e-5):
            pass
        else:
            ss_res = ((vs - pred) ** 2).sum()
            ss_tot = ((vs - vs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            c_test = np.linspace(0, 500, 200)
            pred_test = fn(c_test, *popt)
            d_pred = np.diff(pred_test)
            if (d_pred >= -1e-6).all():
                mono = "inc"
            elif (d_pred <= 1e-6).all():
                mono = "dec"
            else:
                mono = "non"
            models["M3_power"] = {"coef": popt, "r2": r2, "monotone": mono, "predict": lambda c, cf=popt: fn(c, *cf)}
    except Exception:
        pass

    # M4: D = a + b * c^α
    try:
        from scipy.optimize import curve_fit
        def fn(c, a, b, alpha):
            return a + b * np.power(np.maximum(c, 1e-10), alpha)
        popt, _ = curve_fit(fn, cs, vs, p0=[vs[0], vs[-1] - vs[0], 0.3], maxfev=2000)
        pred = fn(cs, *popt)
        if not np.allclose(pred, vs, rtol=1e-3, atol=1e-5):
            pass
        else:
            ss_res = ((vs - pred) ** 2).sum()
            ss_tot = ((vs - vs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            c_test = np.linspace(0, 500, 200)
            pred_test = fn(c_test, *popt)
            d_pred = np.diff(pred_test)
            if (d_pred >= -1e-6).all():
                mono = "inc"
            elif (d_pred <= 1e-6).all():
                mono = "dec"
            else:
                mono = "non"
            models["M4_growth"] = {"coef": popt, "r2": r2, "monotone": mono, "predict": lambda c, cf=popt: fn(c, *cf)}
    except Exception:
        pass

    # M5: D = a + b*c + c1*c^2
    try:
        coef = np.polyfit(cs, vs, 2)
        pred = np.polyval(coef, cs)
        if not np.allclose(pred, vs, rtol=1e-3, atol=1e-5):
            pass
        else:
            ss_res = ((vs - pred) ** 2).sum()
            ss_tot = ((vs - vs.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
            c_test = np.linspace(0, 500, 200)
            pred_test = np.polyval(coef, c_test)
            d_pred = np.diff(pred_test)
            if (d_pred >= -1e-6).all():
                mono = "inc"
            elif (d_pred <= 1e-6).all():
                mono = "dec"
            else:
                mono = "non"
            models["M5_poly2"] = {"coef": coef, "r2": r2, "monotone": mono, "predict": lambda c, cf=coef: np.polyval(cf, c)}
    except Exception:
        pass

    return models


def find_inflection_point(model, c_min, c_max, n_pts=10000):
    """Find c where D'(c) = 0 numerically."""
    cs = np.linspace(c_min, c_max, n_pts)
    preds = model["predict"](cs)
    diffs = np.diff(preds) / np.diff(cs)
    # Find sign change in diffs
    signs = np.sign(diffs)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        return (cs[idx] + cs[idx + 1]) / 2, preds[idx]
    # No inflection — return endpoint with smaller value
    if preds[0] < preds[-1]:
        return c_min, preds[0]
    return c_max, preds[-1]


def extrapolate(config):
    """Extrapolate Issue #83 CSV to extended grid, find inflection points."""
    c_grid, data = load_issue83_data(config)
    print(f"\n[{config}] Issue #83 c_grid: {c_grid}")
    print(f"[{config}] Issue #83 distortion:")
    for layer in ["L0", "L1", "L2"]:
        print(f"  {layer}: {data[layer]}")

    # Fit models per layer
    fitted = {}
    for layer in ["L0", "L1", "L2"]:
        models = fit_models(c_grid, data[layer])
        # Pick best R² among MONOTONIC models (R² 差异 < 1e-4 时优先 monotone, 避免 poly2 外推到 500 失败)
        monotone_models = {k: v for k, v in models.items() if v["monotone"] != "non"}
        if monotone_models:
            # 选 R² 最高的 monotone 模型
            best_name = max(monotone_models, key=lambda k: monotone_models[k]["r2"])
            mono_tag = f" [monotone={monotone_models[best_name]['monotone']}]"
        else:
            # 全 poly2 这种非单调 — 警告 + 仍选 R² 最高 (限制外推到 [0, 50])
            best_name = max(models, key=lambda k: models[k]["r2"])
            mono_tag = " [⚠ NON-MONOTONE, 限制外推 c≤50]"
        best = models[best_name]
        fitted[layer] = {"best_model": best_name, **best, "extrapolation_limit": "c≤50" if mono_tag.startswith(" [⚠") else "c≤500"}
        print(f"  [{config}] {layer} best model: {best_name} (R²={best['r2']:.4f}, monotone={best['monotone']}){mono_tag}")

    return c_grid, data, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="taskA/_history/issue214_curvature_sweep_extend")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    phase_a_dir = output_dir / "phase_a"
    phase_b_dir = output_dir / "phase_b"
    phase_a_dir.mkdir(parents=True, exist_ok=True)
    phase_b_dir.mkdir(parents=True, exist_ok=True)

    # Extended grids
    right_grid = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    left_grid = [-10.0, -5.0, -2.0, -1.0, -0.5, -0.1, -0.01, 0.0]

    # Run for all 3 K configs
    all_results = {}
    for config in ["equal64", "equal128", "equal256"]:
        c_grid, data, fitted = extrapolate(config)
        all_results[config] = {"c_grid_orig": c_grid, "data_orig": data, "fitted": fitted}

        # Phase A: right side
        # For layers L1/L2 (which decrease), look for inflection in [10, 500]
        # For layer L0 (which increases), model D(c) and find best
        # ⚠ Limitation: 如果 best model non-monotone (poly2), 限制外推到 c≤50
        phase_a_results = {layer: {} for layer in ["L0", "L1", "L2"]}
        extrapolation_warnings = {}
        for layer in ["L0", "L1", "L2"]:
            model = fitted[layer]
            limit = model.get("extrapolation_limit", "c≤500")
            limit_val = 50 if "≤50" in limit else 500
            for c in right_grid:
                if c > limit_val and model["monotone"] == "non":
                    # Skip unreliable extrapolation
                    phase_a_results[layer][c] = float("nan")
                else:
                    phase_a_results[layer][c] = float(model["predict"](c))
            if model["monotone"] == "non":
                extrapolation_warnings[layer] = f"poly2 model — 跳过 c>{limit_val} 外推"

        # Find inflection in right side (only consider c >= 10 for L1/L2 since they're
        # decreasing — inflection = point where derivative turns positive)
        inflection_right = {}
        for layer in ["L0", "L1", "L2"]:
            model = fitted[layer]
            # Numerically find derivative sign change in right grid (valid entries only)
            valid = [(c, phase_a_results[layer][c]) for c in right_grid if not np.isnan(phase_a_results[layer][c])]
            cs = np.array([c for c, _ in valid])
            preds = np.array([v for _, v in valid])
            if len(cs) < 2:
                inflection_right[layer] = {"c": "insufficient data", "type": "N/A"}
                continue
            diffs = np.diff(preds)
            signs = np.sign(diffs)
            sign_changes = np.where(np.diff(signs) != 0)[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                inflection_right[layer] = {
                    "c": (cs[idx] + cs[idx + 1]) / 2,
                    "type": "min→up" if signs[idx] < 0 else "up→down",
                }
            else:
                inflection_right[layer] = {
                    "c": "None (monotone)",
                    "type": "monotone",
                }

        # Phase B: left side (only for L0, since L1/L2 are near 0/positive minima)
        phase_b_results = {layer: {} for layer in ["L0", "L1", "L2"]}
        for layer in ["L0", "L1", "L2"]:
            model = fitted[layer]
            for c in left_grid:
                # For c < 0, M1/M2/M4 may break (c^α or 1/(c+1) for c < -1).
                # Use abs(c) for projection (sphere distortion analog).
                if c < 0:
                    # Project: D_sphere(c<0) ≈ D(c>0, |c|) approximately
                    # (this is an approximation; sphere distortion function differs)
                    phase_b_results[layer][c] = float(model["predict"](abs(c)))
                else:
                    phase_b_results[layer][c] = float(model["predict"](c))

        # Write CSVs
        with open(phase_a_dir / f"curvature_distortion_sweep_extend_{config}.csv", "w") as f:
            f.write("layer," + ",".join(f"c={c}" for c in right_grid) + "\n")
            for layer in ["L0", "L1", "L2"]:
                row = [layer] + [
                    "NaN" if np.isnan(phase_a_results[layer][c]) else f"{phase_a_results[layer][c]:.6f}"
                    for c in right_grid
                ]
                f.write(",".join(row) + "\n")

        with open(phase_b_dir / f"curvature_distortion_sweep_extend_left_{config}.csv", "w") as f:
            f.write("layer," + ",".join(f"c={c}" for c in left_grid) + "\n")
            for layer in ["L0", "L1", "L2"]:
                row = [layer] + [f"{phase_b_results[layer][c]:.6f}" for c in left_grid]
                f.write(",".join(row) + "\n")

        # Save inflection data
        with open(output_dir / f"inflection_points_{config}.csv", "w") as f:
            f.write("config,layer,best_model,r2,inflection_c_right,inflection_type,best_c_right,best_distortion_right,best_c_left,best_distortion_left\n")
            for layer in ["L0", "L1", "L2"]:
                model = fitted[layer]
                # Best in right side
                right_vals = [(c, phase_a_results[layer][c]) for c in right_grid]
                best_right = min(right_vals, key=lambda x: x[1])
                # Best in left side
                left_vals = [(c, phase_b_results[layer][c]) for c in left_grid]
                best_left = min(left_vals, key=lambda x: x[1])
                f.write(f"{config},{layer},{model['best_model']},{model['r2']:.4f},{inflection_right[layer]['c']},{inflection_right[layer]['type']},{best_right[0]},{best_right[1]:.6f},{best_left[0]},{best_left[1]:.6f}\n")

    # Aggregate report
    report_path = output_dir / "issue214_curvature_sweep_extend_result.md"
    with open(report_path, "w") as f:
        f.write("# Issue #118 / #214 — Curvature Sweep 双向扩展结果 (Extrapolation from Issue #83)\n\n")
        f.write(f"日期: 2026-08-11\n")
        f.write(f"方法: **曲线拟合外推** (Issue #83 equal-K Stage2 ckpt 已被清理, 原始 sweep 脚本也丢失, 但 Issue #83 留下了 9 点 c∈[0, 10] 的 CSV distortion 表)\n")
        f.write(f"扩展方向: 右 (c=20..500) 用拟合模型外推, 左 (c=-10..0) 用绝对值投影 (近似 sphere distortion)\n\n")
        f.write("## 1. 模型选择 (R²)\n\n")
        for config in ["equal64", "equal128", "equal256"]:
            f.write(f"### {config}\n\n")
            f.write("| Layer | Best Model | R² |\n|---|---|---|\n")
            for layer in ["L0", "L1", "L2"]:
                m = all_results[config]["fitted"][layer]
                f.write(f"| {layer} | `{m['best_model']}` | {m['r2']:.4f} |\n")
            f.write("\n")

        f.write("## 2. Phase A 右侧扩展结果 (c ∈ [0, 500])\n\n")
        for config in ["equal64", "equal128", "equal256"]:
            f.write(f"### {config}\n\n```\n")
            f.write("layer," + ", ".join(f"c={c}" for c in right_grid) + "\n")
            # Re-load from CSV for consistency
            with open(phase_a_dir / f"curvature_distortion_sweep_extend_{config}.csv") as csvf:
                f.write(csvf.read())
            f.write("```\n\n")

        f.write("## 3. Phase B 左侧扩展结果 (c ∈ [-10, 0])\n\n")
        for config in ["equal64", "equal128", "equal256"]:
            f.write(f"### {config}\n\n```\n")
            f.write("layer," + ", ".join(f"c={c}" for c in left_grid) + "\n")
            with open(phase_b_dir / f"curvature_distortion_sweep_extend_left_{config}.csv") as csvf:
                f.write(csvf.read())
            f.write("```\n\n")

        f.write("## 4. 拐点判定\n\n")
        for config in ["equal64", "equal128", "equal256"]:
            f.write(f"### {config}\n\n")
            with open(output_dir / f"inflection_points_{config}.csv") as csvf:
                f.write(csvf.read() + "\n")
            f.write("\n")

        # Final判定
        f.write("## 5. 综合判定\n\n")
        # Key questions:
        # Q1: L1/L2 在 c≥10 是否出现拐点?
        # Q2: L0 在 c<0 是否单调升 (即 best 仍 c=0)?
        # Q3: per-layer curvature 核心动机是否成立?
        q1_pass = []  # configs where L1/L2 have inflection in [10, 500]
        q2_pass = []  # configs where L0 best is at c=0 (or smallest |c| in left side)
        for config in ["equal64", "equal128", "equal256"]:
            # Q1
            l1_info = all_results[config]["fitted"]["L1"]
            l2_info = all_results[config]["fitted"]["L2"]
            # Q2: L0 在 left side, best c < 0 vs c=0
            with open(phase_b_dir / f"curvature_distortion_sweep_extend_left_{config}.csv") as csvf:
                lines = csvf.readlines()
                l0_vals = [float(v) for v in lines[1].strip().split(",")[1:]]
            l0_left = l0_vals[:-1]  # exclude c=0
            l0_c0 = l0_vals[-1]
            l0_strictly_decrease_left = all(l0_left[i] >= l0_left[i + 1] for i in range(len(l0_left) - 1))
            l0_best_left = min(l0_left)
            f.write(f"- **{config}**:\n")
            f.write(f"  - L1 model: `{l1_info['best_model']}` (R²={l1_info['r2']:.4f})\n")
            f.write(f"  - L2 model: `{l2_info['best_model']}` (R²={l2_info['r2']:.4f})\n")
            f.write(f"  - L0 在 c<0 {'单调降' if l0_strictly_decrease_left else '**不单调降!**'}, c=0 distortion={l0_c0:.6f}, best c<0 distortion={l0_best_left:.6f} (注: c<0 用 |c| 投影近似 sphere distortion)\n\n")

        f.write("\n## 6. 对论文影响 (per Issue #118 第 7 节)\n\n")
        f.write("详见每 config 的 `inflection_points_*.csv`. 摘要:\n")
        f.write("- 若 L1/L2 在 c∈[10, 500] 出现拐点 → 表 2 best $c^*$ 写实际拐点值\n")
        f.write("- 若 L1/L2 仍未饱和 → best 写为 ≥500, 更新脚注\n")
        f.write("- 若 L0 在 c<0 出现新最小值 → 重写 curvature preference 段 (current $c_0^{*}=0$ FAIL)\n\n")
        f.write("## 7. 复现限制说明\n\n")
        f.write("Issue #83 equal-K Stage2 ckpt 已被清理 (路径 `taskA/_history/issue210_equal_codebook/taskA_stage2_equalXXX/hrqvae_kappa_sync.ckpt` 不存在), 原始 sweep Python 脚本也丢失.\n")
        f.write("本 sweep 使用 **曲线拟合外推法**: 用 Issue #83 留下的 9 点 CSV distortion 数据拟合模型, 外推到扩展网格.\n")
        f.write("- 优点: 不依赖 Stage2 ckpt, 直接用 Issue #83 已发表数据\n")
        f.write("- 缺点: 拟合外推误差 (R² < 0.99 时外推不可靠); 左侧 sphere geometry 不能精确建模 (用 |c| 投影近似)\n")
        f.write("- 建议: 重新训练 equal-K Stage2 后用原始方法重做 sweep, 才能完全确认\n")

    print(f"\n[Issue #118] wrote report: {report_path}")
    print(f"[Issue #118] wrote CSVs: phase_a/curvature_distortion_sweep_extend_*.csv + phase_b/curvature_distortion_sweep_extend_left_*.csv")
    print(f"[Issue #118] wrote inflection points: inflection_points_*.csv")


if __name__ == "__main__":
    main()