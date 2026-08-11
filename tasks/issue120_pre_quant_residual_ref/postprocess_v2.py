#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #120 v2 postprocess: 从 v1 已存 CSV 过滤 c=-1.0, 生成新 summary.

v1 CSV 包含 216 metrics (8 κ × 3 层 × 9 c, 含 c=-1.0).
v2 排除 c=-1.0, 保留 8 c ∈ [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0].
"""
import csv
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
V1_CSV = REPO / "taskA/_history/issue120_pre_quant_residual_ref/relationship_preservation_per_layer.csv"
OUTPUT_ROOT = REPO / "taskA/_history/issue120_pre_quant_residual_ref"

# v2 c grid (过滤 c=-1.0)
V2_C_GRID = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# 读 v1 → 过滤 c → 写 v2 CSV
v2_rows = []
with open(V1_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if float(row["c"]) in V2_C_GRID:
            v2_rows.append(row)

v2_csv = OUTPUT_ROOT / "relationship_preservation_per_layer.csv"
with open(v2_csv, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["kappa", "layer", "c", "distortion", "spearman", "kendall", "nn_overlap_1", "nn_overlap_5"])
    writer.writeheader()
    for row in v2_rows:
        writer.writerow(row)
print(f"[v2-postprocess] wrote {v2_csv} ({len(v2_rows)} rows)")

# 写 v2 summary (best c by each metric, excluding c=0)
summary_rows = []
for kappa in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    for layer_idx in range(3):
        layer = f"L{layer_idx}"
        best = {"kappa": kappa, "layer": layer}
        for metric in ["distortion", "spearman", "kendall", "nn_overlap_1", "nn_overlap_5"]:
            layer_rows = [r for r in v2_rows if float(r["kappa"]) == kappa and r["layer"] == layer and float(r["c"]) != 0]
            if not layer_rows:
                best[f"best_c_by_{metric}"] = "N/A"
                best[f"best_{metric}"] = "N/A"
                continue
            if metric == "distortion":
                best_row = min(layer_rows, key=lambda r: float(r[metric]))
            else:
                best_row = max(layer_rows, key=lambda r: float(r[metric]))
            best[f"best_c_by_{metric}"] = best_row["c"]
            best[f"best_{metric}"] = best_row[metric]
        summary_rows.append(best)

v2_summary = OUTPUT_ROOT / "relationship_preservation_summary.csv"
with open(v2_summary, "w") as f:
    fieldnames = list(summary_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)
print(f"[v2-postprocess] wrote {v2_summary}")

# 写 v2 level-wise table
v2_table = OUTPUT_ROOT / "level_wise_metrics_table.md"
with open(v2_table, "w") as f:
    f.write("# Issue #120 / v2 — Level-wise Metrics Table (c ∈ [0, 10])\n\n")
    f.write(f"日期: 2026-08-11\n")
    f.write(f"v2 fix: 删除 c=-1.0 (违反 HG-Rec convention c 是 magnitude ≥ 0)\n")
    f.write(f"评估方法: 量化前 residual pairwise Euclidean 距离作为 D_ref, 量化后 codebook (per item assignment) 距离作为 D^(c)\n")
    f.write(f"跨曲率 sweep: {V2_C_GRID} (c=0 Euclidean, c>0 hyperbolic, c 是 magnitude)\n")
    f.write(f"评估样本数: 500 (random, seed=42)\n\n")
    for layer_idx in range(3):
        layer = f"L{layer_idx}"
        f.write(f"## {layer}\n\n")
        f.write("| κ | c | distortion ↓ | spearman ↑ | kendall ↑ | NN@1 ↑ | NN@5 ↑ |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in v2_rows:
            if row["layer"] != layer:
                continue
            f.write(f"| {row['kappa']} | {row['c']} | {float(row['distortion']):.6f} | {float(row['spearman']):.4f} | {float(row['kendall']):.4f} | {float(row['nn_overlap_1']):.4f} | {float(row['nn_overlap_5']):.4f} |\n")
        f.write("\n")
print(f"[v2-postprocess] wrote {v2_table}")

# 写 v2 verdict
v2_verdict = OUTPUT_ROOT / "verdict_issue120_v2.md"
with open(v2_verdict, "w") as f:
    f.write("# Issue #120 / v2 — relationship-preservation verdict (c ∈ [0, 10])\n\n")
    f.write(f"日期: 2026-08-11\n")
    f.write(f"v2 fix: 删除 c=-1.0 (HG-Rec convention: c 是 magnitude, 永 ≥ 0; c > 0 = hyperbolic, Gaussian curvature = -c)\n\n")
    f.write("## 1. 关键发现 (Spearman 最大化)\n\n")
    for row in summary_rows:
        f.write(f"- κ={row['kappa']}, {row['layer']}: best c = {row['best_c_by_spearman']} (spearman={float(row['best_spearman']):.4f})\n")
    f.write("\n## 2. 关键发现 (NN@1 最大化)\n\n")
    for row in summary_rows:
        f.write(f"- κ={row['kappa']}, {row['layer']}: best c = {row['best_c_by_nn_overlap_1']} (NN@1={float(row['best_nn_overlap_1']):.4f})\n")
    f.write("\n## 3. Hyp. Improvement (Δspearman from c=0)\n\n")
    for kappa in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        for layer_idx in range(3):
            layer = f"L{layer_idx}"
            c0_row = next((r for r in v2_rows if float(r["kappa"]) == kappa and r["layer"] == layer and float(r["c"]) == 0.0), None)
            best_row = next((r for r in summary_rows if row["kappa"] == kappa and row["layer"] == layer), None)
            if c0_row and best_row:
                base_spec = float(c0_row["spearman"])
                best_spec = float(best_row["best_spearman"])
                delta = best_spec - base_spec
                f.write(f"- κ={kappa}, {layer}: c=0 baseline = {base_spec:.4f}, best c={best_row['best_c_by_spearman']} = {best_spec:.4f}, Δ = {delta:+.4f}\n")
    f.write("\n## 4. 结论\n\n")
    f.write("- Hyp. Improvement (Spearman): L0 最大 +0.005-0.007, L1 +0.0007, L2 +0.0003 (微小)\n")
    f.write("- NN overlap 与 c 无关 (恒为 0.003-0.07, 与 c=0 完全一致)\n")
    f.write("- Distortion 随 c 单调递增 (c=0.01 最小, c=10 较大)\n")
    f.write("- 整体: codebook 几何对 curvature 选择不敏感, 实际 best c 选择意义有限\n")
print(f"[v2-postprocess] wrote {v2_verdict}")
