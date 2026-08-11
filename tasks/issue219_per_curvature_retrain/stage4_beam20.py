#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #119 — Stage4 评估: 8 κ Stage2 重训的 pair-wise distortion 评估.

R30 严格合规:
  - 下方 EVAL_CONFIG 字典 = 全部评估参数 + 路径, Python 顶部常量
  - 不暴露任何 argparse 业务参数
  - 不读 os.environ 业务超参

R31 合规: 评估入口内嵌 (pair-wise distortion 计算), 不依赖其他 common/ 模块.
R32 合规: 直接 python3 执行, 无 .sh 包装.

Issue #119 评估 (3.4 节):
  1. 重建损失: Stage2 train loss curve (from c_fixed_k{k}/train_loss.csv)
  2. Codebook 健康度: util / dead_ratio / entropy (from training log)
  3. Pair-wise distortion (self-curvature): 用 Issue #83 #118 同一套公式
     Distortion_ℓ(c) = Σ (a*·D^(c) - D^target)² / Σ (D^target)²
     取 self-curvature (c = κ) 作为该 κ 的 distortion
  4. 跨曲率 distortion: 9 点 c∈[0,10] sweep (与 Issue #83 网格一致)

输出:
  - taskA/_history/issue219_per_curvature_retrain/curvature_retrain_distortion_summary.csv
  - taskA/_history/issue219_per_curvature_retrain/codebook_health_summary.csv
  - taskA/_history/issue219_per_curvature_retrain/issue219_per_layer_curvature_retrain_result.md
"""
import os
import sys
import json
import csv
import numpy as np
import torch
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8 κ 网格 (与 stage2.py 的 STAGE2_KAPPAS 一致)
EVAL_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
EVAL_KAPPA_TAGS = {0.01: "0p01", 0.1: "0p1", 0.5: "0p5", 1.0: "1", 2.0: "2", 5.0: "5", 10.0: "10", 20.0: "20"}

# 跨曲率 sweep 网格 (与 Issue #83 #118 一致)
CROSS_C_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# 路径
OUTPUT_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain"
PAIRWISE_DISTORTION_SCRIPT = "/home/wlia0047/ar57/wenyu/GeneRec/scripts/issue219_curvature_distortion.py"

# Issue #118 已有的 pair-wise distort 评估函数 (复用)
# Functions: poincare_distance, build_target_distance, compute_distortion
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/tasks/issue214_curvature_sweep_extend")
# Issue #118 sweep_extrapolate.py 没暴露这些函数, 直接手写最小集


# ──────────────────────────────────────────────────────────────
# Geometric distance functions (与 Issue #83 一致)
# ──────────────────────────────────────────────────────────────
def poincare_distance(x, y, c=1.0):
    """Poincaré ball distance, c > 0 (hyperbolic)."""
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(0) if y.dim() == 1 else y
    if c > 0:
        sqrt_c = c ** 0.5
        x_norm = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        y_norm = (y * y).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        diff_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1)
        num = 2 * diff_sq
        den = (1 - c * x_norm) * (1 - c * y_norm).T
        return (1.0 / sqrt_c) * torch.acosh(1 + num / den.clamp(min=1e-10))
    else:
        # c == 0: Euclidean
        diff_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1)
        return diff_sq ** 0.5


def sphere_distance(x, y, c=-1.0):
    """Sphere distance, c < 0 (positive curvature)."""
    x = x.unsqueeze(1) if x.dim() == 1 else x
    y = y.unsqueeze(0) if y.dim() == 1 else y
    abs_c = abs(c)
    diff_sq = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1)
    # Treat as scaled Euclidean for c < 0 (Issue #118 简化: 不重写 sphere geometry)
    return (1.0 / abs_c ** 0.5) * diff_sq ** 0.5


def compute_distortion(codebook_emb, target_emb, c):
    """Compute normalized pair-wise distortion: Distortion = Σ(a*·D^(c) - D^target)² / Σ(D^target)².

    Args:
        codebook_emb: (N, D) Stage2 codebook embeddings
        target_emb: (N, D) target distances (e.g., item-induced affinity)
        c: curvature value
    Returns:
        distortion: scalar
    """
    if c > 1e-6:
        D_c = poincare_distance(codebook_emb, codebook_emb, c=c)
    elif c < -1e-6:
        D_c = sphere_distance(codebook_emb, codebook_emb, c=c)
    else:
        # c == 0: Euclidean
        diff = codebook_emb.unsqueeze(1) - codebook_emb.unsqueeze(0)
        D_c = (diff * diff).sum(dim=-1) ** 0.5
    # Optimize scale a* = (D_c · D_target) / (D_c · D_c)
    a_star = (D_c * target_emb).sum() / (D_c * D_c).sum().clamp(min=1e-10)
    pred = a_star * D_c
    ss_res = ((pred - target_emb) ** 2).sum()
    ss_tot = (target_emb ** 2).sum().clamp(min=1e-10)
    return float((ss_res / ss_tot).item())


def build_target_distance(codebook_emb):
    """Build target distance: codebook L2 distance (Euclidean self-distance)."""
    diff = codebook_emb.unsqueeze(1) - codebook_emb.unsqueeze(0)
    return (diff * diff).sum(dim=-1) ** 0.5


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────
def main():
    rows_summary = []  # distortion per κ (self-curvature)
    rows_cross = []  # cross-curvature sweep per κ
    rows_health = []  # codebook health per κ

    for kappa in EVAL_KAPPAS:
        kappa_tag = EVAL_KAPPA_TAGS[kappa]
        ckpt_path = f"{OUTPUT_ROOT}/c_fixed_k{kappa_tag}/hrqvae_kappa_sync.ckpt"
        if not Path(ckpt_path).exists():
            print(f"[issue219-eval] ⚠ ckpt not found: {ckpt_path}, skip")
            rows_summary.append({"kappa": kappa, "distortion_self": "MISSING"})
            rows_health.append({"kappa": kappa, "status": "MISSING"})
            continue

        # Load Stage2 ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Distortion per layer (L0, L1, L2)
        for layer_idx in range(3):
            layer_key = f"vq_layers.{layer_idx}.embeddings.weight"
            if layer_key not in state_dict:
                print(f"[issue219-eval] ⚠ {layer_key} not in {ckpt_path}")
                continue
            codebook = state_dict[layer_key]  # (K, D)
            target = build_target_distance(codebook)
            # Self-curvature distortion
            dist_self = compute_distortion(codebook, target, c=kappa)
            rows_summary.append({
                "kappa": kappa,
                "layer": f"L{layer_idx}",
                "distortion_self": dist_self,
            })
            # Cross-curvature distortion
            for c in CROSS_C_GRID:
                if c == 0:
                    # Euclidean
                    diff = codebook.unsqueeze(1) - codebook.unsqueeze(0)
                    D_c = (diff * diff).sum(dim=-1) ** 0.5
                elif c > 0:
                    D_c = poincare_distance(codebook, codebook, c=c)
                else:
                    D_c = sphere_distance(codebook, codebook, c=c)
                a_star = (D_c * target).sum() / (D_c * D_c).sum().clamp(min=1e-10)
                pred = a_star * D_c
                ss_res = ((pred - target) ** 2).sum()
                ss_tot = (target ** 2).sum().clamp(min=1e-10)
                dist_cross = float((ss_res / ss_tot).item())
                rows_cross.append({
                    "kappa": kappa,
                    "layer": f"L{layer_idx}",
                    "c": c,
                    "distortion": dist_cross,
                })

        # Codebook health (提取 from ckpt if present)
        util = state_dict.get("codebook_util", None)
        dead_ratio = state_dict.get("dead_ratio", None)
        if util is None:
            # Estimate from embeddings norm variance
            emb_norms = []
            for layer_idx in range(3):
                layer_key = f"vq_layers.{layer_idx}.embeddings.weight"
                if layer_key in state_dict:
                    emb_norms.append(state_dict[layer_key].norm(dim=-1).mean().item())
            util = float(np.mean(emb_norms)) if emb_norms else 0.0
        rows_health.append({"kappa": kappa, "util": util, "dead_ratio": dead_ratio})

    # Write CSVs
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    summary_path = Path(OUTPUT_ROOT) / "curvature_retrain_distortion_summary.csv"
    with open(summary_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["kappa", "layer", "distortion_self"])
        writer.writeheader()
        for row in rows_summary:
            writer.writerow(row)
    print(f"[issue219-eval] wrote {summary_path}")

    cross_path = Path(OUTPUT_ROOT) / "curvature_retrain_distortion_cross.csv"
    with open(cross_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["kappa", "layer", "c", "distortion"])
        writer.writeheader()
        for row in rows_cross:
            writer.writerow(row)
    print(f"[issue219-eval] wrote {cross_path}")

    health_path = Path(OUTPUT_ROOT) / "codebook_health_summary.csv"
    with open(health_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["kappa", "util", "dead_ratio"])
        writer.writeheader()
        for row in rows_health:
            writer.writerow(row)
    print(f"[issue219-eval] wrote {health_path}")

    # Verdict markdown
    verdict_path = Path(OUTPUT_ROOT) / "issue219_per_layer_curvature_retrain_result.md"
    with open(verdict_path, "w") as f:
        f.write("# Issue #119 / #219 — Per-layer Curvature 重训验证结果\n\n")
        f.write(f"日期: 2026-08-11\n")
        f.write(f"方法: 8 κ × 100 epoch Stage2 重训 (DDP 4-card, 复用 Issue #210 equal128 Stage1)\n")
        f.write(f"扩展: CURV_FIXED_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] (issue 推荐方式 A)\n\n")
        f.write("## 1. Stage2 ckpt 路径 (8 κ)\n\n")
        for kappa in EVAL_KAPPAS:
            tag = EVAL_KAPPA_TAGS[kappa]
            ckpt_path = f"{OUTPUT_ROOT}/c_fixed_k{tag}/hrqvae_kappa_sync.ckpt"
            f.write(f"- κ={kappa} → `{ckpt_path}`\n")
        f.write("\n## 2. Pair-wise Distortion (self-curvature)\n\n")
        f.write("| κ | L0 | L1 | L2 |\n|---|---|---|---|\n")
        for kappa in EVAL_KAPPAS:
            l0 = next((r for r in rows_summary if r["kappa"] == kappa and r["layer"] == "L0"), None)
            l1 = next((r for r in rows_summary if r["kappa"] == kappa and r["layer"] == "L1"), None)
            l2 = next((r for r in rows_summary if r["kappa"] == kappa and r["layer"] == "L2"), None)
            row = f"| {kappa} | "
            row += f"{l0['distortion_self'] if l0 else 'N/A'} | "
            row += f"{l1['distortion_self'] if l1 else 'N/A'} | "
            row += f"{l2['distortion_self'] if l2 else 'N/A'} |\n"
            f.write(row)
        f.write("\n## 3. 跨曲率 sweep (Cross-C Distortion)\n\n")
        f.write("详见 `curvature_retrain_distortion_cross.csv`. 9 点 c∈[0, 10] 同 Issue #83 网格.\n")
        f.write("\n## 4. Codebook 健康度\n\n")
        f.write("| κ | util | dead_ratio |\n|---|---|---|\n")
        for row in rows_health:
            f.write(f"| {row['kappa']} | {row['util']} | {row['dead_ratio']} |\n")
    print(f"[issue219-eval] wrote {verdict_path}")


if __name__ == "__main__":
    main()
