#!/usr/bin/env python3
"""Task #22 Phase 1 — PM-RQ Toy Implementation.

输入: MCKG embedding (Task #99 重建版, M=3 subspaces, dim=64 each)
- 用 3 个 per-subspace embedding (球面 / 欧氏 / 双曲) 作为三段输入
- 不再做 dim split, 因为 per-subspace 已是 κ-专门的
- 单层 K=64, Toys 10K items subset, 100 epochs
- 验证 K^3 搜索可行 + 三分量稳定 + Kendall τ 矩阵

启动:
    python3 scripts/task22_pm_rq_phase1_toy.py \\
        --emb-path products/task99_mckg_rebuild/entity_embedding.pt \\
        --K 64 --epochs 100 --num-items 10000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-path", type=str,
                   default="products/task99_mckg_rebuild/entity_embedding.pt")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--dim", type=int, default=64,
                   help="每段维度 (MCKG per-subspace)")
    p.add_argument("--num-items", type=int, default=10000,
                   help="Toy 子集大小")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default="products/task22_pm_rq/phase1_toy")
    p.add_argument("--report-path", type=str,
                   default="reports/task22_pm_rq/phase1_toy_report.md")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.report_path)).mkdir(parents=True, exist_ok=True)

    print(f"[task22-p1] loading MCKG embedding from {args.emb_path}")
    emb = torch.load(args.emb_path, map_location="cpu", weights_only=False)

    # Per-subspace items: shape (M, N_items, dim)
    sub_item = emb["subspace_item"]  # (3, 11924, 64)
    M, N_total, dim = sub_item.shape
    assert M == 3, f"expected M=3, got {M}"
    assert dim == args.dim, f"expected dim={args.dim}, got {dim}"
    print(f"[task22-p1] subspace_item shape={sub_item.shape}")

    # Toy subset: 前 num-items 个
    N = min(args.num_items, N_total)
    sub_item = sub_item[:, :N, :]  # (3, N, 64)
    print(f"[task22-p1] using {N} items for Toy test")

    # ===== Product Manifold Codebook =====
    print(f"[task22-p1] init ProductManifoldCodebook K={args.K}")
    # 球面子空间 (κ=+): 单位球面归一化
    C_sphere = torch.randn(M, args.K, args.dim)
    C_sphere = C_sphere / C_sphere.norm(dim=-1, keepdim=True)
    # 欧氏子空间 (κ=0): 标准正态
    C_euclid = torch.randn(M, args.K, args.dim)
    # 双曲子空间 (κ=-): Poincaré ball (||x|| < 1)
    C_hyperbolic = torch.randn(M, args.K, args.dim) * 0.5  # 远离边界
    C_hyperbolic = torch.tanh(C_hyperbolic)  # 强制 ||·|| < 1

    # ===== Toy 训练: 重建 + 三码本利用率 =====
    print(f"[task22-p1] training {args.epochs} epochs, batch={args.batch_size}")
    metrics_history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        # 简单随机采样 batch
        idx = torch.randperm(N)[:args.batch_size]
        batch = sub_item[:, idx, :]  # (3, B, 64)

        # 三个子空间各自最近码字
        # batch[0..2] shape (B, dim), C_xxx[0..2] shape (K, dim)
        # 球面: cosine sim (1 - cos sim = 距离)
        d_s = 1 - torch.einsum("bd,kd->bk", batch[0], C_sphere[0])  # (B, K)
        idx_s = d_s.argmin(dim=-1)
        # 欧氏: 欧氏距离
        d_e = torch.cdist(batch[1].unsqueeze(0), C_euclid[1].unsqueeze(0)).squeeze(0)  # (B, K)
        idx_e = d_e.argmin(dim=-1)
        # 双曲: Poincaré ball 距离
        # d(x,y) = arccosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        x = batch[2].unsqueeze(1)  # (B, 1, 64)
        y = C_hyperbolic[2].unsqueeze(0)  # (1, K, 64)
        diff_sq = ((x - y) ** 2).sum(-1)  # (B, K)
        norm_x_sq = (batch[2] ** 2).sum(-1, keepdim=True)  # (B, 1)
        norm_y_sq = (C_hyperbolic[2] ** 2).sum(-1, keepdim=True).T  # (1, K)
        denom = (1 - norm_x_sq) * (1 - norm_y_sq)
        denom = torch.clamp(denom, min=1e-5)
        acosh_arg = 1 + 2 * diff_sq / denom
        acosh_arg = torch.clamp(acosh_arg, min=1.0 + 1e-5)
        d_h = torch.acosh(acosh_arg)
        idx_h = d_h.argmin(dim=-1)

        # Toy loss: 让码本接近 batch (简单 MSE, 不真做反向传播)
        loss_s = ((batch[0] - C_sphere[0, idx_s]) ** 2).sum(-1).mean()
        loss_e = ((batch[1] - C_euclid[1, idx_e]) ** 2).sum(-1).mean()
        loss_h = ((batch[2] - C_hyperbolic[2, idx_h]) ** 2).sum(-1).mean()
        loss = (loss_s + loss_e + loss_h) / 3.0

        # 码本更新: 简单 moving average (toy test 不真做 RSGD)
        with torch.no_grad():
            for m, (idxs, Cs) in enumerate(zip(
                [idx_s, idx_e, idx_h],
                [C_sphere[0], C_euclid[1], C_hyperbolic[2]]
            )):
                for k in range(args.K):
                    mask = (idxs == k)
                    if mask.sum() > 0:
                        Cs[k] = 0.9 * Cs[k] + 0.1 * batch[m, mask].mean(0)
                # 球面归一化
                if m == 0:
                    C_sphere[0] = C_sphere[0] / C_sphere[0].norm(dim=-1, keepdim=True)
                # 双曲强制 ||·|| < 1
                elif m == 2:
                    norms = C_hyperbolic[2].norm(dim=-1, keepdim=True)
                    C_hyperbolic[2] = C_hyperbolic[2] / torch.clamp(norms, min=0.95) * 0.95

        if epoch % 10 == 0:
            metrics_history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "util_s": len(torch.unique(idx_s)) / args.K,
                "util_e": len(torch.unique(idx_e)) / args.K,
                "util_h": len(torch.unique(idx_h)) / args.K,
                "elapsed_sec": time.time() - t0,
            })
            print(f"[epoch {epoch:3d}] loss={loss.item():.4f} "
                  f"util_s={metrics_history[-1]['util_s']:.3f} "
                  f"util_e={metrics_history[-1]['util_e']:.3f} "
                  f"util_h={metrics_history[-1]['util_h']:.3f}")

    # ===== D1.5 决策指标 =====
    # 1. Kendall τ 矩阵 (球面 vs 欧氏 vs 双曲 在 sample 上的排序相关性)
    from scipy.stats import kendalltau
    sample_idx = torch.randperm(N)[:1000]
    sample = sub_item[:, sample_idx, :]

    # 计算每个 sample 到所有码字的距离
    # sample[0..2] shape (1000, dim), C_xxx[0..2] shape (K, dim)
    d_s_full = 1 - torch.einsum("bd,kd->bk", sample[0], C_sphere[0])  # (1000, K)
    d_e_full = torch.cdist(sample[1].unsqueeze(0), C_euclid[1].unsqueeze(0)).squeeze(0)  # (1000, K)
    x = sample[2].unsqueeze(1)
    y = C_hyperbolic[2].unsqueeze(0)
    diff_sq = ((x - y) ** 2).sum(-1)
    norm_x_sq = (sample[2] ** 2).sum(-1, keepdim=True)
    norm_y_sq = (C_hyperbolic[2] ** 2).sum(-1, keepdim=True).T  # (1, K)
    denom = torch.clamp((1 - norm_x_sq) * (1 - norm_y_sq), min=1e-5)
    acosh_arg = torch.clamp(1 + 2 * diff_sq / denom, min=1.0 + 1e-5)
    d_h_full = torch.acosh(acosh_arg)

    tau_se = kendalltau(d_s_full.argmin(dim=-1), d_e_full.argmin(dim=-1)).statistic
    tau_sh = kendalltau(d_s_full.argmin(dim=-1), d_h_full.argmin(dim=-1)).statistic
    tau_eh = kendalltau(d_e_full.argmin(dim=-1), d_h_full.argmin(dim=-1)).statistic

    # 2. 搜索时间 per item
    t1 = time.time()
    for _ in range(100):
        idx = torch.randperm(N)[:args.batch_size]
        batch = sub_item[:, idx, :]
        _ = torch.cdist(batch[1].unsqueeze(0), C_euclid[1].unsqueeze(0))
    search_time_ms = (time.time() - t1) / 100 / args.batch_size * 1000

    final_util = metrics_history[-1]
    final_metrics = {
        "final_loss": final_util["loss"],
        "final_util_s": final_util["util_s"],
        "final_util_e": final_util["util_e"],
        "final_util_h": final_util["util_h"],
        "kendall_tau_se": tau_se,
        "kendall_tau_sh": tau_sh,
        "kendall_tau_eh": tau_eh,
        "search_time_ms_per_item": search_time_ms,
        "n_epochs": args.epochs,
        "total_time_sec": time.time() - t0,
    }

    # ===== D1.5 决策 =====
    decision_pass = (
        final_util["util_s"] > 0.8
        and final_util["util_e"] > 0.8
        and final_util["util_h"] > 0.8
        and abs(tau_se) < 0.7
        and abs(tau_sh) < 0.7
        and abs(tau_eh) < 0.7
        and search_time_ms < 50.0
    )
    decision_partial = (
        final_util["util_s"] > 0.5
        and final_util["util_e"] > 0.5
        and final_util["util_h"] > 0.5
    )

    if decision_pass:
        decision = "PROCEED Phase 2"
    elif decision_partial:
        decision = "INVESTIGATE (utilization OK but tau high or slow)"
    else:
        decision = "STOP / RESTRUCTURE"

    final_metrics["d15_decision"] = decision

    # ===== 输出 =====
    out_path = Path(args.output_dir) / "phase1_metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "history": metrics_history,
            "final": final_metrics,
        }, f, indent=2)
    print(f"[task22-p1] metrics saved → {out_path}")

    # 报告
    report = f"""# Task #22 Phase 1 — PM-RQ Toy Test Report

> **完成日期**: {time.strftime('%Y-%m-%d %H:%M')}
> **状态**: {'✅ PASS' if decision_pass else '⚠️ INVESTIGATE' if decision_partial else '❌ FAIL'}

## 配置
- emb-path: `{args.emb_path}`
- K={args.K}, dim={args.dim}, num_items={N}, epochs={args.epochs}
- seed={args.seed}

## D1.5 决策指标

| 指标 | 值 | 阈值 | 状态 |
|------|----|----|------|
| utilization sphere | {final_util['util_s']:.3f} | > 0.8 | {'✅' if final_util['util_s'] > 0.8 else '❌'} |
| utilization euclid | {final_util['util_e']:.3f} | > 0.8 | {'✅' if final_util['util_e'] > 0.8 else '❌'} |
| utilization hyperbolic | {final_util['util_h']:.3f} | > 0.8 | {'✅' if final_util['util_h'] > 0.8 else '❌'} |
| Kendall τ (s vs e) | {tau_se:.3f} | < 0.7 | {'✅' if abs(tau_se) < 0.7 else '❌'} |
| Kendall τ (s vs h) | {tau_sh:.3f} | < 0.7 | {'✅' if abs(tau_sh) < 0.7 else '❌'} |
| Kendall τ (e vs h) | {tau_eh:.3f} | < 0.7 | {'✅' if abs(tau_eh) < 0.7 else '❌'} |
| search time / item | {search_time_ms:.3f} ms | < 50 ms | {'✅' if search_time_ms < 50 else '❌'} |

## 训练曲线 (每 10 epoch)

| epoch | loss | util_s | util_e | util_h |
|-------|------|--------|--------|--------|
"""
    for m in metrics_history:
        report += f"| {m['epoch']} | {m['loss']:.4f} | {m['util_s']:.3f} | {m['util_e']:.3f} | {m['util_h']:.3f} |\n"

    report += f"""
## D1.5 决策: **{decision}**

{'✅ PROCEED to Phase 2 (Full-Scale Single-Layer, K=256, 全 Toys)' if decision_pass
 else '⚠️ 进一步诊断后再决定 / 调整超参' if decision_partial
 else '❌ 任务提前终止, 写终局 verdict'}

## 产物
- metrics JSON: `{out_path}`
- 训练总时长: {time.time() - t0:.1f} 秒
"""

    with open(args.report_path, "w") as f:
        f.write(report)
    print(f"[task22-p1] report saved → {args.report_path}")
    print()
    print("=" * 60)
    print(f"D1.5 决策: **{decision}**")
    print("=" * 60)


if __name__ == "__main__":
    main()
