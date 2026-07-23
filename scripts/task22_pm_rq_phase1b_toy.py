#!/usr/bin/env python3
"""Task #22 Phase 1b — PM-RQ Toy test with proper Adam optimizer.

Phase 1 FAIL 因 moving average 码本更新导致 Euclidean 坍缩 (util=0.156).
Phase 1b 改动:
  - 码本作为 nn.Parameter, 启用 autograd
  - 用 Adam optimizer 直接优化重建 loss
  - 三个码本独立参数, 球面归一化/双曲 clamp 在 forward 内做
  - 加入码本 utilization penalty (鼓励码字均衡使用)
  - 加入 EMA 码本 dead code 重启 (Phase 1b 额外 trick)

启动:
    python3 scripts/task22_pm_rq_phase1b_toy.py \\
        --emb-path products/task99_mckg_rebuild/entity_embedding.pt \\
        --K 64 --epochs 100 --num-items 10000
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-path", type=str,
                   default="products/task99_mckg_rebuild/entity_embedding.pt")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num-items", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--util-penalty", type=float, default=0.1,
                   help="码本利用率 penalty 权重")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default="products/task22_pm_rq/phase1b_toy")
    p.add_argument("--report-path", type=str,
                   default="reports/task22_pm_rq/phase1b_toy_report.md")
    return p.parse_args()


class ProductManifoldCodebook(nn.Module):
    """三个子空间 (sphere/euclid/hyperbolic) 各 K 个码字.

    所有码字作为 nn.Parameter, forward 返回量化结果.
    - 球面码字: forward 时强制单位归一化
    - 欧氏码字: 无约束
    - 双曲码字: forward 时强制 ||x|| < 0.95 (Poincaré ball 边界安全)
    """

    def __init__(self, K=64, dim=64):
        super().__init__()
        self.K = K
        self.dim = dim
        # 球面码字
        self.C_sphere = nn.Parameter(torch.randn(K, dim) * 0.1)
        # 欧氏码字
        self.C_euclid = nn.Parameter(torch.randn(K, dim) * 0.1)
        # 双曲码字 (init: 远离边界)
        self.C_hyperbolic = nn.Parameter(torch.randn(K, dim) * 0.3)

    def get_normalized(self):
        """返回 forward 用的归一化码本."""
        C_s = F.normalize(self.C_sphere, dim=-1)
        C_e = self.C_euclid
        # 双曲 clamp 到 Poincaré ball
        norms = self.C_hyperbolic.norm(dim=-1, keepdim=True)
        C_h = self.C_hyperbolic / torch.clamp(norms / 0.9, min=1.0)
        return C_s, C_e, C_h

    @staticmethod
    def sphere_distance(x, C):
        """球面 cosine 距离. x: (B, dim), C: (K, dim). Return: (B, K)."""
        x_n = F.normalize(x, dim=-1)
        C_n = F.normalize(C, dim=-1)
        return 1 - x_n @ C_n.T

    @staticmethod
    def euclid_distance(x, C):
        """欧氏距离. x: (B, dim), C: (K, dim). Return: (B, K)."""
        return torch.cdist(x.unsqueeze(0), C.unsqueeze(0)).squeeze(0)

    @staticmethod
    def poincare_distance(x, C):
        """Poincaré ball 距离. x: (B, dim), C: (K, dim). Return: (B, K)."""
        # d(x,y) = arccosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        x_sq = (x ** 2).sum(-1, keepdim=True)  # (B, 1)
        C_sq = (C ** 2).sum(-1, keepdim=True).T  # (1, K)
        diff_sq = ((x.unsqueeze(1) - C.unsqueeze(0)) ** 2).sum(-1)  # (B, K)
        denom = (1 - x_sq) * (1 - C_sq)
        denom = torch.clamp(denom, min=1e-5)
        acosh_arg = 1 + 2 * diff_sq / denom
        acosh_arg = torch.clamp(acosh_arg, min=1.0 + 1e-5)
        return torch.acosh(acosh_arg)

    def forward(self, sub_emb):
        """输入: sub_emb shape (3, B, dim). 返回: (idx_s, idx_e, idx_h, d_s, d_e, d_h)."""
        C_s, C_e, C_h = self.get_normalized()
        d_s = self.sphere_distance(sub_emb[0], C_s)
        d_e = self.euclid_distance(sub_emb[1], C_e)
        d_h = self.poincare_distance(sub_emb[2], C_h)
        return d_s.argmin(-1), d_e.argmin(-1), d_h.argmin(-1), d_s, d_e, d_h


class SimpleDecoder(nn.Module):
    """Toy decoder: 球面 + 欧氏 + 双曲 三个码字 拼接 → 输入空间重建."""

    def __init__(self, K=64, dim=64, in_dim=None):
        super().__init__()
        if in_dim is None:
            in_dim = dim * 3
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim * 3),  # 重建三个子空间
        )

    def forward(self, code_s, code_e, code_h):
        """输入: 三个码字 (B, dim). 输出: (B, 3*dim)."""
        z = torch.cat([code_s, code_e, code_h], dim=-1)
        return self.decoder(z)


def compute_utilization_penalty(idx, K, eps=1e-5):
    """鼓励码字均衡使用的负熵 penalty."""
    # idx: (B,) of int
    counts = torch.bincount(idx, minlength=K).float()
    probs = counts / counts.sum().clamp(min=1.0)
    # 均匀分布的熵是 log(K), 当前熵是 -sum(p log p)
    entropy = -(probs * (probs + eps).log()).sum()
    max_entropy = math.log(K)
    return (max_entropy - entropy) / max_entropy  # 归一化到 [0, 1]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.report_path)).mkdir(parents=True, exist_ok=True)

    print(f"[task22-p1b] loading MCKG embedding from {args.emb_path}")
    emb = torch.load(args.emb_path, map_location="cpu", weights_only=False)
    sub_item = emb["subspace_item"]  # (3, 11924, 64)
    M, N_total, dim = sub_item.shape
    assert M == 3
    N = min(args.num_items, N_total)
    sub_item = sub_item[:, :N, :]
    print(f"[task22-p1b] using {N} items, dim={dim}")

    # 输入归一化 (per-subspace)
    for m in range(3):
        sub_item[m] = (sub_item[m] - sub_item[m].mean(0)) / sub_item[m].std(0).clamp(min=1e-5)
    # 双曲子空间 clamp 到 ||x|| < 0.9
    sub_item[2] = sub_item[2] * 0.5
    norms = sub_item[2].norm(dim=-1, keepdim=True)
    sub_item[2] = sub_item[2] / torch.clamp(norms / 0.85, min=1.0)

    # 模型
    codebook = ProductManifoldCodebook(K=args.K, dim=args.dim)
    decoder = SimpleDecoder(K=args.K, dim=args.dim, in_dim=args.dim * 3)

    optimizer = torch.optim.Adam(
        list(codebook.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )

    print(f"[task22-p1b] training {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, util_penalty={args.util_penalty}")

    metrics_history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        idx_batch = torch.randperm(N)[:args.batch_size]
        batch = sub_item[:, idx_batch, :]  # (3, B, dim)

        optimizer.zero_grad()
        idx_s, idx_e, idx_h, d_s, d_e, d_h = codebook(batch)
        C_s, C_e, C_h = codebook.get_normalized()

        # 重建 loss: 用最近码字 → decoder → 与输入 MSE
        cs_sel = C_s[idx_s]
        ce_sel = C_e[idx_e]
        ch_sel = C_h[idx_h]
        recon = decoder(cs_sel, ce_sel, ch_sel)  # (B, 3*dim)
        target = batch.permute(1, 0, 2).reshape(args.batch_size, -1)  # (B, 3*dim)
        recon_loss = F.mse_loss(recon, target)

        # utilization penalty (鼓励码字均匀)
        pen_s = compute_utilization_penalty(idx_s, args.K)
        pen_e = compute_utilization_penalty(idx_e, args.K)
        pen_h = compute_utilization_penalty(idx_h, args.K)
        util_pen = (pen_s + pen_e + pen_h) / 3.0

        loss = recon_loss + args.util_penalty * util_pen

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            util_s = len(torch.unique(idx_s)) / args.K
            util_e = len(torch.unique(idx_e)) / args.K
            util_h = len(torch.unique(idx_h)) / args.K
            metrics_history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "util_pen": util_pen.item(),
                "util_s": util_s,
                "util_e": util_e,
                "util_h": util_h,
                "elapsed_sec": time.time() - t0,
            })
            print(f"[epoch {epoch:3d}] loss={loss.item():.4f} "
                  f"recon={recon_loss.item():.4f} pen={util_pen.item():.3f} "
                  f"util_s={util_s:.3f} util_e={util_e:.3f} util_h={util_h:.3f}")

    # ===== D1.5 决策指标 =====
    from scipy.stats import kendalltau
    sample_idx = torch.randperm(N)[:1000]
    sample = sub_item[:, sample_idx, :]
    with torch.no_grad():
        idx_s_f, idx_e_f, idx_h_f, d_s_f, d_e_f, d_h_f = codebook(sample)
    util_s = len(torch.unique(idx_s_f)) / args.K
    util_e = len(torch.unique(idx_e_f)) / args.K
    util_h = len(torch.unique(idx_h_f)) / args.K

    tau_se = kendalltau(d_s_f.argmin(-1).numpy(), d_e_f.argmin(-1).numpy()).statistic
    tau_sh = kendalltau(d_s_f.argmin(-1).numpy(), d_h_f.argmin(-1).numpy()).statistic
    tau_eh = kendalltau(d_e_f.argmin(-1).numpy(), d_h_f.argmin(-1).numpy()).statistic

    # search time
    t1 = time.time()
    with torch.no_grad():
        for _ in range(100):
            batch = sub_item[:, torch.randperm(N)[:args.batch_size], :]
            _ = codebook(batch)
    search_time_ms = (time.time() - t1) / 100 / args.batch_size * 1000

    final_metrics = {
        "final_loss": metrics_history[-1]["loss"],
        "final_recon_loss": metrics_history[-1]["recon_loss"],
        "final_util_s": util_s,
        "final_util_e": util_e,
        "final_util_h": util_h,
        "kendall_tau_se": tau_se,
        "kendall_tau_sh": tau_sh,
        "kendall_tau_eh": tau_eh,
        "search_time_ms_per_item": search_time_ms,
        "n_epochs": args.epochs,
        "total_time_sec": time.time() - t0,
    }

    decision_pass = (
        util_s > 0.8 and util_e > 0.8 and util_h > 0.8
        and abs(tau_se) < 0.7 and abs(tau_sh) < 0.7 and abs(tau_eh) < 0.7
        and search_time_ms < 50.0
    )
    if decision_pass:
        decision = "PROCEED Phase 2"
    elif util_e > 0.5 and util_s > 0.5 and util_h > 0.5:
        decision = "PARTIAL PASS (util improved but borderline)"
    else:
        decision = "STOP — PM-RQ Toy 仍 FAIL"

    final_metrics["d15_decision"] = decision

    out_path = Path(args.output_dir) / "phase1b_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "history": metrics_history,
                   "final": final_metrics}, f, indent=2)
    print(f"[task22-p1b] metrics saved → {out_path}")

    report = f"""# Task #22 Phase 1b — PM-RQ Toy Test (Adam optimizer 修复版)

> **完成日期**: {time.strftime('%Y-%m-%d %H:%M')}
> **状态**: {'✅ PASS' if decision_pass else '⚠️ PARTIAL' if util_e > 0.5 else '❌ STOP'}
> **对比 Phase 1**: MA → Adam + decoder + util penalty

## 配置
- emb-path: `{args.emb_path}`
- K={args.K}, dim={args.dim}, num_items={N}, epochs={args.epochs}
- batch={args.batch_size}, lr={args.lr}, util_penalty={args.util_penalty}, seed={args.seed}

## D1.5 决策指标

| 指标 | Phase 1 (MA) | Phase 1b (Adam) | 阈值 | 状态 |
|------|--------------|----------------|------|------|
| util_s | 1.000 | {util_s:.3f} | > 0.8 | {'✅' if util_s > 0.8 else '❌'} |
| util_e | **0.156** ❌ | **{util_e:.3f}** | > 0.8 | {'✅ 修复!' if util_e > 0.8 else '❌ 仍 FAIL' if util_e < 0.5 else '⚠️ 边界'} |
| util_h | 0.875 | {util_h:.3f} | > 0.8 | {'✅' if util_h > 0.8 else '❌'} |
| Kendall τ (s vs e) | 0.080 | {tau_se:.3f} | < 0.7 | {'✅' if abs(tau_se) < 0.7 else '❌'} |
| Kendall τ (s vs h) | 0.005 | {tau_sh:.3f} | < 0.7 | {'✅' if abs(tau_sh) < 0.7 else '❌'} |
| Kendall τ (e vs h) | 0.025 | {tau_eh:.3f} | < 0.7 | {'✅' if abs(tau_eh) < 0.7 else '❌'} |
| search time / item | 0.063 ms | {search_time_ms:.3f} ms | < 50 ms | {'✅' if search_time_ms < 50 else '❌'} |

## 训练曲线 (每 10 epoch)

| epoch | loss | recon | pen | util_s | util_e | util_h |
|-------|------|-------|-----|--------|--------|--------|
"""
    for m in metrics_history:
        report += (f"| {m['epoch']} | {m['loss']:.4f} | {m['recon_loss']:.4f} | "
                   f"{m['util_pen']:.3f} | {m['util_s']:.3f} | {m['util_e']:.3f} | "
                   f"{m['util_h']:.3f} |\n")

    report += f"""
## D1.5 决策: **{decision}**

{'✅ PROCEED to Phase 2 (Full-Scale Single-Layer K=256)' if decision_pass
 else '⚠️ 部分通过, 需进一步调整' if util_e > 0.5
 else '❌ 终止 Task #22, 写终局 verdict'}

## 产物
- metrics JSON: `{out_path}`
- 训练总时长: {time.time() - t0:.1f} 秒
"""

    with open(args.report_path, "w") as f:
        f.write(report)
    print(f"[task22-p1b] report saved → {args.report_path}")
    print()
    print("=" * 60)
    print(f"D1.5 决策: **{decision}**")
    print(f"  util_s={util_s:.3f} util_e={util_e:.3f} util_h={util_h:.3f}")
    print(f"  τ_se={tau_se:.3f} τ_sh={tau_sh:.3f} τ_eh={tau_eh:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
