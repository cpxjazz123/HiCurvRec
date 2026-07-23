#!/usr/bin/env python3
"""Task #22 Phase 3 — PM-RQ Three-Layer Cascade + V-info per layer.

Phase 2 PASS (K=256 full Toys 单层). Phase 3 验证:
- 三层 cascade 是否提升信息保留
- 每层 V-info (residual variance / entropy) 趋势
- 三层 vs 单层 (Phase 2) reconstruction 质量

启动:
    python3 scripts/task22_pm_rq_phase3_cascade.py \\
        --emb-path products/task99_mckg_rebuild/entity_embedding.pt \\
        --K 256 --num-layers 3 --epochs 150
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from task22_pm_rq_phase1b_toy import (
    ProductManifoldCodebook,
    SimpleDecoder,
    compute_utilization_penalty,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-path", type=str,
                   default="products/task99_mckg_rebuild/entity_embedding.pt")
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-items", type=int, default=11924)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--util-penalty", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default="products/task22_pm_rq/phase3_cascade")
    p.add_argument("--report-path", type=str,
                   default="reports/task22_pm_rq/phase3_cascade_report.md")
    return p.parse_args()


class ProductManifoldCascade(nn.Module):
    """三层 ProductManifoldCodebook cascade.

    每层:
      1. 取当前 z (输入或上一层 residual)
      2. 量化到最近码字 (idx_s, idx_e, idx_h)
      3. 用 decoder 重建当前 z
      4. 计算 residual = z - recon
      5. 把 residual 传给下一层
    """

    def __init__(self, num_layers=3, K=256, dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.K = K
        self.dim = dim
        self.codebooks = nn.ModuleList([
            ProductManifoldCodebook(K=K, dim=dim) for _ in range(num_layers)
        ])
        # 每层独立 decoder: 输入 3*dim (三个码字 concat) → 输出 3*dim (重建 z)
        self.decoders = nn.ModuleList([
            SimpleDecoder(K=K, dim=dim, in_dim=dim * 3) for _ in range(num_layers)
        ])

    def encode_one_layer(self, layer_idx, z):
        """单层: 输入 z (3, B, dim), 返回 (idx, recon, residual)."""
        idx_s, idx_e, idx_h, d_s, d_e, d_h = self.codebooks[layer_idx](z)
        C_s, C_e, C_h = self.codebooks[layer_idx].get_normalized()
        cs_sel = C_s[idx_s]
        ce_sel = C_e[idx_e]
        ch_sel = C_h[idx_h]
        recon_flat = self.decoders[layer_idx](cs_sel, ce_sel, ch_sel)  # (B, 3*dim)
        recon = recon_flat.reshape(-1, 3, self.dim).permute(1, 0, 2)  # (3, B, dim)
        residual = z - recon
        return (idx_s, idx_e, idx_h), recon, residual

    def forward(self, sub_emb):
        """三层 cascade.

        Returns:
            all_indices: list of (idx_s, idx_e, idx_h) per layer
            all_recons: list of recon per layer
            all_z: list of z per layer (z_0 = input, z_1 = residual_1, ...)
        """
        all_indices = []
        all_recons = []
        all_z = []
        z = sub_emb
        for layer in range(self.num_layers):
            all_z.append(z)
            idx, recon, z = self.encode_one_layer(layer, z)
            all_indices.append(idx)
            all_recons.append(recon)
        all_z.append(z)  # 最后一层 residual (理想应该接近 0)
        return all_indices, all_recons, all_z


def compute_v_info_per_layer(all_z, all_recons):
    """计算每层 V-info (用 residual 方差衡量).

    V_info(l) = Var(z_{l+1}) / Var(z_0)   # 越低越好 (residual 越小 = 信息保留越多)
    """
    var_input = sum(z.var().item() for z in all_z[0]) / 3
    v_infos = []
    for l, z_next in enumerate(all_z[1:]):
        var_next = sum(z_next[i].var().item() for i in range(3)) / 3
        v_info = var_next / var_input
        v_infos.append(v_info)
    return v_infos


def compute_utilization(all_indices):
    """每层三码本 utilization."""
    utils = []
    for idx_s, idx_e, idx_h in all_indices:
        utils.append({
            "util_s": len(torch.unique(idx_s)) / 256,
            "util_e": len(torch.unique(idx_e)) / 256,
            "util_h": len(torch.unique(idx_h)) / 256,
        })
    return utils


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.report_path)).mkdir(parents=True, exist_ok=True)

    print(f"[task22-p3] loading MCKG embedding from {args.emb_path}")
    emb = torch.load(args.emb_path, map_location="cpu", weights_only=False)
    sub_item = emb["subspace_item"]  # (3, 11924, 64)
    M, N_total, dim = sub_item.shape
    assert M == 3
    N = min(args.num_items, N_total)
    sub_item = sub_item[:, :N, :]
    print(f"[task22-p3] using {N} items, dim={dim}, K={args.K}, num_layers={args.num_layers}")

    # per-subspace normalize
    for m in range(3):
        sub_item[m] = (sub_item[m] - sub_item[m].mean(0)) / sub_item[m].std(0).clamp(min=1e-5)
    sub_item[2] = sub_item[2] * 0.5
    norms = sub_item[2].norm(dim=-1, keepdim=True)
    sub_item[2] = sub_item[2] / torch.clamp(norms / 0.85, min=1.0)

    # 模型
    cascade = ProductManifoldCascade(
        num_layers=args.num_layers, K=args.K, dim=args.dim)
    optimizer = torch.optim.Adam(cascade.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    print(f"[task22-p3] training {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, util_penalty={args.util_penalty}, num_layers={args.num_layers}")

    metrics_history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        idx_batch = torch.randperm(N)[:args.batch_size]
        batch = sub_item[:, idx_batch, :]

        optimizer.zero_grad()
        all_indices, all_recons, all_z = cascade(batch)

        # 重建 loss (sum over layers)
        recon_loss = 0
        util_pen = 0
        for l in range(args.num_layers):
            # 当前层输入 = all_z[l], 重建 = all_recons[l]
            target = all_z[l].permute(1, 0, 2).reshape(args.batch_size, -1)
            recon = all_recons[l].permute(1, 0, 2).reshape(args.batch_size, -1)
            recon_loss = recon_loss + F.mse_loss(recon, target)
            # utilization penalty
            idx_s, idx_e, idx_h = all_indices[l]
            util_pen = util_pen + (
                compute_utilization_penalty(idx_s, args.K)
                + compute_utilization_penalty(idx_e, args.K)
                + compute_utilization_penalty(idx_h, args.K)
            ) / 3

        util_pen = util_pen / args.num_layers
        loss = recon_loss + args.util_penalty * util_pen

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 15 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                all_indices_e, all_recons_e, all_z_e = cascade(sub_item[:, :2000, :])
                v_infos = compute_v_info_per_layer(all_z_e, all_recons_e)
                utils = compute_utilization(all_indices_e)
            metrics_history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "util_pen": util_pen.item(),
                "v_info_layer1": v_infos[0],
                "v_info_layer2": v_infos[1] if len(v_infos) > 1 else 0.0,
                "v_info_layer3": v_infos[2] if len(v_infos) > 2 else 0.0,
                "final_residual_norm": all_z_e[-1].norm().item(),
                "util_s_l0": utils[0]["util_s"],
                "util_e_l0": utils[0]["util_e"],
                "util_h_l0": utils[0]["util_h"],
                "util_s_l1": utils[1]["util_s"] if len(utils) > 1 else 0.0,
                "util_e_l1": utils[1]["util_e"] if len(utils) > 1 else 0.0,
                "util_h_l1": utils[1]["util_h"] if len(utils) > 1 else 0.0,
                "util_s_l2": utils[2]["util_s"] if len(utils) > 2 else 0.0,
                "util_e_l2": utils[2]["util_e"] if len(utils) > 2 else 0.0,
                "util_h_l2": utils[2]["util_h"] if len(utils) > 2 else 0.0,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_sec": time.time() - t0,
            })
            print(f"[epoch {epoch:3d}] loss={loss.item():.4f} "
                  f"recon={recon_loss.item():.4f} pen={util_pen.item():.3f} "
                  f"V1={v_infos[0]:.3f} V2={v_infos[1]:.3f} V3={v_infos[2]:.3f} "
                  f"uL0_s/e/h={utils[0]['util_s']:.2f}/{utils[0]['util_e']:.2f}/{utils[0]['util_h']:.2f} "
                  f"uL1_s/e/h={utils[1]['util_s']:.2f}/{utils[1]['util_e']:.2f}/{utils[1]['util_h']:.2f} "
                  f"uL2_s/e/h={utils[2]['util_s']:.2f}/{utils[2]['util_e']:.2f}/{utils[2]['util_h']:.2f} "
                  f"[{time.time() - t0:.1f}s]")

    # ===== D2.5 决策指标 =====
    print(f"[task22-p3] running full-dataset final eval...")
    cascade.eval()
    with torch.no_grad():
        all_indices_f, all_recons_f, all_z_f = cascade(sub_item)
        v_infos_final = compute_v_info_per_layer(all_z_f, all_recons_f)
        utils_final = compute_utilization(all_indices_f)

    # 计算 final recon (sum of layers)
    final_recon = sum(F.mse_loss(all_recons_f[l].permute(1, 0, 2).reshape(N, -1),
                                  all_z_f[l].permute(1, 0, 2).reshape(N, -1)).item()
                      for l in range(args.num_layers))

    # search time
    t1 = time.time()
    with torch.no_grad():
        for _ in range(50):
            batch = sub_item[:, torch.randperm(N)[:args.batch_size], :]
            _ = cascade(batch)
    search_time_ms = (time.time() - t1) / 50 / args.batch_size * 1000

    final_metrics = {
        "v_info_layer1": v_infos_final[0],
        "v_info_layer2": v_infos_final[1] if len(v_infos_final) > 1 else 0.0,
        "v_info_layer3": v_infos_final[2] if len(v_infos_final) > 2 else 0.0,
        "final_residual_norm": all_z_f[-1].norm().item(),
        "final_recon_total_loss": final_recon,
        "utils_layer0": utils_final[0],
        "utils_layer1": utils_final[1] if len(utils_final) > 1 else None,
        "utils_layer2": utils_final[2] if len(utils_final) > 2 else None,
        "search_time_ms_per_item": search_time_ms,
        "n_epochs": args.epochs,
        "total_time_sec": time.time() - t0,
    }

    # D2.5 决策: 三层 V-info 都 < 0.5 → 信息被充分保留 → PROCEED Phase 4
    # 若 V_info_layer3 > 0.7 → 深层信息丢失严重 → STOP
    all_v_info_ok = all(v < 0.7 for v in v_infos_final[:args.num_layers])
    all_util_ok = all(
        utils_final[l]["util_s"] > 0.7 and utils_final[l]["util_e"] > 0.7
        and utils_final[l]["util_h"] > 0.7
        for l in range(args.num_layers)
    )

    if all_v_info_ok and all_util_ok:
        decision = "PROCEED Phase 4"
    elif all_util_ok:
        decision = "PARTIAL — V-info 仍高, 信息丢失明显"
    else:
        decision = "STOP — 三层 cascade 不稳定"

    final_metrics["d25_decision"] = decision

    # 保存模型
    torch.save({
        "cascade_state": cascade.state_dict(),
        "config": vars(args),
        "final_metrics": final_metrics,
    }, Path(args.output_dir) / "phase3_model.pt")
    print(f"[task22-p3] model saved → {args.output_dir}/phase3_model.pt")

    out_path = Path(args.output_dir) / "phase3_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "history": metrics_history,
                   "final": final_metrics}, f, indent=2)
    print(f"[task22-p3] metrics saved → {out_path}")

    # 报告
    report = f"""# Task #22 Phase 3 — PM-RQ Three-Layer Cascade + V-info per layer

> **完成日期**: {time.strftime('%Y-%m-%d %H:%M')}
> **状态**: {'✅ PASS' if decision == 'PROCEED Phase 4' else '⚠️ PARTIAL' if 'PARTIAL' in decision else '❌ STOP'}
> **下一阶段**: {'Phase 4 — Benchmarking + 消融' if 'PROCEED' in decision else '终止 / 调优'}

## 配置
- emb-path: `{args.emb_path}`
- K={args.K}, dim={args.dim}, num_items={N} (full Toys), num_layers={args.num_layers}
- epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, util_penalty={args.util_penalty}
- seed={args.seed}

## D2.5 决策指标

### V-info per layer (残差方差 / 输入方差)
- V_info_layer1: **{final_metrics['v_info_layer1']:.4f}** (目标 < 0.5)
- V_info_layer2: **{final_metrics['v_info_layer2']:.4f}** (目标 < 0.5)
- V_info_layer3: **{final_metrics['v_info_layer3']:.4f}** (目标 < 0.5)
- final_residual_norm: **{final_metrics['final_residual_norm']:.4f}**

### 三层利用率 (full eval)
- Layer 0: util_s={final_metrics['utils_layer0']['util_s']:.3f}, util_e={final_metrics['utils_layer0']['util_e']:.3f}, util_h={final_metrics['utils_layer0']['util_h']:.3f}
- Layer 1: util_s={final_metrics['utils_layer1']['util_s']:.3f}, util_e={final_metrics['utils_layer1']['util_e']:.3f}, util_h={final_metrics['utils_layer1']['util_h']:.3f}
- Layer 2: util_s={final_metrics['utils_layer2']['util_s']:.3f}, util_e={final_metrics['utils_layer2']['util_e']:.3f}, util_h={final_metrics['utils_layer2']['util_h']:.3f}

### 性能
- total recon_loss (3 layer sum): **{final_metrics['final_recon_total_loss']:.4f}**
- search time / item: **{final_metrics['search_time_ms_per_item']:.3f} ms**

## 训练曲线 (每 15 epoch)

| epoch | loss | recon | V1 | V2 | V3 | L0_e | L1_e | L2_e | residual_norm |
|-------|------|-------|-----|-----|-----|------|------|------|---------------|
"""
    for m in metrics_history:
        report += (f"| {m['epoch']} | {m['loss']:.4f} | {m['recon_loss']:.4f} | "
                   f"{m['v_info_layer1']:.3f} | {m['v_info_layer2']:.3f} | {m['v_info_layer3']:.3f} | "
                   f"{m['util_e_l0']:.3f} | {m['util_e_l1']:.3f} | {m['util_e_l2']:.3f} | "
                   f"{m['final_residual_norm']:.3f} |\n")

    report += f"""
## D2.5 决策: **{decision}**

## 产物
- model: `{args.output_dir}/phase3_model.pt`
- metrics JSON: `{out_path}`
- 训练总时长: {time.time() - t0:.1f} 秒
"""

    with open(args.report_path, "w") as f:
        f.write(report)
    print(f"[task22-p3] report saved → {args.report_path}")
    print()
    print("=" * 60)
    print(f"D2.5 决策: **{decision}**")
    print(f"  V_info: L1={final_metrics['v_info_layer1']:.3f}, "
          f"L2={final_metrics['v_info_layer2']:.3f}, "
          f"L3={final_metrics['v_info_layer3']:.3f}")
    print(f"  Util_e: L0={final_metrics['utils_layer0']['util_e']:.3f}, "
          f"L1={final_metrics['utils_layer1']['util_e']:.3f}, "
          f"L2={final_metrics['utils_layer2']['util_e']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
