#!/usr/bin/env python3
"""
Task #200 双码本隔离测试 v3 — 用户 2026-07-26 修正方案.

用户修正 (vs v2):
  - α_geo: 0.1 → 1.0   (geo_loss 权重,不是 rec_align)
  - 每层都 z_mean 中心化 (不只 L0)
  - emb_rec 保持自由,不归一化

隔离测试 (用户明令不碰流水线):
  A. 加载 task181 baseline encoder, 跑一次 forward, 存 L0/L1/L2 真实 residual latents
  B. 单 HVectorQuantization 三个检查点 (不跑 HRQVAE):
     - A. init_emb 后: util / cos_mean / cos_max
     - B. 一次前向: 分配直方图 bincount
     - C. 100 步梯度: loss / unique SID / emb_geo.grad.norm 演化
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization


# ============================================================
# 步骤 1: 加载 baseline encoder, 收集每层真实 residual latent
# ============================================================
def load_baseline_encoder(ckpt_path: str, device: str = "cpu"):
    """加载 task181 baseline HRQVAE encoder (不跑 quantizer)."""
    from model.hrqvae import HRQVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    model = HRQVAE(
        in_dim=saved_args.in_dim if hasattr(saved_args, 'in_dim') else 768,
        num_emb_list=saved_args.num_emb_list,
        e_dim=saved_args.e_dim,
        layers=saved_args.layers,
        dropout_prob=getattr(saved_args, 'dropout_prob', 0.0),
        bn=getattr(saved_args, 'bn', False),
        loss_type=getattr(saved_args, 'loss_type', 'poincare'),
        quant_loss_weight=getattr(saved_args, 'quant_loss_weight', 1.0),
        beta=getattr(saved_args, 'beta', 1.0),
        kmeans_init=False,  # 已训好,不用再 init
        kmeans_iters=getattr(saved_args, 'kmeans_iters', 100),
        sk_eps=getattr(saved_args, 'sk_epsilons', [0.0, 0.0, 0.0]),
        sk_iters=getattr(saved_args, 'sk_iters', 50),
    )
    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[encoder] ckpt loaded: epoch={ckpt['epoch']}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def collect_residual_latents(model, dataloader, device="cpu"):
    """跑 encoder + 收集每层 residual 输入 (即每层 HVectorQuantization.forward 的 latent)."""
    layer_latents = [[] for _ in model.num_emb_list]
    model.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            # 跑 encoder
            z = model.encoder(x)  # (B, e_dim) 切空间
            residual = z
            for li, vq_layer in enumerate(model.hrq.vq_layers):
                layer_latents[li].append(residual.cpu())
                # 跑 quantizer, 取得 x_q, 更新 residual
                x_q, _, _ = vq_layer(residual)
                residual = residual - x_q
                # 残差进入下一层
    return [torch.cat(ls, dim=0) for ls in layer_latents]


# ============================================================
# 步骤 2: 单 VQ 层三个检查点测试
# ============================================================
@torch.no_grad()
def checkpoint_A_init(vq: HVectorQuantization, latent: torch.Tensor):
    """检查点 A: init_emb 后码本质量."""
    # init_emb (内部包含双码本 init 路径)
    vq.train()  # init_emb 需要 training=True
    vq.init_emb(latent)
    vq.eval()

    emb_geo = F.normalize(vq.emb_geo.weight.data, dim=-1, eps=1e-8)
    cos_pw = emb_geo @ emb_geo.t()
    cos_pw.fill_diagonal_(0.0)
    cos_max = float(cos_pw.max())
    cos_mean = float(cos_pw[torch.triu(torch.ones_like(cos_pw), diagonal=1).bool()].mean())
    n_dup = int((cos_pw > 0.99).sum().item() // 2)

    # 分配 (用 init_emb 后的码本, 不传 batch 触发再次 init)
    from model.utils import expmap0, poincare_distance
    dirs = F.normalize(latent - (vq.z_mean.unsqueeze(0) if vq.use_centering else 0),
                       dim=-1, eps=1e-8)
    dirs_h = expmap0(dirs, vq.c)
    centers_h = expmap0(emb_geo, vq.c)
    B, K = dirs_h.shape[0], centers_h.shape[0]
    d = poincare_distance(
        dirs_h.unsqueeze(1).expand(B, K, -1),
        centers_h.unsqueeze(0).expand(B, K, -1),
        vq.c,
    ).squeeze(-1)
    indices = torch.argmin(d, dim=-1)
    unique_sids = len(set(indices.tolist()))
    util = unique_sids / K

    return {
        "util": float(util),
        "cos_max": cos_max,
        "cos_mean": cos_mean,
        "n_dup": n_dup,
        "latent_norm_mean": float(latent.norm(dim=-1).mean()),
        "latent_norm_max": float(latent.norm(dim=-1).max()),
        "unique_sids_init": int(unique_sids),
    }


def checkpoint_B_one_forward(vq: HVectorQuantization, latent: torch.Tensor):
    """检查点 B: 一次前向分配直方图 (init 后, 看码本是否挤在一处)."""
    vq.train()
    x_q, loss, indices = vq(latent)
    vq.eval()
    bincount = torch.bincount(indices, minlength=vq.n_e)
    return {
        "loss_after_init": float(loss.item()),
        "indices_histogram": bincount.tolist(),
        "top5_indices": bincount.topk(5).values.tolist(),
        "top5_indices_idx": bincount.topk(5).indices.tolist(),
        "unique_sids_one_fwd": int(len(set(indices.tolist()))),
        "min_count": int(bincount.min().item()),
        "max_count": int(bincount.max().item()),
    }


def checkpoint_C_100_steps(vq: HVectorQuantization, latent: torch.Tensor,
                           n_steps: int = 100, lr: float = 1e-3):
    """检查点 C: 100 步梯度演化 (看 collision 是否下降, grad 是否真的传到 emb_geo)."""
    vq.train()
    opt = torch.optim.AdamW(vq.parameters(), lr=lr)
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        x_q, loss, indices = vq(latent)
        loss.backward()
        # 记录 emb_geo 梯度 norm (关键: 看 gradient 是否传到 emb_geo)
        geo_grad_norm = float(vq.emb_geo.weight.grad.norm().item()) \
            if vq.emb_geo.weight.grad is not None else 0.0
        rec_grad_norm = float(vq.emb_rec.weight.grad.norm().item()) \
            if vq.emb_rec.weight.grad is not None else 0.0
        opt.step()

        if step % 10 == 0 or step < 5:
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "unique_sids": int(len(set(indices.tolist()))),
                "geo_grad_norm": geo_grad_norm,
                "rec_grad_norm": rec_grad_norm,
            })
    vq.eval()
    return history


# ============================================================
# 主流程
# ============================================================
def run_isolation_test(args):
    device = "cpu"  # 隔离测试, CPU 即可
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 步骤 1: 加载 baseline, 存真实 latent ---
    print(f"\n[Step 1] 加载 baseline encoder: {args.baseline_ckpt}")
    model = load_baseline_encoder(args.baseline_ckpt, device=device)

    print(f"[Step 1] 加载数据: {args.data_path}")
    from model.utils import EmbDataset
    data = EmbDataset(args.data_path)
    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=2)

    print(f"[Step 1] 收集每层 residual latent ...")
    t0 = time.time()
    layer_latents = collect_residual_latents(model, loader, device=device)
    print(f"[Step 1] 收集完成 ({time.time() - t0:.1f}s)")
    for li, lat in enumerate(layer_latents):
        lat_path = out_dir / f"lat_L{li}.pt"
        torch.save(lat, lat_path)
        print(f"  L{li}: shape={tuple(lat.shape)}, "
              f"norm mean={lat.norm(dim=-1).mean():.3f} "
              f"max={lat.norm(dim=-1).max():.3f}, "
              f"saved → {lat_path}")

    # --- 步骤 2: 单 VQ 层三检查点 (对每层各跑一次) ---
    all_results = {}
    for li in range(len(model.num_emb_list)):
        print(f"\n{'=' * 60}")
        print(f"[Step 2] Layer L{li} 单 VQ 隔离测试")
        print(f"{'=' * 60}")

        latent = layer_latents[li]
        n_e = model.num_emb_list[li]
        e_dim = model.e_dim

        # 构建新 HVectorQuantization (双码本路径, 每层都启用 centering)
        vq = HVectorQuantization(
            n_e=n_e, e_dim=e_dim, sk_eps=0.0, beta=0.5,
            kmeans_init=False, kmeans_iters=100, sk_iters=50,
            curvature=1.0, euclidean_qloss=False, loss_mult_codebook=1.0,
            dual_codebook=True,
        )
        # 每层都启用 centering (用户修正: 不只 L0)
        vq.use_centering = True
        # 初始化 z_mean buffer
        vq.z_mean = torch.zeros(e_dim)
        vq.z_mean_ema = 0.99

        # 检查点 A: 初始化
        print(f"\n  [A] init_emb 后码本质量:")
        res_A = checkpoint_A_init(vq, latent)
        for k, v in res_A.items():
            print(f"    {k}: {v}")

        # 检查点 B: 一次前向分配
        print(f"\n  [B] 一次前向分配直方图:")
        res_B = checkpoint_B_one_forward(vq, latent)
        print(f"    loss_after_init = {res_B['loss_after_init']:.4f}")
        print(f"    unique_sids_one_fwd = {res_B['unique_sids_one_fwd']} / {n_e}")
        print(f"    top5 indices (idx, count): {list(zip(res_B['top5_indices_idx'], res_B['top5_indices']))}")
        print(f"    min/max count = {res_B['min_count']} / {res_B['max_count']}")
        # 看直方图是否单峰
        nonzero_count = sum(1 for c in res_B['indices_histogram'] if c > 0)
        print(f"    nonzero bins = {nonzero_count} / {n_e}  (低 = 全挤在一起)")

        # 检查点 C: 100 步梯度
        print(f"\n  [C] 100 步梯度演化:")
        history_C = checkpoint_C_100_steps(vq, latent, n_steps=100, lr=1e-3)
        for entry in history_C:
            print(f"    step {entry['step']:3d}  loss={entry['loss']:.4f}  "
                  f"unique_sids={entry['unique_sids']:3d}  "
                  f"geo_grad={entry['geo_grad_norm']:.4f}  "
                  f"rec_grad={entry['rec_grad_norm']:.4f}")

        all_results[f"L{li}"] = {
            "A_init": res_A,
            "B_one_forward": res_B,
            "C_100_steps": history_C,
            "n_e": n_e,
            "e_dim": e_dim,
        }

    # 保存 JSON
    out_json = out_dir / "isolation_test_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[isolation_test] 全结果已存: {out_json}")

    # --- 决策摘要 ---
    print(f"\n{'=' * 60}")
    print("三检查点决策摘要")
    print(f"{'=' * 60}")
    for li in range(len(model.num_emb_list)):
        r = all_results[f"L{li}"]
        ok_A = r["A_init"]["util"] >= 0.9 and r["A_init"]["cos_max"] < 0.95 and r["A_init"]["n_dup"] == 0
        ok_B = r["B_one_forward"]["nonzero_bins"] if "nonzero_bins" in r["B_one_forward"] else r["B_one_forward"]["unique_sids_one_fwd"] >= 32
        # 重读 nonzero bins
        nz = sum(1 for c in r["B_one_forward"]["indices_histogram"] if c > 0)
        ok_B = nz >= 32
        # C: 看末 step unique_sids 比初 step 提升
        c_first = r["C_100_steps"][0]["unique_sids"]
        c_last = r["C_100_steps"][-1]["unique_sids"]
        ok_C = c_last > c_first
        print(f"  L{li}: A={ok_A} (util={r['A_init']['util']:.2f}, "
              f"cos_max={r['A_init']['cos_max']:.2f}, n_dup={r['A_init']['n_dup']}), "
              f"B={ok_B} (nonzero_bins={nz}/{r['n_e']}), "
              f"C={ok_C} (unique: {c_first} → {c_last})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt",
                        default=f"{REPO}/products/task181/hrqvae_fix_v2/"
                                "Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000/"
                                "epoch_64_collision_0.0918_model.pth")
    parser.add_argument("--data_path",
                        default=f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--output_dir",
                        default=f"{REPO}/logs/task200/isolation_test_v3")
    args = parser.parse_args()

    run_isolation_test(args)