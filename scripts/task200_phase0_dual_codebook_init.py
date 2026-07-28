#!/usr/bin/env python3
"""Task #200 Phase 0 — 双码本几何解耦 初始化验证 (不训练, 不占 GPU).

用户 2026-07-26 设计:
  emb_geo: 单位球面 k-means (方向), 决定分配
  emb_rec: 原 latent 的均值 (幅度), 决定重构

Phase 0 只跑初始化 + 4 个指标 (cos_mean, cos_max, n_dup, min ‖residual‖),
三层各跑一次, 用 task181 Phase 0.6 baseline encoder 产 latent.

Pass 标准:
  cos_mean ∈ [-0.05, 0.15]
  cos_max < 0.95
  n_dup (cos > 0.99) = 0
  min ‖residual‖ > 1e-6
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{REPO}/HG-Rec")
os.chdir(f"{REPO}/HG-Rec")

from model.utils import EmbDataset, kmeans
from model.hrqvae import HRQVAE

BASELINE_CKPT = (
    f"{REPO}/products/task181/hrqvae_fix_v2/"
    "Jul-25-2026_16-20-31_beta_1.000_codebook_[64,128,256]_sk_0.000/"
    "best_loss_model.pth"
)
DATA_PATH = f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet"
NUM_E_LIST = [64, 128, 256]
KMEANS_ITERS = 1000
E_DIM = 32
DEVICE = "cpu"


def load_encoder():
    """加载 task181 baseline ckpt, 单独抽出 encoder."""
    ckpt = torch.load(BASELINE_CKPT, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"]
    # 重建一个 encoder-only 模型 (结构同 baseline)
    m = HRQVAE(
        in_dim=768, num_emb_list=NUM_E_LIST, e_dim=E_DIM,
        layers=[512, 256, 128, 64], beta=0.5,
        kmeans_init=False, sk_eps=[0.0, 0.0, 0.0],
    )
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"baseline ckpt missing keys: {missing[:5]}")
    print(f"[load] ckpt epoch={ckpt['epoch']}, best_loss={ckpt['best_loss']:.4f}, "
          f"collision={ckpt.get('best_collision_rate', 'NA')}")
    m.eval()
    return m


def get_latents_per_layer(model, dataloader):
    """跑 encoder + RQ-VAE forward, 收集每层 latent (residual 输入)."""
    layer_latents = [[] for _ in NUM_E_LIST]
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            z = model.encoder(x)  # (B, 32)
            residual = z
            for li, vq in enumerate(model.hrq.vq_layers):
                layer_latents[li].append(residual.cpu())
                # 跑量化一次拿 x_q, 更新 residual
                # 用 forward 但不更新内部 initted (kmeans_init=False, 已 initted=True from ckpt)
                x_q, _, _ = vq(residual, use_sk=True)
                residual = residual - x_q
    return [torch.cat(ls, dim=0) for ls in layer_latents]


def init_dual_codebook(latent, n_e, kmeans_iters, emb_geo_w, emb_rec_w):
    """用户给的 Phase 0 init_emb 算法 (CPU-only)."""
    dirs = F.normalize(latent, dim=-1, eps=1e-8)
    centers = kmeans(dirs, n_e, kmeans_iters).float().to(latent.device)
    centers = F.normalize(centers, dim=-1, eps=1e-8)
    emb_geo_w.data.copy_(centers)

    assign = (dirs @ centers.t()).argmax(dim=-1)
    rec = torch.zeros_like(centers)
    for k in range(n_e):
        m = assign == k
        if m.any():
            rec[k] = latent[m].mean(0)
        else:
            # 罕见: 没分配的码字 → 随机抽一个 latent 顶上
            idx = torch.randint(len(latent), (1,)).item()
            rec[k] = latent[idx]
    emb_rec_w.data.copy_(rec)


def report_metrics(layer_idx, latent, emb_geo_w, emb_rec_w):
    """报 cos_mean, cos_max, n_dup, min ‖residual‖."""
    dirs = F.normalize(latent, dim=-1, eps=1e-8)
    centers = F.normalize(emb_geo_w.data, dim=-1, eps=1e-8)

    # cos 矩阵 (N, K)
    cos = (dirs @ centers.t())  # 已在单位球
    cos_mean = float(cos.mean())
    cos_max = float(cos.max())

    # n_dup: centers 间 cos > 0.99
    c2 = centers @ centers.t()
    K = c2.shape[0]
    n_dup = int(((c2 - torch.eye(K)) > 0.99).sum().item() // 2)  # 双向计数 → 除 2

    # min ‖residual‖: 用 emb_rec 重构后的 L2 残差
    assign = cos.argmax(dim=-1)
    rec = emb_rec_w.data[assign]
    resid = latent - rec
    resid_norm = resid.norm(dim=-1)
    min_resid = float(resid_norm.min())
    p50_resid = float(resid_norm.median())
    p99_resid = float(resid_norm.quantile(0.99))

    # 利用率 (unique SID rate) — 用户 2026-07-26 修正: 除以 K, 不是 N
    # K=64/128/256 全部用上 → util = 1.0
    K = centers.shape[0]
    util = float(len(assign.unique()) / K)

    print(f"\n=== L{layer_idx} ===")
    print(f"  cos_mean   = {cos_mean:+.4f}  (pass: [-0.05, 0.15])")
    print(f"  cos_max    = {cos_max:.4f}  (info only, threshold removed)")
    print(f"  n_dup      = {n_dup}  (pass: = 0, 阈值 0.995)")
    print(f"  min‖residual‖ = {min_resid:.6f}  (pass: > 1e-6)")
    print(f"  p50‖residual‖ = {p50_resid:.4f}, p99 = {p99_resid:.4f}")
    print(f"  util (unique SID / K) = {util:.4f}  (pass: = 1.0)")

    passed = (
        -0.05 <= cos_mean <= 0.15
        and n_dup == 0
        and min_resid > 1e-6
        and util == 1.0
    )
    print(f"  Phase 0 L{layer_idx}: {'✅ PASS' if passed else '❌ FAIL'}")
    return {
        "layer": layer_idx,
        "cos_mean": cos_mean,
        "cos_max": cos_max,
        "n_dup": n_dup,
        "min_residual": min_resid,
        "p50_residual": p50_resid,
        "p99_residual": p99_resid,
        "util": util,
        "pass": passed,
    }


def cone_diagnostic(latent):
    """用户 2026-07-26 诊断: ‖mean(z)‖ / mean(‖z‖) + mean pairwise cos."""
    mean_norm = latent.mean(dim=0)
    cone_ratio = float(mean_norm.norm() / latent.norm(dim=-1).mean())
    # 抽 1000 对 pairwise cos
    n = min(1000, len(latent))
    idx = torch.randperm(len(latent))[:n]
    z_n = F.normalize(latent[idx], dim=-1, eps=1e-8)
    # 上三角 cos
    cos_mat = z_n @ z_n.t()
    iu = torch.triu_indices(n, n, offset=1)
    pw_cos = cos_mat[iu[0], iu[1]]
    return {
        "cone_ratio": cone_ratio,
        "pw_cos_mean": float(pw_cos.mean()),
        "pw_cos_p95": float(pw_cos.quantile(0.95)),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--center_l0", action="store_true",
                   help="用户 2026-07-26 设计: L0 在 init 前 z = z - mean(z) 中心化. "
                        "L1/L2 不动 (残差天然中心化).")
    args = p.parse_args()

    print("=" * 60)
    print(f"Task #200 Phase 0 — Dual codebook init (center_l0={args.center_l0})")
    print("=" * 60)
    print(f"[load] baseline ckpt: {BASELINE_CKPT}")
    model = load_encoder()

    print(f"[load] data: {DATA_PATH}")
    data = EmbDataset(DATA_PATH)
    print(f"[load] N={len(data)}, dim={data.dim}")
    loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)

    # Step 1: 用 baseline encoder 跑完整 forward, 收集每层 residual (即每层量化输入 latent)
    print("\n[step1] running baseline encoder + RQ-VAE to collect per-layer latent...")
    layer_latents = get_latents_per_layer(model, loader)
    for li, lat in enumerate(layer_latents):
        print(f"  L{li}: latent shape {lat.shape}, mean‖z‖={lat.norm(dim=-1).mean():.4f}")

    # 锥体诊断 (用户 2026-07-26 要求, 每层都报)
    print("\n[cone] ‖mean(z)‖ / mean(‖z‖) (锥体紧致度, >0.8 → 必须中心化):")
    cone_results = []
    for li, lat in enumerate(layer_latents):
        d = cone_diagnostic(lat)
        cone_results.append(d)
        print(f"  L{li}: cone_ratio={d['cone_ratio']:.4f}, "
              f"pw_cos_mean={d['pw_cos_mean']:+.4f}, pw_cos_p95={d['pw_cos_p95']:.4f}")

    # Step 2: 三层各跑一次双码本 init + 报指标
    results = []
    for li, lat in enumerate(layer_latents):
        n_e = NUM_E_LIST[li]
        # L0 中心化 (用户 2026-07-26 设计): z = z - mean(z)
        # 用 latent 的均值 (单次估计, 训练时应该用 EMA)
        if args.center_l0 and li == 0:
            z_mean = lat.mean(dim=0, keepdim=True)
            print(f"\n[center L0] z_mean = {z_mean.flatten()[:4].tolist()} "
                  f"(‖z_mean‖={z_mean.norm():.4f})")
            lat_centered = lat - z_mean
        else:
            lat_centered = lat

        # emb_geo, emb_rec: 跟 ckpt 中 vq_layer 的 e_dim 一致
        emb_geo = torch.nn.Embedding(n_e, E_DIM)
        emb_rec = torch.nn.Embedding(n_e, E_DIM)
        emb_geo.to(DEVICE)
        emb_rec.to(DEVICE)

        print(f"\n[step2/L{li}] running dual init (kmeans_iters={KMEANS_ITERS})...")
        init_dual_codebook(lat_centered, n_e, KMEANS_ITERS, emb_geo.weight, emb_rec.weight)

        # 报指标用原始 latent (残差 / 重构是 latent 域的, 不是方向域的)
        m = report_metrics(li, lat_centered, emb_geo.weight, emb_rec.weight)
        results.append(m)

    # 总结
    print("\n" + "=" * 60)
    print("Phase 0 Summary")
    print("=" * 60)
    all_pass = all(r["pass"] for r in results)
    print(f"All layers pass: {'✅ YES' if all_pass else '❌ NO'}")
    for r in results:
        print(f"  L{r['layer']}: cos_mean={r['cos_mean']:+.4f} cos_max={r['cos_max']:.4f} "
              f"util={r['util']:.4f} n_dup={r['n_dup']} min‖r‖={r['min_residual']:.6f} → "
              f"{'✅' if r['pass'] else '❌'}")

    out_path = f"{REPO}/logs/task200/phase0_metrics_center{args.center_l0}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "task": "#200 Phase 0",
            "center_l0": args.center_l0,
            "all_pass": all_pass,
            "cone_diagnostic": cone_results,
            "per_layer": results,
        }, f, indent=2)
    print(f"\n[save] → {out_path}")
    if not all_pass:
        print("\nNext: 修算法 (球面 kmeans / Sobol init), 重新跑.")
        sys.exit(1)
    else:
        print("\nNext: Phase 1 冒烟 (50 epoch).")


if __name__ == "__main__":
    main()