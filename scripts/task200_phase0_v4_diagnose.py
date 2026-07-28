#!/usr/bin/env python3
"""
Task #200 Phase 0 v4 — 双码本 standalone 诊断 v4
模拟 HRQVAE.forward 的 latent 形态(logmap0 后切空间),走完整 dual_codebook forward,
看真实 latent 范数 + cos_max + collision + train_loss 期望。

Phase 0 v3 PASS 是因为它 standalone 加载 Poincaré ball 上的真实点(范数 ~0.85)。
v4 加载的是 sentence-transformer 输出的 embedding 范数 ~5-10(切空间),
模拟真实训练场景。
"""
import argparse
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.utils import HVectorQuantization, expmap0, logmap0, poincare_distance


def load_real_latent(parquet_path: str):
    """加载 sentence-transformer 真实 latent (跟训练时 HRQVAE.encoder 输入同尺度)"""
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(parquet_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_") or c.startswith("embedding_")]
    if emb_cols:
        arr = np.stack([np.array(df[c].tolist()) for c in emb_cols], axis=-1)
    elif "embedding" in df.columns:
        arr = np.stack(df["embedding"].values, axis=0)
    else:
        # fall back: assume columns after 'item_id'/'sid' are embeddings
        first_meta = next((i for i, c in enumerate(df.columns)
                           if c.lower() in ("item_id", "sid", "id")), 1)
        arr = df.iloc[:, first_meta:].values
    print(f"[v4] latent shape={arr.shape}, dtype={arr.dtype}")
    print(f"[v4] latent norm: mean={np.linalg.norm(arr, axis=-1).mean():.3f} "
          f"min={np.linalg.norm(arr, axis=-1).min():.3f} "
          f"max={np.linalg.norm(arr, axis=-1).max():.3f}")
    return torch.tensor(arr, dtype=torch.float32)


@torch.no_grad()
def diagnose_dual_codebook_forward(latent, e_dim=32, n_e=64, curvature=1.0):
    """模拟 dual_codebook forward (修复后):
       - emb_geo / emb_rec / latent 全部切空间
       - expmap0 只在算 d 和 geo_loss 一行
       - commit/code 用 MSE (不用 poincare_distance)
    """
    # 截断或投影到 e_dim
    if latent.shape[-1] > e_dim:
        latent = latent[:, :e_dim]
    elif latent.shape[-1] < e_dim:
        latent = F.pad(latent, (0, e_dim - latent.shape[-1]))
    latent = latent.contiguous()

    # L0 centering (z_mean)
    z_mean = latent.mean(dim=0)
    z_centered = latent - z_mean.unsqueeze(0)
    z_for_assign = z_centered

    # emb_geo: F.normalize 球面 kmeans
    dirs = F.normalize(z_for_assign, dim=-1, eps=1e-8)
    from sklearn.cluster import KMeans
    dirs_np = dirs.numpy()
    kmeans = KMeans(n_clusters=n_e, n_init=10, max_iter=300, random_state=42)
    assign = kmeans.fit_predict(dirs_np)
    centers_geo = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    centers_geo = F.normalize(centers_geo, dim=-1, eps=1e-8)

    # 算双曲距离 d
    dirs_h = expmap0(dirs, curvature)
    centers_h = expmap0(centers_geo, curvature)
    B, K = dirs_h.shape[0], centers_h.shape[0]
    d = poincare_distance(
        dirs_h.unsqueeze(1).expand(B, K, -1),
        centers_h.unsqueeze(0).expand(B, K, -1),
        curvature
    ).squeeze(-1)

    # argmin
    indices = torch.argmin(d, dim=-1)

    # emb_rec: latent 均值
    rec = torch.zeros_like(centers_geo)
    for k in range(n_e):
        m = (indices == k)
        if m.any():
            rec[k] = z_for_assign[m].mean(0)
        else:
            idx = torch.randint(len(z_for_assign), (1,)).item()
            rec[k] = z_for_assign[idx]
    x_q = rec[indices]

    # commit/code (MSE, 不再 poincare_distance)
    commit = F.mse_loss(x_q, z_for_assign)
    code = F.mse_loss(x_q, z_for_assign)

    # collision
    codes = [tuple(((indices == k).sum().item() > 0 for k in range(n_e)))]
    sid_per_item = indices.numpy()
    collision = (len(sid_per_item) - len(set(sid_per_item.tolist()))) / len(sid_per_item)

    # cos_max
    cos_pw = (centers_geo @ centers_geo.t()).numpy()
    np.fill_diagonal(cos_pw, 0)
    cos_max = float(cos_pw.max())
    cos_mean = float(cos_pw[np.triu_indices(n_e, k=1)].mean())
    n_dup = int((cos_pw > 0.99).sum() // 2)

    # util = unique SID / K
    unique_sids = len(set(sid_per_item.tolist()))
    util = unique_sids / n_e

    return {
        "latent_norm_mean": float(z_for_assign.norm(dim=-1).mean()),
        "latent_norm_max": float(z_for_assign.norm(dim=-1).max()),
        "centers_norm_mean": float(centers_geo.norm(dim=-1).mean()),
        "d_mean": float(d.mean()),
        "d_max": float(d.max()),
        "d_min": float(d.min()),
        "commit": float(commit),
        "code": float(code),
        "train_loss": float(commit + 0.5 * code),
        "collision": float(collision),
        "util": float(util),
        "cos_max": cos_max,
        "cos_mean": cos_mean,
        "n_dup": n_dup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--center_l0", action="store_true")
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--n_e", type=int, default=64)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    latent = load_real_latent(args.parquet)
    metrics = diagnose_dual_codebook_forward(
        latent, e_dim=args.e_dim, n_e=args.n_e, curvature=args.curvature
    )

    print()
    print("=" * 60)
    print("Phase 0 v4 — dual_codebook forward 诊断")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    # 判据
    ok = []
    ok.append(("collision ≤ 30%", metrics["collision"] <= 0.30))
    ok.append(("cos_max < 0.95", metrics["cos_max"] < 0.95))
    ok.append(("n_dup = 0", metrics["n_dup"] == 0))
    ok.append(("util ≥ 0.90", metrics["util"] >= 0.90))
    ok.append(("latent_norm > 0", metrics["latent_norm_mean"] > 0))

    print("判据:")
    for name, passed in ok:
        print(f"  {'✅' if passed else '❌'} {name}")
    print()
    print("VERDICT:", "✅ PASS" if all(p for _, p in ok) else "❌ FAIL")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"metrics": metrics, "passes": dict(ok)}, f, indent=2)
        print(f"[v4] saved to {args.output}")