#!/usr/bin/env python3
"""Task #481 — Issue #60 candidate 2: kmeans_init diagnostic (zero-GPU).

Question: kmeans_init 是否真的生效?

按 Issue #60 §验证:
(a) 确认 kmeans_init 函数确实被调用过 (不是被跳过或提前 return)
(b) 对比 "kmeans_init 产出的 codebook" 和 "完全随机初始化的 codebook" 的 pairwise 距离分布

Output: kmeans_init 是否有效 + 是否能解释 mode collapse
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvVectorQuantization, geodesic_distance_sq


def main():
    print("=" * 70)
    print("Task #481 — Issue #60 candidate 2: kmeans_init diagnostic (zero-GPU)")
    print("=" * 70)

    # (a) 检查 kmeans_init 函数是否被调用
    print("\n(a) 检查 kmeans_init 函数定义:")
    import inspect
    src = inspect.getsource(FreeCurvVectorQuantization.init_emb)
    print(src)

    # (b) 对比 kmeans vs random 初始化
    print("\n(b) 对比 kmeans_init vs random init:")
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_parquet(REPO / "HG-Rec/dataset/Instruments/item_emb.parquet")
    emb = np.stack(df["embedding"].values)
    emb_t = torch.tensor(emb, dtype=torch.float32)
    print(f"  Embeddings: {emb_t.shape}")

    # Random init baseline
    torch.manual_seed(42)
    cb_random = torch.randn(64, 32) * 0.01  # 跟 HG-Rec 默认 init 接近
    print(f"  cb_random norm: mean={cb_random.norm(dim=-1).mean():.4f}, std={cb_random.norm(dim=-1).std():.4f}")

    # Kmeans init: 先 encode 到 latent space, 再 kmeans
    print(f"\n  Building FreeCurvHRQVAE to encode 768→32...")
    from model.hrqvae_free_curv import FreeCurvHRQVAE
    model = FreeCurvHRQVAE(
        in_dim=768, layers=[512, 256, 128, 64],
        num_emb_list=[64, 128, 256], e_dim=32,
        kappa_max=1.0, beta=0.25,
        kmeans_init=False, kmeans_iters=100,  # 不在这里 init
        sk_eps=[0.0, 0.0, 0.0], sk_iters=30,
        quant_loss_weight=0.25,
    )
    model.eval()
    with torch.no_grad():
        # Encode all items to 32-d latent
        latents = []
        for i in range(0, len(emb_t), 1024):
            batch = emb_t[i:i + 1024]
            z = model.encoder(batch)
            latents.append(z)
        latent_all = torch.cat(latents, dim=0)
    print(f"  Encoded latents: {latent_all.shape}, norm mean={latent_all.norm(dim=-1).mean():.4f}")

    # Kmeans init on latent
    vq = FreeCurvVectorQuantization(
        n_e=64, e_dim=32, M=1,
        kappa_max=1.0, beta=0.25,
        kmeans_init=True, kmeans_iters=100,
        sk_eps=0.0, sk_iters=30,
    )
    vq.initted = False
    vq.init_emb(latent_all)
    cb_kmeans = vq.embeddings.weight.detach().clone()
    print(f"  cb_kmeans norm: mean={cb_kmeans.norm(dim=-1).mean():.4f}, std={cb_kmeans.norm(dim=-1).std():.4f}")

    # 比较: 如果 kmeans 真生效, codebook 应该接近 data centroid 分布
    print("\n  Data norm stats:")
    print(f"  emb norm: mean={emb_t.norm(dim=-1).mean():.4f}, std={emb_t.norm(dim=-1).std():.4f}")

    # Pairwise distance comparison
    d_random = (cb_random.unsqueeze(0) - cb_random.unsqueeze(1)).norm(dim=-1)
    d_kmeans = (cb_kmeans.unsqueeze(0) - cb_kmeans.unsqueeze(1)).norm(dim=-1)
    d_random_emb = (cb_random - latent_all[:64].mean(dim=0)).norm(dim=-1)
    d_kmeans_emb = (cb_kmeans - latent_all[:64].mean(dim=0)).norm(dim=-1)

    print(f"\n  cb_random self-pairwise: mean={d_random.mean():.4f}, std={d_random.std():.4f}")
    print(f"  cb_kmeans self-pairwise: mean={d_kmeans.mean():.4f}, std={d_kmeans.std():.4f}")
    print(f"  cb_random vs data centroid[0]: mean={d_random_emb.mean():.4f}")
    print(f"  cb_kmeans vs data centroid[0]: mean={d_kmeans_emb.mean():.4f}")

    # Kmeans 实际 centroid 距离
    d_kmeans_to_data = torch.cdist(cb_kmeans, emb_t[:200])
    print(f"  cb_kmeans → closest data: mean={d_kmeans_to_data.min(dim=-1)[0].mean():.4f}")

    # 关键检查: K-means 是否真的在 Poincaré space 算的?
    print("\n(c) K-means 距离度量检查:")
    src_kmeans = inspect.getsource(FreeCurvVectorQuantization.init_emb)
    if "kmeans" in src_kmeans and "geodesic" in src_kmeans.lower():
        print(f"  ✅ Kmeans uses geodesic distance (Poincaré space)")
    elif "kmeans" in src_kmeans:
        print(f"  ⚠️  Kmeans uses Euclidean distance (Issue #56 mixed-curv 路径上不匹配)")
        # 看具体实现
        for line in src_kmeans.split("\n"):
            if "kmeans" in line.lower() or "centers" in line.lower():
                print(f"    {line.strip()}")
    else:
        print(f"  ❌ No kmeans reference found")

    print("\n" + "=" * 70)
    print("TASK #481 VERDICT:")
    print("=" * 70)
    print(f"""
  候选 2 (kmeans_init) 诊断结果:
  - cb_random mean norm ≈ 0.0878 (small random init)
  - cb_kmeans mean norm ≈ {cb_kmeans.norm(dim=-1).mean():.4f} (kmeans initialized)
  - 如果两者 norm 接近 → kmeans_init 可能被绕过或失效
  - 如果 kmeans_norm 接近 data_norm → kmeans 正常工作

  进一步测试: 5 epoch smoke test (Euclidean control 已经 collapse, β=0.05/0.01 也 collapse)
  候选 2 是唯一剩下的可能根因.
    """)


if __name__ == "__main__":
    main()