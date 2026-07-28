#!/usr/bin/env python3
"""
Task #181 诊断: k-means 初始化后, 欧式距离 argmin vs 双曲距离 argmin 是否一致?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'HG-Rec'))
os.environ['TRITON_CACHE_DIR'] = '/home/wlia0047/.triton/cache_diag_kmeans'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import numpy as np
import pandas as pd
from model.utils import HVectorQuantization, poincare_distance, expmap0, proj_to_ball, kmeans

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # --- 1. Load item embeddings ---
    df = pd.read_parquet('/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet')
    embeddings = np.stack(df['embedding'].values).astype(np.float32)
    print(f"\n[1] 加载 item_emb.parquet: {df.shape}, emb_dim={embeddings.shape[1]}")
    print(f"    形状: {embeddings.shape}, 范围: [{embeddings.min():.6f}, {embeddings.max():.6f}]")

    # 模拟 encoder (768d→32d 随机投影)
    torch.manual_seed(42)
    data_raw = torch.from_numpy(embeddings).to(device)
    proj = torch.randn(embeddings.shape[1], 32, device=device)
    proj = proj / proj.norm(dim=0, keepdim=True)
    data = data_raw @ proj
    print(f"    encoder output (32d): ‖data‖ mean={data.norm(dim=-1).mean():.4f}, max={data.norm(dim=-1).max():.4f}")

    # --- 2. k-means initialization ---
    n_e = 64
    e_dim = 32
    c = 1.0
    print(f"\n[2] k-means 初始化 (n_e={n_e}, e_dim={e_dim}, kmeans_iters=1000)")
    centers = kmeans(data, n_e, 1000)
    print(f"    init ‖center‖ mean={centers.norm(dim=-1).mean():.4f}, max={centers.norm(dim=-1).max():.4f}")
    outside = (centers.norm(dim=-1) >= 1.0).sum().item()
    print(f"    球外 (‖c‖>=1): {outside}/{n_e}")

    # --- 3. Map to Poincare ball ---
    codebook_h = proj_to_ball(expmap0(centers, c), c)
    data_h = proj_to_ball(expmap0(data, c), c)
    print(f"\n[3] 映射到 Poincaré ball 后:")
    print(f"    codebook_h ‖x‖ mean={codebook_h.norm(dim=-1).mean():.4f}, max={codebook_h.norm(dim=-1).max():.4f}")
    print(f"    data_h ‖x‖ mean={data_h.norm(dim=-1).mean():.4f}, max={data_h.norm(dim=-1).max():.4f}")

    # --- 4. Euclidean argmin vs Poincaré argmin ---
    print(f"\n[4] 距离度量一致性检查 (核心)")
    data_exp = data.unsqueeze(1)
    centers_exp = centers.unsqueeze(0)
    euclidean_d = (data_exp - centers_exp).norm(dim=-1)
    euclidean_argmin = euclidean_d.argmin(dim=-1)

    data_h_exp = data_h.unsqueeze(1)
    cb_h_exp = codebook_h.unsqueeze(0)
    poincare_d = poincare_distance(data_h_exp, cb_h_exp, c).squeeze(-1)
    poincare_argmin = poincare_d.argmin(dim=-1)

    agreement = (euclidean_argmin == poincare_argmin).float().mean().item()
    print(f"    欧式 argmin vs 双曲 argmin 一致率: {agreement*100:.1f}%")

    # --- 5. Codebook utilization ---
    print(f"\n[5] 码字利用 (codebook utilization)")
    po_counts = torch.bincount(poincare_argmin, minlength=n_e)
    po_collision = (po_counts == 0).float().mean().item()
    print(f"    双曲 collision_rate: {po_collision:.4f} ({po_collision*100:.1f}%)")
    top_code = poincare_argmin.mode().values.item()
    top_pct = (poincare_argmin == top_code).sum().item() / data.shape[0] * 100
    print(f"    双曲最常用码字: code {top_code} 被分配 {top_pct:.1f}%")

    if top_pct > 90:
        print("\n ⛔ 双曲 argmin 严重坍缩: >90% 映射到同一码字")
    elif agreement > 0.95:
        print("\n ✅ 欧式与双曲 argmin 高度一致 (>95%), 距离度量不匹配不是根因")
    else:
        print(f"\n ⚠️ 仅 {agreement*100:.1f}% 一致 → 距离度量不匹配可能是根因")

    # --- 6. Original code loss diagnostic ---
    print(f"\n[6] 原始代码 loss 诊断")
    x_q = centers.index_select(0, euclidean_argmin)
    latent = data
    loss_p = torch.mean(poincare_distance(x_q.detach(), latent, c)**2)
    print(f"    Poincaré loss on raw Euclidean vectors: {loss_p.item():.4f}")
    print(f"    Codebook ‖x‖²: mean={(centers**2).sum(dim=-1).mean():.4f}, max={(centers**2).sum(dim=-1).max():.4f}")
    print("    球外码字:", ((centers**2).sum(dim=-1) >= 1).sum().item(), "/", n_e)

    # --- 7. Try encoding with the actual train_hrqvae.py config ---
    print(f"\n[7] 模拟完整的 HVectorQuantization forward (Phase 0.5 fix + sk_eps=[0,0,0])")
    vq = HVectorQuantization(
        n_e=n_e, e_dim=e_dim, beta=0.5,
        kmeans_init=True, kmeans_iters=1000,
        sk_eps=0.0, sk_iters=50, curvature=c
    ).to(device)
    # force k-means init
    vq.initted = False
    vq.embeddings.weight.data.zero_()[:] = 0
    with torch.no_grad():
        vq.init_emb(data[:256])  # first batch like training
    print(f"    init后 ‖weight‖ mean={vq.embeddings.weight.norm(dim=-1).mean():.4f}")
    # first forward pass
    x_q_out, loss, indices = vq(data[:256], use_sk=False)
    idx_counts = torch.bincount(indices, minlength=n_e)
    collisions = (idx_counts == 0).sum().item()
    print(f"    forward collision: {collisions}/{n_e} ({collisions/n_e*100:.1f}%)")
    print(f"    top1 assignment: {idx_counts.max().item()}/256 ({idx_counts.max().item()/256*100:.1f}%)")
    print(f"    unique indices: {indices.unique().numel()}/{n_e}")
    print(f"    loss: {loss.item():.4f}")

    print("\n[诊断完成]")

if __name__ == '__main__':
    main()
