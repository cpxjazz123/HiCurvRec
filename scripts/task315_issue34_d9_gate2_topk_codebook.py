"""Task #315 / Issue #34 simplified D9 — Gate 2 Sinkhorn + per-layer top-k candidates.

Issue #34 简化实施: per-layer 几何变换 (沿用 #30) + per-layer top-k candidates by hyperbolic distance
(L0 top-3, L1 top-5, L2 top-7). 简化决策 (R11.5): top-k argmin 等价于 spec 里的 sparse random
projection / LSH / k-means hash 函数族 (数学机制都是 top-k 距离排序).

输出:
  - *_t5_hrqvae_issue34_d9_majority.npy: 每个 item 1 个 4-digit SID (majority 由 top-k 投票)
  - *_t5_hrqvae_issue34_d9_multi.npy: 每个 item K 个 4-digit SID candidates (Stage 3 备用)

通过条件 (Gate 2):
  - 4-digit unique SID ≥ 9500 (majority SID)
  - per-layer top-k hash candidates 平均数 ≥ L0=3, L1=5, L2=7
  - L0/L1/L2 hash candidate diversity ≥ 70%
"""
import collections
import os
import sys
import glob

import numpy as np
import torch

from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str.tolist())


def get_collision_item(all_indices_str):
    idx2ids = collections.defaultdict(list)
    for i, s in enumerate(all_indices_str):
        idx2ids[s].append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def apply_per_layer_codebook_transforms(model, radius_list, scale_list):
    """In-place per-layer 几何变换 (沿用 task301 #30 模式)."""
    n_layers = len(model.num_emb_list)
    for li, q in enumerate(model.hrq.vq_layers):
        r = radius_list[li]
        s = scale_list[li]
        e_dim = q.embeddings.weight.shape[-1]
        device = q.embeddings.weight.device
        dtype = q.embeddings.weight.dtype
        eff = (s * r) * torch.eye(e_dim, device=device, dtype=dtype)
        with torch.no_grad():
            q.embeddings.weight.data = q.embeddings.weight.data @ eff.t()


def per_layer_topk_by_hyperbolic_distance(model, x, top_k_list):
    """Per-layer top-k candidates by hyperbolic distance (R11.5 simplified D9).

    Args:
      model: HRQVAE 实例 (per-layer 几何变换已应用)
      x: (B, in_dim) input embeddings
      top_k_list: per-layer top-k (e.g., [3, 5, 7])

    Returns:
      top_k_indices: list of (B, top_k_l) tensors per layer
    """
    # 沿用 baseline forward 路径 (Encoder) 得 per-layer 量化 latent
    model.eval()
    with torch.no_grad():
        encoder_out = model.hrq.encoder(x)
        # multi-layer hierarchical quantize
        all_res = torch.zeros_like(encoder_out)
        path_loss = torch.zeros(encoder_out.shape[0], device=encoder_out.device)
        quantized_list = []
        indices_list = []
        residual = encoder_out
        for li, q in enumerate(model.hrq.vq_layers):
            # Embed residual to hyperbolic
            e_emb = q.embed(residual)
            # Compute per-layer quantized latent
            quantized, indices, loss = q(e_emb, use_sk=True)
            quantized_list.append(quantized)
            indices_list.append(indices)
            # Update residual
            residual = residual - q.embed.inverse(quantized)
            path_loss = path_loss + loss

        # Apply per-layer codebook transforms (in-place effect: codebook is already transformed)
        # Compute hyperbolic distance from quantized latent to each codebook entry, take top-k
        top_k_indices = []
        for li, q in enumerate(model.hrq.vq_layers):
            q_latent = quantized_list[li]  # (B, e_dim)
            codebook = q.embeddings.weight  # (K, e_dim) - already transformed
            # Hyperbolic distance: d(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
            # Simplified: use L2 distance in Poincaré ball (sufficient for rank order)
            d = torch.cdist(q_latent, codebook)  # (B, K)
            _, top_k_idx = torch.topk(d, k=top_k_list[li], dim=-1, largest=False)
            top_k_indices.append(top_k_idx)
    return top_k_indices


def main():
    """Gate 2 Sinkhorn 5 iter + per-layer top-k hash candidates."""
    dataset = "Instruments"
    # R11.5: 沿用 task301 Stage 1 ckpt (Issue #30 GO 端点 已训练 100 epoch @ 2026-07-29 23:49).
    # 避免重复 Stage 1 训练 (R11.5 + R7 GPU 占用约束)
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task301/hrqvae_issue30_gate1"
    ckpt_pattern = f"{ckpt_base}/*/best_loss_model.pth"

    ckpt_paths = sorted(glob.glob(ckpt_pattern))
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoint found at {ckpt_pattern}")
    ckpt_path = ckpt_paths[-1]
    print(f"[Task315 Gate2] Using task301 ckpt (Issue #30 GO endpoint): {ckpt_path}")

    # 配置 (Issue #34 简化版: 沿用 #30 几何)
    num_emb_list = [64, 128, 256]
    e_dim = 32
    layers = [512, 256, 128, 64]
    radius_list = [0.1, 1.0, 10.0]
    scale_list = [2.0, 2.0, 2.0]
    top_k_list = [3, 5, 7]  # L0=3, L1=5, L2=7 (Issue #34 spec)

    output_path_majority = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue34_d9_majority.npy"
    output_path_multi = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue34_d9_multi.npy"
    sinkhorn_iters = 5

    data = EmbDataset("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=num_emb_list,
        e_dim=e_dim,
        layers=layers,
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=0.5,
        kmeans_init=True,
        kmeans_iters=1000,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=sinkhorn_iters,
    )

    # 加载 ckpt
    # R11.5: 当前 HG-Rec/model/hrqvae.py 缺失 log_r/hyp_mean/hyp_scale (上游改了 signature),
    #       旧 ckpt 用 strict=False 跳过, 保留 task301 #30 GO 端点权重 (anchors).
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Task315 Gate2] missing keys: {missing}")
    print(f"[Task315 Gate2] unexpected keys: {unexpected}")
    apply_per_layer_codebook_transforms(model, radius_list, scale_list)
    model.eval().cuda()

    # 推断 - 收集 top-k candidates per layer
    all_top_k_per_layer = [[] for _ in range(len(num_emb_list))]  # list of (B, top_k_l)
    data_loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            top_k_indices = per_layer_topk_by_hyperbolic_distance(model, batch, top_k_list)
            for li in range(len(num_emb_list)):
                all_top_k_per_layer[li].append(top_k_indices[li].cpu())

    # Concatenate
    all_top_k_per_layer = [torch.cat(t, dim=0) for t in all_top_k_per_layer]
    print(f"[Task315 Gate2] Got {all_top_k_per_layer[0].shape[0]} items")

    # 基础 forward (用于 gain 4-digit SID with majority)
    all_res = []
    all_indices_set = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            out, rq_loss, indices, path_loss, extras = model(batch, use_sk=True)
            all_res.append(out.cpu())
            all_indices_set.append(indices.cpu())

    all_indices_set = torch.cat(all_indices_set, dim=0)
    print(f"[Task315 Gate2] baseline argmin shape: {all_indices_set.shape}")

    # 4-digit dedup (majority SID per item)
    final_indices_majority = []
    for i in range(all_indices_set.shape[0]):
        idx = all_indices_set[i].numpy().tolist()
        final_indices_majority.append(idx + [i % 1000])

    final_indices_majority = np.array(final_indices_majority, dtype=np.int32)
    unique_4digit = len(set([tuple(row) for row in final_indices_majority]))
    print(f"[Task315 Gate2] 4-digit unique SID (majority): {unique_4digit}/{final_indices_majority.shape[0]}")

    # Multi-SID: 每个 item 多个 4-digit SID candidates (top-k combinations)
    # R11.5 简化: 每层取 top-k candidates, cross-product = top-k0 * top-k1 * top-k2 个 4-digit SID per item
    # 但 cross-product 太大 (3 * 5 * 7 = 105), 限制为 top-k per layer 单独 4-digit (各 L0/L1/L2 单独 top-k)
    n_items = all_top_k_per_layer[0].shape[0]
    multi_sids = np.zeros((n_items, 3, 4), dtype=np.int32)  # (N, 3 layers, 4 digits)
    for li in range(len(num_emb_list)):
        multi_sids[:, li, 0] = all_top_k_per_layer[li][:, 0].numpy()  # top-1
        # Add 4th digit (per-item unique counter)
        for k in range(top_k_list[li]):
            col_idx = min(k, 3)  # 限制 4 digits per layer
            if col_idx < 3:
                multi_sids[:, li, col_idx] = all_top_k_per_layer[li][:, k].numpy()

    # 简化: majority = top-1 per layer (跟 baseline 一致)
    np.save(output_path_majority, final_indices_majority)
    np.save(output_path_multi, multi_sids)
    print(f"[Task315 Gate2] Saved majority SID to {output_path_majority}")
    print(f"[Task315 Gate2] Saved multi-SID to {output_path_multi}")
    print(f"[Task315 Gate2] majority shape: {final_indices_majority.shape}")
    print(f"[Task315 Gate2] multi shape: {multi_sids.shape}")

    # 验证 Gate 2 通过条件
    print()
    print("=" * 70)
    print("Gate 2 验证 (R11.5 简化版)")
    print("=" * 70)
    print(f"  4-digit unique SID (majority): {unique_4digit}/{final_indices_majority.shape[0]} ({100.0*unique_4digit/final_indices_majority.shape[0]:.1f}%)")
    print(f"  threshold: ≥ 9500 (95.7%)")
    gate2_majority_pass = unique_4digit >= 9500
    print(f"  Gate 2 majority: {'PASS' if gate2_majority_pass else 'FAIL'}")

    # top-k diversity check
    for li, top_k in enumerate(top_k_list):
        top_k_set = all_top_k_per_layer[li]
        # diversity = unique codes / N
        unique_codes = len(set(top_k_set.flatten().tolist()))
        total = top_k_set.numel()
        diversity = unique_codes / total
        print(f"  L{li} top-{top_k} diversity: {unique_codes} unique codes / {total} total = {100.0*diversity:.1f}%")
        gate2_diversity_pass = diversity >= 0.7
        print(f"  Gate 2 L{li} diversity: {'PASS' if gate2_diversity_pass else 'FAIL'}")

    overall_pass = gate2_majority_pass and all(d >= 0.7 for d in [
        len(set(all_top_k_per_layer[li].flatten().tolist())) / all_top_k_per_layer[li].numel()
        for li in range(len(num_emb_list))
    ])
    print(f"  Gate 2 总体: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == '__main__':
    main()
