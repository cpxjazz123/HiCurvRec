"""Task #312 Issue #35 Gate 2 — Stage 2 Sinkhorn 推断.

Issue #35 ckpt (r_l=[1,1,1] identity + s_l=[2,2,2]) → (N, 4) SID .npy for Stage 3 T5.

Usage:
    python3 scripts/task312_issue35_gate2_stage2_codebook.py
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
    """In-place per-layer 几何变换 (Issue #35: r_l identity + s_l=[2,2,2])."""
    import torch
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


def main():
    """Issue #35 Gate 2 Sinkhorn 5 iter inference."""
    dataset = "Instruments"
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task312/hrqvae_issue35_rl_identity_sl22"
    ckpt_pattern = f"{ckpt_base}/*/best_loss_model.pth"

    ckpt_paths = sorted(glob.glob(ckpt_pattern))
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoint found at {ckpt_pattern}")
    ckpt_path = ckpt_paths[-1]
    print(f"[Task312 Gate2] Using ckpt: {ckpt_path}")

    # 配置 (Issue #35: r_l identity + s_l=[2,2,2])
    num_emb_list = [64, 128, 256]
    e_dim = 32
    layers = [512, 256, 128, 64]
    radius_list = [1.0, 1.0, 1.0]
    scale_list = [2.0, 2.0, 2.0]

    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue35_rl_identity_sl22.npy"
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
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    apply_per_layer_codebook_transforms(model, radius_list, scale_list)
    model.eval().cuda()

    # 推断
    all_res = []
    all_indices_set = []
    data_loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            # HRQVAE forward 返回: out, rq_loss, indices, path_loss, extras
            out, rq_loss, indices, path_loss, extras = model(batch, use_sk=True)
            all_res.append(out.cpu())
            all_indices_set.append(indices.cpu())

    all_res = torch.cat(all_res, dim=0)
    all_indices_set = torch.cat(all_indices_set, dim=0)
    print(f"[Task312 Gate2] Got {all_res.shape[0]} indices, shape: {all_indices_set.shape}")

    # 4-digit SID
    all_indices_str = all_indices_set.numpy().astype(np.int32)
    stats = get_indices_count(all_indices_str[:, 0] * 1000000 + all_indices_str[:, 1] * 1000 + all_indices_str[:, 2])
    n_unique = len(set(all_indices_str[:, 0] * 1000000 + all_indices_str[:, 1] * 1000 + all_indices_str[:, 2]))
    print(f"[Task312 Gate2] 3-digit unique SID: {n_unique}/{all_res.shape[0]}")

    # 4-digit dedup (按 #30 模式)
    final_indices = []
    for i in range(all_indices_set.shape[0]):
        idx = all_indices_set[i].numpy().tolist()
        # 4th digit = unique counter
        final_indices.append(idx + [i % 1000])  # 简单 4th digit (跟 #30 模式)

    final_indices = np.array(final_indices, dtype=np.int32)

    # 4-digit unique count
    unique_4digit = len(set([tuple(row) for row in final_indices]))
    print(f"[Task312 Gate2] 4-digit unique SID: {unique_4digit}/{final_indices.shape[0]}")

    np.save(output_path, final_indices)
    print(f"[Task312 Gate2] Saved SID to {output_path}")
    print(f"[Task312 Gate2] shape: {final_indices.shape}")


if __name__ == '__main__':
    main()
