#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #297 / Issue #25 Gate 0 — Phase A ckpt 复用验证

设计 (Issue #25 §Gate 0):
- 验证 task287 Arm A Phase A ckpt 存在 + 三层 utilization ≥ 90%
- 加载 9922 items → encode → 4-digit SID via Sinkhorn + dedup
- 通过条件: L0/L1/L2 ≥ 90% + collision ≤ 0.20
- 零 GPU (cpu forward-pass)
- 复用 task287 phase A 配置 (κ frozen=0, 100 epoch)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task297_issue25_gate0")

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')


def _get_args(args, name, default=None):
    """args 可能是 dict (新 ckpt) 或 Namespace (旧 ckpt)."""
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def load_ckpt(ckpt_path, device='cpu'):
    from model.hrqvae import HRQVAE
    from model.utils import EmbDataset
    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args = ckpt['args']
    log.info(f"  num_emb_list={_get_args(args, 'num_emb_list')}, e_dim={_get_args(args, 'e_dim')}, "
             f"beta={_get_args(args, 'beta')}, kappa={_get_args(args, 'kappa', 'N/A')}")
    data = EmbDataset(_get_args(args, 'data_path'))
    log.info(f"  Data: {len(data)} items, dim={data.dim}")
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=_get_args(args, 'num_emb_list'),
        e_dim=_get_args(args, 'e_dim'),
        layers=_get_args(args, 'layers', [512, 256, 128, _get_args(args, 'e_dim')]),
        dropout_prob=_get_args(args, 'dropout_prob', 0.0),
        bn=_get_args(args, 'bn', False),
        loss_type=_get_args(args, 'loss_type'),
        quant_loss_weight=_get_args(args, 'quant_loss_weight', 1.0),
        beta=_get_args(args, 'beta'),
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=_get_args(args, 'sk_epsilons'),
        sk_iters=_get_args(args, 'sk_iters'),
    )
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device).eval()
    return model, data, args


def euc_dist(z, e):
    z_sq = (z ** 2).sum(dim=-1, keepdim=True)
    e_sq = (e ** 2).sum(dim=-1, keepdim=True).t()
    ze = z @ e.t()
    return z_sq + e_sq - 2 * ze


@torch.no_grad()
def encode_all(model, data, batch_size=256, device='cpu'):
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    latents = []
    for batch in loader:
        batch = batch.to(device)
        z = model.encoder(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


@torch.no_grad()
def sinkhorn_(M, eps=0.003, n_iters=5):
    """Simple log-Sinkhorn: returns assignment."""
    n, m = M.shape
    log_a = -torch.log(torch.tensor(n).float())
    log_b = -torch.log(torch.tensor(m).float())
    log_K = -M / eps
    log_u = torch.full((n,), log_a.item())
    log_v = torch.full((m,), log_b.item())
    for _ in range(n_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    log_P = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return log_P.argmax(dim=1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--n_iters", type=int, default=5)
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/.claude/jobs/04ccf474/tmp/task297_issue25_gate0.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #297 / Issue #25 Gate 0 — Phase A ckpt 复用验证")
    log.info("=" * 70)
    log.info(f"CKPT: {args.ckpt_path}")

    model, data, ckpt_args = load_ckpt(args.ckpt_path)
    log.info("\n=== 编码所有 items ===")
    t0 = time.time()
    z_all = encode_all(model, data)
    log.info(f"  Latent shape: {z_all.shape}, encode time: {time.time()-t0:.1f}s")

    n_e_list = _get_args(ckpt_args, 'num_emb_list')
    n_layers = len(n_e_list)

    # Per-layer Sinkhorn + 4-digit dedup
    log.info(f"\n=== Per-layer Sinkhorn (n_iters={args.n_iters}) + 4-digit dedup ===")
    sids = []  # list of (N,) per layer
    per_layer_util = []
    for li in range(n_layers):
        # 残差
        if li == 0:
            cur = z_all
        else:
            cur = z_all.clone()
            for lj in range(li):
                cb = model.hrq.vq_layers[lj].embeddings.weight.detach().cpu()
                d = euc_dist(cur, cb)
                idx = d.argmin(dim=-1)
                cur = cur - cb[idx]

        cb = model.hrq.vq_layers[li].embeddings.weight.detach().cpu()
        K = cb.shape[0]
        d = euc_dist(cur, cb)
        sid_l = sinkhorn_(d, eps=0.003, n_iters=args.n_iters)
        sids.append(sid_l)
        util = len(set(sid_l.tolist())) / K
        per_layer_util.append(util)
        log.info(f"  L{li} (K={K}): unique={len(set(sid_l.tolist()))}/{K} = {util*100:.2f}%")

    # 4-digit 去重: 按 (L0, L1, L2) 拼接, 重复 digit 合并
    sids_array = np.stack(sids, axis=1)  # (N, 3)
    log.info(f"\n  3-digit SID shape: {sids_array.shape}")

    # 原始 3-digit unique count
    sid_3 = [' '.join(str(x) for x in row) for row in sids_array]
    n_unique_3 = len(set(sid_3))
    log.info(f"  3-digit unique: {n_unique_3}/{len(sid_3)} = {n_unique_3/len(sid_3)*100:.2f}%")

    # 4-digit dedup: 找每个 unique 3-digit 的最频繁 idx, 重复的 de-dup
    from collections import Counter
    sid_3_to_indices = {}
    for i, s in enumerate(sid_3):
        sid_3_to_indices.setdefault(s, []).append(i)
    # 4-digit: 重复 3-digit 编号加 4th digit
    dedup_counter = 0
    sids_4 = []
    for s, idx_list in sid_3_to_indices.items():
        if len(idx_list) == 1:
            sids_4.append((s, -1))  # 第 4 digit = -1 (无)
        else:
            for j, idx in enumerate(idx_list):
                sids_4.append((s, j))
    n_unique_4 = len(set(sids_4))
    log.info(f"  4-digit unique: {n_unique_4}/{len(sids_4)} = {n_unique_4/len(sids_4)*100:.2f}%")

    # collision = 1 - n_unique_4/N
    collision = 1.0 - n_unique_4 / len(sids_4)
    log.info(f"  4-digit collision: {collision:.4f}")

    # Gate 0 决策
    log.info("\n" + "=" * 70)
    log.info("Issue #25 Gate 0 决策")
    log.info("=" * 70)
    log.info(f"  L0/L1/L2 utilization: {per_layer_util[0]*100:.2f}% / "
             f"{per_layer_util[1]*100:.2f}% / {per_layer_util[2]*100:.2f}%")
    log.info(f"  4-digit collision: {collision:.4f}")

    util_pass = all(u >= 0.90 for u in per_layer_util)
    collision_pass = collision <= 0.20
    gate0_passed = util_pass and collision_pass

    log.info(f"\n  通过条件: 三层 util ≥ 90% = {util_pass}")
    log.info(f"  通过条件: collision ≤ 0.20 = {collision_pass}")
    if gate0_passed:
        log.info("\n✅ Issue #25 Gate 0 PASS")
        log.info("   → 进 Gate 1 (Phase B per-layer c_k range 30 epoch warm-start)")
    else:
        log.info("\n❌ Issue #25 Gate 0 FAIL")
        log.info("   → 不进 Gate 1, 关 issue")

    # 落盘
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'ckpt_path': args.ckpt_path,
        'n_e_list': n_e_list,
        'n_layers': n_layers,
        'per_layer_util': per_layer_util,
        '4digit_unique_count': n_unique_4,
        '4digit_total': len(sids_4),
        '4digit_collision': collision,
        'util_pass': util_pass,
        'collision_pass': collision_pass,
        'gate0_passed': gate0_passed,
    }
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nOK 结果落盘: {args.output_json}")


if __name__ == "__main__":
    main()
