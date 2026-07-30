"""
Task #339 — Issue #52 / Issue #53 Stage 2 — SID 推断 (无 wrapper, 直接 FreeCurvHRQVAE).

从 Stage 1 best_collision_model.pth 加载, 生成 4-digit SID.
跟 Issue #49 流程一致, 跟 Issue #51 区别: 不含 HypPre wrapper.

Usage:
  # Issue #52 Arm A
  python3 scripts/task339_issue52_stage2_infer.py \\
    --ckpt_path products/task339/arm_a/best_collision_model.pth \\
    --output_npy products/task339/arm_a/sid.npy \\
    --device cuda:0

  # Issue #53 Arm B
  python3 scripts/task339_issue52_stage2_infer.py \\
    --ckpt_path products/task340/arm_b/best_collision_model.pth \\
    --output_npy products/task340/arm_b/sid.npy \\
    --device cuda:0
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae_free_curv import FreeCurvHRQVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', required=True)
    p.add_argument('--output_npy', required=True)
    p.add_argument('--data_path', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--kappa_max', type=float, default=None,
                   help='Override kappa_max from ckpt (e.g. 1.0 for Arm B)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.data_path is None:
        args.data_path = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

    data = EmbDataset(args.data_path)
    in_dim = data.dim
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"Data: {len(data)} items, dim={in_dim}")

    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})

    kappa_max = args.kappa_max if args.kappa_max is not None else ckpt_args.get('kappa_max', 2.0)

    model = FreeCurvHRQVAE(
        in_dim=in_dim,
        num_emb_list=ckpt_args.get('num_emb_list', [64, 128, 256]),
        e_dim=ckpt_args.get('e_dim', 32),
        M=ckpt_args.get('M', 1),
        kappa_max=kappa_max,
        layers=ckpt_args.get('layers', [512, 256, 128, 64]),
        dropout_prob=0.0, bn=False,
        loss_type=ckpt_args.get('loss_type', 'poincare'),
        quant_loss_weight=ckpt_args.get('quant_loss_weight', 1.0),
        beta=ckpt_args.get('beta', 0.25),
        kmeans_init=False, kmeans_iters=100,
        sk_eps=ckpt_args.get('sk_epsilons', [0.01, 0.01, 0.01]),
        sk_iters=ckpt_args.get('sk_iters', 50),
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded (Phase {ckpt.get('phase','?')}, epoch {ckpt.get('epoch','?')})")
    print(f"  kappa_max = {kappa_max}")

    print("κ values (per-layer):")
    for lyr_idx, vq in enumerate(model.hrq.vq_layers):
        k = vq.kappa_m().detach().cpu().tolist()
        print(f"  Layer {lyr_idx}: κ={k[0]:.6f}")

    all_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            indices = model.get_indices(batch, use_sk=True)
            all_indices.append(indices.cpu())
    raw_indices = torch.cat(all_indices, dim=0).numpy()  # (N, 3)
    print(f"Raw indices shape: {raw_indices.shape}")

    from collections import Counter
    sid_raw = [tuple(row) for row in raw_indices]
    counts = Counter(sid_raw)
    collision_raw = sum(c - 1 for c in counts.values()) / len(sid_raw)
    unique_raw = len(counts)
    print(f"Raw 3-digit: unique={unique_raw}/{len(sid_raw)}, collision={collision_raw:.4f}")

    occ_counter: dict = {}
    dedup_indices = []
    for row in raw_indices:
        t = tuple(row)
        occ = occ_counter.get(t, 0)
        occ_counter[t] = occ + 1
        dedup_indices.append(list(row) + [occ])
    sid_4d = np.array(dedup_indices, dtype=np.int32)
    sid_4d_unique = len(set(tuple(row) for row in sid_4d))
    print(f"4-digit dedup: unique={sid_4d_unique}/{len(sid_4d)}")
    assert sid_4d_unique == len(sid_4d), f"Dedup failed: {sid_4d_unique} != {len(sid_4d)}"

    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
    np.save(args.output_npy, sid_4d)
    print(f"SID saved: {args.output_npy}")


if __name__ == "__main__":
    main()
