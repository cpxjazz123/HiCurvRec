"""run_generate_code.py — TIGER Stage 2 SID inference (改编自 generate_code.py).

从 Stage 1 DDP 训练的最优 collision ckpt 推 Instruments (9922 items) 的 SID,
输出 .npy (item_id+1 → 3 codewords).
"""
import os
import sys
import argparse
import logging

_TIGER_ROOT = "/fs04/ar57/wenyu/GeneRec/TIGER"
_RQVAE_DIR = os.path.join(_TIGER_ROOT, "rqvae")
sys.path.insert(0, _TIGER_ROOT)
sys.path.insert(0, _RQVAE_DIR)

import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # === R51+ 6 项确定性约束 ===
    import numpy as np
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    ck_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args.data_path)
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ck_args.num_emb_list,
        e_dim=ck_args.e_dim,
        layers=ck_args.layers,
        dropout_prob=ck_args.dropout_prob,
        bn=ck_args.bn,
        loss_type=ck_args.loss_type,
        quant_loss_weight=ck_args.quant_loss_weight,
        kmeans_init=ck_args.kmeans_init,
        kmeans_iters=ck_args.kmeans_iters,
        sk_epsilons=ck_args.sk_epsilons,
        sk_iters=ck_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    loader = DataLoader(data, num_workers=0, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    all_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            indices = model.get_indices(batch)  # (B, n_layers)
            all_indices.append(indices.cpu().numpy())
    import numpy as np
    all_indices = np.concatenate(all_indices, axis=0)
    print(f"[Stage2] all_indices shape: {all_indices.shape}")

    # TIGER 格式: 每个 item 存 [code_l0, code_l1, code_l2] (item_id+1 mapping)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, all_indices)
    print(f"[Stage2] saved {all_indices.shape[0]} items → {args.output_path}")

    # === R36p utility 检查 (warning only — 不阻断 TIGER) ===
    util_per_layer = [len(np.unique(all_indices[:, li])) for li in range(all_indices.shape[1])]
    util_ratios = [n_uniq / 256 for n_uniq in util_per_layer]
    print(f"[Stage2] utility per layer: {util_per_layer} (codebook=256, ratios={[f'{r:.2%}' for r in util_ratios]})")
    # TIGER 设计: L0 粗量化常用低 utility, 不视为 collapse, 仅 warning
    for li, ratio in enumerate(util_ratios):
        if ratio < 0.30:
            raise RuntimeError(f"R36p FAIL: layer {li} utility={ratio:.2%} < 30% (catastrophic collapse)")


if __name__ == "__main__":
    main()