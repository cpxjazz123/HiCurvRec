#!/usr/bin/env python3
"""Task #312 / Issue #35 — Gate 1 Stage 1 100 epoch 训练

R11.5 决策: r_l=[1,1,1] identity + s_l=[2,2,2] (GO 来自 #30)
目的: 验证 s_l 是不是真杠杆 (isolate r_l 边际效应)
- 如 r_l=[1,1,1]+s_l=[2,2,2] R@10 ≈ 0.1022 → s_l 是真杠杆
- 如 r_l=[1,1,1]+s_l=[2,2,2] R@10 ≪ 0.1022 → r_l 极端值是必要条件, s_l alone 不够

继承 task307 in-place transforms 模式 (直接修改 q.embeddings.weight.data)
"""
import sys, os
import torch

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, os.path.join(REPO, 'HG-Rec'))
sys.path.insert(0, os.path.join(REPO, 'HG-Rec', 'model'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import train_hrqvae as _baseline


def apply_per_layer_codebook_transforms(model, radius_list, scale_list):
    """In-place per-layer 几何变换 (沿用 #30 task301 模式, 直接修改 weight)."""
    n_layers = len(model.num_emb_list)
    assert len(radius_list) == n_layers
    assert len(scale_list) == n_layers

    for li, q in enumerate(model.hrq.vq_layers):
        r = radius_list[li]
        s = scale_list[li]
        e_dim = q.embeddings.weight.shape[-1]
        device = q.embeddings.weight.device
        dtype = q.embeddings.weight.dtype

        # Issue #30 模式: weight = weight @ eff.T  (eff = (s · r) · I = (s · r) · eye)
        eff = (s * r) * torch.eye(e_dim, device=device, dtype=dtype)
        with torch.no_grad():
            q.embeddings.weight.data = q.embeddings.weight.data @ eff.t()

        eff_norm = eff.norm().item()
        weight_norm_after = q.embeddings.weight.norm().item()
        print(f'[TASK312] Layer {li}: r_l={r}, s_l={s}, R_l=I, eff_norm={eff_norm:.4f}, weight_norm_after={weight_norm_after:.4f}', flush=True)


def main():
    args = _baseline.parse_args()

    if args.epochs == 1000:
        args.epochs = 100

    import random, numpy as np
    from torch.utils.data import DataLoader
    from model.utils import EmbDataset
    from model.hrqvae import HRQVAE
    from model.hrqvae_trainer import Trainer

    seed = 42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'[TASK312 GATE1] Issue #35: r_l=[1,1,1] identity + s_l=[2,2,2] (vs #30 [0.1,1,10]+[2,2,2])', flush=True)

    data = EmbDataset(args.data_path)
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    # In-place per-layer transforms (Issue #35: r_l identity + s_l=[2,2,2])
    radius_list = [1.0, 1.0, 1.0]
    scale_list = [2.0, 2.0, 2.0]
    apply_per_layer_codebook_transforms(model, radius_list, scale_list)

    print(f'[TASK312 GATE1] data dim = {data.dim}, dataset size = {len(data)}', flush=True)

    data_loader = DataLoader(
        data, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))

    print(f'[TASK312 GATE1] 启动 Stage 1 100 epoch 训练 (r_l identity + s_l=[2,2,2])', flush=True)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print(f'[TASK312 GATE1] Best Loss = {best_loss}', flush=True)
    print(f'[TASK312 GATE1] Best Collision Rate = {best_collision_rate}', flush=True)


if __name__ == '__main__':
    main()
