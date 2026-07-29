#!/usr/bin/env python3
"""
Task #306 / Issue #33 / D8 — Gate 1 Stage 1 100 epoch training wrapper
(per-item 软分配 on #30 GO 配置 r_l=[0.1,1,10] + s_l=[2,2,2])

继承 task301_issue30_gate1_stage1_train.py 的 per-layer transforms 机制,
替换 HRQVAE 为 PerItemSoftHRQVAE (per-item 软分配 wrapper).

关键 R11.5 决策:
  - per-layer transforms: r_l=[0.1, 1, 10] + s_l=[2, 2, 2] (沿用 #30 极端值)
  - per-item temperature: τ=1.0 (中性, Gate 0 R2 已验证 init finite)
  - β=0.5 (沿用 #30 训练稳定值, baseline 0.25 task301 实际跑过用 0.5)
  - c_k: baseline U(0.5, 5) (不动, 不走 c_k range 双轴)
  - K=[64, 128, 256] (baseline)
  - seed=42 (单 seed, R11.5 禁 multi-seed)
"""

import sys, os, argparse

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, os.path.join(REPO, 'HG-Rec'))
sys.path.insert(0, os.path.join(REPO, 'HG-Rec', 'model'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

# 导入 Gate 0 实现
from task306_issue33_gate0_peritem_soft_vq import PerItemSoftHRQVAE

# 导入 baseline 训练脚本
import train_hrqvae as _baseline


def parse_args_with_extras():
    """继承 baseline parse_args + 新增 --temperature + --seed.

    必须先从 sys.argv 中过滤掉 extras 再调用 baseline.parse_args(),
    否则 baseline 的 parser 不认识 --temperature / --seed 会报错.
    """
    extras_keys = {'--temperature', '--seed'}
    filtered_argv = []
    i = 1  # 跳过脚本名 sys.argv[0]
    while i < len(sys.argv):
        if sys.argv[i] in extras_keys:
            # 跳过 key + value
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    # 用 filtered_argv 替换 sys.argv, 然后调用 baseline parse_args
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]] + filtered_argv
    base = _baseline.parse_args()
    sys.argv = saved_argv

    # 用独立 parser 解析 extras
    parser = argparse.ArgumentParser(description="Task #306 Gate 1 — per-item soft HRQVAE")
    parser.add_argument('--temperature', type=float, default=1.0, help='per-item soft temperature')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    ns, _ = parser.parse_known_args()
    base.temperature = ns.temperature
    base.seed = ns.seed
    return base


def main():
    args = parse_args_with_extras()
    # === 强制覆盖关键参数 (per-item soft D8 配置) ===
    if args.epochs == 1000:
        args.epochs = 100

    # 替换 baseline model 为 PerItemSoftHRQVAE
    import torch
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from model.utils import EmbDataset
    from model.hrqvae_trainer import Trainer

    seed = args.seed
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f'[TASK306 GATE1] per-item soft HRQVAE, temperature = {args.temperature}, seed = {args.seed}')

    data = EmbDataset(args.data_path)
    model = PerItemSoftHRQVAE(
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
        sk_eps=args.sk_epsilons,  # list per-layer
        sk_iters=args.sk_iters,
        temperature=args.temperature,
    )

    # 应用 per-layer Codebook Transforms (r_l + s_l) on top of per-item soft
    r_l = [0.1, 1.0, 10.0]   # Issue #30 GO 配置
    s_l = [2.0, 2.0, 2.0]
    with torch.no_grad():
        for vq, r, s in zip(model.hrq.vq_layers, r_l, s_l):
            # 切空间几何变换: e → (s · r) · e (identity rotation)
            vq.embeddings.weight.data.mul_(s * r)
    print(f'[TASK306 GATE1] per-layer Codebook Transforms 应用: r_l={r_l}, s_l={s_l}')

    print(model)
    print(f'[TASK306 GATE1] data dim = {data.dim}, dataset size = {len(data)}')

    data_loader = DataLoader(
        data, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))

    print(f'[TASK306 GATE1] 启动 Stage 1 100 epoch 训练 (per-item soft + #30 r_l/s_l GO 配置)')
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print(f'[TASK306 GATE1] Best Loss = {best_loss}')
    print(f'[TASK306 GATE1] Best Collision Rate = {best_collision_rate}')


if __name__ == '__main__':
    main()
