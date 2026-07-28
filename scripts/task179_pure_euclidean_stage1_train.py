"""Task #179 Stage 1 fork — 纯欧式 RQ-VAE 训练 (Idea 1 dual-branch hyperbolic T5 Stage 1).

Mirror of HG-Rec/train_hrqvae.py but with EuclideanHRQVAE (no expmap0/logmap0/proj_to_ball).
"""
import argparse
import random
import torch
import numpy as np
import logging

from time import time
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae_euclidean import EuclideanHRQVAE
from model.hrqvae_trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train EuclideanHRQVAE model (Task #179 Stage 1)")

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--eval_step', type=int, default=5, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet",
                        help="input data path")
    parser.add_argument("--weight_decay", type=float, default=0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=True, help="use bn or not (argparse type=bool quirk)")
    # Task #179: 纯欧式强制 mse (poincare 在 EuclideanHRQVAE 中 fallback)
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type (mse for Euclidean)")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=1000, help="max kmeans iters")
    # Task #179: 沿用 phonism baseline 配方
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.003, 0.003, 0.003],
                        help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")
    # Task #179: 跟 phonism 一致 (task69/70/84 用 [64,128,256], 但 phonism 用 [32,64,256])
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[32, 64, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    # Task #179: van den Oord 标准 + 跟 task178 修正版一致 (β 挂 commitment, 不挂 codebook)
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    parser.add_argument("--ckpt_dir", type=str, required=True, help="output directory for model")
    return parser.parse_args()


if __name__ == '__main__':
    # Task #179: seed=42 跟 task178 baseline + phonism 严格对齐 (memory: user-no-multiseed-override)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    """build dataset"""
    data = EmbDataset(args.data_path)
    model = EuclideanHRQVAE(
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
    print(model)
    data_loader = DataLoader(data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)