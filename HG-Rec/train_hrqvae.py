import argparse
import os
import random as _random
import torch
import numpy as np
import logging

# === R51+ 6 确定性约束 (Stage 1 HRQ-VAE 训练, R47 联动) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

from time import time
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from model.hrqvae import HRQVAE
from model.hrqvae_trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train RQVAE model")

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--eval_step', type=int, default=5, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default=f"./dataset/Beauty/item_emb.parquet", help="input data path")
    parser.add_argument("--weight_decay", type=float, default=0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="poincare", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=1000, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.000], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64,128,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')

    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/Beauty", help="please specify output directory for model")

    return parser.parse_args()

if __name__ == '__main__':
    # R51+: Stage 1 HRQ-VAE 单卡, seed=42 单 seed (DDP 由 torchrun 调度)
    seed = 42
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    #logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path)
    model = HRQVAE(in_dim=data.dim,
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
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)


