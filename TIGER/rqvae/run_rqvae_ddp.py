"""run_rqvae_ddp.py — TIGER Stage 1 RQ-VAE DDP 4 卡 wrapper.

包装 TIGER 原始 rqvae/main.py 为 DDP 4 卡运行 + R51+ 6 项确定性约束.
保留 TIGER 原 Trainer.fit() 逻辑不变, 仅在 fit 之前对 DataLoader 用 DistributedSampler 切片,
model 用 DistributedDataParallel 包装. save_ckpt 在 rank 0 写入 unwrapped state_dict.

启动 (R51+ 6 项确定性约束 + DDP 4 卡):
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    torchrun --nproc_per_node=4 --master_port=29510 run_rqvae_ddp.py \
        --data_path /fs04/ar57/wenyu/GeneRec/TIGER/data/Instruments/item_emb.parquet \
        --ckpt_dir /fs04/ar57/wenyu/GeneRec/TIGER/rqvae/ckpt/Instruments \
        --epochs 3000 --batch_size 1024 --lr 1e-3 --seed 42
"""
import os
import sys
import argparse

# 加入 TIGER 仓库路径
_TIGER_ROOT = "/fs04/ar57/wenyu/GeneRec/TIGER"
_RQVAE_DIR = os.path.join(_TIGER_ROOT, "rqvae")
sys.path.insert(0, _TIGER_ROOT)
sys.path.insert(0, _RQVAE_DIR)

# === R51+ 6 项确定性约束 ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--e_dim", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.003])
    p.add_argument("--kmeans_init", action="store_true", default=True)
    p.add_argument("--kmeans_iters", type=int, default=100)
    p.add_argument("--sk_iters", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--quant_loss_weight", type=float, default=1.0)
    p.add_argument("--dropout_prob", type=float, default=0.0)
    p.add_argument("--bn", action="store_true", default=False)
    p.add_argument("--loss_type", default="mse")
    p.add_argument("--learner", default="AdamW")
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--warmup_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_step", type=int, default=50)
    p.add_argument("--save_limit", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # === DDP 初始化 ===
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # === R51+ seed (rank-specific) ===
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")

    # === 数据 (DistributedSampler) ===
    dataset_obj = EmbDataset(args.data_path)
    sampler = DistributedSampler(dataset_obj, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    loader = DataLoader(
        dataset_obj,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # === 模型 ===
    model = RQVAE(
        in_dim=dataset_obj.dim,
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
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)
    model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    # === 训练 (复用 TIGER Trainer) ===
    args.device = device  # Trainer 内部期望 args.device 是 torch.device
    trainer = Trainer(args, model, len(loader))
    best_loss, best_collision = trainer.fit(loader)

    if rank == 0:
        print(f"[DDP Stage1] best_loss={best_loss:.4f} best_collision={best_collision:.4f}")
        print(f"[DDP Stage1] ckpt_dir={trainer.ckpt_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()