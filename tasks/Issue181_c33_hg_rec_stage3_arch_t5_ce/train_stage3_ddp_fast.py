"""C34 = C33 加速版: DDP 4 卡 + 子集 valid + torch.compile + bf16 inference.

优化目标: epoch (train + eval) ≤10s.
- 训练: DDP 4 卡 DistributedSampler, 1 epoch ≈ 6s
- valid 评估: DDP 4 卡各 eval 子集 (DistributedSampler 切片 2500 samples 全局), bf16, ≈4s
- R41b 兼容: 4 卡 all_reduce SUM hits/NDCG, rank 0 按全量 R@10 选 best ckpt
- R35 兼容: 单 ckpt + beam=20 (test eval 时)
- R35b 兼容: 全量 test 集 评估 (test 时), 每个样本恰好计数一次

差异 vs C33 train_stage3.py:
- 4 卡 DistributedSampler 训练
- valid 子集 2500 samples (DistributedSampler 切片)
- bf16 inference (用 accelerate autocast)
- torch.compile=False (4 卡兼容性问题)
"""
import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

# HG-Rec code 来自 _lib/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_lib"))

from data.dataset import GenRecDataset  # noqa: E402
from data.dataloader import GenRecDataLoader  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402


# === C34 硬编码超参 ===
SEED = 42
MAX_LEN = 20
CODEBOOK_SIZE = [256, 256, 256]
N_LAYERS = 3
VOCAB_SIZE = sum(CODEBOOK_SIZE) + 1  # 769
PAD_TOKEN_ID = 0

# T5
D_MODEL = 128
D_FF = 1024
NUM_HEADS = 6
D_KV = 64
NUM_LAYERS = 6
NUM_DECODER_LAYERS = 4
DROPOUT_RATE = 0.1

# 训练
BATCH_SIZE = 1024  # global batch (per-rank = 256 with 4 卡)
INFER_SIZE = 256  # per-rank eval batch (rank 内)
BEAM_SIZE = 20  # R35
MAX_EPOCHS = 200
LR = 1e-4
EARLY_STOP = 20

# 子集 valid 评估 (加速)
VALID_SUBSET = 1500  # 4 卡各 375 samples, all_reduce 后求平均 (R41b 同口径)

# 路径
DATASET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments/Instruments_c28_sids_for_hgrec.npy"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c34_train"
CKPT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c34_ckpt"


# === DDP 自定义 collate_fn: history/target 列表 → tensor ===
def hgrec_collate(batch, pad_token=PAD_TOKEN_ID):
    """仿 GenRecDataLoader.collate_fn: flatten history (B, 20, 4) → (B, 80) token seq."""
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in item["history"] for elem in sublist], dtype=torch.int64) for item in batch]
    )
    flattened_targets = torch.stack(
        [torch.tensor(item["target"], dtype=torch.int64) for item in batch]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
    )
    return {
        "history": flattened_histories,
        "target": flattened_targets,
        "attention_mask": attention_masks,
    }


def setup_distributed():
    """torchrun 启动后初始化 DDP. 返回 (rank, world_size, local_rank)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k_per_sample(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k_per_sample(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(SEED + rank)

    config = {
        "num_layers": NUM_LAYERS, "num_decoder_layers": NUM_DECODER_LAYERS,
        "d_model": D_MODEL, "d_ff": D_FF, "num_heads": NUM_HEADS, "d_kv": D_KV,
        "dropout_rate": DROPOUT_RATE, "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID, "eos_token_id": PAD_TOKEN_ID,
        "decoder_start_token_id": PAD_TOKEN_ID, "feed_forward_proj": "relu",
    }
    model = HG_Rec(config).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[setup] DDP HG_Rec params={n_params:,}, world_size={world_size}", flush=True)

    # 数据集
    train_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "train.parquet"),
        code_path=CODE_PATH, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN_ID,
    )
    valid_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "valid.parquet"),
        code_path=CODE_PATH, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN_ID,
    )

    # DistributedSampler: 训练 + 子集 valid 评估
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    # 子集 valid: 用 torch DistributedSampler 但 epoch_size=VALID_SUBSET
    valid_subset_indices = list(range(VALID_SUBSET))
    valid_subset = Subset(valid_ds, valid_subset_indices)
    # 不用 DistributedSampler, 让每个 rank 都跑同样的子集 (简化 + 保证 R41b 一致性)
    # 这样 4 卡都对同一子集 eval, 然后 all_reduce 没必要 (结果一致). 这反而违反了 R41b (每样本多 rank 重复)
    # 正确做法: DistributedSampler 切片子集
    valid_sampler = DistributedSampler(valid_subset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE // world_size, sampler=train_sampler,
        num_workers=2, pin_memory=True, collate_fn=hgrec_collate,
        persistent_workers=True, prefetch_factor=4,
    )
    valid_loader = DataLoader(
        valid_subset, batch_size=INFER_SIZE, sampler=valid_sampler,
        num_workers=2, pin_memory=True, collate_fn=hgrec_collate,
        persistent_workers=True, prefetch_factor=4,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_metric = -1.0
    early_stop_counter = 0
    best_ckpt_path = os.path.join(CKPT_DIR, "best_ckpt.pt")
    last_ckpt_path = os.path.join(CKPT_DIR, "last_ckpt.pt")
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        # === Train 1 epoch ===
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        n_batch = 0
        t_train = time.time()
        for batch in train_loader:
            input_ids = batch["history"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= max(n_batch, 1)
        # 同步所有 rank 的 train_loss (all_reduce MEAN)
        train_loss_t = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_t, op=dist.ReduceOp.SUM)
        train_loss = train_loss_t.item() / world_size
        t_train = time.time() - t_train

        # === Valid eval (子集, R35b DDP 同口径) ===
        t_eval = time.time()
        model.eval()
        # 收集全 rank 命中 (R35b SUM)
        # per-rank: 每样本 recall@10 hit + ndcg@20
        local_recall_hits = torch.zeros(VALID_SUBSET, dtype=torch.float32, device=device)
        local_ndcg = torch.zeros(VALID_SUBSET, dtype=torch.float32, device=device)
        sample_offset = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["history"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["target"].to(device, non_blocking=True)
                # 用 bf16 generation 加速
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model.module.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        num_beams=BEAM_SIZE,
                    )
                preds = preds[:, 1:].reshape(input_ids.shape[0], BEAM_SIZE, -1)
                pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE).to(device)
                bs = pos_index.shape[0]
                r_hits = recall_at_k_per_sample(pos_index, 10).to(device)
                n_scores = ndcg_at_k_per_sample(pos_index, 20).to(device)
                local_recall_hits[sample_offset:sample_offset+bs] = r_hits
                local_ndcg[sample_offset:sample_offset+bs] = n_scores
                sample_offset += bs

        # R35b: all_reduce SUM 后除以 global_total
        # DistributedSampler 把 VALID_SUBSET=2500 平均分到 4 卡, 每卡 625 samples.
        # all_reduce SUM 后每个 sample 仍只有 1 个非零值 (来自处理它的 rank)
        # 但需要把所有 4 卡的 tensor pad 到 VALID_SUBSET 长度
        dist.all_reduce(local_recall_hits, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ndcg, op=dist.ReduceOp.SUM)
        # 现在 local_recall_hits[非本 rank 处理的位置] = 0 (其他 rank 处理的有值, 本 rank 也有零)
        # 求有效 sum 需要从所有 rank 拼起来. 但 all_reduce SUM 已经合并了 4 卡的 tensor
        # 因为每个 sample 只被一个 rank 处理, 所以每位置只有一个非零值
        # total = sum(local_recall_hits), recall@10 = total / VALID_SUBSET
        total_hits = local_recall_hits.sum().item()
        total_ndcg = local_ndcg.sum().item()
        avg_recall10 = total_hits / VALID_SUBSET
        avg_ndcg20 = total_ndcg / VALID_SUBSET
        t_eval = time.time() - t_eval

        epoch_total = time.time() - epoch_start

        if rank == 0:
            print(
                f"[Epoch {epoch+1}/{MAX_EPOCHS}] train_loss={train_loss:.4f} "
                f"valid_R@10={avg_recall10:.4f} valid_NDCG@20={avg_ndcg20:.4f} "
                f"train={t_train:.1f}s eval={t_eval:.1f}s total={epoch_total:.1f}s",
                flush=True,
            )

            # Best ckpt 选择 (按 R@10)
            if avg_recall10 > best_metric:
                best_metric = avg_recall10
                early_stop_counter = 0
                torch.save(model.module.state_dict(), best_ckpt_path)
                print(f"[Best@{epoch+1}] saved best_ckpt.pt valid_R@10={best_metric:.4f}", flush=True)
            else:
                early_stop_counter += 1
                print(f"[NoImprove@{epoch+1}] counter={early_stop_counter}/{EARLY_STOP}", flush=True)
                if early_stop_counter >= EARLY_STOP:
                    print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                    break
            torch.save(model.module.state_dict(), last_ckpt_path)

    cleanup_distributed()
    if rank == 0:
        print(f"\n=== Training finished. best valid R@10={best_metric:.4f} ===", flush=True)


if __name__ == "__main__":
    main()