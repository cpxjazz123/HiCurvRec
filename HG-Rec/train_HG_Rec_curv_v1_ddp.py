"""DDP 4-card training for HG_Rec_Curv (v1) — Poincaré Embedding + learnable κ.

- torchrun --nproc_per_node=4 train_HG_Rec_curv_v1_ddp.py
- DistributedSampler 切分 train/valid (R35b/R42: 不重复, all_reduce SUM)
- bf16 mixed precision (DDP 加速)
- EARLY_STOP 在 rank 0 触发 (R41, EARLY_STOP=20)
- 每 epoch valid eval (R41b)

预期 epoch 时间: ≤10s (vs 单卡 40s, 4x 加速).
"""
import os
import sys
import math
import random
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 切换工作目录到 HG-Rec (脚本从任意位置调用都能 import)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from model.hg_rec_curv import HG_Rec_Curv
from model.utils import *


def hgrec_collate_fn(batch, pad_token=0):
    """Custom collate: history 是 list-of-list-of-int (item 切分为 L0+L1+L2+0),
    target 是 list-of-int. 拼接成 (B, total_len) 和 (B, target_len)."""
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([e for sub in h for e in sub], dtype=torch.int64) for h in histories]
    )
    flattened_targets = torch.stack(
        [torch.tensor(t, dtype=torch.int64) for t in targets]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if e != pad_token else 0 for e in h], dtype=torch.int64)
         for h in flattened_histories]
    )
    return {'history': flattened_histories, 'target': flattened_targets,
            'attention_mask': attention_masks}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp():
    """Initialize DDP from torchrun env vars."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_sum(value, device):
    """All-reduce SUM scalar tensor across all DDP ranks."""
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def calculate_pos_index(preds, labels, maxk=20):
    """Match predictions to ground truth token sequences. GPU-safe."""
    # preds, labels may be on GPU
    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    assert preds_cpu.shape[1] == maxk, f"preds.shape[1]={preds_cpu.shape[1]} != {maxk}"
    pos_index = torch.zeros((preds_cpu.shape[0], maxk), dtype=torch.bool)
    for i in range(preds_cpu.shape[0]):
        cur_label = labels_cpu[i].tolist()
        for j in range(maxk):
            cur_pred = preds_cpu[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index  # CPU bool tensor


def recall_at_k(pos_index, k, device):
    """Per-sample hits at top-k. Returns CPU tensor of shape (B,)."""
    hits = pos_index[:, :k].any(dim=1).float()
    return hits.to(device)


def ndcg_at_k(pos_index, k, device):
    """Per-sample NDCG at top-k. Returns CPU tensor of shape (B,)."""
    # Rank-based discount: first hit at rank r contributes 1/log2(r+1)
    B = pos_index.shape[0]
    device_pos = pos_index.to(device)
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=device)
    dcg_per_pos = torch.where(device_pos, 1.0 / torch.log2(ranks + 1), torch.tensor(0.0, device=device))
    # For each sample, take dcg at first True position
    has_hit = device_pos.any(dim=1)
    first_hit_idx = device_pos.float().argmax(dim=1)  # if no hit, returns 0
    ndcg = dcg_per_pos.gather(1, first_hit_idx.unsqueeze(1)).squeeze(1)
    ndcg = ndcg * has_hit.float()
    return ndcg


def train_one_epoch(model, train_loader, optimizer, device, rank, epoch):
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    count = 0
    if rank == 0:
        pbar = tqdm(train_loader, ncols=100, desc=f"Train E{epoch}")
    else:
        pbar = train_loader
    for batch in pbar:
        input_ids = batch['history'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['target'].to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss_local = total_loss / max(count, 1)
    avg_loss = all_reduce_sum(avg_loss_local, device) / int(os.environ["WORLD_SIZE"])
    return avg_loss


@torch.no_grad()
def evaluate_ddp(model, eval_loader, topk_list, beam_size, device, rank):
    model.eval()
    # R35b: DistributedSampler guarantees each sample evaluated exactly once
    # Aggregate hits / ndcg as SUM across ranks, divide by global N
    global_hits = {k: 0.0 for k in topk_list}
    global_ndcg_sum = {k: 0.0 for k in topk_list}
    n_local = 0
    if rank == 0:
        pbar = tqdm(eval_loader, ncols=100, desc="Valid")
    else:
        pbar = eval_loader
    for batch in pbar:
        input_ids = batch['history'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['target'].to(device, non_blocking=True)
        # bf16 inference for ~2x speedup
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
        preds = preds[:, 1:].reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
        n_local += pos_index.shape[0]
        for k in topk_list:
            hits = recall_at_k(pos_index, k, device)
            ndcg = ndcg_at_k(pos_index, k, device)
            global_hits[k] += hits.sum().item()
            global_ndcg_sum[k] += ndcg.sum().item()
    # all_reduce SUM across ranks
    world_size = int(os.environ["WORLD_SIZE"])
    n_global = all_reduce_sum(n_local, device)
    metrics = {}
    for k in topk_list:
        hits_global = all_reduce_sum(global_hits[k], device)
        ndcg_global = all_reduce_sum(global_ndcg_sum[k], device)
        metrics[f"R@{k}"] = hits_global / max(n_global, 1)
        metrics[f"NDCG@{k}"] = ndcg_global / max(n_global, 1)
    return metrics, n_global


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--infer_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=1025)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--eos_token_id', type=int, default=0)
    parser.add_argument('--feed_forward_proj', type=str, default='relu')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default='Instruments')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--codebook_size', type=int, nargs='+', default=[64, 128, 256, 1])
    parser.add_argument('--code_path', type=str,
                        default='_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--topk_list', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--kappa_init', type=float, default=1.0)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + rank)

    if rank == 0:
        cur_time = get_local_time()
        log_path = os.path.join(args.log_path, args.dataset_name, cur_time)
        ckpt_path = os.path.join(args.save_path, args.dataset_name, cur_time)
        ensure_dir(log_path)
        ensure_dir(ckpt_path)
        logging.basicConfig(
            filename=os.path.join(log_path, 'HG_Rec_curv_ddp.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        logging.info(f"DDP world_size={world_size}, config={vars(args)}")
        print(f"[DDP] world_size={world_size}", flush=True)
    else:
        log_path = None
        ckpt_path = None

    # Model
    config = dict(
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
        dropout_rate=args.dropout_rate,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        feed_forward_proj=args.feed_forward_proj,
    )
    # 显式重设 PoincareEmbedding 的 kappa_init
    model = HG_Rec_Curv(config).to(device)
    with torch.no_grad():
        model.model.shared.log_kappa.fill_(math.log(args.kappa_init))
    if rank == 0:
        print(model.n_parameters, flush=True)

    # torch.compile 与 genrec_env torch 版本不兼容 (ImportError draw_joint_graph), 改用 bf16 + 大 infer batch 加速
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Datasets
    train_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'train.parquet'),
        code_path=os.path.join(args.dataset_path, args.dataset_name,
                               args.dataset_name + args.code_path),
        mode='train', codebook_size=args.codebook_size, max_len=args.max_len,
    )
    valid_dataset = GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, args.dataset_name, 'valid.parquet'),
        code_path=os.path.join(args.dataset_path, args.dataset_name,
                               args.dataset_name + args.code_path),
        mode='evaluation', codebook_size=args.codebook_size, max_len=args.max_len,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=2,
                              pin_memory=True, drop_last=True, persistent_workers=True,
                              collate_fn=hgrec_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.infer_size,
                              sampler=valid_sampler, num_workers=2,
                              pin_memory=True, persistent_workers=True,
                              collate_fn=hgrec_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, rank, epoch)
        t_train = time.time() - t0

        cur_kappa = model.module.model.shared.kappa.item()
        cur_log_kappa = model.module.model.shared.log_kappa.item()

        t1 = time.time()
        metrics, n_global = evaluate_ddp(model, valid_loader, args.topk_list, args.beam_size, device, rank)
        t_eval = time.time() - t1

        # R41: EARLY_STOP on rank 0 only
        cur_ndcg20 = metrics["NDCG@20"]
        save_best = False
        if rank == 0:
            print(f"[E{epoch}] loss={avg_loss:.4f} train={t_train:.1f}s eval={t_eval:.1f}s "
                  f"κ={cur_kappa:.4f} log_κ={cur_log_kappa:.4f} "
                  f"N={int(n_global)} R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}", flush=True)
            logging.info(f"E{epoch} loss={avg_loss:.4f} κ={cur_kappa:.4f} "
                         f"R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}")
            if cur_ndcg20 > best_ndcg:
                best_ndcg = cur_ndcg20
                best_epoch = epoch
                early_stop_counter = 0
                save_best = True
            else:
                early_stop_counter += 1

        # Broadcast EARLY_STOP across ranks
        stop_tensor = torch.zeros(1, device=device)
        if rank == 0:
            stop_tensor[0] = early_stop_counter
        dist.broadcast(stop_tensor, src=0)
        global_counter = int(stop_tensor.item())

        # Save on rank 0 only
        if save_best:
            _ckpt = os.path.join(ckpt_path, f"HG_Rec_curv_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), _ckpt)
            if rank == 0:
                logging.info(f"Best ckpt saved: {_ckpt} NDCG@20={best_ndcg:.4f}")

        if global_counter >= args.early_stop:
            if rank == 0:
                print(f"[E{epoch}] EARLY_STOP={global_counter} triggered, best E{best_epoch} NDCG@20={best_ndcg:.4f}", flush=True)
            break

    if rank == 0:
        print(f"[DONE] best_epoch={best_epoch} best_ndcg@20={best_ndcg:.4f}", flush=True)
        # Save metadata
        meta = dict(
            best_epoch=best_epoch, best_ndcg_at_20=best_ndcg,
            final_kappa=model.module.model.shared.kappa.item(),
            config=vars(args),
        )
        with open(os.path.join(ckpt_path, "v1_ddp_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    cleanup_ddp()


# time helper (model.utils has get_local_time but no time)
import time
if __name__ == "__main__":
    main()