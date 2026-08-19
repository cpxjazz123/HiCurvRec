"""v9 DDP 4-card 训练: Stage 3 T5 FFN learnable curvature.

唯一改动 vs train_HG-Rec.py:
  - 用 HG_Rec_FFN_Curv 替代 HG_Rec (FFN 后加 learnable c magnitude compression)
  - DDP 4-card + DistributedSampler + all_reduce SUM (R35b/R35c)
  - EARLY_STOP=20 (R41)
  - 每 epoch valid eval (R41b)
  - 监控每层 c 漂移
"""
import os
import sys
import json
import time
import math
import random
import argparse
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.hg_rec_ffn_curv import HG_Rec_FFN_Curv, curvature_reg_loss
from model.utils import *


def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_local_time():
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    if not os.path.exists(p): os.makedirs(p, exist_ok=True)


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu(); labels = labels.detach().cpu()
    assert preds.shape[1] == maxk
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True; break
    return pos_index

def recall_at_k(pos_index, k): return pos_index[:, :k].sum(dim=1).cpu().float()
def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def all_reduce_sum(value, device):
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def train_one_epoch(model, train_loader, optimizer, device, rank, epoch):
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    total_ce = 0.0
    total_reg = 0.0
    count = 0
    if rank == 0:
        pbar = tqdm(train_loader, ncols=100, desc=f"Train E{epoch}")
    else:
        pbar = train_loader
    for batch in pbar:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        optimizer.zero_grad()
        # forward returns (total_loss_with_reg, logits)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # separate ce + reg for logging
        with torch.no_grad():
            ce_only = model.module.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            curv_reg = curvature_reg_loss(model.module.model, model.module.lambda_c)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"NaN/Inf at E{epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_ce += ce_only.item()
        total_reg += curv_reg.item()
        count += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{ce_only.item():.3f}", reg=f"{curv_reg.item():.5f}")
    avg_loss = all_reduce_sum(total_loss, device) / dist.get_world_size() / max(count, 1)
    avg_ce = all_reduce_sum(total_ce, device) / dist.get_world_size() / max(count, 1)
    avg_reg = all_reduce_sum(total_reg, device) / dist.get_world_size() / max(count, 1)
    return avg_loss, avg_ce, avg_reg


@torch.no_grad()
def evaluate_4card(model, eval_loader, topk_list, beam_size, device, rank):
    """4 卡各自分片评估, all_reduce 汇总 recall/NDCG (R35b)."""
    model.eval()
    local_recalls = {f'Recall@{k}': 0.0 for k in topk_list}
    local_ndcgs = {f'NDCG@{k}': 0.0 for k in topk_list}
    local_count = 0
    if rank == 0:
        pbar = tqdm(eval_loader, ncols=100, desc="Valid")
    else:
        pbar = eval_loader
    for batch in pbar:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        preds = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
        preds = preds[:, 1:]
        preds = preds.reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
        bs = input_ids.shape[0]
        local_count += bs
        for k in topk_list:
            local_recalls[f'Recall@{k}'] += recall_at_k(pos_index, k).sum().item()
            local_ndcgs[f'NDCG@{k}'] += ndcg_at_k(pos_index, k).sum().item()
    # all_reduce 聚合总和
    for k in topk_list:
        local_recalls[f'Recall@{k}'] = all_reduce_sum(local_recalls[f'Recall@{k}'], device)
        local_ndcgs[f'NDCG@{k}'] = all_reduce_sum(local_ndcgs[f'NDCG@{k}'], device)
    total_count = all_reduce_sum(local_count, device)
    avg_recalls = {k: v / max(total_count, 1) for k, v in local_recalls.items()}
    avg_ndcgs = {k: v / max(total_count, 1) for k, v in local_ndcgs.items()}
    return avg_recalls, avg_ndcgs


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(2025 + rank)

    # ckpt / log 目录
    if rank == 0:
        cur_time = get_local_time()
        ckpt_path = f"./ckpt/Instruments/{cur_time}_v9_ffn_curv"
        log_path = f"./logs/Instruments/{cur_time}_v9_ffn_curv"
        ensure_dir(ckpt_path); ensure_dir(log_path)
        log_f = open(os.path.join(log_path, "v9_ffn_curv_ddp.log"), "w")
        def log(msg):
            print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()
        log(f"[v9] world_size={world_size}")
    else:
        ckpt_path = None; log = lambda msg: None; log_f = None

    config = {
        'batch_size': 256,
        'infer_size': 96,
        'num_epochs': 200,
        'lr': 1e-4,
        'num_layers': 6,
        'num_decoder_layers': 4,
        'd_model': 128,
        'd_ff': 1024,
        'num_heads': 6,
        'd_kv': 64,
        'dropout_rate': 0.1,
        'vocab_size': 1025,
        'pad_token_id': 0,
        'eos_token_id': 0,
        'feed_forward_proj': 'relu',
        'max_len': 20,
        'dataset_name': 'Instruments',
        'dataset_path': './dataset/',
        'codebook_size': [64, 128, 256, 1],
        'code_path': '_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy',
        'beam_size': 20,
        'topk_list': [5, 10, 20],
        'early_stop': 20,
    }
    if rank == 0:
        log(f"[v9] config: {config}")

    model = HG_Rec_FFN_Curv(config, c_init=1.0, lambda_c=1e-4).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"[v9] params: {n_params:,}")
        log(f"[v9] initial curvatures: {model.get_curvatures()}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # dataset
    train_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'train.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='train', codebook_size=config['codebook_size'], max_len=config['max_len'],
    )
    valid_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'valid.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation', codebook_size=config['codebook_size'], max_len=config['max_len'],
    )

    # DDP sampler
    def genrec_collate(batch, pad_token=0):
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]
        flattened_histories = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )
        flattened_targets = torch.stack([torch.tensor(target, dtype=torch.int64) for target in targets])
        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )
        return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=2, collate_fn=genrec_collate)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config['infer_size'], sampler=valid_sampler, num_workers=2, collate_fn=genrec_collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    curvature_history = []

    if rank == 0:
        log(f"[v9] starting: {config['num_epochs']} epochs, EARLY_STOP={config['early_stop']}")

    for epoch in range(config['num_epochs']):
        t0 = time.time()
        avg_loss, avg_ce, avg_reg = train_one_epoch(model, train_loader, optimizer, device, rank, epoch)
        t_train = time.time() - t0

        t1 = time.time()
        avg_recalls, avg_ndcgs = evaluate_4card(model, valid_loader, config['topk_list'], config['beam_size'], device, rank)
        t_eval = time.time() - t1

        if rank == 0:
            curvs = model.module.get_curvatures()
            curvature_history.append({"epoch": epoch, "loss": avg_loss, "ce": avg_ce, "reg": avg_reg,
                                       "R@5": avg_recalls['Recall@5'], "R@10": avg_recalls['Recall@10'], "R@20": avg_recalls['Recall@20'],
                                       "N@5": avg_ndcgs['NDCG@5'], "N@10": avg_ndcgs['NDCG@10'], "N@20": avg_ndcgs['NDCG@20'],
                                       "curvatures": curvs})
            log(f"[E{epoch:03d}] loss={avg_loss:.4f} ce={avg_ce:.4f} reg={avg_reg:.5f} "
                f"R@10={avg_recalls['Recall@10']:.4f} N@20={avg_ndcgs['NDCG@20']:.4f} "
                f"train={t_train:.1f}s eval={t_eval:.1f}s "
                f"c_avg={np.mean(list(curvs.values())):.3f} c_min={min(curvs.values()):.3f} c_max={max(curvs.values()):.3f}")

        save_best = False
        if rank == 0:
            if avg_ndcgs['NDCG@20'] > best_ndcg:
                best_ndcg = avg_ndcgs['NDCG@20']
                best_epoch = epoch
                early_stop_counter = 0
                save_best = True
            else:
                early_stop_counter += 1

        if save_best:
            save_file = os.path.join(ckpt_path, f"HG_Rec_v9_epoch_{epoch}.pth")
            if rank == 0:
                torch.save(model.module.state_dict(), save_file)
                log(f"  Best ckpt saved: {save_file} NDCG@20={best_ndcg:.4f}")

        if rank == 0 and (early_stop_counter >= config['early_stop']):
            log(f"[E{epoch}] EARLY_STOP={early_stop_counter} triggered, best E{best_epoch} NDCG@20={best_ndcg:.4f}")
            break

    if rank == 0:
        log(f"[DONE] best_epoch={best_epoch} best_NDCG@20={best_ndcg:.4f}")
        with open(os.path.join(log_path, "v9_curvature_history.json"), "w") as f:
            json.dump(curvature_history, f, indent=2)
        log_f.close()
    cleanup_ddp()
    os._exit(0)


if __name__ == "__main__":
    main()