"""DDP 4-card training for HG-Rec v6 — LorentzEmbedding ONLY (no HAB, no regularizer).

v5 关键发现: HAB at attention 是 bottleneck, embedding 几何 (Poincare/Lorentz) 在 HAB 主导下无贡献.
v6 = 最干净的隔离实验:
  - LorentzEmbedding (sinh-based expmap0, τ learnable) — 替换 T5 default Embedding
  - DROP HAB (避免 valid→test ratio 0.80 的 overfitting)
  - DROP 曲率正则项 (避免破坏几何)
  - 纯 CE loss

如果 v6 test R@10 ≥ 0.1074: curvature 单独可用 (without HAB), HAB 是反效果.
如果 v6 test R@10 < 0.1074: HG-Rec baseline 已最优, 任何曲率改造无法提升.

DDP 4-card, bf16 eval, EARLY_STOP=20, EVAL_INTERVAL=1 (R41, R41b).
"""
import os
import sys
import math
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from model.hg_rec_curv_v5 import HG_Rec_Curv_V5, LorentzEmbedding
from model.utils import *


def hgrec_collate_fn(batch, pad_token=0):
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
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_sum(value, device):
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def calculate_pos_index(preds, labels, maxk=20):
    labels_cpu = labels.detach().cpu()
    preds_cpu = preds.detach().cpu()
    assert preds_cpu.shape[1] == maxk
    pos_index = torch.zeros((preds_cpu.shape[0], maxk), dtype=torch.bool)
    for i in range(preds_cpu.shape[0]):
        cur_label = labels_cpu[i].tolist()
        for j in range(maxk):
            cur_pred = preds_cpu[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k, device):
    hits = pos_index[:, :k].any(dim=1).float()
    return hits.to(device)


def ndcg_at_k(pos_index, k, device):
    device_pos = pos_index.to(device)
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=device)
    dcg_per_pos = torch.where(device_pos, 1.0 / torch.log2(ranks + 1),
                                torch.tensor(0.0, device=device))
    has_hit = device_pos.any(dim=1)
    first_hit_idx = device_pos.float().argmax(dim=1)
    ndcg = dcg_per_pos.gather(1, first_hit_idx.unsqueeze(1)).squeeze(1)
    ndcg = ndcg * has_hit.float()
    return ndcg


def train_one_epoch(model, train_loader, optimizer, device, rank, epoch):
    """Pure CE training — no regularizer (v6 isolation)."""
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
        ce_loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce_loss.backward()
        optimizer.step()

        total_loss += ce_loss.item()
        count += 1
        if rank == 0:
            pbar.set_postfix(ce=f"{ce_loss.item():.3f}")
    avg_ce = all_reduce_sum(total_loss, device) / int(os.environ["WORLD_SIZE"]) / max(count, 1)
    return avg_ce


@torch.no_grad()
def evaluate_ddp(model, eval_loader, topk_list, beam_size, device, rank):
    model.eval()
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model.module.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          num_beams=beam_size)
        preds = preds[:, 1:].reshape(input_ids.shape[0], beam_size, -1)
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
        n_local += pos_index.shape[0]
        for k in topk_list:
            hits = recall_at_k(pos_index, k, device)
            ndcg = ndcg_at_k(pos_index, k, device)
            global_hits[k] += hits.sum().item()
            global_ndcg_sum[k] += ndcg.sum().item()
    n_global = all_reduce_sum(n_local, device)
    metrics = {}
    for k in topk_list:
        hits_global = all_reduce_sum(global_hits[k], device)
        ndcg_global = all_reduce_sum(global_ndcg_sum[k], device)
        metrics[f"R@{k}"] = hits_global / max(n_global, 1)
        metrics[f"NDCG@{k}"] = ndcg_global / max(n_global, 1)
    return metrics, n_global


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(2025 + rank)

    if rank == 0:
        cur_time = get_local_time()
        log_path = os.path.join("./logs/", "Instruments", cur_time)
        ckpt_path = os.path.join("./ckpt/", "Instruments", cur_time)
        ensure_dir(log_path)
        ensure_dir(ckpt_path)
        print(f"[DDP] world_size={world_size}", flush=True)
    else:
        log_path = None
        ckpt_path = None

    config = dict(
        num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
        num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
        pad_token_id=0, eos_token_id=0,
        feed_forward_proj="relu",
    )
    model = HG_Rec_Curv_V5(config).to(device)
    if rank == 0:
        print(model.n_parameters, flush=True)
        with torch.no_grad():
            pe = model.model.shared
            v = pe.embedding.weight
            v_norm_mean = v.norm(dim=-1).mean().item()
            sqrt_c = math.sqrt(0.5)
            x_spatial_factor = math.sinh(sqrt_c * v_norm_mean) / (sqrt_c * v_norm_mean)
            x_spatial_norm = v_norm_mean * x_spatial_factor * math.exp(pe.log_tau.item())
            print(f"[V6] init: ||v||≈{v_norm_mean:.4f} τ={math.exp(pe.log_tau.item()):.4f} "
                  f"||x_spatial||≈{x_spatial_norm:.4f}", flush=True)
        print(f"[V6] NO HAB, NO regularizer — pure CE + LorentzEmbedding", flush=True)

    # NO HAB install — v6 isolation
    # NO curvature regularizer

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_dataset = GenRecDataset(
        dataset_path="./dataset/Instruments/train.parquet",
        code_path="./dataset/Instruments/Instruments_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy",
        mode='train', codebook_size=[64, 128, 256, 1], max_len=20,
    )
    valid_dataset = GenRecDataset(
        dataset_path="./dataset/Instruments/valid.parquet",
        code_path="./dataset/Instruments/Instruments_t5_rqvae_beta_0.250_codebook_[64,128,256]_sk_0.000.npy",
        mode='evaluation', codebook_size=[64, 128, 256, 1], max_len=20,
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=1024,
                              sampler=train_sampler, num_workers=2,
                              pin_memory=True, drop_last=True, persistent_workers=True,
                              collate_fn=hgrec_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=256,
                              sampler=valid_sampler, num_workers=2,
                              pin_memory=True, persistent_workers=True,
                              collate_fn=hgrec_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0

    EARLY_STOP = 20
    NUM_EPOCHS = 200

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        avg_ce = train_one_epoch(model, train_loader, optimizer, device, rank, epoch)
        t_train = time.time() - t0

        if rank == 0:
            with torch.no_grad():
                pe = model.module.model.shared
                v = pe.embedding.weight
                c = pe.kappa
                sqrt_c = c.sqrt()
                v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
                factor = torch.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm)
                x_spatial = v * factor
                tau = pe.log_tau.exp()
                x_spatial_scaled = x_spatial * tau
                x_norm_mean = x_spatial_scaled.norm(dim=-1).mean().item()
                tau_val = tau.item()
            print(f"[V6] E{epoch} ||x_spatial||={x_norm_mean:.4f} τ={tau_val:.4f}", flush=True)

        t1 = time.time()
        metrics, n_global = evaluate_ddp(model, valid_loader, [5, 10, 20], 20, device, rank)
        t_eval = time.time() - t1

        cur_ndcg20 = metrics["NDCG@20"]
        save_best = False
        if rank == 0:
            print(f"[E{epoch}] ce={avg_ce:.4f} train={t_train:.1f}s eval={t_eval:.1f}s "
                  f"N={int(n_global)} R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}",
                  flush=True)
            if cur_ndcg20 > best_ndcg:
                best_ndcg = cur_ndcg20
                best_epoch = epoch
                early_stop_counter = 0
                save_best = True
            else:
                early_stop_counter += 1

        stop_tensor = torch.zeros(1, device=device)
        if rank == 0:
            stop_tensor[0] = early_stop_counter
        dist.broadcast(stop_tensor, src=0)
        global_counter = int(stop_tensor.item())

        if save_best:
            _ckpt = os.path.join(ckpt_path, f"HG_Rec_curv_v6_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), _ckpt)
            if rank == 0:
                print(f"  Best ckpt saved: {_ckpt} NDCG@20={best_ndcg:.4f}", flush=True)

        if global_counter >= EARLY_STOP:
            if rank == 0:
                print(f"[E{epoch}] EARLY_STOP={global_counter} triggered, best E{best_epoch} NDCG@20={best_ndcg:.4f}",
                      flush=True)
            break

    if rank == 0:
        print(f"[DONE] best_epoch={best_epoch} best_ndcg@20={best_ndcg:.4f}", flush=True)
        meta = dict(best_epoch=best_epoch, best_ndcg_at_20=best_ndcg)
        with open(os.path.join(ckpt_path, "v6_ddp_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    cleanup_ddp()


if __name__ == "__main__":
    main()