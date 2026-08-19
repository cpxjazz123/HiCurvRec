"""DDP 4-card training for HG_Rec + HAB (v2).

- HG_Rec (baseline T5ForConditionalGeneration, vocab=1025) + HAB at encoder attention
- HAB uses frozen Dbar from Stage2 ckpt (c=1) + learnable λ (Issue #64 v4 mode)
- DDP 4-card, bf16 eval, EARLY_STOP=20
- 不修改 train/valid/test 数据逻辑
"""
import os
import sys
import math
import random
import argparse
import logging
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

# 切换工作目录到 HG-Rec
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from model.hg_rec import HG_Rec
from model.hab import (
    precompute_hab_distances, make_layer_id_lut, HABModule, install_hab,
)
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
    assert preds_cpu.shape[1] == maxk, f"preds.shape[1]={preds_cpu.shape[1]} != {maxk}"
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
    B = pos_index.shape[0]
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
    parser.add_argument('--stage2_ckpt', type=str,
                        default='./ckpt/Instruments/Aug-14-2026_20-04-16_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth')
    parser.add_argument('--hab_lambda_max', type=float, default=0.20)
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
            filename=os.path.join(log_path, 'HG_Rec_hab_v2_ddp.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        logging.info(f"DDP world_size={world_size}, config={vars(args)}")
        print(f"[DDP] world_size={world_size}", flush=True)
    else:
        log_path = None
        ckpt_path = None

    # === HG_Rec 模型 (baseline T5, vocab=1025) ===
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
    model = HG_Rec(config).to(device)
    if rank == 0:
        print(model.n_parameters, flush=True)

    # === HAB 模块: 从 Stage2 ckpt 预计算 Dbar + λ ===
    D_list, Dbar_list, stats_list = precompute_hab_distances(args.stage2_ckpt, c_list=(1.0, 1.0, 1.0))
    if rank == 0:
        for s in stats_list:
            print(f"[HAB] L{s['layer']} K={s['K']} median={s['median']:.4f}", flush=True)
        logging.info(f"HAB stats: {stats_list}")
    hab = HABModule(Dbar_list, lambda_max=args.hab_lambda_max, force_zero_layers=(3,))
    layer_id_lut = make_layer_id_lut(vocab_size=args.vocab_size)
    model = install_hab(model, hab, layer_id_lut)

    # 显式把 hab_module 注册到顶层, 让 DDP 能找到 (model.module.hab_module 在 forward 用)
    # install_hab 已经 add_module 了, 这里仅 sanity check
    if rank == 0:
        n_hab = sum(p.numel() for p in model.hab_module.parameters())
        print(f"[HAB] total params: {n_hab} (Dbar frozen + lambda learnable)", flush=True)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # === 数据集 ===
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

        # 监控 λ
        if rank == 0:
            lambda_eff = model.module.hab_module.lambda_eff.detach().cpu().tolist()
            print(f"[HAB] λ_eff L0={lambda_eff[0]:.4f} L1={lambda_eff[1]:.4f} L2={lambda_eff[2]:.4f}",
                  flush=True)
        else:
            lambda_eff = None

        t1 = time.time()
        metrics, n_global = evaluate_ddp(model, valid_loader, args.topk_list, args.beam_size, device, rank)
        t_eval = time.time() - t1

        cur_ndcg20 = metrics["NDCG@20"]
        save_best = False
        if rank == 0:
            print(f"[E{epoch}] loss={avg_loss:.4f} train={t_train:.1f}s eval={t_eval:.1f}s "
                  f"N={int(n_global)} R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}",
                  flush=True)
            logging.info(f"E{epoch} loss={avg_loss:.4f} λ={[round(x,4) for x in lambda_eff]} "
                         f"R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}")
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
            _ckpt = os.path.join(ckpt_path, f"HG_Rec_hab_v2_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), _ckpt)
            if rank == 0:
                logging.info(f"Best ckpt saved: {_ckpt} NDCG@20={best_ndcg:.4f}")

        if global_counter >= args.early_stop:
            if rank == 0:
                print(f"[E{epoch}] EARLY_STOP={global_counter} triggered, best E{best_epoch} NDCG@20={best_ndcg:.4f}",
                      flush=True)
            break

    if rank == 0:
        print(f"[DONE] best_epoch={best_epoch} best_ndcg@20={best_ndcg:.4f}", flush=True)
        meta = dict(
            best_epoch=best_epoch, best_ndcg_at_20=best_ndcg,
            final_lambda_eff=model.module.hab_module.lambda_eff.detach().cpu().tolist(),
            config=vars(args),
        )
        with open(os.path.join(ckpt_path, "v2_ddp_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    cleanup_ddp()


if __name__ == "__main__":
    main()
