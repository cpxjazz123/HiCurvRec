"""DDP 4-card training for HG-Rec v5 — LorentzEmbedding (sinh-based expmap0) + HAB.

v5 关键变化 vs v4:
  - Embedding: PoincareEmbeddingFixed (tanh-based, 卡在 Poincaré 球边界) → LorentzEmbedding (sinh-based, 无饱和)
  - 曲率正则: 监控 Lorentz spatial norm, 不监控 Poincaré expmap norm
  - τ learnable: 缩放 Lorentz 输出到 T5 默认 magnitude
  - κ fixed at 0.5 (R36 #3: 不可学 κ)
  - HAB frozen Dbar + learnable λ (与 v2/v4 相同)

DDP 4-card, bf16 eval, EARLY_STOP=20, EVAL_INTERVAL=1 (R41, R41b).
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from data.dataset import GenRecDataset
from model.hg_rec_curv_v5 import HG_Rec_Curv_V5, LorentzEmbedding
from model.hab import (
    precompute_hab_distances, make_layer_id_lut, HABModule, install_hab,
)
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


def lorentz_norm_regularizer(model_module, target_norm=2.0):
    """Lorentz embedding 曲率正则: ||x_spatial|| → target_norm (空间维度 norm).

    注意: 这是 ||Lorentz spatial part||, 不是 Poincaré ball radius (R=1/√κ).
    Lorentz hyperboloid 没有 norm 上界, target_norm 选 2.0 让 embedding 保持在中等 magnitude.
    """
    pe = model_module.model.shared
    v = pe.embedding.weight  # (vocab, d)
    c = pe.kappa
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    # Lorentz expmap0 spatial part: sinh(sqrt(c)*||v||)/(sqrt(c)*||v||) * v
    sqrt_c_norm = sqrt_c * v_norm
    factor = torch.sinh(sqrt_c_norm) / sqrt_c_norm
    x_spatial = v * factor
    tau = pe.log_tau.exp()
    x_spatial = x_spatial * tau  # (vocab, d)
    mask = torch.ones_like(x_spatial[:, 0])
    mask[pe.padding_idx] = 0
    x_norms = x_spatial.norm(dim=-1)
    masked_norms = x_norms * mask
    mean_norm = masked_norms.sum() / mask.sum().clamp(min=1)
    return ((mean_norm - target_norm) ** 2)


def train_one_epoch(model, train_loader, optimizer, device, rank, epoch, lambda_reg=0.01, target_norm=2.0):
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
        input_ids = batch['history'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['target'].to(device, non_blocking=True)

        optimizer.zero_grad()
        ce_loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        reg = lorentz_norm_regularizer(model.module, target_norm=target_norm)
        loss = ce_loss + lambda_reg * reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_reg += reg.item()
        count += 1
        if rank == 0:
            pbar.set_postfix(ce=f"{ce_loss.item():.3f}", reg=f"{reg.item():.4f}")
    avg_ce = all_reduce_sum(total_ce, device) / int(os.environ["WORLD_SIZE"]) / max(count, 1)
    avg_reg = all_reduce_sum(total_reg, device) / int(os.environ["WORLD_SIZE"]) / max(count, 1)
    return avg_ce, avg_reg


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
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(2025 + rank)

    if rank == 0:
        cur_time = get_local_time()
        log_path = os.path.join("./logs/", "Instruments", cur_time)
        ckpt_path = os.path.join("./ckpt/", "Instruments", cur_time)
        ensure_dir(log_path)
        ensure_dir(ckpt_path)
        logging.basicConfig(
            filename=os.path.join(log_path, 'HG_Rec_curv_hab_v5_ddp.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
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
            print(f"[V5] init: ||v||≈{v_norm_mean:.4f} τ={math.exp(pe.log_tau.item()):.4f} "
                  f"||x_spatial||≈{x_spatial_norm:.4f}", flush=True)

    # HAB
    stage2_ckpt = "./ckpt/Instruments/Aug-14-2026_20-04-16_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"
    D_list, Dbar_list, stats_list = precompute_hab_distances(stage2_ckpt, c_list=(1.0, 1.0, 1.0))
    if rank == 0:
        for s in stats_list:
            print(f"[HAB] L{s['layer']} K={s['K']} median={s['median']:.4f}", flush=True)
    hab = HABModule(Dbar_list, lambda_max=0.20, force_zero_layers=(3,))
    layer_id_lut = make_layer_id_lut(vocab_size=1025)
    model = install_hab(model, hab, layer_id_lut)

    if rank == 0:
        n_hab = sum(p.numel() for p in model.hab_module.parameters())
        n_pe = sum(p.numel() for p in model.model.shared.parameters())
        print(f"[V5] HAB params: {n_hab}, Embedding params (incl τ): {n_pe}", flush=True)

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
    # v5 fix: target_norm = natural Lorentz ||x_spatial|| ≈ 20 (避免 reg 拉走几何).
    # 如果 target=2.0 (v4 默认), reg=(22-2)^2=400, lambda_reg*reg=4.0 > ce_loss, 破坏训练.
    LAMBDA_REG = 0.01
    TARGET_NORM = 20.0  # 匹配 ||x_spatial|| 自然 magnitude

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        avg_ce, avg_reg = train_one_epoch(model, train_loader, optimizer, device, rank, epoch,
                                           lambda_reg=LAMBDA_REG, target_norm=TARGET_NORM)
        t_train = time.time() - t0

        if rank == 0:
            lambda_eff = model.module.hab_module.lambda_eff.detach().cpu().tolist()
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
            print(f"[V5] E{epoch} ||x_spatial||={x_norm_mean:.4f} τ={tau_val:.4f} "
                  f"λ L0={lambda_eff[0]:.4f} L1={lambda_eff[1]:.4f} L2={lambda_eff[2]:.4f}",
                  flush=True)
        else:
            lambda_eff = None

        t1 = time.time()
        metrics, n_global = evaluate_ddp(model, valid_loader, [5, 10, 20], 20, device, rank)
        t_eval = time.time() - t1

        cur_ndcg20 = metrics["NDCG@20"]
        save_best = False
        if rank == 0:
            print(f"[E{epoch}] ce={avg_ce:.4f} reg={avg_reg:.4f} train={t_train:.1f}s eval={t_eval:.1f}s "
                  f"N={int(n_global)} R@10={metrics['R@10']:.4f} NDCG@20={cur_ndcg20:.4f}",
                  flush=True)
            logging.info(f"E{epoch} ce={avg_ce:.4f} reg={avg_reg:.4f} λ={[round(x,4) for x in lambda_eff]} "
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
            _ckpt = os.path.join(ckpt_path, f"HG_Rec_curv_hab_v5_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), _ckpt)
            if rank == 0:
                logging.info(f"Best ckpt saved: {_ckpt} NDCG@20={best_ndcg:.4f}")

        if global_counter >= EARLY_STOP:
            if rank == 0:
                print(f"[E{epoch}] EARLY_STOP={global_counter} triggered, best E{best_epoch} NDCG@20={best_ndcg:.4f}",
                      flush=True)
            break

    if rank == 0:
        print(f"[DONE] best_epoch={best_epoch} best_ndcg@20={best_ndcg:.4f}", flush=True)
        meta = dict(
            best_epoch=best_epoch, best_ndcg_at_20=best_ndcg,
            final_lambda_eff=model.module.hab_module.lambda_eff.detach().cpu().tolist(),
        )
        with open(os.path.join(ckpt_path, "v5_ddp_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    cleanup_ddp()


if __name__ == "__main__":
    main()