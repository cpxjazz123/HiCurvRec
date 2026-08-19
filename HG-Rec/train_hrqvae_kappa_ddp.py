"""DDP 4-card Stage 2 κ learning — retrain HRQ-VAE with per-branch learnable κ.

机制 (R36 #1):
  Stage 2 训练时, 每层 HVectorQuantization (L0/L1/L2) 学习自己的曲率 c_l = exp(log_c_l).
  默认 c_init=1.0 (与 baseline 一致), 让梯度自动调整.
  损失 = recon_loss (MSE/Poincaré) + quant_loss + commitment_loss.

约束:
  - DDP 4-card (R42)
  - EARLY_STOP=20 (R41)
  - 每 epoch 评估 (R41b)
  - 不调任何超参 (R36): LR/beta/sk_eps/batch_size 都与 baseline 一致
  - 只新增: per-layer log_c 参数 + 学习它

唯一改动 vs train_hrqvae.py:
  - HRQVAE → HRQVAE_Kappa (learnable_c=True)
  - DDP 4-card + DistributedSampler + all_reduce SUM (R35b)
  - EARLY_STOP=20 (R41)
  - 监控 c_l 漂移

产出:
  - ckpt/Instruments/{time}_kappa_ddp/ 目录下:
    - best_collision_model.pth (与 baseline 同名, Stage 3 可直接复用)
    - epoch_*_collision_*.pth (历史快照)
    - kappa_history.json (c_l 训练轨迹)
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

from model.hrqvae_kappa import HRQVAE_Kappa, EmbDataset


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_local_time():
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def all_reduce_sum(value, device):
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def delete_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="DDP HRQ-VAE with learnable κ")
    # 与 baseline train_hrqvae.py 完全一致的默认超参 (R36: 不调参)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--learner', type=str, default="AdamW")
    p.add_argument('--lr_scheduler_type', type=str, default="linear")
    p.add_argument('--warmup_epochs', type=int, default=20)
    p.add_argument('--data_path', type=str, default="./dataset/Instruments/item_emb.parquet")
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--dropout_prob', type=float, default=0.0)
    p.add_argument('--bn', type=bool, default=False)
    p.add_argument('--loss_type', type=str, default="mse")
    p.add_argument('--kmeans_init', type=bool, default=True)
    p.add_argument('--kmeans_iters', type=int, default=1000)
    p.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.000])
    p.add_argument('--sk_iters', type=int, default=50)
    p.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 128, 256])
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--quant_loss_weight', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    p.add_argument('--save_limit', type=int, default=5)
    p.add_argument('--ckpt_dir', type=str, default="./ckpt/Instruments")

    # 新增: κ learning 参数
    p.add_argument('--learnable_c', type=bool, default=True)
    p.add_argument('--c_init', type=float, default=1.0)
    p.add_argument('--c_min', type=float, default=0.05)
    p.add_argument('--c_max', type=float, default=20.0)

    # 新增: 早停 (R41) + DDP
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--eval_interval', type=int, default=1)
    return p.parse_args()


def save_ckpt(state, ckpt_path):
    torch.save(state, ckpt_path, pickle_protocol=4)


def train_one_epoch(model, train_loader, optimizer, scheduler, device, rank, epoch):
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    total_recon = 0.0
    count = 0
    if rank == 0:
        pbar = tqdm(train_loader, ncols=100, desc=f"Train E{epoch}")
    else:
        pbar = train_loader
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        out, rq_loss, indices = model(batch)
        loss, loss_recon = model.module.compute_loss(out, rq_loss, xs=batch)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"Training loss is NaN/Inf at E{epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_recon += loss_recon.item()
        count += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{loss_recon.item():.4f}")
    avg_loss = all_reduce_sum(total_loss, device) / dist.get_world_size() / max(count, 1)
    avg_recon = all_reduce_sum(total_recon, device) / dist.get_world_size() / max(count, 1)
    return avg_loss, avg_recon


@torch.no_grad()
def evaluate_collision(model, eval_loader, device, rank):
    """评估 collision rate (每个样本产生的 unique code 数 / 总样本数).

    4 卡各自分片评估, all_reduce 聚合.
    注意: collision rate 必须基于 rank 本地看到的 unique codes, 然后通过 all_reduce 求和.
    """
    model.eval()
    local_indices_str = set()
    local_num_sample = 0
    if rank == 0:
        pbar = tqdm(eval_loader, ncols=100, desc="Valid ")
    else:
        pbar = eval_loader
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        indices = model.module.get_indices(batch, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        local_num_sample += len(indices)
        for index in indices:
            code = "-".join([str(int(_)) for _ in index])
            local_indices_str.add(code)
    # All-reduce to get total samples + global unique codes
    local_n = torch.tensor(float(local_num_sample), device=device)
    dist.all_reduce(local_n, op=dist.ReduceOp.SUM)
    n_global = int(local_n.item())

    # Serialize local code set, gather to rank 0
    local_codes_list = sorted(local_indices_str)
    gather_list = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_codes_list, gather_list, dst=0)
    if rank == 0:
        global_set = set()
        for codes in gather_list:
            global_set.update(codes)
        n_unique = len(global_set)
        collision_rate = (n_global - n_unique) / max(n_global, 1)
    else:
        collision_rate = 0.0
    return collision_rate, n_global


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(2024 + rank)

    # ckpt / log 目录
    if rank == 0:
        cur_time = get_local_time()
        ckpt_subdir = f"{cur_time}_kappa_ddp_codebook_[{','.join(map(str, args.num_emb_list))}]_sk_{args.sk_epsilons[2]:.3f}"
        ckpt_dir = os.path.join(args.ckpt_dir, ckpt_subdir)
        ensure_dir(ckpt_dir)
        log_path = os.path.join(ckpt_dir, "hrqvae_kappa_ddp.log")
        log_f = open(log_path, "w")
        def log(msg):
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()
        log(f"[DDP] world_size={world_size}")
        log(f"[DDP] args: {vars(args)}")
    else:
        ckpt_dir = None
        log = lambda msg: None
        log_f = None

    # 模型
    data = EmbDataset(args.data_path)
    model = HRQVAE_Kappa(
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
        learnable_c=args.learnable_c,
        c_init=args.c_init,
        c_min=args.c_min,
        c_max=args.c_max,
    ).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"[DDP] model params: {n_params:,}")
        if args.learnable_c:
            for i, q in enumerate(model.hrq.vq_layers):
                log(f"  L{i} init: log_c={q.log_c.item():.4f} → c={q.get_c().item():.4f}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # DDP-safe codebook init: 所有 rank 都调用 init_emb (但只有 rank 0 真做 KMeans, 其他 rank 等 broadcast)
    if rank == 0:
        log("[DDP] running DDP-safe codebook init (rank 0 KMeans, broadcast to all)")
    # 关键: rank 0 用全量数据做 KMeans, 不要 DistributedSampler (否则只看到 1/4 数据)
    if rank == 0:
        init_loader = DataLoader(
            data, batch_size=args.batch_size, num_workers=0,
            shuffle=True, pin_memory=True,
        )
        init_batches = []
        for b in init_loader:
            init_batches.append(b.to(device))
        init_batch_raw = torch.cat(init_batches, dim=0)
        with torch.no_grad():
            init_batch_enc = model.module.encoder(init_batch_raw)
        log(f"[DDP] encoded init batch shape (full dataset): {init_batch_enc.shape}")
        with torch.no_grad():
            for i, q in enumerate(model.module.hrq.vq_layers):
                q.init_emb(init_batch_enc)
        curvs_after_init = [q.get_c().item() for q in model.module.hrq.vq_layers]
        norms_after_init = [q.embeddings.weight.norm(dim=-1).mean().item() for q in model.module.hrq.vq_layers]
        log(f"[DDP] KMeans init done (full dataset). c={curvs_after_init}, codebook norm={norms_after_init}")
        del init_loader, init_batches, init_batch_raw, init_batch_enc
    else:
        # 其他 rank 也必须调用 init_emb (虽然 data 是 None, 不影响, 内部只做 broadcast receive)
        # 但需要准备一个 dummy data tensor, 防止 kmeans 调用 (虽然我们加了 rank check)
        # 实际上 init_emb 在非 rank 0 上不会调用 kmeans, 但 broadcast 需要一个 tensor
        # 先准备 shape 一致的 placeholder
        for i, q in enumerate(model.module.hrq.vq_layers):
            q.init_emb(None)  # rank != 0 会跳过 kmeans 直接 broadcast
    dist.barrier()
    if rank == 0:
        log("[DDP] codebook init complete, all ranks synced")

    # 数据
    train_sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        data, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )

    # Optimizer + scheduler
    if args.learner.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    max_steps = args.epochs * steps_per_epoch
    if args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    else:
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)

    # 训练
    best_collision = float("inf")
    best_loss = float("inf")
    early_stop_counter = 0
    best_epoch = 0
    kappa_history = []
    newest_save_queue = []
    best_save_heap = []

    if rank == 0:
        log(f"[DDP] starting training: {args.epochs} epochs, EARLY_STOP={args.early_stop}")

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss, avg_recon = train_one_epoch(model, train_loader, optimizer, scheduler, device, rank, epoch)
        t_train = time.time() - t0

        # 每 epoch eval (R41b)
        t1 = time.time()
        collision_rate, n_global = evaluate_collision(model, train_loader, device, rank)
        t_eval = time.time() - t1

        # 收集每层 κ
        if rank == 0:
            curvs = [q.get_c().item() for q in model.module.hrq.vq_layers]
            log_c_vals = [q.log_c.item() for q in model.module.hrq.vq_layers]
            log_c_grads = [q.log_c.grad.item() if q.log_c.grad is not None else 0.0 for q in model.module.hrq.vq_layers]
            kappa_history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "recon": avg_recon,
                "collision_rate": collision_rate,
                "n_eval": n_global,
                "curvatures": curvs,
                "log_c": log_c_vals,
                "log_c_grad": log_c_grads,
            })
            log(f"[E{epoch:03d}] loss={avg_loss:.4f} recon={avg_recon:.4f} coll={collision_rate:.4f} N={n_global} "
                f"train={t_train:.1f}s eval={t_eval:.1f}s "
                f"c=[{curvs[0]:.4f}, {curvs[1]:.4f}, {curvs[2]:.4f}] "
                f"grad=[{log_c_grads[0]:.4f}, {log_c_grads[1]:.4f}, {log_c_grads[2]:.4f}]")

        # save best_collision_model.pth
        save_best = False
        if rank == 0:
            if collision_rate < best_collision:
                best_collision = collision_rate
                best_epoch = epoch
                early_stop_counter = 0
                save_best = True
            else:
                early_stop_counter += 1

        if save_best:
            ckpt_path = os.path.join(ckpt_dir, "best_collision_model.pth")
            save_ckpt({
                "args": vars(args),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_collision_rate": best_collision,
                "state_dict": model.module.state_dict(),
                "kappa_history": kappa_history,
            }, ckpt_path)
            if rank == 0:
                log(f"  Best ckpt saved: {ckpt_path} collision={best_collision:.4f}")

        # 保存历史 epoch snapshot (限 save_limit 个)
        if rank == 0:
            epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth")
            save_ckpt({
                "args": vars(args),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_collision_rate": best_collision,
                "state_dict": model.module.state_dict(),
            }, epoch_ckpt)
            now_save = (-collision_rate, epoch_ckpt)
            if len(newest_save_queue) < args.save_limit:
                newest_save_queue.append(now_save)
                heapq.heappush(best_save_heap, now_save)
            else:
                old_save = newest_save_queue.pop(0)
                newest_save_queue.append(now_save)
                if collision_rate < -best_save_heap[0][0]:
                    bad_save = heapq.heappop(best_save_heap)
                    heapq.heappush(best_save_heap, now_save)
                    if bad_save not in newest_save_queue:
                        delete_file(bad_save[1])
                if old_save not in best_save_heap:
                    delete_file(old_save[1])

        # EARLY_STOP (R41)
        if early_stop_counter >= args.early_stop:
            if rank == 0:
                log(f"[E{epoch}] EARLY_STOP={early_stop_counter} triggered, best E{best_epoch} collision={best_collision:.4f}")
            break

    # 训练结束
    if rank == 0:
        log(f"[DONE] best_epoch={best_epoch} best_collision_rate={best_collision:.4f}")
        with open(os.path.join(ckpt_dir, "kappa_history.json"), "w") as f:
            json.dump(kappa_history, f, indent=2)
        log_f.close()
    cleanup_ddp()
    os._exit(0)  # 强制退出, 避免 worker ranks 资源泄漏


if __name__ == "__main__":
    main()
