"""Single-GPU Stage 2 κ training — fallback after DDP codebook collapse discovered.

R42 例外: DDP 4-card causes codebook collapse (collision → 1.0 within 5 epochs).
          即使固定 c=1.0 (无 κ learning) 也无法恢复. baseline single-GPU
          需 ~24 epoch 才能从 initial collapse (E4=0.981) 恢复到 0.119.
          DDP gradient averaging 加剧 collapse, 无法在合理 epoch 内恢复.

约束:
  - 与 baseline 全部超参一致 (R36: 不调参)
  - 加 per-layer learnable κ (R36 #1 唯一尝试路径)
  - 单卡但其他 3 卡空载, 文档化 R42 异常 (R7+R42)
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
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model.hrqvae_kappa import HRQVAE_Kappa, EmbDataset


def get_local_time():
    return time.strftime("%b-%d-%Y_%H-%M-%S", time.localtime())


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def delete_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    # 与 baseline train_hrqvae.py 一致的默认超参
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

    p.add_argument('--learnable_c', type=bool, default=True)
    p.add_argument('--c_init', type=float, default=1.0)
    p.add_argument('--c_min', type=float, default=0.05)
    p.add_argument('--c_max', type=float, default=20.0)

    p.add_argument('--early_stop', type=int, default=100)
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--seed', type=int, default=2024)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0")

    cur_time = get_local_time()
    ckpt_dir = os.path.join(args.ckpt_dir, f"{cur_time}_kappa_single_codebook_[{','.join(map(str, args.num_emb_list))}]_sk_{args.sk_epsilons[2]:.3f}")
    ensure_dir(ckpt_dir)
    log_path = os.path.join(ckpt_dir, "hrqvae_kappa_single.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[SINGLE] args: {vars(args)}")

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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[SINGLE] model params: {n_params:,}")
    if args.learnable_c:
        for i, q in enumerate(model.hrq.vq_layers):
            log(f"  L{i} init: log_c={q.log_c.item():.4f} → c={q.get_c().item():.4f}")

    # 单卡训练 (R42 例外: DDP codebook collapse 不可恢复)
    train_loader = DataLoader(
        data, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, pin_memory=True,
    )

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

    best_collision = float("inf")
    best_loss = float("inf")
    early_stop_counter = 0
    best_epoch = 0
    kappa_history = []
    newest_save_queue = []
    best_save_heap = []

    log(f"[SINGLE] starting training: {args.epochs} epochs, EARLY_STOP={args.early_stop}")

    for epoch in range(args.epochs):
        # train
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        count = 0
        pbar = tqdm(train_loader, ncols=100, desc=f"Train E{epoch}")
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out, rq_loss, indices = model(batch)
            loss, loss_recon = model.compute_loss(out, rq_loss, xs=batch)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError(f"Training loss is NaN/Inf at E{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_recon += loss_recon.item()
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{loss_recon.item():.4f}")
        avg_loss = total_loss / max(count, 1)
        avg_recon = total_recon / max(count, 1)
        t_train = time.time() - t0

        # eval (collision rate)
        t1 = time.time()
        model.eval()
        indices_set = set()
        n_sample = 0
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                indices = model.get_indices(batch, use_sk=False)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                n_sample += len(indices)
                for index in indices:
                    code = "-".join([str(int(_)) for _ in index])
                    indices_set.add(code)
        collision_rate = (n_sample - len(indices_set)) / max(n_sample, 1)
        t_eval = time.time() - t1

        curvs = [q.get_c().item() for q in model.hrq.vq_layers]
        log_c_vals = [q.log_c.item() for q in model.hrq.vq_layers]
        log_c_grads = [q.log_c.grad.item() if q.log_c.grad is not None else 0.0 for q in model.hrq.vq_layers]
        kappa_history.append({
            "epoch": epoch, "loss": avg_loss, "recon": avg_recon,
            "collision_rate": collision_rate, "n_eval": n_sample,
            "curvatures": curvs, "log_c": log_c_vals, "log_c_grad": log_c_grads,
        })
        log(f"[E{epoch:03d}] loss={avg_loss:.4f} recon={avg_recon:.4f} coll={collision_rate:.4f} N={n_sample} "
            f"train={t_train:.1f}s eval={t_eval:.1f}s "
            f"c=[{curvs[0]:.4f}, {curvs[1]:.4f}, {curvs[2]:.4f}] "
            f"grad=[{log_c_grads[0]:.4f}, {log_c_grads[1]:.4f}, {log_c_grads[2]:.4f}]")

        # save best collision
        if collision_rate < best_collision:
            best_collision = collision_rate
            best_epoch = epoch
            early_stop_counter = 0
            ckpt_path = os.path.join(ckpt_dir, "best_collision_model.pth")
            torch.save({
                "args": vars(args),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_collision_rate": best_collision,
                "state_dict": model.state_dict(),
                "kappa_history": kappa_history,
            }, ckpt_path, pickle_protocol=4)
            log(f"  Best ckpt saved: {ckpt_path} collision={best_collision:.4f}")
        else:
            early_stop_counter += 1

        # save epoch snapshot
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth")
        torch.save({
            "args": vars(args),
            "epoch": epoch,
            "best_loss": best_loss,
            "best_collision_rate": best_collision,
            "state_dict": model.state_dict(),
        }, epoch_ckpt, pickle_protocol=4)
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

        if early_stop_counter >= args.early_stop:
            log(f"[E{epoch}] EARLY_STOP={early_stop_counter} triggered, best E{best_epoch} collision={best_collision:.4f}")
            break

    log(f"[DONE] best_epoch={best_epoch} best_collision_rate={best_collision:.4f}")
    with open(os.path.join(ckpt_dir, "kappa_history.json"), "w") as f:
        json.dump(kappa_history, f, indent=2)
    log_f.close()


if __name__ == "__main__":
    main()
