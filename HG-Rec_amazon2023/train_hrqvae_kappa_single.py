"""Single-GPU Stage 1 HRQ-VAE + κ training — fallback after DDP codebook collapse.

R42 例外: DDP 4-card causes codebook collapse (collision → 1.0 within 5 epochs).
          即使固定 c=1.0 (无 κ learning) 也无法恢复. baseline single-GPU
          需 ~24 epoch 才能从 initial collapse (E4=0.981) 恢复到 0.119.
          DDP gradient averaging 加剧 collapse, 无法在合理 epoch 内恢复.

约束:
  - 与 baseline 全部超参一致 (R36: 不调参)
  - 加 per-layer learnable κ (R36 #1 唯一尝试路径)
  - 单卡但其他 3 卡空载, 文档化 R42 异常 (R7+R42)

R53 v3.8 改造 (HG-Rec_amazon2023):
  - argparse 全部数值超参 → 模块级常量 (R30/R43 联动)
  - 唯一 CLI 参数: --tag (路径类)
  - 数据集: Amazon_2023_Instruments (24587 items), codebook=[256,256,256]
  - R51+ 6 确定性约束全开 (PYTHONHASHSEED=42 + CUBLAS_WORKSPACE_CONFIG)
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

# === R51+ 6 确定性约束 (硬编码, 无 setdefault) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

# === R47 HF cache 强制路径 ===
os.environ["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model.hrqvae_kappa import HRQVAE_Kappa, EmbDataset

# === R53 v3.8: 模块级常量 (替代 argparse 数值超参) ===
CONFIG = {
    # 数据集路径 (相对 cwd, 与原版兼容)
    "data_path": "./dataset/Amazon_2023_Instruments/item_emb.parquet",
    "ckpt_dir": "./ckpt/Amazon_2023_Instruments",

    # RQ-VAE 结构 (3 层 256 codebook 与 curvature_base 一致, 给 24K items 充分容量)
    "num_emb_list": [256, 256, 256],
    "layers": [512, 256, 128, 64],
    "e_dim": 32,
    "loss_type": "mse",
    "kmeans_init": True,
    "kmeans_iters": 1000,
    "sk_epsilons": [0.0, 0.0, 0.000],
    "sk_iters": 50,
    "quant_loss_weight": 1.0,
    "beta": 1.0,

    # 可学习 κ
    "learnable_c": True,
    "c_init": 1.0,
    "c_min": 0.05,
    "c_max": 20.0,

    # 优化
    "lr": 1e-3,
    "weight_decay": 0.0,
    "dropout_prob": 0.0,
    "bn": False,
    "learner": "AdamW",
    "lr_scheduler_type": "linear",
    "warmup_epochs": 20,

    # 训练
    "epochs": 500,
    "batch_size": 1024,
    "num_workers": 4,
    "save_limit": 5,

    # 早停 + 评估 (R41 联动)
    "early_stop": 20,   # R41: 统一 20 (原版 100 是历史遗留, 本任务用新规则)
    "eval_interval": 1, # 每 epoch eval (与 R41b 一致)
    "seed": 42,         # R51+: seed 硬编码
}


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
    """R30/R43: 仅保留路径类 CLI 参数 (--tag), 数值超参全部硬编码到 CONFIG."""
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, default='v1_baseline',
                   help='ckpt 子目录名后缀 (避免同名覆盖)')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(CONFIG["seed"])
    device = torch.device("cuda:0")

    cur_time = get_local_time()
    codebook_tag = f"[{','.join(map(str, CONFIG['num_emb_list']))}]"
    sk_tag = f"{CONFIG['sk_epsilons'][2]:.3f}"
    ckpt_dir = os.path.join(
        CONFIG["ckpt_dir"],
        f"{cur_time}_kappa_single_{args.tag}_codebook_{codebook_tag}_sk_{sk_tag}"
    )
    ensure_dir(ckpt_dir)
    log_path = os.path.join(ckpt_dir, "hrqvae_kappa_single.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[SINGLE] config: {CONFIG}")
    log(f"[SINGLE] tag: {args.tag}")

    data = EmbDataset(CONFIG["data_path"])
    model = HRQVAE_Kappa(
        in_dim=data.dim,
        num_emb_list=CONFIG["num_emb_list"],
        e_dim=CONFIG["e_dim"],
        layers=CONFIG["layers"],
        dropout_prob=CONFIG["dropout_prob"],
        bn=CONFIG["bn"],
        loss_type=CONFIG["loss_type"],
        quant_loss_weight=CONFIG["quant_loss_weight"],
        beta=CONFIG["beta"],
        kmeans_init=CONFIG["kmeans_init"],
        kmeans_iters=CONFIG["kmeans_iters"],
        sk_eps=CONFIG["sk_epsilons"],
        sk_iters=CONFIG["sk_iters"],
        learnable_c=CONFIG["learnable_c"],
        c_init=CONFIG["c_init"],
        c_min=CONFIG["c_min"],
        c_max=CONFIG["c_max"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[SINGLE] model params: {n_params:,}")
    if CONFIG["learnable_c"]:
        for i, q in enumerate(model.hrq.vq_layers):
            log(f"  L{i} init: log_c={q.log_c.item():.4f} → c={q.get_c().item():.4f}")

    # 单卡训练 (R42 例外: DDP codebook collapse 不可恢复)
    train_loader = DataLoader(
        data, batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"], shuffle=True, pin_memory=True,
    )

    if CONFIG["learner"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    steps_per_epoch = len(train_loader)
    warmup_steps = CONFIG["warmup_epochs"] * steps_per_epoch
    max_steps = CONFIG["epochs"] * steps_per_epoch
    if CONFIG["lr_scheduler_type"] == "linear":
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

    log(f"[SINGLE] starting training: {CONFIG['epochs']} epochs, EARLY_STOP={CONFIG['early_stop']}")

    for epoch in range(CONFIG["epochs"]):
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
                "args": CONFIG,
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
            "args": CONFIG,
            "epoch": epoch,
            "best_loss": best_loss,
            "best_collision_rate": best_collision,
            "state_dict": model.state_dict(),
        }, epoch_ckpt, pickle_protocol=4)
        now_save = (-collision_rate, epoch_ckpt)
        if len(newest_save_queue) < CONFIG["save_limit"]:
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

        if early_stop_counter >= CONFIG["early_stop"]:
            log(f"[E{epoch}] EARLY_STOP={early_stop_counter} triggered, best E{best_epoch} collision={best_collision:.4f}")
            break

    log(f"[DONE] best_epoch={best_epoch} best_collision_rate={best_collision:.4f}")
    with open(os.path.join(ckpt_dir, "kappa_history.json"), "w") as f:
        json.dump(kappa_history, f, indent=2)
    log_f.close()


if __name__ == "__main__":
    main()
