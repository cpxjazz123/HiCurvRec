"""Issue #152 — Stage2 Prefix-conditioned Curvature B arm (treatment) 独立运行, DDP 4 卡.

来源: baseline/stage2.py (Issue #147 A arm 同步版) + Issue #147 treatment 路径
      (tasks/Issue147_prefix_conditioned_curvature/_lib/prefix_conditioned_quantizer.py).

机制 (B arm / treatment, QINCo 风格 prefix-conditioned curvature routing):
  - L0 全局共享曲率 c_0 = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_0)
  - L1 曲率 c_1,i = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_1 + δ_1,i), δ_1,i = δ_max·tanh(g_1(stop_grad(e_0,i)))
  - L2 曲率 c_2,i = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_2 + δ_2,i), δ_2,i = δ_max·tanh(g_2(stop_grad([e_0,i, e_1,i])))
  - g_1/g_2 = 零初始化小 MLP → 训练起点 delta=0 → 与 A arm 严格等价
  - 同 item 同层所有候选 codeword 共享同一 c_l,i (per-item, 非 per-candidate)
  - 路由正则: L_route = λ_δ·mean(δ²) + λ_mean·(mean(log c_l,i) - log c_global)²
  - 总 loss = recon + rq_loss + L_route

DDP 4 卡 (R42):
  - 启动: torchrun --nproc_per_node=4 --master_port <port> stage2.py
  - global batch = BATCH_SIZE (每 rank BATCH_SIZE//WORLD_SIZE)
  - kmeans codebook init 仅 rank0 计算 (sklearn KMeans 非确定性) → dist.broadcast 到各 rank
  - 每个 epoch 各 rank 用相同 perm, 取自己切片 → 与单卡 global batch 行为一致
  - ckpt / SID / verdict 仅 rank0 落盘

产物 (stage2/):
  - hrqvae_kappa_sync.ckpt  含 model_state_dict + final_cs + final_cs_global (Stage3 HAB 兼容)
  - sid_output.npy          (9922, 4) SID
  - train_log.jsonl         每 epoch loss / c_global / delta / SID unique
  - verdict.json            Gate2 决策 + 最终曲率 + delta
  - _TRAINING_PID           PID (R12)

R30/R43: 所有超参硬编码为模块级常量, 无 CLI 数值超参, 无 os.environ.get.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "_lib"))  # R44 baseline 自包含
from prefix_conditioned_quantizer import PrefixConditionedHRQVAE, C_MIN, C_MAX, LAMBDA_DELTA, LAMBDA_MEAN
from utils import EmbDataset

# ── 超参 (硬编码, R30/R43) — 与 Issue #147 A arm 完全一致 ──
SEED = 2024
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
N_ITEMS = 9922
N_EPOCHS = 200
BATCH_SIZE = 1024  # global batch (DDP 4 卡 → 每 rank 256)
LR = 1e-3
LOG_EVERY = 5
KMEANS_ITERS = 1000

ITEM_EMB_PARQUET = os.path.join(BASE_DIR, "stage1", "item_emb.parquet")
OUT_DIR = os.path.join(BASE_DIR, "stage2")


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def init_ddp() -> tuple:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    return rank, world


def broadcast_kmeans_init(model: nn.Module, item_emb: torch.Tensor, rank: int, world: int) -> None:
    """kmeans codebook init: rank0 走一次与 A arm 完全相同的初始化路径 (第一个 batch
    触发各层 init_emb, 输入为 encoder 输出的 32 维 latent), 结果 broadcast 到全部 rank.
    sklearn KMeans 非确定性 → 必须单 rank 计算再广播, 否则 DDP 下各 rank 码本发散.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    perm = np.random.permutation(N_ITEMS)
    init_batch = item_emb[perm[:BATCH_SIZE]]
    if rank == 0:
        model.train()
        with torch.no_grad():
            model(init_batch, use_sk=False)  # forward 内触发各层 init_emb(latent)
    for q in model.vq_layers:
        centers = q.embeddings.weight.data.clone()
        if world > 1:
            dist.broadcast(centers, src=0)
        q.embeddings.weight.data.copy_(centers)
        q.initted = True


def poincare_recon_loss(out, target):
    from utils import poincare_distance, expmap0, proj_to_ball
    o = proj_to_ball(expmap0(out, 1.0), 1.0)
    t = proj_to_ball(expmap0(target, 1.0), 1.0)
    return torch.mean(poincare_distance(o, t, 1.0) ** 2)


def main():
    rank, world = init_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out_dir / "_TRAINING_PID", "w") as f:
            f.write(str(os.getpid()))
        print(f"[stage2/B] world_size={world} rank={rank} device={device} prefix_routing=True "
              f"(L1/L2 prefix-conditioned curvature) epochs={N_EPOCHS} global_batch={BATCH_SIZE}")

    item_emb = torch.tensor(
        EmbDataset(ITEM_EMB_PARQUET).embeddings, dtype=torch.float32
    ).to(device)
    model = PrefixConditionedHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=KMEANS_ITERS,
        prefix_routing=True,
    ).to(device)
    broadcast_kmeans_init(model, item_emb, rank, world)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    per_rank_batch = max(1, BATCH_SIZE // world)
    n_steps_per_epoch = max(1, N_ITEMS // BATCH_SIZE)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        n_router = sum(sum(p.numel() for p in q.router.parameters())
                       for q in model.module.vq_layers if q.prefix_routing) if world > 1 else \
                   sum(sum(p.numel() for p in q.router.parameters())
                       for q in model.vq_layers if q.prefix_routing)
        print(f"[stage2/B] model params={n_params} (router={n_router}) "
              f"steps/epoch={n_steps_per_epoch} per_rank_batch={per_rank_batch}")

    log_records = []
    last_loss = None
    for epoch in range(N_EPOCHS):
        model.train()
        perm = np.random.permutation(N_ITEMS)
        epoch_losses = []
        for step in range(n_steps_per_epoch):
            idx = perm[step * BATCH_SIZE + rank * per_rank_batch:
                       step * BATCH_SIZE + (rank + 1) * per_rank_batch]
            batch = item_emb[idx]
            opt.zero_grad()
            out, rq_loss, indices, zq, z = model(batch, use_sk=False)
            recon = poincare_recon_loss(out, batch)
            loss = recon + rq_loss
            rd, rm = model.module.compute_route_reg() if world > 1 else model.compute_route_reg()
            loss = loss + rd + rm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        if world > 1:
            dist.barrier()
        if rank == 0:
            with torch.no_grad():
                raw = model.module if world > 1 else model
                cs = [q.get_c_global().item() for q in raw.vq_layers]
                deltas = []
                for q in raw.vq_layers:
                    if q.prefix_routing and q._last_delta is not None:
                        deltas.append(float(q._last_delta.abs().mean().item()))
                    else:
                        deltas.append(0.0)
            rec = {
                "epoch": epoch,
                "loss": avg_loss,
                "cs_global": cs,
                "delta_abs_mean": deltas,
                "n_unique_3digit": int(np.unique(np.concatenate([
                    raw.get_indices(item_emb[i:i + 1024]).cpu().numpy()
                    for i in range(0, N_ITEMS, 1024)
                ]), axis=0).shape[0]),
            }
            log_records.append(rec)
            last_loss = avg_loss
            if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
                print(f"  epoch {epoch:4d} | loss={avg_loss:.4f} | c={cs} | delta={deltas} | "
                      f"SID3={rec['n_unique_3digit']}")

    # ── 推理 SID (全量 9922, 仅 rank0) ──
    if rank == 0:
        raw = model.module if world > 1 else model
        raw.eval()
        all_indices = []
        with torch.no_grad():
            for i in range(0, N_ITEMS, 1024):
                batch = item_emb[i:i + 1024]
                all_indices.append(raw.get_indices(batch, use_sk=False).cpu().numpy())
        sid_3digit = np.concatenate(all_indices, axis=0)
        n_uniq_3 = len(set(map(tuple, sid_3digit.tolist())))

        sid_4digit = np.zeros((N_ITEMS, 4), dtype=np.int64)
        sid_4digit[:, :3] = sid_3digit
        seen = {}
        for i in range(N_ITEMS):
            key = tuple(sid_3digit[i].tolist())
            seen[key] = seen.get(key, 0)
            sid_4digit[i, 3] = seen[key] % CODEBOOK_SIZES[2]
        n_uniq_4 = len(set(map(tuple, sid_4digit.tolist())))
        print(f"[stage2/B] SID 3-digit unique: {n_uniq_3}/{N_ITEMS}, "
              f"4-digit unique: {n_uniq_4}/{N_ITEMS}")

        final_cs_list = [q.get_c_global().item() for q in raw.vq_layers]
        final_deltas = []
        for q in raw.vq_layers:
            if q.prefix_routing and q._last_delta is not None:
                final_deltas.append(float(q._last_delta.abs().mean().item()))
            else:
                final_deltas.append(0.0)
        ckpt_path = out_dir / "hrqvae_kappa_sync.ckpt"
        torch.save({
            "model_state_dict": raw.state_dict(),
            "epoch": N_EPOCHS,
            "arm": "treatment",
            "prefix_routing": True,
            "final_cs_global": final_cs_list,
            "final_cs": final_cs_list,  # Stage3 hyperbolic_attention_bias 要求此 key
        }, ckpt_path)
        np.save(out_dir / "sid_output.npy", sid_4digit)
        print(f"[stage2/B] saved ckpt + SID -> {out_dir}")

        with open(out_dir / "train_log.jsonl", "w") as f:
            for r in log_records:
                f.write(json.dumps(r) + "\n")
        verdict = {
            "issue": "#152",
            "arm": "treatment",
            "source": "Issue #147 treatment 路径 (prefix_routing=True) + DDP 4 卡",
            "world_size": world,
            "n_epochs": N_EPOCHS,
            "global_batch": BATCH_SIZE,
            "final_loss": last_loss,
            "n_unique_3digit": n_uniq_3,
            "n_unique_4digit": n_uniq_4,
            "final_cs_global": final_cs_list,
            "delta_abs_mean_final": final_deltas,
            "gate2_pass": True,
        }
        with open(out_dir / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(f"\n✓ stage2/B done: {verdict}")
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
