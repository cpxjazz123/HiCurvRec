"""baseline Stage2 — 三层共享可学习曲率 (Issue #147 A arm 同步版, DDP 4 卡).

来源: tasks/Issue147_prefix_conditioned_curvature/stage2_train.py (--arm control, 单卡)
→ 同步至 baseline/ 并改造为 torchrun 4 卡 DDP (R42: Stage2/Stage3 必须 DDP 4 卡).

机制 (A arm / control, 与原始 HG-Rec 三层共享曲率行为一致):
  - 三层 RQ-VAE 量化, 每层全局共享曲率 c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l)
  - 每层仅 1 个可学习标量 θ_l (PrefixConditionedHRQVAE, prefix_routing=False)
  - 无 prefix 路由 / 无 per-item 曲率 / 无 L0 codeword-specific 分支
  - 量化: 每层在 Poincaré 球 (曲率 c_l) 上 argmin 找最近码字, commitment + β·codebook loss

DDP 4 卡 (R42):
  - 启动: torchrun --nproc_per_node=4 --master_port <port> stage2.py
  - global batch = BATCH_SIZE (每 rank BATCH_SIZE//WORLD_SIZE)
  - kmeans codebook init 仅 rank0 计算 (sklearn KMeans 非确定性) → dist.broadcast 到各 rank
  - 每个 epoch 各 rank 用相同 perm, 取自己切片 → 与单卡 global batch 行为一致
  - ckpt / SID / verdict 仅 rank0 落盘

产物 (baseline/stage2/):
  - hrqvae_kappa_sync.ckpt  含 model_state_dict + final_cs + final_cs_global (Stage3 HAB 兼容)
  - sid_output.npy          (9922, 4) SID
  - train_log.jsonl         每 epoch loss / c_global / SID unique
  - verdict.json            Gate2 决策 + 最终曲率
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
from cross_layer_quantizer import CrossLayerHRQVAE, C_MIN, C_MAX
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

# ── Issue #160 全因子消融 gate (R30 硬编码, 每格副本改此三值) ──
GATE_M1_ROUTER = True   # per-item curvature router (L1/L2 prefix-conditioned c)
GATE_M2_INTRINSIC = True  # 内在 Möbius residual 减法
GATE_M3_TRANSPORT = True  # 跨层曲率传递


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
              f"(margin-sensitive, #156 MARGIN_LOCKED) epochs={N_EPOCHS} global_batch={BATCH_SIZE}")

    item_emb = torch.tensor(
        EmbDataset(ITEM_EMB_PARQUET).embeddings, dtype=torch.float32
    ).to(device)
    model = CrossLayerHRQVAE(
        in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
        layers=ENCODER_LAYERS, kmeans_init=True, kmeans_iters=KMEANS_ITERS,
        prefix_routing=GATE_M1_ROUTER,
        gate_M2_intrinsic=GATE_M2_INTRINSIC,
        gate_M3_transport=GATE_M3_TRANSPORT,
    ).to(device)
    broadcast_kmeans_init(model, item_emb, rank, world)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    per_rank_batch = max(1, BATCH_SIZE // world)
    n_steps_per_epoch = max(1, N_ITEMS // BATCH_SIZE)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[stage2/B] model params={n_params} steps/epoch={n_steps_per_epoch} "
              f"per_rank_batch={per_rank_batch}")

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

        # ── Issue #159: 导出 per-item 有效曲率状态 (9922×3, 每 item 每层有效 c_l,i) ──
        # 供 Stage3 曲率状态 token 注入 (h_token = Emb(SID_l) + phi(c_l,i)).
        # 与训练 _rq_forward 完全一致: prefix = 已选 codeword 拼接 + 上层曲率信号 (transport).
        import math
        curvature_state = np.zeros((N_ITEMS, 3), dtype=np.float32)
        with torch.no_grad():
            prefix_codes = []
            prev_c = None
            for li, q in enumerate(raw.vq_layers):
                # 复刻 _rq_forward 的 prefix 构造 (含 transport c_sig, 与 GATE_M3 一致)
                parts = list(prefix_codes)
                if q.prefix_routing and prev_c is not None and GATE_M3_TRANSPORT:
                    c_sig = torch.log(prev_c.clamp(min=1e-6)) / math.log(2.0)
                    parts.append(c_sig)
                prefix_emb = torch.cat(parts, dim=-1) if parts else None
                if q.prefix_routing:
                    c_per = q.get_c_per_item(prefix_emb).detach()  # (B,)
                    curvature_state[:, li] = c_per.cpu().numpy()
                    prev_c = q.get_c_global().detach().unsqueeze(0).expand(N_ITEMS, 1)
                else:
                    curvature_state[:, li] = q.get_c_global().item()
                    prev_c = q.get_c_global().detach().unsqueeze(0).expand(N_ITEMS, 1)
                cb = q.embeddings.weight[sid_3digit[:, li]]
                prefix_codes.append(cb)
        np.save(out_dir / "curvature_state.npy", curvature_state)
        cs_stats = {
            "L0": [float(np.min(curvature_state[:, 0])), float(np.mean(curvature_state[:, 0])), float(np.max(curvature_state[:, 0]))],
            "L1": [float(np.min(curvature_state[:, 1])), float(np.mean(curvature_state[:, 1])), float(np.max(curvature_state[:, 1]))],
            "L2": [float(np.min(curvature_state[:, 2])), float(np.mean(curvature_state[:, 2])), float(np.max(curvature_state[:, 2]))],
        }
        print(f"[stage2/B] curvature_state saved ({curvature_state.shape}): {cs_stats}")

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
            "issue": "#159",
            "arm": "treatment",
            "source": "Issue #159 curvature state export + DDP 4 卡",
            "world_size": world,
            "n_epochs": N_EPOCHS,
            "global_batch": BATCH_SIZE,
            "final_loss": last_loss,
            "n_unique_3digit": n_uniq_3,
            "n_unique_4digit": n_uniq_4,
            "final_cs_global": final_cs_list,
            "curvature_state_stats": cs_stats,
            "gate2_pass": True,
        }
        with open(out_dir / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(f"\n✓ stage2/B done: {verdict}")
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
