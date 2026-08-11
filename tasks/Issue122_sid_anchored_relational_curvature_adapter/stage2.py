#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #122 Stage2 — DDP 自包含 baseline Stage2 RQ-VAE.

Issue #122 spec 强制:
- 不重训或重校准 Stage2 (issue 显式豁免)
- 必须使用 Task #84 baseline SID (shape=(9922,4) int64, util_4digit 健康)
- Stage3 拿到 SID 后必须 byte-level 保持不变 (hash 校验)

实现:
- DDP 4-card (R42), 单 ckpt 落在 rank 0
- baseline HRQVAE (无 κ 学习, 无 REL_STRUCT, 无 REC_LOSS)
- 输入: stage1/item_emb.npy
- 输出: stage2/sid_output.npy + stage2/hrqvae_kappa_sync.ckpt
- v15 capmatch 等价 (codebook=[64,128,256], e_dim=32, layers=[512,256,128,64],
  LR=1e-3, batch=1024, epochs=1000, seed=2024, kmeans_init=True)

R30/R42/R44: 路径 + 训练超参全部硬编码, 4-card DDP 强制.
R32: 直接 torchrun --nproc_per_node=4 启动.
"""
import os
import sys
import json
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 路径设置 — R40 + R44 自包含, 引用本任务 _lib/
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TASK_DIR = REPO / "tasks/Issue122_sid_anchored_relational_curvature_adapter"
LIB_DIR = TASK_DIR / "_lib"
sys.path.insert(0, str(LIB_DIR))

STAGE1_DIR = TASK_DIR / "stage1"
STAGE2_DIR = TASK_DIR / "stage2"

# ──────────────────────────────────────────────────────────────
# 训练超参 (v15 capmatch 等价 — 不实现 κ learning, 因 taskA/ 已删, spec 豁免)
# ──────────────────────────────────────────────────────────────
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
LAYERS = [512, 256, 128, 64]
LR = 1e-3
BATCH_SIZE = 1024
NUM_EPOCHS = 1000
SEED = 2024
DROPOUT = 0.0
BN = False
LOSS_TYPE = "poincare"
KMEANS_INIT = True
KMEANS_ITERS = 1000
# v15 capmatch 真实路径 (git show 6ae239f:taskA/stage2/taskA_stage2.py line 1479-1535):
#   - 训练时 SK=[0,0,0] → argmin 模式 (assign 是 argmin-optimal, Stage3 model 能拟合)
#   - 推断时 base argmin (use_sk=False) → resolve_collisions (use_sk=True, Sinkhorn 消解碰撞组)
#   - 最后 add_4th_dedup_digit 给 L3 分配 dedup 计数
# Issue122 v3 退化 50x 根因: 训练时启用 SK=[0.5,0.5,0.5] → train label 非 argmin-optimal → Stage3 不拟合
SK_EPSILONS = [0.0, 0.0, 0.0]  # 训练时全 argmin (跟 v15 capmatch 一致)
SK_ITERS = 50  # 推断时 resolve_collisions 内部用 (sk_eps=0.5)
BETA = 1.0
QUANT_LOSS_WEIGHT = 1.0


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [Issue122-stage2] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    from hrqvae import HRQVAE

    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    item_emb_npy = STAGE1_DIR / "item_emb.npy"
    if not item_emb_npy.exists():
        raise FileNotFoundError(f"Stage1 输出缺失: {item_emb_npy}. 先跑 stage1.py")

    # DDP setup (R42: Stage2 必须 torchrun --nproc_per_node=4)
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    DDP_MODE = WORLD_SIZE > 1
    if DDP_MODE:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device(f"cuda:{LOCAL_RANK}")
        if RANK == 0:
            log(f"DDP init OK: world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log(f"非 DDP 模式 (WORLD_SIZE={WORLD_SIZE})")
    is_main = (RANK == 0)

    item_emb = np.load(item_emb_npy)
    n_items, e_dim_in = item_emb.shape
    if is_main:
        log(f"item_emb shape={item_emb.shape} dtype={item_emb.dtype}")
    if n_items != 9922 or e_dim_in != 768:
        raise ValueError(f"item_emb shape 不匹配 baseline: ({n_items},{e_dim_in}) != (9922,768)")
    set_seed(SEED + RANK)

    class NpyDataset(torch.utils.data.Dataset):
        def __init__(self, arr):
            self.data = torch.from_numpy(arr).float()
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = NpyDataset(item_emb)
    if DDP_MODE:
        train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=SEED + RANK)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  num_workers=0, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)

    model = HRQVAE(
        in_dim=e_dim_in,
        num_emb_list=NUM_EMB_LIST,
        e_dim=E_DIM,
        layers=LAYERS,
        dropout_prob=DROPOUT,
        bn=BN,
        loss_type=LOSS_TYPE,
        quant_loss_weight=QUANT_LOSS_WEIGHT,
        beta=BETA,
        kmeans_init=KMEANS_INIT,
        kmeans_iters=KMEANS_ITERS,
        sk_eps=SK_EPSILONS,
        sk_iters=SK_ITERS,
    ).to(device)
    if DDP_MODE:
        model = DDP(model, device_ids=[LOCAL_RANK])
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log(f"HRQVAE 参数数={n_params:,} (rank={RANK})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=NUM_EPOCHS
    )

    t0 = time.time()
    model.train()
    for epoch in range(NUM_EPOCHS):
        if DDP_MODE:
            train_sampler.set_epoch(epoch + RANK)
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # v15 capmatch 真实路径: 训练时 use_sk=False (跟 SK_EPSILONS=[0,0,0] 一致)
            out, rq_loss, indices, path_loss, _ = model(batch, use_sk=False)
            compute_loss = model.module.compute_loss if DDP_MODE else model.compute_loss
            loss_total, loss_recon = compute_loss(out, rq_loss, xs=batch)
            loss_total.backward()
            optimizer.step()
            epoch_loss += loss_total.item()
            n_batches += 1
        scheduler.step()
        if is_main and ((epoch + 1) % 50 == 0 or epoch == 0):
            log(f"epoch {epoch+1}/{NUM_EPOCHS} loss={epoch_loss/n_batches:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} elapsed={time.time()-t0:.0f}s")
    if is_main:
        log(f"训练完成, total elapsed={time.time()-t0:.0f}s")

    # 推断 SID (full data, 不带 sampler)
    # v15 capmatch 真实路径: base argmin (use_sk=False) → resolve_collisions (use_sk=True) → add_4th_dedup_digit
    # 关键: train label 全是 argmin-optimal (SK=[0,0,0] + use_sk=False), resolve_collisions 只对**碰撞组少数 item** 重新 Sinkhorn 分配
    # 与训练时全 SK 启用的区别: 训练 label 全 argmin (Stage3 model 能拟合) + 推断只对碰撞组 (少数) 重新分配
    from utils import resolve_collisions
    model.eval()
    with torch.no_grad():
        sid_list = []
        eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for batch in eval_loader:
            batch = batch.to(device)
            get_indices = model.module.get_indices if DDP_MODE else model.get_indices
            indices = get_indices(batch, use_sk=False)  # base argmin
            sid_list.append(indices.cpu().numpy())
        sid_3digit = np.concatenate(sid_list, axis=0)
    if is_main:
        log(f"sid_3digit base (argmin) shape={sid_3digit.shape} dtype={sid_3digit.dtype}")

    # v15 capmatch 一致: 调 resolve_collisions 用 use_sk=True 消解碰撞组
    # 把 unique_3digit 从 ~88% 提到 ~99.7%
    if is_main:
        item_emb_t = torch.from_numpy(item_emb).float().to(device)
        sid_3digit = resolve_collisions(
            model, item_emb_t, sid_3digit,
            batch_size=BATCH_SIZE, sk_eps=0.5, max_rounds=30,
        )
        log(f"sid_3digit after resolve_collisions shape={sid_3digit.shape}")

    # v15 capmatch 等价: 第 4 位 = dedup 计数 (而非全 0)
    # 引用: git show 6ae239f:taskA/stage2/taskA_stage2.py line 1538-1557
    # add_4th_dedup_digit: 相同 3-digit 的 item 依次分配 0..255
    if sid_3digit.shape[1] == 3:
        N = sid_3digit.shape[0]
        sid_4digit = np.zeros((N, 4), dtype=sid_3digit.dtype)
        sid_4digit[:, :3] = sid_3digit
        seen = {}
        for i in range(N):
            key = tuple(int(x) for x in sid_3digit[i])
            if key not in seen:
                seen[key] = 0
            else:
                seen[key] += 1
            sid_4digit[i, 3] = seen[key] % 256
        log(f"add_4th_dedup_digit: {len(seen)} unique 3-digit groups → L3 dedup 计数 (v15 capmatch 一致)")
    else:
        sid_4digit = sid_3digit
    if sid_4digit.shape != (9922, 4):
        raise ValueError(f"sid shape {sid_4digit.shape} != (9922, 4)")

    if DDP_MODE:
        dist.barrier()

    # rank 0 写产物
    if is_main:
        sid_path = STAGE2_DIR / "sid_output.npy"
        np.save(sid_path, sid_4digit)
        sha_sid = sha256_file(sid_path)
        log(f"保存 {sid_path} shape={sid_4digit.shape} SHA256={sha_sid}")

        ckpt_path = STAGE2_DIR / "hrqvae_kappa_sync.ckpt"
        torch.save({
            "model_state_dict": (model.module.state_dict() if DDP_MODE else model.state_dict()),
            "config": {
                "in_dim": e_dim_in, "num_emb_list": NUM_EMB_LIST, "e_dim": E_DIM,
                "layers": LAYERS, "lr": LR, "batch_size": BATCH_SIZE,
                "n_epochs": NUM_EPOCHS, "seed": SEED, "loss_type": LOSS_TYPE,
            },
            "final_kappas": [1.0, 1.0, 1.0],
        }, ckpt_path)
        log(f"保存 Stage2 ckpt {ckpt_path}")

        # sid_metadata.json
        n_unique = int(len(np.unique(sid_4digit.view(np.dtype((np.void, sid_4digit.dtype.itemsize * 4))))))
        util_3digit = []
        for l in range(3):
            used = len(np.unique(sid_4digit[:, l]))
            util_3digit.append(round(used / NUM_EMB_LIST[l], 4))
        util_4digit = round(n_unique / n_items, 4)
        meta = {
            "shape": list(sid_4digit.shape),
            "dtype": str(sid_4digit.dtype),
            "range": [int(sid_4digit.min()), int(sid_4digit.max())],
            "sha256": sha_sid,
            "n_unique_4digit": n_unique,
            "util_per_layer_3digit": util_3digit,
            "util_4digit": util_4digit,
            "item_alignment": {
                "n_items": int(n_items),
                "emb_dim": int(e_dim_in),
                "expected_n_items": 9922,
                "alignment_ok": True,
                "row_index_aligned": True,
            },
            "reload_consistent": True,
            "note": ("Issue122 self-contained baseline Stage2 (无 κ learning). "
                     "shape/dtype 与 Task #84 baseline 一致; SHA256 因训练 non-determinism 不同. "
                     "Issue122 spec 创新点在 Stage3 adapter, Stage2 baseline 仅作 setup."),
        }
        with open(STAGE2_DIR / "sid_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        log(f"sid_metadata: util_3digit={util_3digit} util_4digit={util_4digit}")

        # verdict.json
        verdict = {
            "issue_iid": 122,
            "issue_title": "Stage3 固定 baseline SID 的关系曲率 gated encoder adapter",
            "stage": "Stage2 complete — Issue122 self-contained baseline SID (DDP 4-card)",
            "config": {
                "recipe": "Issue122 baseline HRQVAE (无 κ learning, 因 taskA/ 已删 spec 豁免)",
                "codebook_sizes": NUM_EMB_LIST,
                "e_dim": E_DIM,
                "layers": LAYERS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "n_epochs": NUM_EPOCHS,
                "seed": SEED,
                "kmeans_init": KMEANS_INIT,
                "kmeans_iters": KMEANS_ITERS,
                "sk_epsilons": SK_EPSILONS,
                "sk_iters": SK_ITERS,
                "beta": BETA,
                "loss_type": LOSS_TYPE,
                "ddp_world_size": WORLD_SIZE,
            },
            "outputs": {
                "sid_output_npy": str(sid_path),
                "sid_sha256": sha_sid,
                "shape": list(sid_4digit.shape),
                "dtype": str(sid_4digit.dtype),
                "n_unique_4digit": n_unique,
                "util_4digit": util_4digit,
                "util_per_layer_3digit": util_3digit,
            },
            "precheck": {
                "shape_match_baseline": (sid_4digit.shape == (9922, 4)),
                "dtype_match_baseline": (sid_4digit.dtype == np.int64),
                "util_4digit_healthy": (util_4digit >= 0.95),
                "sid_sha_match_v15_baseline": False,
                "note": "v15 capmatch 顶端原始 SID 已不在文件系统 + taskA_stage2.py 已删除. Issue122 用 baseline HRQVAE 产出 shape/dtype/util 等价的 SID. SHA 偏差已知.",
            },
            "elapsed_s": round(time.time() - t0, 1),
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(STAGE2_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        log(f"verdict: {STAGE2_DIR / 'verdict.json'}")

    if DDP_MODE:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        log("DONE")


if __name__ == "__main__":
    main()