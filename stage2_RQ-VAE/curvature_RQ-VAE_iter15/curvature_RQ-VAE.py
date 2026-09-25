"""DDP RQ-VAE training on pre-computed Amazon-2023 Instruments embeddings.

Loads item_emb.npy (24587, 768) and trains RqVae on 4 GPUs via torchrun.

执行流程按 Step1-Step12 拆分（详见 modules/step_checks.py）:
  Step1   配置硬编码不变量检查
  Step2   数据集加载与有限性检查
  Step3   DDP / distributed 初始化
  Step4   模型结构与参数初始化检查
  Step5   optimizer 覆盖全部可训练参数
  Step6   单个 batch 取数与设备一致性检查
  Step7   RQ-VAE 前向 (encoder + n 层量化 + 残差 + 曲率传输)
  Step8   reconstruction + rqvae loss 汇总
  Step9   反向传播后所有可训练参数梯度非零
  Step10  optimizer.step() 至少更新一项参数
  Step11  curriculum step 同步曲率 c(t)
  Step12  checkpoint 落盘且非空

=== 2026-09-17 MIGRATED FROM curvature_base ===
源: /home/wlia0047/ar57/wenyu/GeneRec/curvature_base/train_rqvae_instruments.py (镜像于 curvature_RQ-VAE_iter5/)
迁移内容: 启用 USE_GEODESIC_MIDPOINT_COMMIT=True, USE_M2_INTRINSIC=True, USE_M3_TRANSPORT=True,
         COMMITMENT_WEIGHT=1.0 (HG-Rec default). 保留 cyclic c(t) curriculum 与 4 卡 DDP.
迁移目的: 修复 v400e 版 layer0 collapse (67/256 → 250+/256 codes).
源文件备份: curvature_RQ-VAE.py.v400e_final.bak (md5=4c4c8ad6...).
源 ckpt 备份: out/rqvae/instruments_v400e_pre_migration/rqvae_final.pt.

启动:
  cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE
  python3 curvature_RQ-VAE.py  (内部 spawn torchrun --nproc_per_node=4)
  或 torchrun --standalone --nproc_per_node=4 --master_port=50200 curvature_RQ-VAE.py

R36/R47: items are precomputed by stage1_GeneEmbedding using sentence-t5-xxl.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡) 或兼容单卡 fallback.
R30/R43: 超参硬编码, 无 CLI 数值超参.
R53 v3.8: 路径和训练超参全部从 curvature_config.py 硬编码导入, 不依赖 cwd 或环境变量.
"""
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# R47 imports — RQ-VAE-Recommender modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.rqvae import RqVae
from data.schemas import SeqBatch
from modules.step_checks import (
    check_step1_config,
    check_step2_dataset,
    check_step3_distributed,
    check_step5_optimizer,
    check_step6_batch,
    check_step9_backward,
    check_step10_update,
    check_step11_curvature,
    check_step12_checkpoint,
)
from modules.sid_quality import (
    build_sids_for_corpus,
    evaluate_sid_quality_full,
    format_metrics as format_sid_metrics,
)
from modules.step_checks import check_step0_clean_output

# R52/R53: 路径相对于 cwd, 从 curvature_config.py 硬编码导入
from curvature_config import (
    ITEM_EMB_NPY as EMB_NPY,
    ITEM_IDS_JSON,
    TRAIN_PARQUET,
    RAW_SIDS_NPY,
    RQVAE_OUT_DIR as OUT_DIR,
    MECHANISM_NAME,
    _CONFIG_DIR as _CONFIG_DIR_RESULTS,
)


# === 超参 (硬编码 R30/R43) ===
SEED = 42

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0  # HG-Rec default beta=1.0 — commitment/codebook loss 等权

# === 训练硬约束 (R30/R43): 100k global steps ===
MAX_GLOBAL_STEPS = 100_000       # 100k steps ≈ 7 分钟@4卡
CKPT_EVERY = 10_000              # 每 10k 全球 step 保存一份 ckpt

# codebook 健康检查: 层 unique code 数 < CODEBOOK_SIZE*该阈值 → 打印 [CODEBOOK WARNING]
CODEBOOK_COLLAPSE_THRESHOLD = 0.10

# 当前启用：cyclic curvature、M2/M3；所有层使用 Möbius residual subtraction。
# iter7: 扩大曲率范围 (0.05 .. 1.5) 并拉长周期 (100k), 让 c(t) 在整个训练期间都更激进
# 探索更锐利的曲率梯度, 同时把行为损失权重大幅提升 (0.05 -> 0.20), 让 train-only 的下一物品
# 约束在 RQ-VAE 量化时获得更强梯度.
# (行为损失的权重与温度在 modules/rqvae.py 中硬编码, 与下方 curvature 常量同源.)
C_CYCLIC_MIN = 0.05
C_CYCLIC_MAX = 1.5
C_CYCLIC_PERIOD = 100_000
GRAD_CLIP_NORM = 1.0
MIDPOINT_LAYER_MASK = [False, False, False]
# iter15: log c_l(t) and fixed u_l at these global steps (rank 0).
CURVATURE_SNAPSHOT_STEPS = frozenset({0, 10_000, 50_000, 100_000})

# iter7 (read-only documentation):
#   BEHAVIOR_LOSS_WEIGHT = 0.20  (modules/rqvae.py)
#   BEHAVIOR_TEMPERATURE  = 0.07  (modules/rqvae.py)
#   CURVATURE_REG_WEIGHT  = 0.005 (modules/rqvae.py)

# === 加速调参 (硬编码, R36 加速 OK) ===
BATCH_SIZE = 640                 # per-GPU batch (4 卡 DDP, 总 batch = 2560)
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
COMPILE = False


class TransitionDataset(Dataset):
    """Uniform item reconstruction plus train-only next-item transition pairs."""

    def __init__(self, emb_path: str, item_ids_path: str, train_path: str):
        arr = np.load(emb_path).astype(np.float32)
        self.embeddings = torch.from_numpy(arr)
        check_step2_dataset(self, INPUT_DIM)
        item_ids = json.loads(Path(item_ids_path).read_text(encoding="utf-8"))
        if len(item_ids) != len(self.embeddings) or item_ids != list(range(len(item_ids))):
            raise ValueError("Stage1 item_ids must match dense Stage0 interaction IDs")

        next_items = [[] for _ in range(len(item_ids))]
        id_to_row = {int(item_id): row for row, item_id in enumerate(item_ids)}
        interactions = pd.read_parquet(train_path, columns=["history", "target"])
        for row in interactions.itertuples(index=False):
            history = row.history
            if len(history) == 0:
                continue
            source_id = int(history[-1])
            target_id = int(row.target)
            if source_id not in id_to_row or target_id not in id_to_row:
                raise ValueError(
                    f"train interaction item id outside Stage1 sidecar: "
                    f"{source_id}->{target_id}"
                )
            next_items[id_to_row[source_id]].append(id_to_row[target_id])
        self.next_items = next_items
        self.active_sources = sum(bool(items) for items in next_items)
        if not self.active_sources:
            raise ValueError("train.parquet contains no usable next-item transitions")
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            print(
                f"[Step2] loaded {emb_path}: shape={tuple(self.embeddings.shape)}, "
                f"train transitions={sum(map(len, next_items))}, "
                f"active sources={self.active_sources}/{len(item_ids)}",
                flush=True,
            )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        targets = self.next_items[idx]
        target_idx = int(np.random.choice(targets)) if targets else idx
        return idx, target_idx, self.embeddings[idx], self.embeddings[target_idx]


def collate_items(batch):
    source_ids, target_ids, source_x, target_x = zip(*batch)
    return (
        torch.tensor(source_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
        torch.stack(source_x, dim=0),
        torch.stack(target_x, dim=0),
    )


def setup_distributed():
    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_ckpt(model, optimizer, global_step, out_dir):
    """rank 0 only: 保存不带 DDP ``module.`` 前缀的 TIGER-compatible checkpoint."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, "rqvae_best.pth")
    base_model = model.module if hasattr(model, "module") else model
    state = {
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"[Step12] saved {path} (step={global_step})", flush=True)
    check_step12_checkpoint(path)


def _build_seq_batch(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
) -> SeqBatch:
    bsz = source_x.shape[0]
    return SeqBatch(
        user_ids=torch.zeros(bsz, dtype=torch.long, device=source_x.device),
        ids=source_ids,
        ids_fut=target_ids,
        x=source_x,
        x_fut=target_x,
        seq_mask=torch.ones(bsz, dtype=torch.long, device=source_x.device),
    )


def _load_layer_norms() -> list[float]:
    """加载 calibrate_residual_scales.py 写出的 layer_norms.json；缺省全 1.0.

    返回 list[float]，长度 = N_LAYERS，值 ∈ (0, 1]。
    训练入口只读一次，避免每次 step 重新解析 JSON。
    """
    norm_path = (
        Path(_CONFIG_DIR_RESULTS)
        / "out/decoder/instruments_hgrec_configs"
        / f"hgrec_{MECHANISM_NAME}"
        / "layer_norms.json"
    )
    if not norm_path.is_file():
        # Exact calibration recorded by iter1's calibrate_residual_scales.log.
        baseline_layer_norms = [0.001, 0.932889, 1.0]
        print(
            f"[Step4] WARN: 未找到 {norm_path}, 使用 iter1 已记录的残差尺度 "
            f"{baseline_layer_norms}",
            flush=True,
        )
        return baseline_layer_norms
    payload = json.loads(norm_path.read_text(encoding="utf-8"))
    layer_norms = payload.get("layer_norms")
    if not isinstance(layer_norms, list) or len(layer_norms) != N_LAYERS:
        observed = type(layer_norms).__name__
        raise ValueError(
            f"{norm_path} 内容形状 {observed} 异常, 期望 {N_LAYERS} 项"
        )
    return [float(value) for value in layer_norms]


def _log_curvature_snapshot(model, global_step: int, residual_u: list[float]) -> None:
    """iter15: print c0,c1,c2 and u0,u1,u2 at mandated steps."""
    layers = model.layers
    cs = [float(layer.get_c().item()) for layer in layers]
    us = [layer.get_u_layer() for layer in layers]
    u_cal = "/".join(f"{v:.6f}" for v in residual_u)
    print(
        f"[iter15_curvature] step={global_step} "
        f"c=[{cs[0]:.6f}, {cs[1]:.6f}, {cs[2]:.6f}] "
        f"u_learned=[{us[0]:.6f}, {us[1]:.6f}, {us[2]:.6f}] "
        f"u_calibrated=[{u_cal}]",
        flush=True,
    )


SCRIPT_DIR = Path(os.path.abspath(__file__)).parent


def main():
    # ---------- Step1 配置不变量 ----------
    check_step1_config(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        n_layers=N_LAYERS,
        c_min=C_CYCLIC_MIN,
        c_max=C_CYCLIC_MAX,
        c_period=C_CYCLIC_PERIOD,
        sk_eps=0.05,
        sk_iters=3,
    )

    # ---------- Step3 DDP 初始化 ----------
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    check_step3_distributed(rank, world_size, local_rank)
    if rank == 0:
        print(f"[Step3] DDP rank={rank}/{world_size} local_rank={local_rank}", flush=True)

    # ---------- Step0 每次训练前清空旧产物 (rank 0 删除 + 其他 rank barrier) ----------
    deleted = check_step0_clean_output(OUT_DIR, rank=rank, barrier=True)
    if rank == 0:
        print(f"[Step0] cleaned {deleted} stale artifact(s) from {OUT_DIR}", flush=True)

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(
            f"=== RQ-VAE Step1-Step12 训练循环 (world_size={world_size}) ===",
            flush=True,
        )
        print(f"[Step1] emb: {EMB_NPY}", flush=True)
        print(f"[Step1] out: {OUT_DIR}", flush=True)
        print(
            f"[Step1] per-GPU batch={BATCH_SIZE} | total={BATCH_SIZE*world_size} | "
            f"workers={NUM_WORKERS} | compile={COMPILE}",
            flush=True,
        )
        print(
            f"[Step1] hidden={HIDDEN_DIMS} embed={EMBED_DIM} codebook={CODEBOOK_SIZE} "
            f"layers={N_LAYERS}",
            flush=True,
        )
        print(
            f"[Step1] target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}",
            flush=True,
        )

    # ---------- Step2: all-item reconstruction + train-only transitions ----------
    ds = TransitionDataset(EMB_NPY, ITEM_IDS_JSON, TRAIN_PARQUET)
    n_items = len(ds)
    sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True,
        seed=SEED, drop_last=True,
    )
    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True,
        collate_fn=collate_items,
    )
    if rank == 0:
        print(
            f"[Step2] {n_items} items | {len(train_loader)} batches/rank | "
            f"{len(train_loader)*world_size} batches/epoch",
            flush=True,
        )

    # ---------- Step3.5 iter1: 加载残差归一化尺度（所有 rank 一致）----------
    residual_layer_norms = _load_layer_norms()
    if rank == 0:
        norm_str = "/".join(f"{value:.6f}" for value in residual_layer_norms)
        print(
            f"[Step3.5] residual_layer_norms (fixed u_l, iter15) = [{norm_str}] | "
            f"freeze_layer_scale=True | mechanism={MECHANISM_NAME}",
            flush=True,
        )

    # ---------- Step4 模型 ----------
    model = RqVae(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        n_layers=N_LAYERS,
        commitment_weight=COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=C_CYCLIC_MIN,
        c_cyclic_max=C_CYCLIC_MAX,
        c_cyclic_period=C_CYCLIC_PERIOD,
        midpoint_layer_mask=MIDPOINT_LAYER_MASK,
        residual_layer_norms=residual_layer_norms,
        freeze_layer_scale=True,
    ).to(device)

    # iter15: warm-start from iter8; u_l frozen at residual calibration.
    iter8_ckpt_path = (
        "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
        "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
    )
    if os.path.isfile(iter8_ckpt_path):
        if rank == 0:
            warm_start_state = torch.load(
                iter8_ckpt_path, map_location=device, weights_only=False
            )
            trainable_curvature_keys = {
                f"layers.{index}.c_layer_scale"
                for index in range(N_LAYERS)
            }
            transferred_weights = {
                name: value
                for name, value in warm_start_state["model"].items()
                if name not in trainable_curvature_keys
            }
            missing, unexpected = model.load_state_dict(
                transferred_weights, strict=False
            )
            if unexpected:
                raise RuntimeError(
                    f"iter15 warm-start unexpected keys: {sorted(unexpected)}"
                )
            for layer in model.layers:
                if layer.freeze_layer_scale and abs(
                    layer.get_u_layer() - layer.c_layer_norm
                ) > 1e-5:
                    raise RuntimeError(
                        "iter15 warm-start: frozen u_l diverges from calibration"
                    )
            print(
                f"[Step4] iter15 warm-start loaded {len(transferred_weights)} tensors "
                f"from {iter8_ckpt_path}; c_layer_scale frozen at residual u_l "
                f"(skipped {sorted(trainable_curvature_keys)})",
                flush=True,
            )
        # Make sure every rank waits for the rank-0 warm-start before entering DDP.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    elif rank == 0:
        print(
            f"[Step4] WARN: iter8 warm-start checkpoint missing at {iter8_ckpt_path}; "
            f"iter15 will train from random initialization",
            flush=True,
        )

    if COMPILE:
        model = torch.compile(model)

    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"[Step4] params={n_params} | DDP wrapped", flush=True)

    # ---------- Step5 优化器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    check_step5_optimizer(optimizer, model.module)
    if rank == 0:
        print(f"[Step5] optimizer covers {len(optimizer.param_groups)} group(s)", flush=True)

    global_step = 0   # 全球 step (= 单卡 step × world_size, 用 all_reduce SUM 同步)
    t_start = time.time()
    last_log_t = t_start
    # 仅打印一次的首个 step 触发证据, 避免每 5s log 把 check marker 淹没
    step_first_pass_logged = {
        6: False, 7: False, 8: False, 9: False, 10: False, 11: False,
    }

    if rank == 0:
        _cur_model = model.module if hasattr(model, "module") else model
        _cur_model.set_curriculum_step(0)
        _log_curvature_snapshot(_cur_model, 0, residual_layer_norms)
        print(
            f"[train] start, target={MAX_GLOBAL_STEPS} global steps (Step1-Step12 loop)",
            flush=True,
        )
    model.train()
    done = False
    while not done:
        sampler.set_epoch(global_step)
        for batch in train_loader:
            # ---------- Step6 paired item batch ----------
            source_ids, target_ids, x, x_fut = (
                value.to(device, non_blocking=True) for value in batch
            )
            check_step6_batch(x, INPUT_DIM, device)
            check_step6_batch(x_fut, INPUT_DIM, device)
            if rank == 0 and not step_first_pass_logged[6]:
                print(
                    f"[Step6] first transition batch ok x={tuple(x.shape)} "
                    f"x_fut={tuple(x_fut.shape)} device={x.device}",
                    flush=True,
                )
                step_first_pass_logged[6] = True
            seq_batch = _build_seq_batch(source_ids, target_ids, x, x_fut)

            # ---------- Step7 + Step8 前向与 loss ----------
            optimizer.zero_grad()
            t_step = time.time()
            out = model(seq_batch)
            loss = out.loss
            if rank == 0 and not step_first_pass_logged[7]:
                ids_shape = tuple(out.sem_ids_shape)
                print(f"[Step7] first forward ok sem_ids.shape={ids_shape}", flush=True)
                step_first_pass_logged[7] = True
            if rank == 0 and not step_first_pass_logged[8]:
                print(
                    f"[Step8] first loss ok loss={float(loss.item()):.4f} "
                    f"rl={float(out.reconstruction_loss.item()):.4f} "
                    f"vl={float(out.rqvae_loss.item()):.4f} "
                    f"behavior={float(out.behavior_loss.item()):.4f} "
                    f"curvature_reg={float(out.curvature_regularization.item()):.6f}",
                    flush=True,
                )
                step_first_pass_logged[8] = True

            # ---------- Step9 反向传播 ----------
            loss.backward()
            # HG-Rec fix: gradient clipping at norm 1.0 (实测稳定 +∞ norm 梯度, 防 NaN)
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            check_step9_backward(model.module)
            if rank == 0 and not step_first_pass_logged[9]:
                print(f"[Step9] first backward ok params_with_grad={sum(1 for p in model.module.parameters() if p.grad is not None)}", flush=True)
                step_first_pass_logged[9] = True

            # ---------- Step10 optimizer.step() ----------
            params_before = [
                parameter.detach().clone()
                for parameter in model.module.parameters()
                if parameter.requires_grad
            ]
            optimizer.step()
            check_step10_update(params_before, model.module)
            if rank == 0 and not step_first_pass_logged[10]:
                print(f"[Step10] first optimizer.step ok", flush=True)
                step_first_pass_logged[10] = True
            global_step += 1

            # 同步全球 step 计数 (rank 0 累加其他 rank 的 +1)
            step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            # ---------- Step11 curriculum step 同步曲率 c(t) ----------
            _cur_model = model.module if hasattr(model, "module") else model
            _cur_model.set_curriculum_step(global_step_sync)
            check_step11_curvature(model.module, global_step_sync, C_CYCLIC_MIN, C_CYCLIC_MAX)
            if rank == 0 and not step_first_pass_logged[11]:
                curv_str = "/".join(f"{float(l.get_c().item()):.3f}" for l in model.module.layers)
                print(f"[Step11] first curvature sync step={global_step_sync} c=[{curv_str}]", flush=True)
                step_first_pass_logged[11] = True
            if rank == 0 and global_step_sync in CURVATURE_SNAPSHOT_STEPS:
                _log_curvature_snapshot(
                    model.module, global_step_sync, residual_layer_norms
                )

            now = time.time()
            if rank == 0 and (now - last_log_t >= 5.0):
                elapsed = now - t_start
                ips_global = global_step_sync / elapsed
                ips_per_gpu = (global_step_sync / world_size) / elapsed
                eta_s = (
                    (MAX_GLOBAL_STEPS - global_step_sync) / ips_global
                    if ips_global > 0 else float("inf")
                )
                usage = [int(u) for u in out.per_layer_usage.tolist()]
                usage_str = "/".join(str(u) for u in usage)
                low_usage = [
                    li for li, u in enumerate(usage)
                    if u < CODEBOOK_SIZE * CODEBOOK_COLLAPSE_THRESHOLD
                ]
                curvs = [float(l.get_c().item()) for l in model.module.layers]
                curv_str = "/".join(f"{c:.3f}" for c in curvs)
                print(
                    f"  step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| behavior={float(out.behavior_loss.item()):.4f} "
                    f"| curv_reg={float(out.curvature_regularization.item()):.6f} "
                    f"| codes={usage_str}/{CODEBOOK_SIZE} "
                    f"| c={curv_str} "
                    f"| ips_g={ips_global:.1f} ips/rk={ips_per_gpu:.1f} "
                    f"| elapsed={elapsed:.1f}s | eta={eta_s:.1f}s",
                    flush=True,
                )
                if low_usage:
                    print(
                        f"  [CODEBOOK WARNING] layer {low_usage} usage={usage_str}/"
                        f"{CODEBOOK_SIZE} (< {CODEBOOK_COLLAPSE_THRESHOLD:.0%}), "
                        f"疑似 collapse, 建议终止检查",
                        flush=True,
                    )
                last_log_t = now

            # ---------- Step12 checkpoint + descriptive SID metrics ----------
            if global_step_sync >= MAX_GLOBAL_STEPS or (
                global_step_sync > 0 and global_step_sync % CKPT_EVERY == 0
            ):
                save_ckpt(model, optimizer, global_step_sync, OUT_DIR)
                dist.barrier()

                # Rank 0 performs corpus inference; the status broadcast keeps
                # every rank at the same boundary even if diagnostics fail.
                sid_build_failed = torch.zeros(1, dtype=torch.int64, device=device)
                if rank == 0:
                    try:
                        sids = build_sids_for_corpus(model, EMB_NPY, device)
                        os.makedirs(os.path.dirname(RAW_SIDS_NPY), exist_ok=True)
                        np.save(RAW_SIDS_NPY, sids)
                        print(
                            f"  [SID save step={global_step_sync}] "
                            f"wrote raw 3-token SIDs {RAW_SIDS_NPY}: "
                            f"shape={sids.shape}",
                            flush=True,
                        )
                    except Exception as sids_exc:
                        print(
                            f"  [SID metrics WARN step={global_step_sync}] "
                            f"SID inference/write failed: {sids_exc!r}",
                            flush=True,
                        )
                        sid_build_failed[0] = 1
                    else:
                        try:
                            metrics = evaluate_sid_quality_full(sids, EMB_NPY)
                            print(
                                f"  [SID metrics step={global_step_sync}] "
                                f"{format_sid_metrics(metrics)}",
                                flush=True,
                            )
                        except Exception as metrics_exc:
                            # Descriptive metrics must never stop stage2 training.
                            print(
                                f"  [SID metrics WARN step={global_step_sync}] "
                                f"metric collection failed: {metrics_exc!r}",
                                flush=True,
                            )

                dist.broadcast(sid_build_failed, src=0)
                if (
                    global_step_sync >= MAX_GLOBAL_STEPS
                    and bool(sid_build_failed.item())
                ):
                    raise RuntimeError(
                        "Final 3-token SID generation failed; refusing to report "
                        "a completed stage2 run with stale SID artifacts"
                    )
            if global_step_sync >= MAX_GLOBAL_STEPS:
                done = True
                break

    if rank == 0:
        print(
            f"[train] done at global_step={global_step_sync}, "
            f"total time={(time.time()-t_start):.1f}s",
            flush=True,
        )


# === 单一 Python 入口: `python curvature_RQ-VAE.py` —— 0 CLI args, 全部 torchrun 旗标 hard-code.
import subprocess as _subprocess
import sys as _sys

_LAUNCHER = {
    "torchrun":   "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun",
    "script":     os.path.abspath(__file__),
    "nproc":      4,
    "master_port": 50200,
    "log":        "logs/train_migrated.log",
}


def _launch_via_torchrun():
    """Fork torchrun subprocess with hard-coded flags."""
    log_dir = os.path.dirname(_LAUNCHER["log"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    cmd = [
        _LAUNCHER["torchrun"],
        "--standalone",
        f"--nproc_per_node={_LAUNCHER['nproc']}",
        f"--master_port={_LAUNCHER['master_port']}",
        _LAUNCHER["script"],
        # 2026-09-24: 把 -u 放到 training_script 后面 (torchrun 不接受自己的参数位置加 -u),
        # 强制 unbuffered stdout, 否则 rank 1-3 print 卡在 4096-byte pipe buffer 里.
        "-u",
    ]
    with open(_LAUNCHER["log"], "wb") as fout:
        rc = _subprocess.call(cmd, stdout=fout, stderr=_subprocess.STDOUT)
    _sys.exit(rc)


if __name__ == "__main__":
    # Already a torchrun worker (RANK set) -> main directly.
    # Otherwise, fork torchrun.
    if "RANK" in os.environ:
        try:
            main()
        finally:
            cleanup_distributed()
    else:
        _launch_via_torchrun()
