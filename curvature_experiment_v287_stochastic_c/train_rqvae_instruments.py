"""DDP RQ-VAE training on pre-computed Musical_Instruments embeddings.

Loads item_emb.npy (9922, 768) and trains RqVae on 4 GPUs via torchrun.

启动:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
      /home/wlia0047/ar57/wenyu/GeneRec/RQ-VAE-Recommender/train_rqvae_instruments.py

R36/R47: items are precomputed by preprocess_instruments.py using sentence-t5-xxl.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.

step 计数 = 全球 step (all_reduce SUM 每 step), 与 RQ-VAE-Recommender 论文 400k iter 对齐.
每 50k 全球 step rank 0 保存一次 ckpt; 训完保存 final.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# R47 imports — RQ-VAE-Recommender modules
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode, QuantizeDistance
from modules.tokenizer.semids import SemanticIdTokenizer
from data.schemas import SeqBatch

# R52/R53: 路径相对 cwd, 从 curvature_config.py 硬编码导入
from curvature_config import ITEM_EMB_NPY as EMB_NPY, RQVAE_OUT_DIR as OUT_DIR


# === 超参 (硬编码 R30/R43) ===
SEED = 42

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0  # HG-Rec default beta=1.0 (vs 我们 0.25) — commitment/codebook loss 等权

# === 训练硬约束 (R30/R43): 100k global steps ===
NUM_EPOCHS = 10000               # 语义参考
MAX_GLOBAL_STEPS = 100_000       # 100k steps ≈ 7 分钟@4卡
CKPT_EVERY = 10_000              # 每 10k 全球 step 保存一份 ckpt

# === C5: 曲率 margin 正则 (新曲率正则项, R36) — 实测 L1/L2 93%/92% item margin<0.1 ===
MARGIN_REG_WEIGHT = 1.0          # 正则权重
MARGIN_TARGET = 0.05             # 目标 margin (d2-d1), 当前 L1/L2 中位数 ~0.03

# === F3 v83: Poincaré Spread Loss (反向 v82 Center) — codebook 元素互相远离 ===
# 反向 v82 F2 Center: v82 推 codebook 到原点 (compact, weight=0.01), v83 推 codebook 互相远离 (spread)
# 论文支撑: Poincaré Embeddings (Nickel & Kiela 2017) + Contrastive Loss (Hadsell et al. CVPR 2006)
# MARGIN=2.0 (Poincaré ball 直径 ≈ 5.0, MARGIN=2.0 即 40% of ball, 适中强度)
USE_SPREAD_LOSS = True           # F3 v83: enable spread loss
SPREAD_LOSS_WEIGHT = 0.005       # F3 v83: 正则权重 (低权重, 仅作引导, 不主导 recon loss)
SPREAD_LOSS_MARGIN = 2.5         # F3 v83: pairwise 距离阈值 (Poincaré 单位)
# codebook 健康检查: 层 unique code 数 < CODEBOOK_SIZE*该阈值 → 打印 [CODEBOOK WARNING]
CODEBOOK_COLLAPSE_THRESHOLD = 0.10

# v119 RKHS Gaussian Kernel VQ: kernel(x, c) = exp(-||x-c||²/(2σ²)) 替代 Euclidean/双曲距离
DISTANCE_MODE = QuantizeDistance.L2  # v262 NOVEL: L2 + hyperbolic_distance=True (Poincaré ball argmin) + Möbius commit path (M2 intrinsic)
RBF_BANDWIDTH = 1.0  # 备用
MAHALANOBIS_INIT_VAR = 1.0  # 备用
# θ 曲率学习检查 (M2/M3): 曲率初始值 与 "已学习" 判定阈值
CURV_INIT_C = 1.0                         # 曲率初始值 c=1.0 (HG-Rec 对齐)
CURV_LEARNED_TOL = 0.01                   # |c - 1.25| > 0.01 视为曲率在学习
# C26: 固定 c=1 (HG-Rec 极简, 无可学曲率)
USE_FIXED_CURVATURE = True                # C26 HG-Rec default: c=1 固定 (关 M2/M3/C5)
C_FIXED = 1.0                             # 固定曲率值 (HG-Rec)
# C27: Curriculum Curvature Schedule (ICML 2025)
# 机制: c 从 c_start (近欧氏) 线性增到 c_end (双曲) over curriculum_steps
USE_CURRICULUM_CURVATURE = False          # v270 NOVEL: 关 C27 (cyclic 取代)
C_START = 0.05                            # 初始 c (近欧氏, 训练稳定)
C_END = 0.7                               # v19: 最终 c 1.0→0.7 (缓和曲率调度, R36 框架级变更)
CURRICULUM_STEPS = 50_000                 # 50k 步 ramp up (总 100k 步, 后半段稳定)
# C27b (v270 NOVEL): Cyclic Curriculum Curvature — c(t) = c_min + (c_max-c_min)|sin(πt/T)|
# 论文支撑: SGDR (Loshchilov & Hutter 2017), cyclical LR. 应用到 curvature schedule.
USE_CYCLIC_CURVATURE = True               # v270 启用 cyclic (vs linear curriculum)
C_CYCLIC_MIN = 0.05                       # cyclic c_min (近欧氏)
C_CYCLIC_MAX = 0.7                        # cyclic c_max (双曲)
C_CYCLIC_PERIOD = 25_000                  # cyclic 周期 T (步数, 训练 100k 步 ≈ 4 周期)
# HG-Rec 实现 fix: gradient clipping (HG-Rec 用 clip_grad_norm_(1.0))
GRAD_CLIP_NORM = 1.0                      # 0=关闭, HG-Rec 用 1.0
# C28: 启用 M3 cross-layer transport (与 curriculum 联动)
# 机制: curriculum ramp c 时, M3 防止不同 c 层之间 residual 几何不一致
# 论文支撑: M3 = "Parallel Transport in Hyperbolic Space" (Chami et al. 2019)
USE_M3_TRANSPORT = True                   # C28: 启用 M3 (与 C27 互补)
# C29: 启用 M2 intrinsic Möbius 减法 (R37 rollback: M2 + M3 联合恶化 L0 collapse, test R@10 0.0991→0.0941)
USE_M2_INTRINSIC = True                   # v262 NOVEL: 启用 M2 Möbius intrinsic commit (commit path = Poincaré ball 上测地线 residual subtraction). R36n (e) 几何变换 合规
# v282 NOVEL: Geodesic Midpoint Commit (R36n e variant #3, vs baseline Möbius_sub)
# 公式: res_mid = exp_0((log_0(res) + log_0(emb))/2, c)
# 论文: Ungar 2008 Gyrogroup / Fréchet mean closed-form
USE_GEODESIC_MIDPOINT_COMMIT = True       # v282 启用 geodesic midpoint commit (覆盖 Möbius_sub)
# v287 NOVEL: Stochastic c perturbation (R36n n 几何变换变种 #6)
# commit path 保持 v282 midpoint 不变, c 每 batch 加 N(0, sigma) 噪声.
# 论文支撑: Loshchilov & Hutter 2017 SGDR + Foret et al. ICLR 2021 SAM 噪声鲁棒优化.
# 与 v286 stochastic commit 的本质差异: v286 随机 commit 路径 (破坏 codebook 一致性),
# v287 固定 commit 路径仅扰动 c (保留 v282 R36h baseline 的 commit 几何).
# R36n (a) dynamic c variant #2 — R36h ceiling 第 23 次验证.
USE_STOCHASTIC_C = False                   # v287 启用 stochastic c perturbation
STOCHASTIC_C_SIGMA = 0.05                 # c 噪声标准差 (相对 cyclic 当前 c)
# C22: TCU (τ-Geometric Codebook Update) — Riemannian centroid tracking per batch
USE_TCU = False                  # 默认关闭 (C10 baseline), C22 切到 True 启用
TCU_ALPHA = 0.05                 # EMA momentum (新几何位置混合比)
TCU_ETA = 0.1                    # Riemannian step 大小 (切空间单位)

# C23: MCDQ (Mixed-Curvature Distance Quantization) — 每层 dist = (1-α)·d_P + α·d_E
# 论文支撑: "Learning Mixed-Curvature Representations" (Gu et al. ICLR 2019),
#          "Product Manifolds" (Chami et al. ICML 2021).
USE_MCDQ = False                # C23: 默认关闭 (R37 rollback), C23 切到 True 启用
MCDQ_ALPHA_INIT = 0.5            # α 初始值 (sigmoid(θ)=0.5 → 等权混合)

# C24: SCS (Sinkhorn Curvature Scaling) — Sinkhorn eps ∝ 1/c_l (几何驱动, 非调参)
USE_SCS = False                 # C24 R37 rollback (test R@10=0.0949 < baseline 0.0967)
SCS_EPS_SCALE = 1.0             # SCS 缩放指数 (eps = sk_eps / c_l^SCS_EPS_SCALE)

# === 加速调参 (硬编码, R36 加速 OK) ===
BATCH_SIZE = 640                 # per-GPU batch (4 卡 DDP, 总 batch = 2560)
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
COMPILE = False


class EmbeddingDataset(Dataset):
    def __init__(self, emb_path: str):
        arr = np.load(emb_path).astype(np.float32)
        self.embeddings = torch.from_numpy(arr)
        if dist.get_rank() == 0:
            print(f"[data] loaded {emb_path}: shape={tuple(self.embeddings.shape)}, dtype={self.embeddings.dtype}", flush=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def collate_items(batch):
    return torch.stack(batch, dim=0)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def save_ckpt(model, optimizer, global_step, out_dir, tag):
    """rank 0 only: 保存 model.module.state_dict() + optimizer + step."""
    if dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, f"rqvae_{tag}.pt")
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"[ckpt] saved {path} (step={global_step})", flush=True)


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"=== RQ-VAE Musical_Instruments DDP (world_size={world_size}) ===", flush=True)
        print(f"emb: {EMB_NPY}", flush=True)
        print(f"out: {OUT_DIR}", flush=True)
        print(f"per-GPU batch={BATCH_SIZE} | total={BATCH_SIZE*world_size} | workers={NUM_WORKERS} | compile={COMPILE}", flush=True)
        print(f"hidden={HIDDEN_DIMS} embed={EMBED_DIM} codebook={CODEBOOK_SIZE} layers={N_LAYERS}", flush=True)
        print(f"target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}", flush=True)

    ds = EmbeddingDataset(EMB_NPY)
    n_items = len(ds)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=True)
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
        print(f"[data] {n_items} items | {len(train_loader)} batches/rank | {len(train_loader)*world_size} batches/epoch", flush=True)

    model = RqVae(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=N_LAYERS,
        n_cat_features=0,
        commitment_weight=COMMITMENT_WEIGHT,
        gate_M2_intrinsic=USE_M2_INTRINSIC,  # C28: 开关由 USE_M2_INTRINSIC 控制
        gate_M3_transport=USE_M3_TRANSPORT,  # C28: 启用 M3 transport
        hyperbolic_distance=True,  # v262: Poincaré 距离 argmin + M2 Möbius intrinsic commit path
        sk_eps=0.05,                # v262: Sinkhorn-Knopp 均衡温度 (防 L0 collapse)
        distance_mode=DISTANCE_MODE,  # v262 L2
        rbf_bandwidth=RBF_BANDWIDTH,  # 备用
        mahalanobis_init_var=MAHALANOBIS_INIT_VAR,  # 备用
        prefix_router_layers=None, # Issue #154: L1/L2 per-item 曲率 (默认)
        margin_reg_weight=0.0 if USE_FIXED_CURVATURE else MARGIN_REG_WEIGHT,  # C26 HG-Rec: 关 C5
        margin_target=MARGIN_TARGET,
        spread_loss_weight=SPREAD_LOSS_WEIGHT if USE_SPREAD_LOSS else 0.0,  # F3 v83
        spread_loss_margin=SPREAD_LOSS_MARGIN,  # F3 v83
        use_tcu=False,             # C26 HG-Rec: 关 TCU
        tcu_alpha=TCU_ALPHA,
        tcu_eta=TCU_ETA,
        use_mcdq=False,            # C26 HG-Rec: 关 MCDQ
        mcdq_alpha_init=MCDQ_ALPHA_INIT,
        use_scs=False,             # C26 HG-Rec: 关 SCS
        scs_eps_scale=SCS_EPS_SCALE,
        use_fixed_curvature=USE_FIXED_CURVATURE,  # C26 HG-Rec: 固定 c=1 (C27 优先覆盖)
        c_fixed=C_FIXED,
        use_curriculum_curvature=USE_CURRICULUM_CURVATURE,  # C27: curriculum schedule (优先于 use_fixed)
        c_start=C_START,
        c_end=C_END,
        curriculum_steps=CURRICULUM_STEPS,
        use_cyclic_curvature=USE_CYCLIC_CURVATURE,  # v270: cyclic curriculum (C27b)
        c_cyclic_min=C_CYCLIC_MIN,
        c_cyclic_max=C_CYCLIC_MAX,
        c_cyclic_period=C_CYCLIC_PERIOD,
        stochastic_c_sigma=STOCHASTIC_C_SIGMA if USE_STOCHASTIC_C else 0.0,  # v287: stochastic c perturbation
        use_geodesic_midpoint_commit=USE_GEODESIC_MIDPOINT_COMMIT,  # v282: geodesic midpoint commit (vs Möbius_sub)
        midpoint_layer_mask=[True, False, False],  # v282 R1: 仅 L0 用 midpoint, L1/L2 baseline subtraction (避免 collapse)
    ).to(device)

    if COMPILE:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"[model] params={n_params} | DDP wrapped", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    global_step = 0   # 全球 step (= 单卡 step × world_size, 用 all_reduce SUM 同步)
    t_start = time.time()
    last_log_t = t_start

    if rank == 0:
        print(f"[train] start, target={MAX_GLOBAL_STEPS} global steps", flush=True)
    model.train()
    done = False
    while not done:
        sampler.set_epoch(global_step)
        for batch in train_loader:
            x = batch.to(device, non_blocking=True)
            bsz = x.shape[0]
            seq_batch = SeqBatch(
                user_ids=torch.zeros(bsz, dtype=torch.long, device=device),
                ids=torch.arange(bsz, dtype=torch.long, device=device),
                ids_fut=torch.zeros(bsz, dtype=torch.long, device=device),
                x=x,
                x_fut=x,
                seq_mask=torch.ones(bsz, dtype=torch.long, device=device),
            )
            optimizer.zero_grad()
            t_step = time.time()
            # v287 NOVEL: per-batch stochastic c perturbation (R36n n)
            # 注意: DDP wrap 后 model 是 DistributedDataParallel, 需通过 .module 访问原始 RqVae
            # 注意: c_noise 用 CPU random 生成 (避免 GPU random kernel NCCL deadlock)
            inner_model = model.module if hasattr(model, "module") else model
            if USE_STOCHASTIC_C:
                # 用 CPU random 生成 deterministic noise (受 R51+ manual_seed 控制)
                import random as _random_local
                c_noise = _random_local.gauss(0.0, STOCHASTIC_C_SIGMA)
                for layer in inner_model.layers:
                    layer._current_c_noise = c_noise
            else:
                for layer in inner_model.layers:
                    layer._current_c_noise = 0.0
            out = model(seq_batch, gumbel_t=0.2)
            loss = out.loss
            loss.backward()
            # HG-Rec fix: gradient clipping at norm 1.0 (实测稳定 +∞ norm 梯度, 防 NaN)
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            global_step += 1


            # 同步全球 step 计数 (rank 0 累加其他 rank 的 +1)
            step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            # C27: 更新 curriculum step (各 rank 各自用同步后的 global_step_sync)
            # 这样所有 rank 看到的 c 完全一致 (no broadcast needed)
            model.module.set_curriculum_step(global_step_sync)

            now = time.time()
            if rank == 0 and (now - last_log_t >= 5.0):
                elapsed = now - t_start
                # 全球 it/s = 全球 step / 时间
                ips_global = global_step_sync / elapsed
                ips_per_gpu = (global_step_sync / world_size) / elapsed
                eta_s = (MAX_GLOBAL_STEPS - global_step_sync) / ips_global if ips_global > 0 else float("inf")
                # codebook 健康检查 (rank 0): 各层 unique code 数 + 低使用率告警
                usage = [int(u) for u in out.per_layer_usage.tolist()]
                usage_str = "/".join(str(u) for u in usage)
                low_usage = [
                    li for li, u in enumerate(usage) if u < CODEBOOK_SIZE * CODEBOOK_COLLAPSE_THRESHOLD
                ]
                # θ 曲率学习检查 (M2/M3): 各层 c_l = C_MIN + (C_MAX-C_MIN)*sigmoid(θ_l) (初始 1.25)
                curvs = [float(l.get_c().item()) for l in model.module.layers]
                curv_str = "/".join(f"{c:.3f}" for c in curvs)
                curv_moved = [abs(c - CURV_INIT_C) > CURV_LEARNED_TOL for c in curvs]
                # C5: 各层 mean margin 监控 (目标 MARGIN_TARGET)
                margs = out.per_layer_margin.tolist()
                marg_str = "/".join(f"{m:.4f}" for m in margs)
                # C23: MCDQ 各层 mixing weight α_l 监控
                if USE_MCDQ:
                    alphas = [float(l.get_alpha().item()) for l in model.module.layers]
                    alpha_str = "/".join(f"{a:.3f}" for a in alphas)
                else:
                    alpha_str = "off"
                # C24: SCS 有效 Sinkhorn epsilon per-layer
                if USE_SCS:
                    c_vals = [float(l.get_c().item()) for l in model.module.layers]
                    scs_eps = [f"{0.05 / (c ** SCS_EPS_SCALE):.3f}" for c in c_vals]
                    scs_str = "/".join(scs_eps)
                else:
                    scs_str = "off"
                print(
                    f"  ep {global_step_sync // 10:5d}/{NUM_EPOCHS} | step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| ml={float(out.margin_loss.item() if hasattr(out.margin_loss, 'item') else out.margin_loss):.4f} "
                    f"| codes={usage_str}/{CODEBOOK_SIZE} "
                    f"| c={curv_str} "
                    f"| marg={marg_str} "
                    f"| α={alpha_str} "
                    f"| eps={scs_str} "
                    f"| ips_g={ips_global:.1f} ips/rk={ips_per_gpu:.1f} "
                    f"| elapsed={elapsed:.1f}s | eta={eta_s:.1f}s",
                    flush=True,
                )
                if low_usage:
                    print(
                        f"  [CODEBOOK WARNING] layer {low_usage} usage={usage_str}/{CODEBOOK_SIZE} "
                        f"(< {CODEBOOK_COLLAPSE_THRESHOLD:.0%}), 疑似 collapse, 建议终止检查",
                        flush=True,
                    )
                # θ 学习停滞检查: 训练过半仍无任何层曲率偏离初始值 → 曲率机制疑似失效
                if global_step_sync > MAX_GLOBAL_STEPS // 2 and not any(curv_moved):
                    print(
                        f"  [CURVATURE WARNING] θ 未学习: c={curv_str} 全部 ≈ 初始 {CURV_INIT_C} "
                        f"(梯度可能被 detach/截断, 检查 rqvae.py M2/M3 路径)",
                        flush=True,
                    )
                last_log_t = now

            # ckpt 保存 (rank 0 only, 每 CKPT_EVERY 步一次)
            if global_step_sync >= MAX_GLOBAL_STEPS or (
                global_step_sync > 0 and global_step_sync % CKPT_EVERY == 0
            ):
                tag = f"step{global_step_sync}"
                if global_step_sync >= MAX_GLOBAL_STEPS:
                    tag = "final"
                save_ckpt(model, optimizer, global_step_sync, OUT_DIR, tag)
                dist.barrier()  # 等所有 rank 到齐再继续

            if global_step_sync >= MAX_GLOBAL_STEPS:
                done = True
                break

    if rank == 0:
        print(f"[train] done at global_step={global_step_sync}, total time={(time.time()-t_start):.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
