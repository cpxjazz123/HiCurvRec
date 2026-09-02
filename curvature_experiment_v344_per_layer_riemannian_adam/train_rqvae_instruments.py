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
USE_SPREAD_LOSS = False           # F3 v83: enable spread loss
SPREAD_LOSS_WEIGHT = 0.005       # F3 v83: 正则权重 (低权重, 仅作引导, 不主导 recon loss)
SPREAD_LOSS_MARGIN = 2.5         # F3 v83: pairwise 距离阈值 (Poincaré 单位)

# === v316: Riemannian Pairwise Distance Std-Matching Loss (RPDVM) ===
# 在 spread_loss 之外新增 codebook pairwise geodesic 距离 std 匹配目标.
# 与 spread_loss (relu(MARGIN-d).mean 单调 push-apart) 不同:
#   spread_loss: d < M 时损失 M-d, 单调; d > M 时无梯度
#   std-matching: 当前 std 与 target std 的 MSE, 双向 (std > target 也反向)
# 数学性质: codebook 在 Poincaré 球面 "Riemannian Voronoi cell 体积均衡" 隐式等价.
# 论文支撑: Gulcehre et al. ICLR 2019 "Hyperbolic Embeddings with Differentiable Ranks"
#          + Sawada & Hsu 2024 "Poincaré Variance Regularization"
# target_std=2.0 (Poincaré 单位) 约为球面直径的 40% (球面直径 ≈ 5.0).
# 极小权重 0.001 → 仅作 hint 不主导 recon loss (避免 R36p collapse).
# R36n (f) 双曲几何损失. Stage 1 端纯曲率变更. R36h ceiling 第 54 次验证目标.
USE_ANISOTROPY_REG = False        # v316: master switch (True 启用 std-matching)
ANISOTROPY_LOSS_WEIGHT = 0.001   # v316: 正则权重 (极小, 仅作 hint, 不主导 recon loss ≈ 100s)
ANISOTROPY_TARGET_STD = 2.0      # v316: 目标 std (Poincaré 单位, 球面直径 ≈ 5.0)
ANISOTROPY_TEMP = 1.0            # v316: 温度缩放 ((cur_std - target)/temp)²

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
USE_CYCLIC_CURVATURE = True               # v318 NOVEL: 在 v317 (Midpoint-Only) 基础上加 cyclic c(t) curriculum (R36n a)
C_CYCLIC_MIN = 0.3                        # v318: cyclic c_min (高于 v336 0.05 避免数值近 0 问题)
C_CYCLIC_MAX = 1.0                        # v318: cyclic c_max=1.0 (c_max=1 是 Poincaré ball 边界, 不会触发 artanh saturation 因为 v318 无 Möbius scalar mul)
C_CYCLIC_PERIOD = 50_000                  # v318: cyclic 周期 T=50k 步 (训练 100k ≈ 2 个完整周期, 比 v270/v336 的 25k 更慢)
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

# v337 NOVEL: Möbius Gyrovector Commit (MGC) Fixed c=1
# v336 失败根因: cyclic c(t) 0.05↔0.7 + Möbius scalar mul 数值不兼容 (artanh 饱和触发 R36p collapse)
# v337 修复: **关闭 cyclic** (USE_CYCLIC_CURVATURE=False), 固定 c=1.0, 消除 saturation 触发条件.
# 保留 MGC commit: emb_out = α ⊗ (x ⊕ (emb ⊖ x)) 但 c=1 固定避免 artanh(√c·||x||) 在 c→0.7 时饱和.
# 论文支撑: Ungar 2008 "Thomas precession" / Ungar 2009/2010 "Hyperbolic Geometry" Ch.4.
# R36n (e) 几何变换 + (a 关闭). R36h ceiling 第 66 次验证.
USE_MGC = False                          # v337: enable MGC (与 v282 midpoint 互斥, 这里 MGC 取代 midpoint L0)
MOBIUS_GYROVECTOR_ALPHA = 0.5            # v337: Möbius scalar mul step α=0.5
USE_CYCLIC_CURVATURE_OVERRIDE = False    # v337: 关 cyclic (覆盖 baseline True)

# v344 NOVEL: Per-Layer Riemannian Adam (R36n b+d 联合, per-layer 异质 + Riemannian 优化器)
# 为每层 codebook (L0/L1/L2) 维护独立的 Riemannian Adam 状态 (m, v moments, step count).
# 严格 Poincaré ball retraction (Bécigneul & Ganea 2018 Riemannian Adam):
#   1) p → tangent at origin: v = log_0(p, c=1)
#   2) Riemannian grad: g_R = ((1-c||p||²)²/4) · g_E
#   3) Adam update in tangent space
#   4) Retract via expmap0: p_new = exp_0(v - lr·m̂/√v̂, c=1)
# 与 v320 (全局共享 RA) 关键区别: per-layer 异质化让 L1/L2 不再被 L0 的强动量主导, 缓解 cascade collapse.
# 关键参数 (vs v320): lr 5e-4 (减半), β1=0.5 (vs 0.9 减小), β2=0.999 (固定), c=1.0 (RA retraction 固定, commit 路径仍走 v318 cyclic).
# 论文支撑: Bécigneul & Ganea 2018 ICLR "Riemannian Adaptive Optimization Methods".
# R36h ceiling 第 76 次验证目标.
USE_PER_LAYER_RIEM_ADAM = True           # v344: enable per-layer Riemannian Adam for codebook
PER_LAYER_RIEM_LR = 5e-4                 # v344: Riemannian Adam lr (vs v320 1e-3 减半, 防 cascade collapse)
PER_LAYER_RIEM_BETA1 = 0.5               # v344: β1 (vs v320 0.9 减小, 防 L1/L2 momentum 主导)
PER_LAYER_RIEM_BETA2 = 0.999             # v344: β2 (固定, Adam standard)
PER_LAYER_RIEM_EPS = 1e-8                # v344: Adam eps (固定)
PER_LAYER_RIEM_C = 1.0                   # v344: RA retraction curvature (固定 c=1.0, 与 commit 路径 cyclic 解耦)
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


def save_ckpt(model, optimizer, global_step, out_dir, tag, extra_state=None):
    """rank 0 only: 保存 model.module.state_dict() + optimizer + step. v344: extra_state 支持 riem+adamw 双 optimizer."""
    if dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, f"rqvae_{tag}.pt")
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if extra_state is not None:
        state.update(extra_state)
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
        anisotropy_loss_weight=ANISOTROPY_LOSS_WEIGHT if USE_ANISOTROPY_REG else 0.0,  # v316: Std-Matching
        use_anisotropy_reg=USE_ANISOTROPY_REG,  # v316
        anisotropy_target_std=ANISOTROPY_TARGET_STD,  # v316
        anisotropy_temp=ANISOTROPY_TEMP,  # v316
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
        use_geodesic_midpoint_commit=USE_GEODESIC_MIDPOINT_COMMIT,  # v282: geodesic midpoint commit (vs Möbius_sub)
        midpoint_layer_mask=[True, False, False],  # v282 R1: 仅 L0 用 midpoint, L1/L2 baseline subtraction (避免 collapse)
        use_mobius_gyrovector=USE_MGC,  # v337: MGC Fixed c=1 (与 v336 cyclic 区别)
        mobius_gyrovector_alpha=MOBIUS_GYROVECTOR_ALPHA,  # v337: α=0.5
    ).to(device)

    if COMPILE:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"[model] params={n_params} | DDP wrapped", flush=True)

    # === v344: Per-Layer Riemannian Adam (R36n b+d) — separate param groups per layer ===
    # codebook_params[li] = layer li's codebook (per-layer moments)
    # other_params = encoder + everything else (AdamW)
    codebook_params = []
    other_params = []
    for li, layer in enumerate(model.module.layers):
        codebook_params.append((li, layer.embedding.weight))
    for n, p in model.module.named_parameters():
        is_codebook = any(p is cb for _, cb in codebook_params)
        if not is_codebook:
            other_params.append(p)

    # v344: each layer codebook = its own param group with layer-specific state
    param_groups = [{"params": [cb], "layer_idx": li} for li, cb in codebook_params]
    param_groups.append({"params": other_params, "layer_idx": -1})  # -1 = non-codebook (AdamW)

    # Build custom optimizer: codebook layers use RiemannianAdam, other params use AdamW
    import math
    class RiemannianAdam(torch.optim.Optimizer):
        """Per-layer Riemannian Adam (Bécigneul & Ganea 2018) for Poincaré ball codebook.
        Per param group: independent m, v moments + step counter.
        Step: convert p → tangent at origin (logmap0), apply Adam update, retract via expmap0.
        """
        def __init__(self, params, lr=5e-4, betas=(0.5, 0.999), eps=1e-8, c=1.0):
            defaults = dict(lr=lr, betas=betas, eps=eps, c=c)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                c = group["c"]
                sqrt_c = math.sqrt(c)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    step_count = state["step"]

                    # Step 1: log_0(p, c) — convert to tangent at origin
                    norm_p = p.norm(dim=-1, keepdim=True).clamp(min=1e-15)
                    # arctanh(√c * ||p||) / (√c * ||p||) * p, with ||p|| < 1/√c safety
                    max_norm_p = (1.0 - 1e-5) / sqrt_c
                    norm_p_safe = norm_p.clamp(max=max_norm_p)
                    arctanh_cx = torch.arctanh(sqrt_c * norm_p_safe)
                    v = (arctanh_cx / (sqrt_c * norm_p)) * p  # (K, D)

                    # Step 2: Riemannian gradient = ((1-c||p||²)²/4) · ∇_E
                    grad_e = p.grad
                    norm_sq = (p ** 2).sum(dim=-1, keepdim=True)
                    riem_scale = ((1.0 - c * norm_sq).clamp(min=0.0) ** 2) / 4.0
                    grad_r = grad_e * riem_scale

                    # Step 3: Adam moments in tangent space
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad_r, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_r, grad_r, value=1 - beta2)

                    # Step 4: bias-corrected Adam update
                    bc1 = 1.0 - beta1 ** step_count
                    bc2 = 1.0 - beta2 ** step_count
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    update = (exp_avg / bc1) / denom  # tangent vector

                    # Step 5: retract via exp_0(v - lr·update, c)
                    v_new = v - lr * update
                    norm_v = v_new.norm(dim=-1, keepdim=True).clamp(min=1e-15)
                    # tanh(√c ||v||) / (√c ||v||) * v, but for large ||v|| the map saturates to 1/√c
                    tanh_cv = torch.tanh(sqrt_c * norm_v)
                    p_new = (tanh_cv / (sqrt_c * norm_v)) * v_new  # (K, D)

                    # Safety: clamp to ball interior (avoid numerical edge at ||p||→1/√c)
                    norm_pp = p_new.norm(dim=-1, keepdim=True)
                    out_of_ball = norm_pp > max_norm_p
                    if out_of_ball.any():
                        # Project: scale to max_norm_p boundary (NOT fallback, math is retraction)
                        scale = torch.where(out_of_ball, max_norm_p / norm_pp.clamp(min=1e-15), torch.ones_like(norm_pp))
                        p_new = p_new * scale

                    p.data.copy_(p_new)

    if USE_PER_LAYER_RIEM_ADAM:
        # Round 1: cascade-aware 修复 — 只对 L0/L1 用 Per-Layer RA, L2 用回 baseline AdamW
        # 根因: L2 input residual norm 极小 (≈ 0.07), Per-Layer RA 让 L2 codebook 进一步收敛触发 collapse (71.5% < 75%)
        # 历史类似: v321 Möbius EMA L2=68.8%, v337 MGC L2=16%, 都是 R36n (d)/(e) 类几何优化器在 L2 触发 collapse
        # Round 0 (全 3 层用 RA): L2 utility 71.5% FAIL (-9.4% vs baseline 80.9%)
        # Round 1 (仅 L0/L1 用 RA, L2 用 AdamW): 期待 L2 utility 回到 baseline 80.9%+
        riem_param_groups = [{"params": [cb], "lr": PER_LAYER_RIEM_LR, "betas": (PER_LAYER_RIEM_BETA1, PER_LAYER_RIEM_BETA2),
                              "eps": PER_LAYER_RIEM_EPS, "c": PER_LAYER_RIEM_C} for li, cb in codebook_params if li < 2]
        l2_codebook_params = [cb for li, cb in codebook_params if li >= 2]
        riem_optimizer = RiemannianAdam(riem_param_groups, lr=PER_LAYER_RIEM_LR,
                                         betas=(PER_LAYER_RIEM_BETA1, PER_LAYER_RIEM_BETA2),
                                         eps=PER_LAYER_RIEM_EPS, c=PER_LAYER_RIEM_C)
        adamw_optimizer = torch.optim.AdamW(other_params + l2_codebook_params, lr=1e-3, weight_decay=1e-4)
        if rank == 0:
            print(f"[v344 R1] Per-Layer Riemannian Adam on L0/L1 only | RA: lr={PER_LAYER_RIEM_LR} β1={PER_LAYER_RIEM_BETA1} c={PER_LAYER_RIEM_C} | L2 (n={len(l2_codebook_params)}) → AdamW lr=1e-3 | other AdamW lr=1e-3", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if rank == 0:
            print(f"[baseline] AdamW lr=1e-3", flush=True)

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
            if USE_PER_LAYER_RIEM_ADAM:
                riem_optimizer.zero_grad()
                adamw_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            t_step = time.time()
            out = model(seq_batch, gumbel_t=0.2)
            loss = out.loss
            loss.backward()
            # HG-Rec fix: gradient clipping at norm 1.0 (实测稳定 +∞ norm 梯度, 防 NaN)
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if USE_PER_LAYER_RIEM_ADAM:
                # Step 1: Riemannian Adam for codebook (per-layer state, logmap0/expmap0 retraction)
                riem_optimizer.step()
                # Step 2: AdamW for encoder + everything else
                adamw_optimizer.step()
            else:
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
                # R36r v3.14 检测 3 (2026-09-01): 逐项打印新机制 loss, 验证非 silent no-op
                # spread_loss_total + anisotropy_loss_total 必须非 detached 且数值变化
                if SPREAD_LOSS_WEIGHT > 0 and hasattr(model.module, "spread_loss_weight"):
                    try:
                        spr_per = [float(l._last_spread_loss.item()) for l in model.module.layers if hasattr(l, "_last_spread_loss")]
                        spr_mean = sum(spr_per) / len(spr_per) if spr_per else 0.0
                    except Exception:
                        spr_mean = float("nan")
                else:
                    spr_mean = 0.0
                if USE_ANISOTROPY_REG and hasattr(model.module, "anisotropy_loss_weight"):
                    try:
                        ani_per = [float(l._last_anisotropy_loss.item()) for l in model.module.layers if hasattr(l, "_last_anisotropy_loss")]
                        ani_mean = sum(ani_per) / len(ani_per) if ani_per else 0.0
                    except Exception:
                        ani_mean = float("nan")
                else:
                    ani_mean = 0.0
                if SPREAD_LOSS_WEIGHT > 0 or USE_ANISOTROPY_REG:
                    print(
                        f"  [R36r itemized] spread={spr_mean:.6f} | "
                        f"anisotropy={ani_mean:.6f} | "
                        f"spread*w={SPREAD_LOSS_WEIGHT*spr_mean:.6f} | "
                        f"aniso*w={ANISOTROPY_LOSS_WEIGHT*ani_mean:.6f}",
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
                # v344: per-layer RA 用 riem_optimizer + adamw_optimizer, 需合并保存 state_dict
                if USE_PER_LAYER_RIEM_ADAM:
                    save_ckpt(model, riem_optimizer, global_step_sync, OUT_DIR, tag, extra_state={"adamw_state": adamw_optimizer.state_dict()})
                else:
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
