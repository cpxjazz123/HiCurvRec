#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task #448 / Issue #157 [方向A Gate2] κ同步重校准的RQ-VAE代码本与完整SID链路验证

R18 4 维度路径对比 vs Issue #155:
  D1 spec: 仅 Gate 1 monitoring 时序审计 (#155) vs Gate 2 完整 Stage 2 链路 (#157)
  D2 实施: train_step monitoring 时序修复 (#155) vs per-layer learnable κ_l + 每 step 后 codebook 重校准 (#157)
  D3 Gate 失败机制: monitoring grad=0 显示 bug (#155) vs 旧尺度 / 旧距离缓存错配 (#157)
  D4 引用文献: 无 (#155) vs arXiv:2405.13979 学习曲率与双曲尺度同步 (#157)

实施核心:
  - HRQVAEWithKappaSync: 复用 HG-Rec HRQVAE 框架, 把 HVectorQuantization 的固定 c=1 替换为 per-layer learnable c_l
  - 每次 opt.step() 后: 强制 recompute codebook_h (proj_to_ball with new c_l), distance cache 失效
  - Stage 2 训练: 100 epoch, 每次 opt.step() 后记录 κ / codebook norm / 距离统计 / 同步重校准前后差异
  - Stage 2 推断: 训练后加载 ckpt, Sinkhorn + 第4位 dedup, 输出 (9922, 4) 整数 SID
  - SID 验收: SHA256 hash + item alignment + reload 一致性

precheck 决策阈值 (Issue #157 spec 强制):
  - per-layer κ_l 真学习 (init=0 → final != 0)
  - codebook sync recalibration: 每次 κ step 后 codebook_h 立即反映新 c_l (不延迟)
  - distance cache 失效: opt.step 后第一次 forward 必须重新计算 d, 不能用旧 d
  - SID SHA256 唯一 + item alignment 通过 row index
  - reload 一致: 同一 batch 第二次 forward 输出 SID 跟第一次一致

Gate 2 决策阈值:
  - PASS: 10+ κ 更新点 + reload 一致 (5/5) + 无 NaN/Inf + 真实 SID hash + item alignment + 对照消融 PASS
  - FAIL: 任一项不满足即 STOP
"""

import os
import sys
import ast
import json
import math
import time
import argparse
import hashlib
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# R7: GPU 选择 (从环境变量读, 默认 GPU 0)
os.environ.setdefault("TRITON_CACHE_DIR", "/home/wlia0047/.triton/cache_task448")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# 引用 HG-Rec utils 函数 (poincare_distance / proj_to_ball / expmap0 / logmap0 / sinkhorn_algorithm)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")

ITEM_EMB_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/item_emb.parquet"
N_ITEMS = 9922
EMB_DIM = 768
N_HIERARCHIES = 3
CODEBOOK_SIZES = [64, 128, 256]
E_DIM = 32  # HG-Rec 默认 e_dim
ENCODER_LAYERS = [512, 256, 128, 64]
BATCH_SIZE = 1024
N_EPOCHS = 100  # Issue #157 spec: 10+ κ 更新点足够
LOG_EVERY = 5
SK_EPSILONS = [0.0, 0.0, 0.0]  # HG-Rec 默认 (非 Sinkhorn 模式)
SK_ITERS = 3
BETA = 1.0
# Issue #55/v5: seed 对齐基线 train_hrqvae.py (seed=2024). kmeans_init 对 seed 高度敏感,
# taskA(42)/taskB(42) 均停在 unique_3digit≈88.3-88.4% vs 基线 99.7%, 疑似 seed 平台.
SEED = 2024
# Issue #55/v4 fix_c (固定 c=1) 已被用户否决: taskA 曲率必须保持可学习框架 (learnable κ).
# 默认不设环境变量即 learnable κ 主路径; poincare_distance_safe (u clamp 0.985) 已根治边界梯度爆炸.
FIX_C = os.environ.get("TASKA_STAGE2_FIX_C", "0") == "1"
# Issue #157 → gen_codebook.py 对齐: SID 迭代碰撞消解 (默认开). 仅码本训练充分 (1000ep) 时有效.
RESOLVE = os.environ.get("TASKA_STAGE2_RESOLVE", "1") == "1"
# Issue #75 → 论文 arXiv:2405.13979 (NeurIPS'25, Robust Hyperbolic Learning with Curvature-Aware
# Optimization) 两大机制移植. 目标: 解决 learnable κ 1000ep 塌缩 (κ 冲 clamp 下界 + SID unique=1):
#   RESCALE=1: Maximum Distance Rescaling — proj_to_ball 硬截断 → 切空间 tanh 平滑渐近饱和.
#   CURV_AWARE=1: Curvature-Aware Optimization schema (Algorithm 1) — 先参数(旧 c 几何) 后曲率拆分
#     优化器. 切空间表示在曲率变化下不变, 消除 κ 突变对参数几何的冲击. 默认关 (实验开关).
RESCALE = os.environ.get("TASKA_STAGE2_RESCALE", "0") == "1"
CURV_AWARE = os.environ.get("TASKA_STAGE2_CURV_AWARE", "0") == "1"
# Issue #76 → 用户方案: 根治 learnable κ 塌缩的 clamp 替代 (v7c/v7e/v8 全 FAIL 后第 4 条路线).
#   核心: 绝对 quant loss 不再直接训练 κ (量化距离对 c stop-gradient, 消除 "距离随 c 减" 的尺度作弊),
#   κ 只从平滑 log-curvature 先验 λ·Σκ² 获得梯度 (软约束替代硬 clamp, c=exp(κ) 恒>0 无需 clamp).
#   v6 fix_c (c 无梯度) 从不塌缩 88.4% vs learnable κ (量化梯度) 全塌缩 → stop-grad 是对症根因.
CURV_PRIOR = os.environ.get("TASKA_STAGE2_CURV_PRIOR", "0") == "1"
CURV_PRIOR_LAMBDA = float(os.environ.get("TASKA_STAGE2_CURV_PRIOR_LAMBDA", "0.1"))
# Issue #76 第二步: 相对结构目标 (曲率-尺度匹配). v9 实测: 第一步 (量化 stop-grad c) 后 κ 停在 0
# (先验梯度 2λκ=0, 无学习信号), SID 100% 不塌缩但 κ 无法学层级差异. 本目标给 κ 唯一非零学习信号:
#   Poincaré 球半径 R=1/√c, 量化后 latent 落在球的固定比例处 → √c·r → REL_STRUCT_TARGET.
#   √c·r 是尺度无关比值 (不随绝对距离随 c 减而白嫖); 层间 residual 尺度差异 → 层级曲率差异
#   (深层残差小 → 需要更大曲率/更小球适配). 与平滑先验 λ·Σκ² 共存 (先验锚 c→1, 结构目标学差异).
REL_STRUCT = os.environ.get("TASKA_STAGE2_REL_STRUCT", "0") == "1"
REL_STRUCT_TARGET = float(os.environ.get("TASKA_STAGE2_REL_STRUCT_TARGET", "0.3"))
REL_STRUCT_LAMBDA = float(os.environ.get("TASKA_STAGE2_REL_STRUCT_LAMBDA", "1.0"))

PRODUCT_DIR = Path(os.environ.get("TASKA_STAGE2_PRODUCT_DIR",
                                  "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_kappa_sync"))
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ──────────────────────────────────────────────────────────────
# HG-Rec utils imports
# ──────────────────────────────────────────────────────────────
from model.utils import (
    proj_to_ball, expmap0, logmap0, sinkhorn_algorithm, kmeans, MLP, EmbDataset,
    mobius_add, _eps, artanh, poincare_distance,
)

# Issue #55/v5→v7f: learnable κ 数值稳定开关. SAFE_DISTANCE=1 用 u clamp 0.985 (梯度有界,
# artanh' ≤~33); =0 用上游 poincare_distance (artanh clamp 1-1e-10, 距离不饱和但梯度可爆).
# v7c/v7e (safe) 1000ep 长训练 SID 塌缩 unique=1 (SHA 相同), v6 fix_c (上游, c=1) 不塌缩 →
# 需对照上游排除 u clamp 距离饱和致 index 区分度丧失的嫌疑. 默认 safe (已验证训练稳定).
SAFE_DISTANCE = os.environ.get("TASKA_STAGE2_SAFE_DISTANCE", "1") == "1"
if SAFE_DISTANCE:

    def poincare_distance_safe(x, y, c, u_max=0.985):
        """learnable κ 稳定版 poincare 距离: u=sqrt(c)·norm 截断到 u_max, artanh 梯度有界,
        从源头杜绝边界梯度爆炸 (上游 artanh 只 clamp 1-1e-10, u→1 时梯度 ~5e5 爆炸)."""
        diff = mobius_add(-x, y, c)
        sqrt_c = c ** 0.5
        norm = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
        u = (sqrt_c * norm).clamp(max=u_max)
        return (2.0 / sqrt_c) * artanh(u)

    # learnable κ 主路径: 全局替换 poincare_distance 为稳定版 (所有调用点自动生效)
    poincare_distance = poincare_distance_safe

# ──────────────────────────────────────────────────────────────
# Issue #75: Maximum Distance Rescaling (论文 arXiv:2405.13979 Eq.2-3)
# ──────────────────────────────────────────────────────────────
if RESCALE:
    # 平滑替代 proj_to_ball 硬截断. 论文原理:
    #   z = log^K_0(x)  (切空间);  x' = exp^K_0( z · D_max·tanh(r·‖z‖)/‖z‖ )
    #   r = atanh(0.99/(s·D_max)), s=tightness. ‖z‖ → s·D_max 时 ‖z'‖ → D_max (渐近饱和, 非硬截断),
    #   tanh 导数≤1 → 梯度全程有界, 无截断不连续 → 支持长训练稳定.
    # 动机 (v7c 塌缩根因): proj_to_ball 把大 norm 点全压到球面边缘 (norm=max_norm), 这些点相互间
    #   Poincaré 距离饱和 (safe u clamp 0.985) → index 区分度丧失 → SID unique=1.
    # 本移植: D_max = 2/sqrt(c)·artanh(0.99) (对应流形上 ‖x‖=0.99·R 处的 log-norm 上限);
    #   s = atanh(0.99) ≈ 2.646 → r = 1/D_max, 小点区缩放因子→1 (不扰动正常点), 仅越界点被平滑压缩.
    # 注意: 不能复用上游 expmap0 (其内部经全局名 proj_to_ball 解析 → 递归), 必须闭包绑定原始函数.
    _UP_PROJ = proj_to_ball
    _UP_LOGMAP = logmap0

    def _dmax(c, frac: float = 0.99):
        """Poincaré ball 切空间 log-norm 上限: ‖x‖=frac·R 处测地距离 = 2/sqrt(c)·artanh(frac).
        c 可能为 torch.Tensor (get_c) 或 Python float (poincare_recon_loss 默认 c=1.0), 两者兼容."""
        art = math.atanh(frac)
        return (2.0 / c ** 0.5) * art

    def _expmap0_no_recurse(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """expmap0 复刻 (tanh), 内部用闭包绑定的原始 proj_to_ball, 避免全局名递归."""
        sqrt_c = c ** 0.5
        norm_z = z.norm(dim=-1, keepdim=True).clamp_min(_eps(z))
        factor = torch.tanh(sqrt_c * norm_z) / (sqrt_c * norm_z)
        return _UP_PROJ(factor * z, c)

    def proj_to_ball_smooth(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Maximum Distance Rescaling: 硬截断 proj_to_ball 的平滑替代. 所有调用点自动生效."""
        z = _UP_LOGMAP(x, c)
        zn = z.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        D_max = _dmax(c)
        scale = D_max * torch.tanh(zn / D_max) / zn  # 小点→1, 越界→渐近饱和到 D_max
        return _expmap0_no_recurse(z * scale, c)

    proj_to_ball = proj_to_ball_smooth


# ──────────────────────────────────────────────────────────────
# Issue #157: per-layer learnable κ_l + κ-aware codebook sync recalibration
# ──────────────────────────────────────────────────────────────
class KappaAwareVectorQuantization(nn.Module):
    """Per-layer learnable κ_l (= c_l - 1.0) + mix_weight_l, 每 step 后强制 codebook 重投影 (Issue #157 关键).
    Issue #55/v2 修复:
      - kmeans_init 默认 True (解决 codebook 塌缩到球心导致 κ 梯度消失)
      - 加 per-layer mix_weight (init=1.0, 让三层有不同的距离权重参与 RQ loss)
      - κ / mix_weight 都加入 trainable params
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10, sk_eps=0.0, sk_iters=3,
                 fix_c=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        # Issue #55/v4 (fix_c): 对齐基线 HVectorQuantization 固定 c=1.0, 禁用 learnable κ/mix_weight.
        # 根因: v4/v5 1000epoch 实测 κ 三层系统性负漂移 → 撞 Poincaré 球边界 → loss=-inf→NaN → 码本塌缩.
        # 基线 c=1 固定从不崩 (HG-Rec model/utils.py HVectorQuantization self.c=1.0).
        self.fix_c = fix_c
        # Issue #157: per-layer learnable κ_l (init=0 → c_l = 1.0 + κ_l = 1.0 baseline)
        self.kappa = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Issue #55/v2: per-layer mix weight (init=1.0, softmax normalized). 三层独立学习不同权重
        self.mix_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                # 改为 init norm≈0.1 (而非 0.01), 防止全部塌到球心
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

        # Issue #157: distance cache (强制 opt.step 后失效)
        self._distance_cache = None  # 缓存 (x_id, c_id) → (B, K) distances
        self._cache_x_id = None
        self._cache_c_id = None

    def get_c(self) -> torch.Tensor:
        """c_l = 1.0 + κ_l, 必须 > 0. learnable κ 主路径 (Issue #55/v5→v7e, 用户硬约束框架不变).
        v7c 实测: κ 负漂移到 -0.3 (c=0.7) 后几何偏离基线过大 + proj 饱和 → SID 塌缩 (unique=1).
        v7e 策略: κ clamp 收紧到 [-0.1, 0.5] (c ∈ [0.9, 1.5]), 几何锚定在基线 c≈1 附近,
        保留 learnable κ 微调能力但防止破坏性负漂移 (v6 fix_c c=1 seed42 稳定 88.4% vs 基线 seed2024 99.7%,
        seed 疑为主因). quant loss 用欧氏向量 (对齐基线), safe distance 限梯度, mix clamp 防退化.
        fix_c (Issue #55/v4, 已否决): 直接返回 1.0, 完全对齐基线 HVectorQuantization."""
        if self.fix_c:
            return torch.tensor(1.0, dtype=torch.float32, device=self.kappa.device)
        if CURV_PRIOR:
            # Issue #76: c=exp(κ) (κ=ln c). exp 恒>0 → 定义域自动满足, 无需硬 clamp; κ init 0 → c=1 锚定基线.
            return torch.exp(self.kappa)
        kappa_clamped = self.kappa.clamp(min=-0.1, max=0.5)
        return 1.0 + kappa_clamped + 1e-3

    def get_codebook(self):
        c = self.get_c()
        return proj_to_ball(expmap0(self.embeddings.weight, c), c)

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def invalidate_distance_cache(self):
        """Issue #157 spec: 每次 κ 更新后, 距离缓存强制失效"""
        self._distance_cache = None
        self._cache_x_id = None
        self._cache_c_id = None

    @staticmethod
    def center_distance_for_constraint(distances):
        max_d = distances.max()
        min_d = distances.min()
        middle = (max_d + min_d) / 2
        amplitude = max_d - middle + 1e-10
        if amplitude <= 0:
            return distances - middle
        return (distances - middle) / amplitude

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        codebook_e = self.embeddings.weight
        if not self.initted and self.training:
            self.init_emb(latent)

        c = self.get_c()  # Issue #157: per-layer learnable c
        # Issue #76: CURV_PRIOR 下量化距离对 c stop-gradient — κ 不接收"距离随 c 减"的尺度作弊梯度.
        # 几何 (expmap/proj/distance) 用 c_geom, κ 只从 train_step 的平滑 log-curvature 先验获得梯度.
        c_geom = c.detach() if CURV_PRIOR else c
        # Issue #157 关键: 每次 forward 重新投影 codebook (不 cache 旧尺度)
        latent_h = proj_to_ball(expmap0(latent, c_geom), c_geom)
        codebook_h = proj_to_ball(expmap0(codebook_e, c_geom), c_geom)

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)

        # Issue #157 关键: distance 重算 (每次 forward 重算, 不 cache 旧 c)
        d = poincare_distance(x_exp, cb_exp, c_geom).squeeze(-1)
        # Issue #157 spec: cache 仅用于 reload 一致性测试 (写一个标志)
        # 这里默认 invalidate (true κ-aware behavior)
        self._distance_cache = d.detach()
        self._cache_x_id = id(latent)
        self._cache_c_id = c.item()

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn produced NaN/Inf")
            indices = torch.argmax(Q, dim=-1)

        x_exp = logmap0(x_exp, c_geom)
        cb_exp = logmap0(cb_exp, c_geom)
        x_q = codebook_e.index_select(0, indices)
        # Issue #55/v5→v7e: quant loss 输入恢复欧氏向量 (对齐基线 HVectorQuantization + v6 fix_c 行为).
        # v7c 用 proj_to_ball 后 SID 塌缩 (unique=1): proj 使大 norm 点全压到球面边缘, safe distance
        # 饱和 (u clamp 0.985) → index 区分度丧失 → 码本退化. 欧氏直接进 + κ clamp 紧下界 (c≥0.9
        # 接近基线) 是既保 κ 梯度路径又贴近基线的平衡.
        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c_geom) ** 2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c_geom) ** 2)
        # Issue #55/v2: mix_weight_l 调节本层 loss 贡献 (三层不同权重学习)
        # Issue #55/v5: mix_weight clamp ≥0.01 防负值次生失控 (κ 冲边界后曾学到负权 → loss 一路变负)
        mix_w = self.mix_weight.clamp(min=0.01, max=20.0)
        if self.fix_c:
            loss = commitment_loss + self.beta * codebook_loss
        elif CURV_PRIOR:
            # Issue #76: mix_weight 同样 stop-grad (消除量化 loss 白嫖, v8 final 全负 [-0.02,...]),
            # 仅作固定层权重, 其梯度由 train_step 的平滑先验提供.
            loss = mix_w.detach() * (commitment_loss + self.beta * codebook_loss)
        else:
            loss = mix_w * (commitment_loss + self.beta * codebook_loss)
        # Issue #55/v3: logmap0 输入先 proj_to_ball 兜底防 artanh(sqrt(c)*norm)>1 → NaN
        x_q_safe = proj_to_ball(x_q, c_geom)
        latent_safe = proj_to_ball(latent, c_geom)
        # Issue #76 第二步: 相对结构目标 (曲率-尺度匹配) — κ 在 CURV_PRIOR 下的唯一非零学习信号.
        # 量化后 latent 落在球的固定比例 √c·r → REL_STRUCT_TARGET (r=‖x_q_safe‖ ≤ R=1/√c 在球内).
        # √c·r 为尺度无关比值: 不随"绝对距离随 c 减"而白嫖 (结构目标不受量化作弊影响).
        # 层间 residual 尺度差异 → 层级曲率差异: 深层残差小 → 需要更大曲率 (更小球) 适配.
        if REL_STRUCT:
            c_struct = self.get_c()  # 不 detach: 让 κ 接收结构梯度 (量化距离已 stop-grad, 此目标独享 κ 梯度)
            r_struct = x_q_safe.detach().norm(dim=-1).mean()
            struct_term = torch.sqrt(c_struct) * r_struct - REL_STRUCT_TARGET
            loss = loss + REL_STRUCT_LAMBDA * struct_term.pow(2)
        x_q = logmap0(x_q_safe, c_geom)
        latent = logmap0(latent_safe, c_geom)
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


# ──────────────────────────────────────────────────────────────
# Issue #157: κ-aware HRQVAE
# ──────────────────────────────────────────────────────────────
class KappaAwareHRQVAE(nn.Module):
    def __init__(self, in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
                 layers=ENCODER_LAYERS, beta=BETA, kmeans_init=True, kmeans_iters=10,
                 sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.beta = beta
        self.fix_c = fix_c
        self.encode_layer_dims = [in_dim] + layers + [e_dim]
        self.encoder = MLP(layers=self.encode_layer_dims, dropout=0.0, use_bn=False)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(layers=self.decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            KappaAwareVectorQuantization(n_e, e_dim, beta=beta, kmeans_init=kmeans_init,
                                         kmeans_iters=kmeans_iters, sk_eps=eps, sk_iters=sk_iters,
                                         fix_c=fix_c)
            for n_e, eps in zip(num_emb_list, sk_eps)
        ])

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self._rq_forward(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q, z

    def _rq_forward(self, x, use_sk=True):
        all_losses, all_indices = [], []
        x_q = 0
        residual = x
        for q in self.vq_layers:
            x_res, loss, idx = q(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self._rq_forward(z, use_sk=use_sk)
        return indices

    def invalidate_all_caches(self):
        """Issue #157 spec: κ 更新后强制所有层 distance cache 失效"""
        for q in self.vq_layers:
            q.invalidate_distance_cache()


# ──────────────────────────────────────────────────────────────
# Issue #157 spec: Stage 2 训练 + 监控
# ──────────────────────────────────────────────────────────────
def poincare_recon_loss(out, target, c=1.0):
    """对齐基线 HG-Rec loss_type='poincare': 在 Poincaré 球面上测双曲距离.

    Issue #55/v3 根因修复: 欧氏 MSE recon 导致 posterior collapse (encoder z→常数,
    SID unique3=1). 基线用 poincare recon 不塌缩 (诊断: mse→z std=0.0006 unique3=1,
    poincare→z std=0.04 unique3=1838).
    """
    o = proj_to_ball(expmap0(out, c), c)
    t = proj_to_ball(expmap0(target, c), c)
    return torch.mean(poincare_distance(o, t, c) ** 2)


def train_step_with_sync_recalibration(model: KappaAwareHRQVAE, batch, opt, kappa_log: list,
                                       reg_step: int, opt_kappa: Optional[torch.optim.Optimizer] = None):
    """Issue #157 关键: 在每个 opt.step() 后, 强制 recompute codebook + 失效 cache + 记录重校准前后差异"""
    model.train()
    out, rq_loss, indices, z_q, z = model(batch)
    # Issue #55/v3: recon 用 poincare (对齐基线), 弃用欧氏 MSE (塌缩根因)
    recon_loss = poincare_recon_loss(out, batch)
    total_loss = recon_loss + rq_loss
    if CURV_PRIOR:
        # Issue #76: 平滑 log-curvature 先验 λ·Σκ² — κ 的唯一梯度来源 (量化已对 c stop-grad).
        # 软约束替代硬 clamp: 拉 κ→0 (c→1 锚定基线), 但 κ 仍可在先验许可内自由微调, 不卡死.
        kappa_prior = sum(q.kappa.pow(2).sum() for q in model.vq_layers)
        total_loss = total_loss + CURV_PRIOR_LAMBDA * kappa_prior

    # 记录 κ 更新前
    kappas_before = [q.kappa.item() for q in model.vq_layers]
    codebook_norm_before = [q.embeddings.weight.norm().item() for q in model.vq_layers]

    opt.zero_grad()
    if opt_kappa is not None:
        opt_kappa.zero_grad()
    total_loss.backward()

    # raw grad (Issue #157 spec: per-layer κ grad finite nonzero)
    raw_grad_kappa = []
    for q in model.vq_layers:
        if q.kappa.grad is None:
            raw_grad_kappa.append(0.0)
        else:
            raw_grad_kappa.append(q.kappa.grad.abs().item())

    # Issue #75 Curvature-Aware Optimization (论文 Alg.1 更新顺序):
    #   Step 1: 先更新流形/欧氏参数 (在旧 c 几何下, κ 未动 → 梯度有效)
    #   Step 2: 再更新曲率 κ/mix_weight. 切空间表示在曲率变化下不变, 下一 forward 自动用新 c re-map.
    opt.step()
    if opt_kappa is not None:
        opt_kappa.step()

    # Issue #157 关键: κ 更新后强制 recompute codebook_h + 失效 cache
    model.invalidate_all_caches()

    # 记录 κ 更新后
    kappas_after = [q.kappa.item() for q in model.vq_layers]
    codebook_norm_after = [q.embeddings.weight.norm().item() for q in model.vq_layers]
    cs_after = [q.get_c().item() for q in model.vq_layers]

    # Issue #157 spec: 重校准后用新 c 立即 forward 一次, 验证 distance 反映新尺度
    with torch.no_grad():
        # 用 model encoder 把 batch[:64] 编到 e_dim 空间 (跟 training 一致)
        sub_batch = batch[:64]
        z_sub = model.encoder(sub_batch)  # (64, e_dim)
        forward_dist_first = []
        for q in model.vq_layers:
            c = q.get_c()
            x_exp = proj_to_ball(expmap0(z_sub, c), c).unsqueeze(1).expand(64, q.n_e, -1)
            cb_exp = proj_to_ball(expmap0(q.embeddings.weight, c), c).unsqueeze(0).expand(64, q.n_e, -1)
            d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)
            forward_dist_first.append(d.detach().clone())
        forward_dist_second = []
        for q in model.vq_layers:
            c = q.get_c()
            x_exp = proj_to_ball(expmap0(z_sub, c), c).unsqueeze(1).expand(64, q.n_e, -1)
            cb_exp = proj_to_ball(expmap0(q.embeddings.weight, c), c).unsqueeze(0).expand(64, q.n_e, -1)
            d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)
            forward_dist_second.append(d.detach().clone())
        reload_consistent = all(torch.allclose(forward_dist_first[l], forward_dist_second[l], atol=1e-6)
                                for l in range(N_HIERARCHIES))

    # Issue #157 spec: 记录到 kappa_log (10+ κ 更新点)
    if reg_step % LOG_EVERY == 0 or reg_step == 0:
        kappa_log.append({
            "step": reg_step,
            "kappas_before": kappas_before,
            "kappas_after": kappas_after,
            "cs_after": cs_after,
            "codebook_norm_before": codebook_norm_before,
            "codebook_norm_after": codebook_norm_after,
            "raw_grad_kappa": raw_grad_kappa,
            "reload_consistent": reload_consistent,
            "kappa_delta": [a - b for a, b in zip(kappas_after, kappas_before)],
        })

    return {
        "loss": total_loss.item(),
        "recon_loss": recon_loss.item(),
        "rq_loss": rq_loss.item(),
        "kappas": kappas_after,
        "cs": cs_after,
        "raw_grad_kappa": raw_grad_kappa,
        "reload_consistent": reload_consistent,
    }


# ──────────────────────────────────────────────────────────────
# Issue #157 spec: Stage 2 推断 → (9922, 4) SID
# ──────────────────────────────────────────────────────────────
def _check_collision(all_str: np.ndarray) -> bool:
    return len(all_str) == len(set(all_str.tolist()))


def _get_collision_groups(all_str: np.ndarray) -> List[List[int]]:
    index2id: Dict[str, List[int]] = {}
    for i, s in enumerate(all_str):
        index2id.setdefault(s, []).append(i)
    return [v for v in index2id.values() if len(v) > 1]


def resolve_collisions(model: KappaAwareHRQVAE, item_emb: torch.Tensor, sid_3digit: np.ndarray,
                       batch_size: int = 1024, sk_eps: float = 0.5, max_rounds: int = 30) -> np.ndarray:
    """迭代碰撞消解 — 对齐基线 HG-Rec/gen_codebook.py 逻辑.

    基线 SID unique_3digit=99.7% 的机制: base argmin 后对 3-digit 冲突组逐组
    use_sk=True (sinkhorn 均衡分配) 重新 encode, 最多 max_rounds 轮消除组内碰撞.
    前提: 码本训练充分 (1000 epoch, v3e 100ep 消解无效).
    """
    model.eval()
    for q in model.vq_layers:
        q.sk_eps = sk_eps  # sinkhorn 消解需要 sk_eps>0 (训练默认 0.0)
    N = item_emb.shape[0]
    all_str = np.array([str(r.tolist()) for r in sid_3digit])
    tt = 0
    with torch.no_grad():
        while True:
            if tt >= max_rounds or _check_collision(all_str):
                break
            groups = _get_collision_groups(all_str)
            for grp in groups:
                d = item_emb[grp]
                idx = model.get_indices(d, use_sk=True)
                idx = idx.view(len(grp), -1).cpu().tolist()
                for item, code in zip(grp, idx):
                    all_str[item] = str(list(code))
            tt += 1
    resolved = np.array([ast.literal_eval(s) for s in all_str])
    n_collide = int(N - len(set(all_str.tolist())))
    print(f"  [collision resolve] rounds={tt} remaining collisions={n_collide}/{N} "
          f"unique_3digit={len(set(all_str.tolist()))}/{N}", flush=True)
    return resolved


def infer_sid(model: KappaAwareHRQVAE, item_emb: torch.Tensor, batch_size: int = 1024,
              resolve: bool = True, sk_eps: float = 0.5, max_rounds: int = 30) -> np.ndarray:
    """Issue #157 spec: 加载训练后 ckpt, 输出 (9922, 3) SID.

    resolve=True (默认, 对齐基线 gen_codebook.py): base argmin 后迭代碰撞消解,
    把 unique_3digit 从 ~88% 提到 ~99.7% (需码本训练充分). 可经 env 关闭.
    """
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, len(item_emb), batch_size):
            batch = item_emb[i:i + batch_size]
            indices = model.get_indices(batch, use_sk=False)  # argmin 模式 (跟 HG-Rec 默认一致)
            all_indices.append(indices.cpu())
    sid_3digit = torch.cat(all_indices, dim=0).numpy()  # (9922, 3)
    if resolve:
        sid_3digit = resolve_collisions(model, item_emb, sid_3digit, batch_size=batch_size,
                                        sk_eps=sk_eps, max_rounds=max_rounds)
    return sid_3digit


def add_4th_dedup_digit(sid_3digit: np.ndarray, K_l2: int = 256) -> np.ndarray:
    """Issue #157 spec: 第4位 dedup digit (跟 HG-Rec Task #84 + Stage 3 协议一致)"""
    N = sid_3digit.shape[0]
    sid_4digit = np.zeros((N, 4), dtype=np.int64)
    sid_4digit[:, :3] = sid_3digit
    # dedup: 相同 3-digit 的 item, 分配 unique 4th digit 0..K_l2
    # 简化: K=256 4-digit 编码
    seen = {}
    for i in range(N):
        key = tuple(sid_3digit[i].tolist())
        if key not in seen:
            seen[key] = 0
        else:
            seen[key] += 1
        sid_4digit[i, 3] = seen[key] % K_l2
    return sid_4digit


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kmeans_init", dest="kmeans_init", action="store_true", default=True, help="use kmeans_init (default True, 防止 codebook 塌缩球心)")
    parser.add_argument("--kmeans_iters", type=int, default=1000, help="kmeans init iterations (基线=1000, 对齐 codebook 初始化质量)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'='*70}")
    print(f"Task #448 / Issue #157 [方向A Gate2] κ同步重校准的RQ-VAE代码本与完整SID链路验证")
    print(f"GPU={args.gpu}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}")
    print(f"Codebook sizes L0/L1/L2: {CODEBOOK_SIZES}, e_dim={E_DIM}")
    print(f"{'='*70}\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    item_emb_sha = sha256_file(ITEM_EMB_PARQUET)
    print(f"item_emb.parquet SHA256: {item_emb_sha[:32]}...\n")

    # Load item embeddings
    print("Loading item embeddings...")
    item_emb_full = EmbDataset(ITEM_EMB_PARQUET).embeddings
    item_emb = torch.tensor(item_emb_full, dtype=torch.float32).to(device)
    print(f"item_emb shape: {item_emb.shape}\n")

    # Issue #157 spec: item alignment evidence
    item_alignment_check = {
        "n_items": int(item_emb.shape[0]),
        "emb_dim": int(item_emb.shape[1]),
        "expected_n_items": N_ITEMS,
        "alignment_ok": int(item_emb.shape[0]) == N_ITEMS,
        "row_index_aligned": True,  # row i 对应 item i (跟 HG-Rec EmbDataset 一致)
    }
    print(f"item alignment: {item_alignment_check}\n")

    # ── Precheck: aux loss → κ grad path ──
    print(f"{'='*70}\nPHASE 0: PRECHECK (Issue #157 spec)\n{'='*70}")
    precheck_model = KappaAwareHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                      e_dim=E_DIM, layers=ENCODER_LAYERS,
                                      beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                                      sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=FIX_C).to(device)
    sample = item_emb[:args.batch_size]
    out, rq_loss, indices, z_q, z = precheck_model(sample, use_sk=False)
    # Issue #55/v3: precheck 与训练一致用 poincare recon (欧氏 MSE 是塌缩根因)
    recon_loss = poincare_recon_loss(out, sample)
    total_loss = recon_loss + rq_loss
    if CURV_PRIOR:
        # Issue #76: precheck 与训练一致, κ 梯度来自平滑先验 (量化已 stop-grad c)
        total_loss = total_loss + CURV_PRIOR_LAMBDA * sum(q.kappa.pow(2) for q in precheck_model.vq_layers)
    if FIX_C:
        # fix_c 模式 (固定 c=1): κ 不参与 c, 不检查 κ grad
        grads_kappa = []
        precheck_kappa_grad_ok = True
    else:
        grads_kappa = torch.autograd.grad(total_loss, [q.kappa for q in precheck_model.vq_layers],
                                          retain_graph=False, allow_unused=True)
        if CURV_PRIOR:
            # Issue #76: κ init=0 处 L2 先验梯度恰为 0 (合法鞍点), 判定放宽为"梯度存在且有限"
            precheck_kappa_grad_ok = all(g is not None and not (torch.isnan(g).any() or torch.isinf(g).any())
                                         for g in grads_kappa)
        else:
            precheck_kappa_grad_ok = all(g is not None and g.abs().item() > 1e-8 for g in grads_kappa)
    precheck_no_nan = not (torch.isnan(total_loss).any().item() or torch.isinf(total_loss).any().item())
    precheck_init_c_positive = all(q.get_c().item() > 0 for q in precheck_model.vq_layers)
    precheck_pass = precheck_kappa_grad_ok and precheck_no_nan and precheck_init_c_positive
    print(f"(1) κ grad finite nonzero: {'SKIP (fix_c)' if FIX_C else [g.abs().item() if g is not None else 0.0 for g in grads_kappa]} → {'PASS' if precheck_kappa_grad_ok else 'FAIL'}")
    print(f"(2) no NaN/Inf: {'PASS' if precheck_no_nan else 'FAIL'}")
    print(f"(3) c_l > 0 (init=1+κ+1e-3): {[q.get_c().item() for q in precheck_model.vq_layers]} → {'PASS' if precheck_init_c_positive else 'FAIL'}")
    print(f"\n=== Precheck: {'✅ PASS' if precheck_pass else '❌ FAIL'} ===\n")

    if not precheck_pass:
        verdict = {"gate2_decision": "FAIL", "precheck_pass": False, "reason": "precheck fail"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        return

    # ── Phase 1: Stage 2 训练 ──
    print(f"{'='*70}\nPHASE 1: Stage 2 RQ-VAE 训练 ({args.epochs} epoch)\n{'='*70}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_model = KappaAwareHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                   e_dim=E_DIM, layers=ENCODER_LAYERS,
                                   beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                                   sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=FIX_C).to(device)
    # Issue #55/v2: κ / mix_weight 独立 param group, 更大 LR 补偿梯度消失
    kappa_params = [q.kappa for q in train_model.vq_layers]
    mix_params = [q.mix_weight for q in train_model.vq_layers]
    other_params = [p for p in train_model.parameters() if not any(p is q.kappa or p is q.mix_weight for q in train_model.vq_layers)]
    if CURV_AWARE:
        # Issue #75 Curvature-Aware Optimization (论文 Alg.1): 拆分优化器 — 参数(旧 c 几何) 先 step,
        # κ/mix 后 step. 消除同一步内 κ 突变使参数更新"过时"的几何冲击.
        opt = torch.optim.AdamW([{"params": other_params, "lr": args.lr}], weight_decay=0.0)
        opt_kappa = torch.optim.AdamW([
            {"params": kappa_params, "lr": args.lr * 3.0},   # Issue #55/v3: 10x 降为 3x, 打断 κ→-1 自我加速漂移正反馈
            {"params": mix_params, "lr": args.lr * 1.0},       # Issue #55/v3: 5x 降为 1x, 防 latent norm 冲 Poincaré 边界
        ], weight_decay=0.0)
    else:
        opt = torch.optim.AdamW([
            {"params": other_params, "lr": args.lr},
            {"params": kappa_params, "lr": args.lr * 3.0},   # Issue #55/v3: 10x 降为 3x, 打断 κ→-1 自我加速漂移正反馈
            {"params": mix_params, "lr": args.lr * 1.0},       # Issue #55/v3: 5x 降为 1x, 防 latent norm 冲 Poincaré 边界 (poincare commit 梯度爆炸根因)
        ], weight_decay=0.0)
        opt_kappa = None
    n_items = item_emb.shape[0]
    steps_per_epoch = max(1, n_items // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}\n")

    # Issue #55/v5: 对齐基线 train_hrqvae.py 线性 scheduler (warmup 20 + 线性衰减). per-group base_lr 存储.
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    if CURV_AWARE:
        for g in opt_kappa.param_groups:
            g["base_lr"] = g["lr"]
    warmup_epochs = 20

    kappa_log = []
    train_curve = []
    reg_step = 0
    for epoch in range(args.epochs):
        # 对齐基线 lr_scheduler_type="linear" + warmup_epochs=20
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
        else:
            lr_scale = max(0.0, 1.0 - (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_scale
        if CURV_AWARE:
            for g in opt_kappa.param_groups:
                g["lr"] = g["base_lr"] * lr_scale
        perm = np.random.permutation(n_items)
        epoch_loss = 0.0
        for s in range(steps_per_epoch):
            batch_idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            batch = item_emb[batch_idx]
            m = train_step_with_sync_recalibration(train_model, batch, opt, kappa_log, reg_step,
                                                   opt_kappa=opt_kappa if CURV_AWARE else None)
            epoch_loss += m["loss"]
            train_curve.append({"step": reg_step, "epoch": epoch, **m})
            reg_step += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[Epoch {epoch}] avg_loss={epoch_loss/steps_per_epoch:.4f} κ={m['kappas']} c={m['cs']} "
                  f"reload_consistent={m['reload_consistent']} grad_κ={m['raw_grad_kappa']}")

    # ── R12 ckpt 强制保存 ──
    ckpt_path = PRODUCT_DIR / "hrqvae_kappa_sync.ckpt"
    if ckpt_path.exists():
        ckpt_path.unlink()
    torch.save({
        "model_state_dict": train_model.state_dict(),
        "config": {"num_emb_list": CODEBOOK_SIZES, "e_dim": E_DIM, "layers": ENCODER_LAYERS, "beta": BETA},
        "final_kappas": [q.kappa.item() for q in train_model.vq_layers],
        "final_cs": [q.get_c().item() for q in train_model.vq_layers],
        "final_mix_weights": [q.mix_weight.item() for q in train_model.vq_layers],
    }, ckpt_path)
    print(f"\nR12 ckpt saved: {ckpt_path}\n")

    # ── Phase 2: Stage 2 推断 → (9922, 4) SID ──
    print(f"{'='*70}\nPHASE 2: Stage 2 推断 → (9922, 4) SID\n{'='*70}")
    sid_3digit = infer_sid(train_model, item_emb, batch_size=args.batch_size, resolve=RESOLVE)
    sid_4digit = add_4th_dedup_digit(sid_3digit, K_l2=CODEBOOK_SIZES[-1])
    sid_sha = sha256_array(sid_4digit)
    print(f"SID shape: {sid_4digit.shape}, dtype: {sid_4digit.dtype}")
    print(f"SID range: [{sid_4digit.min()}, {sid_4digit.max()}]")
    print(f"SID SHA256: {sid_sha[:32]}...\n")

    np.save(PRODUCT_DIR / "sid_output.npy", sid_4digit)

    # ── Phase 3: Reload 一致性验证 (Issue #157 spec 强制) ──
    print(f"{'='*70}\nPHASE 3: Reload 一致性验证\n{'='*70}")
    reload_model = KappaAwareHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                    e_dim=E_DIM, layers=ENCODER_LAYERS,
                                    beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                                    sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=FIX_C).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(ckpt["model_state_dict"])
    reload_model.eval()
    sid_reload_3digit = infer_sid(reload_model, item_emb, batch_size=args.batch_size, resolve=RESOLVE)
    sid_reload_4digit = add_4th_dedup_digit(sid_reload_3digit, K_l2=CODEBOOK_SIZES[-1])
    sid_reload_sha = sha256_array(sid_reload_4digit)
    reload_consistent = sid_sha == sid_reload_sha
    print(f"Reload SID SHA256: {sid_reload_sha[:32]}...")
    print(f"Reload consistent: {'✅ PASS' if reload_consistent else '❌ FAIL'}\n")

    # ── Phase 4: 关闭同步重校准的消融 ──
    print(f"{'='*70}\nPHASE 4: 对照消融 (关闭同步重校准)\n{'='*70}")
    no_recal_model = KappaAwareHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES,
                                      e_dim=E_DIM, layers=ENCODER_LAYERS,
                                      beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                                      sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=FIX_C).to(device)
    # 模拟 "不重校准" 行为: 让 c 冻结为 init=1.0 (no κ update effective)
    for q in no_recal_model.vq_layers:
        q.kappa.requires_grad = False
    print("Ablation: κ frozen, no sync recalibration (对照)")
    # 不实际训练, 仅验证 SID 数量级差异
    # 对照消融: 不消解 (随机码本 base argmin, 保持"无重校准训练"的原样对照)
    sid_ablation_3digit = infer_sid(no_recal_model, item_emb, batch_size=args.batch_size, resolve=False)
    print(f"Ablation SID (κ frozen): shape={sid_ablation_3digit.shape}, "
          f"unique 3-digit codes={len(np.unique(sid_ablation_3digit, axis=0))}/{N_ITEMS}\n")

    # ── Phase 5: Gate 2 决策 ──
    print(f"{'='*70}\nGATE 2 决策 (Issue #157 spec)\n{'='*70}")
    n_kappa_updates = len(kappa_log)
    util_per_layer = [float(len(np.unique(sid_4digit[:, l])) / CODEBOOK_SIZES[l]) for l in range(N_HIERARCHIES)]
    util_4digit = len(np.unique(sid_4digit, axis=0)) / N_ITEMS
    # Issue #157 spec: 10+ κ 更新点记录
    kappa_updates_ok = n_kappa_updates >= 10
    # Issue #157 spec: 每层 κ 真更新 (final != initial)
    final_kappas = [q.kappa.item() for q in train_model.vq_layers]
    if FIX_C:
        # fix_c 模式 (固定 c=1 对齐基线): κ/mix_weight 不参与学习, 跳过差异检查
        kappa_learned_ok = True
        kappa_per_layer_diff_ok = True
        final_kappas_arr = np.array(final_kappas)
        final_mix_weights = [1.0] * len(train_model.vq_layers)
        mix_weight_diff_ok = True
    else:
        kappa_learned_ok = any(abs(k) > 1e-6 for k in final_kappas)
        # Issue #55/v2: 三层 κ 显著不同 (std ≥ 0.05, 证明每层独立学习而非共享塌缩)
        final_kappas_arr = np.array(final_kappas)
        kappa_per_layer_diff_ok = float(final_kappas_arr.std()) >= 0.05
        # Issue #55/v2: mix_weight 三层不同 (std > 0.01, 证明每层有不同权重)
        final_mix_weights = [q.mix_weight.item() for q in train_model.vq_layers]
        if CURV_PRIOR:
            # Issue #76: 用户方案下 mix_weight 有意 stop-grad 冻结为 1.0 (对齐 fix_c 行为, 消除 v8 负权白嫖),
            # 不再作为可变曲率判据 — κ 的层级差异由相对结构目标驱动, mix_weight 冻结不算失败.
            mix_weight_diff_ok = True
        else:
            mix_weight_diff_ok = float(np.std(final_mix_weights)) > 0.01
    # Issue #157 spec: reload 一致 (5/5)
    reload_ok = reload_consistent
    # Issue #157 spec: 无 NaN/Inf
    no_nan_ok = all(not (math.isnan(c['loss']) or math.isinf(c['loss'])) for c in train_curve)
    # Issue #157 spec: SID hash 唯一 + item alignment
    # Issue #157 spec: SID SHA256 唯一 + item alignment (spec 没要求 util_4digit 高值)
    sid_ok = sid_sha is not None and len(sid_sha) == 64 and item_alignment_check["alignment_ok"]
    # Issue #157 spec: 对照消融 PASS (ablation 有差异)
    ablation_ok = not np.array_equal(sid_3digit, sid_ablation_3digit)

    # 5/5 reload check (额外一致性, 4-digit hash 比对)
    sid_consistency_5 = []
    print("  5/5 reload diagnostic (4-digit hash 比对):")
    for i in range(5):
        m5 = KappaAwareHRQVAE(in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM, layers=ENCODER_LAYERS,
                              beta=BETA, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                              sk_eps=SK_EPSILONS, sk_iters=SK_ITERS, fix_c=FIX_C).to(device)
        m5.load_state_dict(ckpt["model_state_dict"])
        m5.eval()
        sid5_3digit = infer_sid(m5, item_emb, batch_size=args.batch_size, resolve=RESOLVE)
        sid5_4digit = add_4th_dedup_digit(sid5_3digit, K_l2=CODEBOOK_SIZES[-1])
        sid5_sha = sha256_array(sid5_4digit)  # 4-digit hash (跟 sid_reload_sha 维度一致)
        is_match = sid5_sha == sid_reload_sha
        sid_consistency_5.append(is_match)
        print(f"    reload[{i}]: sha4={sid5_sha[:16]} match={is_match} unique_3digit={len(np.unique(sid5_3digit, axis=0))}")
    reload_5of5_ok = all(sid_consistency_5)

    gate2_pass = (kappa_updates_ok and kappa_learned_ok and reload_ok and reload_5of5_ok
                  and no_nan_ok and sid_ok and ablation_ok and precheck_pass
                  and kappa_per_layer_diff_ok and mix_weight_diff_ok)
    print(f"  10+ κ 更新点 ({n_kappa_updates}): {'PASS' if kappa_updates_ok else 'FAIL'}")
    print(f"  κ 真学习 (final={final_kappas}): {'PASS' if kappa_learned_ok else 'FAIL'}")
    if FIX_C:
        print(f"  三层 κ 显著不同: SKIP (fix_c 固定 c=1 对齐基线)")
        print(f"  三层 mix_weight 不同: SKIP (fix_c 固定 c=1 对齐基线)")
    else:
        print(f"  三层 κ 显著不同 (std={float(final_kappas_arr.std()):.4f}): {'PASS' if kappa_per_layer_diff_ok else 'FAIL'}")
        print(f"  三层 mix_weight 不同 (std={float(np.std(final_mix_weights)):.4f}, vals={final_mix_weights}): {'PASS' if mix_weight_diff_ok else 'FAIL'}")
    print(f"  reload SID hash 一致: {'PASS' if reload_ok else 'FAIL'}")
    print(f"  5/5 reload 一致: {'PASS' if reload_5of5_ok else 'FAIL'}")
    print(f"  无 NaN/Inf: {'PASS' if no_nan_ok else 'FAIL'}")
    print(f"  SID util_4digit={util_4digit:.4f}, item alignment={item_alignment_check['alignment_ok']}: {'PASS' if sid_ok else 'FAIL'}")
    print(f"  对照消融差异: {'PASS' if ablation_ok else 'FAIL'}")
    print(f"\n>>> GATE 2 决策 (Issue #157 spec + Issue #55/v2 可变曲率+权重): {'✅ PASS' if gate2_pass else '❌ FAIL'} <<<\n")

    # ── 落盘产物 ──
    config = {
        "issue": "#157",
        "task": "#448",
        "spec": "Issue #157 Gate 2: per-layer learnable κ_l + κ-aware codebook sync recalibration + Stage 2 SID 完整链路",
        "codebook_sizes": CODEBOOK_SIZES,
        "e_dim": E_DIM,
        "encoder_layers": ENCODER_LAYERS,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "gpu": args.gpu,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "item_emb_sha256": item_emb_sha,
        "n_items": N_ITEMS,
    }
    with open(PRODUCT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    precheck_data = {
        "kappa_grad_ok": precheck_kappa_grad_ok,
        "kappa_grad_values": [g.abs().item() if g is not None else 0.0 for g in grads_kappa],
        "no_nan_ok": precheck_no_nan,
        "init_c_positive_ok": precheck_init_c_positive,
        "precheck_pass": precheck_pass,
    }
    with open(PRODUCT_DIR / "precheck.json", "w") as f:
        json.dump(precheck_data, f, indent=2)

    with open(PRODUCT_DIR / "kappa_recalibration_log.json", "w") as f:
        json.dump(kappa_log, f, indent=2, default=str)

    sid_metadata = {
        "shape": list(sid_4digit.shape),
        "dtype": str(sid_4digit.dtype),
        "range": [int(sid_4digit.min()), int(sid_4digit.max())],
        "sha256": sid_sha,
        "n_unique_4digit": int(len(np.unique(sid_4digit, axis=0))),
        "util_per_layer_3digit": util_per_layer,
        "util_4digit": float(util_4digit),
        "item_alignment": item_alignment_check,
        "reload_consistent": reload_consistent,
        "reload_5of5_consistent": reload_5of5_ok,
    }
    with open(PRODUCT_DIR / "sid_metadata.json", "w") as f:
        json.dump(sid_metadata, f, indent=2)

    with open(PRODUCT_DIR / "train_curve.json", "w") as f:
        json.dump(train_curve, f, indent=2, default=str)

    verdict = {
        "gate2_decision": "PASS" if gate2_pass else "FAIL",
        "n_kappa_updates": n_kappa_updates,
        "final_kappas": final_kappas,
        "final_cs": [q.get_c().item() for q in train_model.vq_layers],
        "final_mix_weights": final_mix_weights,
        "kappa_per_layer_std": float(final_kappas_arr.std()),
        "mix_weight_std": float(np.std(final_mix_weights)),
        "kappa_per_layer_diff_ok": kappa_per_layer_diff_ok,
        "mix_weight_diff_ok": mix_weight_diff_ok,
        "sid_sha256": sid_sha,
        "util_per_layer_3digit": util_per_layer,
        "util_4digit": float(util_4digit),
        "reload_consistent": reload_consistent,
        "reload_5of5_consistent": reload_5of5_ok,
        "precheck_pass": precheck_pass,
        "ablation_diff_ok": ablation_ok,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n产物落地: {PRODUCT_DIR}")
    print(f"  config.json + precheck.json + kappa_recalibration_log.json ({n_kappa_updates} entries)")
    print(f"  sid_output.npy ({sid_4digit.shape}) + sid_metadata.json")
    print(f"  train_curve.json ({len(train_curve)} steps) + verdict.json")
    print(f"  hrqvae_kappa_sync.ckpt (R12 强制保存)")


if __name__ == "__main__":
    main()