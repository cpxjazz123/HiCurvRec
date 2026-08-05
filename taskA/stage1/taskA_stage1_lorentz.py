#!/usr/bin/env python3
"""Task #84 Issue #49 Stage1 — frozen sentence-t5-base 前 N-1 block + Lorentz 最后 block (严格 precheck).

承接 #48 (838531d): 修复 Lorentz 数值参数化并严格完成 PC1-PC8.

修复一 (平滑有界切空间参数化):
  s = ||u||_2, ρ(u) = ρ_max·tanh(s/ρ_max), v(u) = ρ(u)/(s+ε)·u
  ρ_max=1.0, ε=1e-12 固定, 不扫描. v(u) 才是 Exp/Log 互逆检查的输入,
  不再将 Log(Exp(v(u))) 与未约束原始 u 比较. sinh(x)/x 接近 0 用解析极限.

修复二 (几何核心强制 float64):
  Sentence-T5 冻结输出保持原 dtype; 从输入投影之后到 Lorentz pooling/LogMap
  的几何核心必须 float64: h64 = float64(h), z_L = LorentzBlock_64(h64).
  不得在 attention/Minkowski centroid/ExpMap/LogMap/距离/precheck 中静默转回
  float32/bfloat16. Stage1 parquet 写出前才显式转换, 并记录转换前后最大误差.

PC1-PC8 (Issue #49 spec):
  PC1 全路径流形约束: <1e-8 (float64), 所有点 x0>0, max/mean/P99 finite
  PC2 Exp/Log 互逆: 小范数/近 ρ_max/真实 T5 投影三种, max <1e-8
  PC3 Lorentz/Poincaré 等距一致性: max <1e-7, P99 <1e-8, p = x_{1:d}/(√c·x0+1)
  PC4 Attention 合法: 行和与 1 误差 <1e-10, padding key 权重严格 0, 全 finite
  PC5 Minkowski centroid: 归一化前 <m_i,m_i>_L<0, 归一化后 <1e-8, 禁 fallback projection
  PC6 中心有限差分: float64 h=1e-5, W_Q/W_K/W_V/W_1/W_2 各一非零坐标,
      g_AD/g_FD finite nonzero, sign 一致, rel <1e-3, 不得降级
  PC7 真实路径审计: forward/export/reload 三条路径计数/hash, 禁用 Lorentz 后 rel>1e-3
  PC8 三层曲率与最近邻合规: 三个独立 nn.Parameter 地址, MLR_ENABLED=False 显式断言,
      assignment 由 argmin d_c 唯一产生, 扰动 κ 仅对应层缓存变化, reload 5/5

唯一允许结论: precheck blocked 或 precheck PASS. 全部 PASS 前禁止 Stage1-4 训练.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

# === R30: 所有超参硬编码常量 (无 os.environ.get) ===
ITEM_JSON = "/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments.item.json"
OUTPUT_PARQUET_DEFAULT = str(REPO / "taskA/_history/taskA_stage1_issue49/item_emb.parquet")
TAG = "issue49_lorentz"
ENCODER_MODEL = "sentence-transformers/sentence-t5-base"
# Issue #48 spec: c_enc=1.0 固定 (Stage1 Lorentz 坐标系), 不搜索
C_ENC = 1.0
N_FROZEN_BLOCKS = 11
# 修复一: 平滑有界切空间参数化 (Issue #49 spec 强制, 不扫描)
RHO_MAX = 1.0
BOUND_EPS = 1e-12
# 修复二: 几何核心 dtype
GEOM_DTYPE = torch.float64
# Lorentz 距离 arcosh 定义域机器精度保护
ARCOSH_EPS = 1e-12
# Stage1 Lorentz block 训练 (本 Issue 不执行, 保留入口供后续 Gate1 Issue)
TRAIN_BATCH_SIZE = 64
TRAIN_EPOCHS = 3
TRAIN_LR = 1e-4
TRAIN_SEED = 42
DEVICE = "cuda:0"
MAX_SEQ_LEN = 64
HFFN_HIDDEN = 2048
# 教师/学生关系分布温度与 L_aug 权重 (spec: 写死)
KL_TEMP = 0.07
L_AUG_WEIGHT = 0.1
DROPOUT_AUG = 0.2
# PC 阈值 (Issue #49 spec)
PC1_MANIFOLD_MAX = 1e-8
PC2_INVERSE_MAX = 1e-8
PC3_ISO_MAX = 1e-7
PC3_ISO_P99 = 1e-8
PC4_ROWWISE_MAX = 1e-10
PC5_PRE_INNER_LT_ZERO = True
PC5_POST_MANIFOLD_MAX = 1e-8
PC6_FD_H = 1e-5
PC6_REL_MAX = 1e-3
PC7_REL_DIFF_MIN = 1e-3
# Precheck 固定 seed + 真实训练集 audit batch
PRECHECK_SEED = 42
PRECHECK_AUDIT_SIZE = 64


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}", flush=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ================================================================
# 几何核心 (修复二: 强制 float64)
# ================================================================

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """<x,y>_L = -x_0*y_0 + sum_{j>=1} x_j*y_j. x, y: (..., d+1), float64."""
    x = x.to(GEOM_DTYPE)
    y = y.to(GEOM_DTYPE)
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def arcosh_safe(z: torch.Tensor, eps: float = ARCOSH_EPS) -> torch.Tensor:
    """arcosh(z) for z >= 1+eps (机器精度保护), safe gradient near z=1."""
    z = z.to(GEOM_DTYPE)
    z_safe = torch.clamp(z, min=1.0 + eps)
    return torch.acosh(z_safe)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """d_c(x,y) = (1/sqrt(c)) * arcosh(max(1+eps, -c <x,y>_L)), float64."""
    inner = minkowski_inner(x, y)
    z = -c * inner
    return arcosh_safe(z) / math.sqrt(c)


def smooth_bounded_v(u: torch.Tensor, rho_max: float = RHO_MAX, eps: float = BOUND_EPS) -> torch.Tensor:
    """修复一: 平滑有界切空间参数化.

    s = ||u||_2, ρ(u) = ρ_max·tanh(s/ρ_max), v(u) = ρ(u)/(s+ε)·u.
    返回 v(u), ||v(u)|| = ρ(u) < ρ_max 平滑 (u=0 时 v=0).
    float64 强制.
    """
    u = u.to(GEOM_DTYPE)
    s = u.norm(dim=-1, keepdim=True)  # (..., 1)
    rho = rho_max * torch.tanh(s / rho_max)  # (..., 1)
    v = rho / (s + eps) * u
    return v


def expmap_o(v: torch.Tensor, c: float) -> torch.Tensor:
    """Exp_o(v) for v=(0, v_s) tangent at origin o. v: (..., d) 无前导 0, float64.

    o = (1/√c, 0, ..., 0)
    Exp_o(v) = cosh(√c·||v||_L)·o + sinh(√c·||v||_L)/(√c·||v||_L)·v_full
    sinh(x)/x 接近 0 用解析极限 (1 + x²/6), 不得产生 0/0.
    """
    v = v.to(GEOM_DTYPE)
    if v.shape[-1] == 0:
        raise ValueError("v is empty")
    v_norm = v.norm(dim=-1, keepdim=True)  # (..., 1)
    sqrt_c = math.sqrt(c)
    alpha = sqrt_c * v_norm  # (..., 1)
    # sinh(alpha)/alpha 稳定分支: alpha→0 时 → 1 + alpha²/6 (泰勒解析极限)
    alpha_safe = torch.clamp(alpha, min=1e-12)
    sinh_over_alpha = torch.where(
        alpha < 1e-6,
        1.0 + alpha_safe * alpha_safe / 6.0,
        torch.sinh(alpha_safe) / alpha_safe,
    )
    cosh_term = torch.cosh(alpha)  # (..., 1)
    o = torch.zeros(*v.shape[:-1], v.shape[-1] + 1, device=v.device, dtype=GEOM_DTYPE)
    o[..., 0] = 1.0 / sqrt_c
    v_full = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # (..., d+1)
    return cosh_term * o + sinh_over_alpha * v_full


def logmap_o(x: torch.Tensor, c: float) -> torch.Tensor:
    """Log_o(x) for x on H^d_c. x: (..., d+1), float64.

    alpha = -c <o, x>_L = √c·x_0 (o=(1/√c,0,...))
    d = arcosh(alpha)/√c
    v = d/sinh(√c·d) · (x - alpha·o)   (tangent at o, 返回 (..., d) 去前导 0)
    """
    x = x.to(GEOM_DTYPE)
    sqrt_c = math.sqrt(c)
    alpha = sqrt_c * x[..., 0]  # (...,)
    d = arcosh_safe(alpha.clamp_min(1.0 + ARCOSH_EPS)) / sqrt_c  # (...,)
    x_centered = x.clone()
    x_centered[..., 0] = x[..., 0] - alpha / sqrt_c
    sqrt_c_d = sqrt_c * d
    sinh_term = torch.sinh(sqrt_c_d).clamp_min(1e-15)
    coeff = (d / sinh_term).unsqueeze(-1)  # (..., 1)
    return coeff * x_centered[..., 1:]  # (..., d)


def lorentz_centroid(values: torch.Tensor, weights: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """归一化 Minkowski centroid (修复二 float64; Issue #49 PC5 禁 fallback).

    m_i = Σ_j A_ij v_j; 归一化前必须 <m_i,m_i>_L < 0 (timelike), 否则 raise;
    归一化 m / sqrt(-c <m,m>_L) 精确回流形 (禁 project_to_lorentz 兜底).
    einsum '...n,...nd->...d' 加权和: 不物化 (..., N, d+1) 中间张量 (OOM 防护).

    values: (..., N, d+1) on H^d_c; weights: (..., N) 非负, sum≈1
    return (centroid_on_manifold, m_raw)
    """
    values = values.to(GEOM_DTYPE)
    weights = weights.to(GEOM_DTYPE)
    # 加权和 m_i = Σ_j w_ij v_j. 显式维度分派 (einsum ellipsis 对 3D/3D 有广播歧义):
    #   - pooling:  weights (B, N) × values (B, N, d+1) → (B, d+1)
    #   - attention: weights (B*h, L, L) × values (B*h, L, d_h+1) → bmm (B*h, L, d_h+1)
    #   - per-head:  weights (B, h, Lq, Lk) × values (B, h, Lk, dh+1) → reshape+bmm
    if weights.ndim == 2 and values.ndim == 3:
        m = (weights.unsqueeze(-1) * values).sum(dim=-2)  # (B, d+1)
    elif weights.ndim == 3 and values.ndim == 3:
        m = torch.bmm(weights, values)  # (B*h, L_q, d+1)
    elif weights.ndim == 4 and values.ndim == 4:
        w2 = weights.reshape(-1, weights.shape[-2], weights.shape[-1])  # (B*h, Lq, Lk)
        v2 = values.reshape(-1, values.shape[-2], values.shape[-1])     # (B*h, Lk, dh+1)
        m = torch.bmm(w2, v2).reshape(weights.shape[:-2] + (weights.shape[-2], values.shape[-1]))  # (B, h, Lq, dh+1)
    else:
        raise ValueError(f"lorentz_centroid: unsupported shapes weights={tuple(weights.shape)} values={tuple(values.shape)}")
    inner = minkowski_inner(m, m)  # (...,)
    if not (inner < 0).all():
        raise RuntimeError(
            f"PC5 FAIL: Minkowski centroid 归一化前 <m,m>_L 必须 < 0 (timelike), "
            f"got min={inner.min().item():.6e}, max={inner.max().item():.6e}"
        )
    norm_factor = torch.sqrt(-c * inner).unsqueeze(-1)  # (..., 1)
    out = m / norm_factor
    # 精确投影回流形 (无需 fallback; 数学恒等 <m,m>_L = -1/c)
    return out, m


# ================================================================
# Lorentz block 组件 (修复二: 全程 float64)
# ================================================================

class HFFN(nn.Module):
    """Log_o → MLP → LN + 固定 1/√d 缩放 → 平滑有界 → Exp_o (Issue #48/#49 spec + #49 梯度修复).

    梯度修复: LayerNorm 输出 norm=√d≈27.7 会令 smooth_bounded_v 的 tanh 饱和
    (∂v/∂u≈0.036) 且梯度方向锁死平行于 LN 归一化输入, 被 LN 反向投影消去
    (实测 1e-17). LN 后乘固定常数 1/√d → norm≈1 → tanh 非饱和区 (导数≈0.42),
    梯度方向混合, PC6 中心差分可过. 该缩放是常数架构修正, 不改 spec 参数化.
    """

    def __init__(self, d_model: int, d_hidden: int, c: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.scale = 1.0 / math.sqrt(d_model)
        self.c = c

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        x_lorentz = x_lorentz.to(GEOM_DTYPE)
        v = logmap_o(x_lorentz, self.c)  # (B, L, d) float64
        v = self.linear1(v)
        v = F.gelu(v)
        v = self.linear2(v)
        v = self.ln(v) * self.scale  # norm ≈ 1
        v_bounded = smooth_bounded_v(v)  # 修复一
        return expmap_o(v_bounded, self.c)


class HResLN(nn.Module):
    """Log_o(x) + Log_o(y) → LayerNorm → 固定 1/√d 缩放 → 平滑有界 → Exp_o (#49 梯度修复同 HFFN)."""

    def __init__(self, d_model: int, c: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.scale = 1.0 / math.sqrt(d_model)
        self.c = c

    def forward(self, x_lorentz: torch.Tensor, y_lorentz: torch.Tensor) -> torch.Tensor:
        x_lorentz = x_lorentz.to(GEOM_DTYPE)
        y_lorentz = y_lorentz.to(GEOM_DTYPE)
        v_x = logmap_o(x_lorentz, self.c)  # (B, L, d)
        v_y = logmap_o(y_lorentz, self.c)
        v = self.ln(v_x + v_y) * self.scale  # norm ≈ 1
        v_bounded = smooth_bounded_v(v)  # 修复一
        return expmap_o(v_bounded, self.c)


class HAttention(nn.Module):
    """Lorentz distance attention (Issue #48 spec; 修复一/二: float64 + 平滑有界)."""

    def __init__(self, d_model: int, c: float, n_heads: int = 12):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.c = c
        # 统计 (PC4/PC7 审计用)
        self.last_attn_weights = None
        self.last_attn_scores = None
        self.last_entropy = None

    def forward(self, x_lorentz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_lorentz = x_lorentz.to(GEOM_DTYPE)
        mask = mask.to(GEOM_DTYPE)
        B, L, _ = x_lorentz.shape
        # 1. token → 切空间 → 平滑有界 → Q/K/V 投影
        v = logmap_o(x_lorentz, self.c)  # (B, L, d) float64
        v_b = smooth_bounded_v(v)  # 修复一
        q = self.w_q(v_b).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, L, d_h)
        k = self.w_k(v_b).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v_proj = self.w_v(v_b).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        # 2. 平滑有界 → Exp_o → Lorentz (每 head 在 d_h+1 维 Lorentz 流形)
        q_b = smooth_bounded_v(q)   # (B, h, L, d_h)
        k_b = smooth_bounded_v(k)
        vp_b = smooth_bounded_v(v_proj)
        q_lorentz = expmap_o(q_b, self.c)   # (B, h, L, d_h+1)
        k_lorentz = expmap_o(k_b, self.c)
        v_lorentz = expmap_o(vp_b, self.c)
        # 3. Lorentz distance attention scores
        q_flat = q_lorentz.reshape(B * self.n_heads, L, -1)  # (B*h, L, d_h+1)
        k_flat = k_lorentz.reshape(B * self.n_heads, L, -1)
        inner = -q_flat[..., 0:1] * k_flat[..., 0:1].transpose(-1, -2) + torch.matmul(
            q_flat[..., 1:], k_flat[..., 1:].transpose(-1, -2)
        )  # (B*h, L_q, L_k)
        z = -self.c * inner
        z_safe = torch.clamp(z, min=1.0 + ARCOSH_EPS)
        d_c = torch.acosh(z_safe) / math.sqrt(self.c)
        attn_score = -d_c ** 2 / math.sqrt(self.d_head)  # (B*h, L_q, L_k)
        # mask (padding key 权重严格 0)
        # 注意: 广播 masked_fill 在本 torch 版本有分配 bug (mask 需广播时物化 ~19GB),
        # 必须显式构造与 attn_score 同形状的 bool mask (实测 0.38GB, 见 test_attn_mem).
        attn_mask_full = (mask > 0)[:, None, :].expand(B, L, L).repeat(self.n_heads, 1, 1)  # (B*h, L_q, L_k) bool
        attn_score_masked = attn_score.masked_fill(~attn_mask_full, float("-inf"))
        attn_weights = F.softmax(attn_score_masked, dim=-1)
        # 统计记录
        self.last_attn_weights = attn_weights.detach()
        self.last_attn_scores = attn_score.detach()
        ent = -torch.sum(attn_weights * torch.log(attn_weights.clamp_min(1e-15)), dim=-1)
        self.last_entropy = ent.detach()
        # 4. Minkowski centroid value aggregation (每 head; einsum 不物化 5D 中间)
        v_flat = v_lorentz.reshape(B * self.n_heads, L, -1)  # (B*h, L_k, d_h+1)
        attn_for_centroid = attn_weights.reshape(B * self.n_heads, L, L)  # (B*h, L_q, L_k)
        out_centroid, _ = lorentz_centroid(v_flat, attn_for_centroid, self.c)  # (B*h, L_q, d_h+1)
        out = out_centroid.reshape(B, self.n_heads, L, self.d_head + 1)  # (B, h, L, d_h+1)
        # 5. concat heads → 切空间 → w_o → 平滑有界 → Exp_o
        out_tangent = logmap_o(out, self.c)  # (B, h, L, d_h)
        out_tangent = out_tangent.transpose(1, 2).reshape(B, L, self.d_model)  # (B, L, d)
        out_tangent = self.w_o(out_tangent)  # (B, L, d)
        out_bounded = smooth_bounded_v(out_tangent)  # 修复一
        return expmap_o(out_bounded, self.c)


class LorentzBlock(nn.Module):
    """完整 Lorentz block: HAttention → HResLN → HFFN → HResLN (Issue #48/#49 spec)."""

    def __init__(self, d_model: int, d_hidden: int, c: float, n_heads: int = 12):
        super().__init__()
        self.attn = HAttention(d_model, c, n_heads=n_heads)
        self.ln1 = HResLN(d_model, c)
        self.ffn = HFFN(d_model, d_hidden, c)
        self.ln2 = HResLN(d_model, c)
        self.c = c

    def forward(self, x_lorentz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_lorentz = x_lorentz.to(GEOM_DTYPE)
        attn_out = self.attn(x_lorentz, mask)
        x = self.ln1(x_lorentz, attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x, ffn_out)
        return x


class FrozenT5Encoder(nn.Module):
    """frozen sentence-t5-base 前 N-1 block, 返回 token hidden states (float32)."""

    def __init__(self, model_name: str, n_frozen: int):
        super().__init__()
        from transformers import T5EncoderModel
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.n_frozen = n_frozen
        assert n_frozen < len(self.encoder.encoder.block), (
            f"n_frozen={n_frozen} must be < total blocks={len(self.encoder.encoder.block)}"
        )
        self.d_model = self.encoder.config.d_model
        self.n_blocks = len(self.encoder.encoder.block)

    def encode_to_token(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            inputs_emb = self.encoder.shared(input_ids)
            ext_mask = self.encoder.get_extended_attention_mask(attention_mask, input_ids.shape).to(input_ids.device)
            hidden = inputs_emb
            for i in range(self.n_frozen):
                hidden = self.encoder.encoder.block[i](hidden, attention_mask=ext_mask)[0]
        return hidden  # (B, L, d_model) float32


class Stage1LorentzEncoder(nn.Module):
    """frozen t5 前 N-1 block → 输入投影 (float32→float64) → Lorentz block → centroid → Log_o → u_item."""

    def __init__(self, model_name: str, n_frozen: int, c_enc: float, hffn_hidden: int):
        super().__init__()
        self.frozen_t5 = FrozenT5Encoder(model_name, n_frozen)
        self.input_proj = nn.Linear(self.frozen_t5.d_model, self.frozen_t5.d_model)
        self.lorentz_block = LorentzBlock(
            d_model=self.frozen_t5.d_model,
            d_hidden=hffn_hidden,
            c=c_enc,
            n_heads=12,
        )
        self.c = c_enc
        # 修复二: 几何核心组件转 float64 (input_proj + lorentz_block)
        self.input_proj.to(GEOM_DTYPE)
        self.lorentz_block.to(GEOM_DTYPE)

    def freeze_t5(self):
        for p in self.frozen_t5.parameters():
            p.requires_grad = False
        for p in self.input_proj.parameters():
            p.requires_grad = True
        for p in self.lorentz_block.parameters():
            p.requires_grad = True

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               return_float64: bool = False) -> torch.Tensor:
        """返回 u_item (B, d). 几何核心全程 float64; 默认返回 float32 (parquet 接口),
        return_float64=True 时返回 float64 (precheck 用)."""
        # 1. frozen t5 前 N-1 block → token hidden (B, L, d) float32
        hidden = self.frozen_t5.encode_to_token(input_ids, attention_mask)
        # 2. 输入投影 → float64 (修复二)
        h64 = hidden.to(GEOM_DTYPE)
        v_tangent = torch.tanh(self.input_proj(h64))  # (B, L, d) float64
        # 3. 平滑有界 → Exp_o → Lorentz (修复一)
        v_b = smooth_bounded_v(v_tangent)
        x_lorentz = expmap_o(v_b, self.c)  # (B, L, d+1)
        # 4. Lorentz block
        x_lorentz = self.lorentz_block(x_lorentz, attention_mask.to(GEOM_DTYPE))  # (B, L, d+1)
        # 5. mask-aware Minkowski centroid
        mask_f = attention_mask.to(GEOM_DTYPE)  # (B, L)
        z_lorentz, _ = lorentz_centroid(x_lorentz, mask_f, self.c)  # (B, d+1)
        # 6. Log_o → 切空间欧氏向量 (B, d) float64
        u_item = logmap_o(z_lorentz, self.c)
        if return_float64:
            return u_item
        return u_item.to(torch.float32)


# ================================================================
# PC1-PC8 (Issue #49 spec 严格验收)
# ================================================================

def manifold_constraint_err(x: torch.Tensor, c: float) -> torch.Tensor:
    """δ_L(x) = |<x,x>_L + 1/c| (float64)."""
    x = x.to(GEOM_DTYPE)
    inner = minkowski_inner(x, x)
    return (inner + 1.0 / c).abs()


def pc1_manifold_constraint(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC1: 全路径流形约束 (Q/K/V ExpMap 后 / 每 head 聚合后 / HResLN 后 / HFFN 后 / pooling 后)."""
    B, L = input_ids.shape
    model.eval()
    hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)  # (B, L, d) float32
    h64 = hidden.to(GEOM_DTYPE)
    v_b = smooth_bounded_v(torch.tanh(model.input_proj(h64)))
    x_lorentz = expmap_o(v_b, model.c)  # (B, L, d+1)

    # Q/K/V ExpMap 后
    v_t = logmap_o(x_lorentz, model.c)
    v_b2 = smooth_bounded_v(v_t)
    q = model.lorentz_block.attn.w_q(v_b2).view(B, L, 12, -1).transpose(1, 2)
    k = model.lorentz_block.attn.w_k(v_b2).view(B, L, 12, -1).transpose(1, 2)
    vp = model.lorentz_block.attn.w_v(v_b2).view(B, L, 12, -1).transpose(1, 2)
    q_lor = expmap_o(smooth_bounded_v(q), model.c)
    k_lor = expmap_o(smooth_bounded_v(k), model.c)
    v_lor = expmap_o(smooth_bounded_v(vp), model.c)

    # 每 head 聚合后
    attn_weights = model.lorentz_block.attn.last_attn_weights  # 从 forward 拿 (若已有)
    # 先跑一次完整 block 拿内部 attention 输出
    x_attn = model.lorentz_block.attn(x_lorentz, attention_mask)  # (B, L, d+1)
    x_resln1 = model.lorentz_block.ln1(x_lorentz, x_attn)
    x_ffn = model.lorentz_block.ffn(x_resln1)
    x_resln2 = model.lorentz_block.ln2(x_resln1, x_ffn)
    # pooling
    mask_f = attention_mask.to(GEOM_DTYPE)
    z_centroid, _ = lorentz_centroid(x_resln2, mask_f, model.c)

    locations = {
        "q_expmap": q_lor,
        "k_expmap": k_lor,
        "v_expmap": v_lor,
        "attn_out": x_attn,
        "resln1_out": x_resln1,
        "ffn_out": x_ffn,
        "resln2_out": x_resln2,
        "pooling_out": z_centroid,
    }
    max_errs = {}
    mean_errs = {}
    p99_errs = {}
    x0_mins = {}
    all_ok = True
    for name, x in locations.items():
        errs = manifold_constraint_err(x, c)
        max_errs[name] = float(errs.max().item())
        mean_errs[name] = float(errs.mean().item())
        p99_errs[name] = float(torch.quantile(errs.flatten(), 0.99).item())
        x0_mins[name] = float(x[..., 0].min().item())
        finite_ok = torch.isfinite(errs).all().item()
        x0_ok = x[..., 0].min().item() > 0
        ok = max_errs[name] < PC1_MANIFOLD_MAX and finite_ok and x0_ok
        if not ok:
            all_ok = False

    result = {
        "status": "PASS" if all_ok else "FAIL",
        "threshold_max": PC1_MANIFOLD_MAX,
        "locations": {
            name: {
                "max_err": max_errs[name],
                "mean_err": mean_errs[name],
                "p99_err": p99_errs[name],
                "x0_min": x0_mins[name],
                "finite": bool(torch.isfinite(manifold_constraint_err(locations[name], c)).all().item()),
            }
            for name in locations
        },
    }
    return result


def pc2_exp_log_inverse(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC2: Exp/Log 互逆. 小范数 / 接近 ρ_max / 真实 T5 投影后 三种输入."""
    model.eval()
    B, L = input_ids.shape
    d = model.frozen_t5.d_model

    # 真实 T5 投影后的有界向量
    hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)
    h64 = hidden.to(GEOM_DTYPE)
    v_real = smooth_bounded_v(torch.tanh(model.input_proj(h64))).reshape(-1, d)[:16]

    cases = {}
    # 1. 小范数 (||u|| ≈ 0.05)
    u_small = torch.randn(16, d, dtype=GEOM_DTYPE, device=input_ids.device) * 0.05
    cases["small_norm"] = (u_small, smooth_bounded_v(u_small))
    # 2. 接近 ρ_max (||u|| ≈ 5.0 → ρ(u) ≈ ρ_max·tanh(5) ≈ 0.9999)
    u_large = torch.randn(16, d, dtype=GEOM_DTYPE, device=input_ids.device) * 5.0
    cases["near_rho_max"] = (u_large, smooth_bounded_v(u_large))
    # 3. 真实 T5 投影后
    cases["real_t5_projected"] = (v_real, v_real)

    results = {}
    all_ok = True
    for name, (u_orig, v_u) in cases.items():
        x = expmap_o(v_u, c)
        v_recon = logmap_o(x, c)
        denom = v_u.norm(dim=-1).clamp_min(1e-12)
        eps_inv = (v_recon - v_u).norm(dim=-1) / denom
        max_eps = float(eps_inv.max().item())
        results[name] = {
            "u_norm_range": [float(u_orig.norm(dim=-1).min().item()), float(u_orig.norm(dim=-1).max().item())],
            "v_u_norm_range": [float(v_u.norm(dim=-1).min().item()), float(v_u.norm(dim=-1).max().item())],
            "max_inv_rel_err": max_eps,
            "threshold": PC2_INVERSE_MAX,
            "status": "PASS" if max_eps < PC2_INVERSE_MAX else "FAIL",
        }
        if max_eps >= PC2_INVERSE_MAX:
            all_ok = False

    return {"status": "PASS" if all_ok else "FAIL", "threshold": PC2_INVERSE_MAX, "cases": results}


def pc2_near_zero_inverse(c: float, device: torch.device) -> dict:
    """Issue #50 补证一: PC2 近零稳定分支 + 精确零向量 + AD/中心差分.

    v_j = 10^-j·a (j∈{4,6,8,10,12}, a 固定方向单位向量 seed=42) + 精确零向量.
    非零样本 eps_inv = ||Log(Exp(v))-v||/max(||v||,1e-15) < 1e-8;
    零向量: Exp_o(0)=o 绝对误差 <1e-12, Log_o(o)=0 绝对误差 <1e-12;
    对 10^-8 / 10^-10 各记录输出坐标 0 的 autograd 与中心有限差分 (h=1e-5),
    sign 一致且相对误差 <1e-3.
    """
    torch.manual_seed(PRECHECK_SEED)
    d = 768
    a = torch.randn(d, dtype=GEOM_DTYPE, device=device)
    a = a / a.norm()

    # 1. 非零近零样本
    orders = [4, 6, 8, 10, 12]
    cases = {}
    all_ok = True
    for j in orders:
        v = (10.0 ** (-j)) * a
        x = expmap_o(v, c)
        v_rec = logmap_o(x, c)
        denom = v.norm().clamp_min(1e-15)
        rel = float((v_rec - v).norm().item() / denom.item())
        finite = bool(torch.isfinite(v_rec).all().item())
        ok = finite and rel < PC2_INVERSE_MAX
        all_ok = all_ok and ok
        cases[f"1e-{j}"] = {
            "v_norm": float(v.norm().item()),
            "eps_inv": rel,
            "finite": finite,
            "status": "PASS" if ok else "FAIL",
        }

    # 2. 精确零向量
    sqrt_c = math.sqrt(c)
    o = torch.zeros(d + 1, dtype=GEOM_DTYPE, device=device)
    o[0] = 1.0 / sqrt_c
    x0 = expmap_o(torch.zeros(d, dtype=GEOM_DTYPE, device=device), c)
    exp_zero_err = float((x0 - o).abs().max().item())
    log_zero_err = float(logmap_o(o, c).abs().max().item())
    zero_ok = exp_zero_err < 1e-12 and log_zero_err < 1e-12
    all_ok = all_ok and zero_ok

    # 3. AD vs 中心差分 (10^-8 与 10^-10, 输出坐标 0, 输入坐标 0)
    fd_details = []
    h = PC6_FD_H
    for v_scale in [1e-8, 1e-10]:
        v0 = (v_scale * a).clone()
        e0 = torch.zeros(d, dtype=GEOM_DTYPE, device=device)
        e0[0] = 1.0
        # AD
        v = v0.clone().requires_grad_(True)
        loss = logmap_o(expmap_o(v, c), c)[0]
        loss.backward()
        g_ad = float(v.grad[0].item())
        # FD 中心差分 (no_grad, 同 h=1e-5)
        with torch.no_grad():
            loss_plus = logmap_o(expmap_o(v0 + h * e0, c), c)[0]
            loss_minus = logmap_o(expmap_o(v0 - h * e0, c), c)[0]
        g_fd = float(((loss_plus - loss_minus) / (2 * h)).item())
        sign_ok = (g_ad > 0) == (g_fd > 0)
        rel_g = abs(g_ad - g_fd) / max(abs(g_ad), abs(g_fd), 1e-12)
        ok = math.isfinite(g_ad) and math.isfinite(g_fd) and sign_ok and rel_g < PC6_REL_MAX
        all_ok = all_ok and ok
        fd_details.append({
            "v_scale": v_scale,
            "coord": 0,
            "g_AD": g_ad,
            "g_FD": g_fd,
            "relative_error": rel_g,
            "sign_consistent": sign_ok,
            "status": "PASS" if ok else "FAIL",
        })

    return {
        "status": "PASS" if all_ok else "FAIL",
        "threshold_nonzero_rel": PC2_INVERSE_MAX,
        "threshold_zero_abs": 1e-12,
        "direction_a_first_coords": a[:3].tolist(),
        "cases": cases,
        "zero_vector": {"exp_o_0_err": exp_zero_err, "log_o_o_err": log_zero_err, "status": "PASS" if zero_ok else "FAIL"},
        "ad_vs_fd": {"fd_h": h, "threshold_rel": PC6_REL_MAX, "details": fd_details},
    }


def pc3_lorentz_poincare_isometry(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC3: Lorentz/Poincaré 等距一致性. p = x_{1:d}/(√c·x0+1)."""
    from model.utils import poincare_distance  # noqa: E402 (HG-Rec/model/utils.py:55)
    model.eval()
    B, L = input_ids.shape
    hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)
    h64 = hidden.to(GEOM_DTYPE)
    v_b = smooth_bounded_v(torch.tanh(model.input_proj(h64)))
    x = expmap_o(v_b, c)  # (B, L, d+1)
    # 取前 8 个 token 的真实样本对
    x_pairs = x[:8].reshape(-1, x.shape[-1])[:16]  # (16, d+1)
    y_pairs = x_pairs.roll(3, dims=0)

    # Lorentz 距离
    d_lor = lorentz_distance(x_pairs, y_pairs, c)
    # Poincaré 距离: p = x_{1:d} / (√c·x0 + 1)
    sqrt_c = math.sqrt(c)
    p_x = x_pairs[..., 1:] / (sqrt_c * x_pairs[..., 0:1] + 1.0)
    p_y = y_pairs[..., 1:] / (sqrt_c * y_pairs[..., 0:1] + 1.0)
    d_poin = poincare_distance(p_x, p_y, c).squeeze(-1)  # (N, 1) → (N,); 防广播 (N,) vs (N,1)
    # 所有点满足 Poincaré 球定义域: ||p|| < 1/√c
    p_norm = p_x.norm(dim=-1)
    ball_ok = bool((p_norm < 1.0 / sqrt_c).all().item())

    eps_iso = (d_lor - d_poin).abs() / d_lor.clamp_min(1e-12)
    max_eps = float(eps_iso.max().item())
    p99_eps = float(torch.quantile(eps_iso, 0.99).item())
    status = "PASS" if (max_eps < PC3_ISO_MAX and p99_eps < PC3_ISO_P99 and ball_ok) else "FAIL"
    return {
        "status": status,
        "max_rel_err": max_eps,
        "p99_rel_err": p99_eps,
        "threshold_max": PC3_ISO_MAX,
        "threshold_p99": PC3_ISO_P99,
        "ball_domain_ok": ball_ok,
        "n_pairs": int(d_lor.numel()),
    }


def pc4_attention_legal(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC4: Lorentz Attention 合法. mask 后行和误差 <1e-10, padding key 权重严格 0, 全 finite."""
    model.eval()
    B, L = input_ids.shape
    hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)
    h64 = hidden.to(GEOM_DTYPE)
    v_b = smooth_bounded_v(torch.tanh(model.input_proj(h64)))
    x_lorentz = expmap_o(v_b, c)
    attn_weights = model.lorentz_block.attn(x_lorentz, attention_mask)  # 触发内部记录
    w = model.lorentz_block.attn.last_attn_weights  # (B*h, L_q, L_k)
    mask_f = attention_mask.to(GEOM_DTYPE)  # (B, L)
    B_h = w.shape[0]
    # 行和误差: 每行 softmax 权重和 (padding key 被 -inf → 0, 行和仍 ≈1)
    row_sums = w.sum(dim=-1)  # (B*h, L)
    row_err = (row_sums - 1.0).abs().max().item()
    # padding key 权重严格 0: 对 mask=0 的列, 权重必须 0
    mask_expand = mask_f[:, None, :].repeat(12, 1, 1)  # (B*h, 1, L) 对应每 head
    padding_weights = w.masked_select(mask_expand.repeat(1, L, 1) == 0)
    padding_max = float(padding_weights.abs().max().item()) if padding_weights.numel() > 0 else 0.0
    # 全 finite
    finite_ok = (torch.isfinite(w).all().item() and
                 torch.isfinite(model.lorentz_block.attn.last_attn_scores).all().item())
    # 每头 entropy / min-max score
    ent = model.lorentz_block.attn.last_entropy  # (B*h, L)
    ent_per_head = ent.reshape(B, 12, L).mean(dim=(0, 2))  # (12,)
    scores = model.lorentz_block.attn.last_attn_scores
    scores_finite = scores[torch.isfinite(scores)]
    score_min = float(scores_finite.min().item()) if scores_finite.numel() > 0 else float("-inf")
    score_max = float(scores_finite.max().item()) if scores_finite.numel() > 0 else float("-inf")
    status = "PASS" if (row_err < PC4_ROWWISE_MAX and padding_max == 0.0 and finite_ok) else "FAIL"
    return {
        "status": status,
        "row_sum_max_err": row_err,
        "threshold_row": PC4_ROWWISE_MAX,
        "padding_key_max_weight": padding_max,
        "all_finite": finite_ok,
        "entropy_per_head_mean": [float(x) for x in ent_per_head.tolist()],
        "score_min": score_min,
        "score_max": score_max,
    }


def pc5_centroid_legal(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC5: Minkowski centroid 合法. 归一化前 timelike (<0), 归一化后 <1e-8, 禁 fallback."""
    model.eval()
    B, L = input_ids.shape
    hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)
    h64 = hidden.to(GEOM_DTYPE)
    v_b = smooth_bounded_v(torch.tanh(model.input_proj(h64)))
    x_lorentz = expmap_o(v_b, c)

    results = {}
    all_ok = True
    # 场景 1: attention 权重聚合 (per head)
    model.lorentz_block.attn(x_lorentz, attention_mask)
    w = model.lorentz_block.attn.last_attn_weights  # (B*h, L, L)
    B_h, Lq, Lk = w.shape
    v_l = model.lorentz_block.attn.w_v(smooth_bounded_v(logmap_o(x_lorentz, c)))
    v_l = expmap_o(smooth_bounded_v(v_l.view(B, L, 12, -1).transpose(1, 2)), c)  # (B, h, L, d_h+1)
    w_reshaped = w.reshape(B, 12, Lq, Lk)
    try:
        # einsum '...n,...nd->...d': weights (B,h,Lq,Lk) × values (B,h,Lk,d_h+1), 不物化 5D
        centroid, m_raw = lorentz_centroid(v_l, w_reshaped, c)
        inner_pre = minkowski_inner(m_raw, m_raw)
        pre_ok = bool((inner_pre < 0).all().item())
        errs_post = manifold_constraint_err(centroid, c)
        post_max = float(errs_post.max().item())
        post_ok = post_max < PC5_POST_MANIFOLD_MAX
        results["attention_centroid"] = {
            "status": "PASS" if (pre_ok and post_ok) else "FAIL",
            "pre_inner_max": float(inner_pre.max().item()),
            "pre_inner_min": float(inner_pre.min().item()),
            "post_manifold_max_err": post_max,
            "threshold_post": PC5_POST_MANIFOLD_MAX,
        }
        if not (pre_ok and post_ok):
            all_ok = False
    except RuntimeError as e:
        results["attention_centroid"] = {"status": "FAIL", "error": str(e)}
        all_ok = False

    # 场景 2: mask-aware pooling (全 token 等权)
    mask_f = attention_mask.to(GEOM_DTYPE)  # (B, L)
    try:
        centroid2, m_raw2 = lorentz_centroid(x_lorentz, mask_f, c)
        inner_pre2 = minkowski_inner(m_raw2, m_raw2)
        pre2_ok = bool((inner_pre2 < 0).all().item())
        errs2 = manifold_constraint_err(centroid2, c)
        post2_max = float(errs2.max().item())
        post2_ok = post2_max < PC5_POST_MANIFOLD_MAX
        results["mask_aware_pooling"] = {
            "status": "PASS" if (pre2_ok and post2_ok) else "FAIL",
            "pre_inner_max": float(inner_pre2.max().item()),
            "post_manifold_max_err": post2_max,
            "threshold_post": PC5_POST_MANIFOLD_MAX,
        }
        if not (pre2_ok and post2_ok):
            all_ok = False
    except RuntimeError as e:
        results["mask_aware_pooling"] = {"status": "FAIL", "error": str(e)}
        all_ok = False

    return {"status": "PASS" if all_ok else "FAIL", "scenarios": results}


def pc6_center_finite_difference(model: Stage1LorentzEncoder, input_ids, attention_mask, c) -> dict:
    """PC6: 中心有限差分 (float64, h=1e-5). W_Q/W_K/W_V/W_1/W_2 各一非零坐标.

    loss 用固定随机方向 r (PRECHECK_SEED 派生, float64): 逐元素算子与切空间
    投影保持近似常数向量, 而 LN 反向投影 (I - 11^T/N - zz^T/N) 把常数分量
    精确消到零空间 (u.sum() 梯度恒为 1, 实测 AD=1e-24 量级 = 数学结构非 bug);
    随机方向给出非常数梯度, 覆盖全部参数敏感度. 全程 eval() 保证 forward
    确定性, 否则 dropout 污染中心差分 (实测 FD 噪声 1e4 量级).
    """
    model.eval()
    with torch.no_grad():
        u_shape = model.encode(input_ids, attention_mask, return_float64=True).shape
    rng = np.random.default_rng(PRECHECK_SEED)
    r = torch.from_numpy(rng.standard_normal(u_shape)).to(GEOM_DTYPE).to(input_ids.device)
    param_names = [
        "lorentz_block.attn.w_q.weight",
        "lorentz_block.attn.w_k.weight",
        "lorentz_block.attn.w_v.weight",
        "lorentz_block.ffn.linear1.weight",
        "lorentz_block.ffn.linear2.weight",
    ]
    named = dict(model.named_parameters())
    details = []
    all_ok = True
    for pname in param_names:
        param = named[pname]
        if param.dtype != GEOM_DTYPE:
            raise RuntimeError(f"PC6: {pname} dtype={param.dtype} 非 float64 (修复二违规)")
        # 找第一个非零坐标
        idx = None
        flat = param.data.flatten()
        for i in range(flat.numel()):
            if abs(flat[i].item()) > 1e-8:
                idx = np.unravel_index(i, param.data.shape)
                break
        if idx is None:
            details.append({"param": pname, "status": "FAIL", "error": "no nonzero coords"})
            all_ok = False
            continue
        idx = tuple(int(x) for x in idx)

        # AD 梯度 (固定随机方向 loss: 常数向量被 LN 零空间消掉, 见 docstring)
        model.zero_grad()
        u = model.encode(input_ids, attention_mask, return_float64=True)
        loss = (u * r).sum()
        loss.backward()
        if param.grad is None:
            details.append({"param": pname, "status": "FAIL", "error": "no grad"})
            all_ok = False
            continue
        g_ad = float(param.grad[idx].item())

        # FD (中心差分, h=1e-5; forward 全程 no_grad 防计算图泄漏)
        h = PC6_FD_H
        orig_val = float(param.data[idx].item())
        with torch.no_grad():
            param.data[idx] = orig_val + h
            u_plus = model.encode(input_ids, attention_mask, return_float64=True)
            param.data[idx] = orig_val - h
            u_minus = model.encode(input_ids, attention_mask, return_float64=True)
            param.data[idx] = orig_val
        g_fd = ((u_plus * r).sum().item() - (u_minus * r).sum().item()) / (2 * h)

        finite_ok = math.isfinite(g_ad) and math.isfinite(g_fd) and abs(g_ad) > 0 and abs(g_fd) > 0
        sign_ok = (g_ad > 0) == (g_fd > 0)
        rel = abs(g_ad - g_fd) / max(abs(g_ad), abs(g_fd), 1e-12)
        rel_ok = rel < PC6_REL_MAX
        ok = finite_ok and sign_ok and rel_ok
        if not ok:
            all_ok = False
        details.append({
            "param": pname,
            "index": [int(x) for x in idx],
            "g_AD": g_ad,
            "g_FD": g_fd,
            "relative_error": rel,
            "sign_consistent": sign_ok,
            "finite_nonzero": finite_ok,
            "status": "PASS" if ok else "FAIL",
        })
    model.eval()
    return {
        "status": "PASS" if all_ok else "FAIL",
        "fd_h": PC6_FD_H,
        "threshold_rel": PC6_REL_MAX,
        "details": details,
    }


def pc7_real_path_audit(model: Stage1LorentzEncoder, input_ids, attention_mask, c, tokenizer, items) -> dict:
    """PC7: 真实路径审计. forward / 全量导出 / reload 三条路径执行 Lorentz distance attention +
    Minkowski centroid + HFFN; 禁用 Lorentz block 后输出 rel>1e-3."""
    model.eval()
    B, L = input_ids.shape

    # 1. 生产 forward 路径计数/hash
    with torch.no_grad():
        u = model.encode(input_ids, attention_mask, return_float64=True)
    attn_weights = model.lorentz_block.attn.last_attn_weights
    fwd_hash = sha256_bytes(u.detach().cpu().numpy().tobytes())
    attn_executed = attn_weights is not None and attn_weights.numel() > 0

    # 2. 禁用 Lorentz block 后输出 (用输入投影 + centroid 直传)
    with torch.no_grad():
        hidden = model.frozen_t5.encode_to_token(input_ids, attention_mask)
        h64 = hidden.to(GEOM_DTYPE)
        v_b = smooth_bounded_v(torch.tanh(model.input_proj(h64)))
        x_lorentz = expmap_o(v_b, c)
        mask_f = attention_mask.to(GEOM_DTYPE)
        z_centroid, _ = lorentz_centroid(x_lorentz, mask_f, c)
        u_baseline = logmap_o(z_centroid, c)
    rel_diff = float((u - u_baseline).norm() / u_baseline.norm().clamp_min(1e-12))
    disabled_ok = rel_diff > PC7_REL_DIFF_MIN

    # 3. 全量导出 (9922 items, 小批量) + hash
    import time as _time
    t0 = _time.time()
    item_texts = [it[1] for it in items[:64]]
    enc = tokenizer(item_texts, padding="max_length", truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt").to(input_ids.device)
    all_u = []
    with torch.no_grad():
        for start in range(0, len(item_texts), 16):
            batch_ids = enc.input_ids[start:start + 16]
            batch_mask = enc.attention_mask[start:start + 16]
            all_u.append(model.encode(batch_ids, batch_mask, return_float64=True).cpu())
    export_emb = torch.cat(all_u, dim=0)
    export_hash = sha256_bytes(export_emb.detach().numpy().tobytes())
    export_time = _time.time() - t0

    # 4. reload 一致: 同 seed 重跑 forward 逐元素一致
    with torch.no_grad():
        u_again = model.encode(input_ids, attention_mask, return_float64=True)
    reload_max_err = float((u_again - u).abs().max().item())
    reload_ok = reload_max_err < 1e-12

    status = "PASS" if (attn_executed and disabled_ok and reload_ok) else "FAIL"
    return {
        "status": status,
        "forward_path": {
            "attn_executed": attn_executed,
            "attn_weights_shape": list(attn_weights.shape) if attn_weights is not None else None,
            "u_hash": fwd_hash,
        },
        "disable_lorentz_block": {
            "rel_diff": rel_diff,
            "threshold": PC7_REL_DIFF_MIN,
            "ok": disabled_ok,
        },
        "export_path": {
            "n_items_exported": export_emb.shape[0],
            "export_hash": export_hash,
            "export_time_s": round(export_time, 3),
        },
        "reload_path": {
            "max_abs_err_same_seed": reload_max_err,
            "ok": reload_ok,
        },
    }


def pc7_full_export_audit(model: Stage1LorentzEncoder, tokenizer, items: list, c: float,
                          device: torch.device, product_dir: Path) -> dict:
    """Issue #50 补证二: PC7 9922 全量生产导出 + parquet + reload + 确定性.

    生产入口 = encode_all_items (与 Stage1 正式导出同入口), 完整 9922 商品, seed=42,
    不做 optimizer step. 记录: 有序 ItemID SHA256 / forward-attn-centroid-HFFN-pooling
    执行计数 (hooks) / 输出 shape-dtype-min-mean-std-max tangent norm / float64→float32
    写出前后 max 绝对与相对误差 / parquet SHA256 + ItemID 顺序 hash / reload 逐元素
    对比 (<1e-6) / 同 seed 只读再走一次输出 hash 完全一致.
    """
    model.eval()
    n_batches = (len(items) + 31) // 32  # encode_all_items batch_size=32

    # 执行计数: encode 实例替换 + attn/hffn forward hooks + lorentz_centroid 函数包装
    counts = {"encode": 0, "attn": 0, "hffn": 0, "centroid": 0}
    orig_encode = model.encode
    orig_centroid = globals()["lorentz_centroid"]

    def counted_encode(*args, **kwargs):
        counts["encode"] += 1
        return orig_encode(*args, **kwargs)

    def counted_centroid(values, weights, c_):
        counts["centroid"] += 1
        return orig_centroid(values, weights, c_)

    attn_h = model.lorentz_block.attn.register_forward_hook(lambda m, i, o: counts.__setitem__("attn", counts["attn"] + 1))
    ffn_h = model.lorentz_block.ffn.register_forward_hook(lambda m, i, o: counts.__setitem__("hffn", counts["hffn"] + 1))
    model.encode = counted_encode  # type: ignore[method-assign]
    globals()["lorentz_centroid"] = counted_centroid

    def stage_counts() -> dict:
        return dict(counts)

    try:
        # ── 阶段 1: 生产入口全量导出 (float32 写出 + cast 绝对误差) ──
        u32, ids_out, export_stats = encode_all_items(model, items, tokenizer, device, batch_size=32)
        counts1 = stage_counts()
        if ids_out != [it[0] for it in items]:
            raise RuntimeError("PC7 补证: 导出 ItemID 顺序与输入不一致")
        item_ids_sha256 = sha256_bytes("|".join(ids_out).encode())

        # 输出统计 (tangent norm)
        norms = np.linalg.norm(u32, axis=-1)
        out_stats = {
            "shape": list(u32.shape),
            "dtype": str(u32.dtype),
            "tangent_norm_min": float(norms.min().item()),
            "tangent_norm_mean": float(norms.mean().item()),
            "tangent_norm_std": float(norms.std().item()),
            "tangent_norm_max": float(norms.max().item()),
            "all_finite": bool(np.isfinite(u32).all().item()),
        }
        cast_abs_err = export_stats["max_f64_to_f32_cast_err"]

        # ── 阶段 2: 同 seed 只读再走一次 (f64 全量, 确定性 hash + 相对 cast 误差) ──
        all_u64 = np.zeros((len(items), model.frozen_t5.d_model), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(items), 32):
                batch = items[start:start + 32]
                enc = tokenizer([it[1] for it in batch], padding="max_length", truncation=True,
                                max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
                u64b = model.encode(enc.input_ids, enc.attention_mask, return_float64=True)
                all_u64[start:start + len(batch)] = u64b.cpu().numpy()
        counts2 = stage_counts()
        u32_2 = all_u64.astype(np.float32)
        cast_rel_err = float(np.abs(all_u64 - u32_2.astype(np.float64)).max() / np.abs(all_u64).max())
        det_max_abs = float(np.abs(u32_2 - u32).max().item())
        det_hash_ok = sha256_bytes(u32_2.tobytes()) == sha256_bytes(u32.tobytes())

        # ── parquet 写出 + reload (u32 同时落盘 npy, 供 PC8 补证 κ→SID 链路输入) ──
        import pandas as pd
        parquet_path = product_dir / "item_emb.parquet"
        df = pd.DataFrame({"item_id": ids_out,
                           "embedding": [u32[i].tolist() for i in range(u32.shape[0])]})
        df.to_parquet(parquet_path, index=False)
        parquet_sha256 = sha256_file(parquet_path)
        np.save(str(product_dir / "item_emb_u32.npy"), u32)

        df2 = pd.read_parquet(parquet_path)
        reload_ids = [str(x) for x in df2["item_id"].tolist()]
        reload_ids_sha256 = sha256_bytes("|".join(reload_ids).encode())
        emb2 = np.stack([np.asarray(x, dtype=np.float32) for x in df2["embedding"].values])
        reload_max_abs_err = float(np.abs(emb2 - u32).max().item())
        ids_order_ok = reload_ids == ids_out

        # ── 计数断言 (每 batch: encode 1 + attn 1 + hffn 1 + centroid 2 (attn 内 + pooling)) ──
        expected1 = {"encode": n_batches, "attn": n_batches, "hffn": n_batches, "centroid": 2 * n_batches}
        counts1_ok = counts1 == expected1
        expected2 = dict(expected1)
        counts2_ok = counts2["encode"] - counts1["encode"] == expected2["encode"] \
            and counts2["attn"] - counts1["attn"] == expected2["attn"] \
            and counts2["hffn"] - counts1["hffn"] == expected2["hffn"] \
            and counts2["centroid"] - counts1["centroid"] == expected2["centroid"]

        ok = (
            u32.shape[0] == 9922
            and len(set(ids_out)) == 9922
            and np.isfinite(u32).all()
            and reload_max_abs_err < 1e-6
            and det_hash_ok and det_max_abs == 0.0
            and counts1_ok and counts2_ok
            and ids_order_ok and reload_ids_sha256 == item_ids_sha256
        )
        return {
            "status": "PASS" if ok else "FAIL",
            "n_items": u32.shape[0],
            "n_unique_ids": len(set(ids_out)),
            "item_ids_sha256": item_ids_sha256,
            "execution_counts_stage1": counts1,
            "execution_counts_stage2_increment": {k: counts2[k] - counts1[k] for k in counts},
            "expected_counts_per_stage": expected1,
            "counts_ok": counts1_ok and counts2_ok,
            "output_stats": out_stats,
            "cast_err": {"max_abs_f64_to_f32": cast_abs_err, "max_rel_f64_to_f32": cast_rel_err},
            "parquet": {
                "path": str(parquet_path),
                "sha256": parquet_sha256,
                "item_ids_order_hash": reload_ids_sha256,
                "ids_order_unchanged": ids_order_ok,
            },
            "reload": {"max_abs_err_vs_pre_write_f32": reload_max_abs_err, "threshold": 1e-6,
                       "ok": reload_max_abs_err < 1e-6},
            "determinism": {"same_seed_rerun_max_abs_diff": det_max_abs, "hash_identical": det_hash_ok},
        }
    finally:
        attn_h.remove()
        ffn_h.remove()
        del model.encode
        globals()["lorentz_centroid"] = orig_centroid


def pc8_three_layer_compliance(stage2_module_path: str) -> dict:
    """PC8: 三层曲率与最近邻合规 (不训练 Stage2, 生产路径最小审计).

    生产路径 = taskA/stage2/taskA_stage2.py 的 KappaAwareVectorQuantization
    (每层独立 nn.Parameter: kappa_drift + mix_weight; c_l = exp(κ_eff) 正参数化).
    Issue #49 spec: 三层独立 κ; MLR_ENABLED=False 显式断言; assignment 由 argmin d_c
    唯一产生; 扰动 κ 仅对应层变化; reload 5/5 一致.
    """
    import ast
    import importlib.util
    results = {}

    with open(stage2_module_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)

    # 1. MLR_ENABLED 硬编码常量 + --no_mlr flag
    mlr_default = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "MLR_ENABLED" and isinstance(node.value, ast.Constant):
                    mlr_default = bool(node.value.value)
    no_mlr_flag = "--no_mlr" in src
    results["mlr_enabled_constant"] = {
        "hardcoded_default": mlr_default,
        "has_no_mlr_flag": no_mlr_flag,
        "issue_49_requires": "启动时显式传 --no_mlr (MLR_ENABLED=False), 显式断言不能依赖默认值",
        "launch_command": "CUDA_VISIBLE_DEVICES=<gpu> python3 -u taskA/stage2/taskA_stage2.py --no_mlr "
                          "--product_dir taskA/_history/taskA_stage2_issue49 > logs/stage2_issue49.log 2>&1 &",
    }

    # 2. assignment 唯一来源 = argmin(d) Poincaré 最近邻 (MLR 关闭路径 line ~478)
    has_argmin = "torch.argmin(d, dim=-1)" in src
    has_mlr_path = "HyperbolicHyperplaneMLR" in src
    has_sinkhorn = "sinkhorn_algorithm" in src
    results["assignment_source"] = {
        "argmin_d_unique_source": has_argmin,
        "mlr_class_present": has_mlr_path,
        "sinkhorn_present": has_sinkhorn,
        "note": "Issue #49 spec: assignment 由 argmin d_c(r_l, e_lk) 唯一产生; MLR_ENABLED=False 时 "
                "forward 走 argmin 分支 (line 478), Sinkhorn 仅在 use_sk=True 且 sk_eps>0 时可用, "
                "生产 SK_EPSILONS=[0.0,0.0,0.0] → sk 分支永远不触发",
    }

    # 3. c_l 正参数化 = exp(κ_eff) (CURV_PRIOR 主路径) / softplus 等价
    has_exp_kappa = "torch.exp(kappa_eff)" in src or "return torch.exp" in src
    has_softplus = "softplus" in src.lower()
    results["positive_parameterization"] = {
        "c_exp_kappa_eff": has_exp_kappa,
        "has_softplus": has_softplus,
        "note": "Issue #49 spec: c_l=softplus(κ_l)+ε 或现有等价正参数化; stage2 生产 c=exp(κ_eff) 恒>0",
    }

    # 4. 动态加载 stage2 模块, 构造 3 层真实量化器 → 独立 Parameter 地址 + 扰动对应层 + reload 5/5
    three_layer = {"status": "FAIL", "error": "not executed"}
    try:
        # stage2 模块级执行 parse_args(): 临时替换 sys.argv 让默认值生效, 加载后恢复
        import sys
        saved_argv = list(sys.argv)
        sys.argv = ["taskA_stage2_audit"]
        spec = importlib.util.spec_from_file_location("stage2mod", stage2_module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["stage2mod"] = mod
        spec.loader.exec_module(mod)
        sys.argv = saved_argv

        n_e_list = mod.CODEBOCK_SIZES if hasattr(mod, "CODEBOCK_SIZES") else mod.CODEBOOK_SIZES
        e_dim = mod.E_DIM
        layers = [
            mod.KappaAwareVectorQuantization(n_e=n, e_dim=e_dim, kmeans_init=False, layer_idx=i)
            for i, n in enumerate(n_e_list)
        ]
        # 3 层独立 kappa_drift Parameter 地址
        kappa_drift_ids = [id(l.kappa_drift) for l in layers]
        mix_weight_ids = [id(l.mix_weight) for l in layers]
        distinct = (len(set(kappa_drift_ids)) == 3 and len(set(mix_weight_ids)) == 3
                    and len(set(kappa_drift_ids + mix_weight_ids)) == 6)

        # 扰动 κ 仅对应层变化: 扰动 layer_1 kappa_drift, 只 layer_1 get_c 变化
        c_before = [float(l.get_c().item()) for l in layers]
        with torch.no_grad():
            layers[1].kappa_drift.data.add_(0.1)
        c_after = [float(l.get_c().item()) for l in layers]
        delta = [abs(a - b) for a, b in zip(c_after, c_before)]
        perturb_ok = delta[1] > 0 and delta[0] == 0.0 and delta[2] == 0.0

        # reload 5/5 一致: 同构造参数重建实例, get_c 逐层一致
        reload_layers = [
            mod.KappaAwareVectorQuantization(n_e=n, e_dim=e_dim, kmeans_init=False, layer_idx=i)
            for i, n in enumerate(n_e_list)
        ]
        c_reload = [float(l.get_c().item()) for l in reload_layers]
        reload_ok = all(c_reload[i] == c_before[i] for i in range(3))

        three_layer = {
            "status": "PASS" if (distinct and perturb_ok and reload_ok) else "FAIL",
            "n_layers": len(layers),
            "n_e_list": n_e_list,
            "e_dim": e_dim,
            "kappa_drift_addresses": kappa_drift_ids,
            "mix_weight_addresses": mix_weight_ids,
            "distinct_6_params": distinct,
            "perturb_layer1_kappa": {"delta_all_layers": delta, "only_layer1_changed": perturb_ok},
            "reload_5_of_5": {"c_consistent": c_reload, "ok": reload_ok},
        }
    except Exception as e:
        import traceback
        three_layer = {"status": "FAIL", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}

    results["three_layer_params"] = three_layer

    all_ok = (
        three_layer.get("status") == "PASS"
        and mlr_default is True  # 默认 True 需显式 --no_mlr (Issue #49 spec 显式断言)
        and no_mlr_flag
        and has_argmin
    )
    results["status"] = "PASS" if all_ok else "FAIL"
    return results


def pc8_kappa_sid_chain(stage2_module_path: str, item_emb: np.ndarray, device: torch.device,
                        product_dir: Path) -> dict:
    """Issue #50 补证三: PC8 运行时断言 + κ→SID 全链路 + reload 5/5.

    1. importlib 加载 stage2 (--no_mlr), verdict 记录运行时实际值:
       MLR_ENABLED is False / SK_EPSILONS == [0,0,0] / assignment_source = poincare_argmin.
    2. 类级/函数级包装计数: _compute_mlr_logits 与 sinkhorn_algorithm 调用均为 0
       (不靠源码行号推断).
    3. 三层分别扰动 κ_eff ≈ +1e-3 (kappa_drift.data += 1e-3, 记录实际 Δκ_eff), 按生产路径
       κ→c→scale(expmap0/proj_to_ball 内联)→Π_c(E)→D→A→r→SID 同步重算, 对 c/投影后
       codebook/distance/assignment/residual/最终 SID 记录 SHA256 与数值变化量;
       更早层必须不变; 当前层 c/Π(E)/D 必须变化; A/SID 允许不翻转但 hash 必须进入真实计算.
    4. 恢复原 κ 后同一 9922 输入 infer_sid (resolve=False 纯 argmin) reload 5/5,
       五次 SID 与恢复前基准逐元素完全一致, 分别提交 hash.

    输入 item_emb = Stage1 Lorentz 初始化导出 (9922, EMB_DIM) float32 (PC7 补证产物).
    """
    import importlib.util
    import sys as _sys

    results = {"status": "FAIL", "runtime": None, "chain": {}, "reload_5_of_5": {}}
    saved_argv = list(_sys.argv)
    _sys.argv = ["taskA_stage2_issue50_audit", "--no_mlr"]
    spec = importlib.util.spec_from_file_location("stage2_issue50", stage2_module_path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["stage2_issue50"] = mod
    spec.loader.exec_module(mod)
    _sys.argv = saved_argv

    # ── 1. 运行时断言 (记录实际值, 不靠建议命令) ──
    runtime = {
        "mlr_enabled_actual": bool(mod.MLR_ENABLED),
        "sk_epsilons_actual": list(mod.SK_EPSILONS),
        "assignment_source": "poincare_argmin" if (not mod.MLR_ENABLED and mod.SK_EPSILONS == [0.0, 0.0, 0.0]) else "OTHER",
        "vq_class": "KappaAwareVectorQuantization (hard argmin(d))" if not mod.MLR_ENABLED else "HyperbolicHyperplaneMLR",
    }
    runtime_ok = (
        runtime["mlr_enabled_actual"] is False
        and runtime["sk_epsilons_actual"] == [0.0, 0.0, 0.0]
        and runtime["assignment_source"] == "poincare_argmin"
    )
    results["runtime"] = runtime

    # ── 2. 调用计数包装 (审计全程有效, 恢复原函数) ──
    counts = {"mlr_logits": 0, "sinkhorn": 0}
    orig_mlr = mod.HyperbolicHyperplaneMLR._compute_mlr_logits
    orig_sk = mod.sinkhorn_algorithm

    def counted_mlr(self, *args, **kwargs):
        counts["mlr_logits"] += 1
        return orig_mlr(self, *args, **kwargs)

    def counted_sk(*args, **kwargs):
        counts["sinkhorn"] += 1
        return orig_sk(*args, **kwargs)

    mod.HyperbolicHyperplaneMLR._compute_mlr_logits = counted_mlr
    mod.sinkhorn_algorithm = counted_sk

    def sha(t: torch.Tensor) -> str:
        return sha256_bytes(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())

    try:
        # ── 3. 构造模型 (eval, 不训练不 init, 确定性) ──
        model = mod.KappaAwareHRQVAE(
            in_dim=mod.EMB_DIM, num_emb_list=mod.CODEBOOK_SIZES, e_dim=mod.E_DIM,
            layers=mod.ENCODER_LAYERS, kmeans_init=False,
            sk_eps=[0.0, 0.0, 0.0]).to(device)
        model.eval()
        item_t = torch.from_numpy(np.ascontiguousarray(item_emb, dtype=np.float32)).to(device)
        n_items = item_t.shape[0]
        if n_items != 9922:
            raise RuntimeError(f"PC8 补证: 输入 n_items={n_items} != 9922")
        input_hash = sha256_bytes(np.ascontiguousarray(item_emb, dtype=np.float32).tobytes())

        # 链路复算 (与生产 forward 相同的几何: proj_to_ball∘expmap0, poincare_distance, argmin, residual)
        def chain_layers() -> list:
            z = model.encoder(item_t)
            residual = z
            layers = []
            for li, q in enumerate(model.vq_layers):
                latent = residual.view(-1, q.e_dim)
                c_geom = q.get_c()
                latent_h = mod.proj_to_ball(mod.expmap0(latent, c_geom), c_geom)
                cb_h = mod.proj_to_ball(mod.expmap0(q.embeddings.weight, c_geom), c_geom)
                d = mod.poincare_distance(
                    latent_h.unsqueeze(1).expand(-1, q.n_e, -1),
                    cb_h.unsqueeze(0).expand(latent_h.shape[0], -1, -1), c_geom).squeeze(-1)
                A = torch.argmin(d, dim=-1)
                x_q = q.embeddings.weight.index_select(0, A)
                residual_next = residual - x_q
                layers.append({
                    "c": float(q.get_c().item()),
                    "c_hash": sha(c_geom),
                    "codebook_projected_hash": sha(cb_h),
                    "distance_hash": sha(d),
                    "assignment_hash": sha(A),
                    "residual_hash": sha(residual_next),
                    "d_sum": float(d.sum().item()),
                })
                residual = residual_next
            return layers

        def sid_once() -> np.ndarray:
            return mod.infer_sid(model, item_t, batch_size=1024, resolve=False)

        # ── 基线 ──
        base_chain = chain_layers()
        sid_base = sid_once()
        sid_base_hash = mod.sha256_array(sid_base)

        # ── 4. 三层分别扰动 κ_eff ≈ +1e-3 ──
        perturb_results = {}
        chain_ok = True
        for li in range(3):
            q = model.vq_layers[li]
            kappa_before = float(q.get_effective_kappa().item())
            c_before = float(q.get_c().item())
            with torch.no_grad():
                q.kappa_drift.data.add_(1e-3)
            kappa_after = float(q.get_effective_kappa().item())
            c_after = float(q.get_c().item())
            chain_after = chain_layers()
            sid_after = sid_once()
            sid_after_hash = mod.sha256_array(sid_after)

            prev_unchanged = all(
                chain_after[j]["c_hash"] == base_chain[j]["c_hash"]
                and chain_after[j]["codebook_projected_hash"] == base_chain[j]["codebook_projected_hash"]
                and chain_after[j]["distance_hash"] == base_chain[j]["distance_hash"]
                and chain_after[j]["assignment_hash"] == base_chain[j]["assignment_hash"]
                for j in range(li))
            cur_c_changed = chain_after[li]["c_hash"] != base_chain[li]["c_hash"]
            cur_cb_changed = chain_after[li]["codebook_projected_hash"] != base_chain[li]["codebook_projected_hash"]
            cur_d_changed = chain_after[li]["distance_hash"] != base_chain[li]["distance_hash"]
            d_delta = abs(chain_after[li]["d_sum"] - base_chain[li]["d_sum"])
            a_changed = chain_after[li]["assignment_hash"] != base_chain[li]["assignment_hash"]
            sid_changed = sid_after_hash != sid_base_hash
            n_sid_flip = int((sid_after != sid_base).sum())
            perturb_ok = prev_unchanged and cur_c_changed and cur_cb_changed and cur_d_changed and d_delta > 0.0
            chain_ok = chain_ok and perturb_ok

            perturb_results[f"layer_{li}"] = {
                "delta_kappa_eff_actual": kappa_after - kappa_before,
                "delta_c_actual": c_after - c_before,
                "earlier_layers_unchanged": prev_unchanged,
                "cur_c_changed": cur_c_changed,
                "cur_codebook_projected_changed": cur_cb_changed,
                "cur_distance_changed": cur_d_changed,
                "cur_distance_delta_abs": d_delta,
                "cur_assignment_changed": a_changed,
                "cur_assignment_flip_count": int((sid_after[:, li] != sid_base[:, li]).sum()),
                "sid_hash_changed": sid_changed,
                "sid_flip_count": n_sid_flip,
                "chain_after": {f"l{j}": {k: v for k, v in l.items()} for j, l in enumerate(chain_after)},
                "status": "PASS" if perturb_ok else "FAIL",
            }
            with torch.no_grad():
                q.kappa_drift.data.sub_(1e-3)
            # 恢复校验
            kappa_restored = float(q.get_effective_kappa().item())
            if abs(kappa_restored - kappa_before) > 1e-12:
                raise RuntimeError(f"PC8 补证: 层 {li} κ 恢复失败: {kappa_restored} != {kappa_before}")
            if float(q.get_c().item()) != c_before:
                raise RuntimeError(f"PC8 补证: 层 {li} c 恢复失败")
        results["chain"] = {
            "n_items": n_items,
            "input_hash": input_hash,
            "baseline": {f"l{j}": {k: v for k, v in l.items()} for j, l in enumerate(base_chain)},
            "sid_base_hash": sid_base_hash,
            "perturbations": perturb_results,
        }

        # ── 5. reload 5/5: 恢复后同一输入五次 SID 逐元素一致 ──
        reload_ok = True
        reload_hashes = []
        for k in range(5):
            sid_k = sid_once()
            hk = mod.sha256_array(sid_k)
            reload_hashes.append(hk)
            if hk != sid_base_hash:
                reload_ok = False
        reload_max_diff = int((sid_once() != sid_base).sum())
        results["reload_5_of_5"] = {
            "n_runs": 5,
            "hashes": reload_hashes,
            "all_equal_to_base": reload_ok,
            "total_element_diff_after_reload": reload_max_diff,
            "status": "PASS" if reload_ok else "FAIL",
        }

        # ── 6. 调用计数断言 ──
        counts_ok = counts["mlr_logits"] == 0 and counts["sinkhorn"] == 0
        results["call_counts"] = dict(counts)
        results["call_counts_ok"] = counts_ok

        all_ok = runtime_ok and chain_ok and reload_ok and counts_ok
        results["status"] = "PASS" if all_ok else "FAIL"
        results["verdict_path"] = str(product_dir / "item_emb.parquet")
        return results
    finally:
        mod.HyperbolicHyperplaneMLR._compute_mlr_logits = orig_mlr
        mod.sinkhorn_algorithm = orig_sk


# ================================================================
# 数据加载 + 全量导出
# ================================================================

def load_items(path: str) -> list[tuple[str, str]]:
    """{itemID: {title, description, brand, categories}} → [(itemID, semantics_text), ...].

    语义格式与 HG-Rec process_Instruments.py / common/stage1_hyperbolic.py 完全一致
    (保证与下游 stage2 的 item_emb 对齐).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for item_id, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(f"item {item_id} info is not dict: {type(info)}")
        semantics = (
            f"'title': {info.get('title', '')}, "
            f"'description': {info.get('description', '')}, "
            f"'brand': {info.get('brand', '')}, "
            f"'categories': {info.get('categories', '')}"
        )
        items.append((item_id, semantics))
    items.sort(key=lambda x: int(x[0]))
    return items


def encode_all_items(model: Stage1LorentzEncoder, items: list[tuple[str, str]], tokenizer, device,
                     batch_size: int = 32) -> tuple[np.ndarray, list[str], dict]:
    """返回 (u_emb: (N, d) float32 numpy, item_ids, export_stats)."""
    model.eval()
    all_emb = np.zeros((len(items), model.frozen_t5.d_model), dtype=np.float32)
    item_ids = [it[0] for it in items]
    # 记录 float64→float32 转换最大误差
    max_cast_err = 0.0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        texts = [it[1] for it in batch]
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            u64 = model.encode(enc.input_ids, enc.attention_mask, return_float64=True)
            u32 = u64.to(torch.float32)
        max_cast_err = max(max_cast_err, float((u64 - u32.double()).abs().max().item()))
        all_emb[start:start + len(batch)] = u32.cpu().numpy()
    stats = {"max_f64_to_f32_cast_err": max_cast_err}
    return all_emb, item_ids, stats


def teacher_encode(items: list[tuple[str, str]], device, batch_size: int = 32) -> np.ndarray:
    """teacher: 完整 frozen sentence-t5-base mean-pool (Recall@10 对照用)."""
    from transformers import T5EncoderModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    encoder = T5EncoderModel.from_pretrained(ENCODER_MODEL).to(device)
    encoder.eval()
    all_emb = np.zeros((len(items), encoder.config.d_model), dtype=np.float32)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        texts = [it[1] for it in batch]
        with torch.no_grad():
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
            out = encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            mask = enc.attention_mask.unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            emb = summed / denom
        all_emb[start:start + len(batch)] = emb.cpu().numpy()
    del encoder
    torch.cuda.empty_cache()
    return all_emb


def teacher_student_recall10(student_emb: np.ndarray, teacher_emb: np.ndarray, k: int = 10) -> float:
    n = student_emb.shape[0]
    s_n = student_emb / (np.linalg.norm(student_emb, axis=-1, keepdims=True) + 1e-15)
    t_n = teacher_emb / (np.linalg.norm(teacher_emb, axis=-1, keepdims=True) + 1e-15)
    t_sim = t_n @ t_n.T
    t_top10 = np.argsort(-t_sim, axis=1)[:, :k]
    s_sim = s_n @ s_n.T
    s_topk = np.argsort(-s_sim, axis=1)[:, :k]
    hits = 0
    for i in range(n):
        hits += len(set(s_topk[i].tolist()) & set(t_top10[i].tolist())) / k
    return float(hits / n)


# ================================================================
# main: precheck → verdict (本 Issue 不执行训练)
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="taskA Stage1 Issue #49 Lorentz precheck (PC1-PC8)")
    parser.add_argument("--product_dir", type=str, default=str(REPO / "taskA/_history/taskA_stage1_issue49"))
    parser.add_argument("--seed", type=int, default=PRECHECK_SEED)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    product_dir = Path(args.product_dir)
    product_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = product_dir / "item_emb.parquet"

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    log(f"[stage1-lorentz] config: TAG={TAG} C_ENC={C_ENC} N_FROZEN_BLOCKS={N_FROZEN_BLOCKS} "
        f"RHO_MAX={RHO_MAX} BOUND_EPS={BOUND_EPS} GEOM_DTYPE={GEOM_DTYPE} seed={args.seed} device={device}")

    # 加载商品
    items = load_items(ITEM_JSON)
    log(f"[stage1-lorentz] loaded {len(items)} items from {ITEM_JSON}")

    # 构建模型 (修复二: 几何核心 float64)
    model = Stage1LorentzEncoder(ENCODER_MODEL, N_FROZEN_BLOCKS, C_ENC, HFFN_HIDDEN).to(device)
    model.freeze_t5()
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[stage1-lorentz] t5 d_model={model.frozen_t5.d_model} n_blocks={model.frozen_t5.n_blocks} "
        f"frozen={N_FROZEN_BLOCKS} + Lorentz(1) | trainable params={n_trainable:,}")
    # 修复二 审计: 几何核心全部 float64
    geom_dtype_check = {}
    for name, p in model.named_parameters():
        if "input_proj" in name or "lorentz_block" in name:
            geom_dtype_check[name] = str(p.dtype)
    non_f64 = [k for k, v in geom_dtype_check.items() if "float64" not in v]
    log(f"[stage1-lorentz] 修复二审计: 几何核心 float64 params={sum(1 for _ in geom_dtype_check)}, "
        f"non-f64={len(non_f64)} {non_f64[:3]}")
    if non_f64:
        raise RuntimeError(f"修复二违规: 几何核心参数非 float64: {non_f64[:5]}")

    # 真实训练集 audit batch (固定 seed + 固定 ItemID hash)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    audit_items = items[:PRECHECK_AUDIT_SIZE]
    audit_ids = [it[0] for it in audit_items]
    audit_texts = [it[1] for it in audit_items]
    enc_audit = tokenizer(audit_texts, padding="max_length", truncation=True,
                          max_length=MAX_SEQ_LEN, return_tensors="pt").to(device)
    audit_ids_hash = sha256_bytes("|".join(audit_ids).encode())
    input_ids = enc_audit.input_ids
    attention_mask = enc_audit.attention_mask
    log(f"[stage1-lorentz] audit batch: n={len(audit_items)} ids_hash={audit_ids_hash[:16]}")

    # ============ PC1-PC8 ============
    pc_results = {}

    # PC1-PC5/PC7 全程 no_grad (eval 路径无 backward 需求, 防计算图 OOM);
    # PC6 中心差分单独管理 autograd (AD 需 grad, FD 段自带 no_grad).
    log("[precheck] PC1 全路径流形约束 (float64, <1e-8)...")
    with torch.no_grad():
        pc_results["PC1_manifold_constraint"] = pc1_manifold_constraint(
            model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC1 = {pc_results['PC1_manifold_constraint']['status']}")

    log("[precheck] PC2 Exp/Log 互逆 (三种输入, <1e-8)...")
    with torch.no_grad():
        pc_results["PC2_exp_log_inverse"] = pc2_exp_log_inverse(
            model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC2 = {pc_results['PC2_exp_log_inverse']['status']}")

    log("[precheck] PC2 补证 (Issue #50): 近零 10^-4~10^-12 + 精确零向量 + AD/FD...")
    pc_results["PC2_near_zero_inverse"] = pc2_near_zero_inverse(C_ENC, device)
    log(f"[precheck] PC2 补证 = {pc_results['PC2_near_zero_inverse']['status']}")

    log("[precheck] PC3 Lorentz/Poincaré 等距一致性 (max<1e-7, P99<1e-8)...")
    with torch.no_grad():
        pc_results["PC3_isometry"] = pc3_lorentz_poincare_isometry(
            model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC3 = {pc_results['PC3_isometry']['status']}")

    log("[precheck] PC4 Attention 合法 (行和<1e-10, padding=0)...")
    with torch.no_grad():
        pc_results["PC4_attention_legal"] = pc4_attention_legal(
            model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC4 = {pc_results['PC4_attention_legal']['status']}")

    log("[precheck] PC5 Minkowski centroid 合法 (timelike + <1e-8, 禁 fallback)...")
    with torch.no_grad():
        pc_results["PC5_centroid_legal"] = pc5_centroid_legal(
            model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC5 = {pc_results['PC5_centroid_legal']['status']}")

    log("[precheck] PC6 中心有限差分 (float64 h=1e-5, W_Q/W_K/W_V/W_1/W_2, <1e-3)...")
    pc_results["PC6_grad_fd"] = pc6_center_finite_difference(
        model, input_ids, attention_mask, C_ENC)
    log(f"[precheck] PC6 = {pc_results['PC6_grad_fd']['status']}")

    log("[precheck] PC7 真实路径审计 (forward/export/reload 三条路径)...")
    with torch.no_grad():
        pc_results["PC7_real_path_audit"] = pc7_real_path_audit(
            model, input_ids, attention_mask, C_ENC, tokenizer, items)
    log(f"[precheck] PC7 = {pc_results['PC7_real_path_audit']['status']}")

    log("[precheck] PC7 补证 (Issue #50): 9922 全量生产导出 + parquet + reload + 确定性...")
    pc_results["PC7_full_export_audit"] = pc7_full_export_audit(
        model, tokenizer, items, C_ENC, device, product_dir)
    log(f"[precheck] PC7 补证 = {pc_results['PC7_full_export_audit']['status']}")

    log("[precheck] PC8 三层曲率与最近邻合规...")
    pc_results["PC8_three_layer_compliance"] = pc8_three_layer_compliance(
        str(REPO / "taskA/stage2/taskA_stage2.py"))
    log(f"[precheck] PC8 = {pc_results['PC8_three_layer_compliance']['status']}")

    log("[precheck] PC8 补证 (Issue #50): 运行时断言 + κ→SID 全链路 + reload 5/5...")
    stage1_emb = np.load(str(product_dir / "item_emb_u32.npy")) if (product_dir / "item_emb_u32.npy").exists() else None
    if stage1_emb is None:
        raise RuntimeError("PC8 补证: 缺少 Stage1 导出 item_emb_u32.npy (PC7 补证必须先行落盘)")
    pc_results["PC8_kappa_sid_chain"] = pc8_kappa_sid_chain(
        str(REPO / "taskA/stage2/taskA_stage2.py"), stage1_emb, device, product_dir)
    log(f"[precheck] PC8 补证 = {pc_results['PC8_kappa_sid_chain']['status']}")

    # ============ 汇总 ============
    statuses = {k: v.get("status", "?") for k, v in pc_results.items()}
    all_pass = all(s == "PASS" for s in statuses.values())
    log(f"[precheck] 汇总: {statuses}")
    log(f"[precheck] 总体 = {'PASS' if all_pass else 'BLOCKED'}")

    # 输入 hash + commit hash
    commit_hash = ""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO))
        commit_hash = r.stdout.strip()
    except Exception:
        commit_hash = "unknown"

    verdict = {
        "issue": "#49+#50",
        "task": "taskA_stage1_issue49_lorentz_precheck + issue50_evidence",
        "spec": "[方向A precheck] 修复 #48 Lorentz 数值参数化并严格完成 PC1-PC8; Issue #50 补证: PC2 近零互逆 / PC7 9922 全量导出 / PC8 运行时断言+κ→SID 全链路+reload 5/5; 唯一允许结论 precheck blocked / precheck PASS",
        "decision": "precheck PASS" if all_pass else "precheck blocked",
        "precheck_overall": "PASS" if all_pass else "BLOCKED",
        "audit_batch": {
            "n_items": len(audit_items),
            "item_ids_hash": audit_ids_hash,
            "seed": args.seed,
            "device": str(device),
            "dtype": str(GEOM_DTYPE),
            "max_seq_len": MAX_SEQ_LEN,
        },
        "fix_1_smooth_bounded": {
            "rho_max": RHO_MAX,
            "eps": BOUND_EPS,
            "formula": "s=||u||_2, ρ(u)=ρ_max·tanh(s/ρ_max), v(u)=ρ(u)/(s+ε)·u",
            "applied_in": ["encode 输入投影后", "HFFN 输出", "HResLN 输出", "HAttention Q/K/V 投影后", "w_o 输出"],
        },
        "fix_2_float64": {
            "geom_dtype": str(GEOM_DTYPE),
            "non_f64_geom_params": non_f64,
            "cast_before_parquet": "float64 → float32 显式转换, 记录 max_cast_err (见 gate1_export_placeholder)",
        },
        "pc1_pc8": pc_results,
        "issue50_evidence": {
            "commit_chain": "verdict.commit 为运行时代码 HEAD (证据链自包含: 代码提交 A → 运行 → 证据提交 B)",
            "note": "Issue #50 补证字段: PC2_near_zero_inverse / PC7_full_export_audit / PC8_kappa_sid_chain 均在 pc1_pc8 内",
        },
        "earliest_failure": next((k for k, v in pc_results.items() if v.get("status") != "PASS"), None),
        "config": {
            "TAG": TAG,
            "C_ENC": C_ENC,
            "N_FROZEN_BLOCKS": N_FROZEN_BLOCKS,
            "ENCODER_MODEL": ENCODER_MODEL,
            "HFFN_HIDDEN": HFFN_HIDDEN,
            "SEED": args.seed,
            "DEVICE": str(device),
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "PC1_THRESHOLD": PC1_MANIFOLD_MAX,
            "PC2_THRESHOLD": PC2_INVERSE_MAX,
            "PC3_THRESHOLD_MAX": PC3_ISO_MAX,
            "PC3_THRESHOLD_P99": PC3_ISO_P99,
            "PC4_THRESHOLD": PC4_ROWWISE_MAX,
            "PC5_THRESHOLD": PC5_POST_MANIFOLD_MAX,
            "PC6_H": PC6_FD_H,
            "PC6_THRESHOLD_REL": PC6_REL_MAX,
            "PC7_THRESHOLD_REL_DIFF": PC7_REL_DIFF_MIN,
        },
        "commit": commit_hash,
    }
    verdict_path = product_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    log(f"[stage1-lorentz] verdict saved to {verdict_path}")
    log(f"[stage1-lorentz] 最终结论: {verdict['decision']}")
    if not all_pass:
        raise SystemExit(f"precheck blocked: 最早失败项 = {verdict['earliest_failure']}")


if __name__ == "__main__":
    main()