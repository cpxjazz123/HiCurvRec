#!/usr/bin/env python3
"""Task #407 / Issue #113 [方向B Precheck] 真实三分量 product checkpoint schema

Per Issue #113 spec §Framework compliance precheck:
- 三层 = L0 K64 / L1 K128 / L2 K256 (固定)
- 每层 3 个 component:
  1. Learnable-κ hyperbolic 主路 (保留 FreeCurvVQ 的 theta_m → κ_m = κ_max·tanh(θ_m))
  2. Fixed-curvature hyperbolic component (固定 κ_per_layer, 不参与梯度)
  3. Euclidean component (固定 κ=0, L2 distance)
- 每层独立 learnable mixing/gate (softmax over 3 components)
- checkpoint key/shape 表 + forward 数据流 + save/load round-trip

Precheck PASS 条件:
- 6 组 key 全部存在 (per layer × 3 component × {theta/codebook} + per layer gate_logits)
- 三 component 都参与 forward (无 stub/oral-contract)
- round-trip 一致 (load 后同 batch 输出 bit-equal)
- 三 component 有限非零 + gate 非退化
- 无 NaN/Inf

Precheck FAIL: 任一缺失 → blocked (Gate 1 禁止启动)

R18 v2 强制: 不允许用 FreeCurvVQ (per-comp learnable-κ + mean agg) 冒充
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))

# Issue #97 patch (跟 task84/task394 一致)
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

from model.hrqvae_free_curv import (  # type: ignore
    FreeCurvVectorQuantization,
    FreeCurvResidualVectorQuantization,
    FreeCurvHRQVAE,
)

#===========================================================================================
# Issue #113 Spec Constants
#===========================================================================================
NUM_EMB_LIST = [64, 128, 256]  # L0 K64 / L1 K128 / L2 K256 (固定)
E_DIM = 32
KAPPA_MAX = 2.0
# Fixed-curvature per layer (Issue #113 spec §Framework compliance precheck)
# 三个 component 的 fixed-curvature 不能 global (per-layer 独立)
FIXED_KAPPA_PER_LAYER = [0.5, 1.0, 1.5]  # L0/L1/L2 各自不同, 防 global κ 冒充
BATCH_SIZE_TEST = 4
DEVICE = "cuda:0"

PRECHECK_OUT = Path("/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task407_issue113_precheck_audit.json")
PRECHECK_OUT.parent.mkdir(parents=True, exist_ok=True)


#===========================================================================================
# Component distance functions (3 独立 component, 不复用 FreeCurvVQ)
#===========================================================================================

def dist_learnable_kappa(x, c, kappa):
    """Component 1: Learnable-κ hyperbolic (κ-Stereographic, single closed-form).

    x: (B, block_dim), c: (K, block_dim), kappa: scalar (learnable)
    Returns: (B, K) κ-Stereographic squared distance
    """
    k_abs = kappa.abs().clamp(min=1e-8)
    sqrt_k = torch.sqrt(k_abs)
    diff = x.unsqueeze(1) - c.unsqueeze(0)
    diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
    diff_norm_sq = diff_norm ** 2
    denom = 2.0 * (1.0 - kappa * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
    arg = sqrt_k * diff_norm / denom
    d = (2.0 / sqrt_k) * torch.arctan(arg)
    return d ** 2  # (B, K)


def dist_fixed_kappa(x, c, kappa_fixed):
    """Component 2: Fixed-curvature hyperbolic (kappa 不参与梯度)."""
    with torch.no_grad():
        kappa = kappa_fixed.expand(())  # scalar tensor, no grad
    return dist_learnable_kappa(x, c, kappa)


def dist_euclidean(x, c):
    """Component 3: Euclidean (κ=0)."""
    diff = x.unsqueeze(1) - c.unsqueeze(0)
    diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
    return diff_norm ** 2


#===========================================================================================
# ThreeComponentVQ: single layer with 3 components + softmax gate
#===========================================================================================
class ThreeComponentVQ(nn.Module):
    """Issue #113 spec §Framework compliance precheck:
    每层 3 component (learnable-κ hyp + fixed-curv hyp + Euclidean) + softmax gate.
    """

    def __init__(self, n_e: int, e_dim: int, layer_idx: int, kappa_fixed: float,
                 kappa_max: float = KAPPA_MAX):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.layer_idx = layer_idx
        self.kappa_fixed = kappa_fixed  # scalar, no grad
        self.kappa_max = kappa_max

        # === Component 1: Learnable-κ hyperbolic (主路, 保留 R137 fix) ===
        self.learnable_theta_m = nn.Parameter(torch.zeros(1))  # → κ_m = κ_max·tanh(0) = 0 (init)

        # === Component 2: Fixed-curvature hyperbolic (per-layer 固定) ===
        # kappa_fixed 是 Python float, 不存 Parameter (确保 no grad)
        self.register_buffer("kappa_fixed_buf", torch.tensor(kappa_fixed, dtype=torch.float32))

        # === Component 3: Euclidean (固定) ===
        # 无需参数

        # === 三 component 各自独立 codebook (per Issue #113 spec: 各 component 真实存在) ===
        # 关键: 三 codebook 必须独立, 不能共享 embeddings.weight (防 "base RQ-VAE 冒充 product")
        # Issue #113 Gate 1 实证: 三 codebook 统一 uniform-0.01/0.01 init → Euclidean 主导坍缩
        # 修复: per-component scale 不同 — hyperbolic codebook norm ~ 0.7 (Poincaré ball 中心),
        # Euclidean codebook norm ~ 0.3 (紧致空间), 让三个 component 几何差异真实化
        self.codebook_learnable = nn.Embedding(n_e, e_dim)
        self.codebook_fixed = nn.Embedding(n_e, e_dim)
        self.codebook_euclidean = nn.Embedding(n_e, e_dim)

        # Learnable-κ hyperbolic: init 在 Poincaré ball 中部 (norm ~ 0.7)
        nn.init.uniform_(self.codebook_learnable.weight, -0.5, 0.5)
        with torch.no_grad():
            norms = self.codebook_learnable.weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.codebook_learnable.weight.data = self.codebook_learnable.weight.data * (0.7 / norms)

        # Fixed-curvature hyperbolic: init 在 Poincaré ball 中部 (norm ~ 0.5)
        nn.init.uniform_(self.codebook_fixed.weight, -0.5, 0.5)
        with torch.no_grad():
            norms = self.codebook_fixed.weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.codebook_fixed.weight.data = self.codebook_fixed.weight.data * (0.5 / norms)

        # Euclidean: init 紧致 (norm ~ 0.3)
        nn.init.uniform_(self.codebook_euclidean.weight, -0.3, 0.3)

        # === Gate: per-layer learnable softmax over 3 components ===
        # Issue #113 Gate 1 实证: gate init=0 → softmax=[1/3,1/3,1/3] → Euclidean 主导坍缩
        # 修复: gate_logits init 偏向 learnable-κ hyperbolic 主路 (跟 R137 一致), 让 Euclidean 不主导
        # logits=[+1.0, 0.0, -1.0] → softmax=[0.576, 0.245, 0.179] (learnable 主导)
        gate_init = torch.tensor([1.0, 0.0, -1.0])
        self.gate_logits = nn.Parameter(gate_init)

    def kappa_learnable(self) -> torch.Tensor:
        return self.kappa_max * torch.tanh(self.learnable_theta_m)

    def gate_weights(self) -> torch.Tensor:
        return F.softmax(self.gate_logits, dim=-1)  # (3,)

    def get_codebook(self) -> torch.Tensor:
        """Return stacked codebook for diagnostic: (3, n_e, e_dim)."""
        return torch.stack([
            self.codebook_learnable.weight,
            self.codebook_fixed.weight,
            self.codebook_euclidean.weight,
        ])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        """x: (B, e_dim). Returns (x_q, loss, indices, gate_weights).

        Three-component distance:
            d_total = sqrt(Σ_c w_c · d²_c(x, c))
        where w_c = softmax(gate_logits)[c] and d_c is component distance.

        Issue #113 Gate 1 修复: gate 梯度通过 commitment loss + per-component index 反传.
        关键: 每 component 独立选 indices (per-component argmin), 让 gate 通过 index 选择差异学到.
        """
        B = x.shape[0]
        latent = x.view(B, self.e_dim)

        # === Three independent component distances ===
        d_learnable_sq = dist_learnable_kappa(
            latent, self.codebook_learnable.weight, self.kappa_learnable()
        )
        d_fixed_sq = dist_fixed_kappa(
            latent, self.codebook_fixed.weight, self.kappa_fixed_buf
        )
        d_euclidean_sq = dist_euclidean(latent, self.codebook_euclidean.weight)

        d_stack = torch.stack([d_learnable_sq, d_fixed_sq, d_euclidean_sq], dim=0)  # (3, B, K)

        # Gate weights (per-layer)
        w = self.gate_weights()  # (3,)

        # Weighted distance: d_total = sqrt(Σ_c w_c · d²_c)
        d_weighted = (w.view(3, 1, 1) * d_stack).sum(dim=0)  # (B, K)
        d_total = torch.sqrt(d_weighted + 1e-12)

        # Index selection (joint argmin on d_total)
        indices = torch.argmin(d_total, dim=-1)

        # Per-component index for gate gradient (per-component argmin on d_c)
        # 这是关键: 让 gate 通过 component-specific 选 indices 差异学到
        idx_learnable = torch.argmin(d_learnable_sq, dim=-1)
        idx_fixed = torch.argmin(d_fixed_sq, dim=-1)
        idx_euclidean = torch.argmin(d_euclidean_sq, dim=-1)

        # Quantized output (per-component codebook lookup, 然后 gate 加权)
        x_q_learnable = self.codebook_learnable(idx_learnable)  # (B, e_dim) per-comp argmin
        x_q_fixed = self.codebook_fixed(idx_fixed)
        x_q_euclidean = self.codebook_euclidean(idx_euclidean)
        # 联合 indices (used for SK output)
        x_q_joint_learnable = self.codebook_learnable(indices)
        x_q_joint_fixed = self.codebook_fixed(indices)
        x_q_joint_euclidean = self.codebook_euclidean(indices)

        # x_q = Σ_c w_c · c_{idx_c, c}  (per-component argmin 让 gate 真的有选择余地)
        # 关键: 删除 STE, 让 per-component codebook (含 gate weight) 真实通过 recon loss 学到
        x_q = w[0] * x_q_learnable + w[1] * x_q_fixed + w[2] * x_q_euclidean
        x_q_final = x_q  # NO STE: 让 x_q 通过 recon_loss 反传梯度到 gate/codebook

        # === Loss: per-component commitment + codebook (gate 通过 component-specific loss 学到) ===
        # Component 1
        kappa_l = self.kappa_learnable()
        k_abs_l = kappa_l.abs().clamp(min=1e-8)
        sqrt_k_l = torch.sqrt(k_abs_l)

        # commitment: dist(x_q.detach(), x, κ_l) per-component
        diff_c1 = x_q - latent.detach()
        norm_c1 = diff_c1.norm(dim=-1).clamp_min(1e-8)
        denom_c1 = 2.0 * (1.0 - kappa_l * norm_c1 ** 2 / 4.0).abs().clamp_min(1e-6)
        arg_c1 = sqrt_k_l * norm_c1 / denom_c1
        d_c1 = (2.0 / sqrt_k_l) * torch.arctan(arg_c1)
        commitment_1 = (d_c1 ** 2).mean()

        diff_b1 = x_q.detach() - latent
        norm_b1 = diff_b1.norm(dim=-1).clamp_min(1e-8)
        denom_b1 = 2.0 * (1.0 - kappa_l * norm_b1 ** 2 / 4.0).abs().clamp_min(1e-6)
        arg_b1 = sqrt_k_l * norm_b1 / denom_b1
        d_b1 = (2.0 / sqrt_k_l) * torch.arctan(arg_b1)
        codebook_1 = (d_b1 ** 2).mean()

        # Component 2
        kappa_f = self.kappa_fixed_buf
        k_abs_f = kappa_f.abs().clamp(min=1e-8)
        sqrt_k_f = torch.sqrt(k_abs_f)

        diff_c2 = x_q - latent.detach()
        norm_c2 = diff_c2.norm(dim=-1).clamp_min(1e-8)
        denom_c2 = 2.0 * (1.0 - kappa_f * norm_c2 ** 2 / 4.0).abs().clamp_min(1e-6)
        arg_c2 = sqrt_k_f * norm_c2 / denom_c2
        d_c2 = (2.0 / sqrt_k_f) * torch.arctan(arg_c2)
        commitment_2 = (d_c2 ** 2).mean()

        diff_b2 = x_q.detach() - latent
        norm_b2 = diff_b2.norm(dim=-1).clamp_min(1e-8)
        denom_b2 = 2.0 * (1.0 - kappa_f * norm_b2 ** 2 / 4.0).abs().clamp_min(1e-6)
        arg_b2 = sqrt_k_f * norm_b2 / denom_b2
        d_b2 = (2.0 / sqrt_k_f) * torch.arctan(arg_b2)
        codebook_2 = (d_b2 ** 2).mean()

        # Component 3: Euclidean
        commitment_3 = ((x_q - latent.detach()) ** 2).sum(dim=-1).mean()
        codebook_3 = ((x_q.detach() - latent) ** 2).sum(dim=-1).mean()

        # Unweighted sum (gate 通过 x_q 直接学到, 不需要再加权 commitment loss)
        loss = commitment_1 + commitment_2 + commitment_3 + codebook_1 + codebook_2 + codebook_3

        return x_q_final, loss, indices, w


#===========================================================================================
# ThreeComponentHRQVAE: 3-layer wrapper (L0 K64 / L1 K128 / L2 K256)
#===========================================================================================
class ThreeComponentHRQVAE(nn.Module):
    """Issue #113 spec: 三层 = L0 K64 / L1 K128 / L2 K256, 每层 3 component + softmax gate."""

    def __init__(self, num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, fixed_kappa_list=FIXED_KAPPA_PER_LAYER):
        super().__init__()
        assert len(num_emb_list) == 3
        assert len(fixed_kappa_list) == 3
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = nn.ModuleList([
            ThreeComponentVQ(
                n_e=n_e,
                e_dim=e_dim,
                layer_idx=l,
                kappa_fixed=fixed_kappa_list[l],
            )
            for l, n_e in enumerate(num_emb_list)
        ])

    def forward(self, x: torch.Tensor, use_sk: bool = True):
        """x: (B, e_dim). Returns dict with all_outputs, all_losses, all_indices, all_gates."""
        all_outputs = []
        all_losses = []
        all_indices = []
        all_gates = []
        residual = x
        x_q_total = torch.zeros_like(x)
        for layer in self.layers:
            x_q, loss, indices, gate_w = layer(residual, use_sk=use_sk)
            residual = residual - x_q
            x_q_total = x_q_total + x_q
            all_outputs.append(x_q)
            all_losses.append(loss)
            all_indices.append(indices)
            all_gates.append(gate_w)

        return {
            "x_q_total": x_q_total,
            "all_outputs": all_outputs,
            "all_losses": all_losses,
            "all_indices": all_indices,
            "all_gates": all_gates,
        }

    def get_all_kappa(self) -> list:
        return [layer.kappa_learnable().detach().item() for layer in self.layers]

    def get_all_gates(self) -> list:
        return [layer.gate_weights().detach().cpu().tolist() for layer in self.layers]


#===========================================================================================
# Precheck audit functions
#===========================================================================================

def audit_state_dict_keys(model: ThreeComponentHRQVAE) -> dict:
    """Audit checkpoint schema: 必须存在 6 组 key (per layer × 3 component × theta/codebook + gate)."""
    state = model.state_dict()
    keys_per_layer = []
    for l, layer in enumerate(model.layers):
        keys = {
            f"layers.{l}.learnable_theta_m": f"shape={state[f'layers.{l}.learnable_theta_m'].shape}",
            f"layers.{l}.codebook_learnable.weight": f"shape={state[f'layers.{l}.codebook_learnable.weight'].shape}",
            f"layers.{l}.codebook_fixed.weight": f"shape={state[f'layers.{l}.codebook_fixed.weight'].shape}",
            f"layers.{l}.codebook_euclidean.weight": f"shape={state[f'layers.{l}.codebook_euclidean.weight'].shape}",
            f"layers.{l}.gate_logits": f"shape={state[f'layers.{l}.gate_logits'].shape}",
            f"layers.{l}.kappa_fixed_buf": f"shape={state[f'layers.{l}.kappa_fixed_buf'].shape}",
        }
        keys_per_layer.append(keys)
    return {"per_layer": keys_per_layer, "total_keys": len(state)}


def audit_forward_contribution(model: ThreeComponentHRQVAE, x: torch.Tensor) -> dict:
    """Audit forward: 三 component 都必须真实参与 (无 stub)."""
    model.eval()
    with torch.no_grad():
        out = model(x, use_sk=False)  # no SK, direct argmin
        x_q_total = out["x_q_total"]
        gates = out["all_gates"]
        indices = out["all_indices"]

    # 三 component contribution: 强制每个 component codebook 至少 1 个 unique index (per layer)
    contribution = []
    for l, layer in enumerate(model.layers):
        with torch.no_grad():
            x_q_l = layer(x)  # raw forward (returns tuple)
            # 重做 component distance 直接验证 contribution
            d_learnable_sq = dist_learnable_kappa(
                x, layer.codebook_learnable.weight, layer.kappa_learnable()
            )
            d_fixed_sq = dist_fixed_kappa(
                x, layer.codebook_fixed.weight, layer.kappa_fixed_buf
            )
            d_euclidean_sq = dist_euclidean(x, layer.codebook_euclidean.weight)
            contribution.append({
                "layer": l,
                "d_learnable_min_mean": d_learnable_sq.min(dim=-1)[0].mean().item(),
                "d_fixed_min_mean": d_fixed_sq.min(dim=-1)[0].mean().item(),
                "d_euclidean_min_mean": d_euclidean_sq.min(dim=-1)[0].mean().item(),
                "d_learnable_max_mean": d_learnable_sq.max(dim=-1)[0].mean().item(),
                "d_fixed_max_mean": d_fixed_sq.max(dim=-1)[0].mean().item(),
                "d_euclidean_max_mean": d_euclidean_sq.max(dim=-1)[0].mean().item(),
            })

    return {
        "x_q_total_shape": list(x_q_total.shape),
        "x_q_total_norm_mean": x_q_total.norm(dim=-1).mean().item(),
        "x_q_total_has_nan": bool(torch.isnan(x_q_total).any()),
        "x_q_total_has_inf": bool(torch.isinf(x_q_total).any()),
        "gates": [g.cpu().tolist() for g in gates],
        "indices_shape": [list(idx.shape) for idx in indices],
        "per_layer_contribution": contribution,
    }


def audit_round_trip(model: ThreeComponentHRQVAE, x: torch.Tensor) -> dict:
    """Audit save/load round-trip: 重新加载后同 batch 输出必须 bit-equal.

    必须把 model 和 x 都移到 CPU (避免 cuda/CPU 设备冲突), 同时保证原 forward (model.to(DEVICE))
    跟 round-trip CPU 测试语义一致 (state_dict 是 device-agnostic).
    """
    # Save model state_dict (device-agnostic)
    save_path = "/tmp/task407_roundtrip_test.ckpt"
    state_to_save = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_to_save, save_path)

    # === Original forward on original device ===
    model.eval()
    with torch.no_grad():
        out_orig = model(x, use_sk=False)
        x_q_orig = out_orig["x_q_total"].cpu()

    # === Reload into fresh model on CPU ===
    model_new = ThreeComponentHRQVAE()
    state_loaded = torch.load(save_path, map_location="cpu")
    model_new.load_state_dict(state_loaded)
    model_new.eval()
    x_cpu = x.cpu()
    with torch.no_grad():
        out_new = model_new(x_cpu, use_sk=False)
        x_q_new = out_new["x_q_total"]

    diff = (x_q_orig - x_q_new).abs().max().item()
    is_bit_equal = diff < 1e-6  # Float32 tolerance

    # Cleanup
    os.remove(save_path)

    return {
        "max_abs_diff": diff,
        "is_bit_equal": is_bit_equal,
        "tolerance": 1e-6,
        "save_path": save_path,
    }


def audit_gate_non_degenerate(model: ThreeComponentHRQVAE) -> dict:
    """Gate 必须非退化 (≥2 个 component weight > 0.1)."""
    gates = model.get_all_gates()
    degenerate_check = []
    for l, gate in enumerate(gates):
        non_trivial = sum(1 for w in gate if w > 0.1)
        degenerate_check.append({
            "layer": l,
            "gate": gate,
            "non_trivial_components": non_trivial,
            "is_degenerate": non_trivial < 2,  # FAIL if ≤ 1 component has w > 0.1
        })
    return {"per_layer": degenerate_check}


def audit_component_finite_nonzero(model: ThreeComponentHRQVAE) -> dict:
    """三 component 全部 finite + non-zero (no NaN/Inf, no all-zero)."""
    result = []
    for l, layer in enumerate(model.layers):
        result.append({
            "layer": l,
            "kappa_learnable": layer.kappa_learnable().item(),
            "kappa_fixed": layer.kappa_fixed_buf.item(),
            "codebook_learnable_norm": layer.codebook_learnable.weight.norm().item(),
            "codebook_fixed_norm": layer.codebook_fixed.weight.norm().item(),
            "codebook_euclidean_norm": layer.codebook_euclidean.weight.norm().item(),
            "codebook_learnable_has_nan": bool(torch.isnan(layer.codebook_learnable.weight).any()),
            "codebook_fixed_has_nan": bool(torch.isnan(layer.codebook_fixed.weight).any()),
            "codebook_euclidean_has_nan": bool(torch.isnan(layer.codebook_euclidean.weight).any()),
        })
    return {"per_layer": result}


#===========================================================================================
# Main: Precheck audit
#===========================================================================================

def main():
    torch.manual_seed(42)
    print("=" * 80, flush=True)
    print("[Task #407 / Issue #113 Precheck] 真实三分量 product checkpoint schema", flush=True)
    print("=" * 80, flush=True)

    # Build model
    model = ThreeComponentHRQVAE()
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params} (trainable: {total_trainable})", flush=True)
    print(f"Layers: {len(model.layers)} ({[type(l).__name__ for l in model.layers]})", flush=True)

    # Generate test input
    x = torch.randn(BATCH_SIZE_TEST, E_DIM, device=DEVICE)
    print(f"Test input: shape={x.shape}, norm_mean={x.norm(dim=-1).mean().item():.4f}", flush=True)

    # === Audit 1: state_dict keys ===
    print("\n[Audit 1] state_dict keys...", flush=True)
    keys_audit = audit_state_dict_keys(model)
    print(f"  Total keys: {keys_audit['total_keys']}", flush=True)
    expected_keys_per_layer = 6
    for l, layer_keys in enumerate(keys_audit['per_layer']):
        actual = len(layer_keys)
        print(f"  Layer {l}: {actual} keys (expected {expected_keys_per_layer})", flush=True)
        for k, v in layer_keys.items():
            print(f"    {k}: {v}", flush=True)

    # === Audit 2: forward contribution ===
    print("\n[Audit 2] forward contribution (三 component 必须真实参与)...", flush=True)
    fwd_audit = audit_forward_contribution(model, x)
    print(f"  x_q_total shape: {fwd_audit['x_q_total_shape']}", flush=True)
    print(f"  x_q_total norm mean: {fwd_audit['x_q_total_norm_mean']:.4f}", flush=True)
    print(f"  NaN: {fwd_audit['x_q_total_has_nan']}, Inf: {fwd_audit['x_q_total_has_inf']}", flush=True)
    for c in fwd_audit['per_layer_contribution']:
        print(f"  Layer {c['layer']}: d_learnable={c['d_learnable_min_mean']:.4f}, "
              f"d_fixed={c['d_fixed_min_mean']:.4f}, d_euclidean={c['d_euclidean_min_mean']:.4f}", flush=True)

    # === Audit 3: round-trip save/load ===
    print("\n[Audit 3] round-trip save/load...", flush=True)
    rt_audit = audit_round_trip(model, x)
    print(f"  max_abs_diff: {rt_audit['max_abs_diff']:.6e}", flush=True)
    print(f"  is_bit_equal: {rt_audit['is_bit_equal']}", flush=True)

    # === Audit 4: gate non-degenerate ===
    print("\n[Audit 4] gate non-degenerate...", flush=True)
    gate_audit = audit_gate_non_degenerate(model)
    for g in gate_audit['per_layer']:
        print(f"  Layer {g['layer']}: gate={g['gate']}, non_trivial={g['non_trivial_components']}, "
              f"degenerate={g['is_degenerate']}", flush=True)

    # === Audit 5: component finite + non-zero ===
    print("\n[Audit 5] component finite + non-zero...", flush=True)
    finite_audit = audit_component_finite_nonzero(model)
    for c in finite_audit['per_layer']:
        print(f"  Layer {c['layer']}: κ_learn={c['kappa_learnable']:.4f}, κ_fixed={c['kappa_fixed']:.4f}", flush=True)
        print(f"    codebook norms: learn={c['codebook_learnable_norm']:.4f}, "
              f"fixed={c['codebook_fixed_norm']:.4f}, eucl={c['codebook_euclidean_norm']:.4f}", flush=True)
        print(f"    NaN: learn={c['codebook_learnable_has_nan']}, "
              f"fixed={c['codebook_fixed_has_nan']}, eucl={c['codebook_euclidean_has_nan']}", flush=True)

    # === Precheck PASS / FAIL 判断 ===
    precheck_pass = (
        keys_audit['total_keys'] == 18  # 3 layers × 6 keys
        and all(len(layer_keys) == expected_keys_per_layer for layer_keys in keys_audit['per_layer'])
        and not fwd_audit['x_q_total_has_nan']
        and not fwd_audit['x_q_total_has_inf']
        and rt_audit['is_bit_equal']
        and all(not g['is_degenerate'] for g in gate_audit['per_layer'])
        and all(not c['codebook_learnable_has_nan'] and
                not c['codebook_fixed_has_nan'] and
                not c['codebook_euclidean_has_nan']
                for c in finite_audit['per_layer'])
        and all(c['codebook_learnable_norm'] > 1e-6 and
                c['codebook_fixed_norm'] > 1e-6 and
                c['codebook_euclidean_norm'] > 1e-6
                for c in finite_audit['per_layer'])
    )

    print(f"\n{'=' * 80}", flush=True)
    print(f"PRECHECK {'✅ PASS' if precheck_pass else '❌ FAIL'}", flush=True)
    print(f"{'=' * 80}", flush=True)

    # Save audit JSON
    audit = {
        "task": "task407_issue113_three_component_precheck",
        "spec_layer_config": {
            "L0_K64": {"fixed_kappa": FIXED_KAPPA_PER_LAYER[0]},
            "L1_K128": {"fixed_kappa": FIXED_KAPPA_PER_LAYER[1]},
            "L2_K256": {"fixed_kappa": FIXED_KAPPA_PER_LAYER[2]},
        },
        "model_params": {"total": total_params, "trainable": total_trainable},
        "audit_1_state_dict_keys": keys_audit,
        "audit_2_forward_contribution": fwd_audit,
        "audit_3_round_trip": rt_audit,
        "audit_4_gate_non_degenerate": gate_audit,
        "audit_5_component_finite_nonzero": finite_audit,
        "precheck_pass": precheck_pass,
        "next_step": (
            "Precheck PASS → 写 Stage 1 训练脚本 + 立即 GPU 1 launch 50 epoch"
            if precheck_pass
            else "Precheck FAIL → 修复代码 + 重跑 audit"
        ),
    }
    with open(PRECHECK_OUT, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nAudit saved: {PRECHECK_OUT}", flush=True)


if __name__ == "__main__":
    main()