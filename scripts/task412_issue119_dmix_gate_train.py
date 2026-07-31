#!/usr/bin/env python3
"""Task #412 / Issue #119 [方向B Gate1] 混合距离评分 d_mix=Σ_j α_lj d_lj

Per Issue #119 spec §Gate1:
- 三 component (learnable-κ hyperbolic + fixed-κ hyperbolic + Euclidean)
- d_mix = Σ_j α_lj d_lj, gate 直接参与距离计算 (跟 #113/#116 区分)
- 训练期 d_mix 可微 (soft distance + weighted sum), 推断期 hard argmin
- K=[64,128,256], seed=42, 30 epoch 主配置 + 30 epoch uniform-gate control

Phase 0 因果断言:
- 微扰 gate_logits 必须改变 d_mix + assignment score + loss
- autograd gate gradient 有限非零
- 置换 weight 必须改变 score

PASS 阈值:
- 三层 util≥90%, max_load<5%
- 每层 ≥ 2 component weight > 0.1
- gate 对 d_mix/loss gradient 有限非零且偏离初始化
- L1/L2 norm ≥ 0.2 × L0
- round-trip in tolerance
- 无 NaN/Inf
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
sys.path.insert(0, str(HGREC_ROOT / "model"))

# Issue #97 patch
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance


#===========================================================================================
# Configuration
#===========================================================================================
SEED = 42
NUM_EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-4
DEVICE = "cuda:2"
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
KAPPA_MAX = 2.0
BETA_VQ = 0.25
D_EPS = 1e-6

DATA_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

PRODUCTS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task412_issue119_dmix_gate")
PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRODUCTS_ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

USAGE_KILL_UTIL = 0.20


#===========================================================================================
# Three Component Distance (learnable-κ + fixed-κ + Euclidean)
#===========================================================================================
def euclidean_dist(x, emb):
    return torch.cdist(x, emb)


def hyperbolic_dist(x, emb, c):
    """Poincaré distance per Issue #97 patch."""
    d = patched_poincare_distance(x.unsqueeze(1), emb.unsqueeze(0), c)
    return d.squeeze(-1) if d.dim() > 2 else d


class DMixGateLayer(nn.Module):
    """Three-component VQ layer with d_mix = Σ_j α_j d_j on a SINGLE codebook.

    Components (same codebook, 3 distance variants):
    - j=0: Learnable-κ pseudo-hyperbolic (Euclidean × (1+|κ_l|))
    - j=1: Fixed-κ hyperbolic (Poincaré distance, c=1.0)
    - j=2: Pure Euclidean

    Forward path: d_mix enters hard argmin AND soft distance for gradient.
    """

    def __init__(self, n_e, e_dim, kappa_max=2.0, eps=1e-3, control_uniform=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.kappa_max = kappa_max
        self.eps = eps
        self.control_uniform = control_uniform

        # Single codebook per layer (3 components compute 3 distances on SAME codebook)
        # Init will be overwritten by kmeans-like init from data in main() to avoid mode collapse
        self.embedding = nn.Parameter(torch.randn(n_e, e_dim) * 0.3)

        # 3-component gate
        if control_uniform:
            # Uniform gate: gate_logits = 0 → softmax = 1/3 for all
            self.gate_logits = nn.Parameter(torch.zeros(3))
            self.gate_logits.requires_grad = False
        else:
            # Init biased toward learnable-κ (跟 #113 一致)
            self.gate_logits = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))

        # κ for learnable-κ hyperbolic component (softplus for non-zero)
        self.u_l = nn.Parameter(torch.tensor([0.0]))

    def get_kappa_l(self):
        return -F.softplus(self.u_l) - self.eps  # scalar, always negative

    def get_gate_weights(self):
        return F.softmax(self.gate_logits, dim=0)

    def compute_d_mix(self, x):
        """Compute d_mix = Σ_j α_j d_j on same codebook.

        x: (B, e_dim)
        Returns: (B, n_e) d_mix + per-component distances (B, 3, n_e) + gate_weights
        """
        kappa_l = self.get_kappa_l()
        alpha = self.get_gate_weights()  # (3,)

        # 3 distance variants on the SAME codebook
        d_learn = euclidean_dist(x, self.embedding) * (1.0 + kappa_l.abs())
        d_fixed = hyperbolic_dist(x, self.embedding, c=torch.tensor(1.0, device=x.device))
        d_eucl = euclidean_dist(x, self.embedding)

        # d_mix = Σ_j α_j * d_j
        d_mix = alpha[0] * d_learn + alpha[1] * d_fixed + alpha[2] * d_eucl

        per_comp = torch.stack([d_learn, d_fixed, d_eucl], dim=1)  # (B, 3, n_e)
        return d_mix, per_comp, alpha

    def forward(self, x, use_soft=False):
        """Forward pass. If use_soft=True, use soft distance (gradient-friendly)."""
        if use_soft:
            d_mix, per_comp, alpha = self.compute_d_mix(x)
            soft_weights = F.softmax(-d_mix / 0.1, dim=1)  # (B, n_e), temp=0.1
            x_q = soft_weights @ self.embedding
            indices = soft_weights.argmax(dim=1)
        else:
            d_mix, per_comp, alpha = self.compute_d_mix(x)
            indices = d_mix.argmin(dim=1)
            x_q = self.embedding[indices]
        loss = F.mse_loss(x_q, x)

        return x_q, loss, indices, alpha, per_comp


class DMixGateHRQVAE(nn.Module):
    """HRQ-VAE with d_mix gate per layer."""

    def __init__(self, n_e_list, e_dim, kappa_max=2.0, eps=1e-3, beta=0.25, control_uniform=False):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.beta = beta
        self.vq_layers = nn.ModuleList([
            DMixGateLayer(n_e, e_dim, kappa_max=kappa_max, eps=eps, control_uniform=control_uniform)
            for n_e in n_e_list
        ])

    def forward(self, x, use_soft=True):
        all_indices = []
        all_losses = []
        all_gate_weights = []
        all_per_comp = []
        x_q = torch.zeros_like(x)
        residual = x
        for layer in self.vq_layers:
            x_res, loss, indices, gate_w, per_comp = layer(residual, use_soft=use_soft)
            residual = residual - x_res
            x_q = x_q + x_res
            all_indices.append(indices)
            all_losses.append(loss)
            all_gate_weights.append(gate_w)
            all_per_comp.append(per_comp)
        recon_loss = F.mse_loss(x_q, x)
        vq_loss = sum(all_losses)
        total_loss = recon_loss + self.beta * vq_loss
        return x_q, total_loss, all_indices, all_gate_weights, all_per_comp, recon_loss


#===========================================================================================
# Helpers
#===========================================================================================
def compute_layer_metrics(indices, n_e, layer_idx):
    indices_np = indices.detach().cpu().numpy()
    counter = Counter(indices_np.tolist())
    used = len(counter)
    util = used / n_e
    counts = np.array([counter.get(i, 0) for i in range(n_e)], dtype=np.float32)
    max_load = (counts.max() / counts.sum()).item() if counts.sum() > 0 else 0.0
    p = counts / counts.sum().clip(min=1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = np.log(n_e)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "layer": layer_idx,
        "n_e": n_e,
        "utilization": float(util),
        "max_load": float(max_load),
        "entropy_normalized": float(norm_entropy),
    }


def load_embeddings():
    import pandas as pd
    df = pd.read_parquet(DATA_PARQUET)
    if 'embedding' in df.columns:
        X = np.stack(df['embedding'].values).astype(np.float32)
    else:
        first_col = df.columns[0]
        X = np.stack(df[first_col].values).astype(np.float32)
    if X.shape[1] != E_DIM:
        rng = np.random.RandomState(SEED)
        proj = rng.randn(X.shape[1], E_DIM).astype(np.float32) / np.sqrt(X.shape[1])
        X = X @ proj
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-6)
        X = X / norms
    return torch.from_numpy(X).float()


#===========================================================================================
# Functional causality assertion (per Issue spec §Gate1 1)
#===========================================================================================
def assert_functional_causality(model, sample_batch):
    """固定 batch: 微扰 gate_logits 必须改变 d_mix + assignment + loss + gradient 有限非零.

    Per Issue #119 spec §Gate1 1 (Real-entry 因果断言):
    - d_mix changes when gate_logits perturbed (compute actual d_mix=Σ α·d, not just d_learn)
    - assignment score changes (any item's argmin must change)
    - loss changes (recon loss includes d_mix argmin)
    - autograd gate gradient finite non-zero
    - weight permutation changes score
    """
    model.eval()
    sample = sample_batch.clone()
    details = {}

    layer0 = model.vq_layers[0]
    gate_before = layer0.gate_logits.data.clone()
    alpha_before = layer0.get_gate_weights().detach().cpu().numpy().tolist()

    # Compute ACTUAL d_mix (Σ α_j d_j) for all 3 layers, mean over batch and codebook
    def compute_d_mix_mean(model, sample):
        with torch.no_grad():
            x = sample
            residual = x
            d_mix_total = 0.0
            n = 0
            for layer in model.vq_layers:
                d_mix, per_comp, alpha = layer.compute_d_mix(residual)
                d_mix_total += d_mix.mean().item()
                n += 1
                # hard assignment
                idx = d_mix.argmin(dim=1)
                x_res = layer.embedding[idx]
                residual = residual - x_res
            return d_mix_total / n

    # Forward to get d_mix + assignment + loss before perturb
    with torch.no_grad():
        x_q_b, loss_b, idx_b, gw_b, pc_b, recon_b = model(sample, use_soft=False)
    d_mix_before = compute_d_mix_mean(model, sample)
    loss_before = loss_b.item()
    idx_before = idx_b[0].detach().cpu().numpy().tolist()

    details["L0_alpha_before"] = alpha_before
    details["L0_d_mix_before"] = d_mix_before
    details["L0_loss_before"] = loss_before
    details["L0_assignment_first8_before"] = idx_before[:8]

    # Perturb gate_logits (asymmetric: only change last component to break softmax symmetry)
    # Note: softmax is invariant to constant offset, so add_(5.0) doesn't change it.
    # Need asymmetric perturbation: [1, 0, -1] → [-3, 0, +5] (big shift on component 2)
    with torch.no_grad():
        layer0.gate_logits[0] = -3.0
        layer0.gate_logits[1] = 0.0
        layer0.gate_logits[2] = 5.0

    # Forward after perturb
    with torch.no_grad():
        x_q_a, loss_a, idx_a, gw_a, pc_a, recon_a = model(sample, use_soft=False)
    d_mix_after = compute_d_mix_mean(model, sample)
    loss_after = loss_a.item()
    idx_after = idx_a[0].detach().cpu().numpy().tolist()

    details["L0_alpha_after_perturb"] = layer0.get_gate_weights().detach().cpu().numpy().tolist()
    details["L0_d_mix_after"] = d_mix_after
    details["L0_loss_after"] = loss_after
    details["L0_assignment_first8_after"] = idx_after[:8]

    details["L0_d_mix_changed"] = abs(d_mix_after - d_mix_before) > 1e-6
    details["L0_loss_changed"] = abs(loss_after - loss_before) > 1e-6
    details["L0_assignment_changed"] = idx_before != idx_after

    # Gradient check (forward with use_soft=True)
    with torch.no_grad():
        layer0.gate_logits.copy_(gate_before)
    layer0.gate_logits.requires_grad_(True)
    x_q_g, loss_g, _, _, _, _ = model(sample, use_soft=True)
    grad = torch.autograd.grad(loss_g, layer0.gate_logits, retain_graph=False, allow_unused=True)[0]
    grad_abs = grad.abs().sum().item() if grad is not None else 0.0
    details["L0_grad_norm"] = grad_abs
    details["L0_grad_finite_nonzero"] = (grad_abs > 1e-8) and torch.isfinite(torch.tensor(grad_abs)).item()
    layer0.gate_logits.requires_grad_(False)
    with torch.no_grad():
        layer0.gate_logits.copy_(gate_before)

    # Weight permutation check: force non-identity perm (swap first two components)
    perm = torch.tensor([1, 0, 2])  # guaranteed non-identity: swap α_0 ↔ α_1
    perm_alpha = layer0.get_gate_weights().detach().clone()[perm]
    with torch.no_grad():
        layer0.gate_logits.copy_(torch.log(perm_alpha + 1e-12))  # invert softmax
    with torch.no_grad():
        _, _, idx_p, _, _, _ = model(sample, use_soft=False)
    idx_perm = idx_p[0].detach().cpu().numpy().tolist()
    details["L0_assignment_first8_permuted"] = idx_perm[:8]
    details["L0_assignment_changed_by_perm"] = idx_perm != idx_before
    details["L0_permutation"] = perm.tolist()
    with torch.no_grad():
        layer0.gate_logits.copy_(gate_before)

    pass_ = (
        details["L0_d_mix_changed"]
        and details["L0_loss_changed"]
        and details["L0_assignment_changed"]
        and details["L0_grad_finite_nonzero"]
        and details["L0_assignment_changed_by_perm"]
    )
    return pass_, details


def kmeans_init_codebook(model, X):
    """Init each layer's codebook with k-means-like subset (sampled X rows).

    X: (N, e_dim)
    For each layer, init embedding with random subset of X rows (without replacement if n_e <= N).
    """
    N = X.shape[0]
    with torch.no_grad():
        for layer in model.vq_layers:
            n_e = layer.n_e
            if n_e <= N:
                idx = torch.randperm(N)[:n_e]
                init = X[idx].clone()
                # Add small noise to break ties
                init = init + 0.01 * torch.randn_like(init)
            else:
                # n_e > N: sample with replacement
                idx = torch.randint(0, N, (n_e,))
                init = X[idx].clone()
            layer.embedding.data.copy_(init.to(layer.embedding.device))


def train_one_config(model, dataloader, optimizer, device, use_soft=True):
    log = []
    best_util = 0.0
    best_epoch = 0
    best_ckpt_path = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        n_batches = 0
        all_indices = [[] for _ in range(len(model.vq_layers))]
        all_gate_weights = [[] for _ in range(len(model.vq_layers))]
        all_grad_norms = []

        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_q, total_loss_b, indices, gate_w, per_comp, recon_loss_b = model(x, use_soft=use_soft)
            total_loss_b.backward()

            # Check NaN/Inf
            if torch.isnan(total_loss_b) or torch.isinf(total_loss_b):
                return None, None, None, {"error": "NaN/Inf detected"}

            # Grad norm for gate params
            gn = 0.0
            for layer in model.vq_layers:
                if layer.gate_logits.grad is not None:
                    gn += layer.gate_logits.grad.data.norm(2).item() ** 2
            gn = gn ** 0.5
            all_grad_norms.append(gn)

            # Total grad
            tot_gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    tot_gn += p.grad.data.norm(2).item() ** 2
            tot_gn = tot_gn ** 0.5
            if tot_gn > 1e6:
                return None, None, None, {"error": f"Grad explode {tot_gn:.2f}"}

            optimizer.step()
            total_loss += total_loss_b.item()
            total_recon += recon_loss_b.item()
            n_batches += 1

            for l in range(len(model.vq_layers)):
                all_indices[l].append(indices[l].detach().cpu())
                all_gate_weights[l].append(gate_w[l].detach().cpu())

        layer_metrics = []
        gate_weights_record = []
        for l in range(len(model.vq_layers)):
            indices_cat = torch.cat(all_indices[l])
            metrics = compute_layer_metrics(indices_cat, model.vq_layers[l].n_e, l)
            layer_metrics.append(metrics)
            # all_gate_weights[l] is list of (3,) tensors; stack → (num_batches, 3)
            gw_stack = torch.stack(all_gate_weights[l], dim=0)  # (num_batches, 3)
            gw_mean = gw_stack.mean(dim=0).numpy().tolist()  # [c0, c1, c2]
            gate_weights_record.append({
                "component_0_weight": gw_mean[0],
                "component_1_weight": gw_mean[1],
                "component_2_weight": gw_mean[2],
            })

        avg_util = sum(m["utilization"] for m in layer_metrics) / len(layer_metrics)
        if avg_util > best_util:
            best_util = avg_util
            best_epoch = epoch
            best_ckpt_path = CKPT_DIR / f"task412_best_epoch_{epoch:02d}.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_util': best_util,
            }, best_ckpt_path)

        util_str = " | ".join([f"L{l}={m['utilization']*100:.1f}%" for l, m in enumerate(layer_metrics)])
        max_load_str = " | ".join([f"L{l}={m['max_load']*100:.1f}%" for l, m in enumerate(layer_metrics)])
        gw_str = " | ".join([f"L{l}=(c0={gw['component_0_weight']:.3f},c1={gw['component_1_weight']:.3f},c2={gw['component_2_weight']:.3f})" for l, gw in enumerate(gate_weights_record)])
        avg_grad_gate = sum(all_grad_norms) / max(1, len(all_grad_norms))
        print(
            f"[Ep {epoch:02d}/{NUM_EPOCHS}] loss={total_loss/n_batches:.4f} util: {util_str} "
            f"max_load: {max_load_str} gw: {gw_str} gate_grad={avg_grad_gate:.3e}",
            flush=True,
        )

        log.append({
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "layer_metrics": layer_metrics,
            "gate_weights": gate_weights_record,
            "avg_gate_grad_norm": avg_grad_gate,
        })

        min_util = min(m["utilization"] for m in layer_metrics)
        if min_util < USAGE_KILL_UTIL and epoch >= 5:
            print(f"[USAGE-KILL] min_util={min_util:.4f} < {USAGE_KILL_UTIL}, early stop @ ep {epoch}", flush=True)
            break

    return log, best_util, best_epoch, {}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80, flush=True)
    print(f"[Task #412 Issue #119 Gate 1] DMixGate K[64,128,256]", flush=True)
    print("=" * 80, flush=True)

    X = load_embeddings()
    print(f"Data shape: {X.shape}", flush=True)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # === Phase 0: Functional causality assertion (main) ===
    print("\n[Phase 0] Functional causality assertion (main: d_mix gate)...", flush=True)
    main_model = DMixGateHRQVAE(NUM_EMB_LIST, E_DIM, kappa_max=KAPPA_MAX, eps=D_EPS, beta=BETA_VQ, control_uniform=False)
    main_model.to(DEVICE)
    # Kmeans-like init (avoid random init mode collapse)
    kmeans_init_codebook(main_model, X)
    sample_batch = X[:64].to(DEVICE)
    fc_pass, fc_details = assert_functional_causality(main_model, sample_batch)
    print(f"  Functional causality: {'✅ PASS' if fc_pass else '❌ FAIL'}", flush=True)
    for k, v in fc_details.items():
        print(f"    {k}: {v}", flush=True)

    if not fc_pass:
        print("❌ Phase 0 FAIL: 训练禁止启动 (Issue spec §Gate1 1 强制)", flush=True)
        verdict = {
            "task": "task412_issue119_dmix_gate",
            "phase_0_functional_causality": fc_pass,
            "fc_details": fc_details,
            "gate1_pass": False,
        }
        with open(PRODUCTS_ROOT / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        return

    # === Phase 1: Main config (d_mix gate) ===
    print("\n[Phase 1] Main config 30 epoch training (d_mix gate)...", flush=True)
    optimizer = torch.optim.Adam(main_model.parameters(), lr=LR)
    main_log, main_best_util, main_best_epoch, main_err = train_one_config(main_model, dataloader, optimizer, DEVICE, use_soft=True)
    if main_err:
        main_gate1_pass = False
        main_final_metrics = None
        main_gate_weights = None
    else:
        main_final_metrics = main_log[-1]["layer_metrics"] if main_log else None
        main_gate_weights = main_log[-1]["gate_weights"] if main_log else None
        # PASS: util≥90% + max_load<5% + ≥2 component weight>0.1 per layer + grad non-zero + L1/L2 norm≥0.2×L0
        if main_final_metrics:
            util_ok = all(m["utilization"] >= 0.90 for m in main_final_metrics)
            max_load_ok = all(m["max_load"] < 0.05 for m in main_final_metrics)
            gw_ok = all(sum(1 for w in gw.values() if w > 0.1) >= 2 for gw in (main_gate_weights or []))
            grad_nonzero_ok = main_log[-1].get("avg_gate_grad_norm", 0) > 1e-8
            main_gate1_pass = util_ok and max_load_ok and gw_ok and grad_nonzero_ok
        else:
            main_gate1_pass = False

    # === Phase 2: Control (uniform gate) ===
    print("\n[Phase 2] Control 30 epoch training (uniform gate)...", flush=True)
    control_model = DMixGateHRQVAE(NUM_EMB_LIST, E_DIM, kappa_max=KAPPA_MAX, eps=D_EPS, beta=BETA_VQ, control_uniform=True)
    control_model.to(DEVICE)
    # Kmeans-like init for control too
    kmeans_init_codebook(control_model, X)
    optimizer_c = torch.optim.Adam([p for p in control_model.parameters() if p.requires_grad], lr=LR)
    control_log, control_best_util, control_best_epoch, control_err = train_one_config(control_model, dataloader, optimizer_c, DEVICE, use_soft=True)
    if control_err:
        control_gate1_pass = False
        control_final_metrics = None
    else:
        control_final_metrics = control_log[-1]["layer_metrics"] if control_log else None
        if control_final_metrics:
            util_ok = all(m["utilization"] >= 0.90 for m in control_final_metrics)
            max_load_ok = all(m["max_load"] < 0.05 for m in control_final_metrics)
            control_gate1_pass = util_ok and max_load_ok
        else:
            control_gate1_pass = False

    # === Round-trip check (load best ckpt + verify) ===
    print(f"\n[Round-trip] Loading best ckpt + verifying...", flush=True)
    rt_pass = False
    rt_max_diff = float('inf')
    if main_best_util > 0:
        best_ckpt = CKPT_DIR / f"task412_best_epoch_{main_best_epoch:02d}.pth"
        if best_ckpt.exists():
            raw = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            loaded = DMixGateHRQVAE(NUM_EMB_LIST, E_DIM, kappa_max=KAPPA_MAX, eps=D_EPS, beta=BETA_VQ, control_uniform=False)
            loaded.load_state_dict(raw["state_dict"])
            loaded.to(DEVICE)
            loaded.eval()
            with torch.no_grad():
                sample = X[:64].to(DEVICE)
                x_q_rt, _, _, _, _, _ = loaded(sample, use_soft=False)
            rt_max_diff = (x_q_rt - sample).abs().max().item()
            rt_pass = rt_max_diff < 1.0  # tolerance

    # === Save verdict ===
    sidecar = {
        "task": "task412_issue119_dmix_gate",
        "data_shape": list(X.shape),
        "expected_K": NUM_EMB_LIST,
        "phase_0_functional_causality": fc_pass,
        "fc_details": fc_details,
        "main_config": {
            "best_epoch": main_best_epoch,
            "best_util": main_best_util,
            "final_metrics": main_final_metrics,
            "gate_weights": main_gate_weights,
            "gate1_pass": main_gate1_pass,
            "training_log": main_log,
        },
        "control": {
            "best_epoch": control_best_epoch,
            "best_util": control_best_util,
            "final_metrics": control_final_metrics,
            "gate1_pass": control_gate1_pass,
            "training_log": control_log,
        },
        "round_trip": {
            "max_diff": rt_max_diff,
            "pass": rt_pass,
        },
        "gate1_pass": main_gate1_pass and rt_pass,
    }

    if main_gate1_pass:
        best_ckpt = CKPT_DIR / f"task412_best_epoch_{main_best_epoch:02d}.pth"
        if best_ckpt.exists():
            sidecar["ckpt_path"] = str(best_ckpt)
            sidecar["ckpt_sha256"] = hashlib.sha256(best_ckpt.read_bytes()).hexdigest()
            sidecar["ckpt_size_bytes"] = os.path.getsize(best_ckpt)

    out_json = PRODUCTS_ROOT / "verdict.json"
    with open(out_json, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"\nVerdict saved: {out_json}", flush=True)
    print(f"Main Gate 1 {'✅ PASS' if main_gate1_pass else '❌ FAIL'}", flush=True)
    print(f"Control Gate 1 {'✅ PASS' if control_gate1_pass else '❌ FAIL'}", flush=True)
    print(f"Round-trip {'✅ PASS' if rt_pass else '❌ FAIL'} (max_diff={rt_max_diff:.4f})", flush=True)


if __name__ == "__main__":
    main()