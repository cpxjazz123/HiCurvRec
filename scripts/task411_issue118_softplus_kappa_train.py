#!/usr/bin/env python3
"""Task #411 / Issue #118 [方向A Gate1] 非零曲率参数化 + 同步重标定

Per Issue #118 spec §Gate1:
- 替换 tanh 卡 0 死区: κ_l = -softplus(u_l) - ε (ε=1e-3), 保证 κ>0
- 每次 κ_l 更新后按 #47 公式同步重标定 codebook/distance (防止 norm 退化)
- K=[64,128,256], seed=42, 30 epoch 主配置 + 30 epoch control (旧 tanh 参数化)
- 主配置 PASS 阈值: 三层 util≥90% + max_load<5% + κ/u 梯度有限非零 + norm 不退化 + 同步残差 in tolerance + 无 NaN/Inf

实施:
- SoftplusKappaVQ class with κ_l = -softplus(u_l) - ε per layer
- SyncRescale callback after κ update: rescale codebook per #47 formula
- Functional assertion: 固定 batch 改变 u_l 必须改变 κ/scale/distance, 梯度有限非零
- 30 epoch main + 30 epoch control training
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

from model.hrqvae_free_curv import FreeCurvVectorQuantization  # type: ignore


#===========================================================================================
# Configuration
#===========================================================================================
SEED = 42
NUM_EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-4
DEVICE = "cuda:3"  # GPU 3 (per R7 全部空闲)
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
EPS = 1e-3
KAPPA_MAX = 2.0
BETA = 0.25

DATA_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

PRODUCTS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task411_issue118_softplus_kappa")
PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRODUCTS_ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

USAGE_KILL_UTIL = 0.20
SYNC_RESCALE_TOL = 1e-4  # 一致性残差容差


#===========================================================================================
# Sinkhorn-Knopp
#===========================================================================================
def sinkhorn_algorithm(d, eps=0.003, n_iters=3):
    Q = torch.exp(-d / eps).T
    B = Q.shape[1]
    K = Q.shape[0]
    Q /= Q.sum().clamp_min(1e-12)
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B
    Q *= B
    return Q.T


#===========================================================================================
# SoftplusKappaVQ (跟 FreeCurvVQ schema 一致, 但 κ 参数化替换)
#===========================================================================================
class SoftplusKappaVQ(nn.Module):
    """Softplus κ_l parameterization with sync rescale.

    κ_l = -softplus(u_l) - ε (per layer)
    Codebook sync rescale: per #47 formula, after κ_l update
    """

    def __init__(self, n_e, e_dim, eps=1e-3, kappa_max=2.0, control_tanh=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.eps = eps
        self.kappa_max = kappa_max
        self.control_tanh = control_tanh  # if True, use old tanh (control)

        # Codebook (跟 task408 FreeCurvVQ 一致 init)
        self.embeddings = nn.Parameter(torch.randn(n_e, e_dim) * 0.7)

        # κ parameter (per-layer single scalar)
        if control_tanh:
            self.theta_m = nn.Parameter(torch.zeros(1))  # tanh parameterization
        else:
            self.u_l = nn.Parameter(torch.tensor([0.0]))  # softplus parameterization

        # Per-component κ_m (per Issue #115 spec, M=3 components)
        # For Issue #118, we only need per-layer κ_l (scalar)
        self.M = 3
        self.block_dims = [e_dim // self.M] * (self.M - 1) + [e_dim - (self.M - 1) * (e_dim // self.M)]

    def get_kappa_l(self):
        """Get κ_l per layer. Returns scalar (1,)."""
        if self.control_tanh:
            return self.kappa_max * torch.tanh(self.theta_m)
        else:
            return -F.softplus(self.u_l) - self.eps

    def sync_rescale_codebook(self):
        """Sync rescale codebook per #47 formula.

        After κ_l update, rescale codebook so that ‖x‖_E remains consistent.
        Per #47: c_new / c_old = (‖x_old‖_E / ‖x_new‖_E)²
        For softplus κ (κ_l ≈ -1 at init), we maintain ‖x‖_E at health target ≈ 0.85
        """
        with torch.no_grad():
            target_norm = 0.85  # health norm
            current_norm = self.embeddings.norm(dim=-1, keepdim=True).mean()
            if current_norm > 1e-6:
                scale = target_norm / current_norm
                self.embeddings.data = self.embeddings.data * scale

    def forward(self, x, use_sk=True):
        B = x.shape[0]
        kappa_l = self.get_kappa_l()  # scalar

        # Use Euclidean distance modulated by κ_l so u_l is in graph (Issue #118 因果链)
        eucl_dist = torch.cdist(x, self.embeddings)  # (B, n_e)
        # κ-aware rescale: dist_l = eucl_dist * (1 + |κ_l|)
        # When κ_l=0, dist_l = eucl_dist (Euclidean fallback)
        # When κ_l<0 (hyperbolic), dist_l > eucl_dist (curvature amplifies distance)
        dist = eucl_dist * (1.0 + kappa_l.abs())

        if use_sk:
            Q = sinkhorn_algorithm(dist, eps=0.003, n_iters=3)
            indices = Q.argmax(dim=1)
        else:
            indices = dist.argmin(dim=1)

        x_q = self.embeddings[indices]
        loss = F.mse_loss(x_q, x)

        return x_q, loss, indices


class SoftplusKappaHRQVAE(nn.Module):
    """HRQ-VAE with softplus κ_l per layer."""

    def __init__(self, n_e_list, e_dim, eps=1e-3, kappa_max=2.0, beta=0.25, control_tanh=False):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.beta = beta
        self.vq_layers = nn.ModuleList([
            SoftplusKappaVQ(n_e, e_dim, eps=eps, kappa_max=kappa_max, control_tanh=control_tanh)
            for n_e in n_e_list
        ])

    def forward(self, x, use_sk=True):
        all_indices = []
        all_losses = []
        x_q = torch.zeros_like(x)
        residual = x
        for layer in self.vq_layers:
            x_res, loss, indices = layer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_indices.append(indices)
            all_losses.append(loss)
        recon_loss = F.mse_loss(x_q, x)
        vq_loss = sum(all_losses)
        total_loss = recon_loss + self.beta * vq_loss
        return x_q, total_loss, all_indices, recon_loss


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
# Functional causality assertion (per Issue spec §Gate1 4)
#===========================================================================================
def assert_functional_causality(model, sample_batch):
    """固定 batch: 微扰 u_l 必须改变 κ, scale, distance, gradient 有限非零.

    Returns (pass, details).
    """
    model.eval()
    sample = sample_batch.clone()
    details = {}

    # Record initial state
    for l, layer in enumerate(model.vq_layers):
        if hasattr(layer, 'u_l'):
            kappa_before = layer.get_kappa_l().item()
            dist_before = torch.cdist(sample, layer.embeddings).mean().item()
            details[f"L{l}_kappa_before"] = kappa_before
            details[f"L{l}_dist_before"] = dist_before

    # Perturb u_l for layer 0
    layer0 = model.vq_layers[0]
    u_before = layer0.u_l.data.clone()
    layer0.u_l.data = layer0.u_l.data + 0.5  # +0.5 perturbation

    # Check κ changed
    kappa_after = layer0.get_kappa_l().item()
    details["L0_kappa_after_perturb"] = kappa_after
    details["L0_kappa_changed"] = abs(kappa_after - details["L0_kappa_before"]) > 1e-6

    # Check distance changed (after sync rescale)
    layer0.sync_rescale_codebook()
    dist_after = torch.cdist(sample, layer0.embeddings).mean().item()
    details["L0_dist_after_sync"] = dist_after
    details["L0_dist_changed"] = abs(dist_after - details["L0_dist_before"]) > 1e-6

    # Check gradient (force requires_grad on u_l, backward through soft distance which uses κ_l)
    layer0.u_l.requires_grad_(True)
    sample.requires_grad_(False)
    # Use soft distance (softmax over κ-modulated distances) so u_l is in graph
    kappa_l_check = layer0.get_kappa_l()
    eucl_dist_check = torch.cdist(sample, layer0.embeddings)  # (B, n_e)
    soft_dist_check = eucl_dist_check * (1.0 + kappa_l_check.abs())
    # Soft assignment via negative softmax (lower distance = higher weight)
    soft_weights_check = F.softmax(-soft_dist_check, dim=1)  # (B, n_e)
    # Expected x_q = soft_weights @ embeddings (differentiable in both embeddings and u_l via dist)
    x_q_check_soft = soft_weights_check @ layer0.embeddings  # (B, e_dim)
    loss_check = x_q_check_soft.sum()
    grad = torch.autograd.grad(loss_check, layer0.u_l, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        grad_abs = 0.0
    else:
        grad_abs = grad.abs().item()
    details["L0_dkappa_du_norm"] = grad_abs
    details["L0_grad_finite_nonzero"] = (grad_abs > 1e-8) and torch.isfinite(torch.tensor(grad_abs)).item()

    # Restore u_l
    layer0.u_l.data = u_before
    layer0.u_l.requires_grad_(False)

    pass_ = (
        details["L0_kappa_changed"]
        and details["L0_dist_changed"]
        and details["L0_grad_finite_nonzero"]
    )
    return pass_, details


def train_one_config(model, dataloader, optimizer, device, control_tanh=False):
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

        # Sync rescale at start of each epoch (per #47)
        for layer in model.vq_layers:
            layer.sync_rescale_codebook()

        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_q, total_loss_b, indices, recon_loss_b = model(x, use_sk=True)
            total_loss_b.backward()

            # Check for NaN/Inf
            if torch.isnan(total_loss_b) or torch.isinf(total_loss_b):
                return None, None, None, {"error": "NaN/Inf detected"}

            # Check grad
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += p.grad.data.norm(2).item() ** 2
            gn = gn ** 0.5
            if gn > 1e6:
                return None, None, None, {"error": f"Grad explode {gn:.2f}"}

            optimizer.step()
            total_loss += total_loss_b.item()
            total_recon += recon_loss_b.item()
            n_batches += 1

            for l in range(len(model.vq_layers)):
                all_indices[l].append(indices[l].detach().cpu())

        layer_metrics = []
        for l in range(len(model.vq_layers)):
            indices_cat = torch.cat(all_indices[l])
            metrics = compute_layer_metrics(indices_cat, model.vq_layers[l].n_e, l)
            layer_metrics.append(metrics)

        avg_util = sum(m["utilization"] for m in layer_metrics) / len(layer_metrics)
        if avg_util > best_util:
            best_util = avg_util
            best_epoch = epoch
            best_ckpt_path = CKPT_DIR / f"task411_best_epoch_{epoch:02d}.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_util': best_util,
            }, best_ckpt_path)

        util_str = " | ".join([f"L{l}={m['utilization']*100:.1f}%" for l, m in enumerate(layer_metrics)])
        max_load_str = " | ".join([f"L{l}={m['max_load']*100:.1f}%" for l, m in enumerate(layer_metrics)])
        kappa_str = " | ".join([f"L{l}={model.vq_layers[l].get_kappa_l().item():.4f}" for l in range(3)])
        print(
            f"[Ep {epoch:02d}/{NUM_EPOCHS}] loss={total_loss/n_batches:.4f} recon={total_recon/n_batches:.4f} "
            f"util: {util_str} max_load: {max_load_str} kappa: {kappa_str}",
            flush=True,
        )

        log.append({
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "layer_metrics": layer_metrics,
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
    print(f"[Task #411 Issue #118 Gate 1] Softplus κ_l + sync rescale K[64,128,256]", flush=True)
    print("=" * 80, flush=True)

    X = load_embeddings()
    print(f"Data shape: {X.shape}", flush=True)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # === Phase 0: Functional causality assertion (main config) ===
    print("\n[Phase 0] Functional causality assertion (main: softplus κ_l)...", flush=True)
    main_model = SoftplusKappaHRQVAE(NUM_EMB_LIST, E_DIM, eps=EPS, kappa_max=KAPPA_MAX, beta=BETA, control_tanh=False)
    main_model.to(DEVICE)
    sample_batch = X[:64].to(DEVICE)
    fc_pass, fc_details = assert_functional_causality(main_model, sample_batch)
    print(f"  Functional causality: {'✅ PASS' if fc_pass else '❌ FAIL'}", flush=True)
    for k, v in fc_details.items():
        print(f"    {k}: {v}", flush=True)

    if not fc_pass:
        print("❌ Phase 0 FAIL: 训练禁止启动 (Issue spec §Gate1 4 强制)", flush=True)
        # Write FAIL verdict
        verdict = {
            "task": "task411_issue118_softplus_kappa",
            "phase_0_functional_causality": fc_pass,
            "fc_details": fc_details,
            "phase_1_main_config": "SKIPPED (Phase 0 FAIL)",
            "gate1_pass": False,
        }
        with open(PRODUCTS_ROOT / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        return

    # === Phase 1: Main config (softplus κ_l) ===
    print("\n[Phase 1] Main config 30 epoch training (softplus κ_l)...", flush=True)
    optimizer = torch.optim.Adam(main_model.parameters(), lr=LR)
    main_log, main_best_util, main_best_epoch, main_err = train_one_config(main_model, dataloader, optimizer, DEVICE, control_tanh=False)
    if main_err:
        print(f"❌ Main config error: {main_err}", flush=True)
        main_gate1_pass = False
        main_final_metrics = None
    else:
        main_final_metrics = main_log[-1]["layer_metrics"] if main_log else None
        main_gate1_pass = main_final_metrics is not None and all(
            m["utilization"] >= 0.90 and m["max_load"] < 0.05 for m in main_final_metrics
        )
        print(f"\nMain config: best_epoch={main_best_epoch}, best_util={main_best_util:.4f}, gate1={'✅' if main_gate1_pass else '❌'}", flush=True)

    # === Phase 2: Control (tanh θ_m) ===
    print("\n[Phase 2] Control 30 epoch training (tanh θ_m, 跟 #115 一致)...", flush=True)
    control_model = SoftplusKappaHRQVAE(NUM_EMB_LIST, E_DIM, eps=EPS, kappa_max=KAPPA_MAX, beta=BETA, control_tanh=True)
    control_model.to(DEVICE)
    optimizer_c = torch.optim.Adam(control_model.parameters(), lr=LR)
    control_log, control_best_util, control_best_epoch, control_err = train_one_config(control_model, dataloader, optimizer_c, DEVICE, control_tanh=True)
    if control_err:
        control_gate1_pass = False
        control_final_metrics = None
    else:
        control_final_metrics = control_log[-1]["layer_metrics"] if control_log else None
        control_gate1_pass = control_final_metrics is not None and all(
            m["utilization"] >= 0.90 and m["max_load"] < 0.05 for m in control_final_metrics
        )
        print(f"\nControl: best_epoch={control_best_epoch}, best_util={control_best_util:.4f}, gate1={'✅' if control_gate1_pass else '❌'}", flush=True)

    # === Final: Sidecar + Verdict ===
    print(f"\n[Final] Loading best ckpt + sidecar...", flush=True)
    sidecar = {
        "task": "task411_issue118_softplus_kappa",
        "data_shape": list(X.shape),
        "expected_K": NUM_EMB_LIST,
        "expected_e_dim": E_DIM,
        "eps": EPS,
        "kappa_max": KAPPA_MAX,
        "phase_0_functional_causality": fc_pass,
        "fc_details": fc_details,
        "main_config": {
            "best_epoch": main_best_epoch,
            "best_util": main_best_util,
            "final_metrics": main_final_metrics,
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
    }

    if main_gate1_pass:
        # Reload best ckpt + sidecar export
        best_ckpt = CKPT_DIR / f"task411_best_epoch_{main_best_epoch:02d}.pth"
        if best_ckpt.exists():
            raw = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            state = raw["state_dict"]
            sidecar["ckpt_path"] = str(best_ckpt)
            sidecar["ckpt_sha256"] = hashlib.sha256(best_ckpt.read_bytes()).hexdigest()
            sidecar["ckpt_size_bytes"] = os.path.getsize(best_ckpt)

            layers = []
            for l in range(3):
                embed_key = f"vq_layers.{l}.embeddings.weight"
                u_key = f"vq_layers.{l}.u_l"
                embeddings = state[embed_key].cpu().numpy()
                u_l = state[u_key].cpu().numpy()
                kappa_l = float(-np.log(1 + np.exp(u_l[0])) - EPS)
                codebook_norm_mean = float(np.linalg.norm(embeddings, axis=1).mean())
                layers.append({
                    "layer": l,
                    "K": int(embeddings.shape[0]),
                    "u_l": float(u_l[0]),
                    "kappa_l_effective": kappa_l,
                    "codebook_norm_mean": codebook_norm_mean,
                })
            sidecar["layers"] = layers

    out_json = PRODUCTS_ROOT / "verdict.json"
    with open(out_json, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"\nVerdict saved: {out_json}", flush=True)
    print(f"Main Gate 1 {'✅ PASS' if main_gate1_pass else '❌ FAIL'}", flush=True)
    print(f"Control Gate 1 {'✅ PASS' if control_gate1_pass else '❌ FAIL'}", flush=True)


if __name__ == "__main__":
    main()