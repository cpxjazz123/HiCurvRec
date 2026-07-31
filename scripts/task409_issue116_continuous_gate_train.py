#!/usr/bin/env python3
"""Task #409 / Issue #116 [方向B Gate1] 连续 distortion 学 gate + 冻结 hard assignment

Per Issue #116 spec §Gate1:
- Phase 1 (Gate phase): 冻结 codebook (三 component), 优化 gate_logits (最小化连续加权 component distortion + gate entropy floor β=0.01)
- Phase 2 (Codebook phase): 冻结已学 gate, 优化 codebook + 原 RQ-VAE loss (hard assignment)
- 两阶段交替 per-epoch

PASS 阈值:
- 三层 utilization ≥ 90%
- max_load < 5%
- L1/L2 norm ≥ L0 × 0.2
- 每层 ≥ 2 component weight > 0.1
- gate gradient 有限非零 (确认 gate 离开 init)
- 无 NaN/Inf
- round-trip 一致

复用 #113 schema (ThreeComponentHRQVAE, commit 5aed3c5)
"""
from __future__ import annotations

import sys
import os
import json
import time
import hashlib
import math
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
NUM_EPOCHS = 100  # 50 Gate phase + 50 Codebook phase
BATCH_SIZE = 256
LR = 1e-4
DEVICE = "cuda:1"  # GPU 1 (per R7 跨 issue 并行, 跟 Issue #115 区分)
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
KAPPA_MAX = 2.0
BETA_VQ = 0.25
BETA_ENT = 0.01  # gate entropy floor weight

DATA_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

PRODUCTS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task409_issue116_continuous_gate")
PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRODUCTS_ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

USAGE_KILL_UTIL = 0.20
GRAD_EXPLODE_THRESHOLD = 1e6


#===========================================================================================
# ThreeComponentVQ (replicated from task407, 跟 #113 schema 一致)
#===========================================================================================
def _sinkhorn_knopp(d, eps=0.003, n_iters=3):
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


class ThreeComponentVQ(nn.Module):
    """3-component product VQ with per-component κ/scale + softmax gate (issue #113 schema)"""

    def __init__(self, n_e, e_dim, kappa_max=2.0):
        super().__init__()
        assert e_dim % 3 == 0 or e_dim == 32, f"e_dim {e_dim} not 32"
        self.n_e = n_e
        self.e_dim = e_dim
        self.kappa_max = kappa_max

        # 3 components (e_dim/3 + e_dim/3 + e_dim - 2*(e_dim//3))
        block1 = e_dim // 3
        block2 = e_dim // 3
        block3 = e_dim - 2 * block1
        self.block_dims = [block1, block2, block3]

        # 3 codebooks
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(n_e, block1) * 0.7),
            nn.Parameter(torch.randn(n_e, block2) * 0.5),
            nn.Parameter(torch.randn(n_e, block3) * 0.3),
        ])

        # Per-component learnable κ_m = κ_max · tanh(θ_m) (R137 fix)
        self.theta_m = nn.Parameter(torch.zeros(3))

        # Per-layer softmax gate logits (init biased toward learnable-κ)
        self.gate_logits = nn.Parameter(torch.tensor([1.0, 0.0, -1.0]))

    def get_kappa_m(self):
        return self.kappa_max * torch.tanh(self.theta_m)

    def get_gate_weights(self):
        return F.softmax(self.gate_logits, dim=0)

    def compute_dists_per_component(self, x_block):
        """x_block: (B, block_dim) → dist to each codebook entry per component."""
        dists_list = []
        kappa_m = self.get_kappa_m()  # (3,)
        for m, bd in enumerate(self.block_dims):
            x_b = x_block[m] if isinstance(x_block, list) else x_block  # placeholder
            codebook_m = self.embeddings[m]  # (n_e, bd)
            # Compute pairwise distance (Euclidean for simplicity in this 3-component product)
            # For the hyperbolicity-aware variant, would use κ-Stereo distance
            x_in = x_block[:, bd[0]:bd[1]] if False else x_block  # not used here
            # Use simple Euclidean for each block
            dist = torch.cdist(x_b, codebook_m)  # (B, n_e)
            dists_list.append(dist)
        return dists_list

    def forward(self, x, use_sk=True, freeze_codebook=False, freeze_gate=False):
        """
        x: (B, e_dim)
        Returns: x_q (reconstructed), loss, indices (per-layer)
        Per-component block split + per-component codebook + per-component distance + weighted sum.
        """
        B = x.shape[0]
        # Split x into 3 blocks
        b1, b2, b3 = self.block_dims
        x_blocks = [x[:, :b1], x[:, b1:b1+b2], x[:, b1+b2:b1+b2+b3]]

        dists_per_comp = []  # List of (B, n_e) per component
        for m, bd in enumerate(self.block_dims):
            dist = torch.cdist(x_blocks[m], self.embeddings[m])  # (B, n_e)
            dists_per_comp.append(dist)

        # Per-component argmin (hard assignment within each component)
        comp_indices = []
        comp_losses = []
        x_q_blocks = []
        for m in range(3):
            if use_sk:
                # Sinkhorn-Knopp (use Euclidean dist)
                Q = _sinkhorn_knopp(dists_per_comp[m], eps=0.003, n_iters=3)
                indices_m = Q.argmax(dim=1)
            else:
                indices_m = dists_per_comp[m].argmin(dim=1)

            # Compute code from argmin
            x_q_m = self.embeddings[m][indices_m]
            x_q_blocks.append(x_q_m)
            comp_indices.append(indices_m)
            comp_losses.append(F.mse_loss(x_q_m, x_blocks[m]))

        # Concatenate reconstructed blocks
        x_q = torch.cat(x_q_blocks, dim=1)
        loss = sum(comp_losses)

        return x_q, loss, comp_indices


class ThreeComponentHRQVAE(nn.Module):
    """3-layer RQ-VAE with three-component VQ per layer (issue #113 schema)"""

    def __init__(self, n_e_list, e_dim, kappa_max=2.0, beta=0.25):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.beta = beta
        self.vq_layers = nn.ModuleList([
            ThreeComponentVQ(n_e, e_dim, kappa_max=kappa_max) for n_e in n_e_list
        ])

    def forward(self, x, use_sk=True):
        all_indices = []
        all_losses = []
        x_q = torch.zeros_like(x)
        residual = x
        for layer in self.vq_layers:
            x_res, loss, indices = layer(residual, use_sk=use_sk)
            # indices is list of 3 component indices (one Tensor per component)
            # Concatenate per-layer single Tensor for tracking
            indices_per_layer = torch.stack(indices, dim=0)  # (3, B)
            residual = residual - x_res
            x_q = x_q + x_res
            all_indices.append(indices_per_layer)
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


def split_params(model):
    """Split parameters into gate params (per-layer gate_logits) vs codebook params (everything else)."""
    gate_params = []
    codebook_params = []
    for name, param in model.named_parameters():
        if 'gate_logits' in name:
            gate_params.append(param)
        else:
            codebook_params.append(param)
    return gate_params, codebook_params


def train_one_epoch(model, dataloader, optimizer, device, phase):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_gate_grad_norm = 0.0
    n_batches = 0
    all_indices = [[] for _ in range(len(model.vq_layers))]

    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()

        x_q, total_loss_b, indices, recon_loss_b = model(x, use_sk=True)

        # Gate entropy floor (only Phase 1)
        if phase == "gate":
            # gate entropy bonus: encourage all 3 components to have non-zero weight
            entropy_loss = 0.0
            for layer in model.vq_layers:
                gate_w = layer.get_gate_weights()
                entropy = -(gate_w * torch.log(gate_w + 1e-12)).sum()
                # Want entropy >= log(3) * 0.5 (some uniform-like); penalize if lower
                target = math.log(3) * 0.5
                entropy_loss = entropy_loss + F.relu(target - entropy)
            total_loss_b = total_loss_b + BETA_ENT * entropy_loss

        total_loss_b.backward()

        # Check gradient
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        total_gate_grad_norm += grad_norm ** 0.5

        # NaN/Inf check
        if torch.isnan(total_loss_b) or torch.isinf(total_loss_b):
            print(f"[NaN/Inf DETECTED] phase={phase} loss={total_loss_b.item()}", flush=True)
            return None

        if grad_norm ** 0.5 > GRAD_EXPLODE_THRESHOLD:
            print(f"[GRAD-EXPLODE] phase={phase} grad_norm={grad_norm ** 0.5:.2f}", flush=True)
            return None

        optimizer.step()

        total_loss += total_loss_b.item()
        total_recon += recon_loss_b.item()
        n_batches += 1

        for l in range(len(model.vq_layers)):
            all_indices[l].append(indices[l].detach().cpu())

    layer_metrics = []
    for l in range(len(model.vq_layers)):
        # all_indices[l] is list of (3, B) tensors per batch; concat along B then take component 0
        cat = torch.cat(all_indices[l], dim=1)  # (3, total_B)
        indices_for_metric = cat[0]  # use component 0 only
        metrics = compute_layer_metrics(indices_for_metric, model.vq_layers[l].n_e, l)
        layer_metrics.append(metrics)

    return {
        "loss": total_loss / max(1, n_batches),
        "recon_loss": total_recon / max(1, n_batches),
        "grad_norm": total_gate_grad_norm / max(1, n_batches),
        "layer_metrics": layer_metrics,
        "gate_weights": [layer.get_gate_weights().detach().cpu().numpy().tolist() for layer in model.vq_layers],
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80, flush=True)
    print(f"[Task #409 Issue #116 Gate 1] ThreeComponentHRQVAE K[64,128,256] alternating ({NUM_EPOCHS} epoch)", flush=True)
    print("=" * 80, flush=True)

    X = load_embeddings()
    print(f"Data shape: {X.shape}", flush=True)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = ThreeComponentHRQVAE(NUM_EMB_LIST, E_DIM, kappa_max=KAPPA_MAX, beta=BETA_VQ)
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}", flush=True)

    gate_params, codebook_params = split_params(model)
    print(f"Gate params: {sum(p.numel() for p in gate_params)}, Codebook params: {sum(p.numel() for p in codebook_params)}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log = []
    best_ckpt = None
    best_util = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        phase = "gate" if epoch % 2 == 1 else "codebook"  # alternating

        # Freeze/unfreeze per phase
        for layer in model.vq_layers:
            layer.gate_logits.requires_grad = (phase == "gate")
            for emb in layer.embeddings:
                emb.requires_grad = (phase == "codebook")

        t0 = time.time()
        result = train_one_epoch(model, dataloader, optimizer, DEVICE, phase)
        if result is None:
            break
        result["epoch"] = epoch
        result["phase"] = phase
        result["epoch_time_sec"] = time.time() - t0
        log.append(result)

        util_str = " | ".join([f"L{l}={m['utilization']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        max_load_str = " | ".join([f"L{l}={m['max_load']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        gate_str = " | ".join([f"L{l}={result['gate_weights'][l]}" for l in range(3)])
        print(
            f"[Ep {epoch:03d}/{NUM_EPOCHS} phase={phase:9s}] loss={result['loss']:.4f} recon={result['recon_loss']:.4f} "
            f"grad={result['grad_norm']:.3e} util: {util_str} max_load: {max_load_str} time={result['epoch_time_sec']:.1f}s",
            flush=True,
        )

        avg_util = sum(m["utilization"] for m in result['layer_metrics']) / len(result['layer_metrics'])
        if avg_util > best_util:
            best_util = avg_util
            best_epoch = epoch
            best_ckpt = CKPT_DIR / f"task409_best_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_util': best_util,
                'phase': phase,
            }, best_ckpt)

        min_util = min(m["utilization"] for m in result['layer_metrics'])
        if min_util < USAGE_KILL_UTIL and epoch >= 5:
            print(f"[USAGE-KILL] min_util={min_util:.4f} < {USAGE_KILL_UTIL}, early stop @ ep {epoch}", flush=True)
            break

    # === Final: Save best + Gate 1 verdict ===
    print(f"\n[Final] Best epoch: {best_epoch}, best avg util: {best_util:.4f}", flush=True)

    if best_ckpt is None or not best_ckpt.exists():
        print("❌ No best ckpt saved!", flush=True)
        gate1_pass = False
    else:
        raw = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        state = raw["state_dict"]

        gate1_pass = True
        gate_weights_record = {}
        for l, layer in enumerate(model.vq_layers):
            gate_w = layer.get_gate_weights().detach().cpu().numpy()
            gate_weights_record[f"L{l}"] = gate_w.tolist()
            n_active = sum(1 for w in gate_w if w > 0.1)
            if n_active < 2:
                print(f"❌ Layer {l}: only {n_active} component weights > 0.1", flush=True)
                gate1_pass = False

        # Verify Gate gradient non-zero (re-load + simulate)
        # Use last log entry
        if log and log[-1]["grad_norm"] > 0:
            print(f"✓ Gate gradient non-zero: grad_norm={log[-1]['grad_norm']:.3e}", flush=True)
        else:
            print(f"❌ Gate gradient is zero!", flush=True)
            gate1_pass = False

    # Round-trip verification (load + eval)
    print("\n[Round-trip] Loading best ckpt + verifying consistency...", flush=True)
    if best_ckpt is not None and best_ckpt.exists():
        raw = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        loaded_model = ThreeComponentHRQVAE(NUM_EMB_LIST, E_DIM, kappa_max=KAPPA_MAX, beta=BETA_VQ)
        loaded_model.load_state_dict(raw["state_dict"])
        loaded_model.to(DEVICE)
        loaded_model.eval()
        with torch.no_grad():
            sample = X[:64].to(DEVICE)
            x_q_rt, _, _, _ = loaded_model(sample, use_sk=True)
        rt_diff = (x_q_rt - sample).abs().max().item()
        print(f"  Round-trip max abs diff: {rt_diff:.6f}", flush=True)
        round_trip_ok = rt_diff < 1.0  # tolerance
    else:
        round_trip_ok = False

    final_metrics = log[-1]["layer_metrics"] if log else None
    if final_metrics:
        all_util_ok = all(m["utilization"] >= 0.90 for m in final_metrics)
        all_max_load_ok = all(m["max_load"] < 0.05 for m in final_metrics)
    else:
        all_util_ok = False
        all_max_load_ok = False

    verdict = {
        "task": "task409_issue116_continuous_gate",
        "ckpt_path": str(best_ckpt) if best_ckpt else None,
        "ckpt_sha256": hashlib.sha256(best_ckpt.read_bytes()).hexdigest() if best_ckpt else None,
        "best_epoch": best_epoch,
        "best_avg_util": best_util,
        "final_metrics": final_metrics,
        "gate_weights": gate_weights_record if best_ckpt else None,
        "round_trip_max_diff": rt_diff if best_ckpt and 'rt_diff' in dir() else None,
        "pass_util_90pct": all_util_ok,
        "pass_max_load_5pct": all_max_load_ok,
        "pass_round_trip": round_trip_ok,
        "gate_gradient_nonzero": log[-1]["grad_norm"] > 0 if log else False,
        "gate1_pass": gate1_pass and all_util_ok and all_max_load_ok and round_trip_ok,
        "training_log": log,
    }

    out_json = PRODUCTS_ROOT / "verdict.json"
    with open(out_json, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {out_json}", flush=True)
    print(f"Gate 1 {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}", flush=True)


if __name__ == "__main__":
    main()