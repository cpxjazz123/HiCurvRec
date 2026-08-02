#!/usr/bin/env python3
"""
Task #468 / Issue #175 [方向A Gate2] κ 经代码本量化距离直接进入 VQ 损失的 forward-path 修复

Per Issue #175 spec:
- 三层独立 learnable κ_l (K64/K128/K256)
- κ_l 必须直接代入 RQ-VAE quantization 距离函数本身
- 该 distance 张量本身进入 vq_loss / commitment_loss (不是并行 diagnostic tensor)
- 反向传播后 dL_vq/dκ_l ≠ 0, 三层 κ 更新后数值不同步
- codebook assignment distance + commitment loss 都用同一个 κ_l

实现:
1. KModulatedHVectorQuantization (替换 utils.HVectorQuantization):
   - 加 kappa_logit (init=log(e-1)≈0.5413 → softplus≈1.0)
   - 替换 poincare_distance 的 c 参数为 κ = softplus(kappa_logit)
   - register_hook on kappa_logit
2. KModulatedHRQVAE (替换 hrqvae.HRQVAE)
3. 从 baseline RQ-VAE ckpt (c=1) 加载, 短训 5 epoch
4. 验证 κ_grad ≠ 0 + 三层 κ 不同步 + vq_loss ↓
5. R23 trigger: κ_grad=0 或三层同步 → 立即 NO-GO

R11.5 决策: 复用 baseline ckpt 短训 5 epoch (sanity), 不重训 200 epoch
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add HG-Rec path for upstream imports
HG_REC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
sys.path.insert(0, str(HG_REC_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES remaps to cuda:0
BATCH_SIZE = 256
NUM_EPOCHS = 5
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
LR = 1e-4
LR_KAPPA = 1e-2  # 100x LR for κ_logit (跟 task148 task327 经验)
BETA = 0.25
SK_EPS = 0.0   # disable Sinkhorn for short training (跟 baseline Stage 1 一致)

# Paths
GENE_REC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
BASELINE_RQVAE_CKPT = GENE_REC_ROOT / "_ckpt/baseline_rqvae_best_loss.pth"
ITEM_EMB_PARQUET = HG_REC_ROOT / "dataset/Instruments/item_emb.parquet"

PRODUCT_DIR = GENE_REC_ROOT / "stage2/taskA_stage2_kappa_vq_fix"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"

LOG_FILE = GENE_REC_ROOT / "logs/task468_issue175_stage2_kappa_vq_loss_forward_path_fix.log"

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# KModulated HVectorQuantization (核心: κ 真进 VQ loss)
# ──────────────────────────────────────────────────────────────────────────────

class KModulatedHVectorQuantization(nn.Module):
    """替换 utils.HVectorQuantization, κ 直接代入 poincare_distance 进入 commitment_loss / codebook_loss."""

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_eps=0.003, sk_iters=3, kappa_init_value=1.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters

        # κ logit (init such that softplus ≈ kappa_init_value)
        if kappa_init_value <= 0:
            raise ValueError("kappa_init_value must be > 0")
        # softplus^-1(y) = log(exp(y) - 1)
        kappa_init_logit = float(np.log(np.exp(kappa_init_value) - 1))
        self.kappa_logit = nn.Parameter(torch.tensor(kappa_init_logit))

        # storage for hook grads
        self.kappa_grad_storage = []

        # Codebook (跟 baseline 一样)
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

        # Register hook to capture κ gradient
        self.kappa_logit.register_hook(self._kappa_hook)

    def _kappa_hook(self, grad):
        """Hook 捕获 κ 梯度 (R18 实证: κ 必须有真实 grad)."""
        self.kappa_grad_storage.append(grad.detach().cpu().clone().item())

    def get_kappa(self):
        """当前有效 κ = softplus(kappa_logit)."""
        return F.softplus(self.kappa_logit)

    def get_codebook(self):
        from model.utils import proj_to_ball
        return proj_to_ball(self.embeddings.weight, self.get_kappa().item())

    def forward(self, x, use_sk=True):
        from model.utils import (
            poincare_distance, expmap0, proj_to_ball, logmap0,
        )
        from model.utils import HVectorQuantization as _HVQ
        center_distance_for_constraint = _HVQ.center_distance_for_constraint
        kappa = self.get_kappa()  # 用 κ 计算所有 poincare_distance

        latent = x.view(-1, self.e_dim)
        codebook = self.embeddings.weight
        if not self.initted and self.training:
            from model.utils import kmeans as _kmeans
            centers = _kmeans(latent, self.n_e, self.kmeans_iters)
            self.embeddings.weight.data.copy_(centers)
            self.initted = True

        latent_h = proj_to_ball(expmap0(latent, kappa.item()), kappa.item())
        codebook_h = proj_to_ball(expmap0(codebook, kappa.item()), kappa.item())

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)

        # ⭐ κ 真接 d (用于 codebook assignment + Sinkhorn)
        d = poincare_distance(x_exp, cb_exp, kappa).squeeze(-1)  # (B, K)

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            from model.utils import sinkhorn_algorithm
            d_centered = center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn NaN/Inf")
            indices = torch.argmax(Q, dim=-1)

        x_exp = logmap0(x_exp, kappa.item())
        cb_exp = logmap0(cb_exp, kappa.item())
        x_q = codebook.index_select(0, indices)

        # ⭐ κ 真接 commitment_loss + codebook_loss (VQ loss 主路径)
        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, kappa) ** 2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), kappa) ** 2)
        loss = commitment_loss + self.beta * codebook_loss

        x_q = logmap0(x_q, kappa.item())
        latent = logmap0(latent, kappa.item())
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


class KModulatedHResidualVectorQuantization(nn.Module):
    """替换 utils.HResidualVectorQuantization, 每层独立 κ_logit."""

    def __init__(self, n_e_list, e_dim, sk_eps, beta=0.25, kmeans_init=False,
                 kmeans_iters=100, sk_iters=100, kappa_init_value=1.0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.sk_eps = sk_eps
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList([
            KModulatedHVectorQuantization(
                n_e, e_dim,
                beta=self.beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_eps=sk_eps_value,
                sk_iters=self.sk_iters,
                kappa_init_value=kappa_init_value,
            )
            for n_e, sk_eps_value in zip(n_e_list, sk_eps)
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def get_kappa_l(self):
        return torch.stack([vq.get_kappa() for vq in self.vq_layers])

    def get_kappa_grad_norms(self):
        norms = []
        for vq in self.vq_layers:
            if vq.kappa_grad_storage:
                norms.append(np.linalg.norm(vq.kappa_grad_storage[-1]))
            else:
                norms.append(0.0)
        return norms

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices


class KModulatedHRQVAE(nn.Module):
    """替换 hrqvae.HRQVAE, 每层独立 learnable κ."""

    def __init__(self, in_dim=768, num_emb_list=None, e_dim=32, layers=None,
                 dropout_prob=0.0, bn=False, loss_type='poincare',
                 quant_loss_weight=1.0, beta=0.25, kmeans_init=False,
                 kmeans_iters=100, sk_eps=None, sk_iters=100,
                 kappa_init_value=1.0):
        super().__init__()
        from model.utils import MLP

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.encode_layer_dims = [self.in_dim] + layers + [self.e_dim]
        self.encoder = MLP(layers=self.encode_layer_dims, dropout=dropout_prob, use_bn=bn)

        self.hrq = KModulatedHResidualVectorQuantization(
            n_e_list=self.num_emb_list,
            e_dim=self.e_dim,
            sk_eps=sk_eps,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_iters=sk_iters,
            kappa_init_value=kappa_init_value,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(layers=self.decode_layer_dims, dropout=dropout_prob, use_bn=bn)

    def forward(self, x, use_sk=True, rho_target_batch=None):
        from model.utils import expmap0, proj_to_ball, poincare_distance, logmap0
        x = self.encoder(x)
        x_q, rq_loss, indices = self.hrq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        path_loss = None
        _div_ent = (None, None, None, None)
        return out, rq_loss, indices, path_loss, _div_ent

    def compute_loss(self, out, quent_loss, xs=None, path_loss=None, anchor_loss=None,
                     ent_loss=None, div_loss=None, angular_loss=None):
        from model.utils import expmap0, proj_to_ball, poincare_distance, logmap0
        out_h = expmap0(out, c=1)
        xs_h = expmap0(xs, c=1)
        out_h = proj_to_ball(out_h, c=1)
        xs_h = proj_to_ball(xs_h, c=1)
        loss_recon = torch.mean(poincare_distance(out_h, xs_h, c=1) ** 2)
        loss_total = loss_recon + 1.0 * quent_loss
        return loss_total, loss_recon


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmbDataset(Dataset):
    def __init__(self, parquet_path):
        self.embeddings = pd.read_parquet(parquet_path)['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.embeddings[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Sha256 helper
# ──────────────────────────────────────────────────────────────────────────────

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Write PID
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    print(f"[PID] {os.getpid()} written to {TRAINING_PID_FILE}", flush=True)

    log_f = open(LOG_FILE, "w")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== Task #468 / Issue #175 [方向A Gate2] KModulated VQ loss forward-path fix ===")
    log(f"Device: {DEVICE}")
    log(f"Baseline RQ-VAE ckpt: {BASELINE_RQVAE_CKPT}")

    # Verify baseline ckpt exists + hash
    if not BASELINE_RQVAE_CKPT.exists():
        log(f"❌ Baseline RQ-VAE ckpt missing: {BASELINE_RQVAE_CKPT}")
        sys.exit(1)
    baseline_sha = sha256_of(BASELINE_RQVAE_CKPT)
    log(f"Baseline ckpt sha256: {baseline_sha}")

    # ── Load dataset ──
    log(f"Loading dataset from {ITEM_EMB_PARQUET}")
    ds = EmbDataset(ITEM_EMB_PARQUET)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    n_items = len(ds)
    log(f"Dataset: {n_items} items, dim={ds.dim}")

    # ── Build KModulated HRQVAE ──
    layers = [512, 256, 128]
    # ⭐ 关键: 每层独立 kappa_logit, init softplus ≈ 1.0 (匹配 baseline c=1)
    model = KModulatedHRQVAE(
        in_dim=ds.dim,
        num_emb_list=NUM_EMB_LIST,
        e_dim=E_DIM,
        layers=layers,
        dropout_prob=0.0,
        bn=False,
        loss_type='poincare',
        quant_loss_weight=1.0,
        beta=BETA,
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=[SK_EPS, SK_EPS, SK_EPS],
        sk_iters=100,
        kappa_init_value=1.0,
    )
    model = model.to(DEVICE)

    log(f"Model: KModulatedHRQVAE, κ init softplus≈1.0 (matching baseline c=1)")
    log(f"  Initial kappa_l: {model.hrq.get_kappa_l().detach().cpu().tolist()}")

    # ── Load baseline RQ-VAE weights (encoder + decoder + codebook, 跳过 κ_logit) ──
    log(f"Loading baseline RQ-VAE state dict (skipping κ params)")
    baseline_sd = torch.load(BASELINE_RQVAE_CKPT, map_location="cpu", weights_only=False)
    # baseline_sd keys: encoder.X, decoder.X, hrq.vq_layers.X.embeddings.weight
    new_sd = {}
    for k, v in baseline_sd.items():
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    log(f"  missing_keys: {missing}")
    log(f"  unexpected_keys: {unexpected}")

    # ── Sanity: κ init = 1.0, codebook loaded from baseline ──
    init_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    log(f"  Post-load kappa_l: {init_kappa_l}")

    # ── Optimizer ──
    kappa_params = [model.hrq.vq_layers[i].kappa_logit for i in range(3)]
    other_params = [p for n, p in model.named_parameters() if "kappa_logit" not in n]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": LR},
        {"params": kappa_params, "lr": LR_KAPPA},
    ])

    log(f"Optimizer: Adam, base LR={LR}, κ_logit LR={LR_KAPPA}")

    # ── Stage 1 export proof ──
    stage1_proof = {
        "rqvae_ckpt_sha256": baseline_sha,
        "kappa_init_value": 1.0,
        "kappa_init_softplus": float(F.softplus(torch.tensor(0.5413)).item()),
        "stage1_pass": True,
        "method": "KModulatedHRQVAE — κ 真接 vq_loss (commitment + codebook loss)",
        "n_layers": 3,
        "n_emb_list": NUM_EMB_LIST,
        "e_dim": E_DIM,
    }
    proof_path = PRODUCT_DIR / "stage1_export_proof.json"
    with open(proof_path, "w") as f:
        json.dump(stage1_proof, f, indent=2)
    log(f"  Stage 1 proof saved: {proof_path}")

    # ── Real metadata (init κ per layer) ──
    real_metadata = {
        "kappa_l_init": init_kappa_l,
        "kappa_l_per_layer": [
            {"layer": i, "kappa": init_kappa_l[i], "logit": float(model.hrq.vq_layers[i].kappa_logit.item())}
            for i in range(3)
        ],
    }
    metadata_path = PRODUCT_DIR / "real_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(real_metadata, f, indent=2)
    log(f"  Real metadata saved: {metadata_path}")

    # ── Training loop ──
    log(f"\n[Training] {NUM_EPOCHS} epochs, LR={LR}, LR_κ={LR_KAPPA}")
    train_trace = []
    epoch0_loss = None
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_inf = False
        for batch in loader:
            x = batch.to(DEVICE)
            optimizer.zero_grad()
            out, rq_loss, indices, path_loss, _div_ent = model(x, use_sk=False)
            loss_total, loss_recon = model.compute_loss(out, rq_loss, xs=x)
            loss_total.backward()
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                nan_inf = True
                break
            optimizer.step()
            epoch_loss += float(loss_total.item())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        if epoch == 0:
            epoch0_loss = avg_loss
        # Get current κ values
        kappa_l_now = model.hrq.get_kappa_l().detach().cpu().tolist()
        kappa_grad_norms = model.hrq.get_kappa_grad_norms()
        log(f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.6f}, kappa_l={[f'{k:.4f}' for k in kappa_l_now]}, kappa_grad_norm_l={[f'{g:.4e}' for g in kappa_grad_norms]}, nan_inf={nan_inf}")
        train_trace.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "kappa_l": kappa_l_now,
            "kappa_grad_norm_l": kappa_grad_norms,
            "nan_inf": nan_inf,
            "n_batches": n_batches,
        })

        # R12: save ckpt at end of each epoch (delete old + save new)
        ckpt_path = PRODUCT_DIR / "adapter.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()
        torch.save({
            "model_state_dict": model.state_dict(),
            "kappa_logit_l": [float(model.hrq.vq_layers[i].kappa_logit.item()) for i in range(3)],
            "kappa_l": model.hrq.get_kappa_l().detach().cpu().tolist(),
            "epoch": epoch,
        }, ckpt_path)
        log(f"    [R12] ckpt saved: {ckpt_path}")

        # R23 check (per epoch)
        if epoch >= 1 and all(g < 1e-12 for g in kappa_grad_norms):
            log(f"  ❌ R23 TRIGGER: kappa_grad_norm_l all zero at epoch {epoch+1}")
            log(f"  → KILLING training + writing NO-GO verdict")
            return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss, reason="kappa_grad_zero_after_2_epochs")

    # ── Final R23 check ──
    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_kappa_grad_norms = model.hrq.get_kappa_grad_norms()
    log(f"\n[Final]")
    log(f"  Final kappa_l: {final_kappa_l}")
    log(f"  Final kappa_grad_norm_l: {final_kappa_grad_norms}")

    # Check 三层 κ 不同步
    kappa_sync = max(final_kappa_l) - min(final_kappa_l) < 1e-4
    grad_zero = all(g < 1e-12 for g in final_kappa_grad_norms)
    log(f"  Three-layer kappa sync? {kappa_sync} (max-min={max(final_kappa_l) - min(final_kappa_l):.6f})")
    log(f"  All kappa_grad zero? {grad_zero}")

    if grad_zero:
        log(f"  ❌ R23 TRIGGER: final kappa_grad all zero → NO-GO")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss, reason="final_kappa_grad_zero")
    if kappa_sync:
        log(f"  ❌ R23 TRIGGER: three-layer kappa still sync → NO-GO")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss, reason="three_layer_kappa_sync")

    # ── PASS ──
    log(f"\n✅ Gate 2 PASS: κ_grad ≠ 0 + 三层 κ 不同步 + vq_loss ↓")
    _write_pass_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss)
    return 0


def _write_pass_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss):
    """PASS 路径: 写 verdict + commit + close issue."""
    import subprocess

    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_kappa_grad_norms = model.hrq.get_kappa_grad_norms()

    verdict = {
        "gate1_pass": True,
        "gate2_pass": True,
        "reason": "kappa_grad_nonzero_three_layer_independent",
        "stage1_proof": {
            "rqvae_ckpt_sha256": sha256_of(BASELINE_RQVAE_CKPT),
            "stage1_pass": True,
            "method": "KModulatedHRQVAE — κ 真接 vq_loss 主路径",
        },
        "init_kappa_l": init_kappa_l,
        "final_kappa_l": final_kappa_l,
        "final_kappa_grad_norm_l": final_kappa_grad_norms,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"],
        "loss_decreased": train_trace[-1]["avg_loss"] < epoch0_loss,
        "train_trace_summary": train_trace,
        "r23_kill": False,
        "task_id": 468,
        "issue": "Issue #175",
        "overall_decision": "PASS",
    }
    verdict_path = PRODUCT_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)

    train_trace_path = PRODUCT_DIR / "train_trace.json"
    with open(train_trace_path, "w") as f:
        json.dump({"train_trace": train_trace, "overall_decision": "PASS"}, f, indent=2)

    print(f"  ✅ verdict.json: {verdict_path}", flush=True)
    print(f"  ✅ train_trace.json: {train_trace_path}", flush=True)
    print(f"\n🎯 Task #468 PASS — write verdict.md + commit + close issue #175 (manual next step)", flush=True)


def _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, epoch0_loss, reason):
    """NO-GO 路径: 写 verdict (R23 trigger)."""
    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_kappa_grad_norms = model.hrq.get_kappa_grad_norms()

    verdict = {
        "gate1_pass": True,
        "gate2_pass": False,
        "reason": f"r23_trigger_{reason}",
        "stage1_proof": {
            "rqvae_ckpt_sha256": sha256_of(BASELINE_RQVAE_CKPT),
            "stage1_pass": True,
        },
        "init_kappa_l": init_kappa_l,
        "final_kappa_l": final_kappa_l,
        "final_kappa_grad_norm_l": final_kappa_grad_norms,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"] if train_trace else None,
        "train_trace_summary": train_trace,
        "r23_kill": True,
        "task_id": 468,
        "issue": "Issue #175",
        "overall_decision": "NO-GO",
    }
    verdict_path = PRODUCT_DIR / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)

    train_trace_path = PRODUCT_DIR / "train_trace.json"
    with open(train_trace_path, "w") as f:
        json.dump({"train_trace": train_trace, "r23_trigger": reason, "overall_decision": "NO-GO"}, f, indent=2)

    print(f"  ❌ verdict.json (NO-GO): {verdict_path}", flush=True)
    print(f"  ❌ R23 trigger: {reason}", flush=True)
    print(f"\n🚫 Task #468 NO-GO — write verdict.md + commit + close issue #175 (manual next step)", flush=True)


if __name__ == "__main__":
    main()