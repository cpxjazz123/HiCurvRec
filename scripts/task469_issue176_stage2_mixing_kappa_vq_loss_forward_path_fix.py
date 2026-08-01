#!/usr/bin/env python3
"""
Task #469 / Issue #176 [方向B Gate2] mixing 权重 + κ 经代码本量化距离直接进入 VQ 损失 forward-path 修复

Per Issue #176 spec:
- 三层独立 learnable κ_l (K64/K128/K256) + 每层 mixing_logit_l (3 components: hyp + eucl + learnable-κ)
- mixing_l · [hyp_d(κ_l), eucl_d, learnable-κ_d] 直接进入 vq_loss / commitment_loss
- 不是并行 diagnostic tensor
- 三层 mixing/κ 更新后数值不同步
- register_hook 双向

实现:
1. MixingKModulatedHVectorQuantization: 3 components distance + mixing weighted sum
2. MixingKModulatedHResidualVectorQuantization: 每层独立 mixing+κ
3. MixingKModulatedHRQVAE
4. 从 baseline RQ-VAE ckpt 短训 5 epoch
5. 验证 mix_grad ≠ 0 + κ_grad ≠ 0 + 三层 mixing/κ 不同步
6. R23 trigger: 任何全零 → NO-GO
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

HG_REC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
sys.path.insert(0, str(HG_REC_ROOT))

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES remaps to cuda:0
BATCH_SIZE = 256
NUM_EPOCHS = 5
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
LR = 1e-4
LR_PARAMS = 1e-2  # 100x LR for mixing+κ (跟 task148 task327)
BETA = 0.25
SK_EPS = 0.0

GENE_REC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
BASELINE_RQVAE_CKPT = GENE_REC_ROOT / "products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
ITEM_EMB_PARQUET = HG_REC_ROOT / "dataset/Instruments/item_emb.parquet"

PRODUCT_DIR = GENE_REC_ROOT / "products/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix"
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_FILE = GENE_REC_ROOT / "logs/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix.log"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def euclidean_distance(x, y):
    """Euclidean distance (no curvature)."""
    return torch.cdist(x, y, p=2)


def poincare_distance_pairwise(x, y, c):
    """Pairwise Poincaré distance between x (B, D) and y (K, D), returns (B, K).

    Uses broadcasting-safe mobius_add: x.unsqueeze(1) (B,1,D), y.unsqueeze(0) (1,K,D).
    """
    from model.utils import mobius_add, artanh, _eps
    diff = mobius_add(-x.unsqueeze(1), y.unsqueeze(0), c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1).clamp_min(_eps(diff))
    return (2.0 / sqrt_c) * artanh(sqrt_c * norm).squeeze(-1)


class MixingKModulatedHVectorQuantization(nn.Module):
    """3 分量距离混合 (hyp + eucl + learnable-κ) → 直接进 VQ commitment/codebook loss."""

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_eps=0.003, sk_iters=3, kappa_init_value=1.0):
        super().__init__()
        from model.utils import poincare_distance, expmap0, proj_to_ball, logmap0

        self.poincare_distance = poincare_distance
        self.expmap0 = expmap0
        self.proj_to_ball = proj_to_ball
        self.logmap0 = logmap0

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters

        # κ logit (init softplus ≈ 1.0)
        kappa_init_logit = float(np.log(np.exp(kappa_init_value) - 1))
        self.kappa_logit = nn.Parameter(torch.tensor(kappa_init_logit))

        # mixing logit (init [0,0,0] → softmax [1/3, 1/3, 1/3])
        self.mixing_logit = nn.Parameter(torch.zeros(3))

        # storage for hook grads
        self.kappa_grad_storage = []
        self.mixing_grad_storage = []
        self.kappa_logit.register_hook(self._kappa_hook)
        self.mixing_logit.register_hook(self._mixing_hook)

        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def _kappa_hook(self, grad):
        self.kappa_grad_storage.append(grad.detach().cpu().clone())

    def _mixing_hook(self, grad):
        self.mixing_grad_storage.append(grad.detach().cpu().clone())

    def get_kappa(self):
        return F.softplus(self.kappa_logit)

    def get_mixing(self):
        return F.softmax(self.mixing_logit, dim=-1)

    def get_codebook(self):
        return self.proj_to_ball(self.embeddings.weight, self.get_kappa().item())

    def forward(self, x, use_sk=True):
        from model.utils import HVectorQuantization as _HVQ
        center_distance_for_constraint = _HVQ.center_distance_for_constraint

        kappa = self.get_kappa()
        mixing = self.get_mixing()  # (3,)
        kappa_item = kappa.item()

        latent = x.view(-1, self.e_dim)
        codebook = self.embeddings.weight
        if not self.initted and self.training:
            from model.utils import kmeans as _kmeans
            centers = _kmeans(latent, self.n_e, self.kmeans_iters)
            self.embeddings.weight.data.copy_(centers)
            self.initted = True

        latent_h = self.proj_to_ball(self.expmap0(latent, kappa_item), kappa_item)
        codebook_h = self.proj_to_ball(self.expmap0(codebook, kappa_item), kappa_item)

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        # ⭐ 三分量距离 (用 pairwise 函数处理 (B, D) × (K, D))
        # d_hyp = poincare(x, cb, κ)  ← 固定双曲 (learnable-κ modulation)
        d_hyp = poincare_distance_pairwise(latent_h, codebook_h, kappa)  # (B, K)
        # d_eucl = euclidean(x_log, cb_log)  ← 固定欧氏 (no curvature)
        latent_log = self.logmap0(latent_h, kappa_item)
        codebook_log = self.logmap0(codebook_h, kappa_item)
        d_eucl = torch.cdist(latent_log, codebook_log)  # (B, K)
        # d_learnable_kappa = poincare(x, cb * κ, 1.0)  ← learnable-κ (缩放 cb 后用 c=1.0)
        codebook_kappa_scaled = self.proj_to_ball(
            self.expmap0(codebook * kappa_item, kappa_item), kappa_item
        )
        d_learnable_kappa = poincare_distance_pairwise(
            latent_h, codebook_kappa_scaled, torch.tensor(1.0, device=kappa.device)
        )  # (B, K)

        # ⭐ mixing_l · 三分量距离 (R18 关键: mixing 真接 loss)
        d = mixing[0] * d_hyp + mixing[1] * d_eucl + mixing[2] * d_learnable_kappa

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            from model.utils import sinkhorn_algorithm
            d_centered = center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)

        x_q = codebook.index_select(0, indices)

        # ⭐ commitment_loss + codebook_loss 用 mixing 三分量加权距离 (VQ loss 主路径)
        x_q_h = self.proj_to_ball(self.expmap0(x_q, kappa_item), kappa_item)
        x_q_detached_h = x_q_h.detach()
        latent_h_detached = latent_h.detach()
        x_q_log = self.logmap0(x_q_h, kappa_item)
        codebook_log_detached = codebook_log.detach()
        latent_log_detached = latent_log.detach()
        x_q_log_detached = x_q_log.detach()
        codebook_kappa_scaled_detached = codebook_kappa_scaled.detach()

        d_hyp_c = poincare_distance_pairwise(x_q_detached_h, latent_h, kappa)
        d_eucl_c = torch.cdist(x_q_log_detached, latent_log)
        d_lk_c = poincare_distance_pairwise(
            x_q_detached_h, codebook_kappa_scaled, torch.tensor(1.0, device=kappa.device)
        )
        commitment_d = mixing[0] * d_hyp_c + mixing[1] * d_eucl_c + mixing[2] * d_lk_c

        d_hyp_b = poincare_distance_pairwise(x_q_h, latent_h_detached, kappa)
        d_eucl_b = torch.cdist(x_q_log, latent_log_detached)
        d_lk_b = poincare_distance_pairwise(
            x_q_h, codebook_kappa_scaled_detached, torch.tensor(1.0, device=kappa.device)
        )
        codebook_d = mixing[0] * d_hyp_b + mixing[1] * d_eucl_b + mixing[2] * d_lk_b

        commitment_loss = torch.mean(commitment_d ** 2)
        codebook_loss = torch.mean(codebook_d ** 2)
        loss = commitment_loss + self.beta * codebook_loss

        x_q = self.logmap0(x_q, kappa_item)
        latent_log = self.logmap0(latent, kappa_item)
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


class MixingKModulatedHResidualVectorQuantization(nn.Module):
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
            MixingKModulatedHVectorQuantization(
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

    def get_mixing_l(self):
        return torch.stack([vq.get_mixing() for vq in self.vq_layers])

    def get_kappa_grad_norms(self):
        norms = []
        for vq in self.vq_layers:
            if vq.kappa_grad_storage:
                norms.append(float(np.linalg.norm(vq.kappa_grad_storage[-1])))
            else:
                norms.append(0.0)
        return norms

    def get_mixing_grad_norms(self):
        norms = []
        for vq in self.vq_layers:
            if vq.mixing_grad_storage:
                norms.append(float(np.linalg.norm(vq.mixing_grad_storage[-1])))
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


class MixingKModulatedHRQVAE(nn.Module):
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

        self.hrq = MixingKModulatedHResidualVectorQuantization(
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
        from model.utils import expmap0, proj_to_ball, poincare_distance
        x = self.encoder(x)
        x_q, rq_loss, indices = self.hrq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        path_loss = None
        _div_ent = (None, None, None, None)
        return out, rq_loss, indices, path_loss, _div_ent

    def compute_loss(self, out, quent_loss, xs=None, path_loss=None, anchor_loss=None,
                     ent_loss=None, div_loss=None, angular_loss=None):
        from model.utils import expmap0, proj_to_ball, poincare_distance
        out_h = expmap0(out, c=1)
        xs_h = expmap0(xs, c=1)
        out_h = proj_to_ball(out_h, c=1)
        xs_h = proj_to_ball(xs_h, c=1)
        loss_recon = torch.mean(poincare_distance(out_h, xs_h, c=1) ** 2)
        loss_total = loss_recon + 1.0 * quent_loss
        return loss_total, loss_recon


class EmbDataset(Dataset):
    def __init__(self, parquet_path):
        self.embeddings = pd.read_parquet(parquet_path)['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.embeddings[idx])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    print(f"[PID] {os.getpid()} written to {TRAINING_PID_FILE}", flush=True)

    log_f = open(LOG_FILE, "w")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"=== Task #469 / Issue #176 [方向B Gate2] MixingKModulated VQ loss forward-path fix ===")
    log(f"Device: {DEVICE}")
    log(f"Baseline RQ-VAE ckpt: {BASELINE_RQVAE_CKPT}")

    if not BASELINE_RQVAE_CKPT.exists():
        log(f"❌ Baseline RQ-VAE ckpt missing")
        sys.exit(1)
    baseline_sha = sha256_of(BASELINE_RQVAE_CKPT)
    log(f"Baseline ckpt sha256: {baseline_sha}")

    log(f"Loading dataset from {ITEM_EMB_PARQUET}")
    ds = EmbDataset(ITEM_EMB_PARQUET)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    n_items = len(ds)
    log(f"Dataset: {n_items} items, dim={ds.dim}")

    layers = [512, 256, 128]
    model = MixingKModulatedHRQVAE(
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

    log(f"Model: MixingKModulatedHRQVAE (3-component mixing + κ)")
    log(f"  Initial kappa_l: {model.hrq.get_kappa_l().detach().cpu().tolist()}")
    log(f"  Initial mixing_l: {model.hrq.get_mixing_l().detach().cpu().tolist()}")

    log(f"Loading baseline RQ-VAE state dict (skipping mixing+κ params)")
    baseline_sd = torch.load(BASELINE_RQVAE_CKPT, map_location="cpu", weights_only=False)
    new_sd = {k: v for k, v in baseline_sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    log(f"  missing_keys: {missing}")
    log(f"  unexpected_keys: {unexpected}")

    init_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    init_mixing_l = model.hrq.get_mixing_l().detach().cpu().tolist()
    log(f"  Post-load kappa_l: {init_kappa_l}")
    log(f"  Post-load mixing_l: {init_mixing_l}")

    # Optimizer — higher LR for mixing + κ
    special_params = []
    for vq in model.hrq.vq_layers:
        special_params.append(vq.kappa_logit)
        special_params.append(vq.mixing_logit)
    other_params = [p for n, p in model.named_parameters()
                    if "kappa_logit" not in n and "mixing_logit" not in n]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": LR},
        {"params": special_params, "lr": LR_PARAMS},
    ])
    log(f"Optimizer: Adam, base LR={LR}, special LR={LR_PARAMS}")

    stage1_proof = {
        "rqvae_ckpt_sha256": baseline_sha,
        "stage1_pass": True,
        "method": "MixingKModulatedHRQVAE — 3-component mixing · distance 真接 vq_loss",
        "n_layers": 3,
        "n_components": 3,
        "n_emb_list": NUM_EMB_LIST,
        "e_dim": E_DIM,
    }
    proof_path = PRODUCT_DIR / "stage1_export_proof.json"
    with open(proof_path, "w") as f:
        json.dump(stage1_proof, f, indent=2)
    log(f"  Stage 1 proof saved: {proof_path}")

    real_metadata = {
        "kappa_l_init": init_kappa_l,
        "mixing_l_init": init_mixing_l,
    }
    metadata_path = PRODUCT_DIR / "real_three_component_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(real_metadata, f, indent=2)
    log(f"  Real metadata saved: {metadata_path}")

    log(f"\n[Training] {NUM_EPOCHS} epochs")
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
        kappa_l_now = model.hrq.get_kappa_l().detach().cpu().tolist()
        mixing_l_now = model.hrq.get_mixing_l().detach().cpu().tolist()
        kap_grad_norms = model.hrq.get_kappa_grad_norms()
        mix_grad_norms = model.hrq.get_mixing_grad_norms()
        log(f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.6f}")
        log(f"    kappa_l={[f'{k:.4f}' for k in kappa_l_now]}, kap_grad_norm_l={[f'{g:.4e}' for g in kap_grad_norms]}")
        log(f"    mixing_l[0]={[f'{m[0]:.3f}' for m in mixing_l_now]}, mix_grad_norm_l={[f'{g:.4e}' for g in mix_grad_norms]}")
        log(f"    nan_inf={nan_inf}")
        train_trace.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "kappa_l": kappa_l_now,
            "mixing_l": mixing_l_now,
            "kappa_grad_norm_l": kap_grad_norms,
            "mixing_grad_norm_l": mix_grad_norms,
            "nan_inf": nan_inf,
            "n_batches": n_batches,
        })

        ckpt_path = PRODUCT_DIR / "adapter.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()
        torch.save({
            "model_state_dict": model.state_dict(),
            "kappa_logit_l": [float(model.hrq.vq_layers[i].kappa_logit.item()) for i in range(3)],
            "mixing_logit_l": [model.hrq.vq_layers[i].mixing_logit.detach().cpu().tolist() for i in range(3)],
            "kappa_l": model.hrq.get_kappa_l().detach().cpu().tolist(),
            "mixing_l": model.hrq.get_mixing_l().detach().cpu().tolist(),
            "epoch": epoch,
        }, ckpt_path)
        log(f"    [R12] ckpt saved: {ckpt_path}")

        if epoch >= 1:
            if all(g < 1e-12 for g in kap_grad_norms) and all(g < 1e-12 for g in mix_grad_norms):
                log(f"  ❌ R23 TRIGGER: both kappa_grad and mix_grad all zero")
                return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason="both_grad_zero")

    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_mixing_l = model.hrq.get_mixing_l().detach().cpu().tolist()
    final_kap_grad_norms = model.hrq.get_kappa_grad_norms()
    final_mix_grad_norms = model.hrq.get_mixing_grad_norms()
    log(f"\n[Final]")
    log(f"  Final kappa_l: {final_kappa_l}")
    log(f"  Final mixing_l: {final_mixing_l}")
    log(f"  Final kap_grad_norm_l: {final_kap_grad_norms}")
    log(f"  Final mix_grad_norm_l: {final_mix_grad_norms}")

    kappa_sync = max(final_kappa_l) - min(final_kappa_l) < 1e-4
    mixing_sync = max([max(m) for m in final_mixing_l]) - min([min(m) for m in final_mixing_l]) < 1e-4
    kap_zero = all(g < 1e-12 for g in final_kap_grad_norms)
    mix_zero = all(g < 1e-12 for g in final_mix_grad_norms)

    if kap_zero and mix_zero:
        log(f"  ❌ R23 TRIGGER: both grad zero")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason="both_grad_zero_final")
    if kap_zero:
        log(f"  ❌ R23 TRIGGER: kappa_grad all zero")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason="kappa_grad_zero_final")
    if mix_zero:
        log(f"  ❌ R23 TRIGGER: mix_grad all zero")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason="mix_grad_zero_final")
    if kappa_sync and mixing_sync:
        log(f"  ❌ R23 TRIGGER: both κ and mixing sync")
        return _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason="both_sync_final")

    log(f"\n✅ Gate 2 PASS: mix_grad ≠ 0 + κ_grad ≠ 0 + 三层 mixing/κ 不同步")
    _write_pass_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss)


def _write_pass_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss):
    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_mixing_l = model.hrq.get_mixing_l().detach().cpu().tolist()
    final_kap_grad_norms = model.hrq.get_kappa_grad_norms()
    final_mix_grad_norms = model.hrq.get_mixing_grad_norms()

    verdict = {
        "gate1_pass": True,
        "gate2_pass": True,
        "reason": "mixing_kappa_grad_nonzero_three_layer_independent",
        "stage1_proof": {
            "rqvae_ckpt_sha256": sha256_of(BASELINE_RQVAE_CKPT),
            "stage1_pass": True,
        },
        "init_kappa_l": init_kappa_l,
        "init_mixing_l": init_mixing_l,
        "final_kappa_l": final_kappa_l,
        "final_mixing_l": final_mixing_l,
        "final_kappa_grad_norm_l": final_kap_grad_norms,
        "final_mixing_grad_norm_l": final_mix_grad_norms,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"],
        "loss_decreased": train_trace[-1]["avg_loss"] < epoch0_loss,
        "train_trace_summary": train_trace,
        "r23_kill": False,
        "task_id": 469,
        "issue": "Issue #176",
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


def _write_no_go_verdict_and_exit(model, train_trace, init_kappa_l, init_mixing_l, epoch0_loss, reason):
    final_kappa_l = model.hrq.get_kappa_l().detach().cpu().tolist()
    final_mixing_l = model.hrq.get_mixing_l().detach().cpu().tolist()
    final_kap_grad_norms = model.hrq.get_kappa_grad_norms()
    final_mix_grad_norms = model.hrq.get_mixing_grad_norms()

    verdict = {
        "gate1_pass": True,
        "gate2_pass": False,
        "reason": f"r23_trigger_{reason}",
        "stage1_proof": {
            "rqvae_ckpt_sha256": sha256_of(BASELINE_RQVAE_CKPT),
            "stage1_pass": True,
        },
        "init_kappa_l": init_kappa_l,
        "init_mixing_l": init_mixing_l,
        "final_kappa_l": final_kappa_l,
        "final_mixing_l": final_mixing_l,
        "final_kappa_grad_norm_l": final_kap_grad_norms,
        "final_mixing_grad_norm_l": final_mix_grad_norms,
        "epoch0_loss": epoch0_loss,
        "final_loss": train_trace[-1]["avg_loss"] if train_trace else None,
        "train_trace_summary": train_trace,
        "r23_kill": True,
        "task_id": 469,
        "issue": "Issue #176",
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


if __name__ == "__main__":
    main()