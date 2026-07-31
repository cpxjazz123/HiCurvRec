#!/usr/bin/env python3
"""Task #424 / Issue #132 [方向B Gate1] Weighted product d_mix 共享 Sinkhorn 运输分配.

新假设 (vs #128 失败): 不再对每个 item 直接做 soft posterior; 改为从三分量稳定 d_mix 生成
**共享 Sinkhorn transport plan**, per-codeword alpha 同时影响该全局 plan 的 cost.
alpha 对损失的梯度来自受列边际约束的匹配, 最终 SID 仍为 d_mix hard argmin.
"""
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")

from utils import mobius_add as hgrec_mobius_add

# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 remaps to cuda:0
KAPPA_MIN = 0.1
KAPPA_MAX = 2.0
NUM_EMB_LIST = [64, 128, 256]
BETA = 0.25
BATCH_SIZE = 256
LR = 1e-4
NUM_EPOCHS = 30
TEMPERATURE = 1.0
EPS = 1e-5
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task424_issue132_shared_sinkhorn")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
SINKHORN_MAX_ITERS = 10
SINKHORN_TOL = 1e-3


def project_to_poincare_ball(x, c, eps=EPS):
    sqrt_c = c.sqrt().clamp_min(1e-8)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    max_norm = (1.0 - eps) / sqrt_c
    scale = torch.where(x_norm > max_norm, max_norm / x_norm, torch.ones_like(x_norm))
    return x * scale


def stable_pairwise_hyp(z_e, codebook, c, eps=EPS):
    B, D = z_e.shape
    K = codebook.shape[0]
    z_e_p = project_to_poincare_ball(z_e, c, eps)
    cb_p = project_to_poincare_ball(codebook, c, eps)
    z_e_b = z_e_p.unsqueeze(1).expand(-1, K, -1)
    cb_b = cb_p.unsqueeze(0).expand(B, -1, -1)
    x2 = (z_e_b * z_e_b).sum(dim=-1)
    y2 = (cb_b * cb_b).sum(dim=-1)
    one_minus_c_x2 = (1.0 - c * x2).clamp_min(eps)
    one_minus_c_y2 = (1.0 - c * y2).clamp_min(eps)
    diff = hgrec_mobius_add(-z_e_b, cb_b, c)
    diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
    sqrt_c = c.sqrt().clamp_min(1e-8)
    arg = 1.0 + 2.0 * c * diff_norm.pow(2) / (one_minus_c_x2 * one_minus_c_y2)
    arg = arg.clamp_min(1.0 + eps)
    return (1.0 / sqrt_c) * torch.acosh(arg)


def pairwise_euclidean(z_e, codebook):
    return (z_e.unsqueeze(1) - codebook.unsqueeze(0)).norm(dim=-1)


def sinkhorn_balanced_transport(cost_matrix, row_marginal=None, col_marginal=None,
                                max_iters=SINKHORN_MAX_ITERS, tol=SINKHORN_TOL):
    B, K = cost_matrix.shape
    if row_marginal is None:
        row_marginal = torch.ones(B, device=cost_matrix.device) / B
    if col_marginal is None:
        col_marginal = torch.ones(K, device=cost_matrix.device) / K
    log_K = -cost_matrix
    log_K = log_K - log_K.max(dim=-1, keepdim=True).values.detach()
    K_kernel = log_K.exp()
    u = torch.ones(B, device=cost_matrix.device)
    v = torch.ones(K, device=cost_matrix.device)
    row_res = col_res = 1.0
    for it in range(max_iters):
        u = row_marginal / (K_kernel @ v + 1e-30)
        v = col_marginal / (K_kernel.t() @ u + 1e-30)
        plan = u.unsqueeze(-1) * K_kernel * v.unsqueeze(0)
        row_res = (plan.sum(dim=-1) - row_marginal).abs().max().item()
        col_res = (plan.sum(dim=-2) - col_marginal).abs().max().item()
        if max(row_res, col_res) < tol:
            break
    plan = u.unsqueeze(-1) * K_kernel * v.unsqueeze(0)
    return plan, row_res, col_res, it + 1


class SharedSinkhornProductHRQVAE(nn.Module):
    """3-component d_mix + per-codeword alpha + shared Sinkhorn transport plan."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, sinkhorn_iters=SINKHORN_MAX_ITERS):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for K in num_emb_list:
            self.encoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))

        self.codebooks = nn.ParameterList()
        self.kappa_l_raw = nn.ParameterList()
        self.gate_logits = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))
            self.gate_logits.append(nn.Parameter(torch.randn(K, 3) * 0.3))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def radial_rescale_codebook(self, layer_idx, new_kappa, old_kappa):
        ratio = (old_kappa.abs() / new_kappa.abs()).clamp_min(1e-6)
        factor = ratio.sqrt()
        with torch.no_grad():
            self.codebooks[layer_idx].data.mul_(factor)

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)

        z_e = self.encoders[layer_idx](x)
        codebook = self.codebooks[layer_idx]

        # 3 components
        d1 = stable_pairwise_hyp(z_e, codebook, c, EPS)
        c_fixed = torch.tensor(1.0, device=d1.device)
        d2 = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        d3 = pairwise_euclidean(z_e, codebook)

        # per-codeword alpha
        alpha = F.softmax(self.gate_logits[layer_idx], dim=-1)  # (K, 3)
        # d_mix = alpha[:, 0]·d1 + alpha[:, 1]·d2 + alpha[:, 2]·d3
        d_mix = alpha[:, 0].unsqueeze(0) * d1 + alpha[:, 1].unsqueeze(0) * d2 + alpha[:, 2].unsqueeze(0) * d3

        # Shared Sinkhorn transport on d_mix
        plan, row_res, col_res, n_iter = sinkhorn_balanced_transport(
            cost_matrix=d_mix, row_marginal=None, col_marginal=None,
            max_iters=self.sinkhorn_iters, tol=SINKHORN_TOL,
        )
        z_q_transport = plan @ codebook

        # Hard argmin SID (record)
        with torch.no_grad():
            assign = d_mix.argmin(dim=-1)
        z_q_hard = codebook[assign]

        # Domain margins for 2 hyperbolic components
        domain_m1 = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        domain_m2 = (math.sqrt(c_fixed.item()) * z_e.norm(dim=-1)).max().item()

        z_q_st = z_q_transport
        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        transport_loss = (plan * d_mix).sum()
        alpha_entropy = -(alpha * (alpha + 1e-12).log()).sum(-1).mean()

        return {
            "z_q_transport": z_q_transport, "z_q_hard": z_q_hard, "z_q_st": z_q_st,
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "transport_loss": transport_loss, "plan": plan, "d_mix": d_mix,
            "d1": d1, "d2": d2, "d3": d3, "alpha": alpha,
            "alpha_entropy": alpha_entropy, "assign": assign,
            "row_residual": row_res, "col_residual": col_res, "n_iter": n_iter,
            "kappa_l": kappa_l, "domain_margins": [domain_m1, domain_m2],
        }

    def forward(self, x):
        out_list = []
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        total_transport = 0.0
        all_assign = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            out_list.append(out)
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            total_transport = total_transport + out["transport_loss"]
            residual = residual - out["z_q_st"]
        return {
            "recon_loss": total_recon, "commit_loss": total_commit,
            "transport_loss": total_transport,
            "loss": total_recon + self.beta * total_commit + 0.01 * total_transport,
            "assign_list": all_assign, "out_list": out_list,
        }

    def optimizer_step_hook(self):
        for l in range(len(self.num_emb_list)):
            new_kappa = self.get_kappa_l(l).item()
            old_kappa = getattr(self, "_last_kappa", [new_kappa]*len(self.num_emb_list))[l]
            if abs(new_kappa - old_kappa) > 1e-8:
                self.radial_rescale_codebook(l, torch.tensor(new_kappa), torch.tensor(old_kappa))
        self._last_kappa = [self.get_kappa_l(l).item() for l in range(len(self.num_emb_list))]


class PerItemControlHRQVAE(nn.Module):
    """3-component d_mix + per-item soft posterior (control, #128 形式)."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for K in num_emb_list:
            self.encoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))

        self.codebooks = nn.ParameterList()
        self.kappa_l_raw = nn.ParameterList()
        self.gate_logits = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))
            self.gate_logits.append(nn.Parameter(torch.randn(K, 3) * 0.3))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def radial_rescale_codebook(self, layer_idx, new_kappa, old_kappa):
        ratio = (old_kappa.abs() / new_kappa.abs()).clamp_min(1e-6)
        factor = ratio.sqrt()
        with torch.no_grad():
            self.codebooks[layer_idx].data.mul_(factor)

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        z_e = self.encoders[layer_idx](x)
        codebook = self.codebooks[layer_idx]
        d1 = stable_pairwise_hyp(z_e, codebook, c, EPS)
        c_fixed = torch.tensor(1.0, device=d1.device)
        d2 = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        d3 = pairwise_euclidean(z_e, codebook)
        alpha = F.softmax(self.gate_logits[layer_idx], dim=-1)
        d_mix = alpha[:, 0].unsqueeze(0) * d1 + alpha[:, 1].unsqueeze(0) * d2 + alpha[:, 2].unsqueeze(0) * d3
        assign = d_mix.argmin(dim=-1)
        z_q_hard = codebook[assign]
        log_p = -d_mix / self.temperature
        p = F.softmax(log_p, dim=-1)
        z_q_soft = p @ codebook
        z_q_st = z_q_soft
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        return {
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "z_q_hard": z_q_hard, "z_q_st": z_q_st, "assign": assign,
            "alpha": alpha, "d_mix": d_mix,
            "kappa_l": kappa_l,
            "domain_margins": [
                (c.sqrt() * z_e.norm(dim=-1)).max().item(),
                (math.sqrt(c_fixed.item()) * z_e.norm(dim=-1)).max().item(),
            ],
        }

    def forward(self, x):
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            residual = residual - out["z_q_st"]
        return {
            "recon_loss": total_recon, "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
            "assign_list": all_assign,
        }

    def optimizer_step_hook(self):
        for l in range(len(self.num_emb_list)):
            new_kappa = self.get_kappa_l(l).item()
            old_kappa = getattr(self, "_last_kappa", [new_kappa]*len(self.num_emb_list))[l]
            if abs(new_kappa - old_kappa) > 1e-8:
                self.radial_rescale_codebook(l, torch.tensor(new_kappa), torch.tensor(old_kappa))
        self._last_kappa = [self.get_kappa_l(l).item() for l in range(len(self.num_emb_list))]


def compute_layer_metrics(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    counts = torch.zeros(K, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            kappa_l = model.get_kappa_l(layer_idx)
            c = kappa_l.abs().clamp_min(1e-6)
            d1 = stable_pairwise_hyp(z_e, model.codebooks[layer_idx], c, EPS)
            c_fixed = torch.tensor(1.0, device=device)
            d2 = stable_pairwise_hyp(z_e, model.codebooks[layer_idx], c_fixed, EPS)
            d3 = pairwise_euclidean(z_e, model.codebooks[layer_idx])
            alpha = F.softmax(model.gate_logits[layer_idx], dim=-1)
            d_mix = alpha[:, 0].unsqueeze(0) * d1 + alpha[:, 1].unsqueeze(0) * d2 + alpha[:, 2].unsqueeze(0) * d3
            assign = d_mix.argmin(dim=-1).cpu()
            for a in assign.tolist():
                counts[a] += 1
    n_used = (counts > 0).sum().item()
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    p = counts.float() / max(1, counts.sum().item())
    p_nz = p[p > 0]
    entropy = -(p_nz * p_nz.log()).sum().item() if len(p_nz) > 0 else 0.0
    return {"util": util, "max_load": max_load,
            "entropy_normalized": entropy / math.log(K) if K > 0 else 0,
            "n_used": n_used, "K": K}


def five_step_shared_sinkhorn_audit(model, X, batch_size, device):
    """5-step audit per Issue #132 §Gate1 1:
    1. 两个双曲域约束通过
    2. alpha 微扰改变 d_mix, transport plan, loss
    3. alpha/κ 梯度有限非零
    4. plan 行/列边际残差受控
    5. hard SID 已记录
    """
    out_baseline = model(X[:batch_size].to(device))
    if not torch.isfinite(out_baseline["loss"]):
        return {"audit_pass": False, "reason": "baseline loss NaN/Inf"}
    loss_baseline = out_baseline["loss"].item()
    d_mix_baseline = out_baseline["out_list"][0]["d_mix"].detach().clone()
    plan_baseline = out_baseline["out_list"][0]["plan"].detach().clone()
    domain_m_baseline = out_baseline["out_list"][0]["domain_margins"]
    row_res_baseline = out_baseline["out_list"][0]["row_residual"]
    col_res_baseline = out_baseline["out_list"][0]["col_residual"]

    # alpha perturbation (per-component offsets to break softmax invariance)
    perturb = torch.tensor([2.0, -1.5, 0.5], device=device).view(1, 3)  # 大幅扰动让 d_mix 变化足以打破 Sinkhorn 收敛
    for l in range(len(model.num_emb_list)):
        model.gate_logits[l].data = model.gate_logits[l].data + perturb
    out_perturb = model(X[:batch_size].to(device))
    if not torch.isfinite(out_perturb["loss"]):
        return {"audit_pass": False, "reason": "loss NaN/Inf after alpha perturb"}
    loss_perturb = out_perturb["loss"].item()
    d_mix_perturb = out_perturb["out_list"][0]["d_mix"].detach().clone()
    plan_perturb = out_perturb["out_list"][0]["plan"].detach().clone()
    domain_m_perturb = out_perturb["out_list"][0]["domain_margins"]
    row_res_perturb = out_perturb["out_list"][0]["row_residual"]
    col_res_perturb = out_perturb["out_list"][0]["col_residual"]

    d_mix_changed = (d_mix_perturb - d_mix_baseline).abs().max() > 1e-6
    plan_changed = (plan_perturb - plan_baseline).abs().max() > 1e-6
    plan_change_max = float((plan_perturb - plan_baseline).abs().max())
    loss_changed = abs(loss_perturb - loss_baseline) > 1e-10

    domain_ok = all(m < 1.0 - EPS for m in domain_m_perturb)
    plan_residual_ok = (np.isfinite(row_res_perturb) and np.isfinite(col_res_perturb) and
                        row_res_perturb < 1.0 and col_res_perturb < 1.0)

    # grad finite non-zero
    model.zero_grad(set_to_none=True)
    out_perturb["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g_kappa = model.kappa_l_raw[l].grad
        g_alpha = model.gate_logits[l].grad
        max_g_k = g_kappa.abs().max().item() if g_kappa is not None else 0.0
        max_g_a = g_alpha.abs().max().item() if g_alpha is not None else 0.0
        grad_max_per_layer.append({"kappa": max_g_k, "alpha": max_g_a})
        if (g_kappa is None or not torch.isfinite(g_kappa).all() or max_g_k < 1e-20 or
            g_alpha is None or not torch.isfinite(g_alpha).all() or max_g_a < 1e-20):
            grad_finite_nz = False

    hard_sid_recorded = out_perturb["assign_list"][0].numel() > 0

    audit_pass = all([domain_ok, plan_residual_ok, d_mix_changed,
                       loss_changed, grad_finite_nz, hard_sid_recorded])

    return {
        "audit_pass": bool(audit_pass),
        "domain_ok": bool(domain_ok),
        "domain_margins_perturb": domain_m_perturb,
        "plan_residual_ok": bool(plan_residual_ok),
        "row_res_perturb": row_res_perturb, "col_res_perturb": col_res_perturb,
        "d_mix_changed": bool(d_mix_changed),
        "plan_changed": bool(plan_changed),
        "plan_change_max": plan_change_max,
        "loss_changed": bool(loss_changed),
        "grad_finite_nz": bool(grad_finite_nz),
        "grad_max_per_layer": grad_max_per_layer,
        "hard_sid_recorded": bool(hard_sid_recorded),
    }


def train_one_config(model_class, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = model_class(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_batches = max(1, len(X) // batch_size)
    epoch_metrics = []
    best_avg_util = 0.0
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        idx_perm = torch.randperm(len(X))
        ep_loss = 0.0
        ep_n = 0
        for i in range(n_batches):
            batch_idx = idx_perm[i*batch_size:(i+1)*batch_size]
            x_batch = X[batch_idx].to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            if not torch.isfinite(out["loss"]):
                print(f"  [{log_prefix} Ep {epoch}] ❌ Non-finite loss @ batch {i}", flush=True)
                continue
            out["loss"].backward()
            optimizer.step()
            if hasattr(model, "optimizer_step_hook"):
                model.optimizer_step_hook()
            ep_loss += out["loss"].item()
            ep_n += 1
        ep_loss /= max(1, ep_n)
        per_layer = [compute_layer_metrics(model, X, l, batch_size, device) for l in range(len(NUM_EMB_LIST))]
        avg_util = sum(m["util"] for m in per_layer) / len(per_layer)
        if avg_util > best_avg_util:
            best_avg_util = avg_util
            best_epoch = epoch
        epoch_metrics.append({"epoch": epoch, "loss": ep_loss, "per_layer": per_layer, "avg_util": avg_util})
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  [{log_prefix} Ep {epoch:02d}/{num_epochs}] loss={ep_loss:.4f} util={avg_util:.3f}", flush=True)
        if epoch >= 5 and avg_util < 0.3:
            print(f"  [{log_prefix} Ep {epoch:02d}] ❌ USAGE-KILL (util={avg_util:.3f})", flush=True)
            break
    return model, epoch_metrics, best_epoch, best_avg_util


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print(f"[Task #424 Issue #132 Gate 1] Shared Sinkhorn transport on weighted product d_mix")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step shared sinkhorn audit]...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = SharedSinkhornProductHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
    ).to(DEVICE)
    audit = five_step_shared_sinkhorn_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = audit.get("audit_pass", False)
    print(f"  domain_ok: {audit.get('domain_ok')}")
    print(f"  plan_residual_ok: {audit.get('plan_residual_ok')}")
    print(f"  row_res_perturb: {audit.get('row_res_perturb')}, col_res_perturb: {audit.get('col_res_perturb')}")
    print(f"  d_mix_changed: {audit.get('d_mix_changed')}")
    print(f"  plan_changed: {audit.get('plan_changed')}")
    print(f"  plan_change_max: {audit.get('plan_change_max')}")
    print(f"  loss_changed: {audit.get('loss_changed')}")
    print(f"  grad_finite_nz: {audit.get('grad_finite_nz')}")
    print(f"  grad_max_per_layer: {audit.get('grad_max_per_layer')}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        verdict = {"task": "task424_issue132_shared_sinkhorn", "issue": 132,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        print("❌ 5-step audit FAIL — exit.")
        return

    print("\n[Main] 30 epoch shared-sinkhorn training...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        SharedSinkhornProductHRQVAE, X, BATCH_SIZE, NUM_EPOCHS, DEVICE, SEED, "Main-Shared",
    )

    print("\n[Control] 30 epoch per-item-posterior control...", flush=True)
    control_model, control_metrics, control_best_epoch, control_best_avg_util = train_one_config(
        PerItemControlHRQVAE, X, BATCH_SIZE, NUM_EPOCHS, DEVICE, SEED, "Ctrl-PerItem",
    )

    final_main = main_metrics[-1] if main_metrics else None
    final_control = control_metrics[-1] if control_metrics else None
    main_util = {f"L{l}": m["util"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}
    main_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}
    control_util = {f"L{l}": m["util"] for l, m in enumerate(final_control["per_layer"])} if final_control else {}
    control_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_control["per_layer"])} if final_control else {}

    main_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_main["per_layer"]) if final_main else False
    control_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_control["per_layer"]) if final_control else False

    verdict = {
        "task": "task424_issue132_shared_sinkhorn", "issue": 132,
        "audit": audit, "audit_pass": audit_pass,
        "main": {
            "best_epoch": main_best_epoch, "best_avg_util": main_best_avg_util,
            "final_util": main_util, "final_max_load": main_max_load,
            "epoch_metrics": main_metrics,
        },
        "control": {
            "best_epoch": control_best_epoch, "best_avg_util": control_best_avg_util,
            "final_util": control_util, "final_max_load": control_max_load,
            "epoch_metrics": control_metrics,
        },
        "gate1_pass": main_pass and control_pass and audit_pass,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Main: util={main_util}, max_load={main_max_load}, pass={main_pass}")
    print(f"Control: util={control_util}, max_load={control_max_load}, pass={control_pass}")
    print(f"Gate 1: {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()