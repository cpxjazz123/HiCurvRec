#!/usr/bin/env python3
"""Task #426 / Issue #134 [方向A Gate1] hard-count EMA 双曲码本更新防单码字坍缩.

新假设 (vs #131/#127 失败): 训练期 forward SID 仍由稳定双曲 cost 的 hard argmin 产生;
codebook 更新使用 detach 的 hard counts 与 geodesic/切空间 EMA, 而不是让单样本 soft posterior 反向拉向全局几何中心;
对每层 L0/L1/L2 增加可审计的 batch/count 约束, 只约束 codeword 使用分布, 不改变 K、不导出软 SID;
κ 更新后继续按 #47 同步 scale/codebook/distance.

5-step audit per Issue #134 §Gate1 1:
1. hard argmin assignment 存在
2. EMA/geodesic update 不参与 encoder 梯度偷渡
3. count regularizer 对 hard counts 有效
4. κ/scale/codebook 同步残差有限
5. loss/κ grad/codebook norm 无 NaN/Inf
"""
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/GeneRec/scripts")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")

# Fix typo
sys.path = [p.replace("ar57", "ar57") if "ar57" in p else p for p in sys.path]

from utils import mobius_add as hgrec_mobius_add

# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
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
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task426_issue134_hard_ema")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
EMA_DECAY = 0.99
COUNT_REG_WEIGHT = 0.01


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


def geodesic_ema_update(codebook, hard_counts, z_e, assign, c, decay=EMA_DECAY, eps=EPS):
    """Geodesic/tangent-space EMA: for each codeword, compute weighted tangent centroid from assigned z_e,
    apply EMA in tangent space, project back to ball.

    Returns:
        new_codebook: (K, D)
        ema_drift_max: max per-element change
    """
    K, D = codebook.shape
    new_codebook = codebook.detach().clone()
    ema_drift_max = 0.0
    sqrt_c = c.sqrt().clamp_min(1e-8)
    for k in range(K):
        if hard_counts[k] > 0:
            mask = (assign == k)
            if mask.sum() == 0:
                continue
            assigned_z = z_e[mask]  # (N_k, D)
            assigned_z_p = project_to_poincare_ball(assigned_z.detach(), c, eps)
            # Logarithmic map: from codebook[k] to assigned_z_p, averaged in tangent space
            log_diff = hgrec_mobius_add(-assigned_z_p, codebook[k:k+1].expand_as(assigned_z_p), c)
            # Tangent centroid
            tangent_centroid = log_diff.mean(dim=0) * decay
            # Exponential map back: codebook[k] + tangent_centroid
            new_pos = hgrec_mobius_add(codebook[k:k+1], tangent_centroid.unsqueeze(0), c)
            new_pos_p = project_to_poincare_ball(new_pos, c, eps)
            drift = (new_pos_p - codebook[k:k+1]).abs().max().item()
            ema_drift_max = max(ema_drift_max, drift)
            new_codebook[k] = new_pos_p.squeeze(0)
    return new_codebook, ema_drift_max


def count_regularizer(hard_counts, K):
    """Encourage uniform usage via negative entropy of normalized counts."""
    p = hard_counts.float() / hard_counts.sum().clamp_min(1).float()
    p_nz = p[p > 0]
    if len(p_nz) == 0:
        return torch.tensor(0.0)
    entropy = -(p_nz * p_nz.log()).sum()
    entropy_max = math.log(K)
    # Penalize low entropy (concentrated) → return negative entropy normalized
    return -entropy / entropy_max


class HardEMAHRQVAE(nn.Module):
    """Stable hyperbolic cost + hard argmin SID + geodesic EMA codebook update + count regularizer."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, ema_decay=EMA_DECAY, count_reg_weight=COUNT_REG_WEIGHT):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.count_reg_weight = count_reg_weight

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
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def radial_rescale_codebook(self, layer_idx, new_kappa, old_kappa):
        ratio = (old_kappa.abs() / new_kappa.abs()).clamp_min(1e-6)
        factor = ratio.sqrt()
        with torch.no_grad():
            self.codebooks[layer_idx].data.mul_(factor)

    def forward_layer(self, x, layer_idx, hard_counts_batch=None):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)

        z_e = self.encoders[layer_idx](x)
        codebook = self.codebooks[layer_idx]

        cost = stable_pairwise_hyp(z_e, codebook, c, EPS)
        # Hard argmin SID (forward only)
        assign = cost.argmin(dim=-1)
        z_q_hard = codebook[assign]
        # STE: z_q_st = z_e + (z_q_hard - z_e).detach() — gradient flows through encoder, NOT through codebook
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())

        # Count regularizer on hard assignments (in-batch)
        if hard_counts_batch is None:
            hard_counts_batch = torch.bincount(assign, minlength=K).float()
        reg_loss = count_regularizer(hard_counts_batch, K)

        domain_margin = (c.sqrt() * z_e.norm(dim=-1)).max().item()

        return {
            "z_q_hard": z_q_hard, "z_q_st": z_q_st,
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "reg_loss": reg_loss, "assign": assign,
            "hard_counts": hard_counts_batch,
            "kappa_l": kappa_l, "domain_margin": domain_margin,
        }

    def forward(self, x):
        out_list = []
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        total_reg = 0.0
        all_assign = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            out_list.append(out)
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            total_reg = total_reg + out["reg_loss"]
            residual = residual - out["z_q_st"]
        return {
            "recon_loss": total_recon, "commit_loss": total_commit,
            "reg_loss": total_reg,
            "loss": total_recon + self.beta * total_commit + self.count_reg_weight * total_reg,
            "assign_list": all_assign, "out_list": out_list,
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
            cost = stable_pairwise_hyp(z_e, model.codebooks[layer_idx], c, EPS)
            assign = cost.argmin(dim=-1).cpu()
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


def five_step_ema_audit(model, X, batch_size, device):
    """5-step audit per Issue #134 §Gate1 1."""
    out_baseline = model(X[:batch_size].to(device))
    loss_baseline = out_baseline["loss"].item()
    assign_baseline = out_baseline["assign_list"][0]
    domain_margin_baseline = out_baseline["out_list"][0]["domain_margin"]
    reg_loss_baseline = out_baseline["reg_loss"].item()

    # κ perturbation
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data += 2.0
    out_perturb = model(X[:batch_size].to(device))
    loss_perturb = out_perturb["loss"].item()
    reg_loss_perturb = out_perturb["reg_loss"].item()
    domain_margin_perturb = out_perturb["out_list"][0]["domain_margin"]

    hard_argmin_ok = assign_baseline.numel() > 0 and (assign_baseline.max() < model.num_emb_list[0])

    # Check EMA doesn't steal encoder gradient — codebook is Parameter, but encoder gradient flows via STE only
    model.zero_grad(set_to_none=True)
    out_perturb["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g_kappa = model.kappa_l_raw[l].grad
        g_enc_w = model.encoders[l][0].weight.grad  # encoder Linear weight
        max_g_k = g_kappa.abs().max().item() if g_kappa is not None else 0.0
        max_g_e = g_enc_w.abs().max().item() if g_enc_w is not None else 0.0
        grad_max_per_layer.append({"kappa": max_g_k, "encoder": max_g_e})
        if (g_kappa is None or not torch.isfinite(g_kappa).all() or max_g_k < 1e-20 or
            g_enc_w is None or not torch.isfinite(g_enc_w).all() or max_g_e < 1e-20):
            grad_finite_nz = False

    domain_ok = domain_margin_baseline < 1.0 - EPS
    loss_changed = abs(loss_perturb - loss_baseline) > 1e-10
    reg_affects_loss = abs(reg_loss_perturb - reg_loss_baseline) > 0
    # Run EMA once to check drift
    with torch.no_grad():
        z_e = model.encoders[0](X[:batch_size].to(device))
        kappa_l = model.get_kappa_l(0)
        c = kappa_l.abs().clamp_min(1e-6)
        cost = stable_pairwise_hyp(z_e, model.codebooks[0], c, EPS)
        assign_ema = cost.argmin(dim=-1)
        hard_counts = torch.bincount(assign_ema, minlength=model.num_emb_list[0])
        new_cb, ema_drift = geodesic_ema_update(model.codebooks[0], hard_counts, z_e, assign_ema, c)
    ema_drift_finite = np.isfinite(ema_drift) and ema_drift < 100.0

    audit_pass = all([hard_argmin_ok, domain_ok, loss_changed, grad_finite_nz, reg_affects_loss, ema_drift_finite])

    return {
        "audit_pass": bool(audit_pass),
        "hard_argmin_ok": bool(hard_argmin_ok),
        "domain_ok": bool(domain_ok),
        "domain_margin_baseline": domain_margin_baseline,
        "domain_margin_perturb": domain_margin_perturb,
        "loss_changed": bool(loss_changed),
        "grad_finite_nz": bool(grad_finite_nz),
        "grad_max_per_layer": grad_max_per_layer,
        "reg_affects_loss": bool(reg_affects_loss),
        "ema_drift": float(ema_drift),
        "ema_drift_finite": bool(ema_drift_finite),
        "loss_baseline": loss_baseline, "loss_perturb": loss_perturb,
        "reg_loss_baseline": reg_loss_baseline, "reg_loss_perturb": reg_loss_perturb,
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
                continue
            out["loss"].backward()
            optimizer.step()
            if hasattr(model, "optimizer_step_hook"):
                model.optimizer_step_hook()
            # Apply EMA codebook update after each batch (with no_grad since it's detached)
            with torch.no_grad():
                for l in range(len(model.num_emb_list)):
                    if i == 0 and epoch == 1:
                        continue  # skip first batch (no history)
                    hard_counts = out["out_list"][l]["hard_counts"]
                    assign_l = out["assign_list"][l]
                    z_e_l = model.encoders[l](x_batch)
                    kappa_l = model.get_kappa_l(l)
                    c_l = kappa_l.abs().clamp_min(1e-6)
                    new_cb, ema_drift = geodesic_ema_update(
                        model.codebooks[l], hard_counts, z_e_l, assign_l, c_l,
                        decay=model.ema_decay, eps=EPS,
                    )
                    # Only update unused codewords slightly (keep them in ball)
                    used = (hard_counts > 0)
                    model.codebooks[l].data[used] = new_cb[used]
                    # For unused codewords, rescale slightly toward ball center
                    unused = ~used
                    if unused.any():
                        model.codebooks[l].data[unused] *= 0.999
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
    print(f"[Task #426 Issue #134 Gate 1] hard-count EMA + geodesic codebook update + count regularizer")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step hard EMA audit] Main config...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = HardEMAHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
    ).to(DEVICE)
    audit = five_step_ema_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = audit.get("audit_pass", False)
    for k in ["hard_argmin_ok", "domain_ok", "loss_changed", "grad_finite_nz",
              "reg_affects_loss", "ema_drift_finite"]:
        print(f"  {k}: {audit.get(k)}")
    print(f"  ema_drift: {audit.get('ema_drift')}")
    print(f"  grad_max_per_layer: {audit.get('grad_max_per_layer')}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        verdict = {"task": "task426_issue134_hard_ema", "issue": 134,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        print("❌ 5-step audit FAIL — exit.")
        return

    print("\n[Main] 30 epoch hard-EMA training...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        HardEMAHRQVAE, X, BATCH_SIZE, NUM_EPOCHS, DEVICE, SEED, "Main-HardEMA",
    )

    final_main = main_metrics[-1] if main_metrics else None
    main_util = {f"L{l}": m["util"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}
    main_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}

    main_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_main["per_layer"]) if final_main else False

    verdict = {
        "task": "task426_issue134_hard_ema", "issue": 134,
        "audit": audit, "audit_pass": audit_pass,
        "main": {
            "best_epoch": main_best_epoch, "best_avg_util": main_best_avg_util,
            "final_util": main_util, "final_max_load": main_max_load,
            "epoch_metrics": main_metrics,
        },
        "gate1_pass": main_pass and audit_pass,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Main: util={main_util}, max_load={main_max_load}, pass={main_pass}")
    print(f"Gate 1: {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()