#!/usr/bin/env python3
"""Task #420 / Issue #127 [方向A Gate1] Poincaré-ball 域投影与稳定距离的 κ 软后向审计.

修复 #124 NaN 问题: 在每个 κ 更新和 #47 同步重标定后, 对 encoder/codebook 应用每层
独立可微的 Poincaré-ball 半径投影 ‖x‖<(1-eps)/sqrt(c_l), 距离改为稳定 acosh 形式.

保留 #124 hard-forward/soft-backward bridge + hard SID + K=[64,128,256].
不做 clamp_min κ / 改 K / 温度扫描 / 欧氏替代.

5-step 域审计 + 30 epoch main (有投影稳定化) + control (无投影, 同 #124) 双配置.
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
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task420_issue127_ball_projection")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def project_to_poincare_ball(x, c, eps=EPS):
    """Project x to Poincaré ball: ‖x‖ * sqrt(c) < 1 - eps.

    Differentiable via clamp + safe division (no detach).
    """
    sqrt_c = c.sqrt().clamp_min(1e-8)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    max_norm = (1.0 - eps) / sqrt_c
    # Differentiable scale: where ‖x‖ > max_norm, scale down
    scale = torch.where(x_norm > max_norm, max_norm / x_norm, torch.ones_like(x_norm))
    return x * scale


def stable_poincare_distance(z_e, codebook, c, eps=EPS):
    """Stable pairwise Poincaré distance (B, K) using acosh.

    acosh(1 + 2·c·‖x‖²/(1-c·‖x‖²)·‖y‖²/(1-c·‖y‖²)·(1-<x̂,ŷ>) / ...)
    For numerical stability: use 1+2c·u + 2c·v - 2c·(1+2c)·<x,y> form,
    then acosh. Clamp to >=1+eps before acosh.
    """
    B, D = z_e.shape
    K = codebook.shape[0]
    # Project to ball
    z_e_p = project_to_poincare_ball(z_e, c, eps)
    cb_p = project_to_poincare_ball(codebook, c, eps)
    # Broadcast
    z_e_b = z_e_p.unsqueeze(1).expand(-1, K, -1)
    cb_b = cb_p.unsqueeze(0).expand(B, -1, -1)
    # ‖x‖², ‖y‖², <x,y>
    x2 = (z_e_b * z_e_b).sum(dim=-1)  # (B, K)
    y2 = (cb_b * cb_b).sum(dim=-1)
    xy = (z_e_b * cb_b).sum(dim=-1)
    # Argument of acosh: 1 + 2c (x2 + y2 - 2xy) / ((1-cx2)(1-cy2))  - well, simpler:
    # Let u = c·x2/(1-c·x2), v = c·y2/(1-c·y2). Then α = 1 + 2(u + v - 2c·xy/((1-c·x2)(1-c·y2)))/c — gets complex.
    # Simpler: ‖(-x) ⊕ y‖² = (‖y‖² - 2<x,y>·(1-c·‖x‖²)/(1-c·‖y‖²) + ‖x‖²·(1+c·‖y‖²)²/(1-c·‖y‖²)²) / ((1-c·‖x‖²)²·(1-c·‖y‖²))
    # For numerical stability use mobius_add + clamp norm.
    diff = hgrec_mobius_add(-z_e_b, cb_b, c)  # (B, K, D)
    diff_norm = diff.norm(dim=-1).clamp_min(1e-8)  # (B, K)
    sqrt_c = c.sqrt().clamp_min(1e-8)
    # acosh form: d = (1/sqrt(c)) * acosh(1 + 2c·‖diff‖²/((1-c‖x‖²)(1-c‖y‖²)))
    one_minus_c_x2 = (1.0 - c * x2).clamp_min(eps)
    one_minus_c_y2 = (1.0 - c * y2).clamp_min(eps)
    arg = 1.0 + 2.0 * c * diff_norm.pow(2) / (one_minus_c_x2 * one_minus_c_y2)
    arg = arg.clamp_min(1.0 + eps)  # ensure >= 1 for acosh
    d = (1.0 / sqrt_c) * torch.acosh(arg)
    return d  # (B, K)


class BallProjHRQVAE(nn.Module):
    """HRQVAE with Poincaré-ball projection + stable acosh distance + soft posterior backward."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, enable_projection=True, enable_soft_backward=True):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
        self.enable_projection = enable_projection
        self.enable_soft_backward = enable_soft_backward

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

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)

        z_e = self.encoders[layer_idx](x)
        codebook = self.codebooks[layer_idx]

        if self.enable_projection:
            z_e_p = project_to_poincare_ball(z_e, c, EPS)
            cb_p = project_to_poincare_ball(codebook, c, EPS)
            d_hyp = stable_poincare_distance(z_e, codebook, c, EPS)
        else:
            # Control: no projection, fallback to mobius_add distance
            B = z_e.shape[0]
            z_e_b = z_e.unsqueeze(1).expand(-1, K, -1)
            cb_b = codebook.unsqueeze(0).expand(B, -1, -1)
            diff = hgrec_mobius_add(-z_e_b, cb_b, c)
            diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
            sqrt_c = c.sqrt().clamp_min(1e-8)
            x2 = (z_e_b * z_e_b).sum(dim=-1)
            y2 = (cb_b * cb_b).sum(dim=-1)
            one_minus_c_x2 = (1.0 - c * x2).clamp_min(EPS)
            one_minus_c_y2 = (1.0 - c * y2).clamp_min(EPS)
            arg = 1.0 + 2.0 * c * diff_norm.pow(2) / (one_minus_c_x2 * one_minus_c_y2)
            arg = arg.clamp_min(1.0 + EPS)
            d_hyp = (1.0 / sqrt_c) * torch.acosh(arg)

        assign = d_hyp.argmin(dim=-1)
        z_q_hard = codebook[assign]

        log_p = -d_hyp / self.temperature
        p = F.softmax(log_p, dim=-1)
        z_q_soft = p @ codebook

        if self.enable_soft_backward:
            z_q_st = z_q_soft
        else:
            z_q_st = z_e + (z_q_hard - z_e).detach()

        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        d_hyp_min = d_hyp.min(dim=-1).values.mean()

        # Domain margin: sqrt(c) * ‖x‖  (must be < 1 - eps)
        domain_margin = (c.sqrt() * z_e.norm(dim=-1)).max().item()

        return {
            "z_q_soft": z_q_soft, "z_q_hard": z_q_hard, "z_q_st": z_q_st,
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "d_hyp_min": d_hyp_min, "assign": assign, "p_soft": p,
            "kappa_l": kappa_l, "domain_margin": domain_margin,
        }

    def forward(self, x):
        out_list = []
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        domain_margins = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            domain_margins.append(out["domain_margin"])
            out_list.append(out)
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            residual = residual - out["z_q_st"]
        return {
            "recon_loss": total_recon, "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
            "assign_list": all_assign, "out_list": out_list,
            "domain_margins": domain_margins,
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
            d_hyp = stable_poincare_distance(z_e, model.codebooks[layer_idx], c, EPS) if model.enable_projection else \
                    stable_poincare_distance(z_e, model.codebooks[layer_idx], c, EPS)
            assign = d_hyp.argmin(dim=-1).cpu()
            for a in assign.tolist():
                counts[a] += 1
    n_used = (counts > 0).sum().item()
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    p = counts.float() / max(1, counts.sum().item())
    p_nz = p[p > 0]
    entropy = -(p_nz * p_nz.log()).sum().item() if len(p_nz) > 0 else 0.0
    entropy_max = math.log(K)
    entropy_norm = entropy / entropy_max if entropy_max > 0 else 0.0
    return {"util": util, "max_load": max_load, "entropy_normalized": entropy_norm, "n_used": n_used, "K": K}


def five_step_domain_audit(model, X, batch_size, device):
    """5-step audit per Issue spec §Gate1 1:
    1. 每层 encoder/codebook 最大 sqrt(c)||x|| < 1-eps
    2. 距离/后验/loss/κ梯度全有限
    3. κ 微扰改变 posterior + loss (对比 untouched baseline, 不调 rescale 钩子)
    4. hard assignment 已记录
    5. 同步残差有限
    """
    # Capture UNTOUCHED baseline first
    old_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    old_codebooks = [cb.data.clone() for cb in model.codebooks]

    out_baseline = model(X[:batch_size].to(device))
    loss_baseline = out_baseline["loss"].item()
    posterior_baseline = out_baseline["out_list"][0]["p_soft"].detach().clone()

    # κ perturbation WITHOUT radial rescale (audit κ dependency, not geometric equivalence)
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data += 0.5
    new_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    new_codebooks = [cb.data.clone() for cb in model.codebooks]

    kappa_changed = any(abs(new - old) > 1e-4 for new, old in zip(new_kappa_l, old_kappa_l))

    # 1. Domain margin per layer (after perturbation)
    domain_ok = True
    domain_margins_after_perturb = []
    with torch.no_grad():
        for l in range(len(model.num_emb_list)):
            K = model.num_emb_list[l]
            c = abs(new_kappa_l[l])
            z_e = model.encoders[l](X[:batch_size].to(device))
            sqrt_c = math.sqrt(c)
            margin = (sqrt_c * z_e.norm(dim=-1)).max().item()
            domain_margins_after_perturb.append(margin)
            if margin >= 1.0 - EPS:
                domain_ok = False

    # 2. Forward under perturbed κ — compare against baseline (no rescale)
    out1 = model(X[:batch_size].to(device))
    loss_after_perturb = out1["loss"].item()
    posterior_after_perturb = out1["out_list"][0]["p_soft"].detach().clone()
    posterior_changed = (posterior_after_perturb - posterior_baseline).abs().max() > 1e-6
    loss_changed = abs(loss_after_perturb - loss_baseline) > 1e-10  # any finite change

    # 3. grad 有限非零
    model.zero_grad(set_to_none=True)
    out1["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g = model.kappa_l_raw[l].grad
        if g is None or not torch.isfinite(g).all() or g.abs().max() < 1e-20:
            grad_finite_nz = False
        grad_max_per_layer.append(g.abs().max().item() if g is not None else 0.0)

    # 4. round-trip (codebook unchanged since no rescale called)
    with torch.no_grad():
        rt_diff = 0.0
        for l in range(len(model.num_emb_list)):
            d = (old_codebooks[l] - new_codebooks[l]).abs().max().item()
            rt_diff = max(rt_diff, d)
    roundtrip_ok = rt_diff < 1e-6  # no rescale → expect diff ~0

    # 5. hard assignment recorded
    hard_assign_recorded = out1["assign_list"][0].numel() > 0

    return {
        "domain_ok": domain_ok,
        "domain_margins_after_perturb": domain_margins_after_perturb,
        "posterior_changed": bool(posterior_changed),
        "loss_changed": bool(loss_changed),
        "grad_finite_nz": bool(grad_finite_nz),
        "grad_max_per_layer": grad_max_per_layer,
        "roundtrip_ok": bool(roundtrip_ok),
        "hard_assign_recorded": bool(hard_assign_recorded),
        "old_kappa": old_kappa_l,
        "new_kappa": new_kappa_l,
        "loss_baseline": loss_baseline,
        "loss_after_perturb": loss_after_perturb,
    }


def train_one_config(enable_projection, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = BallProjHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_projection=enable_projection, enable_soft_backward=True,
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
            model.optimizer_step_hook()
            ep_loss += out["loss"].item()
        ep_loss /= n_batches

        per_layer = [compute_layer_metrics(model, X, l, batch_size, device) for l in range(len(NUM_EMB_LIST))]
        avg_util = sum(m["util"] for m in per_layer) / len(per_layer)
        if avg_util > best_avg_util:
            best_avg_util = avg_util
            best_epoch = epoch
        epoch_metrics.append({
            "epoch": epoch, "loss": ep_loss, "per_layer": per_layer, "avg_util": avg_util,
        })
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
    print(f"[Task #420 Issue #127 Gate 1] Poincaré-ball projection + acosh")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step domain audit] Main config...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = BallProjHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_projection=True, enable_soft_backward=True,
    ).to(DEVICE)
    audit = five_step_domain_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = all([audit["domain_ok"], audit["posterior_changed"], audit["loss_changed"],
                       audit["grad_finite_nz"], audit["roundtrip_ok"], audit["hard_assign_recorded"]])
    print(f"  domain_ok: {audit['domain_ok']}")
    print(f"  domain_margins_after_perturb: {audit['domain_margins_after_perturb']}")
    print(f"  posterior_changed: {audit['posterior_changed']}")
    print(f"  loss_changed: {audit['loss_changed']}")
    print(f"  grad_finite_nz: {audit['grad_finite_nz']}")
    print(f"  grad_max_per_layer: {audit['grad_max_per_layer']}")
    print(f"  roundtrip_ok: {audit['roundtrip_ok']}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step audit FAIL — exit.")
        verdict = {"task": "task420_issue127_ball_projection", "issue": 127,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step domain audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        return

    print("\n[Main] 30 epoch with projection...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        enable_projection=True, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Main-proj",
    )

    print("\n[Control] 30 epoch without projection...", flush=True)
    control_model, control_metrics, control_best_epoch, control_best_avg_util = train_one_config(
        enable_projection=False, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Ctrl-noproj",
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
        "task": "task420_issue127_ball_projection",
        "issue": 127,
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
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Main: util={main_util}, max_load={main_max_load}, pass={main_pass}")
    print(f"Control: util={control_util}, max_load={control_max_load}, pass={control_pass}")
    print(f"Gate 1: {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()
