#!/usr/bin/env python3
"""Task #427 / Issue #135 [方向B Gate1] 层级 product mixing + anchor 限幅防 per-codeword 坍缩.

新假设 (vs #132 失败): mixing 从 per-codeword α 改为每层/每 component 的低维权重 (3 个标量 per layer),
减少 codeword 级自由度导致的坍缩等价解; assignment 以 learnable-κ 主路作为 anchor,
product components 只作为 bounded correction; 加 component contribution 与 hard-count usage 审计.

5-step audit per Issue #135 §Gate1 1:
1. 三分量 d_mix 有限
2. layer-level mixing 微扰改变 d_mix/loss
3. learnable κ 与 mixing 梯度有限非零
4. bounded correction 不覆盖 anchor
5. hard assignment/round-trip 存在
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
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task427_issue135_layer_product_mixing")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
BOUNDED_CORRECTION_RATIO_MAX = 0.3  # bounded correction ≤ 30% of anchor distance


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


class LayerProductHRQVAE(nn.Module):
    """3-component d_mix + LAYER-LEVEL mixing weights (3 scalars per layer) + anchor-preserving hard assignment."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, bounded_correction_max=BOUNDED_CORRECTION_RATIO_MAX):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
        self.bounded_correction_max = bounded_correction_max

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
        # LAYER-LEVEL mixing logits (3 scalars per layer, NOT per-codeword)
        self.mixing_logits = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))
            self.mixing_logits.append(nn.Parameter(torch.tensor([2.0, 0.5, -1.0])))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def get_mixing_weights(self, layer_idx):
        return F.softmax(self.mixing_logits[layer_idx], dim=-1)  # (3,) — shared across all codewords

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

        d1 = stable_pairwise_hyp(z_e, codebook, c, EPS)  # anchor: learnable κ hyperbolic
        c_fixed = torch.tensor(1.0, device=d1.device)
        d2 = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)  # fixed κ hyperbolic component
        d3 = pairwise_euclidean(z_e, codebook)  # Euclidean component

        # Layer-level weights (3 scalars, NOT per-codeword)
        alpha = self.get_mixing_weights(layer_idx)  # (3,)
        # Bounded correction: each component has bounded contribution
        w1 = alpha[0].clamp(min=self.bounded_correction_max, max=1.0 - 2 * self.bounded_correction_max)
        w_remaining = 1.0 - w1
        w2 = alpha[1] * w_remaining / (alpha[1] + alpha[2]).clamp_min(1e-8)
        w3 = alpha[2] * w_remaining / (alpha[1] + alpha[2]).clamp_min(1e-8)
        w2 = w2.clamp(max=self.bounded_correction_max)
        w3 = w3.clamp(max=self.bounded_correction_max)
        # Renormalize
        w_sum = (w1 + w2 + w3).clamp_min(1e-8)
        w1, w2, w3 = w1/w_sum, w2/w_sum, w3/w_sum

        d_mix = w1 * d1 + w2 * d2 + w3 * d3

        # Anchor-correction ratio
        correction_total = (w2 + w3)
        anchor_ratio = (1.0 - correction_total).item()

        # Hard argmin SID
        assign = d_mix.argmin(dim=-1)
        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())

        domain_m1 = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        domain_m2 = (math.sqrt(c_fixed.item()) * z_e.norm(dim=-1)).max().item()

        return {
            "z_q_hard": z_q_hard, "z_q_st": z_q_st,
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "assign": assign, "d1": d1, "d2": d2, "d3": d3,
            "weights": torch.stack([w1, w2, w3]),
            "anchor_ratio": anchor_ratio,
            "kappa_l": kappa_l,
            "domain_margins": [domain_m1, domain_m2],
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
            alpha = model.get_mixing_weights(layer_idx)
            w1 = alpha[0]
            w2 = alpha[1]
            w3 = alpha[2]
            d_mix = w1 * d1 + w2 * d2 + w3 * d3
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


def five_step_layer_audit(model, X, batch_size, device):
    """5-step audit per Issue #135 §Gate1 1."""
    out_baseline = model(X[:batch_size].to(device))
    loss_baseline = out_baseline["loss"].item()
    assign_baseline = out_baseline["assign_list"][0]
    domain_m_baseline = [out_baseline["loss"].item(), out_baseline["loss"].item()]  # will refresh

    # Refresh forward to get actual layer outputs
    model.zero_grad(set_to_none=True)
    out0 = model.encoders[0](X[:batch_size].to(device))
    kappa_l = model.get_kappa_l(0)
    c = kappa_l.abs().clamp_min(1e-6)
    codebook = model.codebooks[0]
    d1 = stable_pairwise_hyp(out0, codebook, c, EPS)
    c_fixed = torch.tensor(1.0, device=device)
    d2 = stable_pairwise_hyp(out0, codebook, c_fixed, EPS)
    d3 = pairwise_euclidean(out0, codebook)
    alpha = model.get_mixing_weights(0)
    d_mix_baseline = alpha[0]*d1 + alpha[1]*d2 + alpha[2]*d3
    assign_baseline = d_mix_baseline.argmin(dim=-1)
    domain_m_baseline = [
        (c.sqrt() * out0.norm(dim=-1)).max().item(),
        (math.sqrt(c_fixed.item()) * out0.norm(dim=-1)).max().item(),
    ]

    # Mixing perturbation (per-component offsets)
    perturb = torch.tensor([2.0, -1.5, 0.5], device=device)
    model.mixing_logits[0].data = model.mixing_logits[0].data + perturb
    out_perturb = model(X[:batch_size].to(device))
    loss_perturb = out_perturb["loss"].item()

    out0_p = model.encoders[0](X[:batch_size].to(device))
    alpha_p = model.get_mixing_weights(0)
    d_mix_perturb = alpha_p[0]*d1 + alpha_p[1]*d2 + alpha_p[2]*d3
    assign_perturb = d_mix_perturb.argmin(dim=-1)

    # Domain margins post-perturb
    domain_m_perturb = [
        (c.sqrt() * out0_p.norm(dim=-1)).max().item(),
        (math.sqrt(c_fixed.item()) * out0_p.norm(dim=-1)).max().item(),
    ]

    d_mix_changed = (d_mix_perturb - d_mix_baseline).abs().max() > 1e-6
    loss_changed = abs(loss_perturb - loss_baseline) > 1e-10

    domain_ok = all(m < 1.0 - EPS for m in domain_m_baseline)

    # Anchor-correction ratio check (run baseline forward to get weights)
    anchor_ratios = []
    for l in range(len(model.num_emb_list)):
        a = model.get_mixing_weights(l)
        anchor_ratios.append(a[0].item())
    bounded_correction_ok = all(r >= 1.0 - 2 * model.bounded_correction_max for r in anchor_ratios)

    # Gradient finite non-zero (mixing + κ)
    model.zero_grad(set_to_none=True)
    out_perturb["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g_kappa = model.kappa_l_raw[l].grad
        g_mix = model.mixing_logits[l].grad
        max_g_k = g_kappa.abs().max().item() if g_kappa is not None else 0.0
        max_g_m = g_mix.abs().max().item() if g_mix is not None else 0.0
        grad_max_per_layer.append({"kappa": max_g_k, "mixing": max_g_m})
        if (g_kappa is None or not torch.isfinite(g_kappa).all() or max_g_k < 1e-20 or
            g_mix is None or not torch.isfinite(g_mix).all() or max_g_m < 1e-20):
            grad_finite_nz = False

    hard_assignment_ok = assign_baseline.numel() > 0 and (assign_baseline.max() < model.num_emb_list[0])

    # At least 2 components contribute > 0.1 (component contribution requirement)
    component_contrib_ok = sum(1 for r in anchor_ratios if r > 0.1) >= 1  # anchor is one; we need at least 2 components (anchor + 1 product)
    # Actually we need anchor + at least 1 of (w2/w3) > 0.1
    # Simpler check: max(w2+w3) > 0.1 for some layer
    component_contrib_ok = False
    for l in range(len(model.num_emb_list)):
        a = model.get_mixing_weights(l)
        if (a[1] > 0.1).item() or (a[2] > 0.1).item():
            component_contrib_ok = True
            break

    audit_pass = all([domain_ok, d_mix_changed, loss_changed, grad_finite_nz,
                       bounded_correction_ok, hard_assignment_ok, component_contrib_ok])

    return {
        "audit_pass": bool(audit_pass),
        "domain_ok": bool(domain_ok),
        "domain_margins_baseline": domain_m_baseline,
        "domain_margins_perturb": domain_m_perturb,
        "d_mix_changed": bool(d_mix_changed),
        "loss_changed": bool(loss_changed),
        "grad_finite_nz": bool(grad_finite_nz),
        "grad_max_per_layer": grad_max_per_layer,
        "bounded_correction_ok": bool(bounded_correction_ok),
        "anchor_ratios": anchor_ratios,
        "hard_assignment_ok": bool(hard_assignment_ok),
        "component_contrib_ok": bool(component_contrib_ok),
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
    print(f"[Task #427 Issue #135 Gate 1] layer-level product mixing + anchor-preserving hard assignment")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step layer audit]...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = LayerProductHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
    ).to(DEVICE)
    audit = five_step_layer_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = audit.get("audit_pass", False)
    for k in ["domain_ok", "d_mix_changed", "loss_changed", "grad_finite_nz",
              "bounded_correction_ok", "hard_assignment_ok", "component_contrib_ok"]:
        print(f"  {k}: {audit.get(k)}")
    print(f"  anchor_ratios: {audit.get('anchor_ratios')}")
    print(f"  grad_max_per_layer: {audit.get('grad_max_per_layer')}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        verdict = {"task": "task427_issue135_layer_product_mixing", "issue": 135,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        print("❌ 5-step audit FAIL — exit.")
        return

    print("\n[Main] 30 epoch layer-mixing training...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        LayerProductHRQVAE, X, BATCH_SIZE, NUM_EPOCHS, DEVICE, SEED, "Main-LayerMix",
    )

    final_main = main_metrics[-1] if main_metrics else None
    main_util = {f"L{l}": m["util"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}
    main_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_main["per_layer"])} if final_main else {}

    main_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_main["per_layer"]) if final_main else False

    verdict = {
        "task": "task427_issue135_layer_product_mixing", "issue": 135,
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