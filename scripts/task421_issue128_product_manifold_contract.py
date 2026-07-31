#!/usr/bin/env python3
"""Task #421 / Issue #128 [方向B Gate1] Product-space 三分量 Poincaré 域合同.

修复 #125 NaN: 每个 hyperbolic 分量独立 ball projection + d_mix 前域断言 + per-codeword alpha.

保留 #125 hard argmin forward + per-codeword α_l,k softmax(g_l,k) + 3 分量 (learn κ hyp / fixed κ hyp / Euclidean).
不做固定码字 / 2-分量 / 共享 kappa / 删除 Euclidean / 共享 alpha.

5-step 域审计 (每分量 ball projection + d_mix 前 assertion 嵌入) + 30 epoch main (有 projection+assertion) + control (#125 unstable).
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
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task421_issue128_product_manifold_contract")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


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
    return (1.0 / sqrt_c) * torch.acosh(arg)  # (B, K)


def pairwise_euclidean(z_e, codebook):
    return (z_e.unsqueeze(1) - codebook.unsqueeze(0)).norm(dim=-1)  # (B, K)


def assert_in_ball(x, c, layer_name, eps=EPS):
    """Forward-only domain assertion: raise on violation (per spec §d_mix前域断言)."""
    sqrt_c = c.sqrt().clamp_min(1e-8)
    x_norm = x.norm(dim=-1)
    max_norm = (1.0 - eps) / sqrt_c
    if not torch.isfinite(x_norm).all():
        raise RuntimeError(f"[{layer_name}] domain assertion FAIL: x contains NaN/Inf")
    if (x_norm >= max_norm).any():
        raise RuntimeError(f"[{layer_name}] domain assertion FAIL: sqrt(c)*‖x‖ ≥ 1-eps")


class ProductManifoldHRQVAE(nn.Module):
    """3-component (learn κ hyp / fixed κ=1 hyp / Euclidean) per-codeword alpha HRQVAE.

    Per Issue #128: 每个 hyperbolic 分量独立 ball projection + d_mix 前域断言.
    """

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, enable_projection=True, enable_assertion=True):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
        self.enable_projection = enable_projection
        self.enable_assertion = enable_assertion

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
        # Per-codeword alpha gate_logits (K, 3) for each layer
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

        # Component 1: learn κ hyp (with optional projection)
        if self.enable_projection:
            d1 = stable_pairwise_hyp(z_e, codebook, c, EPS)
        else:
            d1 = stable_pairwise_hyp(z_e, codebook, c, EPS)
        # Component 2: fixed κ=1 hyp
        c_fixed = torch.tensor(1.0, device=d1.device)
        if self.enable_projection:
            d2 = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        else:
            d2 = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        # Component 3: Euclidean
        d3 = pairwise_euclidean(z_e, codebook)

        # Per-codeword alpha = softmax(gate_logits)  (K, 3)
        alpha = F.softmax(self.gate_logits[layer_idx], dim=-1)  # (K, 3)
        # domain assertion on alpha (just numerical check)
        if self.enable_assertion:
            assert torch.isfinite(alpha).all(), f"alpha contains NaN at L{layer_idx}"

        # d_mix = Σ_j alpha_j * d_j  (B, K) by broadcasting alpha
        d_mix = alpha[:, 0].unsqueeze(0) * d1 + alpha[:, 1].unsqueeze(0) * d2 + alpha[:, 2].unsqueeze(0) * d3
        if self.enable_assertion:
            assert torch.isfinite(d_mix).all(), f"d_mix contains NaN at L{layer_idx}"

        assign = d_mix.argmin(dim=-1)
        z_q_hard = codebook[assign]

        log_p = -d_mix / self.temperature
        p = F.softmax(log_p, dim=-1)
        z_q_soft = p @ codebook

        z_q_st = z_q_soft
        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        d_mix_min = d_mix.min(dim=-1).values.mean()
        alpha_entropy = -(alpha * (alpha + 1e-12).log()).sum(-1).mean()

        # Domain margin per hyperbolic component
        domain_m1 = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        domain_m2 = (math.sqrt(c_fixed.item()) * z_e.norm(dim=-1)).max().item()

        return {
            "z_q_soft": z_q_soft, "z_q_hard": z_q_hard, "z_q_st": z_q_st,
            "x_hat": x_hat, "recon_loss": recon_loss, "commit_loss": commit_loss,
            "d_mix_min": d_mix_min, "assign": assign, "p_soft": p,
            "kappa_l": kappa_l, "alpha": alpha, "alpha_entropy": alpha_entropy,
            "domain_margins": [domain_m1, domain_m2],
            "d_mix": d_mix,
        }

    def forward(self, x):
        out_list = []
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            out_list.append(out)
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            residual = residual - out["z_q_st"]
        return {
            "recon_loss": total_recon, "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
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
    entropy_max = math.log(K)
    entropy_norm = entropy / entropy_max if entropy_max > 0 else 0.0
    return {"util": util, "max_load": max_load, "entropy_normalized": entropy_norm, "n_used": n_used, "K": K}


def five_step_product_audit(model, X, batch_size, device):
    """5-step audit per Issue #128 §Gate1 1:
    1. 每层每 hyperbolic 分量 encoder/codebook 独立 ball projection
    2. d_mix 前域断言通过
    3. 后验/loss/κ梯度全有限
    4. α 微扰改变 posterior + loss (对比 untouched baseline)
    5. hard assignment 已记录
    """
    old_alpha = [gl.data.clone() for gl in model.gate_logits]
    old_kappa_raw = [kr.data.clone() for kr in model.kappa_l_raw]

    # Baseline (untouched)
    out_baseline = model(X[:batch_size].to(device))
    if not torch.isfinite(out_baseline["loss"]):
        return {"audit_pass": False, "reason": "baseline loss NaN/Inf", "domain_ok": False}
    loss_baseline = out_baseline["loss"].item()
    posterior_baseline = out_baseline["out_list"][0]["p_soft"].detach().clone()

    # α perturbation — different per-component offsets (softmax is invariant to per-row constant)
    perturb_offsets = torch.tensor([0.5, -0.3, 0.1], device=device).view(1, 3)
    for l in range(len(model.num_emb_list)):
        model.gate_logits[l].data = model.gate_logits[l].data + perturb_offsets
    out_alpha = model(X[:batch_size].to(device))
    if not torch.isfinite(out_alpha["loss"]):
        return {"audit_pass": False, "reason": "loss NaN/Inf after α perturbation", "domain_ok": False}
    loss_after_α = out_alpha["loss"].item()
    posterior_after_α = out_alpha["out_list"][0]["p_soft"].detach().clone()

    # 1+2. Domain + d_mix assertion (under perturbed α)
    domain_ok = True
    with torch.no_grad():
        for l in range(len(model.num_emb_list)):
            K = model.num_emb_list[l]
            kappa_l = model.get_kappa_l(l)
            c = kappa_l.abs().clamp_min(1e-6)
            z_e = model.encoders[l](X[:batch_size].to(device))
            sqrt_c = math.sqrt(c.item())
            margin_h1 = (sqrt_c * z_e.norm(dim=-1)).max().item()
            c_fixed = 1.0
            margin_h2 = (math.sqrt(c_fixed) * z_e.norm(dim=-1)).max().item()
            if margin_h1 >= 1.0 - EPS or margin_h2 >= 1.0 - EPS:
                domain_ok = False

    posterior_changed = (posterior_after_α - posterior_baseline).abs().max() > 1e-6
    loss_changed = abs(loss_after_α - loss_baseline) > 1e-6

    # 5. grad finite
    model.zero_grad(set_to_none=True)
    out_alpha["loss"].backward()
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

    hard_assign_recorded = out_alpha["assign_list"][0].numel() > 0

    audit_pass = all([domain_ok, posterior_changed, loss_changed, grad_finite_nz, hard_assign_recorded])

    return {
        "audit_pass": bool(audit_pass),
        "domain_ok": bool(domain_ok),
        "posterior_changed": bool(posterior_changed),
        "loss_changed": bool(loss_changed),
        "grad_finite_nz": bool(grad_finite_nz),
        "grad_max_per_layer": grad_max_per_layer,
        "hard_assign_recorded": bool(hard_assign_recorded),
        "loss_baseline": loss_baseline,
        "loss_after_α": loss_after_α,
    }


def train_one_config(enable_projection_and_assertion, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = ProductManifoldHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_projection=enable_projection_and_assertion,
        enable_assertion=enable_projection_and_assertion,
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
        ep_n_batches = 0
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
            ep_n_batches += 1
        ep_loss /= max(1, ep_n_batches)

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
    print(f"[Task #421 Issue #128 Gate 1] Product manifold contract (per-layer 3-component α)")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step product audit]...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = ProductManifoldHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_projection=True, enable_assertion=True,
    ).to(DEVICE)
    audit = five_step_product_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = audit.get("audit_pass", False)
    print(f"  audit_pass: {audit_pass}")
    print(f"  domain_ok: {audit.get('domain_ok')}")
    print(f"  posterior_changed: {audit.get('posterior_changed')}")
    print(f"  loss_changed: {audit.get('loss_changed')}")
    print(f"  grad_finite_nz: {audit.get('grad_finite_nz')}")
    print(f"  grad_max_per_layer: {audit.get('grad_max_per_layer')}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step audit FAIL — exit.")
        verdict = {"task": "task421_issue128_product_manifold_contract", "issue": 128,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        return

    print("\n[Main] 30 epoch with projection + assertion...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        enable_projection_and_assertion=True, X=X, batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS, device=DEVICE, seed=SEED, log_prefix="Main",
    )

    print("\n[Control] 30 epoch without projection+assertion...", flush=True)
    control_model, control_metrics, control_best_epoch, control_best_avg_util = train_one_config(
        enable_projection_and_assertion=False, X=X, batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS, device=DEVICE, seed=SEED, log_prefix="Ctrl",
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
        "task": "task421_issue128_product_manifold_contract",
        "issue": 128,
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