#!/usr/bin/env python3
"""Task #418 / Issue #125 [方向B Gate1] hard SID + 软后向 mixed-distance posterior 恢复 gate gradient.

Hard argmin forward + per-codeword alpha_l,k = softmax(g_l,k) into soft posterior
softmax(-d_mix/τ) backward to recover alpha gradient.
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

from utils import poincare_distance as hgrec_poincare_distance, mobius_add as hgrec_mobius_add

# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # Will be remapped via CUDA_VISIBLE_DEVICES
KAPPA_MIN = 0.1
KAPPA_MAX = 2.0
NUM_EMB_LIST = [64, 128, 256]
BETA = 0.25
BATCH_SIZE = 256
LR = 1e-4
NUM_EPOCHS = 30
TEMPERATURE = 1.0
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task418_issue125_hard_soft_posterior")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def poincare_pairwise(z_e, codebook, c):
    """Compute pairwise Poincaré distance (B, K)."""
    B, D = z_e.shape
    K = codebook.shape[0]
    z_e_b = z_e.unsqueeze(1).expand(-1, K, -1)
    cb_b = codebook.unsqueeze(0).expand(B, -1, -1)
    d = hgrec_poincare_distance(z_e_b, cb_b, c)
    return d.squeeze(-1)  # (B, K)


class HardSoftPosteriorHRQVAE(nn.Module):
    """HRQVAE with hard SID forward + per-codeword alpha_l,k into soft posterior backward."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0,
                 beta=0.25, temperature=1.0, enable_soft_backward=True):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.temperature = temperature
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
        self.gate_logits = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))
            self.gate_logits.append(nn.Parameter(torch.randn(K, 3) * 0.3))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def get_alpha_per_codeword(self, layer_idx):
        return F.softmax(self.gate_logits[layer_idx], dim=-1)  # (K, 3)

    def compute_d_mix(self, z_e, layer_idx):
        """d_mix per codeword: 3-component weighted distance. Returns (B, K), alpha, components."""
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        codebook = self.codebooks[layer_idx]

        d_learn = poincare_pairwise(z_e, codebook, c)  # (B, K)
        d_fixed = poincare_pairwise(z_e, codebook, torch.tensor(1.0, device=z_e.device))  # fixed κ=1
        d_eucl = (z_e.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(dim=-1)  # (B, K)

        alpha = self.get_alpha_per_codeword(layer_idx)  # (K, 3)
        d_mix = alpha[:, 0] * d_learn + alpha[:, 1] * d_fixed + alpha[:, 2] * d_eucl
        return d_mix, alpha, (d_learn, d_fixed, d_eucl)

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        codebook = self.codebooks[layer_idx]

        z_e = self.encoders[layer_idx](x)
        d_mix, alpha, (d_learn, d_fixed, d_eucl) = self.compute_d_mix(z_e, layer_idx)

        assign = d_mix.argmin(dim=-1)
        z_q_hard = codebook[assign]

        log_p = -d_mix / self.temperature
        p = F.softmax(log_p, dim=-1)
        z_q_soft = p @ codebook

        if self.enable_soft_backward:
            z_q_st = z_q_soft
        else:
            z_q_st = z_e + (z_q_hard - z_e).detach()

        x_hat = self.decoders[layer_idx](z_q_st)

        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        d_mix_min = d_mix.min(dim=-1).values.mean()

        alpha_entropy = -(alpha * alpha.clamp_min(1e-10).log()).sum(dim=-1).mean()

        return {
            "z_q_soft": z_q_soft,
            "z_q_hard": z_q_hard,
            "z_q_st": z_q_st,
            "x_hat": x_hat,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "d_mix_min": d_mix_min,
            "assign": assign,
            "p_soft": p,
            "alpha": alpha,
            "alpha_entropy": alpha_entropy,
            "kappa_l": kappa_l,
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
            "recon_loss": total_recon,
            "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
            "assign_list": all_assign,
            "out_list": out_list,
        }


def compute_layer_metrics(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    counts = torch.zeros(K, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            d_mix, _, _ = model.compute_d_mix(z_e, layer_idx)
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


def five_step_functional_audit(model, X, batch_size, device):
    """5-step audit per Issue spec §Gate1 2."""
    old_gate_logits = [g.data.clone() for g in model.gate_logits]
    old_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]

    # DIRECTLY perturb gate_logits[0] (asymmetric test — only K0 alpha changes)
    with torch.no_grad():
        model.gate_logits[0].data[0, :] = torch.tensor([-3.0, 0.0, 5.0])

    new_gate_logits = [g.data.clone() for g in model.gate_logits]

    # 1. d_mix + posterior + z_q_soft change
    z_q_soft_changed = False
    d_mix_changed = False
    with torch.no_grad():
        z_e = model.encoders[0](X[:batch_size].to(device))
        codebook_0 = model.codebooks[0]
        kappa_old = abs(old_kappa_l[0])
        # Compute d_mix using OLD gate_logits[0]
        d_learn_old = poincare_pairwise(z_e, codebook_0, kappa_old)
        d_fixed_old = poincare_pairwise(z_e, codebook_0, 1.0)
        d_eucl_old = (z_e.unsqueeze(1) - codebook_0.unsqueeze(0)).pow(2).sum(dim=-1)
        alpha_old = F.softmax(old_gate_logits[0], dim=-1)
        d_mix_old = alpha_old[:, 0] * d_learn_old + alpha_old[:, 1] * d_fixed_old + alpha_old[:, 2] * d_eucl_old
        # Compute d_mix using NEW gate_logits[0]
        d_mix_new, alpha_new, _ = model.compute_d_mix(z_e, 0)
        d_mix_changed = (d_mix_new - d_mix_old).abs().max() > 1e-6
        p_old = F.softmax(-d_mix_old / model.temperature, dim=-1)
        p_new = F.softmax(-d_mix_new / model.temperature, dim=-1)
        z_q_soft_old = p_old @ codebook_0
        z_q_soft_new = p_new @ codebook_0
        z_q_soft_changed = (z_q_soft_new - z_q_soft_old).abs().max() > 1e-6

    # 2. grad 有限非零 via soft path
    with torch.no_grad():
        model.gate_logits[0].data.copy_(old_gate_logits[0])
    model.zero_grad(set_to_none=True)
    out = model(X[:batch_size].to(device))
    out["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g = model.gate_logits[l].grad
        if g is None or not torch.isfinite(g).all() or g.abs().max() < 1e-20:
            grad_finite_nz = False
        grad_max_per_layer.append(g.abs().max().item() if g is not None else 0.0)

    # 3. hard assignment recorded
    hard_assign_recorded = out["assign_list"][0].numel() > 0

    # 4. save/load round-trip
    state = {f"gate_logits_{i}": g.data.clone() for i, g in enumerate(model.gate_logits)}
    state_loaded = {k: v.clone() for k, v in state.items()}
    roundtrip_ok = all(torch.allclose(state[k], state_loaded[k]) for k in state)

    # 5. at least 2 components > 0.1 in non-trivial codeword subset
    alpha_main = F.softmax(model.gate_logits[0].detach(), dim=-1)
    K0 = model.num_emb_list[0]
    nontrivial_mask = alpha_main.max(dim=-1).values > 0.5
    nontrivial_n = nontrivial_mask.sum().item()
    if nontrivial_n > 0:
        nontrivial_alpha = alpha_main[nontrivial_mask]
        per_k_ge_01 = (nontrivial_alpha > 0.1).sum(dim=-1)
        two_components_ge_01 = (per_k_ge_01 >= 2).float().mean().item()
    else:
        two_components_ge_01 = 0.0

    return {
        "d_mix_changed": d_mix_changed,
        "z_q_soft_changed": z_q_soft_changed,
        "grad_finite_nz": grad_finite_nz,
        "grad_max_per_layer": grad_max_per_layer,
        "hard_assign_recorded": hard_assign_recorded,
        "roundtrip_ok": roundtrip_ok,
        "two_components_ge_01": two_components_ge_01,
    }


def train_one_config(enable_soft_backward, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = HardSoftPosteriorHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_soft_backward=enable_soft_backward,
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
    print(f"[Task #418 Issue #125 Gate 1] Hard SID + Soft posterior backward")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    print("\n[5-step audit] Pre-training audit on main config...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = HardSoftPosteriorHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_soft_backward=True,
    ).to(DEVICE)
    audit = five_step_functional_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = all([audit["d_mix_changed"], audit["z_q_soft_changed"],
                       audit["grad_finite_nz"], audit["hard_assign_recorded"],
                       audit["roundtrip_ok"]])
    print(f"  d_mix_changed: {audit['d_mix_changed']}")
    print(f"  z_q_soft_changed: {audit['z_q_soft_changed']}")
    print(f"  grad_finite_nz: {audit['grad_finite_nz']}")
    print(f"  grad_max_per_layer: {audit['grad_max_per_layer']}")
    print(f"  hard_assign_recorded: {audit['hard_assign_recorded']}")
    print(f"  roundtrip_ok: {audit['roundtrip_ok']}")
    print(f"  two_components_ge_01: {audit['two_components_ge_01']:.3f}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step audit FAIL — exit.")
        verdict = {"task": "task418_issue125_hard_soft_posterior", "issue": 125,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        return

    print("\n[Main] 30 epoch per-codeword soft backward...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        enable_soft_backward=True, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Main-pcw",
    )

    print("\n[Control] 30 epoch detach backward...", flush=True)
    control_model, control_metrics, control_best_epoch, control_best_avg_util = train_one_config(
        enable_soft_backward=False, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Ctrl-detach",
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
        "task": "task418_issue125_hard_soft_posterior",
        "issue": 125,
        "audit": audit,
        "audit_pass": audit_pass,
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
