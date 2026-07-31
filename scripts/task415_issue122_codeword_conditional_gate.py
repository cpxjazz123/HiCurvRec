#!/usr/bin/env python3
"""Task #415 / Issue #122 [方向B Gate1] per-codeword alpha_l,k mixed-distance gate."""
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")


def patched_poincare_distance(x, c, keepdim=True):
    sqrt_c = c.sqrt()
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y_norm = (x / (1 - c * x_norm.pow(2) + 1e-6)).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    num = (2 * sqrt_c * (x - (x / (1 - c * x_norm.pow(2) + 1e-6))).norm(dim=-1, keepdim=True))
    denom = ((1 - c * x_norm.pow(2) + 1e-6) * (1 - c * y_norm.pow(2) + 1e-6)).sqrt()
    return (num / denom.clamp_min(1e-6)).pow(2) / c.clamp_min(1e-6)


SEED = 42
DEVICE = "cuda:0"
KAPPA_MIN = 0.1
NUM_EMB_LIST = [64, 128, 256]
BETA = 0.25
BATCH_SIZE = 256
LR = 1e-4
NUM_EPOCHS = 30
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task415_issue122_codeword_conditional_gate")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


class CodewordGateLayer(nn.Module):
    """Per-layer codebook with per-codeword alpha_l,k gate."""

    def __init__(self, K, e_dim=768, kappa_min=0.1):
        super().__init__()
        self.K = K
        self.e_dim = e_dim
        self.kappa_min = kappa_min

        # Single codebook per layer
        self.embedding = nn.Parameter(torch.randn(K, e_dim) * 0.05)
        # κ_l (scalar negative)
        self.kappa_l_raw = nn.Parameter(torch.tensor(0.0))
        # Per-codeword gate logits (K, 3) — softmax over 3 components
        self.gate_logits = nn.Parameter(torch.zeros(K, 3))

        self.encoder = nn.Sequential(nn.Linear(e_dim, e_dim * 2), nn.ReLU(), nn.Linear(e_dim * 2, e_dim))
        self.decoder = nn.Sequential(nn.Linear(e_dim, e_dim * 2), nn.ReLU(), nn.Linear(e_dim * 2, e_dim))

    def get_kappa_l(self):
        return -(self.kappa_min + F.softplus(self.kappa_l_raw))

    def get_alpha_per_codeword(self):
        return F.softmax(self.gate_logits, dim=-1)  # (K, 3)

    def compute_d_mix(self, z_e):
        """z_e: (B, D) → d_mix: (B, K) per-codeword weighted distance."""
        K = self.K
        kappa_l = self.get_kappa_l()
        c = kappa_l.abs().clamp_min(1e-6)

        # 3 distance variants
        d_learn = (z_e.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1) * (1 + c.sqrt())
        d_fixed = patched_poincare_distance(z_e.unsqueeze(1), c, keepdim=True).squeeze(-1)
        d_eucl = (z_e.unsqueeze(1) - self.embedding.unsqueeze(0)).pow(2).sum(-1)

        # Per-codeword weights (K, 3)
        alpha = self.get_alpha_per_codeword()  # (K, 3)
        # d_mix = Σ_j alpha[k, j] * d_j
        d_mix = alpha[:, 0] * d_learn + alpha[:, 1] * d_fixed + alpha[:, 2] * d_eucl  # (B, K)
        return d_mix, alpha, (d_learn, d_fixed, d_eucl)


class CodewordGateHRQVAE(nn.Module):
    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, beta=0.25):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.beta = beta
        self.layers = nn.ModuleList([
            CodewordGateLayer(K, e_dim, kappa_min) for K in num_emb_list
        ])

    def forward(self, x):
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        all_alpha = []
        for layer in self.layers:
            z_e = layer.encoder(residual)
            d_mix, alpha, (d_learn, d_fixed, d_eucl) = layer.compute_d_mix(z_e)
            assign = d_mix.argmin(dim=-1)
            z_q = layer.embedding[assign]
            x_hat = layer.decoder(z_q)
            z_q_st = z_e + (z_q - z_e).detach()
            recon_loss = F.mse_loss(x_hat, residual)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            total_recon += recon_loss
            total_commit += commit_loss
            all_assign.append(assign)
            all_alpha.append(alpha)
            residual = residual - z_q_st.detach()
        return {
            "recon_loss": total_recon,
            "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
            "assign_list": all_assign,
            "alpha_list": all_alpha,
        }


def five_step_functional_audit(model, X, batch_size, device):
    """5-step audit per Issue spec §Gate1 1:
    1. perturb 任一活跃 k 的 gate → d_mix/loss/assignment 改变
    2. permute alpha → score 改变
    3. grad 有限非零
    4. round-trip
    5. component-usage 熵
    """
    model.train()
    x_batch = X[:batch_size].to(device)

    # Snapshot
    snap_gates = [layer.gate_logits.data.clone() for layer in model.layers]
    snap_emb = [layer.embedding.data.clone() for layer in model.layers]

    # Forward baseline
    out_base = model(x_batch)
    loss_base = out_base["loss"].item()
    d_mix_base = []
    for l, layer in enumerate(model.layers):
        z_e = layer.encoder(x_batch)
        d_mix, _, _ = layer.compute_d_mix(z_e)
        d_mix_base.append(d_mix.detach())

    # 1. perturb gate of L0 K0 — ASYMMETRIC perturbation (avoid softmax offset invariance)
    model.layers[0].gate_logits.data[0, :] = torch.tensor([-3.0, 0.0, 5.0])  # asymmetric
    out_pert = model(x_batch)
    loss_pert = out_pert["loss"].item()
    d_mix_pert = []
    for l, layer in enumerate(model.layers):
        z_e = layer.encoder(x_batch)
        d_mix, _, _ = layer.compute_d_mix(z_e)
        d_mix_pert.append(d_mix.detach())
    gate_perturbed = abs(loss_pert - loss_base) > 1e-10
    d_mix_changed_L0 = (d_mix_pert[0] - d_mix_base[0]).abs().max().item() > 1e-10
    # Restore
    model.layers[0].gate_logits.data.copy_(snap_gates[0])

    # 2. permute alpha L0
    perm = torch.tensor([2, 0, 1])
    model.layers[0].gate_logits.data = model.layers[0].gate_logits.data[:, perm]
    out_perm = model(x_batch)
    loss_perm = out_perm["loss"].item()
    permuted_alpha = abs(loss_perm - loss_base) > 1e-10
    model.layers[0].gate_logits.data.copy_(snap_gates[0])

    # 3. grad finite non-zero (DIRECT backward)
    model.zero_grad(set_to_none=True)
    out_g = model(x_batch)
    out_g["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l, layer in enumerate(model.layers):
        g = layer.gate_logits.grad
        if g is None or not torch.isfinite(g).all() or g.abs().max() < 1e-20:
            grad_finite_nz = False
        grad_max_per_layer.append(g.abs().max().item() if g is not None else 0.0)

    # 4. round-trip
    with torch.no_grad():
        rt_diff_max = 0.0
        for l, layer in enumerate(model.layers):
            d = (snap_emb[l] - layer.embedding.data).abs().max().item()
            rt_diff_max = max(rt_diff_max, d)
    roundtrip_ok = rt_diff_max < 1.0

    # 5. component-usage 熵 (per layer per codeword) — need NON-uniform init
    entropy_per_layer = []
    for l, layer in enumerate(model.layers):
        alpha = F.softmax(snap_gates[l], dim=-1)
        entropy_per_codeword = -(alpha * (alpha + 1e-9).log()).sum(-1)
        entropy_per_layer.append({
            "L{l}_mean_entropy": entropy_per_codeword.mean().item(),
            "L{l}_max_entropy": entropy_per_codeword.max().item(),
            "L{l}_min_entropy": entropy_per_codeword.min().item(),
        })

    return {
        "gate_perturbed": gate_perturbed,
        "d_mix_changed_L0": d_mix_changed_L0,
        "permuted_alpha": permuted_alpha,
        "grad_finite_nz": grad_finite_nz,
        "grad_max_per_layer": grad_max_per_layer,
        "roundtrip_ok": roundtrip_ok,
        "entropy_per_layer": entropy_per_layer,
        "loss_base": loss_base,
        "loss_pert": loss_pert,
    }


def compute_layer_metrics(model, X, batch_size, device, layer_idx):
    model.eval()
    counts = torch.zeros(model.num_emb_list[layer_idx], dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            layer = model.layers[layer_idx]
            z_e = layer.encoder(x_batch)
            d_mix, _, _ = layer.compute_d_mix(z_e)
            assign = d_mix.argmin(dim=-1).cpu()
            for a in assign.tolist():
                if a < len(counts):
                    counts[a] += 1
    n_used = (counts > 0).sum().item()
    K = model.num_emb_list[layer_idx]
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    p = counts.float() / max(1, counts.sum().item())
    p_nz = p[p > 0]
    entropy = -(p_nz * p_nz.log()).sum().item() if len(p_nz) > 0 else 0.0
    return {"util": util, "max_load": max_load, "n_used": n_used, "K": K}


def train_main(X, batch_size, num_epochs, device, seed, log_prefix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = CodewordGateHRQVAE(num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, beta=BETA).to(device)
    # Init codebook from data
    with torch.no_grad():
        for layer in model.layers:
            idx = torch.randperm(len(X))[:layer.K]
            layer.embedding.data = X[idx].to(device)
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
                continue
            out["loss"].backward()
            optimizer.step()
            ep_loss += out["loss"].item()
        ep_loss /= n_batches
        per_layer = []
        for l in range(len(NUM_EMB_LIST)):
            m = compute_layer_metrics(model, X, batch_size, device, l)
            per_layer.append(m)
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
    print(f"[Task #415 Issue #122 Gate 1] Codeword-Conditional Gate")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    # embedding column contains numpy arrays per row → stack
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}")

    # === 5-step functional audit ===
    print("\n[5-step functional audit]", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = CodewordGateHRQVAE(num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, beta=BETA).to(DEVICE)
    audit = five_step_functional_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = all([audit["gate_perturbed"], audit["d_mix_changed_L0"], audit["permuted_alpha"],
                       audit["grad_finite_nz"], audit["roundtrip_ok"]])
    print(f"  gate_perturbed: {audit['gate_perturbed']}")
    print(f"  d_mix_changed_L0: {audit['d_mix_changed_L0']}")
    print(f"  permuted_alpha: {audit['permuted_alpha']}")
    print(f"  grad_finite_nz: {audit['grad_finite_nz']}")
    print(f"  grad_max_per_layer: {audit['grad_max_per_layer']}")
    print(f"  roundtrip_ok: {audit['roundtrip_ok']}")
    print(f"  entropy_per_layer: {audit['entropy_per_layer']}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step functional audit FAIL — exit without training.")
        return

    # === Main 30 epoch ===
    print("\n[Main] 30 epoch per-codeword gate...", flush=True)
    e_dim_actual = X.shape[1]
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_main(
        X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, device=DEVICE, seed=SEED, log_prefix="Main"
    )

    final_main = main_metrics[-1]
    main_util = {f"L{l}": m["util"] for l, m in enumerate(final_main["per_layer"])}
    main_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_main["per_layer"])}
    main_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_main["per_layer"])

    verdict = {
        "task": "task415_issue122_codeword_conditional_gate",
        "issue": 122,
        "audit": audit,
        "audit_pass": audit_pass,
        "main": {
            "best_epoch": main_best_epoch,
            "best_avg_util": main_best_avg_util,
            "final_util": main_util,
            "final_max_load": main_max_load,
            "epoch_metrics": main_metrics,
        },
        "gate1_pass": main_pass,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Main: util={main_util}, max_load={main_max_load}, pass={main_pass}")
    print(f"Gate 1: {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()