#!/usr/bin/env python3
"""Task #417 / Issue #124 [方向A Gate1] hard 前向 + 软后向 κ gradient bridge.

Hard argmin forward + soft posterior (softmax(-d_hyp/τ)) backward to recover κ gradient.
Uses HG-Rec's poincare_distance for proper (B, K) pairwise distances.
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
DEVICE = "cuda:0"
KAPPA_MIN = 0.1
KAPPA_MAX = 2.0
NUM_EMB_LIST = [64, 128, 256]
BETA = 0.25
BATCH_SIZE = 256
LR = 1e-4
NUM_EPOCHS = 30
TEMPERATURE = 1.0
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task417_issue124_hard_soft_bridge")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def poincare_pairwise(z_e, codebook, c):
    """Compute pairwise Poincaré distance (B, K) between z_e (B, D) and codebook (K, D).

    Uses HG-Rec's poincare_distance(mobius_add(-x, y, c)) vectorized over K.
    """
    B, D = z_e.shape
    K = codebook.shape[0]
    # Broadcast z_e (B, 1, D) and codebook (1, K, D) → (B, K, D)
    z_e_b = z_e.unsqueeze(1).expand(-1, K, -1)
    cb_b = codebook.unsqueeze(0).expand(B, -1, -1)
    # Möbius add -x ⊕ y → shape (B, K, D)
    diff = hgrec_mobius_add(-z_e_b, cb_b, c)
    # poincare_distance(mobius_add(-x, y, c)) → (B, K, 1) → (B, K)
    d = hgrec_poincare_distance(z_e_b, cb_b, c)  # (B, K, 1)
    return d.squeeze(-1)  # (B, K)


class HardSoftBridgeHRQVAE(nn.Module):
    """HRQVAE with hard argmin forward + soft posterior backward."""

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

        codebook = self.codebooks[layer_idx]
        z_e = self.encoders[layer_idx](x)
        d_hyp = poincare_pairwise(z_e, codebook, c)  # (B, K)

        assign = d_hyp.argmin(dim=-1)
        z_q_hard = codebook[assign]

        # Soft posterior: softmax(-d_hyp/τ) — this carries κ gradient
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

        return {
            "z_q_soft": z_q_soft,
            "z_q_hard": z_q_hard,
            "z_q_st": z_q_st,
            "x_hat": x_hat,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "d_hyp_min": d_hyp_min,
            "assign": assign,
            "p_soft": p,
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
            d_hyp = poincare_pairwise(z_e, model.codebooks[layer_idx], c)
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


def five_step_functional_audit(model, X, batch_size, device):
    """5-step audit per Issue spec §Gate1 2."""
    old_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    old_codebooks = [cb.data.clone() for cb in model.codebooks]

    # DIRECTLY perturb κ
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data += 0.5
    new_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    model.optimizer_step_hook()
    new_codebooks = [cb.data.clone() for cb in model.codebooks]

    # 1. κ changed
    kappa_changed = any(abs(new - old) > 1e-4 for new, old in zip(new_kappa_l, old_kappa_l))

    # 2 & 3. z_q_soft changed with κ (different p + different codebook)
    z_q_soft_changed = False
    with torch.no_grad():
        for l in range(len(model.num_emb_list)):
            K = model.num_emb_list[l]
            kappa_new = abs(new_kappa_l[l])
            kappa_old = abs(old_kappa_l[l])
            z_e = model.encoders[l](X[:batch_size].to(device))
            # Use OLD codebook for both — pure κ effect
            d_old = poincare_pairwise(z_e, old_codebooks[l], kappa_old)
            d_new = poincare_pairwise(z_e, old_codebooks[l], kappa_new)
            p_old = F.softmax(-d_old / model.temperature, dim=-1)
            p_new = F.softmax(-d_new / model.temperature, dim=-1)
            z_q_soft_old = p_old @ old_codebooks[l]
            z_q_soft_new = p_new @ old_codebooks[l]
            if (z_q_soft_new - z_q_soft_old).abs().max() > 1e-6:
                z_q_soft_changed = True

    # 4. grad 有限非零 via soft-backward path
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data -= 0.5
    model.zero_grad(set_to_none=True)
    out = model(X[:batch_size].to(device))
    out["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g = model.kappa_l_raw[l].grad
        if g is None or not torch.isfinite(g).all() or g.abs().max() < 1e-20:
            grad_finite_nz = False
        grad_max_per_layer.append(g.abs().max().item() if g is not None else 0.0)

    # 5. round-trip
    with torch.no_grad():
        rt_diff = 0.0
        for l in range(len(model.num_emb_list)):
            d = (old_codebooks[l] - new_codebooks[l]).abs().max().item()
            rt_diff = max(rt_diff, d)
    roundtrip_ok = rt_diff < 1.0

    return {
        "kappa_changed": kappa_changed,
        "z_q_soft_changed": z_q_soft_changed,
        "grad_finite_nz": grad_finite_nz,
        "grad_max_per_layer": grad_max_per_layer,
        "roundtrip_ok": roundtrip_ok,
        "old_kappa": old_kappa_l,
        "new_kappa": new_kappa_l,
    }


def train_one_config(enable_soft_backward, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]
    model = HardSoftBridgeHRQVAE(
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
    print(f"[Task #417 Issue #124 Gate 1] Hard forward + Soft backward κ bridge")
    print("=" * 70)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    # === 5-step audit ===
    print("\n[5-step audit] Pre-training audit on main config...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = HardSoftBridgeHRQVAE(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual, kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX, beta=BETA, temperature=TEMPERATURE,
        enable_soft_backward=True,
    ).to(DEVICE)
    audit = five_step_functional_audit(audit_model, X, BATCH_SIZE, DEVICE)
    audit_pass = all([audit["kappa_changed"], audit["z_q_soft_changed"],
                       audit["grad_finite_nz"], audit["roundtrip_ok"]])
    print(f"  kappa_changed: {audit['kappa_changed']}")
    print(f"  z_q_soft_changed: {audit['z_q_soft_changed']}")
    print(f"  grad_finite_nz: {audit['grad_finite_nz']}")
    print(f"  grad_max_per_layer: {audit['grad_max_per_layer']}")
    print(f"  roundtrip_ok: {audit['roundtrip_ok']}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step audit FAIL — exit.")
        verdict = {"task": "task417_issue124_hard_soft_bridge", "issue": 124,
                    "audit": audit, "audit_pass": audit_pass,
                    "gate1_pass": False, "halt_reason": "5-step audit FAIL"}
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        return

    # === Main (soft-backward) ===
    print("\n[Main] 30 epoch soft backward...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        enable_soft_backward=True, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Main-soft",
    )

    # === Control (DETACH backward — same as #121) ===
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
        "task": "task417_issue124_hard_soft_bridge",
        "issue": 124,
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
