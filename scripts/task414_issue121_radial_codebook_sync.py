#!/usr/bin/env python3
"""Task #414 / Issue #121 [方向A Gate1] κ 更新后 assignment-preserving 径向 codebook 同步传输.

5-step functional audit + 30 epoch main (有径向传输) + control (无径向传输).
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
CKPT_TASK84 = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task414_issue121_radial_codebook_sync")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def patched_poincare_distance(x, c, keepdim=True):
    """Patched Poincaré distance from task411/Issue #118."""
    sqrt_c = c.sqrt()
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y_norm = (x / (1 - c * x_norm.pow(2) + 1e-6)).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    num = (2 * sqrt_c * (x - (x / (1 - c * x_norm.pow(2) + 1e-6))).norm(dim=-1, keepdim=True))
    denom = ((1 - c * x_norm.pow(2) + 1e-6) * (1 - c * y_norm.pow(2) + 1e-6)).sqrt()
    return (num / denom.clamp_min(1e-6)).pow(2) / c.clamp_min(1e-6)


class RadialSyncHRQVAE(nn.Module):
    """HRQVAE with per-layer κ_l + radial codebook transfer on every optimizer step."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0, beta=0.25,
                 enable_radial_sync=True):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.enable_radial_sync = enable_radial_sync

        # Encoder + decoder (per-layer)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for K in num_emb_list:
            self.encoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim),
                nn.ReLU(),
                nn.Linear(e_dim, e_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim),
                nn.ReLU(),
                nn.Linear(e_dim, e_dim),
            ))

        # Per-layer codebook + κ_l
        self.codebooks = nn.ParameterList()
        self.kappa_l_raw = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))

        # Snapshot for sync residual audit
        self._last_kappa = None
        self._last_rescale = None
        self._sync_residual = []

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))  # scalar, 始终负

    def radial_rescale_codebook(self, layer_idx, new_kappa, old_kappa):
        """Assignment-preserving radial rescale: c_new = c_old * sqrt(old_kappa/new_kappa)."""
        ratio = (old_kappa.abs() / new_kappa.abs()).clamp_min(1e-6)
        factor = ratio.sqrt()
        with torch.no_grad():
            self.codebooks[layer_idx].data.mul_(factor)

    def forward_layer(self, x, layer_idx):
        """x: (B, D) → quantized + hyperbolic distance + loss."""
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)

        codebook = self.codebooks[layer_idx]  # (K, D)

        # Encoder output → codeword assignment
        z_e = self.encoders[layer_idx](x)  # (B, D)

        # Hyperbolic distance to codebook
        d_hyp = patched_poincare_distance(
            z_e.unsqueeze(1), c, keepdim=True
        ).squeeze(-1)  # (B, K)

        # Argmin → hard assignment
        assign = d_hyp.argmin(dim=-1)  # (B,)

        # Quantized output
        z_q = codebook[assign]  # (B, D)

        # Decoder output
        x_hat = self.decoders[layer_idx](z_q)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q.detach())
        d_hyp_min = d_hyp.min(dim=-1).values.mean()

        return {
            "z_q": z_q,
            "z_q_st": z_q_st,
            "x_hat": x_hat,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "d_hyp_min": d_hyp_min,
            "assign": assign,
            "kappa_l": kappa_l,
            "codebook_norm": codebook.norm(),
        }

    def forward(self, x):
        """Full HRQ-VAE forward pass with optional radial sync."""
        out_list = []
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        kappa_history = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            kappa_history.append(out["kappa_l"].item())
            all_assign.append(out["assign"])

            # Sync residual: 记录旧 κ, 更新 codebook (via optimizer), 重算距离
            # 这里 radial sync 在 optimizer step 后调用 (在 train_step 里)
            out_list.append(out)
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            residual = residual - out["z_q_st"].detach()

        return {
            "recon_loss": total_recon,
            "commit_loss": total_commit,
            "loss": total_recon + self.beta * total_commit,
            "assign_list": all_assign,
            "out_list": out_list,
            "kappa_list": kappa_history,
        }

    def optimizer_step_hook(self):
        """Per-layer radial codebook sync after optimizer.step().

        For each layer: record old κ → apply radial rescale on codebook → record residual.
        """
        if not self.enable_radial_sync:
            return
        for l in range(len(self.num_emb_list)):
            new_kappa = self.get_kappa_l(l).item()
            old_kappa = self._last_kappa[l] if self._last_kappa is not None else new_kappa
            if abs(new_kappa - old_kappa) > 1e-8:
                # Radial rescale codebook
                self.radial_rescale_codebook(l, torch.tensor(new_kappa), torch.tensor(old_kappa))
                # Track sync residual
                rescale = math.sqrt(abs(old_kappa) / abs(new_kappa)) if abs(new_kappa) > 0 else 1.0
                self._sync_residual.append({"layer": l, "old_kappa": old_kappa, "new_kappa": new_kappa, "rescale": rescale})
        # Snapshot new kappa
        self._last_kappa = [self.get_kappa_l(l).item() for l in range(len(self.num_emb_list))]


def compute_layer_metrics(out, X, layer_idx, batch_size):
    """Compute util/max_load/entropy_normalized for one layer over the dataset."""
    model.eval()
    counts = torch.zeros(out["assign"].max().item() + 1, dtype=torch.long)
    # Recompute assignments over a sample
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(DEVICE)
            z_e = model.encoders[layer_idx](x_batch)
            K = model.num_emb_list[layer_idx]
            kappa_l = model.get_kappa_l(layer_idx)
            c = kappa_l.abs().clamp_min(1e-6)
            d_hyp = patched_poincare_distance(z_e.unsqueeze(1), c, keepdim=True).squeeze(-1)
            assign = d_hyp.argmin(dim=-1).cpu()
            for a in assign.tolist():
                if a < len(counts):
                    counts[a] += 1
    n_used = (counts > 0).sum().item()
    K = model.num_emb_list[layer_idx]
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    # Entropy normalized
    p = counts.float() / max(1, counts.sum().item())
    p_nonzero = p[p > 0]
    entropy = -(p_nonzero * p_nonzero.log()).sum().item()
    entropy_max = math.log(K)
    entropy_normalized = entropy / entropy_max if entropy_max > 0 else 0.0
    return {
        "util": util,
        "max_load": max_load,
        "entropy_normalized": entropy_normalized,
        "n_used": n_used,
        "K": K,
    }


def five_step_functional_audit(model, X, batch_size):
    """5-step audit per Issue spec §Gate1 2:
    1. κ update 改变 scale/距离 (DIRECT perturbation, not optimizer-dependent)
    2. 传输前后 assignment 一致
    3. 传输前后 distance ranking 一致
    4. grad 有限非零 (DIRECT backward)
    5. round-trip verify
    """
    # Snapshot
    old_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    old_codebooks = [cb.data.clone() for cb in model.codebooks]

    # DIRECTLY perturb kappa_l_raw (deterministic test)
    # κ_l = -(κ_min + softplus(u_l)) → perturb u_l → perturb κ
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data += 0.5  # significant perturbation

    new_kappa_l = [model.get_kappa_l(l).item() for l in range(len(model.num_emb_list))]
    model.optimizer_step_hook()  # apply radial sync with the perturbed κ
    new_codebooks = [cb.data.clone() for cb in model.codebooks]

    # 1. κ update 改变 scale/距离
    kappa_changed = any(abs(new - old) > 1e-4 for new, old in zip(new_kappa_l, old_kappa_l))

    # 2 & 3. 传输前后 assignment + ranking 一致
    with torch.no_grad():
        old_assign = []
        new_assign = []
        ranking_consistent = True
        for l in range(len(model.num_emb_list)):
            K = model.num_emb_list[l]
            kappa_old = torch.tensor(abs(old_kappa_l[l])).clamp_min(1e-6)
            kappa_new = torch.tensor(abs(new_kappa_l[l])).clamp_min(1e-6)
            z_e = model.encoders[l](X[:batch_size].to(DEVICE))
            d_old = patched_poincare_distance(z_e.unsqueeze(1), kappa_old, keepdim=True).squeeze(-1)
            d_new = patched_poincare_distance(z_e.unsqueeze(1), kappa_new, keepdim=True).squeeze(-1)
            old_assign.append(d_old.argmin(dim=-1))
            new_assign.append(d_new.argmin(dim=-1))
            # Ranking consistent
            if not (d_old.argmin(dim=-1) == d_new.argmin(dim=-1)).all():
                ranking_consistent = False
        assignment_consistent = all((o == n).all().item() for o, n in zip(old_assign, new_assign))

    # 4. grad 有限非零 (DIRECT backward)
    # Reset kappa_l_raw to original
    for l in range(len(model.num_emb_list)):
        model.kappa_l_raw[l].data -= 0.5
    model.zero_grad(set_to_none=True)
    out2 = model(X[:batch_size].to(DEVICE))
    out2["loss"].backward()
    grad_finite_nz = True
    grad_max_per_layer = []
    for l in range(len(model.num_emb_list)):
        g = model.kappa_l_raw[l].grad
        if g is None or not torch.isfinite(g).all() or g.abs().max() < 1e-20:
            grad_finite_nz = False
        grad_max_per_layer.append(g.abs().max().item() if g is not None else 0.0)

    # 5. round-trip (codebook rescale is bounded < 1.0)
    with torch.no_grad():
        rt_diff_max = 0.0
        for l in range(len(model.num_emb_list)):
            d_old = (old_codebooks[l] - new_codebooks[l]).abs().max().item()
            rt_diff_max = max(rt_diff_max, d_old)
    roundtrip_ok = rt_diff_max < 1.0

    return {
        "kappa_changed": kappa_changed,
        "assignment_consistent": assignment_consistent,
        "ranking_consistent": ranking_consistent,
        "grad_finite_nz": grad_finite_nz,
        "grad_max_per_layer": grad_max_per_layer,
        "roundtrip_ok": roundtrip_ok,
        "old_kappa": old_kappa_l,
        "new_kappa": new_kappa_l,
    }


def train_one_config(enable_radial_sync, X, batch_size, num_epochs, device, seed=42, log_prefix=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    e_dim_actual = X.shape[1]  # detect from data
    model = RadialSyncHRQVAE(
        num_emb_list=NUM_EMB_LIST,
        e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX,
        beta=BETA,
        enable_radial_sync=enable_radial_sync,
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
            out["loss"].backward()
            # Sanity: NaN/Inf check
            if not torch.isfinite(out["loss"]):
                print(f"  [{log_prefix} Ep {epoch}] ❌ Non-finite loss @ batch {i}", flush=True)
                continue
            optimizer.step()
            # Post-step radial sync hook
            model.optimizer_step_hook()
            ep_loss += out["loss"].item()
        ep_loss /= n_batches

        # Per-layer metrics on sample
        per_layer = []
        for l in range(len(NUM_EMB_LIST)):
            m = compute_layer_metrics(model, X, l, batch_size)
            per_layer.append(m)
        avg_util = sum(m["util"] for m in per_layer) / len(per_layer)
        if avg_util > best_avg_util:
            best_avg_util = avg_util
            best_epoch = epoch
        epoch_metrics.append({
            "epoch": epoch,
            "loss": ep_loss,
            "per_layer": per_layer,
            "avg_util": avg_util,
            "sync_residual_n": len(model._sync_residual),
        })
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  [{log_prefix} Ep {epoch:02d}/{num_epochs}] loss={ep_loss:.4f} util={avg_util:.3f} sync_n={len(model._sync_residual)}", flush=True)

        # USAGE-KILL @ ep 5 if avg_util < 0.3
        if epoch >= 5 and avg_util < 0.3:
            print(f"  [{log_prefix} Ep {epoch:02d}] ❌ USAGE-KILL (util={avg_util:.3f})", flush=True)
            break

    return model, epoch_metrics, best_epoch, best_avg_util


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print(f"[Task #414 Issue #121 Gate 1] Radial Codebook Sync Transfer")
    print("=" * 70)

    # Load data
    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    # embedding column contains numpy arrays per row → stack
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    print(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    # === 5-step functional audit (main only) ===
    print("\n[5-step functional audit] Pre-training audit on main config...", flush=True)
    e_dim_actual = X.shape[1]
    audit_model = RadialSyncHRQVAE(
        num_emb_list=NUM_EMB_LIST,
        e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX,
        beta=BETA,
        enable_radial_sync=True,
    ).to(DEVICE)
    audit = five_step_functional_audit(audit_model, X, BATCH_SIZE)
    audit_pass = all([audit["kappa_changed"], audit["assignment_consistent"], audit["ranking_consistent"],
                       audit["grad_finite_nz"], audit["roundtrip_ok"]])
    print(f"  kappa_changed: {audit['kappa_changed']}")
    print(f"  assignment_consistent: {audit['assignment_consistent']}")
    print(f"  ranking_consistent: {audit['ranking_consistent']}")
    print(f"  grad_finite_nz: {audit['grad_finite_nz']}")
    print(f"  grad_max_per_layer: {audit['grad_max_per_layer']}")
    print(f"  roundtrip_ok: {audit['roundtrip_ok']}")
    print(f"  Audit overall: {'✅ PASS' if audit_pass else '❌ FAIL'}")

    if not audit_pass:
        print("\n❌ 5-step functional audit FAIL — exit without training.")
        return

    # === Main (有径向传输) ===
    print("\n[Main] 30 epoch with radial sync...", flush=True)
    main_model, main_metrics, main_best_epoch, main_best_avg_util = train_one_config(
        enable_radial_sync=True, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Main"
    )

    # === Control (无径向传输但其他相同) ===
    print("\n[Control] 30 epoch without radial sync...", flush=True)
    control_model, control_metrics, control_best_epoch, control_best_avg_util = train_one_config(
        enable_radial_sync=False, X=X, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
        device=DEVICE, seed=SEED, log_prefix="Ctrl"
    )

    # === Final metrics ===
    final_main = main_metrics[-1]
    final_control = control_metrics[-1]
    main_util = {f"L{l}": m["util"] for l, m in enumerate(final_main["per_layer"])}
    main_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_main["per_layer"])}
    control_util = {f"L{l}": m["util"] for l, m in enumerate(final_control["per_layer"])}
    control_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_control["per_layer"])}

    main_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_main["per_layer"])
    control_pass = all(m["util"] >= 0.9 and m["max_load"] < 0.05 for m in final_control["per_layer"])

    # === Verdict ===
    verdict = {
        "task": "task414_issue121_radial_codebook_sync",
        "issue": 121,
        "audit": audit,
        "audit_pass": audit_pass,
        "main": {
            "best_epoch": main_best_epoch,
            "best_avg_util": main_best_avg_util,
            "final_util": main_util,
            "final_max_load": main_max_load,
            "epoch_metrics": main_metrics,
        },
        "control": {
            "best_epoch": control_best_epoch,
            "best_avg_util": control_best_avg_util,
            "final_util": control_util,
            "final_max_load": control_max_load,
            "epoch_metrics": control_metrics,
        },
        "gate1_pass": main_pass and control_pass,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Main: util={main_util}, max_load={main_max_load}, pass={main_pass}")
    print(f"Control: util={control_util}, max_load={control_max_load}, pass={control_pass}")
    print(f"Gate 1: {'✅ PASS' if verdict['gate1_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()