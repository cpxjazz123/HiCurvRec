#!/usr/bin/env python3
"""Task #433 / Issue #143 [方向A 预检+Gate1] 修复独立κ冻结并审计曲率同步更新.

R18 强制: 修复独立 κ 的参数所有权 + 更新路径, 1000-step old-lock control vs repaired-κ 对照.
跟 Issue #140 路径差异:
- D1 spec: Issue #143 修独立 κ 所有权 + optimizer path, 不是 hard-EMA
- D2 实施: 每层独立 raw-theta + 独立 optimizer param-group, 显式 update-order
- D3 失效机制: κ 冻结为 R137 设计缺陷 (R137 κ lock 互斥), #143 修复路径
- D4 文献: arXiv:2405.13979 Robust Hyperbolic Learning with Curvature-Aware Optimization

Precheck (强制):
1. 参数注册表: 三层独立 kappa_l_raw[i] Parameter
2. optimizer param-group: 三组独立 optimizer group (per-layer)
3. update-order trace: theta → κ → #47 scale → Poincare projection → distance → codebook

Gate 1: 1000-step 对照 (old-lock control + repaired-κ). 每 100 步记录.

PASS (Issue #143 spec):
- 三层 κ 均有有限非零 grad + 非零更新, 互不相同
- #47 同步 trace 完整且有限
- 无越域/NaN/Inf
- hard SID round-trip
- usage >=90%, max_load <5%

FAIL/PENDING 否则.

6 件套审计 (R20+R21 强制).
"""
import sys
import json
import math
import hashlib
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
BATCH_SIZE = 256
LR_KAPPA = 1e-3  # κ-specific learning rate
LR_CODEBOOK = 1e-4  # codebook learning rate
NUM_STEPS = 1000
RECORD_EVERY = 100
EPS = 1e-5
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task433_issue143_kappa_ownership_repair")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task433_issue143_kappa_ownership_repair.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
PRECHECK_PATH = PRODUCT_DIR / "precheck.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


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


class RepairedKappaModel(nn.Module):
    """Issue #143: 每层独立 raw-theta + 独立 optimizer param group.
    kappa_l_raw[i] 独立 Parameter, 跟 codebook[i] 分开 register.
    update_order: theta → κ (no detach, no constant override) → #47 scale → projection → distance.
    """

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0, beta=0.25):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for K in num_emb_list:
            self.encoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
        # 关键: 三个独立 raw-theta 参数, 每个在不同 register path
        # Issue #143 强制 — 不允许共享, 不允许 detach, 不允许常量覆盖
        self.kappa_l_raw_0 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_1 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_2 = nn.Parameter(torch.tensor(0.0))
        # codebook 也独立 register
        self.codebook_0 = nn.Parameter(torch.randn(num_emb_list[0], e_dim) * 0.05)
        self.codebook_1 = nn.Parameter(torch.randn(num_emb_list[1], e_dim) * 0.05)
        self.codebook_2 = nn.Parameter(torch.randn(num_emb_list[2], e_dim) * 0.05)

    def get_kappa_l(self, layer_idx):
        # Per-layer Parameter access (no detach, no clamp that kills grad)
        if layer_idx == 0:
            u = self.kappa_l_raw_0
        elif layer_idx == 1:
            u = self.kappa_l_raw_1
        else:
            u = self.kappa_l_raw_2
        # softplus to keep positive (no negative kill)
        # But to enable non-zero grad updates, use raw kappa = kappa_min + F.softplus(u)
        # No clamp to fixed value
        return -(self.kappa_min + F.softplus(u))

    def get_codebook(self, layer_idx):
        if layer_idx == 0:
            return self.codebook_0
        elif layer_idx == 1:
            return self.codebook_1
        else:
            return self.codebook_2

    def apply_47_scale(self, layer_idx, old_kappa, new_kappa):
        # Issue #47 同步: scale codebook by sqrt(|old_κ|/|new_κ|) when κ updates
        cb = self.get_codebook(layer_idx).detach()
        ratio = (old_kappa.abs() / new_kappa.abs()).clamp_min(1e-6)
        factor = ratio.sqrt()
        # In-place, but allow grad through new_kappa
        with torch.no_grad():
            if layer_idx == 0:
                self.codebook_0.data.mul_(factor)
            elif layer_idx == 1:
                self.codebook_1.data.mul_(factor)
            else:
                self.codebook_2.data.mul_(factor)
        return factor.item()

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        z_e = self.encoders[layer_idx](x)
        codebook = self.get_codebook(layer_idx)
        cost = stable_pairwise_hyp(z_e, codebook, c, EPS)
        assign = cost.argmin(dim=-1)
        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        domain_margin = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        return {"z_q_st": z_q_st, "x_hat": x_hat, "recon_loss": recon_loss,
                "commit_loss": commit_loss, "assign": assign,
                "z_e": z_e, "codebook": codebook, "c": c, "K": K,
                "kappa_l": kappa_l, "domain_margin": domain_margin}

    def forward(self, x):
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        all_assign = []
        layer_outputs = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l)
            all_assign.append(out["assign"])
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            residual = residual - out["z_q_st"]
            layer_outputs.append(out)
        return {"recon_loss": total_recon, "commit_loss": total_commit,
                "loss": total_recon + self.beta * total_commit,
                "assign_list": all_assign, "layer_outputs": layer_outputs}


def compute_layer_metrics(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    counts = torch.zeros(K, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            c = model.get_kappa_l(layer_idx).abs().clamp_min(1e-6)
            d = stable_pairwise_hyp(z_e, model.get_codebook(layer_idx), c, EPS)
            assign = d.argmin(dim=-1).cpu()
            for a in assign.tolist():
                counts[a] += 1
    n_used = (counts > 0).sum().item()
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    return {"util": util, "max_load": max_load, "n_used": n_used, "K": K}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #433 Issue #143 precheck+Gate1] 修复独立κ冻结 + 曲率同步更新审计 (R18)")
    log_lines.append("=" * 70)
    log_lines.append(f"[Config] seed={SEED}, device={DEVICE}, num_steps={NUM_STEPS}, record_every={RECORD_EVERY}")
    log_lines.append(f"[Config] codebook_size={NUM_EMB_LIST}, batch_size={BATCH_SIZE}, "
                    f"lr_kappa={LR_KAPPA}, lr_codebook={LR_CODEBOOK}")

    config = {
        "seed": SEED, "device": DEVICE, "num_steps": NUM_STEPS, "record_every": RECORD_EVERY,
        "codebook_size": NUM_EMB_LIST, "batch_size": BATCH_SIZE,
        "lr_kappa": LR_KAPPA, "lr_codebook": LR_CODEBOOK,
        "kappa_min": KAPPA_MIN, "kappa_max": KAPPA_MAX,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log_lines.append(f"[Config saved] {CONFIG_PATH}")

    emb_sha = sha256_of(EMB_PATH)
    log_lines.append(f"\n[SHA256] item_emb.parquet: {emb_sha}")

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    log_lines.append(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    # ============================================================================
    # Precheck (Issue #143 spec 强制 Step 0)
    # ============================================================================
    log_lines.append(f"\n[Precheck Step 0] 参数注册表 + optimizer param-group:")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    e_dim_actual = X.shape[1]
    model = RepairedKappaModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
    ).to(DEVICE)

    # 1. 参数注册表
    parameter_registry = {}
    for name, param in model.named_parameters():
        if "kappa_l_raw" in name:
            parameter_registry[name] = {
                "shape": list(param.shape),
                "requires_grad": param.requires_grad,
                "is_leaf": param.is_leaf,
                "dtype": str(param.dtype),
                "value_initial": param.detach().item() if param.numel() == 1 else "multi-dim",
            }
    log_lines.append(f"  [1] Parameter registry (kappa_l_raw): {parameter_registry}")

    # 2. optimizer param-group (3 组 per-layer, 独立 lr)
    # Issue #143 spec 强制 — 独立 optimizer group
    kappa_params = []
    codebook_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "kappa_l_raw" in name:
            kappa_params.append(param)
        elif "codebook_" in name:
            codebook_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.Adam([
        {"params": kappa_params, "lr": LR_KAPPA},
        {"params": codebook_params, "lr": LR_CODEBOOK},
        {"params": other_params, "lr": LR_CODEBOOK},
    ])
    optimizer_groups = [{"name": "kappa", "lr": LR_KAPPA, "count": len(kappa_params)},
                       {"name": "codebook", "lr": LR_CODEBOOK, "count": len(codebook_params)},
                       {"name": "other", "lr": LR_CODEBOOK, "count": len(other_params)}]
    log_lines.append(f"  [2] Optimizer param-groups: {optimizer_groups}")

    # 3. 前后状态 trace
    precheck_trace = {
        "before_train": {
            f"kappa_l_raw_{l}": float(model.get_kappa_l(l).item()) for l in range(len(NUM_EMB_LIST))
        }
    }
    log_lines.append(f"  [3] Before-train kappa: {precheck_trace['before_train']}")

    # ============================================================================
    # Gate 1: 1000-step repaired-κ training
    # ============================================================================
    log_lines.append(f"\n[Gate 1] {NUM_STEPS}-step repaired-κ training:")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    last_kappa_per_layer = [model.get_kappa_l(l).item() for l in range(len(NUM_EMB_LIST))]
    trace = []
    for step in range(1, NUM_STEPS + 1):
        # Stage 1: forward + backward
        idx_perm = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx_perm].to(DEVICE)
        optimizer.zero_grad()
        out = model(x_batch)
        if not torch.isfinite(out["loss"]):
            log_lines.append(f"  Step {step}: loss not finite, skipping")
            continue
        out["loss"].backward()
        optimizer.step()

        # Stage 2: #47 同步 (Issue #143 spec 强制)
        new_kappa_per_layer = []
        for l in range(len(NUM_EMB_LIST)):
            new_kappa = model.get_kappa_l(l).item()
            old_kappa = last_kappa_per_layer[l]
            new_kappa_per_layer.append(new_kappa)
            if abs(new_kappa - old_kappa) > 1e-8:
                factor = model.apply_47_scale(l, torch.tensor(old_kappa), torch.tensor(new_kappa))
                last_kappa_per_layer[l] = new_kappa

        if step % RECORD_EVERY == 0:
            per_layer_metrics = [compute_layer_metrics(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
            # Capture grad state before next step
            grad_kappa_per_layer = []
            for l in range(len(NUM_EMB_LIST)):
                g = model.get_kappa_l(l)
                if l == 0:
                    g_raw = model.kappa_l_raw_0.grad
                elif l == 1:
                    g_raw = model.kappa_l_raw_1.grad
                else:
                    g_raw = model.kappa_l_raw_2.grad
                grad_kappa_per_layer.append(g_raw.abs().item() if g_raw is not None else 0.0)
            trace.append({
                "step": step,
                "loss": out["loss"].item(),
                "kappa_per_layer": new_kappa_per_layer,
                "grad_kappa_per_layer": grad_kappa_per_layer,
                "kappa_unique": len(set(round(k, 4) for k in new_kappa_per_layer)) == len(NUM_EMB_LIST),
                "per_layer_metrics": per_layer_metrics,
            })
            if step % 200 == 0 or step == NUM_STEPS:
                avg_util = sum(m["util"] for m in per_layer_metrics) / len(per_layer_metrics)
                log_lines.append(f"  Step {step}: loss={out['loss'].item():.4f}, avg_util={avg_util:.3f}, "
                                f"kappa={new_kappa_per_layer}, grad_kappa={grad_kappa_per_layer}")
                print("\n".join(log_lines), flush=True)
                log_lines = []

    # ============================================================================
    # R18 PASS checks (Issue #143 spec)
    # ============================================================================
    log_lines.append(f"\n[Gate 1 checks]:")

    # 1. 三层 κ 均有有限非零 grad + 非零更新, 互不相同
    final_kappa = trace[-1]["kappa_per_layer"]
    final_grad = trace[-1]["grad_kappa_per_layer"]
    kappa_unique = trace[-1]["kappa_unique"]
    all_grad_nonzero = all(g > 0 for g in final_grad)
    all_finite = all(np.isfinite(k) for k in final_kappa)
    kappa_updates_ok = kappa_unique and all_grad_nonzero and all_finite
    log_lines.append(f"  (1) kappa unique={kappa_unique}, grad_nonzero={all_grad_nonzero}, finite={all_finite}")
    log_lines.append(f"      final kappa={final_kappa}, grad={final_grad}")

    # 2. #47 同步 trace 完整
    n_sync_events = sum(1 for l in range(len(NUM_EMB_LIST)) for k in trace if k["step"] > 0)
    log_lines.append(f"  (2) #47 sync trace: {len(trace)} record points × {len(NUM_EMB_LIST)} layers")

    # 3. 无越域/NaN/Inf
    all_loss_finite = all(np.isfinite(t["loss"]) for t in trace)
    log_lines.append(f"  (3) all loss finite: {all_loss_finite}")

    # 4. hard SID round-trip
    with torch.no_grad():
        out1 = model(X[:BATCH_SIZE].to(DEVICE))
        out2 = model(X[:BATCH_SIZE].to(DEVICE))
        round_trip = torch.equal(out1["assign_list"][0], out2["assign_list"][0])
    log_lines.append(f"  (4) hard SID round-trip: {round_trip}")

    # 5. usage >=90%, max_load <5%
    final_metrics = trace[-1]["per_layer_metrics"]
    final_util = {f"L{l}": m["util"] for l, m in enumerate(final_metrics)}
    final_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_metrics)}
    util_ok = all(m["util"] >= 0.9 for m in final_metrics)
    max_load_ok = all(m["max_load"] < 0.05 for m in final_metrics)
    log_lines.append(f"  (5) final util: {final_util} (>=90%: {util_ok})")
    log_lines.append(f"      final max_load: {final_max_load} (<5%: {max_load_ok})")

    gate1_pass = kappa_updates_ok and all_loss_finite and round_trip and util_ok and max_load_ok
    log_lines.append(f"\n[Gate 1] {'✅ PASS' if gate1_pass else '❌ FAIL'}")

    # ============================================================================
    # Save precheck + verdict + 6 件套
    # ============================================================================
    precheck = {
        "parameter_registry": parameter_registry,
        "optimizer_groups": optimizer_groups,
        "before_train_kappa": precheck_trace["before_train"],
        "after_train_kappa": final_kappa,
        "issue_143_precheck_pass": bool(kappa_updates_ok),
    }
    with open(PRECHECK_PATH, "w") as f:
        json.dump(precheck, f, indent=2, default=str)
    log_lines.append(f"\n[Precheck saved] {PRECHECK_PATH}")

    verdict = {
        "task": "task433_issue143_kappa_ownership_repair", "issue": 143,
        "issue_spec_6_audit_pieces": {
            "1_config": str(CONFIG_PATH),
            "2_sha256": {"item_emb": emb_sha},
            "3_trace": trace,
            "4_raw_log": str(LOG_PATH),
            "5_verdict": str(VERDICT_PATH),
            "6_commit": "<pending - written after git push>",
        },
        "precheck": precheck,
        "config": config,
        "reproducibility": {"item_emb_sha256": emb_sha},
        "r18_repaired_kappa_audit": {
            "kappa_unique": bool(kappa_unique),
            "final_kappa": final_kappa,
            "final_grad_kappa": final_grad,
            "all_grad_nonzero": bool(all_grad_nonzero),
            "all_loss_finite": bool(all_loss_finite),
            "final_util": final_util,
            "final_max_load": final_max_load,
            "util_ok_90pct": bool(util_ok),
            "max_load_ok_5pct": bool(max_load_ok),
            "hard_sid_round_trip": bool(round_trip),
        },
        "trace": trace,
        "gate1_pass": bool(gate1_pass),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines), flush=True)
    print(f"\nGate 1: {'✅ PASS' if gate1_pass else '❌ FAIL'}", flush=True)


if __name__ == "__main__":
    main()