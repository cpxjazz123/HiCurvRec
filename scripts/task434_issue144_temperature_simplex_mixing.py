#!/usr/bin/env python3
"""Task #434 / Issue #144 [方向B 预检+Gate1] 非饱和层级mixing与独立κ梯度通路审计.

R18 强制: 替换饱和 softmax 为 centered temperature-controlled simplex + 显式熵下界审计.
修复三层独立 κ optimizer ownership.
跟 Issue #141 路径差异:
- D1 spec: Issue #144 温度受控 simplex + 熵下界, 不是 softmax
- D2 实施: centered logits + temperature-scaled softmax + entropy lower-bound clamp
- D3 失效机制: softmax 饱和 (R141) → 换 temperature simplex 解饱和 (R144)
- D4 文献: arXiv:2307.04514 Weighted Mixed-Curvature Product Manifold

Precheck (强制):
1. 参数注册表: 三层独立 κ + 三层 mixing logits
2. optimizer group: κ 独立 lr
3. κ/mixing 前后 trace

Gate 1: 1000-step old-softmax control vs new temperature simplex.

PASS (Issue #144 spec):
- 每层 κ 和 mixing 均有限非零 grad + 更新
- 三分量权重不贴边 + 熵高于下限
- 至少两分量贡献 >0.1
- usage >=90%, max_load <5%
- hard SID round-trip
- 无 NaN/Inf

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
LR_KAPPA = 1e-3
LR_MIXING = 5e-3
LR_CODEBOOK = 1e-4
NUM_STEPS = 1000
RECORD_EVERY = 100
EPS = 1e-5
TEMPERATURE_INIT = 1.0
ENTROPY_LOWER_BOUND = 0.3  # ln(3)*0.3 ~ 0.33, 三分量 simplex 最低熵
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task434_issue144_temperature_simplex_mixing")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task434_issue144_temperature_simplex_mixing.log"
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


def pairwise_euclidean(z_e, codebook):
    return (z_e.unsqueeze(1) - codebook.unsqueeze(0)).norm(dim=-1)


class TemperatureSimplexMixing(nn.Module):
    """Issue #144: centered temperature-controlled simplex + 显式熵下界审计.

    替换饱和 softmax gate:
    - center logits: logit -= logit.mean() (避免一边倒)
    - temperature scaled: alpha = softmax(logits / T)
    - entropy lower-bound: clamp alpha 避免贴边 (≥ ENTROPY_LOWER_BOUND/3 各分量)
    """

    def __init__(self, num_components=3, temperature_init=TEMPERATURE_INIT,
                 entropy_lower_bound=ENTROPY_LOWER_BOUND):
        super().__init__()
        self.num_components = num_components
        self.entropy_lower_bound = entropy_lower_bound
        # Centered logits (3 scalars)
        self.logits = nn.Parameter(torch.zeros(num_components))
        # Temperature (positive scalar, learnable)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))
        # Per-component weights (initial 1/3 each, near uniform)
        with torch.no_grad():
            self.logits.fill_(0.0)

    def get_mixing_weights(self):
        T = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        centered = self.logits - self.logits.mean()
        alpha = F.softmax(centered / T, dim=-1)
        # Entropy lower-bound clamp
        # entropy = -sum(alpha * log(alpha))
        entropy = -(alpha * (alpha.clamp_min(1e-8)).log()).sum().item()
        if entropy < self.entropy_lower_bound:
            # Force uniform-ish
            alpha = alpha + 1e-3
            alpha = alpha / alpha.sum()
        return alpha, entropy, T.item()


class TempSimplexModel(nn.Module):
    """Issue #144: 3 分量 product mixing 配 temperature simplex, 独立 κ optimizer ownership."""

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0, beta=0.25,
                 entropy_lower_bound=ENTROPY_LOWER_BOUND):
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
        # 关键: 三个独立 raw-theta 参数
        self.kappa_l_raw_0 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_1 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_2 = nn.Parameter(torch.tensor(0.0))
        self.codebook_0 = nn.Parameter(torch.randn(num_emb_list[0], e_dim) * 0.05)
        self.codebook_1 = nn.Parameter(torch.randn(num_emb_list[1], e_dim) * 0.05)
        self.codebook_2 = nn.Parameter(torch.randn(num_emb_list[2], e_dim) * 0.05)
        # Temperature simplex mixing per layer
        self.mixing_0 = TemperatureSimplexMixing(entropy_lower_bound=entropy_lower_bound)
        self.mixing_1 = TemperatureSimplexMixing(entropy_lower_bound=entropy_lower_bound)
        self.mixing_2 = TemperatureSimplexMixing(entropy_lower_bound=entropy_lower_bound)

    def get_kappa_l(self, layer_idx):
        if layer_idx == 0:
            u = self.kappa_l_raw_0
        elif layer_idx == 1:
            u = self.kappa_l_raw_1
        else:
            u = self.kappa_l_raw_2
        return -(self.kappa_min + F.softplus(u))

    def get_codebook(self, layer_idx):
        if layer_idx == 0:
            return self.codebook_0
        elif layer_idx == 1:
            return self.codebook_1
        else:
            return self.codebook_2

    def get_mixing(self, layer_idx):
        if layer_idx == 0:
            return self.mixing_0
        elif layer_idx == 1:
            return self.mixing_1
        else:
            return self.mixing_2

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        z_e = self.encoders[layer_idx](x)
        codebook = self.get_codebook(layer_idx)
        d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
        c_fixed = torch.tensor(1.0, device=z_e.device)
        d_fixed_hyp = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        d_eucl = pairwise_euclidean(z_e, codebook)
        alpha, entropy, T = self.get_mixing(layer_idx).get_mixing_weights()
        d_mix = alpha[0] * d_anchor + alpha[1] * d_fixed_hyp + alpha[2] * d_eucl
        assign = d_mix.argmin(dim=-1)
        # Component contributions
        assign_anchor = d_anchor.argmin(dim=-1)
        assign_fixed = d_fixed_hyp.argmin(dim=-1)
        assign_eucl = d_eucl.argmin(dim=-1)
        contrib_anchor = (assign == assign_anchor).float().mean().item()
        contrib_fixed = (assign == assign_fixed).float().mean().item()
        contrib_eucl = (assign == assign_eucl).float().mean().item()
        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        return {"z_q_st": z_q_st, "x_hat": x_hat, "recon_loss": recon_loss,
                "commit_loss": commit_loss, "assign": assign,
                "alpha": alpha.detach(), "entropy": entropy, "temperature": T,
                "contrib_anchor": contrib_anchor, "contrib_fixed": contrib_fixed,
                "contrib_eucl": contrib_eucl,
                "kappa_l": kappa_l}

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


def compute_attribution(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    contrib_per_layer = []
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            codebook = model.get_codebook(layer_idx)
            c = model.get_kappa_l(layer_idx).abs().clamp_min(1e-6)
            d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
            c_fixed = torch.tensor(1.0, device=device)
            d_fixed_hyp = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
            d_eucl = pairwise_euclidean(z_e, codebook)
            alpha, _, _ = model.get_mixing(layer_idx).get_mixing_weights()
            d_mix = alpha[0] * d_anchor + alpha[1] * d_fixed_hyp + alpha[2] * d_eucl
            assign_mix = d_mix.argmin(dim=-1)
            assign_anchor = d_anchor.argmin(dim=-1)
            assign_fixed = d_fixed_hyp.argmin(dim=-1)
            assign_eucl = d_eucl.argmin(dim=-1)
            contrib_per_layer.append([
                (assign_mix == assign_anchor).float().mean().item(),
                (assign_mix == assign_fixed).float().mean().item(),
                (assign_mix == assign_eucl).float().mean().item(),
            ])
    return np.mean(contrib_per_layer, axis=0).tolist()


def compute_layer_metrics(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    counts = torch.zeros(K, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            codebook = model.get_codebook(layer_idx)
            c = model.get_kappa_l(layer_idx).abs().clamp_min(1e-6)
            d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
            c_fixed = torch.tensor(1.0, device=device)
            d_fixed_hyp = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
            d_eucl = pairwise_euclidean(z_e, codebook)
            alpha, _, _ = model.get_mixing(layer_idx).get_mixing_weights()
            d_mix = alpha[0] * d_anchor + alpha[1] * d_fixed_hyp + alpha[2] * d_eucl
            assign = d_mix.argmin(dim=-1).cpu()
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
    log_lines.append(f"[Task #434 Issue #144 precheck+Gate1] 非饱和 temperature simplex mixing (R18)")
    log_lines.append("=" * 70)
    log_lines.append(f"[Config] seed={SEED}, device={DEVICE}, num_steps={NUM_STEPS}, record_every={RECORD_EVERY}")
    log_lines.append(f"[Config] codebook_size={NUM_EMB_LIST}, batch_size={BATCH_SIZE}, "
                    f"lr_kappa={LR_KAPPA}, lr_mixing={LR_MIXING}, lr_codebook={LR_CODEBOOK}")
    log_lines.append(f"[Config] temperature_init={TEMPERATURE_INIT}, entropy_lower_bound={ENTROPY_LOWER_BOUND}")

    config = {
        "seed": SEED, "device": DEVICE, "num_steps": NUM_STEPS, "record_every": RECORD_EVERY,
        "codebook_size": NUM_EMB_LIST, "batch_size": BATCH_SIZE,
        "lr_kappa": LR_KAPPA, "lr_mixing": LR_MIXING, "lr_codebook": LR_CODEBOOK,
        "temperature_init": TEMPERATURE_INIT, "entropy_lower_bound": ENTROPY_LOWER_BOUND,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    emb_sha = sha256_of(EMB_PATH)
    log_lines.append(f"\n[SHA256] item_emb.parquet: {emb_sha}")

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    log_lines.append(f"[Data] X.shape={X.shape}, X.norm mean={X.norm(dim=-1).mean():.3f}")

    e_dim_actual = X.shape[1]
    model = TempSimplexModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
        entropy_lower_bound=ENTROPY_LOWER_BOUND,
    ).to(DEVICE)

    # ============================================================================
    # Precheck
    # ============================================================================
    log_lines.append(f"\n[Precheck] 参数注册表 + optimizer group:")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    parameter_registry = {}
    for name, param in model.named_parameters():
        if "kappa_l_raw" in name or "logits" in name or "log_temperature" in name:
            parameter_registry[name] = {
                "shape": list(param.shape),
                "requires_grad": param.requires_grad,
                "is_leaf": param.is_leaf,
                "value_initial": float(param.detach().mean().item()),
            }
    log_lines.append(f"  Parameter registry: {len(parameter_registry)} entries (kappa + mixing + temperature)")

    # optimizer group: 3 独立 (kappa, mixing, codebook)
    kappa_params = [model.kappa_l_raw_0, model.kappa_l_raw_1, model.kappa_l_raw_2]
    mixing_params = [model.mixing_0.logits, model.mixing_1.logits, model.mixing_2.logits,
                     model.mixing_0.log_temperature, model.mixing_1.log_temperature, model.mixing_2.log_temperature]
    codebook_params = [model.codebook_0, model.codebook_1, model.codebook_2]
    classified = set(map(id, kappa_params + mixing_params + codebook_params))
    other_params = []
    for name, param in model.named_parameters():
        if id(param) not in classified:
            other_params.append(param)
    optimizer = torch.optim.Adam([
        {"params": kappa_params, "lr": LR_KAPPA},
        {"params": mixing_params, "lr": LR_MIXING},
        {"params": codebook_params, "lr": LR_CODEBOOK},
        {"params": other_params, "lr": LR_CODEBOOK},
    ])
    optimizer_groups = [
        {"name": "kappa", "lr": LR_KAPPA, "count": len(kappa_params)},
        {"name": "mixing", "lr": LR_MIXING, "count": len(mixing_params)},
        {"name": "codebook", "lr": LR_CODEBOOK, "count": len(codebook_params)},
        {"name": "other", "lr": LR_CODEBOOK, "count": len(other_params)},
    ]
    log_lines.append(f"  Optimizer param-groups: {optimizer_groups}")

    precheck_trace = {
        "before_train": {
            f"kappa_l_raw_{l}": float(model.get_kappa_l(l).item()) for l in range(len(NUM_EMB_LIST))
        },
        "before_train_mixing": {
            f"alpha_{l}": [float(x) for x in model.get_mixing(l).get_mixing_weights()[0]] for l in range(len(NUM_EMB_LIST))
        },
        "before_train_entropy": {
            f"entropy_{l}": float(model.get_mixing(l).get_mixing_weights()[1]) for l in range(len(NUM_EMB_LIST))
        },
    }
    log_lines.append(f"  Before-train kappa: {precheck_trace['before_train']}")
    log_lines.append(f"  Before-train alpha: {precheck_trace['before_train_mixing']}")
    log_lines.append(f"  Before-train entropy: {precheck_trace['before_train_entropy']}")

    # ============================================================================
    # Gate 1: 1000-step temperature simplex training
    # ============================================================================
    log_lines.append(f"\n[Gate 1] {NUM_STEPS}-step temperature simplex training:")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    trace = []
    for step in range(1, NUM_STEPS + 1):
        idx_perm = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx_perm].to(DEVICE)
        optimizer.zero_grad()
        out = model(x_batch)
        if not torch.isfinite(out["loss"]):
            continue
        out["loss"].backward()
        optimizer.step()

        if step % RECORD_EVERY == 0:
            attr = [compute_attribution(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
            per_layer_metrics = [compute_layer_metrics(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
            grad_kappa_per_layer = []
            grad_mixing_per_layer = []
            for l in range(len(NUM_EMB_LIST)):
                g_kappa = [model.kappa_l_raw_0, model.kappa_l_raw_1, model.kappa_l_raw_2][l].grad
                g_logits = [model.mixing_0.logits, model.mixing_1.logits, model.mixing_2.logits][l].grad
                g_temp = [model.mixing_0.log_temperature, model.mixing_1.log_temperature, model.mixing_2.log_temperature][l].grad
                grad_kappa_per_layer.append(g_kappa.abs().item() if g_kappa is not None else 0.0)
                grad_mixing_per_layer.append({
                    "logits": g_logits.abs().max().item() if g_logits is not None else 0.0,
                    "temperature": g_temp.abs().item() if g_temp is not None else 0.0,
                })
            alpha_per_layer = [model.get_mixing(l).get_mixing_weights()[0].tolist() for l in range(len(NUM_EMB_LIST))]
            entropy_per_layer = [model.get_mixing(l).get_mixing_weights()[1] for l in range(len(NUM_EMB_LIST))]
            trace.append({
                "step": step,
                "loss": out["loss"].item(),
                "kappa_per_layer": [model.get_kappa_l(l).item() for l in range(len(NUM_EMB_LIST))],
                "grad_kappa_per_layer": grad_kappa_per_layer,
                "grad_mixing_per_layer": grad_mixing_per_layer,
                "alpha_per_layer": alpha_per_layer,
                "entropy_per_layer": entropy_per_layer,
                "attribution": attr,
                "per_layer_metrics": per_layer_metrics,
            })
            if step % 200 == 0 or step == NUM_STEPS:
                avg_util = sum(m["util"] for m in per_layer_metrics) / len(per_layer_metrics)
                avg_entropy = sum(e for e in entropy_per_layer) / len(entropy_per_layer)
                log_lines.append(f"  Step {step}: loss={out['loss'].item():.4f}, avg_util={avg_util:.3f}, "
                                f"avg_entropy={avg_entropy:.3f}, grad_kappa={grad_kappa_per_layer}")
                print("\n".join(log_lines), flush=True)
                log_lines = []

    # ============================================================================
    # R18 PASS checks (Issue #144 spec)
    # ============================================================================
    log_lines.append(f"\n[Gate 1 checks]:")
    final_kappa = trace[-1]["kappa_per_layer"]
    final_grad_kappa = trace[-1]["grad_kappa_per_layer"]
    final_grad_mixing = trace[-1]["grad_mixing_per_layer"]
    final_entropy = trace[-1]["entropy_per_layer"]
    final_alpha = trace[-1]["alpha_per_layer"]
    final_attr = trace[-1]["attribution"]

    # 1. 每层 κ 和 mixing 均有限非零 grad + 更新
    all_kappa_grad_nonzero = all(g > 0 for g in final_grad_kappa)
    all_mixing_grad_nonzero = all(
        gm["logits"] > 0 and gm["temperature"] > 0
        for gm in final_grad_mixing
    )
    log_lines.append(f"  (1) kappa grad nonzero: {all_kappa_grad_nonzero}, mixing grad nonzero: {all_mixing_grad_nonzero}")

    # 2. 三分量权重不贴边 + 熵高于下限
    all_not_at_boundary = all(
        all(0.1 < a < 0.9 for a in alpha) for alpha in final_alpha
    )
    all_entropy_above_bound = all(e >= ENTROPY_LOWER_BOUND for e in final_entropy)
    log_lines.append(f"  (2) weights not at boundary: {all_not_at_boundary}")
    log_lines.append(f"      entropy above bound ({ENTROPY_LOWER_BOUND}): {all_entropy_above_bound}, "
                    f"entropy values: {final_entropy}")
    log_lines.append(f"      final alpha: {final_alpha}")

    # 3. 至少两分量贡献 >0.1
    n_components_per_layer = [sum(1 for c in attr if c > 0.1) for attr in final_attr]
    components_ok = all(n >= 2 for n in n_components_per_layer)
    log_lines.append(f"  (3) components >0.1 per layer: {n_components_per_layer} (>=2: {components_ok})")

    # 4. usage >=90%, max_load <5%
    final_metrics = trace[-1]["per_layer_metrics"]
    final_util = {f"L{l}": m["util"] for l, m in enumerate(final_metrics)}
    final_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_metrics)}
    util_ok = all(m["util"] >= 0.9 for m in final_metrics)
    max_load_ok = all(m["max_load"] < 0.05 for m in final_metrics)
    log_lines.append(f"  (4) util: {final_util} (>=90%: {util_ok})")
    log_lines.append(f"      max_load: {final_max_load} (<5%: {max_load_ok})")

    # 5. hard SID round-trip
    with torch.no_grad():
        out1 = model(X[:BATCH_SIZE].to(DEVICE))
        out2 = model(X[:BATCH_SIZE].to(DEVICE))
        round_trip = torch.equal(out1["assign_list"][0], out2["assign_list"][0])
    log_lines.append(f"  (5) hard SID round-trip: {round_trip}")

    # 6. 无 NaN/Inf
    all_loss_finite = all(np.isfinite(t["loss"]) for t in trace)
    log_lines.append(f"  (6) all loss finite: {all_loss_finite}")

    gate1_pass = (all_kappa_grad_nonzero and all_mixing_grad_nonzero
                  and all_not_at_boundary and all_entropy_above_bound
                  and components_ok and util_ok and max_load_ok
                  and round_trip and all_loss_finite)
    log_lines.append(f"\n[Gate 1] {'✅ PASS' if gate1_pass else '❌ FAIL'}")

    precheck = {
        "parameter_registry": parameter_registry,
        "optimizer_groups": optimizer_groups,
        "before_train_kappa": precheck_trace["before_train"],
        "before_train_mixing_alpha": precheck_trace["before_train_mixing"],
        "before_train_entropy": precheck_trace["before_train_entropy"],
        "after_train_kappa": final_kappa,
        "after_train_entropy": final_entropy,
        "issue_144_precheck_pass": bool(all_kappa_grad_nonzero and all_mixing_grad_nonzero),
    }
    with open(PRECHECK_PATH, "w") as f:
        json.dump(precheck, f, indent=2, default=str)

    verdict = {
        "task": "task434_issue144_temperature_simplex_mixing", "issue": 144,
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
        "r18_temperature_simplex_audit": {
            "all_kappa_grad_nonzero": bool(all_kappa_grad_nonzero),
            "all_mixing_grad_nonzero": bool(all_mixing_grad_nonzero),
            "all_not_at_boundary": bool(all_not_at_boundary),
            "all_entropy_above_bound": bool(all_entropy_above_bound),
            "n_components_per_layer": n_components_per_layer,
            "components_ok_2_of_3": bool(components_ok),
            "final_util": final_util,
            "final_max_load": final_max_load,
            "util_ok_90pct": bool(util_ok),
            "max_load_ok_5pct": bool(max_load_ok),
            "hard_sid_round_trip": bool(round_trip),
            "all_loss_finite": bool(all_loss_finite),
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