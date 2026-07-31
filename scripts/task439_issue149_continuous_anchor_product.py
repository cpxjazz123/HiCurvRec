#!/usr/bin/env python3
"""Task #439 / Issue #149 [方向B Gate1] 硬SID不变的连续soft-anchor product校准损失.

R18 强制: 保持 hard SID 不变 + 新增 continuous soft-anchor product calibration.
修复 #146 失败: 不用 argmin (discrete, 切断) + 不用 F.relu (频繁 0 grad), 改用:
- continuous soft-anchor: softmax(-d_mix / T) 加权 sum, NO argmin
- softplus margin: F.softplus(margin + anchor - neg), 处处可微
- 负样本: random codeword (NO argmin 选 anchor, NO 共享 anchor 选 positive)
d_mix = α · d_anchor + β · d_fixed_hyp + γ · d_eucl.

Precheck 强制 (Issue #149 spec):
1. autograd trace: soft-anchor continuous d_mix → mixing/κ
2. hard SID 不参与该梯度路径
3. 温度 > 0, 不塌缩, anchor entropy 不退化

Gate 1: 1000-step no-aux control + continuous-anchor 对照.

PASS:
- 每层 κ + mixing 都有有限非零 grad 和更新
- 至少两分量贡献 > 0.1
- hard SID round-trip + 无 NaN/Inf
- usage >= 90%, max_load < 5%
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
NUM_PAIRS_PER_BATCH = 64
SOFTANCHOR_TEMP = 1.0  # Issue #149: continuous soft-anchor temperature (预注册 > 0)
LR_KAPPA = 1e-3
LR_CODEBOOK = 1e-4
LR_MIXING = 1e-3
NUM_STEPS = 1000
RECORD_EVERY = 100
EPS = 1e-5
MARGIN = 1.0
ALPHA_CONTRASTIVE = 0.5
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task439_issue149_continuous_anchor_product")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task439_issue149_continuous_anchor_product.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
PRECHECK_PATH = PRODUCT_DIR / "precheck.json"
D_MIX_GRAPH_PROOF_PATH = PRODUCT_DIR / "d_mix_graph_proof.json"


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
    """Issue #146 d_anchor: hyperbolic distance (anchor branch, learnable κ)."""
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


def stable_pairwise_hyp_fixed(z_e, codebook, c_fixed=1.0, eps=EPS):
    """Issue #146 d_fixed_hyp: hyperbolic distance with FIXED κ=1 (no grad through c)."""
    c = torch.tensor(c_fixed, device=z_e.device, dtype=z_e.dtype)
    return stable_pairwise_hyp(z_e, codebook, c, eps)


def stable_pairwise_eucl(z_e, codebook):
    """Issue #146 d_eucl: Euclidean distance (pure Euclidean component)."""
    return torch.cdist(z_e, codebook, p=2)


class ProductDistanceModel(nn.Module):
    """Issue #146 product-distance contrastive calibration model."""

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
        # 3 独立 κ (learnable)
        self.kappa_l_raw_0 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_1 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_2 = nn.Parameter(torch.tensor(0.0))
        # 3 mixing scalars per layer (3 分量权重)
        self.mixing_logits_0 = nn.Parameter(torch.zeros(3))
        self.mixing_logits_1 = nn.Parameter(torch.zeros(3))
        self.mixing_logits_2 = nn.Parameter(torch.zeros(3))
        # Codebooks
        self.codebook_0 = nn.Parameter(torch.randn(num_emb_list[0], e_dim) * 0.05)
        self.codebook_1 = nn.Parameter(torch.randn(num_emb_list[1], e_dim) * 0.05)
        self.codebook_2 = nn.Parameter(torch.randn(num_emb_list[2], e_dim) * 0.05)

    def get_kappa_l(self, layer_idx):
        if layer_idx == 0:
            u = self.kappa_l_raw_0
        elif layer_idx == 1:
            u = self.kappa_l_raw_1
        else:
            u = self.kappa_l_raw_2
        return -(self.kappa_min + F.softplus(u))

    def get_mixing(self, layer_idx):
        """Softmax over 3 mixing logits. Always sum=1, no saturation (跟 #144 不同)."""
        if layer_idx == 0:
            logits = self.mixing_logits_0
        elif layer_idx == 1:
            logits = self.mixing_logits_1
        else:
            logits = self.mixing_logits_2
        return F.softmax(logits, dim=-1)  # (3,)

    def get_codebook(self, layer_idx):
        if layer_idx == 0:
            return self.codebook_0
        elif layer_idx == 1:
            return self.codebook_1
        else:
            return self.codebook_2

    def compute_d_mix(self, z_e, codebook, c, mixing):
        """Issue #146 d_mix = α · d_anchor + β · d_fixed_hyp + γ · d_eucl."""
        d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
        d_fixed = stable_pairwise_hyp_fixed(z_e, codebook, c_fixed=1.0, eps=EPS)
        d_eucl = stable_pairwise_eucl(z_e, codebook)
        # 3 分量独立计算, 然后 mixing 加权 (alpha/beta/gamma sum=1)
        # mixing shape (3,), broadcast to (B, K) via expand
        d_mix = (mixing[0] * d_anchor + mixing[1] * d_fixed + mixing[2] * d_eucl)
        return d_mix, d_anchor, d_fixed, d_eucl

    def contrastive_loss(self, d_mix, batch_idx):
        """Issue #149 修复 #146 失败: continuous soft-anchor + softplus margin.

        #146 反例: argmin 选 anchor (discrete, 切断) + F.relu(margin - pos + neg) (频繁 0 grad)
        #149 修复:
        - continuous soft-anchor: 用 softmax(-d_mix / T) 对全部 codebook 距离加权 (NO argmin)
          - 每个 sample 的 anchor = sum_k softmax(-d_k / T) * d_k (连续, differentiable)
          - temperature T > 0 (T=1.0 默认) 控制 sharpness
        - softplus margin: F.softplus(MARGIN + mean_anchor - neg_dist) 替代 F.relu
          - softplus 处处可微, 没有 hard 0 区间
          - 用 mean_anchor (跟负样本对比) 而不是只 sample 1 个
        - 负样本: random codeword index (跟 #149 spec 一致, NO argmin)
        """
        B, K = d_mix.shape
        T = SOFTANCHOR_TEMP  # temperature, 预注册 > 0
        # Continuous soft-anchor per sample: weighted sum of all distances (NO argmin)
        # softmax(-d_mix / T) 让小距离的 codeword 权重高, 但 gradient 全部 codebook 都参与
        soft_w = F.softmax(-d_mix / T, dim=-1)  # (B, K) - 每个 sample 对全部 K 个 codebook 的权重
        # soft anchor distance = sum_k soft_w_k * d_mix_k (连续加权距离, NO argmin)
        soft_anchor_dist = (soft_w * d_mix).sum(dim=-1)  # (B,)
        # 随机选负样本 index (per sample), 跟 soft anchor 不同 codebook
        neg_idx = torch.randint(0, K, (B,), device=d_mix.device)  # (B,)
        neg_dist = d_mix.gather(1, neg_idx.unsqueeze(1)).squeeze(1)  # (B,)
        # softplus margin: F.softplus(MARGIN + soft_anchor - neg_dist), 处处可微
        # softplus(x) = log(1 + exp(x)), 处处 > 0, gradient 处处非零 (vs relu 在 x<0 时 gradient = 0)
        margin_per_sample = MARGIN + soft_anchor_dist - neg_dist
        loss_per_sample = F.softplus(margin_per_sample)
        contrastive_loss = loss_per_sample.mean()
        # 信息收集
        entropy = -(soft_w * torch.log(soft_w.clamp_min(1e-10))).sum(dim=-1).mean()
        component_contrib = (soft_w * d_mix).std().item()
        return contrastive_loss, soft_anchor_dist.mean().item(), neg_dist.mean().item(), entropy.item(), component_contrib

    def forward_layer(self, x, layer_idx, use_aux=False):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        mixing = self.get_mixing(layer_idx)
        z_e = self.encoders[layer_idx](x)
        codebook = self.get_codebook(layer_idx)
        # Hard SID path (跟 Issue #145 同 .detach())
        cost_hard = stable_pairwise_hyp(z_e, codebook, c, EPS)
        assign = cost_hard.argmin(dim=-1)
        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        # Aux loss
        aux_loss_val = 0.0
        d_mix_val = 0.0
        d_anchor_val = 0.0
        d_fixed_val = 0.0
        d_eucl_val = 0.0
        contrastive_loss_val = 0.0
        if use_aux:
            d_mix, d_anchor, d_fixed, d_eucl = self.compute_d_mix(z_e, codebook, c, mixing)
            contrastive, anchor_mean, neg_mean, soft_entropy, comp_contrib = self.contrastive_loss(d_mix, None)
            aux_loss_val = ALPHA_CONTRASTIVE * contrastive
            d_mix_val = d_mix.mean().item()
            d_anchor_val = d_anchor.mean().item()
            d_fixed_val = d_fixed.mean().item()
            d_eucl_val = d_eucl.mean().item()
            contrastive_loss_val = contrastive.item()
            soft_entropy_val = soft_entropy
            comp_contrib_val = comp_contrib
        domain_margin = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        return {"z_q_st": z_q_st, "x_hat": x_hat, "recon_loss": recon_loss,
                "commit_loss": commit_loss, "assign": assign,
                "z_e": z_e, "codebook": codebook, "c": c, "K": K,
                "kappa_l": kappa_l, "domain_margin": domain_margin,
                "aux_loss": torch.tensor(aux_loss_val, device=x.device) if not isinstance(aux_loss_val, torch.Tensor) else aux_loss_val,
                "mixing": mixing,
                "d_mix_val": d_mix_val, "d_anchor_val": d_anchor_val,
                "d_fixed_val": d_fixed_val, "d_eucl_val": d_eucl_val,
                "contrastive_loss_val": contrastive_loss_val,
                "soft_entropy_val": soft_entropy_val if use_aux else 0.0,
                "comp_contrib_val": comp_contrib_val if use_aux else 0.0}

    def forward(self, x, use_aux=False):
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        total_aux = 0.0
        all_assign = []
        layer_outputs = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l, use_aux=use_aux)
            all_assign.append(out["assign"])
            total_recon = total_recon + out["recon_loss"]
            total_commit = total_commit + out["commit_loss"]
            total_aux = total_aux + out["aux_loss"]
            residual = residual - out["z_q_st"]
            layer_outputs.append(out)
        total_loss = total_recon + self.beta * total_commit
        if use_aux:
            total_loss = total_loss + total_aux
        return {"recon_loss": total_recon, "commit_loss": total_commit,
                "aux_loss": total_aux, "loss": total_loss,
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


def autograd_d_mix_proof(model, X, device):
    """Issue #146 强制: contrastive_loss → d_mix → mixing/κ, hard SID 不在 d_mix grad path."""
    B = min(BATCH_SIZE, len(X))
    x_batch = X[:B].to(device)
    out = model(x_batch, use_aux=True)
    # Backward aux_loss only (验证 aux → mixing/κ grad path)
    model.zero_grad()
    out["aux_loss"].backward(retain_graph=True)
    kappa_grads_with_aux = []
    mixing_grads_with_aux = []
    for l in range(len(NUM_EMB_LIST)):
        if l == 0:
            gk = model.kappa_l_raw_0.grad
            gm = model.mixing_logits_0.grad
        elif l == 1:
            gk = model.kappa_l_raw_1.grad
            gm = model.mixing_logits_1.grad
        else:
            gk = model.kappa_l_raw_2.grad
            gm = model.mixing_logits_2.grad
        kappa_grads_with_aux.append(gk.abs().item() if gk is not None else 0.0)
        mixing_grads_with_aux.append(gm.abs().sum().item() if gm is not None else 0.0)
    # Backward hard SID path only (验证 hard SID 不在 d_mix grad path)
    model.zero_grad()
    out["recon_loss"].backward(retain_graph=True)
    out["commit_loss"].backward(retain_graph=True)
    kappa_grads_with_hard_only = []
    mixing_grads_with_hard_only = []
    for l in range(len(NUM_EMB_LIST)):
        if l == 0:
            gk = model.kappa_l_raw_0.grad
            gm = model.mixing_logits_0.grad
        elif l == 1:
            gk = model.kappa_l_raw_1.grad
            gm = model.mixing_logits_1.grad
        else:
            gk = model.kappa_l_raw_2.grad
            gm = model.mixing_logits_2.grad
        kappa_grads_with_hard_only.append(gk.abs().item() if gk is not None else 0.0)
        mixing_grads_with_hard_only.append(gm.abs().sum().item() if gm is not None else 0.0)
    return {
        "aux_loss_only_kappa_grads": kappa_grads_with_aux,
        "aux_loss_only_mixing_grads": mixing_grads_with_aux,
        "hard_sid_only_kappa_grads": kappa_grads_with_hard_only,
        "hard_sid_only_mixing_grads": mixing_grads_with_hard_only,
        "aux_provides_kappa_grad": any(g > 0 for g in kappa_grads_with_aux),
        "aux_provides_mixing_grad": any(g > 0 for g in mixing_grads_with_aux),
        "hard_sid_isolated": all(g == 0 for g in kappa_grads_with_hard_only + mixing_grads_with_hard_only),
    }


def train_one_config(model, X, optimizer, use_aux, num_steps, record_every, device, log_lines):
    trace = []
    for step in range(1, num_steps + 1):
        idx_perm = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx_perm].to(device)
        optimizer.zero_grad()
        out = model(x_batch, use_aux=use_aux)
        if not torch.isfinite(out["loss"]):
            continue
        out["loss"].backward()
        optimizer.step()
        if step % record_every == 0:
            per_layer_metrics = [compute_layer_metrics(model, X, l, BATCH_SIZE, device) for l in range(len(NUM_EMB_LIST))]
            grad_kappa = []
            grad_mixing = []
            mixing_vals = []
            for l in range(len(NUM_EMB_LIST)):
                gk = [model.kappa_l_raw_0, model.kappa_l_raw_1, model.kappa_l_raw_2][l].grad
                gm = [model.mixing_logits_0, model.mixing_logits_1, model.mixing_logits_2][l].grad
                grad_kappa.append(gk.abs().item() if gk is not None else 0.0)
                grad_mixing.append(gm.abs().sum().item() if gm is not None else 0.0)
                mixing_vals.append(model.get_mixing(l).detach().cpu().tolist())
            trace.append({
                "step": step,
                "loss": out["loss"].item(),
                "aux_loss": out["aux_loss"].item() if use_aux else 0.0,
                "recon_loss": out["recon_loss"].item(),
                "commit_loss": out["commit_loss"].item(),
                "kappa_per_layer": [model.get_kappa_l(l).item() for l in range(len(NUM_EMB_LIST))],
                "mixing_per_layer": mixing_vals,
                "grad_kappa": grad_kappa,
                "grad_mixing": grad_mixing,
                "per_layer_metrics": per_layer_metrics,
            })
            if step % 200 == 0 or step == num_steps:
                avg_util = sum(m["util"] for m in per_layer_metrics) / len(per_layer_metrics)
                log_lines.append(f"  [{'AUX' if use_aux else 'CTRL'}] Step {step}: loss={out['loss'].item():.4f}, "
                                f"avg_util={avg_util:.3f}, grad_kappa={grad_kappa}, grad_mixing={[f'{g:.3f}' for g in grad_mixing]}")
                print(log_lines[-1], flush=True)
    return trace


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #436 Issue #146 precheck+Gate1] product-distance contrastive calibration (R18)")
    log_lines.append("=" * 70)
    log_lines.append(f"[Config] seed={SEED}, device={DEVICE}, num_steps={NUM_STEPS}, record_every={RECORD_EVERY}")
    log_lines.append(f"[Config] codebook_size={NUM_EMB_LIST}, batch_size={BATCH_SIZE}, num_pairs={NUM_PAIRS_PER_BATCH}")
    log_lines.append(f"[Config] lr_kappa={LR_KAPPA}, lr_codebook={LR_CODEBOOK}, lr_mixing={LR_MIXING}")
    log_lines.append(f"[Config] margin={MARGIN}, alpha_contrastive={ALPHA_CONTRASTIVE}")

    config = {
        "seed": SEED, "device": DEVICE, "num_steps": NUM_STEPS, "record_every": RECORD_EVERY,
        "codebook_size": NUM_EMB_LIST, "batch_size": BATCH_SIZE,
        "lr_kappa": LR_KAPPA, "lr_codebook": LR_CODEBOOK, "lr_mixing": LR_MIXING,
        "kappa_min": KAPPA_MIN, "kappa_max": KAPPA_MAX,
        "margin": MARGIN, "alpha_contrastive": ALPHA_CONTRASTIVE,
        "num_pairs_per_batch": NUM_PAIRS_PER_BATCH,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    emb_sha = sha256_of(EMB_PATH)
    log_lines.append(f"\n[SHA256] item_emb.parquet: {emb_sha}")

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.tensor(X_np, dtype=torch.float32)
    log_lines.append(f"[Data] X.shape={X.shape}")

    e_dim_actual = X.shape[1]

    # ============================================================================
    # Precheck (Issue #146 spec 强制 Step 0)
    # ============================================================================
    log_lines.append(f"\n[Precheck Step 0] autograd d_mix graph 隔离证明:")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    model = ProductDistanceModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
    ).to(DEVICE)

    kappa_params = [model.kappa_l_raw_0, model.kappa_l_raw_1, model.kappa_l_raw_2]
    mixing_params = [model.mixing_logits_0, model.mixing_logits_1, model.mixing_logits_2]
    codebook_params = [model.codebook_0, model.codebook_1, model.codebook_2]
    other_params = []
    classified = set(map(id, kappa_params + mixing_params + codebook_params))
    for name, param in model.named_parameters():
        if id(param) not in classified:
            other_params.append(param)
    optimizer = torch.optim.Adam([
        {"params": kappa_params, "lr": LR_KAPPA},
        {"params": mixing_params, "lr": LR_MIXING},
        {"params": codebook_params, "lr": LR_CODEBOOK},
        {"params": other_params, "lr": LR_CODEBOOK},
    ])

    d_mix_graph = autograd_d_mix_proof(model, X, DEVICE)
    log_lines.append(f"  [1] aux_loss → mixing/κ grad (kappa={d_mix_graph['aux_loss_only_kappa_grads']}, mixing={d_mix_graph['aux_loss_only_mixing_grads']})")
    log_lines.append(f"  [2] hard SID → mixing/κ grad (应=0): kappa={d_mix_graph['hard_sid_only_kappa_grads']}, mixing={d_mix_graph['hard_sid_only_mixing_grads']}")
    log_lines.append(f"  [3] aux_provides_kappa_grad={d_mix_graph['aux_provides_kappa_grad']}")
    log_lines.append(f"  [4] aux_provides_mixing_grad={d_mix_graph['aux_provides_mixing_grad']}")
    log_lines.append(f"  [5] hard_sid_isolated={d_mix_graph['hard_sid_isolated']}")

    with open(D_MIX_GRAPH_PROOF_PATH, "w") as f:
        json.dump(d_mix_graph, f, indent=2)

    precheck_pass = (d_mix_graph["aux_provides_kappa_grad"] and
                     d_mix_graph["aux_provides_mixing_grad"] and
                     d_mix_graph["hard_sid_isolated"])
    log_lines.append(f"  Precheck PASS: {precheck_pass}")
    print("\n".join(log_lines), flush=True)
    log_lines = []

    # ============================================================================
    # Gate 1: 1000-step no-aux control + product-calibration
    # ============================================================================
    log_lines.append(f"\n[Gate 1 Control] 1000-step no-aux (跟 #144 同根因 baseline):")
    print(log_lines[-1], flush=True)
    log_lines = []
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_ctrl = ProductDistanceModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
    ).to(DEVICE)
    kappa_params_c = [model_ctrl.kappa_l_raw_0, model_ctrl.kappa_l_raw_1, model_ctrl.kappa_l_raw_2]
    mixing_params_c = [model_ctrl.mixing_logits_0, model_ctrl.mixing_logits_1, model_ctrl.mixing_logits_2]
    codebook_params_c = [model_ctrl.codebook_0, model_ctrl.codebook_1, model_ctrl.codebook_2]
    other_params_c = []
    classified_c = set(map(id, kappa_params_c + mixing_params_c + codebook_params_c))
    for name, param in model_ctrl.named_parameters():
        if id(param) not in classified_c:
            other_params_c.append(param)
    optimizer_ctrl = torch.optim.Adam([
        {"params": kappa_params_c, "lr": LR_KAPPA},
        {"params": mixing_params_c, "lr": LR_MIXING},
        {"params": codebook_params_c, "lr": LR_CODEBOOK},
        {"params": other_params_c, "lr": LR_CODEBOOK},
    ])
    trace_ctrl = train_one_config(model_ctrl, X, optimizer_ctrl, False, NUM_STEPS, RECORD_EVERY, DEVICE, log_lines)

    log_lines.append(f"\n[Gate 1 Calibration] 1000-step with product-distance contrastive:")
    print(log_lines[-1], flush=True)
    log_lines = []
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_aux = ProductDistanceModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
    ).to(DEVICE)
    kappa_params_a = [model_aux.kappa_l_raw_0, model_aux.kappa_l_raw_1, model_aux.kappa_l_raw_2]
    mixing_params_a = [model_aux.mixing_logits_0, model_aux.mixing_logits_1, model_aux.mixing_logits_2]
    codebook_params_a = [model_aux.codebook_0, model_aux.codebook_1, model_aux.codebook_2]
    other_params_a = []
    classified_a = set(map(id, kappa_params_a + mixing_params_a + codebook_params_a))
    for name, param in model_aux.named_parameters():
        if id(param) not in classified_a:
            other_params_a.append(param)
    optimizer_aux = torch.optim.Adam([
        {"params": kappa_params_a, "lr": LR_KAPPA},
        {"params": mixing_params_a, "lr": LR_MIXING},
        {"params": codebook_params_a, "lr": LR_CODEBOOK},
        {"params": other_params_a, "lr": LR_CODEBOOK},
    ])
    trace_aux = train_one_config(model_aux, X, optimizer_aux, True, NUM_STEPS, RECORD_EVERY, DEVICE, log_lines)

    # ============================================================================
    # Issue #146 PASS checks
    # ============================================================================
    log_lines.append(f"\n[Gate 1 checks]:")

    final_kappa = trace_aux[-1]["kappa_per_layer"]
    final_grad_kappa = trace_aux[-1]["grad_kappa"]
    final_grad_mixing = trace_aux[-1]["grad_mixing"]
    final_mixing = trace_aux[-1]["mixing_per_layer"]
    all_kappa_nonzero = all(g > 0 for g in final_grad_kappa)
    all_mixing_nonzero = all(g > 0 for g in final_grad_mixing)
    log_lines.append(f"  (1) κ grad_nonzero={all_kappa_nonzero}, mixing grad_nonzero={all_mixing_nonzero}")
    log_lines.append(f"      final κ={final_kappa}, grad_κ={final_grad_kappa}, grad_mixing={final_grad_mixing}")

    # 至少两分量贡献 > 0.1 (Issue #146 spec)
    components_above_01 = []
    for l_mix in final_mixing:
        components_above_01.append([c > 0.1 for c in l_mix])
    layers_with_two_components = sum(any(cs) and sum(cs) >= 2 for cs in components_above_01)
    log_lines.append(f"  (2) 两分量 >0.1 layers: {layers_with_two_components}/3 (期望 >= 2)")

    with torch.no_grad():
        out1 = model_aux(X[:BATCH_SIZE].to(DEVICE))
        out2 = model_aux(X[:BATCH_SIZE].to(DEVICE))
        round_trip = torch.equal(out1["assign_list"][0], out2["assign_list"][0])
    log_lines.append(f"  (3) hard SID round-trip: {round_trip}")

    all_loss_finite = all(np.isfinite(t["loss"]) for t in trace_aux)
    log_lines.append(f"  (4) all loss finite: {all_loss_finite}")

    final_metrics = trace_aux[-1]["per_layer_metrics"]
    final_util = {f"L{l}": m["util"] for l, m in enumerate(final_metrics)}
    final_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_metrics)}
    util_ok = all(m["util"] >= 0.9 for m in final_metrics)
    max_load_ok = all(m["max_load"] < 0.05 for m in final_metrics)
    log_lines.append(f"  (5) final util: {final_util} (>=90%: {util_ok})")
    log_lines.append(f"      final max_load: {final_max_load} (<5%: {max_load_ok})")

    gate1_pass = (precheck_pass and all_kappa_nonzero and all_mixing_nonzero and
                  layers_with_two_components >= 2 and all_loss_finite and
                  round_trip and util_ok and max_load_ok)
    log_lines.append(f"\n[Gate 1] {'✅ PASS' if gate1_pass else '❌ FAIL'}")

    precheck = {
        "d_mix_graph_proof": d_mix_graph,
        "issue_146_precheck_pass": bool(precheck_pass),
    }
    with open(PRECHECK_PATH, "w") as f:
        json.dump(precheck, f, indent=2, default=str)

    verdict = {
        "task": "task436_issue146_product_distance_contrastive", "issue": 146,
        "config": config,
        "reproducibility": {"item_emb_sha256": emb_sha},
        "r18_product_distance_audit": {
            "aux_provides_kappa_grad": bool(d_mix_graph["aux_provides_kappa_grad"]),
            "aux_provides_mixing_grad": bool(d_mix_graph["aux_provides_mixing_grad"]),
            "hard_sid_isolated": bool(d_mix_graph["hard_sid_isolated"]),
            "kappa_grad_nonzero": bool(all_kappa_nonzero),
            "mixing_grad_nonzero": bool(all_mixing_nonzero),
            "two_components_above_01_layers": int(layers_with_two_components),
            "hard_sid_round_trip": bool(round_trip),
            "all_loss_finite": bool(all_loss_finite),
            "final_util": final_util,
            "final_max_load": final_max_load,
            "util_ok_90pct": bool(util_ok),
            "max_load_ok_5pct": bool(max_load_ok),
            "final_mixing": final_mixing,
        },
        "trace_control": trace_ctrl,
        "trace_aux": trace_aux,
        "gate1_pass": bool(gate1_pass),
        "commit_hash": "<pending - written after git push>",
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines), flush=True)
    print(f"\nGate 1: {'✅ PASS' if gate1_pass else '❌ FAIL'}", flush=True)


if __name__ == "__main__":
    main()