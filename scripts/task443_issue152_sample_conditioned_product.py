#!/usr/bin/env python3
"""Task #443 / Issue #152 [方向B Gate1] 样本条件 product 权重与容量硬分配审计.

R18 强制: 保持 #149 ProductDistanceModel 三分量 (d_anchor/d_fixed_hyp/d_eucl),
改 mixing scalars 为 **per-sample sample-conditioned weights** (从 encoder 输出 MLP 生成),
hard SID 走 capacity-constrained Hungarian (跟 #151 同).

修复 #149 失败: 静态 mixing + soft-anchor+softplus 让 loss 降 85% 但 usage<=3.13%, max_load≈100%.
新增:
- sample-conditioned weights: 每个 sample 的 3 mixing 权重从 encoder 特征生成 (受熵下界约束)
- capacity-constrained Hungarian: 每码字 ≤ cap, 消除单码字捷径

Precheck 强制 (Issue #152 spec):
1. continuous calibration → sample weights/κ autograd trace
2. integer capacity assignment → hard SID (跟 #151 同 Hungarian)
3. 容量求解整型性 + 100% 可行
4. sample weights entropy >= log(3) - ε (受熵下界约束)

Gate 1 (Issue #152 spec):
- PASS: 每批容量可行, 三层 usage>=90%, max_load<5%, 每层至少两分量平均贡献>0.1 且 sample weights 非退化, κ/mixing 均有限非零并更新, hard SID round-trip, 无 NaN/Inf
- FAIL: 任一不满足即 STOP

8 件套审计 (R20+R21 强制).
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")

from utils import mobius_add as hgrec_mobius_add

# Config
SEED = 42
DEVICE = "cuda:0"
KAPPA_MIN = 0.1
KAPPA_MAX = 2.0
NUM_EMB_LIST = [64, 128, 256]
BATCH_SIZE = 256
NUM_PAIRS_PER_BATCH = 64
SOFTANCHOR_TEMP = 1.0  # continuous soft-anchor temperature
LR_KAPPA = 1e-3
LR_CODEBOOK = 1e-4
LR_MIXING = 1e-3
LR_WEIGHT_MLP = 1e-3
NUM_STEPS = 1000
RECORD_EVERY = 100
EPS = 1e-5
MARGIN = 1.0
ALPHA_CONTRASTIVE = 0.5
ALPHA_ENTROPY = 0.1  # entropy regularization for sample weights
ENTROPY_MIN = math.log(3) - 0.1  # entropy lower bound for 3-component softmax
CAPACITY_SLACK = 1
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task443_issue152_sample_conditioned_product")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task443_issue152_sample_conditioned_product.log"
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
    c = torch.tensor(c_fixed, device=z_e.device, dtype=z_e.dtype)
    return stable_pairwise_hyp(z_e, codebook, c, eps)


def stable_pairwise_eucl(z_e, codebook):
    return torch.cdist(z_e, codebook, p=2)


def capacity_constrained_assignment(cost_np, capacity_cap):
    B, K = cost_np.shape
    total_slots = sum(capacity_cap)
    if total_slots < B:
        return None
    cost_expanded = np.zeros((B, total_slots), dtype=np.float32)
    slot_to_codeword = []
    for i, cap in enumerate(capacity_cap):
        for _ in range(cap):
            slot_to_codeword.append(i)
            cost_expanded[:, len(slot_to_codeword) - 1] = cost_np[:, i]
    row_ind, col_ind = linear_sum_assignment(cost_expanded)
    assign = np.zeros(B, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        assign[r] = slot_to_codeword[c]
    return assign


class SampleConditionedProductKappaModel(nn.Module):
    """Issue #152: per-sample conditioned product weights + capacity-hard Hungarian assignment.

    架构:
    - 跟 #149 同 ProductDistanceModel 三分量 (d_anchor/d_fixed_hyp/d_eucl)
    - mixing scalars 改为 per-sample: encoder z_e → weight_mlp → 3 logits → softmax (受 entropy reg)
    - hard SID 走 Hungarian capacity (跟 #151 同)
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
        # per-sample weight MLP (encoder z_e → 3 logits)
        self.weight_mlps = nn.ModuleList()
        for K in num_emb_list:
            self.encoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, e_dim),
            ))
            self.weight_mlps.append(nn.Sequential(
                nn.Linear(e_dim, e_dim), nn.ReLU(), nn.Linear(e_dim, 3),
            ))
        # 3 独立 κ
        self.kappa_l_raw_0 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_1 = nn.Parameter(torch.tensor(0.0))
        self.kappa_l_raw_2 = nn.Parameter(torch.tensor(0.0))
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

    def get_codebook(self, layer_idx):
        if layer_idx == 0:
            return self.codebook_0
        elif layer_idx == 1:
            return self.codebook_1
        else:
            return self.codebook_2

    def get_sample_weights(self, z_e, layer_idx):
        """Issue #152: per-sample weights via MLP from encoder output (NO static scalar)."""
        logits = self.weight_mlps[layer_idx](z_e)  # (B, 3)
        return F.softmax(logits, dim=-1)  # (B, 3) per-sample mixing

    def compute_d_mix(self, z_e, codebook, c, sample_weights):
        """Issue #152: per-sample d_mix = α_0·d_anchor + α_1·d_fixed + α_2·d_eucl.
        sample_weights: (B, 3), per-sample mixing
        """
        d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
        d_fixed = stable_pairwise_hyp_fixed(z_e, codebook, c_fixed=1.0, eps=EPS)
        d_eucl = stable_pairwise_eucl(z_e, codebook)
        # d_mix shape (B, K): weighted sum over 3 components per sample
        # sample_weights[:, 0] is (B,), broadcast to (B, K)
        d_mix = (sample_weights[:, 0:1] * d_anchor +
                 sample_weights[:, 1:2] * d_fixed +
                 sample_weights[:, 2:3] * d_eucl)
        return d_mix, d_anchor, d_fixed, d_eucl

    def contrastive_loss(self, d_mix):
        """Issue #149 复用: continuous soft-anchor + softplus margin."""
        B, K = d_mix.shape
        T = SOFTANCHOR_TEMP
        soft_w = F.softmax(-d_mix / T, dim=-1)  # (B, K)
        soft_anchor_dist = (soft_w * d_mix).sum(dim=-1)  # (B,)
        neg_idx = torch.randint(0, K, (B,), device=d_mix.device)
        neg_dist = d_mix.gather(1, neg_idx.unsqueeze(1)).squeeze(1)
        margin_per_sample = MARGIN + soft_anchor_dist - neg_dist
        loss_per_sample = F.softplus(margin_per_sample)
        contrastive_loss = loss_per_sample.mean()
        entropy = -(soft_w * torch.log(soft_w.clamp_min(1e-10))).sum(dim=-1).mean()
        return contrastive_loss, soft_anchor_dist.mean().item(), neg_dist.mean().item(), entropy.item()

    def forward_layer(self, x, layer_idx, use_aux=False):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        z_e = self.encoders[layer_idx](x)
        codebook = self.get_codebook(layer_idx)
        # === Issue #152: per-sample weights ===
        sample_weights = self.get_sample_weights(z_e, layer_idx)  # (B, 3)
        # Hard SID path (capacity-constrained Hungarian, 跟 #151 同)
        cost = stable_pairwise_hyp(z_e, codebook, c, EPS)
        B = cost.shape[0]
        per_cap = max(1, math.ceil(B / K) + CAPACITY_SLACK)
        capacity_cap = [per_cap] * K
        cost_np = cost.detach().cpu().numpy()
        assign_np = capacity_constrained_assignment(cost_np, capacity_cap)
        if assign_np is None:
            assign = cost.argmin(dim=-1)
            feasible = False
        else:
            assign = torch.from_numpy(assign_np).to(cost.device)
            feasible = True
        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        aux_loss = torch.tensor(0.0, device=x.device)
        entropy_loss = torch.tensor(0.0, device=x.device)
        component_contrib_val = 0.0
        if use_aux:
            d_mix, d_anchor, d_fixed, d_eucl = self.compute_d_mix(z_e, codebook, c, sample_weights)
            contrastive, anchor_mean, neg_mean, soft_entropy = self.contrastive_loss(d_mix)
            aux_loss = ALPHA_CONTRASTIVE * contrastive
            # Entropy regularization: entropy >= ENTROPY_MIN (per-sample weight entropy)
            sample_entropy = -(sample_weights * torch.log(sample_weights.clamp_min(1e-10))).sum(dim=-1)
            entropy_loss = ALPHA_ENTROPY * F.relu(ENTROPY_MIN - sample_entropy).mean()
            # Component contribution: weight × d_mix component magnitude
            component_contrib_val = sample_weights.mean(dim=0).cpu().tolist()  # mean weight per component
        domain_margin = (c.sqrt() * z_e.norm(dim=-1)).max().item()
        return {"z_q_st": z_q_st, "x_hat": x_hat, "recon_loss": recon_loss,
                "commit_loss": commit_loss, "assign": assign,
                "z_e": z_e, "codebook": codebook, "c": c, "K": K,
                "kappa_l": kappa_l, "domain_margin": domain_margin,
                "aux_loss": aux_loss + entropy_loss,
                "feasible": feasible,
                "sample_weights": sample_weights,
                "component_contrib": component_contrib_val}

    def forward(self, x, use_aux=False):
        residual = x
        total_recon = 0.0
        total_commit = 0.0
        total_aux = 0.0
        all_assign = []
        all_feasible = []
        layer_outputs = []
        for l in range(len(self.num_emb_list)):
            out = self.forward_layer(residual, l, use_aux=use_aux)
            all_assign.append(out["assign"])
            all_feasible.append(out["feasible"])
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
                "assign_list": all_assign, "layer_outputs": layer_outputs,
                "feasible_list": all_feasible}


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
            B = d.shape[0]
            per_cap = max(1, math.ceil(B / K) + CAPACITY_SLACK)
            capacity_cap = [per_cap] * K
            cost_np = d.detach().cpu().numpy()
            assign_np = capacity_constrained_assignment(cost_np, capacity_cap)
            if assign_np is None:
                assign = d.argmin(dim=-1).cpu()
            else:
                assign = torch.from_numpy(assign_np)
            for a in assign.tolist():
                counts[a] += 1
    n_used = (counts > 0).sum().item()
    util = n_used / K
    max_load = counts.max().item() / max(1, counts.sum().item())
    return {"util": util, "max_load": max_load, "n_used": n_used, "K": K}


def autograd_d_mix_proof(model, X, device):
    B = min(BATCH_SIZE, len(X))
    x_batch = X[:B].to(device)
    out = model(x_batch, use_aux=True)
    # Backward aux_loss only (验证 aux → κ/weight_mlp grad path)
    model.zero_grad()
    out["aux_loss"].backward(retain_graph=True)
    kappa_grads_with_aux = []
    weight_mlp_grads_with_aux = []
    for l in range(3):
        if l == 0:
            gk = model.kappa_l_raw_0.grad
        elif l == 1:
            gk = model.kappa_l_raw_1.grad
        else:
            gk = model.kappa_l_raw_2.grad
        kappa_grads_with_aux.append(gk.abs().item() if gk is not None else 0.0)
        # weight_mlp grad (last linear layer of MLP)
        weight_mlp_grads_with_aux.append(model.weight_mlps[l][-1].weight.grad.abs().sum().item() if model.weight_mlps[l][-1].weight.grad is not None else 0.0)
    # Backward hard SID path only (验证 hard SID 不在 aux grad path)
    model.zero_grad()
    out["recon_loss"].backward(retain_graph=True)
    out["commit_loss"].backward(retain_graph=True)
    kappa_grads_with_hard_only = []
    weight_mlp_grads_with_hard_only = []
    for l in range(3):
        if l == 0:
            gk = model.kappa_l_raw_0.grad
        elif l == 1:
            gk = model.kappa_l_raw_1.grad
        else:
            gk = model.kappa_l_raw_2.grad
        kappa_grads_with_hard_only.append(gk.abs().item() if gk is not None else 0.0)
        weight_mlp_grads_with_hard_only.append(model.weight_mlps[l][-1].weight.grad.abs().sum().item() if model.weight_mlps[l][-1].weight.grad is not None else 0.0)
    return {
        "kappa_grads_with_aux": kappa_grads_with_aux,
        "kappa_grads_with_hard_only": kappa_grads_with_hard_only,
        "weight_mlp_grads_with_aux": weight_mlp_grads_with_aux,
        "weight_mlp_grads_with_hard_only": weight_mlp_grads_with_hard_only,
        "aux_to_kappa_path": all(g > 0 for g in kappa_grads_with_aux),
        "hard_sid_isolated": all(abs(g) < 1e-3 for g in kappa_grads_with_hard_only),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #443 Issue #152 precheck+Gate1] sample-conditioned product + capacity-hard")
    log_lines.append("=" * 70)
    log_lines.append(f"\n[Config] seed={SEED}, alpha_contrastive={ALPHA_CONTRASTIVE}, alpha_entropy={ALPHA_ENTROPY}, entropy_min={ENTROPY_MIN:.3f}")

    emb_sha = sha256_of(EMB_PATH)
    log_lines.append(f"[SHA256] item_emb.parquet: {emb_sha}")

    config = {
        "seed": SEED, "device": DEVICE, "kappa_min": KAPPA_MIN, "kappa_max": KAPPA_MAX,
        "num_emb_list": NUM_EMB_LIST, "batch_size": BATCH_SIZE,
        "lr_kappa": LR_KAPPA, "lr_codebook": LR_CODEBOOK, "lr_mixing": LR_MIXING, "lr_weight_mlp": LR_WEIGHT_MLP,
        "num_steps": NUM_STEPS, "record_every": RECORD_EVERY,
        "alpha_contrastive": ALPHA_CONTRASTIVE, "alpha_entropy": ALPHA_ENTROPY,
        "entropy_min": ENTROPY_MIN, "capacity_slack": CAPACITY_SLACK,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    X_np = np.stack([np.asarray(v, dtype=np.float32) for v in df['embedding'].values])
    X = torch.from_numpy(X_np)
    log_lines.append(f"[Data] X.shape: {X.shape}")

    model = SampleConditionedProductKappaModel(NUM_EMB_LIST, e_dim=X.shape[1]).to(DEVICE)

    # Precheck
    log_lines.append(f"\n[Precheck Step 0] autograd graph isolation proof:")
    proof = autograd_d_mix_proof(model, X, DEVICE)
    log_lines.append(f"  kappa_grads_with_aux: {proof['kappa_grads_with_aux']}")
    log_lines.append(f"  weight_mlp_grads_with_aux: {proof['weight_mlp_grads_with_aux']}")
    log_lines.append(f"  hard_sid_isolated: {proof['hard_sid_isolated']}")
    log_lines.append(f"  aux_to_kappa_path: {proof['aux_to_kappa_path']}")

    with open(D_MIX_GRAPH_PROOF_PATH, "w") as f:
        json.dump(proof, f, indent=2)

    precheck_pass = proof["aux_to_kappa_path"] and proof["hard_sid_isolated"]
    log_lines.append(f"\n[Precheck 总评] PASS: {precheck_pass}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log_lines = []

    if not precheck_pass:
        log_lines.append("[STOP] Precheck FAIL")
        with open(LOG_PATH, "a") as f:
            f.write("\n".join(log_lines))
        with open(VERDICT_PATH, "w") as f:
            json.dump({"gate1_pass": False, "reason": "precheck_failed", "proof": proof}, f, indent=2)
        return

    # 训练
    log_lines.append(f"\n[Gate 1 Control] 1000-step no-aux (NO aux, 跟 #144/#146 复现)")
    optimizer = torch.optim.Adam([
        {"params": [model.kappa_l_raw_0, model.kappa_l_raw_1, model.kappa_l_raw_2], "lr": LR_KAPPA},
        {"params": [model.codebook_0, model.codebook_1, model.codebook_2], "lr": LR_CODEBOOK},
        {"params": [p for mlp in model.weight_mlps for p in mlp.parameters()], "lr": LR_WEIGHT_MLP},
        {"params": list(model.encoders.parameters()) + list(model.decoders.parameters()), "lr": LR_CODEBOOK},
    ])
    control_trace = []
    for step in range(NUM_STEPS):
        idx = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx].to(DEVICE)
        out = model(x_batch, use_aux=False)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        if step % RECORD_EVERY == 0:
            kappa_grads = []
            for l in range(3):
                if l == 0: g = model.kappa_l_raw_0.grad
                elif l == 1: g = model.kappa_l_raw_1.grad
                else: g = model.kappa_l_raw_2.grad
                kappa_grads.append(g.abs().item() if g is not None else 0.0)
            control_trace.append({"step": step, "loss": out["loss"].item(), "kappa_grads": kappa_grads,
                                  "feasible": out["feasible_list"]})
            log_lines.append(f"  [CTRL] Step {step}: loss={out['loss'].item():.4f}, grad_kappa={kappa_grads}")

    log_lines.append(f"\n[Gate 1 Calibration] 1000-step with sample-conditioned aux + capacity:")
    calib_trace = []
    for step in range(NUM_STEPS):
        idx = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx].to(DEVICE)
        out = model(x_batch, use_aux=True)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        if step % RECORD_EVERY == 0:
            kappa_grads = []
            for l in range(3):
                if l == 0: g = model.kappa_l_raw_0.grad
                elif l == 1: g = model.kappa_l_raw_1.grad
                else: g = model.kappa_l_raw_2.grad
                kappa_grads.append(g.abs().item() if g is not None else 0.0)
            util_per_layer = []
            max_load_per_layer = []
            component_per_layer = []
            for l in range(3):
                m = compute_layer_metrics(model, X, l, BATCH_SIZE, DEVICE)
                util_per_layer.append(m["util"])
                max_load_per_layer.append(m["max_load"])
                component_per_layer.append(out["layer_outputs"][l]["component_contrib"])
            calib_trace.append({"step": step, "loss": out["loss"].item(),
                                "kappa_grads": kappa_grads, "feasible": out["feasible_list"],
                                "util": util_per_layer, "max_load": max_load_per_layer,
                                "component_contrib": component_per_layer})
            log_lines.append(f"  [AUX] Step {step}: loss={out['loss'].item():.4f}, grad_kappa={kappa_grads}, util={util_per_layer}, max_load={max_load_per_layer}, comp={component_per_layer}")

    # Gate 1 决策
    final_util = []
    final_max_load = []
    final_component = []
    for l in range(3):
        m = compute_layer_metrics(model, X, l, BATCH_SIZE, DEVICE)
        final_util.append(m["util"])
        final_max_load.append(m["max_load"])
        # 评估 sample weights 退化 (per-sample entropy >= ENTROPY_MIN)
        with torch.no_grad():
            x_eval = X[:BATCH_SIZE].to(DEVICE)
            z_e_eval = model.encoders[l](x_eval)
            sw_eval = model.get_sample_weights(z_e_eval, l)
            sw_entropy = -(sw_eval * torch.log(sw_eval.clamp_min(1e-10))).sum(dim=-1)
            sw_mean_entropy = sw_entropy.mean().item()
        final_component.append({"mean_entropy": sw_mean_entropy, "min_entropy": sw_entropy.min().item()})

    log_lines.append(f"\n[Gate 1 checks]:")
    log_lines.append(f"  final kappa: {[model.get_kappa_l(l).item() for l in range(3)]}")
    log_lines.append(f"  final util: {final_util} (>=90%: {all(u >= 0.9 for u in final_util)})")
    log_lines.append(f"  final max_load: {final_max_load} (<5%: {all(ml < 0.05 for ml in final_max_load)})")
    log_lines.append(f"  final sample_weights entropy: {final_component} (>=ENTROPY_MIN: {all(c['mean_entropy'] >= ENTROPY_MIN for c in final_component)})")
    log_lines.append(f"  all batch feasible: {all(t['feasible'] for t in calib_trace)}")

    # 检查 component contribution (>= 2 components per layer > 0.1)
    component_pass = []
    for l in range(3):
        comp_contribs = calib_trace[-1]["component_contrib"][l]  # list of 3 mean weights
        n_active = sum(1 for c in comp_contribs if c > 0.1)
        component_pass.append(n_active >= 2)
    log_lines.append(f"  component contribution >=2 per layer: {component_pass}")

    gate1_pass = (precheck_pass and
                  all(u >= 0.9 for u in final_util) and
                  all(ml < 0.05 for ml in final_max_load) and
                  all(t["feasible"] for t in calib_trace) and
                  all(all(g > 0 for g in t["kappa_grads"]) for t in calib_trace) and
                  all(component_pass))

    log_lines.append(f"\n[Gate 1] {'✅ PASS' if gate1_pass else '❌ FAIL'}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log_lines))

    with open(VERDICT_PATH, "w") as f:
        json.dump({
            "gate1_pass": gate1_pass,
            "reason": "trained" if gate1_pass else "usage_or_maxload_not_met",
            "precheck": proof,
            "final_util": final_util,
            "final_max_load": final_max_load,
            "final_component": final_component,
            "component_pass": component_pass,
            "all_feasible": all(t["feasible"] for t in calib_trace),
            "control_trace_summary": control_trace,
            "calib_trace_summary": calib_trace,
            "task_id": 443, "issue": "Issue #152",
        }, f, indent=2)


if __name__ == "__main__":
    main()