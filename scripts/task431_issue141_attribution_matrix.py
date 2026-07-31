#!/usr/bin/env python3
"""Task #431 / Issue #141 [方向B Gate1] product 分量归因矩阵与 anchor 限幅最小反证实验.

R18 强制: 1000-step 最小复现, 不直接宣布 layer-mixing 有效.
构建 component attribution matrix: 逐层测量 learnable-hyperbolic anchor / fixed-hyperbolic / Euclidean
三分量对每个 hard assignment 的距离排序贡献; 只允许受限低维 gate 改变 correction,
不允许改变 anchor 的 top-1 排序超过预声明比例 (ranking flip ratio).

PASS (Issue #141 spec):
- 三层至少两分量贡献 > 0.1
- anchor-correction 与 ranking flip 均不超过预声明上限
- usage >= 90%, max_load < 5%
- kappa/mixing 梯度有限非零
- hard SID round-trip 可复现
- 附原始日志 + 配置 + 矩阵 + checkpoint SHA256 + verdict + commit (R20+R21 强制 6 件套)

FAIL/PENDING 否则.
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
LR = 1e-4
NUM_STEPS = 1000  # R18 强制最小复现 1000-step
RECORD_EVERY = 100  # 每 100 step 记录 1 次
EPS = 1e-5
EMB_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task431_issue141_attribution_matrix")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task431_issue141_attribution_matrix.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
RANKING_FLIP_MAX = 0.05  # 预声明 ranking flip 上限 (5%)
ANCHOR_RATIO_MIN = 0.4  # anchor ≥ 40%
COMPONENT_CONTRIB_MIN = 0.1  # 至少两分量 > 0.1


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


class AttributionMatrixModel(nn.Module):
    """Issue #141: 逐层 3 分量 d_mix = w1*d_anchor + w2*d_fixed_hyp + w3*d_eucl,
    低维 gate (3 标量 per layer), 不允许改变 anchor top-1 ranking 超过 5%.
    """

    def __init__(self, num_emb_list, e_dim=768, kappa_min=0.1, kappa_max=2.0, beta=0.25,
                 ranking_flip_max=RANKING_FLIP_MAX, anchor_ratio_min=ANCHOR_RATIO_MIN):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.beta = beta
        self.ranking_flip_max = ranking_flip_max
        self.anchor_ratio_min = anchor_ratio_min

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
        # Layer-level mixing (3 scalars per layer, NOT per-codeword)
        self.mixing_logits = nn.ParameterList()
        for K in num_emb_list:
            self.codebooks.append(nn.Parameter(torch.randn(K, e_dim) * 0.05))
            self.kappa_l_raw.append(nn.Parameter(torch.tensor(0.0)))
            self.mixing_logits.append(nn.Parameter(torch.tensor([1.5, 0.3, -0.7])))

    def get_kappa_l(self, layer_idx):
        u = self.kappa_l_raw[layer_idx]
        return -(self.kappa_min + F.softplus(u))

    def get_mixing_weights(self, layer_idx):
        # Enforce anchor >= anchor_ratio_min, w2/w3 ∈ [0, 0.5]
        alpha = F.softmax(self.mixing_logits[layer_idx], dim=-1)
        w1 = alpha[0].clamp(min=self.anchor_ratio_min, max=1.0)
        # Remaining mass split between w2 and w3, bounded
        remaining = (1.0 - w1).clamp_min(1e-8)
        w2 = alpha[1] * remaining
        w3 = alpha[2] * remaining
        # Bound each
        w2 = w2.clamp(max=0.5)
        w3 = w3.clamp(max=0.5)
        # Renormalize
        wsum = (w1 + w2 + w3).clamp_min(1e-8)
        return w1 / wsum, w2 / wsum, w3 / wsum

    def forward_layer(self, x, layer_idx):
        K = self.num_emb_list[layer_idx]
        kappa_l = self.get_kappa_l(layer_idx)
        c = kappa_l.abs().clamp_min(1e-6)
        z_e = self.encoders[layer_idx](x)
        codebook = self.codebooks[layer_idx]
        d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)  # learnable-κ hyperbolic
        c_fixed = torch.tensor(1.0, device=z_e.device)
        d_fixed_hyp = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
        d_eucl = pairwise_euclidean(z_e, codebook)
        w1, w2, w3 = self.get_mixing_weights(layer_idx)
        d_mix = w1 * d_anchor + w2 * d_fixed_hyp + w3 * d_eucl
        # Hard SID
        assign = d_mix.argmin(dim=-1)
        # Anchor-only assignment (for ranking flip ratio)
        assign_anchor = d_anchor.argmin(dim=-1)
        ranking_flip_ratio = (assign != assign_anchor).float().mean().item()

        z_q_hard = codebook[assign]
        z_q_st = z_e + (z_q_hard - z_e).detach()
        x_hat = self.decoders[layer_idx](z_q_st)
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z_e, z_q_hard.detach())
        return {"z_q_st": z_q_st, "x_hat": x_hat, "recon_loss": recon_loss,
                "commit_loss": commit_loss, "assign": assign,
                "assign_anchor": assign_anchor,
                "d_anchor": d_anchor, "d_fixed_hyp": d_fixed_hyp, "d_eucl": d_eucl,
                "w1": w1, "w2": w2, "w3": w3,
                "ranking_flip_ratio": ranking_flip_ratio,
                "kappa_l": kappa_l,
                "domain_margin": (c.sqrt() * z_e.norm(dim=-1)).max().item()}

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


def compute_attribution_matrix(model, X, layer_idx, batch_size, device):
    """R18 attribution matrix: per-sample contribution of (d_anchor, d_fixed_hyp, d_eucl)
    to final d_mix ranking.
    Returns: attribution (B, 3), top1_ranking_flip (B,)
    """
    model.eval()
    K = model.num_emb_list[layer_idx]
    contrib_per_layer = []
    ranking_flip_per_layer = []
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            codebook = model.codebooks[layer_idx]
            c = model.get_kappa_l(layer_idx).abs().clamp_min(1e-6)
            d_anchor = stable_pairwise_hyp(z_e, codebook, c, EPS)
            c_fixed = torch.tensor(1.0, device=device)
            d_fixed_hyp = stable_pairwise_hyp(z_e, codebook, c_fixed, EPS)
            d_eucl = pairwise_euclidean(z_e, codebook)
            w1, w2, w3 = model.get_mixing_weights(layer_idx)
            d_mix = w1 * d_anchor + w2 * d_fixed_hyp + w3 * d_eucl
            assign_mix = d_mix.argmin(dim=-1)
            assign_anchor = d_anchor.argmin(dim=-1)
            assign_fixed = d_fixed_hyp.argmin(dim=-1)
            assign_eucl = d_eucl.argmin(dim=-1)
            # Attribution: how often each component's top-1 matches mix's top-1
            contrib_anchor = (assign_mix == assign_anchor).float().mean().item()
            contrib_fixed = (assign_mix == assign_fixed).float().mean().item()
            contrib_eucl = (assign_mix == assign_eucl).float().mean().item()
            contrib_per_layer.append([contrib_anchor, contrib_fixed, contrib_eucl])
            ranking_flip = (assign_mix != assign_anchor).float().mean().item()
            ranking_flip_per_layer.append(ranking_flip)
    return {
        "contributions_mean": np.mean(contrib_per_layer, axis=0).tolist(),
        "ranking_flip_mean": float(np.mean(ranking_flip_per_layer)),
    }


def compute_layer_metrics(model, X, layer_idx, batch_size, device):
    model.eval()
    K = model.num_emb_list[layer_idx]
    counts = torch.zeros(K, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, min(len(X), 2000), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            z_e = model.encoders[layer_idx](x_batch)
            c = model.get_kappa_l(layer_idx).abs().clamp_min(1e-6)
            d_anchor = stable_pairwise_hyp(z_e, model.codebooks[layer_idx], c, EPS)
            c_fixed = torch.tensor(1.0, device=device)
            d_fixed_hyp = stable_pairwise_hyp(z_e, model.codebooks[layer_idx], c_fixed, EPS)
            d_eucl = pairwise_euclidean(z_e, model.codebooks[layer_idx])
            w1, w2, w3 = model.get_mixing_weights(layer_idx)
            d_mix = w1 * d_anchor + w2 * d_fixed_hyp + w3 * d_eucl
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
    log_lines.append(f"[Task #431 Issue #141 Gate 1] product 分量归因矩阵 + anchor 限幅 1000-step 最小复现 (R18)")
    log_lines.append("=" * 70)
    log_lines.append(f"[Config] seed={SEED}, device={DEVICE}, num_steps={NUM_STEPS}, record_every={RECORD_EVERY}")
    log_lines.append(f"[Config] codebook_size={NUM_EMB_LIST}, batch_size={BATCH_SIZE}, lr={LR}")
    log_lines.append(f"[Config] ranking_flip_max={RANKING_FLIP_MAX}, anchor_ratio_min={ANCHOR_RATIO_MIN}, component_contrib_min={COMPONENT_CONTRIB_MIN}")

    config = {
        "seed": SEED, "device": DEVICE, "num_steps": NUM_STEPS, "record_every": RECORD_EVERY,
        "codebook_size": NUM_EMB_LIST, "batch_size": BATCH_SIZE, "lr": LR,
        "ranking_flip_max": RANKING_FLIP_MAX, "anchor_ratio_min": ANCHOR_RATIO_MIN,
        "component_contrib_min": COMPONENT_CONTRIB_MIN,
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
    for line in log_lines[-3:]:
        print(line, flush=True)

    e_dim_actual = X.shape[1]
    model = AttributionMatrixModel(
        num_emb_list=NUM_EMB_LIST, e_dim=e_dim_actual,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, beta=0.25,
        ranking_flip_max=RANKING_FLIP_MAX, anchor_ratio_min=ANCHOR_RATIO_MIN,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_lines.append(f"\n[Step 0] Initial attribution matrix:")
    init_attr = [compute_attribution_matrix(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
    for l, attr in enumerate(init_attr):
        log_lines.append(f"  L{l} contributions (anchor, fixed_hyp, eucl): {attr['contributions_mean']}")
        log_lines.append(f"      ranking_flip: {attr['ranking_flip_mean']:.4f}")
    for line in log_lines[-6:]:
        print(line, flush=True)

    # R18 minimal repro 1000-step
    log_lines.append(f"\n[Step 1] {NUM_STEPS}-step R18 anchor-bounded training:")
    print(log_lines[-1], flush=True)
    trace = []
    for step in range(1, NUM_STEPS + 1):
        idx_perm = torch.randperm(len(X))[:BATCH_SIZE]
        x_batch = X[idx_perm].to(DEVICE)
        optimizer.zero_grad()
        out = model(x_batch)
        if not torch.isfinite(out["loss"]):
            continue
        out["loss"].backward()
        # R18 spec: kappa/mixing 梯度有限非零
        grad_ok = True
        grad_max_per_layer = []
        for l in range(len(NUM_EMB_LIST)):
            g_kappa = model.kappa_l_raw[l].grad
            g_mix = model.mixing_logits[l].grad
            max_g_k = g_kappa.abs().max().item() if g_kappa is not None else 0.0
            max_g_m = g_mix.abs().max().item() if g_mix is not None else 0.0
            grad_max_per_layer.append({"kappa": max_g_k, "mixing": max_g_m})
            if g_kappa is None or g_mix is None or max_g_k == 0 or max_g_m == 0:
                grad_ok = False
        optimizer.step()

        if step % RECORD_EVERY == 0:
            attr = [compute_attribution_matrix(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
            per_layer_metrics = [compute_layer_metrics(model, X, l, BATCH_SIZE, DEVICE) for l in range(len(NUM_EMB_LIST))]
            trace.append({
                "step": step,
                "loss": out["loss"].item(),
                "attributions": attr,
                "per_layer_metrics": per_layer_metrics,
                "grad_max_per_layer": grad_max_per_layer,
                "grad_ok": bool(grad_ok),
                "kappa_per_layer": [model.get_kappa_l(l).item() for l in range(len(NUM_EMB_LIST))],
                "mixing_per_layer": [list(model.get_mixing_weights(l)) for l in range(len(NUM_EMB_LIST))],
            })
            if step % 200 == 0 or step == NUM_STEPS:
                avg_util = sum(m["util"] for m in per_layer_metrics) / len(per_layer_metrics)
                avg_flip = sum(a["ranking_flip_mean"] for a in attr) / len(attr)
                log_lines.append(f"  Step {step}: loss={out['loss'].item():.4f}, avg_util={avg_util:.3f}, "
                                f"avg_ranking_flip={avg_flip:.4f}")
                print(log_lines[-1], flush=True)

    # R18 PASS checks (Issue #141 spec)
    log_lines.append(f"\n[Step 2] R18 PASS checks:")
    final_attr = trace[-1]["attributions"]
    final_metrics = trace[-1]["per_layer_metrics"]

    # 1. 三层至少两分量贡献 > 0.1
    n_components_per_layer = []
    for l, attr in enumerate(final_attr):
        n_above_thresh = sum(1 for c in attr["contributions_mean"] if c > COMPONENT_CONTRIB_MIN)
        n_components_per_layer.append(n_above_thresh)
    components_ok = all(n >= 2 for n in n_components_per_layer)
    log_lines.append(f"  (1) components > 0.1 per layer: {n_components_per_layer} (>=2: {components_ok})")
    print(log_lines[-1], flush=True)

    # 2. anchor-correction ratio ≤ ranking_flip_max 5%
    anchor_ratios_per_layer = [mix[0] for mix in trace[-1]["mixing_per_layer"]]
    ranking_flips = [a["ranking_flip_mean"] for a in final_attr]
    anchor_ratio_ok = all(r >= ANCHOR_RATIO_MIN for r in anchor_ratios_per_layer)
    ranking_flip_ok = all(f <= RANKING_FLIP_MAX for f in ranking_flips)
    log_lines.append(f"  (2) anchor_ratio: {anchor_ratios_per_layer} (>=0.4: {anchor_ratio_ok})")
    log_lines.append(f"      ranking_flip: {ranking_flips} (<=0.05: {ranking_flip_ok})")
    print(log_lines[-1], flush=True)
    print(log_lines[-1], flush=True)

    # 3. usage >= 90%, max_load < 5%
    final_util = {f"L{l}": m["util"] for l, m in enumerate(final_metrics)}
    final_max_load = {f"L{l}": m["max_load"] for l, m in enumerate(final_metrics)}
    util_ok = all(m["util"] >= 0.9 for m in final_metrics)
    max_load_ok = all(m["max_load"] < 0.05 for m in final_metrics)
    log_lines.append(f"  (3) final util: {final_util} (>=90%: {util_ok})")
    log_lines.append(f"      final max_load: {final_max_load} (<5%: {max_load_ok})")
    print(log_lines[-1], flush=True)
    print(log_lines[-1], flush=True)

    # 4. kappa/mixing 梯度有限非零 (final)
    final_grad_ok = trace[-1]["grad_ok"]
    log_lines.append(f"  (4) kappa/mixing grad finite nonzero: {final_grad_ok} (final grads: {trace[-1]['grad_max_per_layer']})")
    print(log_lines[-1], flush=True)

    # 5. hard SID round-trip
    with torch.no_grad():
        out1 = model(X[:BATCH_SIZE].to(DEVICE))
        out2 = model(X[:BATCH_SIZE].to(DEVICE))
        round_trip = torch.equal(out1["assign_list"][0], out2["assign_list"][0])
    log_lines.append(f"  (5) hard SID round-trip: {round_trip}")
    print(log_lines[-1], flush=True)

    gate1_pass = components_ok and anchor_ratio_ok and ranking_flip_ok and util_ok and max_load_ok and final_grad_ok and round_trip
    log_lines.append(f"\n[Gate 1] {'✅ PASS' if gate1_pass else '❌ FAIL'}")

    verdict = {
        "task": "task431_issue141_attribution_matrix", "issue": 141,
        "issue_spec_6_audit_pieces": {
            "1_config": str(CONFIG_PATH),
            "2_sha256": {"item_emb": emb_sha},
            "3_trace": trace,
            "4_raw_log": str(LOG_PATH),
            "5_verdict": str(VERDICT_PATH),
            "6_commit": "<pending - written after git push>",
        },
        "config": config,
        "reproducibility": {"item_emb_sha256": emb_sha},
        "r18_attribution_matrix_audit": {
            "init_attribution": init_attr,
            "final_attribution": final_attr,
            "n_components_above_thresh_per_layer": n_components_per_layer,
            "components_ok_2_of_3": bool(components_ok),
            "anchor_ratios": anchor_ratios_per_layer,
            "ranking_flips": ranking_flips,
            "anchor_ratio_ok": bool(anchor_ratio_ok),
            "ranking_flip_ok": bool(ranking_flip_ok),
            "final_util": final_util,
            "final_max_load": final_max_load,
            "util_ok_90pct": bool(util_ok),
            "max_load_ok_5pct": bool(max_load_ok),
            "grad_ok": bool(final_grad_ok),
            "hard_sid_round_trip": bool(round_trip),
        },
        "trace": trace,
        "gate1_pass": bool(gate1_pass),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Verdict saved] {VERDICT_PATH}")
    print(log_lines[-1], flush=True)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log_lines.append(f"[Log saved] {LOG_PATH}")

    log_lines.append(f"\nGate 1: {'✅ PASS' if gate1_pass else '❌ FAIL'}")
    print(log_lines[-1], flush=True)


if __name__ == "__main__":
    main()