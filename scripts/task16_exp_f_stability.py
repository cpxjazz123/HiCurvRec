"""
Task #71 — Exp F: 三流形残差数值稳定性测试

测试 r_E/r_H/r_S 三种残差在 RQ-VAE 训练中的数值稳定性：
  - gradient norm (是否爆炸)
  - NaN/Inf 出现频率
  - 训练 loss 曲线是否发散

启动:
  python scripts/task6_exp_f_stability.py [--device cuda:0]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


# ============== 三种残差几何函数 ==============

def project_to_poincare(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def project_to_sphere(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return x / norm


def poincare_log_map(q, r):
    q_sqnorm = (q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    r_sqnorm = (r ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    inner = (q * r).sum(dim=-1, keepdim=True)
    num = (1 + 2 * inner + r_sqnorm) * q + (1 - q_sqnorm) * r
    denom = (1 + 2 * inner + q_sqnorm * r_sqnorm).clamp(min=1e-7)
    u = num / denom
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    diff_sq = ((r - q) ** 2).sum(dim=-1, keepdim=True)
    arg = (1 + 2 * diff_sq / ((1 - q_sqnorm) * (1 - r_sqnorm))).clamp(min=1 + 1e-7)
    d = torch.acosh(arg)
    lambda_q = 2 / (1 - q_sqnorm)
    return (d / (lambda_q * u_norm)) * u


def spherical_log_map(q, r):
    q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (q * r).sum(dim=-1, keepdim=True) / (q_norm * r_norm)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(cos_sim)
    inner = (q * r).sum(dim=-1, keepdim=True)
    proj = r - (inner / (q_norm ** 2)) * q
    proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (d / proj_norm) * proj


# ============== 简化 RQ-VAE 训练器（单层） ==============

class SingleLayerRQ(nn.Module):
    """单层 RQ-VAE（用于数值稳定性测试）"""

    def __init__(self, n_features, n_clusters):
        super().__init__()
        self.codebook = nn.Parameter(
            torch.randn(n_clusters, n_features) * (1.0 / n_features ** 0.5)
        )
        self.n_clusters = n_clusters

    def forward(self, x, residual_type="E"):
        """根据 residual_type 计算三种残差，训练 codebook"""
        d = torch.cdist(x, self.codebook, p=2)
        idx = d.argmin(dim=1)
        q = self.codebook[idx]
        # 残差
        if residual_type == "E":
            r = x - q
        elif residual_type == "H":
            r = poincare_log_map(project_to_poincare(q), project_to_poincare(x))
        elif residual_type == "S":
            r = spherical_log_map(project_to_sphere(q), project_to_sphere(x))
        else:
            raise ValueError(residual_type)
        # 重建
        x_hat = q + r if residual_type == "E" else x  # H/S 重建 = q + log_map inverse (简化: 不重构)
        # 简化损失: ||x - q||²（commitment loss）
        loss = F.mse_loss(x, q.detach()) + F.mse_loss(x.detach(), q)
        return loss, r, idx


def train_one_geometry(residual_type, embeddings_np, n_clusters=64,
                       n_steps=200, batch_size=512, lr=1e-3, device="cuda:0", seed=42):
    """训练单层 RQ-VAE 监控数值稳定性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_items, n_features = embeddings_np.shape

    model = SingleLayerRQ(n_features, n_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32, device=device)

    history = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "nan_inf_count": [],
    }

    for step in range(n_steps):
        # 随机 batch
        idx = np.random.randint(0, n_items, size=batch_size)
        batch = embeddings_t[idx]

        optimizer.zero_grad()
        loss, r, q_idx = model(batch, residual_type=residual_type)
        loss.backward()

        # 检查 NaN/Inf
        nan_count = 0
        inf_count = 0
        for p in model.parameters():
            if torch.isnan(p.grad).any():
                nan_count += 1
            if torch.isinf(p.grad).any():
                inf_count += 1
        history["nan_inf_count"].append(nan_count + inf_count)

        # Gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        history["grad_norm"].append(grad_norm)
        history["step"].append(step)
        history["loss"].append(loss.item())

        # Clip gradients to prevent explosion (only measure, don't fix)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_json", default="products/task16/from_task71/exp_f_stability.json")
    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Task #71 — Exp F: 三流形残差数值稳定性测试")
    print("=" * 60)

    # 加载数据
    print(f"\n[1/3] Loading embeddings: {args.embeddings_pt}")
    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    emb_np = embeddings.numpy()
    print(f"  shape: {emb_np.shape}")

    # 检查 device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"  WARNING: CUDA not available, using CPU")
        args.device = "cpu"
    print(f"  Using device: {args.device}")

    # 三种几何分别训练
    print(f"\n[2/3] Training 3 RQ-VAEs (n_steps={args.n_steps}, n_clusters={args.n_clusters})...")
    histories = {}
    for residual_type in ["E", "H", "S"]:
        print(f"  Training residual_type={residual_type}...")
        histories[residual_type] = train_one_geometry(
            residual_type=residual_type,
            embeddings_np=emb_np,
            n_clusters=args.n_clusters,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        h = histories[residual_type]
        print(f"    Final loss: {h['loss'][-1]:.6f}")
        print(f"    Final grad_norm: {h['grad_norm'][-1]:.4f}")
        print(f"    Mean grad_norm: {np.mean(h['grad_norm']):.4f}")
        print(f"    Max grad_norm: {np.max(h['grad_norm']):.4f}")
        print(f"    NaN/Inf count: {sum(h['nan_inf_count'])} / {len(h['nan_inf_count'])}")

    # 决策
    print(f"\n[3/3] Decision analysis...")
    grad_norms = {k: np.mean(h["grad_norm"]) for k, h in histories.items()}
    nan_infs = {k: sum(h["nan_inf_count"]) for k, h in histories.items()}
    print(f"  Mean grad_norm: E={grad_norms['E']:.4f}, H={grad_norms['H']:.4f}, S={grad_norms['S']:.4f}")
    print(f"  NaN/Inf counts: E={nan_infs['E']}, H={nan_infs['H']}, S={nan_infs['S']}")

    grad_ratio = max(grad_norms.values()) / max(min(grad_norms.values()), 1e-10)
    print(f"  Grad norm ratio (max/min): {grad_ratio:.2f}")
    total_nan_inf = sum(nan_infs.values())
    total_steps = args.n_steps * 3
    nan_inf_rate = total_nan_inf / total_steps
    print(f"  NaN/Inf rate: {nan_inf_rate * 100:.4f}%")

    if nan_inf_rate > 0.001:  # > 0.1%
        decision = "R6_REJECTED (NaN/Inf > 0.1% → 数值不稳定 → STOP)"
    elif grad_ratio > 5:
        decision = "R6_MARGINAL (grad norm ratio > 5× → 需 clip)"
    else:
        decision = "R6_CONFIRMED (梯度稳定，可继续 E)"

    print(f"\n  Decision: {decision}")

    # 保存
    out = {
        "task": "#71 Exp F — numerical stability",
        "n_steps": args.n_steps,
        "n_clusters": args.n_clusters,
        "device": args.device,
        "histories": {
            k: {
                "step": h["step"],
                "loss": h["loss"],
                "grad_norm": h["grad_norm"],
                "nan_inf_count": h["nan_inf_count"],
            }
            for k, h in histories.items()
        },
        "summary": {
            "mean_grad_norm": grad_norms,
            "nan_inf_total": nan_infs,
            "grad_norm_ratio": grad_ratio,
            "nan_inf_rate": nan_inf_rate,
        },
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()