"""
Task #71 — Exp E Stage 2.2: 三流形 L=3 多层 RQ-VAE trainer

L=3 层 cascade:
  Layer 1: q_1 = nearest(x, C1);  r_1 = compute_residual(x, q_1, geom)
  Layer 2: q_2 = nearest(r_1, C2); r_2 = compute_residual(r_1, q_2, geom)
  Layer 3: q_3 = nearest(r_2, C3); r_3 = compute_residual(r_2, q_3, geom)
  Final: x_hat = q_1 + q_2 + q_3

残差几何（E/H/S）只影响 r_1/r_2/r_3 的计算。
最近码字选择始终用欧氏距离（与 Task #70 D1 一致：任何距离排序相同）。

启动:
  python scripts/task6_exp_e_s22_train_multilayer.py \
    --residual_type {E,H,S} \
    --device cuda:0 \
    --n_steps 1500
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


# ============== 流形几何函数 ==============

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


def compute_residual(x, q, residual_type):
    """按几何类型计算残差"""
    if residual_type == "E":
        return x - q
    elif residual_type == "H":
        return poincare_log_map(project_to_poincare(q), project_to_poincare(x))
    elif residual_type == "S":
        return spherical_log_map(project_to_sphere(q), project_to_sphere(x))
    else:
        raise ValueError(residual_type)


# ============== 多层 RQ-VAE ==============

class MultiLayerRQ(nn.Module):
    """L 层 RQ-VAE，支持 r_E / r_H / r_S 残差几何"""

    def __init__(self, n_features, n_clusters=256, n_layers=3, residual_type="E",
                 decay=0.99, eps=1e-5):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_layers = n_layers
        self.residual_type = residual_type
        self.decay = decay
        self.eps = eps

        # 每层独立 codebook + EMA buffers
        self.codebooks = nn.ParameterList([
            nn.Parameter(
                torch.randn(n_clusters, n_features) * (1.0 / n_features ** 0.5),
                requires_grad=True,
            )
            for _ in range(n_layers)
        ])
        self.cluster_sizes = nn.ParameterList([
            nn.Parameter(torch.zeros(n_clusters), requires_grad=False)
            for _ in range(n_layers)
        ])
        self.codebook_avgs = nn.ParameterList([
            nn.Parameter(torch.zeros(n_clusters, n_features), requires_grad=False)
            for _ in range(n_layers)
        ])

    @torch.no_grad()
    def ema_update(self, layer_idx, x, idx):
        """EMA 更新单层 codebook"""
        cb = self.codebooks[layer_idx]
        cs = self.cluster_sizes[layer_idx]
        ca = self.codebook_avgs[layer_idx]
        onehot = F.one_hot(idx, num_classes=self.n_clusters).float()  # (B, K)
        new_cs = onehot.sum(dim=0)
        cs.mul_(self.decay).add_(new_cs, alpha=1 - self.decay)
        embed_sum = onehot.t() @ x  # (K, D)
        ca.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = cs.sum()
        cluster_size_smoothed = ((cs + self.eps) / (n + self.n_clusters * self.eps) * n)
        embed_normalized = ca / cluster_size_smoothed.unsqueeze(1)
        cb.copy_(embed_normalized)

    def quantize_layer(self, x, layer_idx):
        """单层量化（返回 codeword + index）"""
        cb = self.codebooks[layer_idx]
        d = torch.cdist(x, cb, p=2)
        idx = d.argmin(dim=1)
        q = cb[idx]
        return q, idx

    def forward(self, x, update_codebook=True):
        """L 层 cascade forward"""
        all_q = []
        all_idx = []
        all_r = []
        cur = x
        total_commit_loss = 0.0
        total_codebook_loss = 0.0

        for l in range(self.n_layers):
            q, idx = self.quantize_layer(cur, l)
            r = compute_residual(cur, q, self.residual_type)

            # Commitment + codebook loss（每层独立）
            cl = F.mse_loss(cur, q.detach())
            bl = F.mse_loss(q, cur.detach())
            total_commit_loss = total_commit_loss + cl
            total_codebook_loss = total_codebook_loss + bl

            all_q.append(q)
            all_idx.append(idx)
            all_r.append(r)

            if self.training and update_codebook:
                self.ema_update(l, cur.detach(), idx)

            # 下一层输入：原始 RQ-VAE 残差 (Euclidean) — 但这里我们要测几何差异
            # 关键设计：cur = r（按几何计算的残差），传到下一层
            cur = r

        # 最终重建 = 所有 q 之和（按几何逆运算很难严格恢复 → 用求和近似）
        x_hat = sum(all_q)

        # 重建损失
        recon_loss = F.mse_loss(x_hat, x)

        loss = recon_loss + total_commit_loss + total_codebook_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach().item(),
            "commit_loss": total_commit_loss.detach().item(),
            "codebook_loss": total_codebook_loss.detach().item(),
            "indices": all_idx,  # list of (B,) per layer
            "q_sum": x_hat,
            "all_q": all_q,
            "all_r": all_r,
        }


def train_multilayer(residual_type, embeddings_np, n_clusters=256, n_layers=3,
                     n_steps=1500, batch_size=512, lr=1e-3, device="cuda:0",
                     seed=42, val_split=0.1, log_interval=150):
    """训练 L 层 RQ-VAE"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_items, n_features = embeddings_np.shape

    perm = np.random.permutation(n_items)
    n_val = int(n_items * val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32, device=device)
    train_data = embeddings_t[train_idx]
    val_data = embeddings_t[val_idx]

    print(f"  Init {n_layers} codebooks with K-Means...")
    model = MultiLayerRQ(n_features, n_clusters, n_layers, residual_type).to(device)
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        batch_size=1024, n_init=3, max_iter=100,
    )
    km.fit(embeddings_np[train_idx])
    # L1 codebook = K-Means on raw data
    model.codebooks[0].data.copy_(
        torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    )
    # L2/L3 codebooks = K-Means on Stage 2.1 residuals (欧氏, 与 E 流一致)
    cur = embeddings_t  # use full data for residual init
    for l in range(1, n_layers):
        with torch.no_grad():
            # cascade init: use E residual to find q_l-1
            r = _get_q_at_layer(model, cur, l)
            r_np = r.cpu().numpy()
        km_l = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed,
            batch_size=1024, n_init=3, max_iter=100,
        )
        km_l.fit(r_np)
        model.codebooks[l].data.copy_(
            torch.tensor(km_l.cluster_centers_, dtype=torch.float32, device=device)
        )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = {
        "step": [],
        "train_loss": [],
        "train_recon": [],
        "val_recon": [],
        "layer_usage": [[] for _ in range(n_layers)],
    }

    print(f"  Training {residual_type} ({n_layers}-layer) on {len(train_idx)} items, "
          f"val on {len(val_idx)} items, {n_steps} steps...")

    for step in range(n_steps):
        model.train()
        idx = np.random.randint(0, len(train_idx), size=batch_size)
        batch = train_data[idx]

        optimizer.zero_grad()
        out = model(batch, update_codebook=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % log_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data, update_codebook=False)
                train_out = model(batch, update_codebook=False)

            layer_usage = [
                len(set(out["indices"][l].tolist()))
                for l in range(n_layers)
            ]

            history["step"].append(step)
            history["train_loss"].append(out["loss"].item())
            history["train_recon"].append(train_out["recon_loss"])
            history["val_recon"].append(val_out["recon_loss"])
            for l in range(n_layers):
                history["layer_usage"][l].append(layer_usage[l])

            usage_str = " | ".join([f"L{l+1}:{u}/{n_clusters}" for l, u in enumerate(layer_usage)])
            print(f"    step {step + 1}/{n_steps}: "
                  f"loss={out['loss'].item():.4f}, "
                  f"train_recon={train_out['recon_loss']:.6f}, "
                  f"val_recon={val_out['recon_loss']:.6f} | {usage_str}")

    return model, history


def _get_q_at_layer(model, x, target_layer):
    """Helper: 沿 E 残差 cascade 到第 target_layer"""
    cur = x
    with torch.no_grad():
        for l in range(target_layer):
            q, _ = model.quantize_layer(cur, l)
            cur = cur - q
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_type", choices=["E", "H", "S"], required=True)
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_dir", default="products/task16/from_task71/exp_e_s22")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Task #71 Exp E Stage 2.2 — L={args.n_layers} Multi-Layer Three-Manifold RQ-VAE")
    print(f"  residual_type: {args.residual_type}")
    print(f"  n_clusters: {args.n_clusters}")
    print(f"  n_steps: {args.n_steps}")
    print(f"  device: {args.device}")
    print("=" * 70)

    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    emb_np = embeddings.numpy()
    print(f"\nEmbeddings shape: {emb_np.shape}")

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    model, history = train_multilayer(
        residual_type=args.residual_type,
        embeddings_np=emb_np,
        n_clusters=args.n_clusters,
        n_layers=args.n_layers,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    codebook_path = os.path.join(args.out_dir, f"codebooks_{args.residual_type}.pt")
    torch.save({
        "codebooks": [cb.detach().cpu() for cb in model.codebooks],
        "residual_type": args.residual_type,
        "n_features": model.n_features,
        "n_clusters": model.n_clusters,
        "n_layers": model.n_layers,
    }, codebook_path)
    print(f"\n  Saved codebooks: {codebook_path}")

    history_path = os.path.join(args.out_dir, f"history_{args.residual_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved history: {history_path}")

    print(f"\n{'='*70}")
    print(f"Stage 2.2 Summary (residual_type={args.residual_type})")
    print(f"{'='*70}")
    print(f"  Final train recon loss: {history['train_recon'][-1]:.6f}")
    print(f"  Final val recon loss: {history['val_recon'][-1]:.6f}")
    for l in range(args.n_layers):
        print(f"  Layer {l+1} usage: {history['layer_usage'][l][-1]}/{args.n_clusters}")


if __name__ == "__main__":
    main()
