"""
Task #71 — Exp E Stage 2.1: 三流形 RQ-VAE 单层 trainer

支持 r_E / r_H / r_S 三种残差几何。
单层简化版（暂只测 L1），验证三流形 RQ-VAE 训练可行性。

启动:
  python scripts/task6_exp_e_s21_train.py \
    --residual_type {E,H,S} \
    --device cuda:0 \
    --n_clusters 256 \
    --n_steps 500
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


# ============== 单层三流形 RQ-VAE ==============

class ThreeManifoldRQ(nn.Module):
    """单层 RQ-VAE，支持 r_E / r_H / r_S 残差几何"""

    def __init__(self, n_features, n_clusters, residual_type="E", decay=0.99):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.residual_type = residual_type

        # Codebook 用 K-Means 初始化（避免 EMA 坍缩问题）
        self.codebook = nn.Parameter(
            torch.randn(n_clusters, n_features) * (1.0 / n_features ** 0.5),
            requires_grad=False,
        )
        self.register_buffer("cluster_size", torch.zeros(n_clusters))
        self.register_buffer("codebook_avg", torch.zeros(n_clusters, n_features))

        # Decoder: simple linear from codeword to reconstruction
        self.decoder = nn.Linear(n_features, n_features)

    def compute_residual(self, x, q):
        if self.residual_type == "E":
            return x - q
        elif self.residual_type == "H":
            return poincare_log_map(project_to_poincare(q), project_to_poincare(x))
        elif self.residual_type == "S":
            return spherical_log_map(project_to_sphere(q), project_to_sphere(x))
        else:
            raise ValueError(self.residual_type)

    def forward(self, x, update_codebook=True):
        """Forward pass: 量化 + 残差 + 重建"""
        # 1. 找最近码字
        d = torch.cdist(x, self.codebook, p=2)
        idx = d.argmin(dim=1)
        q = self.codebook[idx]

        # 2. 残差（按几何）
        r = self.compute_residual(x, q)

        # 3. 重建（用 decoder from q）
        x_hat = self.decoder(q)
        recon_loss = F.mse_loss(x_hat, x)

        # 4. Commitment loss（鼓励 encoder 接近 codeword）
        commitment_loss = F.mse_loss(x, q.detach())

        # 5. Codebook loss（鼓励 codebook 接近 encoder）
        codebook_loss = F.mse_loss(q, x.detach())

        loss = recon_loss + commitment_loss + codebook_loss

        # 6. EMA codebook 更新
        if self.training and update_codebook:
            with torch.no_grad():
                # One-hot encoding of assignments
                onehot = F.one_hot(idx, num_classes=self.n_clusters).float()  # (B, K)
                # Cluster size update
                new_cluster_size = onehot.sum(dim=0)  # (K,)
                self.cluster_size.mul_(0.99).add_(new_cluster_size, alpha=0.01)
                # Codebook avg update
                embed_sum = onehot.t() @ x.detach()  # (K, D)
                self.codebook_avg.mul_(0.99).add_(embed_sum, alpha=0.01)
                # Update codebook
                n = self.cluster_size.sum()
                cluster_size_smoothed = (
                    (self.cluster_size + 1e-5) / (n + self.n_clusters * 1e-5) * n
                )
                embed_normalized = self.codebook_avg / cluster_size_smoothed.unsqueeze(1)
                self.codebook.copy_(embed_normalized)

        return {
            "loss": loss,
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "codebook_loss": codebook_loss.item(),
            "indices": idx,
            "reconstruction": x_hat,
            "residual": r,
            "q": q,
        }


def train_one_layer(residual_type, embeddings_np, n_clusters=256, n_steps=2000,
                    batch_size=512, lr=1e-3, device="cuda:0", seed=42,
                    val_split=0.1, log_interval=200):
    """训练单层 RQ-VAE"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_items, n_features = embeddings_np.shape

    # Train/val split
    perm = np.random.permutation(n_items)
    n_val = int(n_items * val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32, device=device)
    train_data = embeddings_t[train_idx]
    val_data = embeddings_t[val_idx]

    # Init codebook with K-Means on training data
    print(f"  Init codebook with K-Means (n_clusters={n_clusters})...")
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed,
        batch_size=1024, n_init=3, max_iter=100,
    )
    km.fit(embeddings_np[train_idx])
    initial_codebook = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    model = ThreeManifoldRQ(n_features, n_clusters, residual_type).to(device)
    model.codebook.data.copy_(initial_codebook.to(device))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = {
        "step": [],
        "train_loss": [],
        "train_recon": [],
        "val_recon": [],
        "codebook_usage": [],  # unique clusters used
    }

    print(f"  Training {residual_type} on {len(train_idx)} items, "
          f"val on {len(val_idx)} items, {n_steps} steps...")

    for step in range(n_steps):
        model.train()
        # Random batch
        idx = np.random.randint(0, len(train_idx), size=batch_size)
        batch = train_data[idx]

        optimizer.zero_grad()
        out = model(batch, update_codebook=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        if (step + 1) % log_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data, update_codebook=False)
                train_out = model(batch, update_codebook=False)

            unique_clusters = len(set(out["indices"].tolist()))
            history["step"].append(step)
            history["train_loss"].append(out["loss"].item())
            history["train_recon"].append(train_out["recon_loss"])
            history["val_recon"].append(val_out["recon_loss"])
            history["codebook_usage"].append(unique_clusters)

            print(f"    step {step + 1}/{n_steps}: "
                  f"train_loss={out['loss'].item():.4f}, "
                  f"train_recon={train_out['recon_loss']:.4f}, "
                  f"val_recon={val_out['recon_loss']:.4f}, "
                  f"codebook_usage={unique_clusters}/{n_clusters}")

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_type", choices=["E", "H", "S"], required=True)
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_dir", default="products/task16/from_task71/exp_e_s21")
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Task #71 Exp E Stage 2.1 — Single Layer Three-Manifold RQ-VAE")
    print(f"  residual_type: {args.residual_type}")
    print(f"  n_clusters: {args.n_clusters}")
    print(f"  n_steps: {args.n_steps}")
    print(f"  device: {args.device}")
    print("=" * 60)

    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    emb_np = embeddings.numpy()
    print(f"\nEmbeddings shape: {emb_np.shape}")

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        print(f"  CUDA not available, using CPU")

    os.makedirs(args.out_dir, exist_ok=True)

    # 训练
    model, history = train_one_layer(
        residual_type=args.residual_type,
        embeddings_np=emb_np,
        n_clusters=args.n_clusters,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    # 保存 codebook + history
    codebook_path = os.path.join(args.out_dir, f"codebook_{args.residual_type}.pt")
    torch.save({
        "codebook": model.codebook.detach().cpu(),
        "decoder_weight": model.decoder.weight.detach().cpu(),
        "decoder_bias": model.decoder.bias.detach().cpu(),
        "residual_type": args.residual_type,
        "n_features": model.n_features,
        "n_clusters": model.n_clusters,
    }, codebook_path)
    print(f"\n  Saved codebook: {codebook_path}")

    history_path = os.path.join(args.out_dir, f"history_{args.residual_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Saved history: {history_path}")

    # 总结
    print(f"\n{'='*60}")
    print(f"Stage 2.1 Summary (residual_type={args.residual_type})")
    print(f"{'='*60}")
    print(f"  Final train recon loss: {history['train_recon'][-1]:.6f}")
    print(f"  Final val recon loss: {history['val_recon'][-1]:.6f}")
    print(f"  Final codebook usage: {history['codebook_usage'][-1]}/{args.n_clusters}")


if __name__ == "__main__":
    main()