"""
Task 62 — 训练 3 层 RQ-VAE（EMA 更新，防码本坍缩）
用法: python scripts/task6_train_rqvae.py [--device cuda:0]
输出: products/task16/from_task62/rqvae_ckpt.pt

使用 EMA（指数移动平均）更新码本，而非 SGD。
这是 VQ-VAE 论文 (NeurIPS 2017) 的标准做法，可有效防止码本坍缩。
"""

import os, sys, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizeEMA(nn.Module):
    """单层 VQ，EMA 码本更新"""
    def __init__(self, n_features, n_clusters, beta=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        # 码本: (n_clusters, n_features)，用 kaiming 初始化
        self.codebook = nn.Parameter(
            torch.randn(n_clusters, n_features) * (1.0 / n_features ** 0.5),
            requires_grad=False,  # EMA 更新，不需要梯度
        )
        # EMA 统计量
        self.register_buffer("cluster_size", torch.zeros(n_clusters))
        self.register_buffer("codebook_avg", torch.zeros(n_clusters, n_features))

    def forward(self, x, update_codebook=True):
        """
        x: (B, D)
        update_codebook: 训练时 True，推理时 False
        returns: z_q (B, D), indices (B,), commitment_loss (scalar)
        """
        B, D = x.shape

        # (B, K) 距离矩阵: cdist((B,D), (K,D)) → (B,K)
        dist = torch.cdist(x, self.codebook, p=2)
        indices = dist.argmin(dim=1)  # (B,)

        z_q = self.codebook[indices]  # (B, D)

        # Commitment loss （只在训练时使用）
        commitment_loss = self.beta * F.mse_loss(x, z_q.detach(), reduction='mean')

        # EMA 更新码本
        if update_codebook and self.training:
            # One-hot encoding of assignments
            enc = F.one_hot(indices, num_classes=self.n_clusters).float()  # (B, K)
            # Sum over batch
            batch_cluster_size = enc.sum(dim=0)  # (K,)
            # Weighted sum of assigned vectors
            batch_codebook_avg = enc.T @ x  # (K, D)

            # EMA update
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * batch_cluster_size
            self.codebook_avg.data = self.decay * self.codebook_avg + (1 - self.decay) * batch_codebook_avg

            # Normalize: divide by cluster size and update codebook
            n = self.cluster_size.sum()
            laplace_smooth = (self.cluster_size + self.epsilon) / (n + self.n_clusters * self.epsilon) * n
            codebook_new = self.codebook_avg / laplace_smooth.unsqueeze(1)
            self.codebook.data = codebook_new

        # Straight-through estimator
        z_q = x + (z_q - x).detach()

        return z_q, indices, commitment_loss


class ResidualQuantization(nn.Module):
    """N 层残差量化"""
    def __init__(self, n_features, n_layers=3, n_clusters=256, decay=0.99):
        super().__init__()
        self.n_layers = n_layers
        self.quantizers = nn.ModuleList([
            VectorQuantizeEMA(n_features, n_clusters, decay=decay) for _ in range(n_layers)
        ])

    def forward(self, x, update_codebook=True):
        residual = x
        z_q_list = []
        indices_list = []
        residuals_list = []
        total_commit_loss = 0.0

        for i, q in enumerate(self.quantizers):
            z_q, indices, commit_loss = q(residual, update_codebook=update_codebook)
            z_q_list.append(z_q)
            indices_list.append(indices)
            residuals_list.append(residual - z_q)
            residual = residual - z_q.detach()
            total_commit_loss += commit_loss

        return z_q_list, indices_list, residuals_list, total_commit_loss

    def get_codebook_usage(self, x, batch_size=4096):
        """返回每层码本利用率 (%)"""
        with torch.no_grad():
            usage = []
            subset = x[:batch_size].to(x.device)
            for q in self.quantizers:
                _, indices, _ = q(subset, update_codebook=False)
                n_used = indices.unique().numel()
                usage.append((n_used / q.n_clusters) * 100)
        return usage

    def encode(self, x):
        """只编码，返回每层索引 (n_layers, B)"""
        residual = x
        all_indices = []
        with torch.no_grad():
            for q in self.quantizers:
                dist = torch.cdist(residual, q.codebook, p=2)
                indices = dist.argmin(dim=1)
                all_indices.append(indices)
                residual = residual - q.codebook[indices]
        return torch.stack(all_indices, dim=0)


def init_codebooks_with_kmeans(model, embeddings, device='cuda:0'):
    """用 k-means 初始化每层码本"""
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    print("Initializing codebooks with MiniBatchKMeans...")
    emb_np = embeddings.cpu().numpy()
    residual = torch.tensor(emb_np, device='cpu')
    for layer_idx, q in enumerate(model.quantizers):
        print(f"  Layer {layer_idx + 1}: k-means (K={q.n_clusters})...")
        kmeans = MiniBatchKMeans(n_clusters=q.n_clusters, batch_size=1024,
                                 random_state=42, n_init=3, max_iter=100)
        kmeans.fit(residual.numpy())
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        # 更新码本
        q.codebook.data = centroids.to(device)
        # 重置 EMA 统计量
        q.cluster_size.data.zero_()
        q.codebook_avg.data.zero_()
        # 计算残差（cdist 直接支持 (N, D) x (K, D) → (N, K)）
        dist = torch.cdist(residual, centroids, p=2)
        ind = dist.argmin(dim=1)
        z_q = centroids[ind]
        residual = residual - z_q
        usage = (ind.unique().numel() / q.n_clusters) * 100
        print(f"    Usage: {usage:.1f}%")
    print("  K-means initialization done.")


def train_rqvae_ema(embeddings, n_layers=3, n_clusters=256, n_steps=2000,
                    batch_size=512, decay=0.99, device='cuda:0',
                    init_kmeans=True):
    """
    EMA 方式训练 RQ-VAE，比 SGD 更稳定。
    """
    N, D = embeddings.shape
    model = ResidualQuantization(
        D, n_layers=n_layers, n_clusters=n_clusters, decay=decay
    ).to(device)
    model.train()

    # K-means 初始化
    if init_kmeans:
        init_codebooks_with_kmeans(model, embeddings, device)

    dataset = embeddings.to(device)
    n_batches = max(1, N // batch_size)

    print(f"\nTraining RQ-VAE (EMA): N={N}, D={D}, layers={n_layers}, clusters={n_clusters}")
    print(f"  Steps: {n_steps}, batch_size: {batch_size}, decay={decay}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        perm = torch.randperm(N, device=device)
        shuffled = dataset[perm]
        total_loss = 0.0

        for i in range(n_batches):
            batch = shuffled[i * batch_size: (i + 1) * batch_size]
            _, _, _, commit_loss = model(batch, update_codebook=True)
            # EMA 更新码本时不需要 loss.backward()（码本不通过梯度更新）
            # 但 commitment_loss 用于训练 encoder（如果有 encoder 的话）
            # 目前我们没有独立的 encoder，所以 commitment_loss 直接忽略
            # 码本更新由 EMA 在前向传播中自动完成
            total_loss += commit_loss.item()

        avg_loss = total_loss / n_batches

        if step % 100 == 0 or step == 1:
            usage = model.get_codebook_usage(dataset)
            usage_str = ", ".join([f"L{i+1}={u:.1f}%" for i, u in enumerate(usage)])
            elapsed = time.time() - t0
            print(f"  Step {step:6d}/{n_steps} | cmt_loss={avg_loss:.6f} | {usage_str} | {elapsed:.0f}s")

            # 检查是否所有码本利用率都 > 50% — 可以提前结束
            if all(u > 50 for u in usage) and step >= 500:
                print(f"  All codebooks > 50% utilized at step {step}. Early stopping.")
                break

    total_time = time.time() - t0
    print(f"Training done. Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay")
    parser.add_argument("--no_kmeans_init", action="store_true", help="Skip k-means init")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    emb_path = "products/task16/from_task62/item_embeddings.pt"
    if not os.path.exists(emb_path):
        print(f"ERROR: {emb_path} not found. Run task6_extract_data.py first.")
        sys.exit(1)

    embeddings = torch.load(emb_path, map_location="cpu")
    print(f"Loaded embeddings: {embeddings.shape}")

    model = train_rqvae_ema(
        embeddings,
        n_layers=args.n_layers,
        n_clusters=args.n_clusters,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        decay=args.decay,
        device=device,
        init_kmeans=not args.no_kmeans_init,
    )

    out_dir = "products/task16/from_task62"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "rqvae_ckpt.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "n_layers": args.n_layers,
        "n_clusters": args.n_clusters,
        "n_features": embeddings.shape[1],
        "config": vars(args),
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
