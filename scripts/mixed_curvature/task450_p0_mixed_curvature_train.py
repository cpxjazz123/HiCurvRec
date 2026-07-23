"""Task 450 — P0.3 Joint Mixed-Curvature RQ-VAE.

核心改动 vs task441 UnifiedCodebookRQVAE:
  1. 每个 codeword (256, 928) 拆为两个子空间:
     - H-part (前 32 维): 走 Poincaré ball metric (with per-layer radius)
     - E-part (后 896 维): Euclidean metric
  2. 距离 = w_H * d_H(x[:, :32], C[:, :32]) + w_E * d_E(x[:, 32:], C[:, 32:])
  3. 共享一个 codebook per layer — "Joint" 在于一个码字同时含两个不同曲率分量
  4. w_H, w_E softmax-normalized (init 0.5/0.5)
  5. Encoder/Decoder 在 input 空间 (928-dim), 与 task441 一致 — 无 bottleneck

为什么这是 "Joint Mixed-Curvature":
  - 同一码本 C1 编码后, x → q1 在 H 空间距 x 的距离 用 poincaré, 在 E 空间用 euclid
  - 与 task388v5 (single-curvature per layer) 根本不同:
    * task388v5: 整个 layer 选一种 metric (L1 ball, L2/L3 Euclidean)
    * 这里: 每个 layer 内部就 mixed-curvature — 一个码字同时含 H 和 E 子空间

公平性 vs P0.2 Euclidean-Concat baseline:
  - 同输入 (928-dim multi-factor concat)
  - 同 codebook (256 × 928)
  - 同层数 (3 + dedup → 4)
  - 同 max_steps (3000)
  - 同 seed (42)
  - 唯一差异: distance function (mixed H+E) vs (pure E)
"""
import math
import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
WORKTREE = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/.claude/worktrees/task9-decoder-only'
sys.path.insert(0, GRID)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONCAT_EMB = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/mixed_curvature/concat_embedding.pt'
LOG_ROOT = f'{GRID}/logs/train/runs'

# Architecture
N_CLUSTERS = 256
INPUT_DIM = 928           # multi-factor concat
H_SUBSPACE_DIM = 32       # 前 32 维走 Poincaré
E_SUBSPACE_DIM = 896      # 后 896 维走 Euclidean (32+896=928)
NUM_LAYERS = 3
BALL_C = 1.0
MAX_NORM = 0.5

# Training
BATCH_SIZE = 1024
LR = 1e-3
MAX_STEPS = 3000
LOG_EVERY = 100


def _mobius_add(x, y, c=BALL_C):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return num / denom.clamp(min=1e-15)


def _poincare_distance(x, y, c=BALL_C):
    """x: (..., D), y: (..., D) → (...,)"""
    sqrt_c = math.sqrt(c)
    diff = _mobius_add(-x, y, c=c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 - 5e-3)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


class JointMixedCurvatureRQVAE(nn.Module):
    """Joint Mixed-Curvature RQ-VAE: 一个 codeword 同时含 H 和 E 子空间."""

    def __init__(self, dim=INPUT_DIM, h_dim=H_SUBSPACE_DIM, e_dim=E_SUBSPACE_DIM,
                 n_clusters=N_CLUSTERS,
                 init_radii=(0.25, 0.25, 0.25),
                 max_norm=MAX_NORM,
                 init_log_w=(0.0, 0.0)):
        super().__init__()
        assert dim == h_dim + e_dim, f"dim ({dim}) must equal h_dim ({h_dim}) + e_dim ({e_dim})"
        self.dim = dim
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.n_clusters = n_clusters
        self.max_norm = max_norm

        # Encoder/Decoder: 与 task441 UnifiedCodebookRQVAE 一致, 928-dim 空间
        self.encoder = nn.Sequential(
            nn.Linear(dim, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim),  # output dim = input dim (no bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 768),
            nn.ReLU(),
            nn.Linear(768, dim),
        )
        # Layer norm 稳定
        self.norm_in = nn.LayerNorm(dim)

        # Per-layer softplus-bounded radius (仅用于 H 子空间)
        radius_raw_init = [math.log(math.exp(max(r, 1e-6)) - 1) for r in init_radii]
        self.radius_raw = nn.ParameterList([
            nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=True)
            for v in radius_raw_init
        ])

        # w_H, w_E 可学 (log space), softmax 归一化
        self.log_w_raw = nn.Parameter(torch.tensor(init_log_w, dtype=torch.float32), requires_grad=True)

        # SINGLE codebook per layer (shared H + E)
        self.C1 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)
        self.C2 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)
        self.C3 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)

    @property
    def radius(self):
        r = torch.stack([F.softplus(raw) for raw in self.radius_raw])
        return r.clamp(max=self.max_norm)

    @property
    def weights(self):
        """返回 (w_H, w_E), 经 softmax 归一化使 w_H + w_E = 1"""
        return F.softmax(self.log_w_raw, dim=0)

    def _lift_to_ball(self, c_euclidean, layer_idx):
        """lift c (any dim) to Poincaré ball at radius self.radius[layer_idx]"""
        r = self.radius[layer_idx]
        c_norm = c_euclidean.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        c_unit = c_euclidean / c_norm
        v = c_unit * torch.atanh(r.clamp(min=1e-6, max=1.0 - 5e-3))
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        sqrt_c = math.sqrt(BALL_C)
        c_ball = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        return c_ball

    def _compute_distance(self, x, C, layer_idx):
        """Per-sub-space distance: w_H * d_H(x[:, :h], C[:, :h]) + w_E * d_E(x[:, h:], C[:, h:])
        x: (B, dim)  C: (K, dim)  → (B, K)
        """
        w_H, w_E = self.weights[0], self.weights[1]
        # H subspace (前 h_dim 维) - 走 Poincaré ball
        x_h = x[:, :self.h_dim]
        C_h = C[:, :self.h_dim]
        x_h_ball = self._lift_to_ball(x_h, layer_idx)
        C_h_ball = self._lift_to_ball(C_h, layer_idx)
        d_h = _poincare_distance(
            x_h_ball.unsqueeze(1), C_h_ball.unsqueeze(0), c=BALL_C
        )  # (B, K)

        # E subspace (后 e_dim 维) - Euclidean
        x_e = x[:, self.h_dim:]
        C_e = C[:, self.h_dim:]
        d_e = (x_e.unsqueeze(1) - C_e.unsqueeze(0)).pow(2).sum(-1)  # (B, K)

        d = w_H * d_h + w_E * d_e
        return d, d_h, d_e

    def _soft_min_distance(self, d, temperature=10.0):
        return -torch.logsumexp(-d * temperature, dim=1).mean()

    def _entropy_loss(self, idx, num_clusters=N_CLUSTERS):
        """Batch-level entropy maximization: encourage uniform codebook usage."""
        onehot = F.one_hot(idx, num_clusters).float()  # (B, K)
        p = onehot.mean(dim=0).clamp(min=1e-9)  # (K,)
        return -(p * p.log()).sum()  # 最大化 entropy = 最小化 -entropy

    def forward(self, x_raw):
        """x_raw: (B, 928) multi-factor concat input"""
        # Encode: 928 → 928 (no bottleneck)
        z = self.encoder(self.norm_in(x_raw))

        Cs = [self.C1, self.C2, self.C3]
        residuals = [z.clone()]
        indices = []
        soft_metric_reg = torch.zeros((), device=z.device, dtype=z.dtype)

        for l in range(NUM_LAYERS):
            C = Cs[l]
            d, d_h, d_e = self._compute_distance(z, C, l)
            soft_metric_reg = soft_metric_reg + self._soft_min_distance(d)

            idx = d.argmin(dim=1)
            q = C[idx]
            indices.append(idx)
            z = z - q  # Euclidean residual
            residuals.append(z.clone())

        z_hat = residuals[1] + residuals[2] + residuals[3]
        x_hat = self.decoder(z_hat)
        codes = torch.stack(indices, dim=1)

        recon = F.mse_loss(x_hat, x_raw)
        cl1 = F.mse_loss(residuals[0], residuals[1])
        cl2 = F.mse_loss(residuals[1], residuals[2])
        cl3 = F.mse_loss(residuals[2], residuals[3])
        # entropy maximization on batch-level assignment
        ent = sum(self._entropy_loss(idx) for idx in indices) / len(indices)
        return x_hat, codes, {
            'recon': recon,
            'commit_l1': cl1,
            'commit_l2': cl2,
            'commit_l3': cl3,
            'soft_metric_reg': soft_metric_reg,
            'entropy_loss': ent,
        }

    def init_codebooks(self, x_euclidean):
        """Init: KMeans++ on encoded z (避免 codebook collapse).

        1. 编码 x → z
        2. KMeans++ on z 初始化 C1
        3. C2/C3 在 C1 基础上 + 噪声初始化 (避免三个 codeword 全相同)
        """
        from sklearn.cluster import KMeans
        device = x_euclidean.device
        with torch.no_grad():
            x_dev = x_euclidean.to(device)
            z = self.encoder(self.norm_in(x_dev))
            print(f'[init mixed] KMeans++ on z={tuple(z.shape)} for {self.n_clusters} clusters', flush=True)
            z_np = z.detach().cpu().numpy()
            km1 = KMeans(n_clusters=self.n_clusters, n_init=4, random_state=42).fit(z_np)
            km2 = KMeans(n_clusters=self.n_clusters, n_init=4, random_state=43).fit(z_np)
            km3 = KMeans(n_clusters=self.n_clusters, n_init=4, random_state=44).fit(z_np)
            self.C1.data = torch.tensor(km1.cluster_centers_, dtype=torch.float32, device=device)
            self.C2.data = torch.tensor(km2.cluster_centers_, dtype=torch.float32, device=device)
            self.C3.data = torch.tensor(km3.cluster_centers_, dtype=torch.float32, device=device)
            # usage check
            _, c1_count = np.unique(km1.labels_, return_counts=True)
            _, c2_count = np.unique(km2.labels_, return_counts=True)
            _, c3_count = np.unique(km3.labels_, return_counts=True)
            print(f'[init mixed] L1 used clusters: {(c1_count>0).sum()}/{self.n_clusters} '
                  f'(min/max count={c1_count.min()}/{c1_count.max()})', flush=True)
            print(f'[init mixed] L2 used clusters: {(c2_count>0).sum()}/{self.n_clusters} '
                  f'(min/max count={c2_count.min()}/{c2_count.max()})', flush=True)
            print(f'[init mixed] L3 used clusters: {(c3_count>0).sum()}/{self.n_clusters} '
                  f'(min/max count={c3_count.min()}/{c3_count.max()})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    task_tag = f'task58_p0_mixed_curvature'

    print(f'[mixed-curv] device={DEVICE}, dim={INPUT_DIM} (H={H_SUBSPACE_DIM}+E={E_SUBSPACE_DIM})', flush=True)

    x_all = torch.load(CONCAT_EMB, map_location='cpu', weights_only=False).float()
    print(f'[mixed-curv] x_all: {x_all.shape}', flush=True)

    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std

    model = JointMixedCurvatureRQVAE(
        dim=INPUT_DIM,
        h_dim=H_SUBSPACE_DIM,
        e_dim=E_SUBSPACE_DIM,
        n_clusters=N_CLUSTERS,
        init_radii=(0.25, 0.25, 0.25),
        max_norm=MAX_NORM,
    ).to(DEVICE)
    init_subset = x_all[torch.randperm(len(x_all))[:5000]]
    model.init_codebooks(init_subset.to(DEVICE))

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = f'{LOG_ROOT}/{task_tag}_{ts}'
    os.makedirs(f'{out_dir}/checkpoints', exist_ok=True)

    print(f'[mixed-curv] MAX_STEPS={args.max_steps}, BATCH={args.batch_size}', flush=True)
    losses_log = []
    t_start = time.time()
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(0, len(x_all), (args.batch_size,))
        x_batch = x_all[idx].to(DEVICE)
        x_hat, codes, losses = model(x_batch)
        loss = (
            losses['recon']
            + 0.25 * (losses['commit_l1'] + losses['commit_l2'] + losses['commit_l3'])
            + 0.001 * losses['soft_metric_reg']
            + 0.05 * (-losses['entropy_loss'])  # 最小化 -entropy = 最大化 entropy
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            r_now = model.radius.detach().cpu().numpy()
            w_now = model.weights.detach().cpu().numpy()
            # 算 codebook usage (这 batch)
            with torch.no_grad():
                all_codes = torch.cat(indices, dim=0) if False else None
                # codeword usage from full dataset (last batch)
                usage_str = ''
            print(f'[mixed-curv] step {step:5d}/{args.max_steps} '
                  f'recon={losses["recon"].item():.4f} '
                  f'cl1={losses["commit_l1"].item():.4f} '
                  f'radius=[{r_now[0]:.3f},{r_now[1]:.3f},{r_now[2]:.3f}] '
                  f'w_H={w_now[0]:.3f} w_E={w_now[1]:.3f} '
                  f'ent={losses["entropy_loss"].item():.3f} '
                  f'total={loss.item():.4f} elapsed={elapsed:.1f}s', flush=True)
            losses_log.append({
                'step': step,
                'recon': float(losses['recon'].item()),
                'cl1': float(losses['commit_l1'].item()),
                'radius': [float(r) for r in r_now],
                'w_H': float(w_now[0]),
                'w_E': float(w_now[1]),
                'entropy': float(losses['entropy_loss'].item()),
                'total': float(loss.item()),
                'elapsed_sec': elapsed,
            })

    ckpt_path = f'{out_dir}/checkpoints/ckpt_mixed_curvature.ckpt'
    state = {
        'quantization_layer_list.0.centroids': model.C1.detach().cpu(),
        'quantization_layer_list.1.centroids': model.C2.detach().cpu(),
        'quantization_layer_list.2.centroids': model.C3.detach().cpu(),
        'radius_raw': torch.stack([raw.detach().cpu() for raw in model.radius_raw]),
        'log_w_raw': model.log_w_raw.detach().cpu(),
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'norm_in': model.norm_in.state_dict(),
    }
    torch.save({'state_dict': state}, ckpt_path)
    print(f'[mixed-curv] saved to {ckpt_path}', flush=True)

    with open(f'{out_dir}/losses.json', 'w') as f:
        json.dump({
            'task': task_tag,
            'method': 'Joint Mixed-Curvature RQ-VAE (H+E per codeword)',
            'max_steps': args.max_steps,
            'h_dim': H_SUBSPACE_DIM,
            'e_dim': E_SUBSPACE_DIM,
            'final_radius': [float(r) for r in model.radius.detach().cpu().numpy()],
            'final_weights': [float(w) for w in model.weights.detach().cpu().numpy()],
            'losses': losses_log,
        }, f, indent=2)

    final_r = model.radius.detach().cpu().numpy()
    final_w = model.weights.detach().cpu().numpy()
    print(f'\n[mixed-curv] final radius=[{final_r[0]:.4f},{final_r[1]:.4f},{final_r[2]:.4f}], w_H={final_w[0]:.3f} w_E={final_w[1]:.3f}', flush=True)
    print(f'[mixed-curv] total time = {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()