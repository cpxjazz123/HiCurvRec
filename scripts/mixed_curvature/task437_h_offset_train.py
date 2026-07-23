#!/usr/bin/env python3
"""task56_h_offset_train.py — Task #437 Step 1: 支持 h_offset 参数的 Mixed-Curvature 训练

修改:
  - h_offset 参数: H 子空间起始位置 (默认 0, 与 P0.3 一致)
  - h_dim=0 时取消 H 子空间 (全 E, 等价 P0.2 baseline)
  - h_dim=32, h_offset=768 时 H 在 brand 范围
  - h_dim=96, h_offset=800 时 H 在 taxonomy 范围
  - h_dim=32, h_offset=896 时 H 在 behavior 范围

用法:
  python3 task56_h_offset_train.py --h_offset=0  --h_dim=32   # P0.3 baseline
  python3 task56_h_offset_train.py --h_offset=0  --h_dim=0    # 反事实 A: 取消 H
  python3 task56_h_offset_train.py --h_offset=768 --h_dim=32  # 反事实 B: H 在 brand
  python3 task56_h_offset_train.py --h_offset=800 --h_dim=96  # 反事实 C: H 在 taxonomy
  python3 task56_h_offset_train.py --h_offset=896 --h_dim=32  # 反事实 D: H 在 behavior
"""
import os
import sys
import math
import argparse
import time
from pathlib import Path
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
INPUT_DIM = 928
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


class HOffsetMixedCurvatureRQVAE(nn.Module):
    """Mixed-Curvature RQ-VAE with configurable H subspace offset."""

    def __init__(self, dim=INPUT_DIM, h_dim=32, h_offset=0,
                 n_clusters=N_CLUSTERS,
                 init_radii=(0.25, 0.25, 0.25),
                 max_norm=MAX_NORM,
                 init_log_w=(0.0, 0.0)):
        super().__init__()
        assert h_offset >= 0
        assert h_offset + h_dim <= dim, f"h_offset ({h_offset}) + h_dim ({h_dim}) must <= dim ({dim})"
        self.dim = dim
        self.h_dim = h_dim
        self.h_offset = h_offset
        self.n_clusters = n_clusters
        self.max_norm = max_norm

        # Encoder/Decoder: same as task441 UnifiedCodebookRQVAE
        self.encoder = nn.Sequential(
            nn.Linear(dim, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim),
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
        self.norm_in = nn.LayerNorm(dim)

        # Per-layer softplus-bounded radius (only used when h_dim > 0)
        radius_raw_init = [math.log(math.exp(max(r, 1e-6)) - 1) for r in init_radii]
        self.radius_raw = nn.ParameterList([
            nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=True)
            for v in radius_raw_init
        ])

        # w_H, w_E learnable (log space), softmax normalized
        self.log_w_raw = nn.Parameter(torch.tensor(init_log_w, dtype=torch.float32), requires_grad=True)

        # Codebooks (one per layer, shared H + E)
        self.C1 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)
        self.C2 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)
        self.C3 = nn.Parameter(torch.zeros(n_clusters, dim), requires_grad=True)

    @property
    def radius(self):
        r = torch.stack([F.softplus(raw) for raw in self.radius_raw])
        return r.clamp(max=self.max_norm)

    @property
    def weights(self):
        return F.softmax(self.log_w_raw, dim=0)

    def _lift_to_ball(self, c_euclidean, layer_idx):
        r = self.radius[layer_idx]
        c_norm = c_euclidean.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        c_unit = c_euclidean / c_norm
        v = c_unit * torch.atanh(r.clamp(min=1e-6, max=1.0 - 5e-3))
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        sqrt_c = math.sqrt(BALL_C)
        c_ball = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        return c_ball

    def _compute_distance(self, x, C, layer_idx):
        """Per-sub-space distance with configurable H offset.

        H subspace: x[:, h_offset:h_offset+h_dim], C[:, h_offset:h_offset+h_dim]
        E subspace: concatenation of x[:, :h_offset] and x[:, h_offset+h_dim:]
        """
        w_H, w_E = self.weights[0], self.weights[1]

        # E subspace first (always present)
        if self.h_offset > 0:
            x_e_pre = x[:, :self.h_offset]
            C_e_pre = C[:, :self.h_offset]
        else:
            x_e_pre = x[:, :0]
            C_e_pre = C[:, :0]
        if self.h_offset + self.h_dim < self.dim:
            x_e_post = x[:, self.h_offset + self.h_dim:]
            C_e_post = C[:, self.h_offset + self.h_dim:]
        else:
            x_e_post = x[:, :0]
            C_e_post = C[:, :0]
        x_e = torch.cat([x_e_pre, x_e_post], dim=-1)
        C_e = torch.cat([C_e_pre, C_e_post], dim=-1)
        d_e = (x_e.unsqueeze(1) - C_e.unsqueeze(0)).pow(2).sum(-1)

        # H subspace (only if h_dim > 0)
        if self.h_dim > 0:
            x_h = x[:, self.h_offset:self.h_offset + self.h_dim]
            C_h = C[:, self.h_offset:self.h_offset + self.h_dim]
            x_h_ball = self._lift_to_ball(x_h, layer_idx)
            C_h_ball = self._lift_to_ball(C_h, layer_idx)
            d_h = _poincare_distance(
                x_h_ball.unsqueeze(1), C_h_ball.unsqueeze(0), c=BALL_C
            )
            d = w_H * d_h + w_E * d_e
            return d, d_h, d_e
        else:
            # No H subspace, pure Euclidean
            return d_e, torch.zeros_like(d_e), d_e

    def _soft_min_distance(self, d, temperature=10.0):
        return -torch.logsumexp(-d * temperature, dim=1).mean()

    def _entropy_loss(self, idx, num_clusters=N_CLUSTERS):
        onehot = F.one_hot(idx, num_clusters).float()
        p = onehot.mean(dim=0).clamp(min=1e-9)
        return -(p * p.log()).sum()

    def forward(self, x_raw):
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
            z = z - q
            residuals.append(z.clone())

        z_hat = residuals[1] + residuals[2] + residuals[3]
        x_hat = self.decoder(z_hat)
        codes = torch.stack(indices, dim=1)

        recon = F.mse_loss(x_hat, x_raw)
        cl1 = F.mse_loss(residuals[0], residuals[1])
        cl2 = F.mse_loss(residuals[1], residuals[2])
        cl3 = F.mse_loss(residuals[2], residuals[3])
        ent = self._entropy_loss(indices[-1])
        total = recon + 0.25 * (cl1 + cl2 + cl3) + 0.01 * ent + 0.001 * soft_metric_reg
        return total, codes, recon, ent, soft_metric_reg


def init_codebooks(model: HOffsetMixedCurvatureRQVAE, x_euclidean: torch.Tensor, device='cuda'):
    """K-means init for codebooks — 在 encoded z 上跑 (task58 路径).

    关键改动 vs 之前:
      1. 先 encode x → z (与 forward 一致)
      2. 在 z 上跑 KMeans (而非 raw x)
      3. 3 个 codebook 用 3 个不同 random_state (而非 3 段切片)
      4. 用 5000 samples 子集 (task58 同款, 加速 init)
    """
    from sklearn.cluster import KMeans
    n = x_euclidean.shape[0]
    print(f'[init] encoding {n} samples → z...', flush=True)
    with torch.no_grad():
        x_dev = x_euclidean.to(device)
        z = model.encoder(model.norm_in(x_dev))
        z_np = z.detach().cpu().numpy()
    # Subsample 5000 for KMeans speed (matches task58 init_subset=5000)
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(n, size=min(5000, n), replace=False)
    z_sub = z_np[subset_idx]
    print(f'[init] running k-means on z={z_sub.shape} subset (3 random seeds)...', flush=True)
    codebooks = [model.C1, model.C2, model.C3]
    with torch.no_grad():
        for i, cb in enumerate(codebooks):
            km = KMeans(n_clusters=N_CLUSTERS, n_init=4, random_state=42 + i)
            km.fit(z_sub)
            cb.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
            used = (np.unique(km.labels_, return_counts=True)[1] > 0).sum()
            print(f'[init] C{i+1} initialized: {used}/{N_CLUSTERS} clusters used', flush=True)
    print('[init] codebooks initialized', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h_offset', type=int, default=0, help='H subspace start offset')
    ap.add_argument('--h_dim', type=int, default=32, help='H subspace dim (0=disable H)')
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    ap.add_argument('--log-every', type=int, default=LOG_EVERY)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--task-tag', type=str, default='task56_hoffset')
    args = ap.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    variant = f'hoff{args.h_offset:03d}_hdim{args.h_dim:03d}'
    task_tag = f'{args.task_tag}_{variant}'
    print(f'[task56] device={DEVICE}, h_offset={args.h_offset}, h_dim={args.h_dim}', flush=True)
    print(f'[task56] task_tag={task_tag}', flush=True)

    out_dir = f'{LOG_ROOT}/{task_tag}'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load concat embedding
    print(f'[task56] loading {CONCAT_EMB}', flush=True)
    full = torch.load(CONCAT_EMB, map_location='cpu', weights_only=False).float()
    print(f'[task56] full shape={tuple(full.shape)}', flush=True)

    # Z-score normalize (task58 path) — critical to prevent codebook collapse
    full_mean = full.mean(dim=0, keepdim=True)
    full_std = full.std(dim=0, keepdim=True).clamp(min=1e-6)
    full = (full - full_mean) / full_std
    print(f'[task56] z-score normalized: mean={full.mean():.4f}, std={full.std():.4f}', flush=True)

    # Build model
    model = HOffsetMixedCurvatureRQVAE(
        dim=INPUT_DIM,
        h_dim=args.h_dim,
        h_offset=args.h_offset,
    ).to(DEVICE)

    # Init codebooks with K-means on encoded z
    init_codebooks(model, full, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = full.shape[0]
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        x = full[idx].to(DEVICE)
        total, codes, recon, ent, smr = model(x)
        opt.zero_grad()
        total.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / elapsed
            print(f'[{task_tag}] step {step:5d}/{args.max_steps} | '
                  f'recon={recon.item():.4f} ent={ent.item():.2f} smr={smr.item():.4f} | '
                  f'{sps:.1f} step/s | elapsed {elapsed:.0f}s', flush=True)

    # Save ckpt
    ckpt_path = f'{out_dir}/ckpt_{variant}.ckpt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'h_offset': args.h_offset,
        'h_dim': args.h_dim,
        'variant': variant,
        'task_tag': task_tag,
    }, ckpt_path)
    print(f'[task56] saved ckpt to {ckpt_path}', flush=True)


if __name__ == '__main__':
    main()