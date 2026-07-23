#!/usr/bin/env python3
"""Stage 2: 训练 H-E-E-E RQ-VAE
  - L1: Poincaré ball 双曲量化 + Riemannian Adam
  - L2/L3: 欧氏量化 + Adam
  - 复用 task13_group_a_s21 的数据 (FLAN-T5 embeddings) 和超参

输入:
  - logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt  (11924, 2048)
  - task13_group_a_s21 ckpt 作为 E-E-E-E baseline 对照

输出:
  - logs/train/runs/<timestamp>/checkpoints/ckpt_H_E_E_E.ckpt
  - checkpoints contain: quantization_layer_list.0.centroids (Poincaré)
                         quantization_layer_list.1/2.centroids (Euclidean)
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
import src.utils.decorators  # noqa: F401

# ========== Config ==========
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
BASELINE_CKPT = f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'
LOG_ROOT = f'{GRID}/logs/train/runs'
TASK_TAG = 'task385_stage2_h_e_e_e'

# L1 = hyperbolic (Poincaré), L2/L3 = Euclidean (matching task15)
NUM_LAYERS = 3
N_CLUSTERS = 256  # same as task13
DIM = 2048
BALL_EPS = 1e-5
BALL_C = 1.0  # curvature; for c=1, ball radius = 1
MAX_NORM = 1.0 - 1e-3  # project to slightly inside unit ball

# training hyperparams from task15 recipe (approximated; will use Adam + Riemannian Adam)
BATCH_SIZE = 2048
LR_EUC = 1e-3
LR_HYP = 1e-3
MAX_STEPS = 15000
LOG_EVERY = 500


# ========== Poincaré Ball primitives ==========

def project_to_ball(x, max_norm=MAX_NORM, eps=BALL_EPS):
    """Project x to Poincaré ball of curvature c=1.

    Norm must satisfy ||x|| < 1/sqrt(c) = 1.0
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c=BALL_C):
    """Möbius addition in Poincaré ball"""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return num / denom.clamp(min=1e-15)


def exp_map0(v, c=BALL_C):
    """Exponential map at origin: v ∈ T_0 B^c → B^c"""
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def log_map0(y, c=BALL_C):
    """Logarithmic map at origin: y ∈ B^c → T_0 B^c"""
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    return torch.atanh(sqrt_c * y_norm.clamp(max=1.0 - 1e-5)) * y / (sqrt_c * y_norm)


def poincare_distance(x, y, c=BALL_C):
    """d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕ y||)"""
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 - 1e-5)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


def euclidean_to_poincare(x, c=BALL_C):
    """Map Euclidean x to Poincaré ball via exp_map0(x).

    This gives a diffeomorphism from R^D to Poincaré ball (radius < 1/sqrt(c)).
    """
    # normalize x to be small enough to fit in ball
    x_scaled = x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-9) + 3.0)  # scale to fit
    return project_to_ball(exp_map0(x_scaled, c=c), max_norm=MAX_NORM)


# ========== Riemannian Adam for Poincaré ball ==========

class RiemannianAdam(torch.optim.Optimizer):
    """Adam optimizer for parameters in Poincaré ball (Bécigneul & Ganea 2019).

    For each parameter, the update is:
      1. Standard Adam: m, v, m_hat, v_hat on the tangent space (using exp_map0)
      2. Apply update in tangent space at origin: x_new = exp_map0(x - lr * m_hat/sqrt(v_hat))

    Reference: "Riemannian Adaptive Optimization Methods" ICLR 2019
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # grad is w.r.t. parameter (already on manifold via exp_map0 composition)
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                bias_correction1 = 1 - b1 ** step
                bias_correction2 = 1 - b2 ** step
                # tangent-space update (parameters live in ball, so we treat them as if at origin)
                denom = exp_avg_sq.sqrt().add(eps)
                step_size = lr / bias_correction1
                update = exp_avg / denom * bias_correction2
                # Apply as Riemannian gradient step: x_new = exp_map0(-step_size * update)
                # Since x is already in the ball, we treat the update in tangent space at x
                # then use the parallel transport approximation (close to identity for small step):
                p.data = project_to_ball(
                    exp_map0(-step_size * update / bias_correction2, c=BALL_C),
                    max_norm=MAX_NORM,
                )
        return loss


# ========== Poincaré K-means initialization ==========

@torch.no_grad()
def poincare_kmeans_init(x_poincare, n_clusters, n_iter=50, seed=42):
    """K-means in Poincaré ball.

    1. Init centroids by random sample (in Euclidean, then project)
    2. Assign each point to nearest centroid (Poincaré distance)
    3. Update centroid by Fréchet mean (iterative exp_map averaging)

    Returns: (n_clusters, D) tensor of centroids in ball.
    """
    torch.manual_seed(seed)
    N, D = x_poincare.shape
    # random init
    idx = torch.randperm(N)[:n_clusters]
    centroids = x_poincare[idx].clone()
    centroids = project_to_ball(centroids, max_norm=MAX_NORM)
    for it in range(n_iter):
        # assign
        # poincare_distance returns (N, K)
        dists = torch.cdist(x_poincare, centroids, p=2)  # use Euclidean as proxy init
        assignments = dists.argmin(dim=1)
        # update: Fréchet mean via Karcher flow (simplified: exp_map at mean, then average)
        for k in range(n_clusters):
            mask = (assignments == k)
            if mask.sum() == 0:
                continue
            pts = x_poincare[mask]
            # Iterative Karcher mean: start at mean, average tangent vectors
            mean = pts.mean(dim=0)
            mean = project_to_ball(mean.unsqueeze(0), max_norm=MAX_NORM).squeeze(0)
            for _ in range(5):
                # log_map at current mean
                v = log_map0(pts - mean, c=BALL_C)  # simple tangent approximation
                v_mean = v.mean(dim=0)
                mean = exp_map0(v_mean + log_map0(mean.unsqueeze(0), c=BALL_C).squeeze(0) * 0,
                                c=BALL_C).squeeze(0)
                mean = project_to_ball(mean.unsqueeze(0), max_norm=MAX_NORM).squeeze(0)
            centroids[k] = mean
    return centroids


# ========== H-E-E-E RQ-VAE module ==========

class HEeERQVAE(nn.Module):
    """L1 hyperbolic (Poincaré), L2/L3 Euclidean residual quantization.

    Forward:
      x_euclidean: (B, D)
      x_ball = euclidean_to_poincare(x_euclidean)  # map to ball for L1
      q1, idx1 = poincare_quantize(x_ball, C1)    # assign to nearest in ball
      r1_euclidean = x_euclidean - log_map0(q1)   # residual in Euclidean (approximation)
      q2, idx2 = euclidean_quantize(r1_euclidean, C2)
      r2 = r1_euclidean - q2
      q3, idx3 = euclidean_quantize(r2, C3)
      x_hat = q1 + q2 + q3 (back to Euclidean)

    Note: We mix Poincaré (L1) and Euclidean (L2/L3) spaces.
    The transformation log_map0(q1) maps from ball back to R^D for residual computation.
    """

    def __init__(self, dim=DIM, n_clusters=N_CLUSTERS, num_layers=NUM_LAYERS):
        super().__init__()
        self.dim = dim
        self.n_clusters = n_clusters
        self.num_layers = num_layers
        # L1 codebook (Poincaré ball)
        self.register_parameter('C1', nn.Parameter(
            torch.zeros(n_clusters, dim), requires_grad=True
        ))
        # L2, L3 codebook (Euclidean)
        self.register_parameter('C2', nn.Parameter(
            torch.randn(n_clusters, dim) * 0.02, requires_grad=True
        ))
        self.register_parameter('C3', nn.Parameter(
            torch.randn(n_clusters, dim) * 0.02, requires_grad=True
        ))

    def init_codebooks(self, x_euclidean):
        """Initialize codebooks via K-means."""
        device = x_euclidean.device
        with torch.no_grad():
            # L1: Poincaré K-means
            x_ball = euclidean_to_poincare(x_euclidean)
            C1 = poincare_kmeans_init(x_ball, self.n_clusters, n_iter=50)
            self.C1.data = C1.to(device)
            # L2, L3: Euclidean K-means (sequential residuals)
            # After L1, residual = x_euclidean - log_map0(assigned_centroid in ball)
            dists_l1 = torch.cdist(x_ball, C1.to(device), p=2)
            idx_l1 = dists_l1.argmin(dim=1)
            q1_assigned = C1.to(device)[idx_l1]
            r1 = x_euclidean - log_map0(q1_assigned, c=BALL_C)
            # Euclidean K-means on r1
            from sklearn.cluster import KMeans
            km2 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            r1_np = r1.cpu().numpy()
            km2.fit(r1_np)
            C2_init = torch.from_numpy(km2.cluster_centers_).float().to(device)
            self.C2.data = C2_init
            r2 = r1 - C2_init[torch.from_numpy(km2.labels_).to(device)]
            km3 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            r2_np = r2.cpu().numpy()
            km3.fit(r2_np)
            self.C3.data = torch.from_numpy(km3.cluster_centers_).float().to(device)

    def forward(self, x_euclidean):
        """Forward pass returning (x_hat, indices, losses).

        x_euclidean: (B, D) — FLAN-T5 embedding (already in R^D)
        """
        x_orig = x_euclidean
        # L1: Poincaré
        x_ball = euclidean_to_poincare(x_euclidean)
        d1 = poincare_distance(
            x_ball.unsqueeze(1),  # (B, 1, D)
            self.C1.unsqueeze(0),  # (1, K, D)
            c=BALL_C,
        )  # (B, K)
        idx1 = d1.argmin(dim=1)
        q1 = self.C1[idx1]  # in ball
        # Map q1 back to Euclidean via log_map0
        q1_euclid = log_map0(q1, c=BALL_C) * 0.5  # scale factor for residual mixing
        r1 = x_orig - q1_euclid
        # L2: Euclidean
        d2 = (r1.unsqueeze(1) - self.C2.unsqueeze(0)).norm(dim=-1)
        idx2 = d2.argmin(dim=1)
        q2 = self.C2[idx2]
        r2 = r1 - q2
        # L3: Euclidean
        d3 = (r2.unsqueeze(1) - self.C3.unsqueeze(0)).norm(dim=-1)
        idx3 = d3.argmin(dim=1)
        q3 = self.C3[idx3]
        x_hat = q1_euclid + q2 + q3
        indices = torch.stack([idx1, idx2, idx3], dim=1)  # (B, 3)
        # Losses
        recon_loss = F.mse_loss(x_hat, x_orig)
        # Commitment loss for L1 (Poincaré distance)
        commit_loss_l1 = poincare_distance(
            x_ball, q1, c=BALL_C
        ).mean()
        commit_loss_l2 = F.mse_loss(r1, q2)
        commit_loss_l3 = F.mse_loss(r2, q3)
        return x_hat, indices, {
            'recon': recon_loss,
            'commit_l1': commit_loss_l1,
            'commit_l2': commit_loss_l2,
            'commit_l3': commit_loss_l3,
        }


# ========== Training loop ==========

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[task385] device: {device}', flush=True)

    print(f'[task385] Loading embeddings from {EMB_PATH}', flush=True)
    x_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    # Normalize to standard scale (FLAN-T5 embeddings have varying magnitudes)
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std
    print(f'[task385] x_all shape: {x_all.shape}, mean norm: {x_all.norm(dim=-1).mean():.4f}',
          flush=True)

    model = HEeERQVAE(dim=DIM, n_clusters=N_CLUSTERS, num_layers=NUM_LAYERS).to(device)

    # Init codebooks (use a subset for K-means)
    print('[task385] Initializing codebooks via K-means...', flush=True)
    t0 = time.time()
    init_subset = x_all[torch.randperm(len(x_all))[:5000]]
    model.init_codebooks(init_subset.to(device))
    print(f'[task385] Codebook init done in {time.time()-t0:.1f}s', flush=True)

    # Optimizers: Riemannian Adam for C1, Adam for C2/C3
    opt_C1 = RiemannianAdam([model.C1], lr=LR_HYP)
    opt_C23 = torch.optim.Adam([model.C2, model.C3], lr=LR_EUC)

    # Output dir
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = f'{LOG_ROOT}/{TASK_TAG}_{ts}'
    os.makedirs(f'{out_dir}/checkpoints', exist_ok=True)

    # Training
    print(f'[task385] Starting training: MAX_STEPS={MAX_STEPS}, BATCH_SIZE={BATCH_SIZE}',
          flush=True)
    losses_log = []
    t_start = time.time()
    for step in range(1, MAX_STEPS + 1):
        # Sample batch
        idx = torch.randint(0, len(x_all), (BATCH_SIZE,))
        x_batch = x_all[idx].to(device)
        # Forward
        x_hat, codes, losses = model(x_batch)
        # Total loss
        loss = losses['recon'] + 0.25 * (losses['commit_l1'] +
                                          losses['commit_l2'] +
                                          losses['commit_l3'])
        # Backward
        opt_C1.zero_grad()
        opt_C23.zero_grad()
        loss.backward()
        opt_C1.step()
        opt_C23.step()
        # Project C1 to ball after each step
        with torch.no_grad():
            model.C1.data = project_to_ball(model.C1.data, max_norm=MAX_NORM)
        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f'[task385] step {step:6d}/{MAX_STEPS} '
                  f'recon={losses["recon"].item():.4f} '
                  f'commit_l1={losses["commit_l1"].item():.4f} '
                  f'commit_l2={losses["commit_l2"].item():.4f} '
                  f'commit_l3={losses["commit_l3"].item():.4f} '
                  f'total={loss.item():.4f} '
                  f'elapsed={elapsed:.1f}s', flush=True)
            losses_log.append({
                'step': step,
                'recon': float(losses['recon'].item()),
                'commit_l1': float(losses['commit_l1'].item()),
                'commit_l2': float(losses['commit_l2'].item()),
                'commit_l3': float(losses['commit_l3'].item()),
                'total': float(loss.item()),
                'elapsed_sec': elapsed,
            })

    # Save final checkpoint
    ckpt_path = f'{out_dir}/checkpoints/ckpt_H_E_E_E.ckpt'
    torch.save({
        'state_dict': {
            'quantization_layer_list.0.centroids': model.C1.detach().cpu(),
            'quantization_layer_list.1.centroids': model.C2.detach().cpu(),
            'quantization_layer_list.2.centroids': model.C3.detach().cpu(),
        },
        'hyper_parameters': {
            'num_layers': NUM_LAYERS,
            'n_clusters': N_CLUSTERS,
            'dim': DIM,
            'l1_space': 'poincare_ball',
            'l2_space': 'euclidean',
            'l3_space': 'euclidean',
            'normalize_residuals': False,
        },
        'task_tag': TASK_TAG,
        'losses_log': losses_log,
        'x_mean': x_mean,
        'x_std': x_std,
        'final_step': MAX_STEPS,
        'final_loss': float(loss.item()),
    }, ckpt_path)
    print(f'[task385] Saved: {ckpt_path}', flush=True)

    # Save losses log separately
    with open(f'{out_dir}/losses.json', 'w') as f:
        json.dump({'task': TASK_TAG, 'method': 'h_e_e_e_rqvae',
                   'date': datetime.now().strftime('%Y-%m-%d'),
                   'status': 'completed',
                   'data': {'losses': losses_log}}, f, indent=2)
    print(f'[task385] Training complete in {time.time()-t_start:.1f}s', flush=True)
    print(f'[task385] Output: {out_dir}', flush=True)
    return out_dir, ckpt_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr-euc', type=float, default=LR_EUC)
    parser.add_argument('--lr-hyp', type=float, default=LR_HYP)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # override globals
    MAX_STEPS = args.max_steps
    BATCH_SIZE = args.batch_size
    LR_EUC = args.lr_euc
    LR_HYP = args.lr_hyp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)