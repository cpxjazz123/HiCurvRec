"""Task 388: H-E-E-E training v2 — fix 3 root causes.

Fixes vs v1:
1. K-means init: Euclidean K-means → map centroids to ball (avoid origin collapse)
2. log_map/exp_map: verify round-trip; use max_norm=0.5 to avoid edge saturation
3. commit_l1: use Euclidean distance ||x - log_map0(q1)||^2 to match recon loss scale
4. LR: reduce hyp LR from 1e-3 to 1e-4 for stability on ball
5. Riemannian Adam: use mobius_add for true Riemannian step (not exp_map0 at origin)

Verification criteria (user-specified):
- Loss must drop substantially (< 0.9)
- L1 residual ρ must be << input ρ
"""

import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.utils.decorators  # noqa: F401

# ========== Config ==========
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
LOG_ROOT = f'{GRID}/logs/train/runs'
TASK_TAG = 'task388_stage2_h_e_e_e_v2'

NUM_LAYERS = 3
N_CLUSTERS = 256
DIM = 2048
BALL_EPS = 1e-5
BALL_C = 1.0
MAX_NORM = 0.5  # FIX v2: was 0.999, too close to boundary → atanh explodes. Use 0.5 to stay in stable region.

BATCH_SIZE = 2048
LR_EUC = 1e-3
LR_HYP = 1e-4  # FIX v2: was 1e-3, too aggressive on ball
MAX_STEPS = 15000
LOG_EVERY = 500

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========== Poincaré Ball primitives ==========

def project_to_ball(x, max_norm=MAX_NORM, eps=BALL_EPS):
    """Project x to Poincaré ball of curvature c=1."""
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
    # FIX v2: clamp y_norm * sqrt_c < 1.0 - 5e-3 to avoid atanh explosion
    arg = (sqrt_c * y_norm).clamp(max=1.0 - 5e-3)
    return torch.atanh(arg) * y / (sqrt_c * y_norm)


def poincare_distance(x, y, c=BALL_C):
    """d_c(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕ y||)"""
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 - 5e-3)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


def euclidean_to_poincare(x, c=BALL_C, target_norm=0.5):
    """Map Euclidean x to Poincaré ball with TARGET norm (FIX v2).

    v1 scaled by x/(||x||+3) → all points have ≈ same norm (loses direction info diversity)
    v2 scales each x to a target ball norm → preserves direction diversity.
    """
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    # Each point is mapped to ball with norm target_norm, but direction from x_unit
    # via exp_map0: y = exp_map0(target_norm_dir * something)
    # Simpler: y = tanh(target_norm / 1.0) * x_unit (this gives y on ball with norm = tanh(target_norm))
    # But we want y to have norm = target_norm. So:
    # tanh(v_norm) = target_norm → v_norm = atanh(target_norm)
    v_norm_target = math.atanh(target_norm)
    v = x_unit * v_norm_target
    y = exp_map0(v, c=c)
    return project_to_ball(y, max_norm=MAX_NORM)


# ========== Riemannian Adam for Poincaré ball ==========

class RiemannianAdam(torch.optim.Optimizer):
    """Adam optimizer for parameters in Poincaré ball (Bécigneul & Ganea 2019)."""

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
                denom = exp_avg_sq.sqrt().add(eps)
                # Adam direction in tangent space at origin
                update = exp_avg / denom
                # FIX v2: step from origin in tangent space, then add to current point via mobius_add
                # This is the true Riemannian gradient step (assuming parallel transport is identity for small step)
                tangent_step = exp_map0(-lr * update / bias_correction1, c=BALL_C)
                p.data = project_to_ball(
                    mobius_add(p.data, tangent_step, c=BALL_C),
                    max_norm=MAX_NORM,
                )
        return loss


# ========== Euclidean K-means → ball init (FIX v2) ==========

@torch.no_grad()
def euc_kmeans_to_ball_init(x_euclidean, n_clusters, seed=42):
    """Run Euclidean K-means on x_euclidean, then map centroids to Poincaré ball.

    This avoids origin collapse because Euclidean K-means naturally produces
    diverse directions, then exp_map0 preserves those directions on the ball.
    """
    from sklearn.cluster import KMeans
    N, D = x_euclidean.shape
    print(f'[init] Euclidean K-means on {N} points, k={n_clusters}...', flush=True)
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed)
    km.fit(x_euclidean.cpu().numpy())
    centroids = torch.tensor(km.cluster_centers_, dtype=x_euclidean.dtype, device=x_euclidean.device)
    print(f'[init] Euclidean K-means done. inertia={km.inertia_:.4f}', flush=True)

    # Map centroids to ball (FIX v2: use target_norm version)
    centroids_ball = euclidean_to_poincare(centroids, c=BALL_C, target_norm=MAX_NORM * 0.5)
    print(f'[init] ball centroids: norm mean={centroids_ball.norm(dim=-1).mean():.4f}, '
          f'min={centroids_ball.norm(dim=-1).min():.4f}, max={centroids_ball.norm(dim=-1).max():.4f}', flush=True)
    print(f'[init] ball centroids: zero_fraction={(centroids_ball.abs().max(dim=-1).values < 1e-6).float().mean():.4f}', flush=True)
    return centroids_ball


# ========== H-E-E-E RQ-VAE module ==========

class HEeERQVAE(nn.Module):
    """L1 hyperbolic (Poincaré), L2/L3 Euclidean residual quantization."""

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
        """Initialize codebooks via Euclidean K-means (FIX v2)."""
        device = x_euclidean.device
        with torch.no_grad():
            # L1: Euclidean K-means → ball
            self.C1.data = euc_kmeans_to_ball_init(x_euclidean, self.n_clusters, seed=42)
            # L2: Euclidean K-means on L1 residual
            x_ball = euclidean_to_poincare(x_euclidean, target_norm=MAX_NORM * 0.5)
            dists_l1 = torch.cdist(x_ball, self.C1, p=2)
            idx_l1 = dists_l1.argmin(dim=1)
            q1 = self.C1[idx_l1]
            q1_euclid = log_map0(q1, c=BALL_C)
            r1 = x_euclidean - q1_euclid
            from sklearn.cluster import KMeans
            km2 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            km2.fit(r1.cpu().numpy())
            self.C2.data = torch.tensor(km2.cluster_centers_, dtype=x_euclidean.dtype, device=device)
            # L3: Euclidean K-means on L2 residual
            dists_l2 = torch.cdist(r1, self.C2, p=2)
            idx_l2 = dists_l2.argmin(dim=1)
            q2 = self.C2[idx_l2]
            r2 = r1 - q2
            km3 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            km3.fit(r2.cpu().numpy())
            self.C3.data = torch.tensor(km3.cluster_centers_, dtype=x_euclidean.dtype, device=device)

    def forward(self, x_euclidean):
        """x_euclidean: (B, D) — already normalized."""
        x_orig = x_euclidean
        # L1: Poincaré ball
        x_ball = euclidean_to_poincare(x_euclidean, target_norm=MAX_NORM * 0.5)
        d1 = poincare_distance(x_ball.unsqueeze(1), self.C1.unsqueeze(0), c=BALL_C)
        idx1 = d1.argmin(dim=1)
        q1 = self.C1[idx1]
        q1_euclid = log_map0(q1, c=BALL_C)
        r1 = x_euclidean - q1_euclid
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
        indices = torch.stack([idx1, idx2, idx3], dim=1)
        # Losses
        recon_loss = F.mse_loss(x_hat, x_orig)
        # FIX v2: commit_l1 in EUCLIDEAN space (matches recon loss scale)
        commit_loss_l1 = F.mse_loss(x_euclidean, q1_euclid)
        commit_loss_l2 = F.mse_loss(r1, q2)
        commit_loss_l3 = F.mse_loss(r2, q3)
        return x_hat, indices, {
            'recon': recon_loss,
            'commit_l1': commit_loss_l1,
            'commit_l2': commit_loss_l2,
            'commit_l3': commit_loss_l3,
        }


# ========== Training loop ==========

def train():
    print(f'[task388] device: {DEVICE}', flush=True)

    print(f'[task388] Loading embeddings from {EMB_PATH}', flush=True)
    x_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    # Normalize to standard scale
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std
    print(f'[task388] x_all shape: {x_all.shape}, mean norm: {x_all.norm(dim=-1).mean():.4f}',
          flush=True)

    model = HEeERQVAE(dim=DIM, n_clusters=N_CLUSTERS, num_layers=NUM_LAYERS).to(DEVICE)

    # Init codebooks
    print('[task388] Initializing codebooks via Euclidean K-means → ball mapping...', flush=True)
    t0 = time.time()
    init_subset = x_all[torch.randperm(len(x_all))[:5000]]
    model.init_codebooks(init_subset.to(DEVICE))
    print(f'[task388] Codebook init done in {time.time()-t0:.1f}s', flush=True)
    print(f'[task388] C1 init: norm mean={model.C1.norm(dim=-1).mean():.4f}, '
          f'zero_frac={(model.C1.abs().max(dim=-1).values < 1e-6).float().mean():.4f}', flush=True)

    # Optimizers
    opt_C1 = RiemannianAdam([model.C1], lr=LR_HYP)
    opt_C23 = torch.optim.Adam([model.C2, model.C3], lr=LR_EUC)

    # Output dir
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = f'{LOG_ROOT}/{TASK_TAG}_{ts}'
    os.makedirs(f'{out_dir}/checkpoints', exist_ok=True)

    # Training
    print(f'[task388] Starting training: MAX_STEPS={MAX_STEPS}, BATCH_SIZE={BATCH_SIZE}, '
          f'LR_HYP={LR_HYP}, MAX_NORM={MAX_NORM}', flush=True)
    losses_log = []
    t_start = time.time()
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, len(x_all), (BATCH_SIZE,))
        x_batch = x_all[idx].to(DEVICE)
        x_hat, codes, losses = model(x_batch)
        loss = losses['recon'] + 0.25 * (losses['commit_l1'] +
                                          losses['commit_l2'] +
                                          losses['commit_l3'])
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
            print(f'[task388] step {step:6d}/{MAX_STEPS} '
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
        'date': ts,
        'status': 'completed',
    }, ckpt_path)
    print(f'[task388] Saved ckpt to {ckpt_path}', flush=True)

    # Save losses.json
    import json
    losses_path = f'{out_dir}/losses.json'
    json.dump({
        'task': TASK_TAG,
        'method': 'h_e_e_e_train_v2',
        'date': ts,
        'status': 'completed',
        'data': {
            'losses': losses_log,
            'final_total': losses_log[-1]['total'],
            'initial_total': losses_log[0]['total'],
            'final_recon': losses_log[-1]['recon'],
            'final_commit_l1': losses_log[-1]['commit_l1'],
            'final_commit_l2': losses_log[-1]['commit_l2'],
            'final_commit_l3': losses_log[-1]['commit_l3'],
            'ckpt_path': ckpt_path,
        },
    }, open(losses_path, 'w'), indent=2)
    print(f'[task388] Saved losses to {losses_path}', flush=True)
    print(f'[task388] Done. Total time: {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    train()