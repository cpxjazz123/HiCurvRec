"""Task 388 v3: H-E-E-E training with proper reconstruction.

Insight: log_map0 maps ball → T_0 (tangent at origin, which is Euclidean).
But ||log_map0(q1)|| ≈ atanh(||q1||) << ||x||. So q1 cannot contribute meaningfully to recon.

Fix: Use DUAL codebooks:
- C1 (on ball): used for Poincaré distance-based assignment
- C1_euclid (in Euclidean space): used for reconstruction directly

This decouples geometry (assignment) from reconstruction (residual quantization).
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

GRID = '/home/wlia0047/wenyu/GeneRec'
GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.utils.decorators  # noqa: F401

# ========== Config ==========
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
LOG_ROOT = f'{GRID}/logs/train/runs'
TASK_TAG = 'task388_stage2_h_e_e_e_v3'

NUM_LAYERS = 3
N_CLUSTERS = 256
DIM = 2048
BALL_EPS = 1e-5
BALL_C = 1.0
MAX_NORM = 0.5

BATCH_SIZE = 2048
LR_EUC = 1e-3
LR_HYP = 1e-4
MAX_STEPS = 15000
LOG_EVERY = 500

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ========== Poincaré primitives (same as v2) ==========

def project_to_ball(x, max_norm=MAX_NORM, eps=BALL_EPS):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c=BALL_C):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return num / denom.clamp(min=1e-15)


def exp_map0(v, c=BALL_C):
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def log_map0(y, c=BALL_C):
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    sqrt_c = math.sqrt(c)
    arg = (sqrt_c * y_norm).clamp(max=1.0 - 5e-3)
    return torch.atanh(arg) * y / (sqrt_c * y_norm)


def poincare_distance(x, y, c=BALL_C):
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c)
    diff_norm = diff.norm(dim=-1).clamp(max=1.0 - 5e-3)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


def euclidean_to_poincare(x, c=BALL_C, target_norm=0.5):
    """Map Euclidean x to ball with target norm, preserving direction."""
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    v_norm_target = math.atanh(target_norm)
    v = x_unit * v_norm_target
    y = exp_map0(v, c=c)
    return project_to_ball(y, max_norm=MAX_NORM)


class RiemannianAdam(torch.optim.Optimizer):
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
                denom = exp_avg_sq.sqrt().add(eps)
                update = exp_avg / denom
                tangent_step = exp_map0(-lr * update / bias_correction1, c=BALL_C)
                p.data = project_to_ball(
                    mobius_add(p.data, tangent_step, c=BALL_C),
                    max_norm=MAX_NORM,
                )
        return loss


# ========== H-E-E-E RQ-VAE v3: DUAL codebook ==========

class HEeERQVAEv3(nn.Module):
    """L1 uses Poincaré geometry for assignment, Euclidean for reconstruction.

    C1 (ball):     codebook on Poincaré ball for distance-based assignment
    C1_euclid:     codebook in Euclidean space for reconstruction
    C2, C3 (Euclidean): standard RQ-VAE L2/L3
    """

    def __init__(self, dim=DIM, n_clusters=N_CLUSTERS, num_layers=NUM_LAYERS):
        super().__init__()
        self.dim = dim
        self.n_clusters = n_clusters
        self.num_layers = num_layers
        # L1 codebook (Poincaré ball) — for ASSIGNMENT
        self.register_parameter('C1_ball', nn.Parameter(
            torch.zeros(n_clusters, dim), requires_grad=True
        ))
        # L1 codebook (Euclidean) — for RECONSTRUCTION
        self.register_parameter('C1_euclid', nn.Parameter(
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
        """Init: Euclidean K-means on x_euclidean → C1_euclid; map → C1_ball."""
        device = x_euclidean.device
        with torch.no_grad():
            from sklearn.cluster import KMeans
            N = x_euclidean.shape[0]
            print(f'[init] Euclidean K-means on {N} points, k={self.n_clusters}...', flush=True)
            km = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            km.fit(x_euclidean.cpu().numpy())
            centroids = torch.tensor(km.cluster_centers_, dtype=x_euclidean.dtype, device=device)
            print(f'[init] K-means done. inertia={km.inertia_:.4f}', flush=True)

            # C1_euclid = raw K-means centroids (full magnitude)
            self.C1_euclid.data = centroids.clone()

            # C1_ball = map centroids to ball (for assignment via Poincaré distance)
            self.C1_ball.data = euclidean_to_poincare(
                centroids, c=BALL_C, target_norm=MAX_NORM * 0.5
            )
            print(f'[init] C1_ball: norm mean={self.C1_ball.norm(dim=-1).mean():.4f}, '
                  f'zero_frac={(self.C1_ball.abs().max(dim=-1).values < 1e-6).float().mean():.4f}', flush=True)
            print(f'[init] C1_euclid: norm mean={self.C1_euclid.norm(dim=-1).mean():.4f}, '
                  f'min={self.C1_euclid.norm(dim=-1).min():.4f}, max={self.C1_euclid.norm(dim=-1).max():.4f}', flush=True)

            # L2/L3: standard Euclidean K-means on residuals
            x_ball = euclidean_to_poincare(x_euclidean, target_norm=MAX_NORM * 0.5)
            dists_l1 = poincare_distance(x_ball.unsqueeze(1), self.C1_ball.unsqueeze(0), c=BALL_C)
            idx_l1 = dists_l1.argmin(dim=1)
            q1_e = self.C1_euclid[idx_l1]
            r1 = x_euclidean - q1_e
            km2 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            km2.fit(r1.cpu().numpy())
            self.C2.data = torch.tensor(km2.cluster_centers_, dtype=x_euclidean.dtype, device=device)

            dists_l2 = torch.cdist(r1, self.C2, p=2)
            idx_l2 = dists_l2.argmin(dim=1)
            q2 = self.C2[idx_l2]
            r2 = r1 - q2
            km3 = KMeans(n_clusters=self.n_clusters, n_init=3, random_state=42)
            km3.fit(r2.cpu().numpy())
            self.C3.data = torch.tensor(km3.cluster_centers_, dtype=x_euclidean.dtype, device=device)
            print(f'[init] C2: norm mean={self.C2.norm(dim=-1).mean():.4f}', flush=True)
            print(f'[init] C3: norm mean={self.C3.norm(dim=-1).mean():.4f}', flush=True)

    def forward(self, x_euclidean):
        """x_euclidean: (B, D) — already normalized."""
        x_orig = x_euclidean
        # L1: assignment via Poincaré distance, reconstruction via C1_euclid
        x_ball = euclidean_to_poincare(x_euclidean, target_norm=MAX_NORM * 0.5)
        d1 = poincare_distance(x_ball.unsqueeze(1), self.C1_ball.unsqueeze(0), c=BALL_C)
        idx1 = d1.argmin(dim=1)
        q1_e = self.C1_euclid[idx1]
        r1 = x_euclidean - q1_e
        # L2/L3: Euclidean
        d2 = (r1.unsqueeze(1) - self.C2.unsqueeze(0)).norm(dim=-1)
        idx2 = d2.argmin(dim=1)
        q2 = self.C2[idx2]
        r2 = r1 - q2
        d3 = (r2.unsqueeze(1) - self.C3.unsqueeze(0)).norm(dim=-1)
        idx3 = d3.argmin(dim=1)
        q3 = self.C3[idx3]
        x_hat = q1_e + q2 + q3
        indices = torch.stack([idx1, idx2, idx3], dim=1)
        # Losses
        recon_loss = F.mse_loss(x_hat, x_orig)
        commit_loss_l1 = F.mse_loss(x_euclidean, q1_e)
        commit_loss_l2 = F.mse_loss(r1, q2)
        commit_loss_l3 = F.mse_loss(r2, q3)
        return x_hat, indices, {
            'recon': recon_loss,
            'commit_l1': commit_loss_l1,
            'commit_l2': commit_loss_l2,
            'commit_l3': commit_loss_l3,
        }


def train():
    print(f'[task388 v3] device: {DEVICE}', flush=True)

    print(f'[task388 v3] Loading embeddings from {EMB_PATH}', flush=True)
    x_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    x_mean = x_all.mean(dim=0, keepdim=True)
    x_std = x_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_all = (x_all - x_mean) / x_std
    print(f'[task388 v3] x_all shape: {x_all.shape}, mean norm: {x_all.norm(dim=-1).mean():.4f}',
          flush=True)

    model = HEeERQVAEv3(dim=DIM, n_clusters=N_CLUSTERS, num_layers=NUM_LAYERS).to(DEVICE)

    print('[task388 v3] Initializing codebooks...', flush=True)
    t0 = time.time()
    init_subset = x_all[torch.randperm(len(x_all))[:5000]]
    model.init_codebooks(init_subset.to(DEVICE))
    print(f'[task388 v3] Codebook init done in {time.time()-t0:.1f}s', flush=True)

    # Two optimizers: Riemannian for C1_ball, Adam for C1_euclid + C2/C3
    opt_C1_ball = RiemannianAdam([model.C1_ball], lr=LR_HYP)
    opt_euc = torch.optim.Adam([model.C1_euclid, model.C2, model.C3], lr=LR_EUC)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = f'{LOG_ROOT}/{TASK_TAG}_{ts}'
    os.makedirs(f'{out_dir}/checkpoints', exist_ok=True)

    print(f'[task388 v3] Starting training: MAX_STEPS={MAX_STEPS}, BATCH_SIZE={BATCH_SIZE}, '
          f'LR_HYP={LR_HYP}, LR_EUC={LR_EUC}, MAX_NORM={MAX_NORM}', flush=True)
    losses_log = []
    t_start = time.time()
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, len(x_all), (BATCH_SIZE,))
        x_batch = x_all[idx].to(DEVICE)
        x_hat, codes, losses = model(x_batch)
        loss = losses['recon'] + 0.25 * (losses['commit_l1'] +
                                          losses['commit_l2'] +
                                          losses['commit_l3'])
        opt_C1_ball.zero_grad()
        opt_euc.zero_grad()
        loss.backward()
        opt_C1_ball.step()
        opt_euc.step()
        with torch.no_grad():
            model.C1_ball.data = project_to_ball(model.C1_ball.data, max_norm=MAX_NORM)
        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f'[task388 v3] step {step:6d}/{MAX_STEPS} '
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

    ckpt_path = f'{out_dir}/checkpoints/ckpt_H_E_E_E.ckpt'
    torch.save({
        'state_dict': {
            'quantization_layer_list.0.centroids': model.C1_euclid.detach().cpu(),
            'quantization_layer_list.1.centroids': model.C2.detach().cpu(),
            'quantization_layer_list.2.centroids': model.C3.detach().cpu(),
            'C1_ball': model.C1_ball.detach().cpu(),
        },
        'hyper_parameters': {
            'num_layers': NUM_LAYERS,
            'n_clusters': N_CLUSTERS,
            'dim': DIM,
            'l1_space': 'poincare_ball_dual',
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
    print(f'[task388 v3] Saved ckpt to {ckpt_path}', flush=True)

    import json
    losses_path = f'{out_dir}/losses.json'
    json.dump({
        'task': TASK_TAG,
        'method': 'h_e_e_e_train_v3_dual_codebook',
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
    print(f'[task388 v3] Saved losses to {losses_path}', flush=True)
    print(f'[task388 v3] Done. Total time: {time.time()-t_start:.1f}s', flush=True)


if __name__ == '__main__':
    train()