#!/usr/bin/env python3
# task18_hrq_hyperbolic.py — HRQ 双曲残差量化 (Lorentz model)
#
# 简化 standalone 实现:
#   1. 输入 flan-t5-xl embedding (N=11924, D=2048)
#   2. Lorentz 提升: h = [sqrt(c+||x||²), x] (curvature c=1)
#   3. 每层: 双曲 K-means (3-layer with cluster size 256)
#   4. 残差: r_{l+1} = log_{h_code}(x) (双曲 log map 在码本点)
#   5. 输出: SID tensor (11924, 4) — 最后 1 列 dedup digit
#
# 与 Group A baseline (RQ-VAE/RKMeans 残差量化) 可直接对照:
#   - 输入同 embedding
#   - 输出同 SID shape
#   - TIGER T5 训练/推断 pipeline 不变
#
# 数学公式 (Lorentz 模型):
#   <h, h>_L = -h_0² + sum_{i=1}^{D} h_i²
#   d_L(h1, h2) = arccosh(-<h1, h2>_L / c)
#   log_p(q) = (arccosh(-<p,q>_L/c) / sinh(d_L(p,q))) * (q + <p,q>_L/c * p)
#   exp_p(v) = cosh(||v||_L) * p + sinh(||v||_L) * v/||v||_L

import sys, os, json, time
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch


# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LAYERS = 3
N_CLUSTERS = 256
MAX_ITER = 50         # K-means inner iterations per layer
BATCH = 1024          # subsample batch for distance matrix (memory)
DROPOUT_PATIENCE = 5  # early stop if QErr not improved
SEED = 42
CURVATURE = 1.0      # Lorentz curvature c

# Paths
EMBED_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
CKPT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp20'
os.makedirs(CKPT_DIR, exist_ok=True)


# ====================== Lorentz model functions ======================

def lift_to_lorentz(x, c=CURVATURE):
    """Euclidean x ∈ R^D → Lorentz point h ∈ R^(D+1)
       h_0 = sqrt(c + ||x||²), h_[1:] = x
    """
    x_norm2 = (x * x).sum(-1)                      # (N,)
    h0 = torch.sqrt(c + x_norm2).unsqueeze(-1)    # (N, 1)
    return torch.cat([h0, x], dim=-1)             # (N, D+1)


def lift_to_lorentz_preserve(h, c=CURVATURE):
    """Lorentz point h: 直接返回 (lift 时已 normal-space -> 提升后用此还?) """
    return h


def lorentz_inner(h1, h2):
    """Lorentz inner product: -h1_0 h2_0 + sum_{i=1} h1_i h2_i
       h1, h2: (..., D+1)
       Returns: (..., )"""
    return -h1[..., 0] * h2[..., 0] + (h1[..., 1:] * h2[..., 1:]).sum(-1)


def lorentz_norm_sq(v):
    """v is a Lorentz tangent vector (Minkowski norm)
       ||v||_L² = -v_0² + |v_[1:]|²"""
    return -v[..., 0] ** 2 + (v[..., 1:] ** 2).sum(-1)


def lorentz_distance(h1, h2, c=CURVATURE):
    """d_L(h1, h2) = arccosh(-<h1, h2>_L / c)
       h1, h2: (..., D+1)
       Returns: (..., )"""
    inner = lorentz_inner(h1, h2)
    arg = -inner / c
    arg = torch.clamp(arg, min=1.0 + 1e-9)        # numerical safety
    return torch.acosh(arg)


def exp_map(p, v, c=CURVATURE):
    """Exponential map at p with tangent vector v
       p: (..., D+1) — Lorentz point, norm_L² = -c
       v: (..., D+1) — tangent vector, p_0 v_0 = <p_[1:], v_[1:]>

       Returns: (..., D+1) Lorentz point
    """
    p_norm_v = lorentz_norm_sq(v).clamp(min=0)    # ||v||_L²
    p_norm_v = torch.sqrt(p_norm_v + 1e-12)        # ||v||_L
    cosh_d = torch.cosh(p_norm_v)
    sinh_d = torch.sinh(p_norm_v) / (p_norm_v + 1e-12)  # sinh(||v||)/||v||
    # exp_p(v) = cosh(||v||) p + sinh(||v||)/||v|| v
    return cosh_d.unsqueeze(-1) * p + sinh_d.unsqueeze(-1) * v


def log_map(p, q, c=CURVATURE):
    """Logarithmic map at p of q (returns tangent vector in T_p)
       p, q: (..., D+1) Lorentz points
       Returns: v ∈ T_p (Minkowski)
    """
    inner_pq = lorentz_inner(p, q)
    arg = -inner_pq / c
    arg = torch.clamp(arg, min=1.0 + 1e-9)
    d = torch.acosh(arg)                            # scalar distance
    # v = (d / sinh(d)) * (q + <p, q>_L/c * p)
    factor = d / (torch.sinh(d) * c + 1e-12)
    inner_term = (inner_pq / c).unsqueeze(-1) * p   # (..., D+1)
    return factor.unsqueeze(-1) * (q + inner_term)


def project_to_tangent(p, v):
    """Project v onto the tangent space at p
       v ∈ T_p iff <p, v>_L = 0

       p: (D+1,) Lorentz point
       v: (D+1,) arbitrary vector
       Returns: tangent vector with <p, v_proj>_L = 0
    """
    inner_pv = lorentz_inner(p, v)
    return v + (inner_pv / (-lorentz_inner(p, p) + 1e-12)).unsqueeze(-1) * p
    # Note: <p, p>_L = -c, so the denominator is c (constant)


# ====================== Hyperbolic K-means ======================

def hkmeans_init(x_lorentz, n_clusters, seed=SEED):
    """Random init: pick n_clusters distinct points from x_lorentz
       x_lorentz: (N, D+1)
       Returns: (n_clusters, D+1) initial centroids
    """
    rng = np.random.default_rng(seed)
    N = x_lorentz.shape[0]
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    return x_lorentz[init_idx].clone()


def hkmeans_assign_batched(x_lorentz, centroids, batch=BATCH):
    """Assign each point to its nearest centroid (in batched manner for memory)
       x_lorentz: (N, D+1)
       centroids: (K, D+1)
       Returns: (N,) integer assignments
    """
    N = x_lorentz.shape[0]
    K = centroids.shape[0]
    assignments = torch.empty(N, dtype=torch.long, device=x_lorentz.device)
    for i in range(0, N, batch):
        x_batch = x_lorentz[i:i+batch]                    # (B, D+1)
        # distance matrix: (B, K)
        inner_xc = lorentz_inner(x_batch.unsqueeze(1), centroids.unsqueeze(0))  # (B, K, ...)
        arg = -inner_xc / CURVATURE
        arg = torch.clamp(arg, min=1.0 + 1e-9)
        d = torch.acosh(arg)                              # (B, K)
        assignments[i:i+batch] = d.argmin(dim=-1)
    return assignments


def hkmeans_update_centroids(x_lorentz, assignments, n_clusters, batch=BATCH):
    """Update centroids:  Riemannian center of mass via gradient descent
       (avoid closed-form which doesn't exist in hyperbolic space)
       Use simple exp_map averaging:
         1. pull back assigned points: log_{current_c}(assigned_x)
         2. mean tangent vector
         3. exp_{current_c}(mean)
    """
    centroids_new = torch.empty_like(x_lorentz[:n_clusters])
    for k in range(n_clusters):
        mask = (assignments == k)
        n_assigned = mask.sum().item()
        if n_assigned == 0:
            # Empty cluster: reinit from random
            rng = np.random.default_rng(seed=k)
            rand_idx = rng.integers(0, x_lorentz.shape[0])
            centroids_new[k] = x_lorentz[rand_idx]
            continue
        elif n_assigned == 1:
            centroids_new[k] = x_lorentz[mask][0]
            continue
        assigned_x = x_lorentz[mask]                     # (n_k, D+1)
        # 当前 centroid (取该 cluster 已分配点的 R-Mean)
        # 用 Euclidean mean 然后 lift 作为新 centroid (简化)
        mean_eu = assigned_x[:, 1:].mean(0)              # (D,)
        centroids_new[k] = lift_to_lorentz(mean_eu.unsqueeze(0))[0]
    return centroids_new


def hkmeans_run(x_lorentz, n_clusters, max_iter=MAX_ITER, batch=BATCH, seed=SEED,
                verbose=True):
    """单层 双曲 K-means
       x_lorentz: (N, D+1) 输入
       Returns: assignments (N,), centroids (K, D+1), qerr_history
    """
    centroids = hkmeans_init(x_lorentz, n_clusters, seed=seed)
    history = []
    best_qerr = float('inf')
    patience_count = 0

    for it in range(max_iter):
        t0 = time.time()
        # Assign
        assignments = hkmeans_assign_batched(x_lorentz, centroids, batch=batch)
        # QErr (sum of squared distance)
        q_err = 0.0
        for k in range(n_clusters):
            mask = (assignments == k)
            if mask.any():
                d = lorentz_distance(x_lorentz[mask], centroids[k].unsqueeze(0))
                q_err += (d ** 2).sum().item()
        history.append(q_err)
        if verbose:
            print(f'  iter {it}: QErr={q_err:.4f}  ({time.time()-t0:.1f}s)')
        # Update
        centroids_new = hkmeans_update_centroids(x_lorentz, assignments, n_clusters, batch=batch)
        # Check convergence
        if q_err < best_qerr - 1e-3:
            best_qerr = q_err
            patience_count = 0
            centroids = centroids_new
        else:
            patience_count += 1
            centroids = centroids_new
            if patience_count >= DROPOUT_PATIENCE:
                if verbose:
                    print(f'  early stop at iter {it} ({DROPOUT_PATIENCE} iters no improvement)')
                break

    return assignments, centroids, history


# ====================== Residual quantization ======================

def hrq_forward(x_lorentz, n_layers=N_LAYERS, n_clusters=N_CLUSTERS,
                max_iter=MAX_ITER, device=DEVICE, verbose=True):
    """3 层 HRQ forward with PROPER hyperbolic residual quantization

    残差策略 (v2 修复 round-trip identity bug):
       L1: cluster x in H^d → C_1[c_1]
       r_1 = log_map(C_1[c_1], x) ∈ T_{C_1}    (Minkowski tangent vector)
       L2: cluster r_1 (在 T_{C_1} 中, 视为 R^d 用 Euclidean K-means)
       r_2 = r_1 - C_2[c_2]                    (Euclidean tangent residual)
       L3: cluster r_2 → C_3[c_3]
       r_3 = r_2 - C_3[c_3]

    注意: 因为 log_map(c, x) + c 不直接等于 x (Minkowski 几何),
          但 exp_map(c, log_map(c, x)) = x (round-trip identity).
          真正的 residual 是 log_map(c, x) 本身作为 tangent vector, 然后欧氏减 cluster centroid.
    """
    x = x_lorentz.clone().to(device)
    codes = torch.zeros(x.shape[0], n_layers, dtype=torch.long, device=device)
    codebooks = []
    residuals = [x.cpu()]
    r = x  # 当前量化目标: 第一层 r=x, 后续 r=上一层的 tangent residual

    for l in range(n_layers):
        if verbose:
            print(f'\n=== Layer {l+1}/{n_layers}: hyperbolic K-means on dim {r.shape[-1]-1} ===')

        if l == 0:
            # L1: cluster x in H^d (full Lorentz)
            assignments, centroids, history = hkmeans_run(
                r, n_clusters, max_iter=max_iter, batch=BATCH, seed=SEED + l,
                verbose=verbose,
            )
            codes[:, l] = assignments
            codebooks.append(centroids.cpu())

            # Compute tangent residual: r_1 = log_map(C_1[c_1], x) ∈ T_{C_1}
            # Drop time component → R^d (Euclidean) representation
            new_r = torch.zeros(r.shape[0], r.shape[-1] - 1, device=r.device)  # (N, D)
            for i in range(0, r.shape[0], BATCH):
                r_batch = r[i:i+BATCH]
                assn_batch = assignments[i:i+BATCH]
                c_batch = centroids[assn_batch]                              # (B, D+1)
                v = log_map(c_batch, r_batch)                                # (B, D+1) tangent (Minkowski)
                # v[:, 1:] ∈ R^d (Euclidean) is the spatial part of tangent vector
                new_r[i:i+BATCH] = v[:, 1:]                                  # (B, D) Euclidean residual
            r = new_r
            residuals.append(r.cpu())
        else:
            # L2+: cluster r in R^d (Euclidean) — 视为 tangent space 欧氏 K-means
            # Use standard Euclidean K-means (not hyperbolic)
            assignments, centroids, history = euclidean_kmeans(
                r, n_clusters, max_iter=max_iter, batch=BATCH, seed=SEED + l,
                verbose=verbose,
            )
            codes[:, l] = assignments
            codebooks.append(centroids.cpu())                                # (K, D) Euclidean

            # Residual: r_{l+1} = r - C_l[c_l] (Euclidean subtraction)
            new_r = torch.zeros_like(r)
            for i in range(0, r.shape[0], BATCH):
                r_batch = r[i:i+BATCH]
                assn_batch = assignments[i:i+BATCH]
                c_batch = centroids[assn_batch]                              # (B, D)
                new_r[i:i+BATCH] = r_batch - c_batch
            r = new_r
            residuals.append(r.cpu())
    return codes, codebooks, residuals


def euclidean_kmeans(x, n_clusters, max_iter=MAX_ITER, batch=BATCH, seed=SEED,
                     verbose=True):
    """标准 Euclidean K-means (用于 tangent space residual quantization)
       x: (N, D) Euclidean
       Returns: assignments (N,), centroids (K, D), history
    """
    rng = np.random.default_rng(seed)
    N, D = x.shape
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centroids = x[init_idx].clone()
    history = []
    best_qerr = float('inf')
    patience_count = 0

    for it in range(max_iter):
        t0 = time.time()
        # Distance: ||x - c||²
        # Chunked
        dist = torch.zeros(N, n_clusters, device=x.device)
        for i in range(0, N, batch):
            xb = x[i:i+batch]
            d = ((xb.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)
            dist[i:i+batch] = d
        assignments = dist.argmin(dim=1)

        # QErr
        q_err = 0.0
        for k in range(n_clusters):
            mask = (assignments == k)
            if mask.any():
                d = (x[mask] - centroids[k]).norm(dim=-1)
                q_err += (d ** 2).sum().item()
        history.append(q_err)
        if verbose:
            print(f'  iter {it}: QErr={q_err:.4f}  ({time.time()-t0:.1f}s)')

        # Update: mean of assigned
        new_centroids = torch.empty_like(centroids)
        for k in range(n_clusters):
            mask = (assignments == k)
            if mask.any():
                new_centroids[k] = x[mask].mean(0)
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids

        # Convergence
        if q_err < best_qerr - 1e-3:
            best_qerr = q_err
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= DROPOUT_PATIENCE:
                if verbose:
                    print(f'  early stop at iter {it}')
                break
    return assignments, centroids, history


def add_dedup_digit(codes):
    """Add dedup digit (4th column): integer encoding of duplicate handling
       Codes: (N, n_layers) → (N, n_layers+1)
       Dedup column = 0 if unique; otherwise index of duplicate
       For Toys single dataset, no major collisions expected at 256^3 = 16.7M capacity
    """
    N, L = codes.shape
    full_codes = torch.cat([codes, torch.zeros(N, 1, dtype=torch.long, device=codes.device)], dim=1)
    # Dedup: assign 0/1 (1 if first occurrence)
    seen = set()
    for i in range(N):
        key = tuple(codes[i].cpu().tolist())
        if key in seen:
            full_codes[i, L] = 1
        else:
            seen.add(key)
    return full_codes


# ====================== Main ======================

def main():
    print('=' * 70)
    print('task20 — HRQ 双曲残差量化 (Lorentz model)')
    print('=' * 70)
    print(f'device: {DEVICE}')

    # 1. Load embedding
    print('\n=== Load flan-t5-xl Toys embedding ===')
    x = torch.load(EMBED_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0)                                # (N, 2048)
    else:
        x_eu = x
    if hasattr(x_eu, 'numpy'):
        x_eu = x_eu.numpy()
    x_eu = torch.tensor(x_eu, dtype=torch.float32)
    N, D = x_eu.shape
    print(f'  x_eu shape: {x_eu.shape}')

    # Optional subsample for speed (remove for full dataset)
    # x_eu = x_eu[:5000]
    # N = x_eu.shape[0]

    # 2. Lift to Lorentz
    print('\n=== Lift to Lorentz ===')
    x_lorentz = lift_to_lorentz(x_eu.to(DEVICE))
    print(f'  x_lorentz shape: {x_lorentz.shape}')

    # 3. HRQ forward (3 layers)
    print('\n=== HRQ forward: 3 layers ===')
    codes, codebooks, residuals = hrq_forward(x_lorentz, n_layers=N_LAYERS,
                                               n_clusters=N_CLUSTERS,
                                               max_iter=MAX_ITER, device=DEVICE)

    codes_cpu = codes.cpu()
    print(f'\n  codes shape (N, L): {codes_cpu.shape}')
    print(f'  layer 1 unique: {codes_cpu[:, 0].unique().shape[0]}/{N_CLUSTERS}')
    print(f'  layer 2 unique: {codes_cpu[:, 1].unique().shape[0]}/{N_CLUSTERS}')
    print(f'  layer 3 unique: {codes_cpu[:, 2].unique().shape[0]}/{N_CLUSTERS}')
    full_unique = torch.unique(codes_cpu, dim=0).shape[0]
    print(f'  full code tuple unique: {full_unique}/{N}')

    # 4. Save SID with dedup digit
    print('\n=== Save SID ===')
    full_codes = add_dedup_digit(codes_cpu)
    sid_path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
    os.makedirs(os.path.dirname(sid_path), exist_ok=True)
    sid = full_codes.t()                                                # (4, 11924) — match RQ shape
    torch.save(sid, sid_path)
    print(f'  SID shape: {sid.shape} → {sid_path}')

    # 5. Save codebooks + residuals for diagnostic
    codebook_path = os.path.join(CKPT_DIR, 'task18_hrq_codebooks.pt')
    torch.save({
        'codebooks': codebooks,
        'residuals': residuals,
        'codes': codes_cpu,
        'curvature': CURVATURE,
        'n_layers': N_LAYERS,
        'n_clusters': N_CLUSTERS,
    }, codebook_path)
    print(f'  codebooks → {codebook_path}')

    # 6. Basic diagnostic
    print('\n=== Diagnostic ===')
    print(f'  RQ 3 码本 (256^3=16.7M capacity) 唯一率: {full_unique/N:.4f} (碰撞率 {1 - full_unique/N:.4f})')
    if full_unique == N:
        print('  ✅ 无碰撞 — dedup digit 全 0')
    else:
        print(f'  ✓ {N - full_unique} 项碰撞')

    # 7. Pairwise Lorentz distance sample
    print('\n=== Pairwise Lorentz distance sample (sample 500 for speed) ===')
    sample_n = 500
    h1 = x_lorentz[:sample_n]
    h2 = x_lorentz[1:sample_n + 1]
    d = lorentz_distance(h1, h2)
    print(f'  mean: {d.mean().item():.4f}, std: {d.std().item():.4f}')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
