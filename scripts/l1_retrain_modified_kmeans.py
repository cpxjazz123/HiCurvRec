#!/usr/bin/env python3
"""L1 真重训 — 改造版 K-means 解决后验重分配被 KILL 的问题

原始 Idea 2 (post-hoc λ scan) 被 KILL_POST_HOC, 因为后验重分配不能保证码本利用率.
这次**真改 L1 K-means 训练损失**, 在训练时直接优化 task-coherence.

新损失:
  L_1^new = E[‖x - q_1‖²] - λ · E[‖U^T q_1‖²]
  q_1 是 x 在 L1 量化器最近码字
  U 是 task 标签矩阵 (N_items × N_classes 的 SVD top-k)

码字更新 (修改版):
  c_j ← mean(x_i + μ · UU^T x_i) for i assigned to j
  (UU^T 是 task 子空间投影; μ 控制 task-bias 强度)

两阶段退火:
  λ(t) = λ_max · min(1, t / T_warmup)
  μ(t) = μ_max · min(1, t / T_warmup)

超参扫描:
  λ_max ∈ {0.1, 0.3, 0.5, 1.0}
  μ ∈ {0, 0.2, 0.5}
  = 12 组

每组跟踪 3 条曲线:
  ρ^task(q_1; t) — 量化器对 task 的敏感度
  recon_error(t)
  codeword usage(t) — 实际使用码字数

判定 (用户给定):
  KILL: 任何 (λ_max, μ) 都无法同时满足 ρ ≥ 0.20 + recon growth ≤ 20% + codewords ≥ 240/256
  PASS: 任一组满足以上 + 端到端 R@10 稳定提升 (3-seed)

复用:
- Stage 1 embedding (X_raw, 2048-d)  + PCA-50
- toys metadata → 24 类 cat_sub (平衡, Gini=0.459) → top-4 task 信号
"""
import os, json, time
import numpy as np
import torch
from scipy.stats import spearmanr

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/l1_retrain_modified_kmeans'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'

PCA_DIM = 50
N_CLUSTERS = 256
N_ITERS = 50
SEED = 42
T_WARMUP = 10        # 退火热身步
LAMBDA_GRID = [0.1, 0.3, 0.5, 1.0]
MU_GRID = [0.0, 0.2, 0.5]


def whiten_pca(X, n_components=PCA_DIM):
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh[:n_components].T
    return (X_centered @ V) / (S[:n_components].clamp(min=1e-8))


def build_task_subspace(cat_sub, X, n_components=4):
    """Build U: (D, n_components) — task directions in the D-dim embedding space.

       Compute per-class mean embeddings, center, SVD top-k → Vh of shape (k, D).
       So U = Vh[:k] is (k, D). Then for any x ∈ R^D: U @ U.T @ x = U (U x)
       is the projection of x onto task subspace (D-dim vector).
       Returns: U (k, D), cls_idx (N,)
    """
    N, D = X.shape
    classes = sorted(set(cat_sub))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    cls_idx = np.array([cls_to_idx[c] for c in cat_sub])
    one_hot = np.zeros((N, len(classes)), dtype=np.float32)
    one_hot[np.arange(N), cls_idx] = 1.0
    # Per-class mean embedding
    counts = one_hot.sum(0)  # (n_classes,)
    means = (one_hot.T @ X.numpy()) / np.maximum(counts, 1).reshape(-1, 1)  # (n_classes, D)
    means_t = torch.from_numpy(means).float()
    mean_global = means_t.mean(0, keepdim=True)
    means_centered = means_t - mean_global
    k = min(n_components, means_centered.shape[0])
    U_S, S, Vh = torch.linalg.svd(means_centered, full_matrices=False)
    # Vh is (k, D). We want U of shape (k, D), so U^T @ x = Vh[:k] @ x = (k,)
    U = Vh[:k]  # (k, D)
    return U, cls_idx


def compute_rho_task(assignments, codebook_spatial, U, cls_idx):
    """ρ^task: Spearman correlation between assignment (proxy for codeword)
       and task class. We use one-hot agreement:
       For each pair (i, j) in same cluster: fraction same task class.
       Normalized by random baseline.
    """
    N = assignments.shape[0]
    same_cluster_same_task = 0
    pair_total = 0
    rng = np.random.default_rng(SEED)
    cluster_ids = assignments.unique().tolist()
    for k in cluster_ids:
        mask = assignments == k
        ids = torch.where(mask)[0].numpy()
        if len(ids) < 2:
            continue
        # sample up to 100 pairs
        n_pairs = min(100, len(ids) * (len(ids) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(ids, size=2, replace=False)
            if cls_idx[i] == cls_idx[j]:
                same_cluster_same_task += 1
            pair_total += 1
    if pair_total == 0:
        return 0.0
    return same_cluster_same_task / pair_total


def modified_kmeans(X, U, cls_idx, lam, mu, n_clusters=N_CLUSTERS, n_iters=N_ITERS,
                    seed=SEED, t_warmup=T_WARMUP):
    """Modified K-means with task-aware loss:
       L = E[‖x - q_1‖²] - λ · E[‖U^T q_1‖²]
       (UU^T is the task projector onto k-dim task subspace)

       Assignment: argmin over clusters of ‖x - c_j‖² - λ · ‖U^T c_j‖²
       Update: c_j = mean( x_i + μ · UU^T x_i  for i in cluster j )

       Annealing:
       λ(t) = λ · min(1, t / t_warmup)
       μ(t) = μ · min(1, t / t_warmup)

       Returns: assignments, codebook, history
    """
    N, D = X.shape
    k = U.shape[0]  # U is (k, D)
    # UU^T x = U^T (U x). Compute the projector matrix P = U^T @ U is (D, D),
    # then P @ x is the (D,) task-projection.

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    codebook = X[init_idx].clone()  # (K, D)
    assignments = torch.zeros(N, dtype=torch.long)

    history = {
        'rho_task': [],
        'recon_error': [],
        'codewords_used': [],
    }

    P = U.T @ U  # (D, D) task projector

    for it in range(n_iters):
        # Annealing
        anneal = min(1.0, (it + 1) / t_warmup)
        lam_t = lam * anneal
        mu_t = mu * anneal

        # Compute task-bias for each item: x_i + mu_t · P x_i  (P = U^T U, D×D)
        X_task_biased = X + mu_t * (X @ P.T)  # (N, D)

        # Assignment with task-modified distance
        #   d(x_i, c_j)² - λ_t · ‖U c_j‖²   (U c_j is (k,), so ‖U c_j‖² is scalar)
        Uc = U @ codebook.T  # (k, K) — projection of each codeword onto task subspace
        task_penalty = lam_t * (Uc ** 2).sum(0)  # (K,)
        for i in range(0, N, 2048):
            xb = X[i:i+2048]                          # (B, D)
            xb_tb = X_task_biased[i:i+2048]           # (B, D)
            # squared dist (B, K)
            dists = (xb.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
            dists = dists - task_penalty.unsqueeze(0)  # (B, K)
            assignments[i:i+2048] = dists.argmin(dim=-1)

        # Update: weighted mean of biased items per cluster
        new_codebook = torch.zeros_like(codebook)
        for kk in range(n_clusters):
            mask = assignments == kk
            n_k = mask.sum().item()
            if n_k == 0:
                rng2 = np.random.default_rng(seed=seed + kk + it)
                new_codebook[kk] = X[rng2.integers(0, N)]
                continue
            elif n_k == 1:
                new_codebook[kk] = X_task_biased[mask][0]
                continue
            new_codebook[kk] = X_task_biased[mask].mean(0)
        codebook = new_codebook

        # Track metrics
        recon = (X - codebook[assignments]).pow(2).sum(-1).mean().item()
        rho = compute_rho_task(assignments, codebook, U, cls_idx)
        n_used = assignments.unique().shape[0]

        history['rho_task'].append(float(rho))
        history['recon_error'].append(float(recon))
        history['codewords_used'].append(int(n_used))

    return assignments, codebook, history


def main():
    print('=' * 70)
    print('L1 真重训: 改造版 K-means 12 组扫描')
    print('=' * 70)

    # Load
    print('\n[Step 1] 加载数据 + PCA-50')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N, D = x_eu.shape
    print(f'  embedding: {N} × {D}')
    x_white = whiten_pca(x_eu, n_components=PCA_DIM)
    print(f'  PCA-{PCA_DIM}: {tuple(x_white.shape)}')

    # Load metadata
    print('\n[Step 2] 加载 metadata + 构建 task subspace')
    with open(META_PATH) as f:
        md = json.load(f)
    cat_sub = [md[str(i)]['cat_sub'] for i in range(N)]
    U, cls_idx = build_task_subspace(cat_sub, x_white, n_components=4)
    print(f'  task subspace U: {tuple(U.shape)}, n_classes = {len(set(cat_sub))}')
    print(f'  class distribution: {dict(zip(*np.unique(cls_idx, return_counts=True)))}')

    # Scan
    print('\n[Step 3] 扫描 (λ_max, μ) 12 组')
    results = {}
    for lam in LAMBDA_GRID:
        for mu in MU_GRID:
            t0 = time.time()
            print(f'\n  --- λ_max={lam}, μ={mu} ---')
            assignments, codebook, history = modified_kmeans(
                x_white, U, cls_idx, lam, mu,
                n_clusters=N_CLUSTERS, n_iters=N_ITERS, seed=SEED, t_warmup=T_WARMUP
            )
            elapsed = time.time() - t0
            final_rho = history['rho_task'][-1]
            final_recon = history['recon_error'][-1]
            final_used = history['codewords_used'][-1]
            initial_recon = history['recon_error'][0]
            recon_growth = (final_recon - initial_recon) / max(initial_recon, 1e-9)
            print(f'    final rho_task = {final_rho:.4f}')
            print(f'    final recon    = {final_recon:.4f} (initial {initial_recon:.4f}, '
                  f'growth {recon_growth * 100:+.1f}%)')
            print(f'    codewords used = {final_used}/{N_CLUSTERS}')
            print(f'    elapsed = {elapsed:.1f}s')
            results[(lam, mu)] = {
                'lambda': lam, 'mu': mu,
                'history': history,
                'final_rho': float(final_rho),
                'final_recon': float(final_recon),
                'initial_recon': float(initial_recon),
                'recon_growth_pct': float(recon_growth * 100),
                'codewords_used': int(final_used),
                'elapsed_sec': float(elapsed),
            }

    # Step 4: kill criterion (user-defined)
    print('\n[Step 4] Kill 判定 (用户给定)')
    print('  任一 (λ, μ) 满足: ρ ≥ 0.20 + recon growth ≤ 20% + codewords ≥ 240/256')
    print('  → 任一不满足 → KILL')
    print('  → 任一满足 → 进入 Step 5 (端到端 R@10)')
    survivors = []
    for (lam, mu), r in results.items():
        if r['final_rho'] >= 0.20 and r['recon_growth_pct'] <= 20 and r['codewords_used'] >= 240:
            survivors.append((lam, mu, r))
    print(f'\n  通过 kill 线的组数: {len(survivors)}/{len(results)}')
    if survivors:
        for lam, mu, r in survivors:
            print(f'    ✓ (λ={lam}, μ={mu}): ρ={r["final_rho"]:.3f}, '
                  f'recon_growth={r["recon_growth_pct"]:+.1f}%, '
                  f'codewords={r["codewords_used"]}')

    if not survivors:
        verdict = (
            f'L1_RETRAIN_KILL: 12 组 (λ_max, μ) 全部未通过 kill 线 '
            f'(ρ≥0.20 + recon growth≤20% + codewords≥240). '
            f'修改版 K-means 不能同时保 task coherence, 重构质量, 和码本利用率. '
            f'当前最优 (按 ρ): '
            f'{max(results.items(), key=lambda x: x[1]["final_rho"])}'
        )
        outcome = 'kill_l1_retrain'
    else:
        best = max(survivors, key=lambda x: x[2]['final_rho'])
        verdict = (
            f'L1_RETRAIN_PASS: {len(survivors)} 组通过 kill 线, '
            f'best (λ={best[0]}, μ={best[1]}): ρ={best[2]["final_rho"]:.3f}. '
            f'下一步: 用这组跑 full pipeline + 3-seed R@10 验证.'
        )
        outcome = 'pass_l1_retrain'

    print(f'\n[Verdict] {verdict}')

    # Save
    out_json = os.path.join(OUT_DIR, 'l1_retrain_12group.json')
    with open(out_json, 'w') as f:
        json.dump({
            'method': 'modified_kmeans_with_task_loss',
            'description': (
                'Modified K-means with task-aware loss L = E[||x-q||²] - λE[||U^T q||²] '
                'and task-biased centroid update c_j ← mean(x_i + μ UU^T x_i). '
                'Annealing: λ(t) = λ_max · min(1, t/T_warmup), μ same.'
            ),
            'params': {
                'pca_dim': PCA_DIM,
                'n_clusters': N_CLUSTERS,
                'n_iters': N_ITERS,
                't_warmup': T_WARMUP,
                'seed': SEED,
                'lambda_grid': LAMBDA_GRID,
                'mu_grid': MU_GRID,
            },
            'task_subspace': {
                'method': 'SVD of one-hot task labels',
                'n_components': int(U.shape[1]),
                'n_classes': int(len(set(cat_sub))),
            },
            'results_per_group': {
                f'lam{lam}_mu{mu}': r for (lam, mu), r in results.items()
            },
            'survivors': [
                {'lambda': lam, 'mu': mu, **r} for lam, mu, r in survivors
            ],
            'kill_criteria': {
                'rho_min': 0.20,
                'recon_growth_max_pct': 20,
                'codewords_min': 240,
            },
            'verdict': verdict,
            'outcome': outcome,
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Final rho per (lam, mu)
        ax = axes[0, 0]
        for mu in MU_GRID:
            ys = [results[(lam, mu)]['final_rho'] for lam in LAMBDA_GRID]
            ax.plot(LAMBDA_GRID, ys, 'o-', label=f'μ={mu}')
        ax.set_xlabel('λ_max')
        ax.set_ylabel('final ρ_task')
        ax.set_title('Final ρ_task per (λ, μ)')
        ax.axhline(0.20, color='red', linestyle='--', alpha=0.5, label='kill ρ=0.20')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final recon
        ax = axes[0, 1]
        for mu in MU_GRID:
            ys = [results[(lam, mu)]['final_recon'] for lam in LAMBDA_GRID]
            ax.plot(LAMBDA_GRID, ys, 'o-', label=f'μ={mu}')
        ax.set_xlabel('λ_max')
        ax.set_ylabel('final recon_error')
        ax.set_title('Final reconstruction error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Codewords used
        ax = axes[1, 0]
        for mu in MU_GRID:
            ys = [results[(lam, mu)]['codewords_used'] for lam in LAMBDA_GRID]
            ax.plot(LAMBDA_GRID, ys, 'o-', label=f'μ={mu}')
        ax.set_xlabel('λ_max')
        ax.set_ylabel('codewords used')
        ax.set_title(f'Codeword usage (kill ≥ {240})')
        ax.axhline(240, color='red', linestyle='--', alpha=0.5)
        ax.set_ylim(0, N_CLUSTERS + 5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Recon growth
        ax = axes[1, 1]
        for mu in MU_GRID:
            ys = [results[(lam, mu)]['recon_growth_pct'] for lam in LAMBDA_GRID]
            ax.plot(LAMBDA_GRID, ys, 'o-', label=f'μ={mu}')
        ax.set_xlabel('λ_max')
        ax.set_ylabel('recon growth (%)')
        ax.set_title('Reconstruction growth (kill ≤ 20%)')
        ax.axhline(20, color='red', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('L1 真重训: 12 组 (λ_max, μ) 扫描')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, 'l1_retrain_12group.png')
        plt.savefig(out_png, dpi=120)
        print(f'[SAVED] {out_png}')
    except Exception as e:
        print(f'[WARN] plot failed: {e}')

    return verdict, results


if __name__ == '__main__':
    main()
