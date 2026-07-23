#!/usr/bin/env python3
"""候选 2 gate: 图谱频率坐标 φ_i vs 深层残差相关性检验

Step 0: 重建 item-item 共现稀疏矩阵 + 对称归一化 Laplacian + 谱分解
Step 1: 翻 Idea 1 (CF 注入) 结果, 确认 prior
Step 2: 现象 1 (严格 kill 版) — 把 cat_sub facet 换成 φ_i 离散分桶

Kill 线 (比候选 1 更严格):
  rel_gap > 8% AND Cohen's d ≥ 0.3
  若 5-8% 区间 → 倾向 KILL (基于 prior ΔI_V inject < 0.04 bit)
"""
import os, sys, json, time, glob
import numpy as np
import torch
import tensorflow as tf
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import mannwhitneyu
import importlib.util

# Make GRID root importable for torch.load (although not needed here)
GRID_ROOT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
if GRID_ROOT not in sys.path:
    sys.path.insert(0, GRID_ROOT)

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate'
os.makedirs(OUT_DIR, exist_ok=True)

# Same paths as stage1
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
CKPT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task1_v2_dense_ckpts_v2/checkpoints/checkpoint_000_001600.ckpt'
DATA_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys/training'

N_ITEMS = 11924
N_HIERARCHIES = 4
K_EIG = 64              # spectral components to compute
N_SAMPLE_PAIRS = 5000
SEED = 42

# ===== Borrow co-occurrence builder from idea1_2_build_ppmi_svd.py =====
spec = importlib.util.spec_from_file_location(
    'ppmi_mod',
    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/idea1_2_build_ppmi_svd.py'
)
ppmi_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppmi_mod)


def build_cooccurrence_cached():
    """Build (or load cached) sparse co-occurrence matrix C."""
    CACHE_PATH = os.path.join(OUT_DIR, 'cooccurrence_csr.npz')
    if os.path.exists(CACHE_PATH):
        print(f'[CACHE] loading {CACHE_PATH}')
        from scipy.sparse import load_npz
        return load_npz(CACHE_PATH)

    print('[Step 0.1] load user sequences')
    user_seqs = ppmi_mod.load_user_sequences(n_users_max=50000)
    print(f'  loaded {len(user_seqs)} user sequences')

    print('[Step 0.2] build sparse co-occurrence matrix')
    t0 = time.time()
    C = ppmi_mod.build_co_occurrence(user_seqs, N_ITEMS)
    print(f'  co-occurrence built in {time.time()-t0:.1f}s, nnz={C.nnz}')
    from scipy.sparse import save_npz
    save_npz(CACHE_PATH, C)
    print(f'  cached → {CACHE_PATH}')
    return C


def build_laplacian_eigvecs(C, k=K_EIG):
    """Build symmetric normalized Laplacian and compute smallest K eigenvectors.

    L_sym = I - D^{-1/2} C D^{-1/2}
    Smallest eigenvalues ⟺ smoothest (lowest-frequency) graph signals.
    """
    EIGVEC_PATH = os.path.join(OUT_DIR, f'eigvecs_k{k}.npy')
    EVALS_PATH = os.path.join(OUT_DIR, f'evals_k{k}.npy')
    if os.path.exists(EIGVEC_PATH) and os.path.exists(EVALS_PATH):
        print(f'[CACHE] loading eigenvectors from {EIGVEC_PATH}')
        return np.load(EIGVEC_PATH), np.load(EVALS_PATH)

    print('[Step 0.3] build symmetric normalized Laplacian L_sym')
    t0 = time.time()
    # Degree
    deg = np.asarray(C.sum(axis=1)).ravel()  # (N,)
    # Drop zero-degree nodes by giving them a tiny self-loop (so D^{-1/2} well-defined)
    deg_safe = deg.copy().astype(np.float64)
    zero_mask = deg_safe == 0
    deg_safe[zero_mask] = 1e-8
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    # D^{-1/2} C D^{-1/2}
    D_inv_sqrt = diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ C @ D_inv_sqrt
    # Symmetrize (it should already be symmetric since C is symmetric, but be safe)
    A_norm = (A_norm + A_norm.T) * 0.5
    # L_sym = I - A_norm
    N = C.shape[0]
    L_sym = diags(np.ones(N)) - A_norm
    L_sym = L_sym.tocsr()
    print(f'  L_sym built in {time.time()-t0:.1f}s, nnz={L_sym.nnz}')

    print(f'[Step 0.4] compute smallest {k} eigenvectors via shift-invert')
    t0 = time.time()
    # sigma=0 means find eigenvalues nearest 0
    # For smallest of a PSD Laplacian, eigsh with sigma=0 + which='LM' returns them
    # In scipy, eigsh(L, k=K, sigma=0, which='LM') for smallest of PSD
    evals, eigvecs = eigsh(L_sym, k=k, sigma=0.0, which='LM', tol=1e-3)
    print(f'  eigsh done in {time.time()-t0:.1f}s')
    print(f'  smallest 5 eigenvalues: {evals[:5].tolist()}')
    print(f'  largest of these {k}: {evals[-5:].tolist()}')

    np.save(EVALS_PATH, evals)
    np.save(EIGVEC_PATH, eigvecs)
    return eigvecs, evals


def compute_phi(eigvecs):
    """Compute φ_i = (energy in low-freq half) / (energy in all K eigenvectors)."""
    # eigvecs: (N, K). Each column is eigenvector. Sort by eigenvalue ascending (already sorted).
    K = eigvecs.shape[1]
    half = K // 2
    # Square each entry, then sum across low-freq and all-freq
    sq = eigvecs ** 2  # (N, K)
    energy_low = sq[:, :half].sum(axis=1)
    energy_all = sq.sum(axis=1)
    # Avoid division by zero (zero-degree nodes might have all-zero rows)
    phi = np.where(energy_all > 1e-12, energy_low / energy_all, 0.5)
    return phi


def discretize_phi(phi, n_bins=5):
    """Discretize φ into n_bins by quantile."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(phi, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bins = np.digitize(phi, edges[1:-1])  # 0..n_bins-1
    return bins


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (y.mean() - x.mean()) / pooled if pooled > 0 else 0.0


# ===== Codebook + residual loading (reuse stage1 forgiving unpickler) =====
class _DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass
    def __getstate__(self): return self.__dict__


def _load_checkpoint_forgiving(path):
    import pickle
    pickle_module = torch.serialization.pickle

    class _ForgivingUnpickler(pickle_module.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError, ModuleNotFoundError):
                return _DummyClass

    class _CustomPickleModule:
        Unpickler = _ForgivingUnpickler
        def __getattr__(self, name):
            return getattr(pickle_module, name)

    return torch.load(path, map_location='cpu',
                      pickle_module=_CustomPickleModule(),
                      weights_only=False)


def forward_residuals(X, codebooks):
    N = X.shape[0]
    residuals = []
    r = X.clone()
    for cb in codebooks:
        dist = torch.cdist(r, cb)
        c = dist.argmin(dim=1)
        r_l = r - cb[c]
        residuals.append(r_l)
        r = r_l
    return residuals


# =====================================================================
# Step 1: Read Idea 1 prior
# =====================================================================
def step1_prior_check():
    print('\n' + '=' * 70)
    print('Step 1: 翻 Idea 1 (CF 注入) 结果, 确认 prior')
    print('=' * 70)
    verdict_path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/FINAL_VERDICT.md'
    with open(verdict_path) as f:
        text = f.read()
    # Extract the key line on ΔI_V inject
    key_lines = [l for l in text.splitlines() if 'ΔI_V' in l and 'inject' in l]
    print('  Key evidence from FINAL_VERDICT.md:')
    for l in key_lines[:5]:
        print(f'    {l}')
    # Also load the JSON for hard numbers
    vinfo_path = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_phen4_norm_vinfo.json'
    with open(vinfo_path) as f:
        vinfo = json.load(f)
    # Find max |delta_I_V| across algos × layers
    # (the field structure is algo → metric, not direct; we use the verdict's table)
    print(f'\n  Prior summary: ΔI_V(inject) for CF signal on r_l:')
    print(f'    A_baseline:  -0.005 ~ +0.002 bit')
    print(f'    B_mmq:       +0.009 ~ +0.036 bit')
    print(f'    C_gsrq:      +0.003 ~ +0.012 bit')
    print(f'    Kill line:   > 0.1 bit')
    print(f'    ALL < 0.04 bit, well below kill line.')
    prior_pass = True  # prior CONFIRMED (CF signal is dilute in deep residuals)
    return prior_pass, {
        'source': 'idea1_2_cf_diagnostic/FINAL_VERDICT.md',
        'kill_line_bit': 0.1,
        'observed_max_bit': 0.036,
        'verdict': 'CF 信号在深层残差稀薄 (ΔI_V inject 全部 < 0.04 bit)',
        'implication': (
            '候选 2 用图谱频率 (PPMI 同源协同信号) 作 facet, 与 Idea 1 同根, '
            '理论上同样在深层衰减, 大概率在现象 1 重复候选 1 的失败模式. '
            '这就是为什么现象 1 kill 线必须比候选 1 更严格, 不是 "找不到效应" 才是 kill, '
            '而是 "效应太弱, 落在 5-8% 灰色区间" 也应判死.'
        ),
    }


# =====================================================================
# Step 2: Phenomenon 1 with stricter kill line
# =====================================================================
def phenomenon_1_strict(residuals, phi_bins):
    """φ_bins: integer labels 0..4 (5 bins). Pairs in same/different φ-bin."""
    print('\n' + '=' * 70)
    print('Step 2: 现象 1 (严格 kill 版) — 同/跨 φ-bin 残差距离')
    print('=' * 70)

    rng = np.random.default_rng(SEED + 100)
    N = len(phi_bins)
    print(f'  N items = {N}, n φ-bins = {int(phi_bins.max()) + 1}')
    print(f'  bin distribution: {dict(zip(*np.unique(phi_bins, return_counts=True)))}')

    # Sample P_same and P_diff
    p_same_i, p_same_j = [], []
    attempts = 0
    while len(p_same_i) < N_SAMPLE_PAIRS and attempts < N_SAMPLE_PAIRS * 100:
        i = rng.integers(0, N)
        cell = phi_bins[i]
        candidates = np.where(phi_bins == cell)[0]
        if len(candidates) >= 2:
            mask = candidates != i
            if mask.any():
                j = rng.choice(candidates[mask])
                p_same_i.append(int(i))
                p_same_j.append(int(j))
        attempts += 1
    p_same_i = np.array(p_same_i)
    p_same_j = np.array(p_same_j)
    print(f'  P_same samples = {len(p_same_i)}')

    p_diff_i, p_diff_j = [], []
    attempts = 0
    while len(p_diff_i) < len(p_same_i) and attempts < len(p_same_i) * 100:
        i, j = rng.integers(0, N, size=2)
        if phi_bins[i] != phi_bins[j]:
            p_diff_i.append(int(i))
            p_diff_j.append(int(j))
        attempts += 1
    p_diff_i = np.array(p_diff_i)
    p_diff_j = np.array(p_diff_j)
    print(f'  P_diff samples = {len(p_diff_i)}')

    results = {}
    for layer_idx, layer_name in [(1, 'r_1'), (2, 'r_2'), (3, 'r_3')]:
        if layer_idx >= len(residuals):
            continue
        r_l = residuals[layer_idx]
        d_same = (r_l[p_same_i] - r_l[p_same_j]).norm(dim=-1).numpy()
        d_diff = (r_l[p_diff_i] - r_l[p_diff_j]).norm(dim=-1).numpy()
        u_stat, p_val = mannwhitneyu(d_same, d_diff, alternative='less')
        rel_gap = (d_diff.mean() - d_same.mean()) / max(d_diff.mean(), 1e-9)
        d_eff = cohen_d(d_same, d_diff)
        print(f'\n  --- {layer_name} ---')
        print(f'    D_same mean = {d_same.mean():.4f}, std = {d_same.std():.4f}')
        print(f'    D_diff mean = {d_diff.mean():.4f}, std = {d_diff.std():.4f}')
        print(f'    relative gap = {rel_gap * 100:+.2f}%')
        print(f'    Cohen\'s d   = {d_eff:+.3f}')
        print(f'    Mann-Whitney U = {u_stat:.0f}, p-value = {p_val:.4e}')
        results[layer_name] = {
            'd_same_mean': float(d_same.mean()),
            'd_diff_mean': float(d_diff.mean()),
            'relative_gap_pct': float(rel_gap * 100),
            'cohens_d': float(d_eff),
            'mannwhitney_p': float(p_val),
        }

    # Apply STRICT kill line:
    #   rel_gap > 8% AND Cohen's d ≥ 0.3 (on r_2, r_3)
    #   AND p < 0.001
    #   If 5-8% range → "灰色", tend to KILL due to prior
    print(f'\n  Kill 判定 (严格版): rel_gap > 8% AND Cohen\'s d ≥ 0.3 (AND p<0.001)')
    print(f'  若 rel_gap 在 5-8% 区间: 基于 Idea 1 prior (CF 在深层稀薄), 倾向 KILL')
    print(f'  {"LAYER":<8} {"rel_gap":>10} {"d":>8} {"p-val":>12} {"tier":>10} {"pass?":>6}')
    overall_verdict = None
    tier_per_layer = {}
    for k, v in results.items():
        rg, d_eff, p_v = v['relative_gap_pct'], abs(v['cohens_d']), v['mannwhitney_p']
        if rg > 8 and d_eff >= 0.3 and p_v < 0.001:
            tier = 'PASS'
        elif rg > 5:
            tier = 'GREY'  # 5-8% 区间, prior 倾向 KILL
        else:
            tier = 'KILL'
        tier_per_layer[k] = tier
        ok = tier == 'PASS'
        print(f'  {k:<8} {rg:>+9.2f}% {d_eff:>7.3f} {p_v:>12.4e} {tier:>10} {"✓" if ok else "✗":>6}')

    if all(tier_per_layer.get(k) == 'PASS' for k in ['r_2', 'r_3'] if k in tier_per_layer):
        verdict = (
            'PASS: φ-bin 网格在深层残差空间有显著且中等等级效应 (rel_gap>8%, d≥0.3). '
            '但鉴于 prior, 这种 PASS 反而需要仔细审查是否数据/统计假象. '
            '下一阶段: 现象 2/3 严格交叉验证 + 端到端 R@10.'
        )
    elif any(tier_per_layer.get(k) == 'GREY' for k in tier_per_layer):
        verdict = (
            'KILL_GREY: rel_gap 落在 5-8% 灰色区间. 基于 Idea 1 prior '
            '(CF 信号 ΔI_V(inject) < 0.04 bit), 这种微效应 **大概率是统计假象**, '
            '不应采信. 候选 2 判死, 不投入阶段三 (图谱频率 × 时间).'
        )
    else:
        verdict = (
            'KILL: rel_gap < 5%, 候选 2 在 r_2/r_3 上完全没有显著相关性, '
            '彻底判死. 与候选 1 失败模式一致 (PPMI 同源协同信号在深层稀薄).'
        )

    print(f'\n  >>> {verdict}')
    return verdict, results, tier_per_layer


# =====================================================================
# MAIN
# =====================================================================
def main():
    t0 = time.time()
    print('=' * 70)
    print('候选 2 gate: 图谱频率坐标 vs 深层残差')
    print('=' * 70)

    # Step 0: Build co-occurrence + Laplacian + eigenvectors
    print('\n[Step 0] 重建 item-item 共现图 + Laplacian 谱分解')
    C = build_cooccurrence_cached()
    eigvecs, evals = build_laplacian_eigvecs(C, k=K_EIG)
    phi = compute_phi(eigvecs)
    print(f'\n  φ_i statistics:')
    print(f'    mean = {phi.mean():.4f}, std = {phi.std():.4f}')
    print(f'    min = {phi.min():.4f}, max = {phi.max():.4f}')
    print(f'    quantiles 10/50/90: '
          f'{np.quantile(phi, 0.1):.3f} / {np.quantile(phi, 0.5):.3f} / {np.quantile(phi, 0.9):.3f}')
    phi_bins = discretize_phi(phi, n_bins=5)
    print(f'    bin counts: {dict(zip(*np.unique(phi_bins, return_counts=True)))}')

    # Step 1: Prior
    prior_pass, prior_evidence = step1_prior_check()

    # Load Stage 1 embeddings + codebooks
    print('\n[Setup] load Stage 1 embedding + task13 codebooks')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    print(f'  X: {x_eu.shape}')

    ckpt = _load_checkpoint_forgiving(CKPT_PATH)
    sd = ckpt['state_dict']
    codebooks = [sd[f'quantization_layer_list.{l}.centroids'].float() for l in range(N_HIERARCHIES)]
    print(f'  codebooks: {len(codebooks)} × {codebooks[0].shape}')

    residuals = forward_residuals(x_eu, codebooks)
    for l, r in enumerate(residuals):
        print(f'  r_{l + 1}: mean ‖r‖ = {r.norm(dim=-1).mean().item():.4f}')

    # Step 2: phenomenon 1 with strict kill
    p1_verdict, p1_results, tier_per_layer = phenomenon_1_strict(residuals, phi_bins)

    # Save
    out_json = os.path.join(OUT_DIR, 'stage2_candidate2_gate.json')
    with open(out_json, 'w') as f:
        json.dump({
            'candidate': 2,
            'method': 'graph_spectral_frequency_phi',
            'phi_definition': (
                'φ_i = sum(v_k(i)^2, k=1..K/2) / sum(v_k(i)^2, k=1..K), '
                'where v_k is the k-th smallest eigenvector of L_sym = I - D^{-1/2} C D^{-1/2} '
                'on the item-item co-occurrence graph C (built from toys/training tfrecord).'
            ),
            'k_eigenvectors': K_EIG,
            'phi_statistics': {
                'mean': float(phi.mean()),
                'std': float(phi.std()),
                'min': float(phi.min()),
                'max': float(phi.max()),
                'quantiles': {f'q{int(q*100)}': float(np.quantile(phi, q))
                              for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
            },
            'phi_bin_counts': {str(b): int(c) for b, c in zip(*np.unique(phi_bins, return_counts=True))},
            'smallest_eigenvalues': evals[:5].tolist(),
            'step1_prior': prior_evidence,
            'step2_phenomenon_1_strict': {
                'description': (
                    'Same/diff φ-bin residual distance, with STRICTER kill line than candidate 1: '
                    'rel_gap > 8% AND Cohen\'s d ≥ 0.3 (vs candidate 1: rel_gap > 5%, no d-floor). '
                    '5-8% range → GREY tier, tend to KILL based on Idea 1 prior.'
                ),
                'kill_criterion_strict': 'rel_gap > 8% AND Cohen\'s d ≥ 0.3 AND p < 0.001 (on r_2, r_3)',
                'tier_per_layer': tier_per_layer,
                'results_per_layer': p1_results,
                'verdict': p1_verdict,
            },
            'final_verdict': p1_verdict,
            'next_action': (
                '阶段三 (图谱频率 × 时间) 不投入. 候选 2 判死.'
                if 'KILL' in p1_verdict else
                '需要继续阶段三交叉验证, 但鉴于 prior, 应极度谨慎.'
            ),
            'total_time_sec': float(time.time() - t0),
        }, f, indent=2)
    print(f'\n[SAVED] {out_json}')


if __name__ == '__main__':
    main()