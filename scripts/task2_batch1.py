#!/usr/bin/env python3
# task4_batch1.py — 11 诊断量批 1：4 个最便宜诊断量
# 量 1: μ_l / σ_l (residual amplitude profile)
# 量 2: erank_l (spectral effective rank)
# 量 3: m_l (water-filling dimension)
# 量 4: f_radial + cell CV (gain-shape decomposition)
#
# Usage: python task4_batch1.py
# 输出: result/task14_<quantity>.json + .png (per algorithm)
#       result/task4_verdict.md
#
# 数据：Toys RKMeans/RKMeans-GSRQ 3-layer，每层 256 codebook

import sys, os, json, types
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

# ---------- 1. Stub GRID modules to bypass circular import ----------
class _Stub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__dict__['__path__'] = []
    def __getattr__(self, name):
        cls = type(name, (), {})
        self.__dict__[name] = cls
        return cls

# Pre-stub all data loading modules that might be triggered by torch.load
for mod in [
    'src.data.loading.components.interfaces',
    'src.data.loading.components.iterators',
    'src.data.loading.components.dataloading',
    'src.data.loading.components.collate_functions',
    'src.data.loading.components.pre_processing',
    'src.data.loading.components.interfaces',
    'src.data.loading.datamodules.sequence_datamodule',
    'src.data.loading.datamodules',
    'src.utils.decorators',
    'src.utils.utils',
    'src.utils',
]:
    if mod not in sys.modules:
        m = _Stub(mod)
        m.__file__ = '/tmp/dummy.py'
        sys.modules[mod] = m

# ---------- 2. Config ----------
EMB_PATH = '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'

ALGORITHMS = {
    'A_baseline': '/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-07/23-51-55/checkpoints/checkpoint_000_003000.ckpt',
    'C_gsrq':     '/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
}

L = 3
W = 256
N, D = 11924, 2048

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 3. Load ckpt + extract codebooks ----------
def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    codebooks = []
    has_gains = False
    gains = []
    for l in range(L):
        key = f'quantization_layer_list.{l}.centroids'
        if key not in state:
            raise KeyError(f'ckpt missing {key}: keys={list(state.keys())[:10]}')
        codebooks.append(state[key].float())
        gain_key = f'quantization_layer_list.{l}.cluster_gains'
        if gain_key in state:
            has_gains = True
            gains.append(state[gain_key].float())
        else:
            gains.append(None)
    return codebooks, gains, has_gains

# ---------- 4. Forward pass: residual decomposition ----------
def forward_residual(x, codebooks, gains, has_gains):
    """x: (N, D) tensor
       returns: r_lst (L+1), q_lst (L), idx_lst (L)
    """
    r_lst = [x.clone()]
    q_lst = []
    idx_lst = []
    for l in range(L):
        r = r_lst[-1]
        C = codebooks[l]                              # (W, D)
        # Effective centroids = C * g if has_gains else C
        if has_gains:
            eff_C = C * gains[l].unsqueeze(-1)        # (W, D)
        else:
            eff_C = C
        # Nearest neighbor (Euclidean) — chunked to avoid (N, W, D) intermediate
        # Direct matrix multiply is faster:
        # d2 = ||r||^2 - 2 r @ C^T + ||C||^2
        r_norm2 = (r ** 2).sum(-1, keepdim=True)      # (N, 1)
        C_norm2 = (eff_C ** 2).sum(-1)                # (W,)
        d2 = r_norm2 - 2 * (r @ eff_C.T) + C_norm2    # (N, W)
        idx = d2.argmin(dim=1)                        # (N,)
        q = eff_C[idx]                                # (N, D)
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
    return r_lst, q_lst, idx_lst

# ---------- 5. Quantities ----------
def compute_layer_norm_profile(r_lst):
    """量 1: μ_l / σ_l (residual amplitude per layer)"""
    mus, sigmas = [], []
    for l in range(L):
        r_norm = r_lst[l].norm(dim=-1)                # (N,)
        mus.append(r_norm.mean().item())
        sigmas.append(r_norm.std().item())
    return {'mu_l': mus, 'sigma_l': sigmas}

def compute_erank(r_lst, sample_n=2000):
    """量 2: erank_l = exp(H(λ̂)) of residual covariance
    在 (N, D) 上直接做 SVD，但只用前 sample_n 个样本（erank 对样本量鲁棒）
    """
    eranks = []
    rng = np.random.default_rng(42)
    for l in range(L):
        r = r_lst[l]                                  # (N, D)
        # Subsample for speed
        if r.shape[0] > sample_n:
            idx = rng.choice(r.shape[0], size=sample_n, replace=False)
            r_sub = r[idx]
        else:
            r_sub = r
        r_centered = r_sub - r_sub.mean(0)
        # SVD on (sample_n, D=2048) is fast
        # Use randomized SVD via torch.svd_lowrank for speed
        try:
            U, S, V = torch.svd_lowrank(r_centered, q=128, niter=2)
            s2 = (S ** 2).clamp(min=1e-12)
        except Exception:
            S = torch.linalg.svdvals(r_centered)
            s2 = (S ** 2).clamp(min=1e-12)
        p = s2 / s2.sum()
        H = -(p * p.clamp(min=1e-12).log()).sum()
        eranks.append(H.exp().item())
    return {'erank_l': eranks}

def compute_water_filling(r_lst, codebook_width=W, sample_n=2000):
    """量 3: m_l = number of eigvals above water-filling threshold θ_l at rate R_l = log2(W)
    在 (N, D) 上做 SVD，只用前 sample_n 个样本。
    """
    ms = []
    rng = np.random.default_rng(42)
    for l in range(L):
        r = r_lst[l]
        if r.shape[0] > sample_n:
            idx = rng.choice(r.shape[0], size=sample_n, replace=False)
            r_sub = r[idx]
        else:
            r_sub = r
        r_centered = r_sub - r_sub.mean(0)
        try:
            U, S, V = torch.svd_lowrank(r_centered, q=128, niter=2)
        except Exception:
            S = torch.linalg.svdvals(r_centered)
        s2 = (S ** 2).clamp(min=1e-12)
        # Capacity = sum(s2) (total variance)
        R = np.log2(codebook_width)
        capacity = s2.sum().item()
        threshold = capacity / (R * 2.0)
        m_l = (s2 > threshold).sum().item()
        ms.append(int(m_l))
    return {'m_l': ms}

def compute_gain_shape_decomp(r_lst, q_lst, idx_lst):
    """量 4: f_radial + cell CV"""
    f_radial_list = []
    cell_cv_list = []
    for l in range(L):
        r = r_lst[l]
        q = q_lst[l]
        idx = idx_lst[l]
        # Magnitudes
        r_norm = r.norm(dim=-1)
        q_norm = q.norm(dim=-1)
        # Radial error (magnitude difference)
        e_r = (r_norm - q_norm).abs()
        # Total squared error
        e_total = ((r - q) ** 2).sum(dim=-1)          # (N,)
        # f_radial = sum(e_r^2) / sum(e_total)
        f_radial_l = (e_r ** 2).sum() / e_total.sum().clamp(min=1e-12)
        f_radial_list.append(f_radial_l.item())
        # Cell CV: for each unique cluster, std(r_norm)/mean(r_norm)
        cvs = []
        for k in idx.unique():
            mask = (idx == k)
            r_in_cell = r_norm[mask]
            if r_in_cell.numel() > 1 and r_in_cell.std() > 0:
                cvs.append((r_in_cell.std() / r_in_cell.mean()).item())
        cell_cv_list.append(float(np.mean(cvs)) if cvs else 0.0)
    return {'f_radial_l': f_radial_list, 'cell_cv_l': cell_cv_list}

# ---------- 6. Main ----------
def main():
    print(f'Loading embedding: {EMB_PATH}')
    x = torch.load(EMB_PATH, weights_only=False)
    assert x.shape == (N, D), f'expected ({N}, {D}), got {x.shape}'
    print(f'  shape={x.shape}, dtype={x.dtype}')

    all_results = {}
    for name, ckpt_path in ALGORITHMS.items():
        print(f'\n=== {name}: {os.path.basename(ckpt_path)} ===')
        if not os.path.exists(ckpt_path):
            print(f'  ckpt not found, skipping')
            continue
        codebooks, gains, has_gains = load_codebooks(ckpt_path)
        print(f'  has_gains={has_gains}, codebook shapes={[c.shape for c in codebooks]}')

        # Forward
        r_lst, q_lst, idx_lst = forward_residual(x, codebooks, gains, has_gains)

        # Save r/q/idx for task17 (as .pt)
        cache_path = os.path.join(OUT_DIR, f'{name}_rqidx.pt')
        torch.save({
            'r_lst': [r.cpu() for r in r_lst],
            'q_lst': [q.cpu() for q in q_lst],
            'idx_lst': [idx.cpu() for idx in idx_lst],
            'codebooks': [c.cpu() for c in codebooks],
            'gains': [g.cpu() if g is not None else None for g in gains],
        }, cache_path)
        print(f'  cached r/q/idx → {cache_path}')

        # Quantities
        q1 = compute_layer_norm_profile(r_lst)
        q2 = compute_erank(r_lst)
        q3 = compute_water_filling(r_lst)
        q4 = compute_gain_shape_decomp(r_lst, q_lst, idx_lst)

        result = {
            'algorithm': name,
            'ckpt': ckpt_path,
            'has_gains': has_gains,
            'layers': []
        }
        for l in range(L):
            result['layers'].append({
                'l': l + 1,
                'mu_l': q1['mu_l'][l],
                'sigma_l': q1['sigma_l'][l],
                'erank_l': q2['erank_l'][l],
                'm_l': q3['m_l'][l],
                'f_radial_l': q4['f_radial_l'][l],
                'cell_cv_l': q4['cell_cv_l'][l],
            })
        all_results[name] = result
        print(f'  q1 (mu_l):  {[round(v, 4) for v in q1["mu_l"]]}')
        print(f'  q2 (erank): {[round(v, 1) for v in q2["erank_l"]]}')
        print(f'  q3 (m_l):   {q3["m_l"]}')
        print(f'  q4 (f_rad): {[round(v, 4) for v in q4["f_radial_l"]]}, cell_cv={[round(v, 4) for v in q4["cell_cv_l"]]}')

    # Save combined results
    combined_path = os.path.join(OUT_DIR, 'task4_all_algorithms.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== Combined JSON → {combined_path} ===')

    # Per-algorithm JSON
    for name, result in all_results.items():
        p = os.path.join(OUT_DIR, f'task14_{name}.json')
        with open(p, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  → {p}')

    # ---------- 7. Plots ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: μ_l / σ_l (all 3 algorithms)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            mus = [r['mu_l'] for r in result['layers']]
            sigmas = [r['sigma_l'] for r in result['layers']]
            ax[0].plot(ls, mus, marker='o', label=name)
            ax[0].fill_between(ls, [m - s for m, s in zip(mus, sigmas)],
                               [m + s for m, s in zip(mus, sigmas)], alpha=0.15)
        ax[0].set_xlabel('layer l'); ax[0].set_ylabel('||r_l||'); ax[0].legend(); ax[0].set_title('量 1: 残差幅度剖面')

        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            eranks = [r['erank_l'] for r in result['layers']]
            ms = [r['m_l'] for r in result['layers']]
            ax[1].plot(ls, eranks, marker='o', label=f'{name} erank')
            ax[1].plot(ls, ms, marker='s', linestyle='--', label=f'{name} m_l')
        ax[1].set_xlabel('layer l'); ax[1].set_ylabel('effective rank'); ax[1].legend(); ax[1].set_title('量 2 + 3: erank_l vs m_l')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'task14_layer_norm_and_rank.png'), dpi=120)
        plt.close()

        # Plot 2: f_radial + cell CV
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            fr = [r['f_radial_l'] for r in result['layers']]
            cv = [r['cell_cv_l'] for r in result['layers']]
            ax[0].plot(ls, fr, marker='o', label=name)
        ax[0].set_xlabel('layer l'); ax[0].set_ylabel('f_radial'); ax[0].legend(); ax[0].set_title('量 4a: 量化误差中径向占比')
        for name, result in all_results.items():
            ls = [r['l'] for r in result['layers']]
            cv = [r['cell_cv_l'] for r in result['layers']]
            ax[1].plot(ls, cv, marker='o', label=name)
        ax[1].set_xlabel('layer l'); ax[1].set_ylabel('cell CV'); ax[1].legend(); ax[1].set_title('量 4b: cell 内幅度变异')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'task14_gain_shape_decomp.png'), dpi=120)
        plt.close()
        print(f'\n=== PNG plots saved to {OUT_DIR} ===')
    except Exception as e:
        print(f'plot failed: {e}')

    # ---------- 8. Verdict ----------
    verdict_lines = ['# Task 18 (Diag11 批 1) 一段话判据\n']
    for name, result in all_results.items():
        verdict_lines.append(f'\n## {name}\n')
        for r in result['layers']:
            l = r['l']
            verdict_lines.append(
                f"- Layer {l}: μ={r['mu_l']:.4f}±{r['sigma_l']:.4f}, "
                f"erank={r['erank_l']:.1f}, m_l={r['m_l']}, "
                f"f_radial={r['f_radial_l']:.4f}, cell CV={r['cell_cv_l']:.4f}"
            )
    verdict_lines.append('\n## 整体判定\n')
    verdict_lines.append('- (待后续分析基于此 JSON + PNG 写出)')
    verdict_path = os.path.join(OUT_DIR, 'task4_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\n=== Verdict → {verdict_path} ===')

if __name__ == '__main__':
    main()