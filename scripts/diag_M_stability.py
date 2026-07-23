#!/usr/bin/env python3
"""Task #5: 跨 M 构造稳定性推广 — test if causal_gain(M1) - causal_gain(M_baseline)
varies systematically across algorithms (differentiation test).

For each algorithm × M ∈ {M1=PPMI-SVD, M2=cooc, M3=graph eigvecs, M4=random proj, M5=identity}:
  project residual r_l through M (low-rank projection if M has fewer rows)
  MI(M·r_l → brand) per layer (residual's remaining brand-info)
  ΔI_V_M_l = MI(M·r_l → brand) - MI(M·r_{l+1} → brand)
  causal_gain_M = sum_l ΔI_V_M_l  (total brand-info "causally dropped" by M projection)
Then:
  stability_score(algo) = std(causal_gain_M across M1..M4) / |mean(causal_gain_M)|

Kill line:
  - variance(stability_score across algos) < 0.1 → not differentiating
  - cluster pattern matches known failure (GSRQ collision high, WF warning high) → valid
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators  # noqa: F401  (break Lightning ckpt circular import)

import os
import json
import numpy as np
import torch
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_M_stability'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
PPMI_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/cf_ppmi_svd256.pt'
COOCC_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate/cooccurrence_csr.npz'
EIGV_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate/eigvecs_k64.npy'

ALGOS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
    'WF_retrained': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt',
}
# Known failure mode color-coding (for scatter)
ALGO_FAILURE_MODE = {
    'A_baseline': 'normal',      # baseline
    'B_mmq':      'normal',      # baseline-like
    'C_gsrq':     'collision',   # 3-layer collision 65% (high)
    'WF_retrained': 'warning_flag',  # reduced K, warning
}

PCA_DIM = 16           # compress projected residuals before MI (cf idea1_WF pattern)
N_NEIGHBORS = 5
N_SAMPLE = 2000         # subsample for MI speed
SEED = 42
TOP_BRANDS = 50         # brand label cardinality cap


def build_brand_labels(N):
    """Return y_brand of shape (N,) with integer labels for top-50 brands, others 'OTHER'."""
    metadata = json.load(open(META_PATH))
    brand_arr = np.array([metadata.get(str(i), {}).get('brand', 'NO_BRAND') for i in range(N)])
    cnt = Counter(brand_arr.tolist())
    top = [b for b, _ in cnt.most_common(TOP_BRANDS) if b != 'NO_BRAND']
    brand_labels = np.array([b if b in top else 'OTHER' for b in brand_arr])
    subs = sorted(set(brand_labels.tolist()))
    brand_to_id = {b: i for i, b in enumerate(subs)}
    y = np.array([brand_to_id[b] for b in brand_labels], dtype=np.int64)
    return y


def fast_mi_bits(x_2d, y, n_sample=N_SAMPLE, seed=SEED, n_neighbors=N_NEIGHBORS):
    """Mean MI bits per feature after PCA-16 reduction + subsample-2000."""
    if x_2d.ndim != 2:
        raise ValueError(f"expected 2D, got {x_2d.shape}")
    if x_2d.shape[1] > PCA_DIM:
        x_2d = PCA(n_components=PCA_DIM, random_state=seed).fit_transform(x_2d)
    N = x_2d.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x_2d[idx], y[idx]
    else:
        x_s, y_s = x_2d, y
    mi_nats = mutual_info_classif(x_s, y_s, n_neighbors=n_neighbors,
                                   random_state=seed, discrete_features=False, n_jobs=4)
    return float(mi_nats.mean() / np.log(2))


def build_M_matrices(N, D, seed=SEED):
    """Construct 5 M matrices (projections over the feature dimension D).

    Each M has shape (m, D) where m ≤ D.  Then u_l = r_l @ M.T  ∈ R^{N, m}.
    If M has fewer rows, it's a low-rank projection of the residual's features.

    M1: PPMI-SVD basis (256, D) — fit PPMI projection of the flan-t5 embedding.
        Since PPMI-SVD lives in item space, build a feature basis by running PCA
        on the (N, D) embedding and taking its components as M1; for the CF
        semantics we instead use the *CF columns*: project through per-item CF vec.
        Use CF feature basis = PCA-256 on f_norm @ f_norm.T projected back, or
        simpler: M1 = f_norm.T @ embedding (256, D), the linear map defined by
        CF item similarity.
    M2: raw co-occurrence rows (256, D) — same trick using C-truncated SVD rows
    M3: graph Laplacian eigenvectors over items — also item-space; map via
        embedding.T @ eigvec.T → (D, 64), then transpose to (64, D)
    M4: random Gaussian (256, D), seed=42
    M5: identity (D, D) — too large, return None (= no projection)
    """
    Ms = {}

    # Load embedding (N, D) to back the item-space bases
    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    print(f'  emb shape = {emb.shape}')

    # M1: CF basis in feature space — linear map defined by f_norm (N, 256).
    # M1 shape = (256, D).  u = r @ M1.T = r @ (emb.T @ f_norm).T = r @ f_norm.T @ emb
    cf = torch.load(PPMI_PATH, map_location='cpu', weights_only=False)
    f_norm = cf['f_normalized'].numpy().astype(np.float32)  # (N, 256)
    M1 = f_norm.T @ emb  # (256, D)
    Ms['M1_PPMI_SVD'] = M1.astype(np.float32)
    print(f'  M1 shape = {Ms["M1_PPMI_SVD"].shape}')

    # M2: raw co-occurrence basis
    coo_data = np.load(COOCC_PATH, allow_pickle=True)
    if 'data' in coo_data.files and 'indices' in coo_data.files and 'indptr' in coo_data.files:
        C = csr_matrix((coo_data['data'], coo_data['indices'], coo_data['indptr']),
                        shape=tuple(coo_data['shape']) if 'shape' in coo_data.files else (N, N))
    else:
        C = csr_matrix(coo_data[list(coo_data.files)[0]])
    print(f'  cooccurrence shape={C.shape}, nnz={C.nnz}, density={C.nnz/C.size:.6f}')
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=256, random_state=seed, n_iter=8)
    u_cooc = svd.fit_transform(C).astype(np.float32)  # (N, 256)
    print(f'  M2 raw cooc SVD explained_var={svd.explained_variance_ratio_.sum():.3f}')
    M2 = u_cooc.T @ emb  # (256, D)
    Ms['M2_raw_cooc'] = M2.astype(np.float32)
    print(f'  M2 shape = {Ms["M2_raw_cooc"].shape}')

    # M3: graph Laplacian eigenvectors (N, k=64).  Lift to feature space: (64, D)
    eigv = np.load(EIGV_PATH).astype(np.float32)  # (N, 64)
    print(f'  M3 eigvecs shape={eigv.shape}')
    M3 = eigv.T @ emb  # (64, D) — feature basis via item-coordinate projection
    Ms['M3_graph_eig'] = M3.astype(np.float32)
    print(f'  M3 shape = {Ms["M3_graph_eig"].shape}')

    # M4: random Gaussian (256, D)
    rng = np.random.default_rng(seed)
    M4 = rng.standard_normal((256, D)).astype(np.float32) / np.sqrt(D)
    Ms['M4_random_gauss'] = M4
    print(f'  M4 shape = {M4.shape}')

    # M5: identity → no projection (return None)
    Ms['M5_identity'] = None

    return Ms


def compute_algo_M_sensitivity(name, rqidx_path, Ms, y_brand):
    """For one algorithm, compute causal_gain for each M.

    Returns dict: M_name -> {'mi_per_layer': [...], 'delta_I_V_per_layer': [...], 'causal_gain': float}
    """
    bundle = torch.load(rqidx_path, map_location='cpu', weights_only=False)
    r_lst = [r.float() for r in bundle['r_lst']]  # list of (N, D)
    N, D = r_lst[0].shape
    n_layers = len(r_lst) - 1  # r[0]=x, r[l+1]=input to layer l+1
    assert N == y_brand.shape[0], f"N mismatch: r={N}, y={y_brand.shape[0]}"

    print(f'\n--- {name} ({n_layers} layers, N={N}, D={D}) ---')

    out = {}
    for M_name, M in Ms.items():
        mi_per_layer = []
        for l in range(n_layers):
            r = r_lst[l + 1]  # input to layer l+1 (post quantization of layers 0..l)
            r_np = r.numpy().astype(np.float32)
            if M is None:
                proj = r_np  # identity, no projection
            else:
                # M: (m, D) — feature projection.  u = r @ M.T → (N, m)
                proj = (r_np @ M.T).astype(np.float32)  # (N, m)
            mi = fast_mi_bits(proj, y_brand)
            mi_per_layer.append(mi)
        delta_per_layer = [mi_per_layer[l] - mi_per_layer[l + 1]
                            for l in range(len(mi_per_layer) - 1)]
        causal_gain = sum(delta_per_layer)  # sum of drops = first-layer MI - last-layer MI
        out[M_name] = {
            'mi_per_layer': mi_per_layer,
            'delta_I_V_per_layer': delta_per_layer,
            'causal_gain': float(causal_gain),
        }
        print(f'  {M_name}: mi_layers={[f"{m:.3f}" for m in mi_per_layer]}, '
              f'causal_gain={causal_gain:+.4f}')
    return out


def main():
    print('=' * 70)
    print('Task #5 — 跨 M 构造稳定性推广')
    print('=' * 70)

    # Build brand labels
    metadata = json.load(open(META_PATH))
    N = max(int(k) for k in metadata.keys()) + 1
    print(f'  N (item universe) = {N}')
    y_brand = build_brand_labels(N)
    cnt = np.bincount(y_brand)
    p = cnt / cnt.sum()
    H_brand = float(-np.sum(p * np.log2(p + 1e-12)))
    print(f'  H(brand_{TOP_BRANDS}) = {H_brand:.4f} bits, n_classes={len(cnt)}')

    # Build M matrices
    print('\nBuilding M matrices:')
    Ms = build_M_matrices(N, D=2048)

    # Compute per-algo × per-M sensitivity
    print('\n=== Per-algo × per-M MI diagnostics ===')
    all_results = {}
    for name, path in ALGOS.items():
        if not os.path.exists(path):
            print(f'  {name}: NOT FOUND ({path})')
            continue
        all_results[name] = compute_algo_M_sensitivity(name, path, Ms, y_brand)

    # Compute stability score per algo
    print('\n=== Stability score per algo ===')
    stability = {}
    for algo, m_dict in all_results.items():
        cg = np.array([m_dict[m]['causal_gain'] for m in ['M1_PPMI_SVD', 'M2_raw_cooc',
                                                          'M3_graph_eig', 'M4_random_gauss']])
        cg_mean = float(cg.mean())
        cg_std = float(cg.std())
        score = cg_std / abs(cg_mean) if abs(cg_mean) > 1e-9 else float('inf')
        stability[algo] = {
            'causal_gains_M1_M4': cg.tolist(),
            'M_baseline_identity': float(m_dict['M5_identity']['causal_gain']),
            'mean': cg_mean,
            'std': cg_std,
            'stability_score': float(score),
            'failure_mode': ALGO_FAILURE_MODE.get(algo, 'unknown'),
        }
        print(f'  {algo}: cg={[f"{c:+.4f}" for c in cg]}, '
              f'M5_id={m_dict["M5_identity"]["causal_gain"]:+.4f}, '
              f'mean={cg_mean:+.4f}, std={cg_std:.4f}, '
              f'stability={score:.3f}, mode={ALGO_FAILURE_MODE.get(algo)}')

    # Verdict
    print('\n=== Verdict ===')
    scores = np.array([stability[a]['stability_score'] for a in all_results])
    if len(scores) >= 2:
        variance_of_scores = float(scores.std())
    else:
        variance_of_scores = 0.0
    print(f'  variance of stability scores across algos = {variance_of_scores:.4f}')
    if variance_of_scores < 0.1:
        kill = '✗ KILL — algos have similar stability_score (variance < 0.1), not differentiating'
    else:
        # Check cluster pattern
        by_mode = {m: [stability[a]['stability_score'] for a in all_results
                       if stability[a]['failure_mode'] == m]
                   for m in set(stability[a]['failure_mode'] for a in all_results)}
        collision_scores = by_mode.get('collision', [])
        warn_scores = by_mode.get('warning_flag', [])
        normal_scores = by_mode.get('normal', [])
        collision_high = (collision_scores and normal_scores and
                          np.mean(collision_scores) > np.mean(normal_scores))
        warn_high = (warn_scores and normal_scores and
                     np.mean(warn_scores) > np.mean(normal_scores))
        if collision_high or warn_high:
            kill = (f'✓ PASS — cluster pattern aligns with known failure modes '
                    f'(collision_high={collision_high}, warn_high={warn_high})')
        else:
            kill = ('⚠ WEAK — variance > 0.1 but cluster pattern does not align '
                    'with known failure types')

    print(f'  KILL LINE: {kill}')

    # Save CSV
    import csv
    csv_path = os.path.join(OUT_DIR, 'per_algo_M_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algo', 'M', 'mi_layer1', 'mi_layer2', 'mi_layer3',
                    'delta_I_V_12', 'delta_I_V_23', 'causal_gain',
                    'stability_score', 'failure_mode'])
        for algo in all_results:
            m_dict = all_results[algo]
            score = stability[algo]['stability_score']
            mode = stability[algo]['failure_mode']
            for M_name in m_dict:
                rec = m_dict[M_name]
                mi = rec['mi_per_layer']
                d = rec['delta_I_V_per_layer']
                w.writerow([algo, M_name,
                            f'{mi[0]:.4f}' if len(mi) > 0 else '',
                            f'{mi[1]:.4f}' if len(mi) > 1 else '',
                            f'{mi[2]:.4f}' if len(mi) > 2 else '',
                            f'{d[0]:.4f}' if len(d) > 0 else '',
                            f'{d[1]:.4f}' if len(d) > 1 else '',
                            f'{rec["causal_gain"]:.4f}',
                            f'{score:.4f}', mode])
    print(f'  saved → {csv_path}')

    # Save scatter PNG
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        color_map = {'normal': 'tab:blue', 'collision': 'tab:red',
                     'warning_flag': 'tab:orange'}
        marker_map = {'normal': 'o', 'collision': 'X', 'warning_flag': '^'}

        fig, ax = plt.subplots(figsize=(8, 6))
        for algo in all_results:
            score = stability[algo]['stability_score']
            mode = stability[algo]['failure_mode']
            ax.scatter(score, stability[algo]['mean'],
                       s=120, c=color_map.get(mode, 'gray'),
                       marker=marker_map.get(mode, 'o'),
                       edgecolors='black', linewidths=0.8, label=mode if mode not in ax.get_legend_handles_labels()[1] else None)
            ax.annotate(algo, (score, stability[algo]['mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('stability_score = std(cg_M1..M4) / |mean(cg_M1..M4)|')
        ax.set_ylabel('mean(causal_gain) across M1..M4')
        ax.set_title(f'Task #5: Cross-M Stability  (kill={kill.split()[0]})')
        ax.grid(True, alpha=0.3)
        # legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker=marker_map[m], color='w',
                                   markerfacecolor=color_map[m], markersize=12, label=m)
                            for m in color_map]
        ax.legend(handles=legend_elements, loc='best')
        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, 'scatter_algo_stability.png')
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f'  saved → {png_path}')
    except Exception as e:
        print(f'  WARN: plotting failed: {e}')

    # Save verdict.md
    md_path = os.path.join(OUT_DIR, 'verdict.md')
    with open(md_path, 'w') as f:
        f.write(f'# Task #5 — 跨 M 构造稳定性推广\n\n')
        f.write(f'## Kill line outcome\n{kill}\n\n')
        f.write(f'## Per-algo stability summary\n\n')
        f.write(f'| algo | mean(cg_M1..M4) | std | stability_score | failure_mode |\n')
        f.write(f'|------|-----------------|-----|-----------------|--------------|\n')
        for algo, s in stability.items():
            f.write(f'| {algo} | {s["mean"]:+.4f} | {s["std"]:.4f} | '
                    f'{s["stability_score"]:.3f} | {s["failure_mode"]} |\n')
        f.write(f'\n## Per-algo × per-M causal_gain table\n\n')
        f.write(f'| algo | M1_PPMI_SVD | M2_raw_cooc | M3_graph_eig | M4_random_gauss | M5_identity |\n')
        f.write(f'|------|-------------|-------------|--------------|-----------------|-------------|\n')
        for algo in all_results:
            m_dict = all_results[algo]
            row = [algo] + [f'{m_dict[m]["causal_gain"]:+.4f}'
                            for m in ['M1_PPMI_SVD', 'M2_raw_cooc',
                                       'M3_graph_eig', 'M4_random_gauss',
                                       'M5_identity']]
            f.write(f'| {" | ".join(row)} |\n')
        f.write(f'\n## Variance of stability scores across algos = {variance_of_scores:.4f}\n')
    print(f'  saved → {md_path}')

    # Save full JSON
    json_path = os.path.join(OUT_DIR, 'all_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'H_brand': H_brand,
            'top_brands': TOP_BRANDS,
            'n_classes': int(len(cnt)),
            'stability': stability,
            'kill_line': kill,
            'variance_of_scores': variance_of_scores,
        }, f, indent=2)
    print(f'  saved → {json_path}')


if __name__ == '__main__':
    main()