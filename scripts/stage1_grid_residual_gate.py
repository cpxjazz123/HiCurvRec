#!/usr/bin/env python3
"""阶段一 gate: 类目深度 × facet 网格 vs 深层残差相关性检验
====================================================================

三现象, 每个有明确 kill 线:
  现象1: 同/跨网格深层残差距离对比 (总闸)
  现象2: 深度轴 g 与 facet 轴 f 独立性检验 (排除 f-only 简化)
  现象3: 网格坐标 V-info 上限 (判候选1是否有实际价值)

数据来源 (全部复用, 不重训):
  - Stage 1 embedding: 11924 × 2048
  - task11_v2_dense_ckpts_v2 checkpoint: 4 层 RQ-VAE codebook (256 × 2048)
  - toys metadata: cat_top, cat_sub, cats[], brand

网格坐标 (g_i, f_i):
  - g_i = cats[] 路径深度分桶: {1, 2, 3, 4, 5+}
  - f_i = cat_sub label (24 类, 平衡, 已知干净)

深层残差 r_2, r_3:
  - 用 task13 checkpoint 4 层 codebook 对每个 item 做前向
  - r_l = r_{l-1} - codebook_l[c_l]
  - 我们取 l=2, 3 作为"深层残差"
"""
import os, sys, json, time, pickle
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

# Make GRID root importable so torch.load can unpickle `src.*` classes
GRID_ROOT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
if GRID_ROOT not in sys.path:
    sys.path.insert(0, GRID_ROOT)

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage1_grid_gate'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
CKPT_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task1_v2_dense_ckpts_v2/checkpoints/checkpoint_000_001600.ckpt'

N_SAMPLE_PAIRS = 5000     # 每组 pair 采样数
SEED = 42
N_HIERARCHIES = 4         # task13 用 4 层
N_CLUSTERS = 256


class _DummyClass:
    """Stand-in for any class that can't be imported (used during unpickle)."""
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass
    def __getstate__(self): return self.__dict__


def _load_checkpoint_forgiving(path):
    """Load a torch checkpoint, substituting dummy classes for anything we can't import.

    Uses torch's own pickle_module (which has the persistent_load that handles tensor
    storage references in ZIP-format checkpoints), but monkey-patches find_class to
    return a dummy on import failures. This bypasses the circular import in
    src.data.loading.components.interfaces.
    """
    import importlib
    pickle_module = torch.serialization.pickle

    # Build a subclass of torch's Unpickler with forgiving find_class
    class _ForgivingUnpickler(pickle_module.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError, ModuleNotFoundError):
                return _DummyClass

    # Wrap torch.load: monkey-patch its pickle_module.Unpickler with our forgiving version.
    # Easier route: call _legacy_load directly with our unpickler class.
    #
    # torch.serialization._legacy_load takes (file_like, pickle_module, ...)
    # and uses pickle_module.Unpickler internally.

    # Use torch.load with a custom pickle_module where Unpickler is our forgiving class.
    class _CustomPickleModule:
        Unpickler = _ForgivingUnpickler
        # Keep other attributes torch.serialization may need
        def __getattr__(self, name):
            return getattr(pickle_module, name)

    return torch.load(
        path,
        map_location='cpu',
        pickle_module=_CustomPickleModule(),
        weights_only=False,
    )


def load_codebooks():
    """Load 4-layer codebook from task13 checkpoint via forgiving unpickler.

    The full checkpoint contains pickled Lightning objects whose classes depend on
    a circular import in `src.data.loading.components.interfaces`. We don't need
    those objects — only the model state_dict — so we substitute dummy classes for
    anything we can't import.
    """
    ckpt = _load_checkpoint_forgiving(CKPT_PATH)
    sd = ckpt['state_dict']
    codebooks = []
    for l in range(N_HIERARCHIES):
        key = f'quantization_layer_list.{l}.centroids'
        if key not in sd:
            raise ValueError(f'codebook layer {l} not in checkpoint, available keys (first 5): {list(sd.keys())[:5]}')
        codebooks.append(sd[key].float())  # (256, 2048)
    print(f'  loaded {len(codebooks)} codebooks, each shape {codebooks[0].shape}')
    return codebooks


def forward_residuals(X, codebooks):
    """Forward pass: assign codes + compute residuals for each layer.

    For each layer l:
      - Find nearest centroid for each item
      - r_l = r_{l-1} - centroid_l[c_l]

    Returns:
      residuals: list of (N, D) for r_1, r_2, ..., r_N_HIERARCHIES
      codes: list of (N,) for c_1, ..., c_N_HIERARCHIES
    """
    N = X.shape[0]
    residuals = []
    codes = []
    r = X.clone()
    for l, cb in enumerate(codebooks):
        # Distance to each centroid: (N, K)
        # dist[i, k] = ||r_i - cb_k||²
        dist = torch.cdist(r, cb)  # (N, K)
        c = dist.argmin(dim=1)  # (N,)
        r_l = r - cb[c]  # (N, D)
        residuals.append(r_l)
        codes.append(c)
        r = r_l
    return residuals, codes


def build_grid_coords(metadata):
    """Build (g_i, f_i) grid coordinates.

    g_i: depth bucket of cats[] path
        1 -> cats depth 1
        2 -> cats depth 2
        3 -> cats depth 3 (most common: 7521 items)
        4 -> cats depth 4
        5 -> cats depth >= 5

    f_i: cat_sub label (24 classes)
    """
    N = len(metadata)
    g = np.zeros(N, dtype=np.int64)
    f_str = [None] * N
    f_idx = np.zeros(N, dtype=np.int64)
    cat_sub_set = sorted(set(metadata[str(i)]['cat_sub'] for i in range(N)))
    cat_sub_to_id = {c: i for i, c in enumerate(cat_sub_set)}

    depth_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for i in range(N):
        d = len(metadata[str(i)]['cats'])
        if d >= 5:
            g[i] = 5
            depth_count[5] += 1
        else:
            g[i] = d
            depth_count[d] += 1
        f_str[i] = metadata[str(i)]['cat_sub']
        f_idx[i] = cat_sub_to_id[f_str[i]]
    print(f'  depth distribution: {depth_count}')
    print(f'  cat_sub classes: {len(cat_sub_set)}')
    return g, f_idx, cat_sub_set


def cohen_d(x, y):
    """Cohen's d for two samples."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (y.mean() - x.mean()) / pooled if pooled > 0 else 0.0


# =====================================================================
# 现象 1: 同/跨网格深层残差距离对比
# =====================================================================
def phenomenon_1(residuals, g, f):
    print('\n' + '=' * 70)
    print('现象 1: 同/跨网格深层残差距离对比 (总闸)')
    print('=' * 70)

    rng = np.random.default_rng(SEED)
    N = len(g)
    # Build a single grid-id = g * (n_facets) + f
    n_f = int(f.max()) + 1
    grid_id = g * n_f + f  # unique id per (g, f) cell
    grid_set = set(grid_id.tolist())
    print(f'  N items = {N}, n_grid_cells = {len(grid_set)}')

    # Sample P_same: pairs in same grid cell
    p_same_idx_i = []
    p_same_idx_j = []
    attempts = 0
    while len(p_same_idx_i) < N_SAMPLE_PAIRS and attempts < N_SAMPLE_PAIRS * 100:
        i = rng.integers(0, N)
        # find another item in same grid cell
        cell = grid_id[i]
        candidates = np.where(grid_id == cell)[0]
        if len(candidates) >= 2:
            j = rng.choice(candidates[candidates != i]) if (candidates != i).any() else candidates[0]
            p_same_idx_i.append(i)
            p_same_idx_j.append(int(j))
        attempts += 1
    p_same_i = np.array(p_same_idx_i)
    p_same_j = np.array(p_same_idx_j)
    print(f'  P_same samples = {len(p_same_i)}')

    # Sample P_diff: pairs in different grid cells (matching distribution)
    p_diff_idx_i = []
    p_diff_idx_j = []
    attempts = 0
    while len(p_diff_idx_i) < len(p_same_i) and attempts < len(p_same_i) * 100:
        i, j = rng.integers(0, N, size=2)
        if grid_id[i] != grid_id[j]:
            p_diff_idx_i.append(int(i))
            p_diff_idx_j.append(int(j))
        attempts += 1
    p_diff_i = np.array(p_diff_idx_i)
    p_diff_j = np.array(p_diff_idx_j)
    print(f'  P_diff samples = {len(p_diff_i)}')

    results = {}
    for layer_idx, layer_name in [(1, 'r_1'), (2, 'r_2'), (3, 'r_3')]:
        if layer_idx >= len(residuals):
            continue
        r_l = residuals[layer_idx]
        # Compute pairwise distances
        d_same = (r_l[p_same_i] - r_l[p_same_j]).norm(dim=-1).numpy()
        d_diff = (r_l[p_diff_i] - r_l[p_diff_j]).norm(dim=-1).numpy()
        # Mann-Whitney U (one-sided: d_same < d_diff)
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
            'd_same_std': float(d_same.std()),
            'd_diff_mean': float(d_diff.mean()),
            'd_diff_std': float(d_diff.std()),
            'relative_gap_pct': float(rel_gap * 100),
            'cohens_d': float(d_eff),
            'mannwhitney_u': float(u_stat),
            'mannwhitney_p': float(p_val),
        }

    # Apply kill line: ALL layers must have rel_gap > 5% AND p < 0.001
    all_pass = all(
        v['relative_gap_pct'] > 5 and v['mannwhitney_p'] < 0.001
        for v in results.values()
    )
    print(f'\n  Kill 判定: rel_gap > 5% 且 p < 0.001 (L2, L3)')
    print(f'  {"LAYER":<8} {"rel_gap":>10} {"p-val":>12} {"pass?":>6}')
    for k, v in results.items():
        ok = v['relative_gap_pct'] > 5 and v['mannwhitney_p'] < 0.001
        print(f'  {k:<8} {v["relative_gap_pct"]:>+9.2f}% {v["mannwhitney_p"]:>12.4e} {"✓" if ok else "✗":>6}')
    if all_pass:
        verdict = 'PASS: 网格与深层残差空间显著相关, 进入现象 2 验证深度轴独立性'
    else:
        verdict = (
            'KILL_CANDIDATE_1: 网格坐标与深层残差空间没有显著相关性, '
            '候选1判死. 不需要继续 patchify 机制投入.'
        )
    print(f'\n  >>> {verdict}')
    return all_pass, results, verdict


# =====================================================================
# 现象 2: 深度轴 g 与 facet 轴 f 独立性检验
# =====================================================================
def phenomenon_2(residuals, g, f):
    print('\n' + '=' * 70)
    print('现象 2: 深度轴 g 与 facet 轴 f 独立性检验')
    print('=' * 70)

    rng = np.random.default_rng(SEED + 1)
    N = len(g)
    n_f = int(f.max()) + 1

    # D_same_g_diff_f: same depth (g_i = g_j), different facet (f_i ≠ f_j)
    p_sg_df_i, p_sg_df_j = [], []
    attempts = 0
    while len(p_sg_df_i) < N_SAMPLE_PAIRS and attempts < N_SAMPLE_PAIRS * 100:
        i = rng.integers(0, N)
        # find another item in same depth but different facet
        same_g = (g == g[i])
        diff_f = (f != f[i])
        candidates = np.where(same_g & diff_f)[0]
        if len(candidates) > 0:
            j = rng.choice(candidates)
            p_sg_df_i.append(int(i))
            p_sg_df_j.append(int(j))
        attempts += 1

    # D_diff_g_same_f: different depth (g_i ≠ g_j), same facet (f_i = f_j)
    p_dg_sf_i, p_dg_sf_j = [], []
    attempts = 0
    while len(p_dg_sf_i) < N_SAMPLE_PAIRS and attempts < N_SAMPLE_PAIRS * 100:
        i = rng.integers(0, N)
        diff_g = (g != g[i])
        same_f = (f == f[i])
        candidates = np.where(diff_g & same_f)[0]
        if len(candidates) > 0:
            j = rng.choice(candidates)
            p_dg_sf_i.append(int(i))
            p_dg_sf_j.append(int(j))
        attempts += 1

    # Random baseline: complete random pairs
    p_ran_i = rng.integers(0, N, size=N_SAMPLE_PAIRS)
    p_ran_j = rng.integers(0, N, size=N_SAMPLE_PAIRS)

    p_sg_df_i = np.array(p_sg_df_i)
    p_sg_df_j = np.array(p_sg_df_j)
    p_dg_sf_i = np.array(p_dg_sf_i)
    p_dg_sf_j = np.array(p_dg_sf_j)
    p_ran_i = np.array(p_ran_i)
    p_ran_j = np.array(p_ran_j)

    print(f'  P_same_g_diff_f samples = {len(p_sg_df_i)}')
    print(f'  P_diff_g_same_f samples = {len(p_dg_sf_i)}')
    print(f'  P_random samples       = {len(p_ran_i)}')

    results = {}
    for layer_idx, layer_name in [(2, 'r_2'), (3, 'r_3')]:
        if layer_idx >= len(residuals):
            continue
        r_l = residuals[layer_idx]
        d_sg_df = (r_l[p_sg_df_i] - r_l[p_sg_df_j]).norm(dim=-1).numpy()
        d_dg_sf = (r_l[p_dg_sf_i] - r_l[p_dg_sf_j]).norm(dim=-1).numpy()
        d_ran = (r_l[p_ran_i] - r_l[p_ran_j]).norm(dim=-1).numpy()

        # Gap of "same_g diff_f" vs random
        rel_gap_g = (d_ran.mean() - d_sg_df.mean()) / max(d_ran.mean(), 1e-9)
        # Gap of "diff_g same_f" vs random
        rel_gap_f = (d_ran.mean() - d_dg_sf.mean()) / max(d_ran.mean(), 1e-9)
        # Difference between the two — if similar, both axes contribute equally
        # if d_sg_df ≈ d_ran, g-axis has no info beyond f
        print(f'\n  --- {layer_name} ---')
        print(f'    D(same g, diff f) = {d_sg_df.mean():.4f}  (rel_gap vs random = {rel_gap_g * 100:+.2f}%)')
        print(f'    D(diff g, same f) = {d_dg_sf.mean():.4f}  (rel_gap vs random = {rel_gap_f * 100:+.2f}%)')
        print(f'    D(random)         = {d_ran.mean():.4f}')

        # Mann-Whitney: is d_sg_df significantly smaller than d_ran?
        u1, p1 = mannwhitneyu(d_sg_df, d_ran, alternative='less')
        # Is d_dg_sf significantly smaller than d_ran?
        u2, p2 = mannwhitneyu(d_dg_sf, d_ran, alternative='less')
        print(f'    D(same_g diff_f) < D(rand)? U = {u1:.0f}, p = {p1:.4e}')
        print(f'    D(diff_g same_f) < D(rand)? U = {u2:.0f}, p = {p2:.4e}')

        results[layer_name] = {
            'd_same_g_diff_f_mean': float(d_sg_df.mean()),
            'd_diff_g_same_f_mean': float(d_dg_sf.mean()),
            'd_random_mean': float(d_ran.mean()),
            'rel_gap_g_pct': float(rel_gap_g * 100),
            'rel_gap_f_pct': float(rel_gap_f * 100),
            'mw_p_same_g_diff_f': float(p1),
            'mw_p_diff_g_same_f': float(p2),
        }

    # Kill line: BOTH d_sg_df < d_ran AND d_dg_sf < d_ran, both with p < 0.001
    # meaning BOTH axes contribute info
    all_pass = all(
        v['mw_p_same_g_diff_f'] < 0.001 and v['mw_p_diff_g_same_f'] < 0.001
        and v['rel_gap_g_pct'] > 3 and v['rel_gap_f_pct'] > 3
        for v in results.values()
    )
    print(f'\n  Kill 判定: 两个轴都 < random (p<0.001, rel_gap>3%)')
    print(f'  {"LAYER":<8} {"g-axis rel_gap":>16} {"f-axis rel_gap":>16} {"pass?":>6}')
    for k, v in results.items():
        ok_g = v['mw_p_same_g_diff_f'] < 0.001 and v['rel_gap_g_pct'] > 3
        ok_f = v['mw_p_diff_g_same_f'] < 0.001 and v['rel_gap_f_pct'] > 3
        ok = ok_g and ok_f
        print(f'  {k:<8} {v["rel_gap_g_pct"]:>+15.2f}% {v["rel_gap_f_pct"]:>+15.2f}% {"✓" if ok else "✗":>6}')
    if all_pass:
        verdict = 'PASS: 深度轴和 facet 轴各自独立贡献, 二维结构成立'
    else:
        # Check which axis fails
        g_fails = any(v['mw_p_same_g_diff_f'] >= 0.001 or v['rel_gap_g_pct'] <= 3
                      for v in results.values())
        if g_fails:
            verdict = (
                'FAIL_1D: 深度轴 g 没有独立信息 (D_same_g_diff_f ≈ D_random), '
                '网格退化成 facet 一维. 候选1的"二维"卖点不成立, 但 facet 维度仍可能有效, '
                '可退回更简单的一维方向 (不放弃整个 idea, 但降级).'
            )
        else:
            verdict = 'INSUFFICIENT: 两个轴都没有显著独立贡献, 候选1判死'
    print(f'\n  >>> {verdict}')
    return all_pass, results, verdict


# =====================================================================
# 现象 3: 网格坐标 V-info 上限
# =====================================================================
def phenomenon_3(g, f, metadata):
    print('\n' + '=' * 70)
    print('现象 3: 网格坐标 V-info 上限 ΔI_V^grid')
    print('=' * 70)

    # Pick TWO Y targets that grid can predict:
    # Y1: brand (top-50 brands, multi-class)
    # Y2: popularity bucket (cold/warm/hot — from earlier diag)

    # === Y1: brand (top 30 brands) ===
    brand_str = [metadata[str(i)]['brand'] for i in range(len(metadata))]
    from collections import Counter
    brand_counter = Counter(brand_str)
    top_brands = [b for b, _ in brand_counter.most_common(30)]
    brand_to_id = {b: i for i, b in enumerate(top_brands)}
    y1 = np.array([brand_to_id.get(b, 29) for b in brand_str])  # 29 = "other"
    # but we want clean labels — let's filter to top-30 + a special "other" class
    print(f'  Y1 = brand: top-30 + other, {len(top_brands)} + 1 = {len(set(y1))} effective classes')

    # === Y2: popularity bucket ===
    # Need popularity data — load from earlier cache if exists
    POP_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag2_ollivier_ricci/toys_item_popularity.npy'
    if os.path.exists(POP_PATH):
        pop = np.load(POP_PATH)
        pop_q33 = np.percentile(pop[pop > 0], 33)
        pop_q67 = np.percentile(pop[pop > 0], 67)
        y2 = np.zeros(len(pop), dtype=np.int64)  # 0=cold, 1=warm, 2=hot
        y2[(pop > pop_q33) & (pop < pop_q67)] = 1
        y2[pop >= pop_q67] = 2
        print(f'  Y2 = popularity: cold={int((y2==0).sum())}, warm={int((y2==1).sum())}, hot={int((y2==2).sum())}')
    else:
        y2 = None
        print(f'  Y2 = popularity: not available, skip')

    # Build features: (g, f) one-hot encoded
    n_g = int(g.max()) + 1
    n_f = int(f.max()) + 1
    X_grid = np.column_stack([g, f])  # (N, 2)

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_onehot = enc.fit_transform(X_grid)  # (N, n_g + n_f)
    print(f'  X_grid one-hot: {X_onehot.shape}')

    rng = np.random.default_rng(SEED + 2)
    perm = rng.permutation(len(g))
    n_train = int(0.8 * len(g))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def fit_and_score(Y, name, n_classes):
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # Train logistic regression
        clf = LogisticRegression(max_iter=500, multi_class='multinomial', n_jobs=-1,
                                  solver='lbfgs', C=1.0)
        clf.fit(X_onehot[train_idx], Y_train)
        proba = clf.predict_proba(X_onehot[test_idx])
        # Clip to avoid log(0)
        proba = np.clip(proba, 1e-9, 1.0)
        ce = log_loss(Y_test, proba, labels=list(range(n_classes)))
        # Entropy baseline
        p_prior = np.bincount(Y, minlength=n_classes) / len(Y)
        p_prior = np.clip(p_prior, 1e-9, 1.0)
        h_y = -np.sum(p_prior * np.log2(p_prior))
        # ΔI_V ≈ H(Y) - CE_loss (in bits)
        delta_iv = h_y - ce / np.log(2)
        # Cross-check: accuracy
        acc = clf.score(X_onehot[test_idx], Y_test)
        print(f'  {name}: H(Y) = {h_y:.4f} bits, CE_loss = {ce:.4f} nats '
              f'= {ce / np.log(2):.4f} bits')
        print(f'  ΔI_V^{{grid}} ({name}) = {delta_iv:.4f} bits, accuracy = {acc:.4f}')
        return {
            'h_y_bits': float(h_y),
            'ce_loss_bits': float(ce / np.log(2)),
            'delta_iv_bits': float(delta_iv),
            'accuracy': float(acc),
        }

    results = {}
    results['Y_brand'] = fit_and_score(y1, 'Y=brand', 31)
    if y2 is not None:
        results['Y_pop_bucket'] = fit_and_score(y2, 'Y=pop-bucket', 3)

    # Kill line: at least ONE Y target has ΔI_V^grid > 0.1 bit
    any_pass = any(v['delta_iv_bits'] > 0.1 for v in results.values())
    print(f'\n  Kill 判定: 至少一个 Y 的 ΔI_V^grid > 0.1 bit')
    print(f'  {"TARGET":<15} {"ΔI_V (bits)":>14} {"pass?":>6}')
    for k, v in results.items():
        ok = v['delta_iv_bits'] > 0.1
        print(f'  {k:<15} {v["delta_iv_bits"]:>14.4f} {"✓" if ok else "✗":>6}')
    if any_pass:
        verdict = (
            'PASS: 网格坐标携带可观测的下游信息 (>0.1 bit), '
            '候选1有实际价值. 进入阶段二 (图谱频率 vs cat_sub 比较).'
        )
    else:
        verdict = (
            'KILL_NO_DOWNSTREAM_VALUE: 即使网格与残差空间相关 (现象1 pass), '
            '这个相关性也不携带下游有用的信息. 候选1判死.'
        )
    print(f'\n  >>> {verdict}')
    return any_pass, results, verdict


# =====================================================================
# MAIN
# =====================================================================
def main():
    print('=' * 70)
    print('阶段一 gate: 类目深度 × facet 网格 vs 深层残差')
    print('=' * 70)

    print('\n[Step 1] 加载 Stage 1 embedding')
    x = torch.load(EMB_PATH, weights_only=False, map_location='cpu')
    if hasattr(x, 'samples'):
        emb_list = [s['embeddings'] for s in x.samples]
        x_eu = torch.cat(emb_list, dim=0).float()
    else:
        x_eu = x.float()
    N, D = x_eu.shape
    print(f'  X: {N} × {D}')

    print('\n[Step 2] 加载 4-layer codebook (task11_v2_dense_ckpts_v2)')
    codebooks = load_codebooks()

    print('\n[Step 3] 前向算 residuals r_1..r_4')
    residuals, codes = forward_residuals(x_eu, codebooks)
    for l, r in enumerate(residuals):
        norm = r.norm(dim=-1).mean().item()
        print(f'  r_{l + 1}: mean ‖r‖ = {norm:.4f}')

    print('\n[Step 4] 构建网格坐标 (g, f)')
    with open(META_PATH) as f_meta:
        metadata = json.load(f_meta)
    g, f, cat_sub_set = build_grid_coords(metadata)

    # Run all 3 phenomena
    p1_pass, p1_results, p1_verdict = phenomenon_1(residuals, g, f)
    p2_pass, p2_results, p2_verdict = (None, None, 'SKIP — 现象 1 已 KILL')
    p3_pass, p3_results, p3_verdict = (None, None, 'SKIP — 现象 1 已 KILL')

    if p1_pass:
        p2_pass, p2_results, p2_verdict = phenomenon_2(residuals, g, f)
        if p2_pass:
            p3_pass, p3_results, p3_verdict = phenomenon_3(g, f, metadata)
        else:
            p3_pass, p3_results, p3_verdict = (
                None, None,
                'SKIP — 现象 2 失败 (网格退化为 1D, 网格本身仍可作 facet 简化方向但不升级到候选 2)'
            )
    overall_pass = p1_pass and (p2_pass is not False) and (p3_pass is not False)

    # Save
    out_json = os.path.join(OUT_DIR, 'stage1_gate.json')
    with open(out_json, 'w') as f_out:
        json.dump({
            'stage': '1 (gate, free)',
            'phenomena': {
                'phenomenon_1': {
                    'description': '同/跨网格深层残差距离对比 (Mann-Whitney U + Cohen\'s d)',
                    'kill_criterion': (
                        'r_2, r_3 上的 rel_gap > 5% AND p < 0.001; '
                        '否则候选1 KILL, 不投入 patchify 实现'
                    ),
                    'pass': p1_pass,
                    'verdict': p1_verdict,
                    'results': p1_results,
                },
                'phenomenon_2': {
                    'description': '深度轴 g 与 facet 轴 f 独立性检验',
                    'kill_criterion': (
                        'BOTH D_same_g_diff_f < D_random (p<0.001, rel_gap>3%) AND '
                        'D_diff_g_same_f < D_random (p<0.001, rel_gap>3%); '
                        '若 g-axis 失败 → 网格退化为 1D'
                    ),
                    'pass': p2_pass,
                    'verdict': p2_verdict,
                    'results': p2_results,
                },
                'phenomenon_3': {
                    'description': '网格坐标 V-info 上限 ΔI_V^grid',
                    'kill_criterion': (
                        '至少一个 Y target (brand / pop-bucket) 的 ΔI_V^grid > 0.1 bit; '
                        '否则网格相关性无下游价值'
                    ),
                    'pass': p3_pass,
                    'verdict': p3_verdict,
                    'results': p3_results,
                },
            },
            'overall_pass': overall_pass,
            'next_action': (
                '阶段二 (图谱频率 vs cat_sub facet) 仅在阶段一全部 PASS 才执行'
                if overall_pass else '阶段一未通过, 不投入阶段二/三'
            ),
        }, f_out, indent=2, default=str)
    print(f'\n[SAVED] {out_json}')

    return overall_pass


if __name__ == '__main__':
    main()
