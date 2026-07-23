#!/usr/bin/env python3
"""Stage 0 + Stage 1 统一实验

Stage 0: 准备 D_tax, D_trans, D_residual_l 三组对齐距离矩阵（n=2000 商品）
Stage 1: partial Mantel test (999 perm) + 回归残差交叉验证

判定:
  - partial ρ 保留 ≥ 50% 强度且 p<0.05 → 独立 → 进入 Stage 2
  - partial ρ 下降 > 70% → 是 taxonomy 影子 → END (L1 纯双曲足够)
  - 30-70% 区间 → 部分重叠 → 进入 Stage 2 但记录重叠度
"""

import os
import sys
import json
import argparse

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
# 先注册 src 路径再导入 torch,避免 ckpt 反序列化碰到循环 import
import src.utils.decorators  # noqa: F401

import time
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

# NOTE: tensorflow 延迟到函数内导入,避免顶层加载扰乱 torch.load 反序列化

OUT_DIR = f'{GRID}/result/task380_stage0_1_partial_mantel'
os.makedirs(OUT_DIR, exist_ok=True)

# 复用 task351 forward_residual
def forward_residual(x, codebooks, gains=None):
    r_lst = [x.clone()]
    q_lst, idx_lst = [], []
    L = len(codebooks)
    for l in range(L):
        r = r_lst[-1]
        C = codebooks[l]
        if gains is not None and gains[l] is not None:
            eff_C = C * gains[l].unsqueeze(-1)
        else:
            eff_C = C
        r_norm2 = (r ** 2).sum(-1, keepdim=True)
        C_norm2 = (eff_C ** 2).sum(-1)
        d2 = r_norm2 - 2 * (r @ eff_C.T) + C_norm2
        idx = d2.argmin(dim=1)
        q = eff_C[idx]
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
    return r_lst, q_lst, idx_lst


def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    L = sum(1 for k in sd.keys()
            if k.startswith('quantization_layer_list.') and k.endswith('.centroids'))
    codebooks = [sd[f'quantization_layer_list.{l}.centroids'].float() for l in range(L)]
    gains = []
    for l in range(L):
        gk = f'quantization_layer_list.{l}.cluster_gains'
        if gk in sd:
            gains.append(sd[gk].float())
        else:
            gains.append(None)
    return codebooks, gains, L


# ========== Distance Helpers ==========

def upper_tri(m):
    n = m.shape[0]
    return m[np.triu_indices(n, k=1)]


def mantel_pearson(D1, D2):
    return float(pearsonr(upper_tri(D1), upper_tri(D2))[0])


def partial_mantel_perm(D_A, D_B, D_C, n_perm=999, seed=42):
    """Partial Mantel permutation test (regression-residual approach).

    标准方法 (Legendre 2000, Smouse 1986 改进版):
      1) 线性回归 vec(A) ~ vec(C) → A_resid
      2) 线性回归 vec(B) ~ vec(C) → B_resid
      3) r_obs = Pearson(A_resid, B_resid)
      4) permutation: 置换 B_resid 行序(只置换 B 的上三角),重算 Pearson r*
      5) p = (count of |r*| ≥ |r_obs| + 1) / (n_perm + 1)

    这种实现避免了"同步置换 A 和 B"导致 partial ρ 不变的陷阱。
    """
    rng = np.random.default_rng(seed)
    n = D_A.shape[0]

    v_A = upper_tri(D_A)
    v_B = upper_tri(D_B)
    v_C = upper_tri(D_C)

    # 1+2: 计算 A_resid 和 B_resid
    def make_resid(target, control):
        slope, intercept = np.polyfit(control, target, 1)
        return target - (slope * control + intercept)

    A_resid = make_resid(v_A, v_C)
    B_resid = make_resid(v_B, v_C)

    # 3: 观察 partial r
    obs, _ = pearsonr(A_resid, B_resid)

    # 4: permutation distribution (置换 B_resid 的行,保持 A_resid 原序)
    m = len(B_resid)
    ge_count = 0
    abs_obs = abs(obs)
    for _ in range(n_perm):
        perm = rng.permutation(m)
        B_perm = B_resid[perm]
        r_p, _ = pearsonr(A_resid, B_perm)
        if abs(r_p) >= abs_obs:
            ge_count += 1

    p_value = (ge_count + 1) / (n_perm + 1)
    return float(obs), float(p_value)


def regression_residual_r(D_residual, D_tax, D_trans):
    """线性回归残差法 (cross-validation via residualize)."""
    v_r = upper_tri(D_residual)
    v_t = upper_tri(D_tax)
    v_g = upper_tri(D_trans)
    sl_r, int_r = np.polyfit(v_t, v_r, 1)
    sl_g, int_g = np.polyfit(v_t, v_g, 1)
    R_r = v_r - (sl_r * v_t + int_r)
    R_g = v_g - (sl_g * v_t + int_g)
    return float(pearsonr(R_r, R_g)[0])


# ========== Stage 0 Data Prep ==========

def build_tax_dist(metadata, sample_items):
    n = len(sample_items)
    cat_top = []
    cat_sub = []
    for it in sample_items:
        m = metadata.get(str(it), metadata.get(it, {}))
        cat_top.append(m.get('cat_top', 'Unknown'))
        cat_sub.append(m.get('cat_sub', 'Unknown'))
    cat_top = np.array(cat_top)
    cat_sub = np.array(cat_sub)

    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            if cat_sub[i] == cat_sub[j] and cat_sub[i] != 'Unknown':
                d = 0.0
            elif cat_top[i] == cat_top[j] and cat_top[i] != 'Unknown':
                d = 1.0
            else:
                d = 2.0
            D[i, j] = d
            D[j, i] = d
    return D


def build_transition_graph(n_items, n_users=3000, min_count=2):
    import tensorflow as tf  # 延迟导入
    transitions = defaultdict(lambda: defaultdict(int))
    user_count = 0
    train_files = sorted(os.listdir(f'{GRID}/data/amazon_data/toys/training/'))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(f'{GRID}/data/amazon_data/toys/training/', fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            seq = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            for pos in range(len(seq) - 1):
                a, b = seq[pos], seq[pos + 1]
                if a < n_items and b < n_items:
                    transitions[a][b] += 1
        if user_count > n_users:
            break

    adj = defaultdict(set)
    for a, neigh in transitions.items():
        for b, c in neigh.items():
            if c >= min_count:
                adj[a].add(b)
                adj[b].add(a)
    return dict(adj)


def shortest_path_dist(adj, sample_set, n_max=2000):
    from collections import deque
    item_list = list(sample_set)
    item_to_idx = {it: i for i, it in enumerate(item_list)}
    n = len(item_list)
    INF = 999
    D = np.full((n, n), INF, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    for src in item_list:
        if src not in adj:
            continue
        visited = {src: 0}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited[nxt] = visited[cur] + 1
                    queue.append(nxt)
        for dst, d in visited.items():
            if dst in item_to_idx:
                D[item_to_idx[src], item_to_idx[dst]] = d
    return D


def build_sample(metadata, n_sample=2000, seed=42, trans_adj=None):
    """筛选有 cat_sub 且在 transition graph 中的商品"""
    valid = [int(k) for k, v in metadata.items()
             if v.get('cat_sub', 'Unknown') != 'Unknown']
    if trans_adj is not None:
        valid = [i for i in valid if i in trans_adj and len(trans_adj[i]) >= 1]
    rng = np.random.default_rng(seed)
    if len(valid) > n_sample:
        sample = sorted(rng.choice(valid, size=n_sample, replace=False).tolist())
    else:
        sample = sorted(valid)
    return sample


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=3000)
    parser.add_argument('--n-users', type=int, default=5000)
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--n-perm', type=int, default=999)
    parser.add_argument('--quick', action='store_true', help='Quick: 500 items + 99 perm')
    args = parser.parse_args()

    if args.quick:
        args.n_sample = 500
        args.n_perm = 99

    n_sample = args.n_sample
    n_perm = args.n_perm
    print(f'=== Stage 0 + 1: Partial Mantel ({n_sample} items, {n_perm} perm) ===')

    # Load FLAN-T5 embeddings
    print('\n[1] Load FLAN-T5 embeddings')
    emb_path = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
    emb = torch.load(emb_path, weights_only=False, map_location='cpu').float()
    if emb.shape[0] == 2048 and emb.shape[1] == 11924:
        emb = emb.T
    n_items = emb.shape[0]
    print(f'  embedding shape: {emb.shape}, dim={emb.shape[1]}')

    # Load item metadata
    print('\n[2] Load item metadata')
    with open(f'{GRID}/result/ideaB_causal/toys_metadata.json') as f:
        d = json.load(f)
    metadata = d.get('data', d)
    print(f'  {len(metadata)} items')

    # Load RQ-VAE ckpt + forward_residual
    print('\n[3] Load RQ-VAE ckpt + compute per-layer residuals')
    ckpt_path = f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'
    codebooks, gains, L = load_codebooks(ckpt_path)
    print(f'  L={L}, codebook dim={codebooks[0].shape[-1]}')
    r_lst, q_lst, idx_lst = forward_residual(emb, codebooks, gains)
    for l, r in enumerate(r_lst):
        print(f'  r^({l}) shape={tuple(r.shape)}, norm_mean={r.norm(dim=-1).mean().item():.4f}')

    # Build transition graph
    print(f'\n[4] Build transition graph (n_users={args.n_users}, min_count={args.min_count})')
    t0 = time.time()
    trans_adj = build_transition_graph(n_items=11924, n_users=args.n_users, min_count=args.min_count)
    print(f'  {len(trans_adj)} active items, {sum(len(v) for v in trans_adj.values())//2} edges, time={time.time()-t0:.1f}s')

    # Sample items
    print('\n[5] Sample items')
    sample_items = build_sample(metadata, n_sample=n_sample, seed=42, trans_adj=trans_adj)
    print(f'  sampled: {len(sample_items)} items')
    sample_set = set(sample_items)
    sample_idx = np.array(sample_items)

    # Build D_tax
    print('\n[6] Build D_tax (cat_sub distance)')
    D_tax = build_tax_dist(metadata, sample_items)
    print(f'  D_tax shape={D_tax.shape}, unique={np.unique(D_tax)[:5]}')

    # Build D_trans (filtered to sample, BFS)
    print('\n[7] Build D_trans (BFS shortest path on sample subgraph)')
    adj_kept = defaultdict(set)
    for a in sample_set:
        if a in trans_adj:
            for b in trans_adj[a]:
                if b in sample_set:
                    adj_kept[a].add(b)
    D_trans = shortest_path_dist(adj_kept, sample_items)
    reachable_frac = (D_trans < 999).sum() / D_trans.size
    print(f'  D_trans shape={D_trans.shape}, reachable={reachable_frac*100:.1f}%, mean dist={D_trans[D_trans<999].mean():.2f}')

    # Build D_residual_l (Euclidean over each layer's residuals)
    print('\n[8] Build D_residual_l (Euclidean in RQ-VAE residual space)')
    D_residual_l = {}
    for l in range(L + 1):
        emb_l = r_lst[l][sample_idx].cpu().numpy()
        D = cdist(emb_l, emb_l, metric='euclidean').astype(np.float32)
        D_residual_l[l] = D
        print(f'  l={l}: mean dist={D.mean():.4f}, max={D.max():.4f}')

    # Save matrices
    print('\n[9] Save matrices')
    np.savez(f'{OUT_DIR}/dist_matrices.npz',
             sample_items=sample_items,
             D_tax=D_tax,
             D_trans=D_trans,
             **{f'D_residual_l{l}': D_residual_l[l] for l in range(L + 1)})
    print(f'  Saved: {OUT_DIR}/dist_matrices.npz')

    # ============ Stage 1: Partial Mantel ============
    print(f'\n=== Stage 1: Partial Mantel Test (n_perm={n_perm}) ===')
    results = {}
    for layer_l in range(L + 1):
        print(f'\n[L={layer_l}]')
        t0 = time.time()
        D_r = D_residual_l[layer_l]
        naive_r = mantel_pearson(D_r, D_trans)
        partial_r, partial_p = partial_mantel_perm(D_r, D_trans, D_tax,
                                                    n_perm=n_perm, seed=42 + layer_l)
        cv_r = regression_residual_r(D_r, D_tax, D_trans)

        # Compare with naive Mantel (tree only) for sanity
        naive_tax_r = mantel_pearson(D_r, D_tax)
        results[f'l={layer_l}'] = {
            'naive_trans_r': naive_r,
            'partial_trans_r_tax_control': partial_r,
            'partial_p': partial_p,
            'regression_resid_r': cv_r,
            'naive_tax_r': naive_tax_r,
            'preserved_pct': (partial_r / naive_r * 100) if abs(naive_r) > 1e-6 else 0,
        }
        print(f'  naive Mantel(trans): {naive_r:+.4f}')
        print(f'  naive Mantel(tax):   {naive_tax_r:+.4f}')
        print(f'  partial Mantel(trans|tax): {partial_r:+.4f}, p={partial_p:.4f}')
        print(f'  regression resid r:  {cv_r:+.4f}')
        print(f'  preserved:           {(partial_r/naive_r*100) if naive_r else 0:.1f}%')
        print(f'  time: {time.time()-t0:.1f}s')

    # Decision based on L1 (most relevant per task description)
    print('\n' + '=' * 70)
    print('DECISION (based on L=1)')
    print('=' * 70)
    r1 = results['l=1']
    naive = r1['naive_trans_r']
    partial = r1['partial_trans_r_tax_control']
    p = r1['partial_p']
    preserved = r1['preserved_pct']

    print(f'  L1 naive ρ (transition only) = {naive:+.4f}')
    print(f'  L1 partial ρ (tax controlled) = {partial:+.4f}, p={p:.4f}')
    print(f'  L1 preserved strength = {preserved:.1f}%')

    if preserved >= 50 and p < 0.05:
        decision = 'INDEPENDENT'
        next_step = 'Stage 2: L1 hyperbolic training'
        narrative = '转移图信号在 L1 与 taxonomy 相对独立 → 双曲化检验可行'
    elif preserved < 30 and abs(partial) < 0.01:
        decision = 'TAXONOMY_SHADOW'
        next_step = 'END (L1 纯双曲足够)'
        narrative = '转移图信号主要是 taxonomy 的影子 → 不需要拆分'
    else:
        decision = 'PARTIAL_OVERLAP'
        next_step = 'Stage 2 with monitoring'
        narrative = '部分重叠 → Stage 2 需监控重叠度变化'

    print(f'\n  ▶ Decision: {decision}')
    print(f'  ▶ {narrative}')
    print(f'  ▶ Next: {next_step}')

    # Save
    out = {
        'task': 'task380_stage0_1_partial_mantel',
        'method': 'partial_mantel + regression residual',
        'n_sample': n_sample,
        'n_perm': n_perm,
        'per_layer': results,
        'decision': {
            'layer_focus': 'l=1',
            'naive_r': float(naive),
            'partial_r': float(partial),
            'partial_p': float(p),
            'preserved_pct': float(preserved),
            'verdict': decision,
            'next_step': next_step,
        },
        'date': '2026-07-14',
        'status': 'completed',
    }

    def to_json_safe(obj):
        """递归转换 numpy 类型为 Python 原生类型"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_json_safe(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(f'{OUT_DIR}/partial_mantel_results.json', 'w') as f:
        json.dump(to_json_safe(out), f, indent=2, ensure_ascii=False)

    # Verdict md
    write_verdict(out, OUT_DIR)
    print(f'\n✅ Saved: {OUT_DIR}/partial_mantel_results.json')
    print(f'✅ Saved: {OUT_DIR}/verdict.md')


def write_verdict(out, OUT_DIR):
    dec = out['decision']
    lines = [
        '# Task 380 Verdict: Partial Mantel Test (Stage 1)',
        '',
        '## 设计',
        '',
        f'- **目标**: 控制 D_tax 后，D_residual(transition) 信号是否仍显著',
        f'- **方法**: partial Mantel (999 perm) + 回归残差交叉验证',
        f'- **数据**: n_sample={out["n_sample"]} 商品 (cat_sub ≠ Unknown AND transition-active)',
        '',
        '## 现象',
        '',
        '### 各层 partial Mantel 结果',
        '',
        '| Layer | naive ρ (trans) | partial ρ (trans|tax) | p-value | 保留度 | regression resid r |',
        '|-------|-----------------|------------------------|---------|--------|---------------------|',
    ]
    for layer_name, vals in out['per_layer'].items():
        lines.append(
            f'| {layer_name} | {vals["naive_trans_r"]:+.4f} | '
            f'{vals["partial_trans_r_tax_control"]:+.4f} | '
            f'{vals["partial_p"]:.4f} | {vals["preserved_pct"]:.1f}% | '
            f'{vals["regression_resid_r"]:+.4f} |'
        )

    lines.extend([
        '',
        '## 结论 (L=1 关键层)',
        '',
        f'- **naive ρ** = {dec["naive_r"]:+.4f}',
        f'- **partial ρ** (taxonomy controlled) = {dec["partial_r"]:+.4f}, p={dec["partial_p"]:.4f}',
        f'- **保留强度** = {dec["preserved_pct"]:.1f}%',
        '',
        f'**判定**: {dec["verdict"]}',
        '',
        '## 建议',
        '',
    ])
    if dec['verdict'] == 'INDEPENDENT':
        lines.append('→ 转移图信号与 taxonomy 在 L1 相对独立 → 进入 **Stage 2**')
        lines.append('→ 训练 L1 双曲量化器，验证双曲化是否牺牲转移图信息')
    elif dec['verdict'] == 'TAXONOMY_SHADOW':
        lines.append('→ 转移图信号主要是 taxonomy 的影子 → L1 纯双曲即可')
        lines.append('→ 不需要拆分架构，实验到此结束')
    else:
        lines.append('→ 部分重叠 → 进入 Stage 2 但需持续监控重叠度')

    with open(f'{OUT_DIR}/verdict.md', 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
