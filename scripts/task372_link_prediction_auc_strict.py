#!/usr/bin/env python3
"""Task 372: 链路预测 AUC bootstrap CI 严格化

弥补 task370 的两个缺陷:
  1. n_bootstrap=200 是 n_pos_sample(正边采样数),不是真 bootstrap 重复
  2. 只有一种负采样策略 (random non-edge)

task372 严格化:
  - 真 bootstrap: 对正边集合重采样 n_iter 次
  - 3 种负采样策略:
    * random: 完全随机非边
    * pop_matched: 按 endpoint 流行度匹配的非边
    * hard_negative: cosine 相似度高但不是真边的对
  - 每 (layer × 策略) 报 mean + 95% CI
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict

import numpy as np
import torch

GRID = '/home/wlia0047/wenyu/GeneRec/GRID' if False else '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
sys.path.insert(0, GRID)
import src.utils.decorators  # noqa: F401

# 复用 task370 的函数
sys.path.insert(0, f'{GRID}/task_artifacts/scripts')
from task370_network_structure import (
    load_codebooks, forward_residual,
    build_co_purchase_graph, build_transition_graph,
    build_adj_list,
)

OUT_DIR = f'{GRID}/result/task372_link_prediction_auc_strict'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
CKPT = f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'

N_ITEMS = 11924


# ========== 严格 AUC bootstrap ==========

def collect_pos_edges(adj_list, sampled_items):
    """返回 sampled_items 内的正边集合(去重 i<j)"""
    s = set(sampled_items)
    edges = []
    for i, neighs in adj_list.items():
        if i not in s:
            continue
        for j in neighs:
            if j > i and j in s:
                edges.append((i, j))
    return list(set(edges))


def collect_neg_random(sampled_items, pos_edges_set, n_neg, rng):
    """完全随机非边"""
    s = list(sampled_items)
    n = len(s)
    neg = []
    attempts = 0
    max_a = n_neg * 20
    while len(neg) < n_neg and attempts < max_a:
        i = rng.choice(n)
        j = rng.choice(n)
        if i == j:
            attempts += 1
            continue
        a, b = (s[i], s[j]) if s[i] < s[j] else (s[j], s[i])
        if (a, b) in pos_edges_set:
            attempts += 1
            continue
        if (a, b) in neg:
            attempts += 1
            continue
        neg.append((a, b))
        attempts += 1
    return neg


def collect_neg_pop_matched(sampled_items, pos_edges, item_pop, n_neg, rng):
    """流行度匹配的非边: 按 item_pop 分布采样"""
    s = list(sampled_items)
    # 按流行度排序, 取与正边 endpoint 流行度区间匹配的非边
    pos_pop_pairs = [(item_pop.get(a, 0), item_pop.get(b, 0)) for a, b in pos_edges]
    if not pos_pop_pairs:
        return collect_neg_random(sampled_items, set(pos_edges), n_neg, rng)
    pos_edges_set = set(pos_edges)
    neg = []
    # 流行度分桶
    pop_sorted = sorted([(item_pop.get(it, 0), it) for it in s], reverse=True)
    pop_rank = {it: r for r, (_, it) in enumerate(pop_sorted)}
    n = len(s)
    for _ in range(n_neg * 20):
        if len(neg) >= n_neg:
            break
        i_idx = rng.randint(n)
        j_idx = rng.randint(n)
        if i_idx == j_idx:
            continue
        a, b = (s[i_idx], s[j_idx]) if s[i_idx] < s[j_idx] else (s[j_idx], s[i_idx])
        if (a, b) in pos_edges_set or (a, b) in neg:
            continue
        # 检查流行度差异在正边范围内
        a_pop = item_pop.get(a, 0)
        b_pop = item_pop.get(b, 0)
        # 简单判定: 两端流行度的几何均值在正边流行度的 0.5x ~ 2x 之间
        avg_pop = (a_pop * b_pop) ** 0.5
        pos_avg_pops = [(p_a * p_b) ** 0.5 for p_a, p_b in pos_pop_pairs]
        if pos_avg_pops:
            med = np.median(pos_avg_pops)
            if 0.5 * med <= avg_pop <= 2.0 * med:
                neg.append((a, b))
    return neg[:n_neg]


def collect_neg_hard(sampled_items, pos_edges_set, emb, item_to_idx, n_neg, rng, top_k=50):
    """Hard negative: cosine 相似度高但不是真边的对

    策略: 对每个正边的 endpoint i, 找与 i cosine 相似度 top-K 但不在正边里的 j
    """
    s = list(sampled_items)
    # 预计算 cosine 相似度 (sampled_items 内)
    n = len(s)
    idx_map = item_to_idx
    # 取所有 sampled 的 embedding (按 item_to_idx 顺序)
    emb_sub = emb[[idx_map[it] for it in s]]  # (n, D)
    # cosine 相似度矩阵
    norm = np.linalg.norm(emb_sub, axis=1, keepdims=True) + 1e-9
    emb_n = emb_sub / norm
    cos_sim = emb_n @ emb_n.T  # (n, n)

    # 对每个 item, 取 top-K 相似(排除自己)
    np.fill_diagonal(cos_sim, -np.inf)
    top_k_idx = np.argpartition(-cos_sim, top_k, axis=1)[:, :top_k]  # (n, top_k)

    neg = []
    s_to_pos = {a: set() for a in s}
    for a, b in pos_edges_set:
        s_to_pos[a].add(b)
        s_to_pos[b].add(a)
    for _ in range(n_neg * 20):
        if len(neg) >= n_neg:
            break
        i_idx = rng.randint(n)
        candidates = top_k_idx[i_idx]
        j_idx = int(rng.choice(candidates))
        if i_idx == j_idx:
            continue
        a, b = (s[i_idx], s[j_idx]) if s[i_idx] < s[j_idx] else (s[j_idx], s[i_idx])
        if (a, b) in pos_edges_set or (a, b) in neg:
            continue
        neg.append((a, b))
    return neg[:n_neg]


def auc_one_pass(emb, item_to_idx, pos_edges, neg_edges):
    """单次 AUC 计算 (欧氏距离, 越小越相似, 所以 score = -dist)"""
    pos_scores = []
    for a, b in pos_edges:
        diff = emb[item_to_idx[a]] - emb[item_to_idx[b]]
        pos_scores.append(-np.sqrt((diff ** 2).sum()))
    neg_scores = []
    for a, b in neg_edges:
        diff = emb[item_to_idx[a]] - emb[item_to_idx[b]]
        neg_scores.append(-np.sqrt((diff ** 2).sum()))
    if not pos_scores or not neg_scores:
        return None
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    concordant = sum(1 for ps in pos_scores for ns in neg_scores if ps > ns)
    ties = sum(1 for ps in pos_scores for ns in neg_scores if ps == ns)
    return (concordant + 0.5 * ties) / (n_pos * n_neg)


def bootstrap_auc(emb, item_to_idx, pos_edges_all, sampled_items,
                  item_pop, strategy='random',
                  n_iter=200, n_pos=500, n_neg=5, seed=42):
    """严格 bootstrap AUC: 每次重采样正边 + 重新生成负边

    Returns:
        dict with mean, std, ci95, ci_lo, ci_hi, raw_values
    """
    rng = np.random.RandomState(seed)
    pos_edges_set = set(pos_edges_all)
    aucs = []
    for it in range(n_iter):
        # 重采样正边
        if len(pos_edges_all) <= n_pos:
            pos_boot = pos_edges_all
        else:
            idx = rng.choice(len(pos_edges_all), n_pos, replace=False)
            pos_boot = [pos_edges_all[i] for i in idx]
        # 重生成负边
        if strategy == 'random':
            neg_boot = collect_neg_random(sampled_items, pos_edges_set, n_pos * n_neg, rng)
        elif strategy == 'pop_matched':
            neg_boot = collect_neg_pop_matched(sampled_items, pos_boot, item_pop,
                                                n_pos * n_neg, rng)
        elif strategy == 'hard_negative':
            neg_boot = collect_neg_hard(sampled_items, pos_edges_set, emb,
                                         item_to_idx, n_pos * n_neg, rng)
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
        if len(neg_boot) < n_pos * n_neg * 0.8:
            continue  # 跳过无效迭代
        auc = auc_one_pass(emb, item_to_idx, pos_boot, neg_boot)
        if auc is not None:
            aucs.append(auc)
    aucs = np.array(aucs)
    return {
        'n_iter': int(len(aucs)),
        'n_iter_requested': n_iter,
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs, ddof=1)),
        'ci95_lo': float(np.percentile(aucs, 2.5)),
        'ci95_hi': float(np.percentile(aucs, 97.5)),
        'ci90_lo': float(np.percentile(aucs, 5)),
        'ci90_hi': float(np.percentile(aucs, 95)),
        'raw': aucs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-iter', type=int, default=200,
                        help='bootstrap 重复次数 (默认 200)')
    parser.add_argument('--n-pos', type=int, default=500,
                        help='每次 bootstrap 抽多少正边 (默认 500)')
    parser.add_argument('--n-neg', type=int, default=5,
                        help='每个正边配几个负样本 (默认 5)')
    parser.add_argument('--n-users', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'[task372] Loading embeddings from {EMB_PATH}', flush=True)
    emb_all = torch.load(EMB_PATH, map_location='cpu', weights_only=False)
    # emb shape: (N_items, D_emb) for FLAN-T5
    print(f'[task372] emb shape: {emb_all.shape}')

    print(f'[task372] Loading codebook from {CKPT}', flush=True)
    codebooks, gains, has_gains, normalize, L = load_codebooks(CKPT)
    print(f'[task372] L={L} layers, has_gains={has_gains}, normalize={normalize}')

    # 计算 residual (l=0~3)
    print('[task372] Computing residuals...', flush=True)
    x = emb_all.float()
    r_lst, _, _ = forward_residual(x, codebooks, gains if has_gains else None)
    # r_lst[0]=input, r_lst[1]=l=0 res, r_lst[2]=l=1 res, r_lst[3]=l=2 res, r_lst[4]=l=3 res
    residuals = {l: r_lst[l + 1].numpy() for l in range(L)}
    print(f'[task372] Residual shapes: {[(l, residuals[l].shape) for l in range(L)]}')

    # 抽 3000 商品
    rng = np.random.RandomState(args.seed)
    sample_items = sorted(rng.choice(N_ITEMS, size=3000, replace=False).tolist())
    item_to_idx = {it: i for i, it in enumerate(sample_items)}
    sampled_set = set(sample_items)
    print(f'[task372] Sampled {len(sample_items)} items', flush=True)

    # 构建 copurchase + transition graphs
    print('[task372] Building copurchase graph...', flush=True)
    t0 = time.time()
    copurchase_edges, _ = build_co_purchase_graph(N_ITEMS, n_users=args.n_users,
                                                  window=5, min_weight=2)
    from task370_network_structure import build_adj_list
    copurchase_adj = build_adj_list(copurchase_edges, N_ITEMS, directed=False)
    print(f'[task372] copurchase graph: {sum(len(v) for v in copurchase_adj.values())//2} edges, '
          f'{len(copurchase_adj)} nodes, {time.time()-t0:.1f}s', flush=True)

    print('[task372] Building transition graph...', flush=True)
    t0 = time.time()
    transition_edges, transition_counts = build_transition_graph(N_ITEMS, n_users=args.n_users, min_count=2)
    transition_adj = build_adj_list(transition_edges, N_ITEMS, directed=True)
    print(f'[task372] transition graph: {len(transition_edges)} directed edges, '
          f'{len(transition_adj)} nodes, {time.time()-t0:.1f}s', flush=True)

    # 收集正边
    pos_copurchase = collect_pos_edges(copurchase_adj, sampled_set)
    pos_transition = collect_pos_edges(transition_adj, sampled_set)
    print(f'[task372] copurchase pos edges in sample: {len(pos_copurchase)}', flush=True)
    print(f'[task372] transition pos edges in sample: {len(pos_transition)}', flush=True)

    # 计算 item_pop (来自 transition counts, 因为 transition 自带 frequency 信息)
    item_pop = defaultdict(int)
    for (a, b), c in transition_counts.items():
        item_pop[a] += c
        item_pop[b] += c
    # 也并入 copurchase 的度数(粗略, 不区分窗口)
    for a, neighs in copurchase_adj.items():
        item_pop[a] += len(neighs)

    # 对每个 layer × 每个 graph × 每个策略 跑 bootstrap
    strategies = ['random', 'pop_matched', 'hard_negative']
    graphs = {
        'copurchase': (copurchase_adj, pos_copurchase),
        'transition': (transition_adj, pos_transition),
    }

    all_results = {}
    for graph_name, (adj, pos_edges) in graphs.items():
        all_results[graph_name] = {}
        for l in range(L):
            all_results[graph_name][l] = {}
            emb_l = residuals[l]
            for strategy in strategies:
                print(f'[task372] {graph_name} l={l} strategy={strategy} '
                      f'(n_iter={args.n_iter})...', flush=True)
                t0 = time.time()
                res = bootstrap_auc(
                    emb_l, item_to_idx, pos_edges, sampled_set, dict(item_pop),
                    strategy=strategy,
                    n_iter=args.n_iter, n_pos=args.n_pos, n_neg=args.n_neg,
                    seed=args.seed + l,
                )
                elapsed = time.time() - t0
                all_results[graph_name][l][strategy] = res
                print(f'  mean={res["mean"]:.4f} std={res["std"]:.4f} '
                      f'CI95=[{res["ci95_lo"]:.4f}, {res["ci95_hi"]:.4f}] '
                      f'({res["n_iter"]}/{res["n_iter_requested"]} iters, '
                      f'{elapsed:.1f}s)', flush=True)

    # 整合 verdict
    # 1. 看 transition 是否在 CI 95 上显著 > 0.5 (chance)
    # 2. 看 copurchase 同理
    # 3. 看 hard_negative 是否低于 random (这是深层信号被 hard neg 考验的检验)
    summary = {}
    for graph_name in graphs:
        summary[graph_name] = {}
        for l in range(L):
            row = {}
            for strategy in strategies:
                r = all_results[graph_name][l][strategy]
                row[strategy] = {
                    'mean': r['mean'],
                    'std': r['std'],
                    'ci95_lo': r['ci95_lo'],
                    'ci95_hi': r['ci95_hi'],
                    'n_iter': r['n_iter'],
                    'sig_above_chance': bool(r['ci95_lo'] > 0.5),
                }
            row['hard_vs_random'] = {
                'delta_mean': row['hard_negative']['mean'] - row['random']['mean'],
                'overlap_ci': bool(
                    row['hard_negative']['ci95_hi'] >= row['random']['ci95_lo']
                    and row['random']['ci95_hi'] >= row['hard_negative']['ci95_lo']
                ),
            }
            summary[graph_name][l] = row

    out = {
        'task': 'task372_link_prediction_auc_strict',
        'method': 'strict_bootstrap_auc_multi_neg',
        'date': '2026-07-14',
        'status': 'completed',
        'data': {
            'config': {
                'n_iter': args.n_iter,
                'n_pos': args.n_pos,
                'n_neg': args.n_neg,
                'n_users_graph': args.n_users,
                'seed': args.seed,
            },
            'n_sample': len(sample_items),
            'n_copurchase_pos': len(pos_copurchase),
            'n_transition_pos': len(pos_transition),
            'raw': {
                graph_name: {
                    str(l): {
                        strategy: all_results[graph_name][l][strategy]
                        for strategy in strategies
                    }
                    for l in range(L)
                }
                for graph_name in graphs
            },
            'summary': summary,
        }
    }
    out_path = f'{OUT_DIR}/auc_strict_bootstrap_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[task372] Saved: {out_path}')

    # 打印最终判定表
    print('\n=== task372 Final Verdict ===')
    print(f'{"Graph":<12}{"Layer":<8}{"Strategy":<18}{"Mean":<10}{"CI95":<22}{"Sig>0.5":<10}')
    print('-' * 80)
    for graph_name in graphs:
        for l in range(L):
            for strategy in strategies:
                r = summary[graph_name][l][strategy]
                sig = 'YES' if r['sig_above_chance'] else 'NO'
                print(f'{graph_name:<12}l={l:<6}{strategy:<18}{r["mean"]:.4f}     '
                      f'[{r["ci95_lo"]:.4f}, {r["ci95_hi"]:.4f}]   {sig}')


if __name__ == '__main__':
    main()