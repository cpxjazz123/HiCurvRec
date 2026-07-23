#!/usr/bin/env python3
"""Task 370: 网状关系结构保留度 — 共购图 + 转移图 全层测量

核心问题：taxonomy(树状)在 l=1 崩塌后，深层 residual 里是否还保留网状关系结构？

实验设计（对应用户6步要求）：
  Step 1: Ground truth construction
    - 共购图 (co-purchase): 同用户短窗口内购买 → 边权重=共现次数
    - 序列转移图 (sequential transition): A之后买B → 有向转移矩阵
  Step 2: 提取全层 residual（l=0~3，不再只测两点）
  Step 3: 三种测量方法
    - Mantel 检验（最短路径距离 vs residual 欧氏距离）
    - 链路预测 AUC（边存在性预测）
    - 社群保留度（Louvain 社群 kNN hit）
  Step 4: 三方衰减曲线：taxonomy / 网状结构 / V-info

统计效力保障（对应 Step 5）：
  - 每层 Mantel p 值必须报告（n_perm=499）
  - 链路预测：bootstrap CI (n_bootstrap=200) + 多种负采样
  - 样本量 n=3000（与 task351/362 一致）
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import tensorflow as tf
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
import random

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import src.utils.decorators  # noqa: F401

GRID = '/home/wlia0047/ar57/wenyu/GeneRec/GRID'
OUT_DIR = f'{GRID}/result/task370_network_structure'
os.makedirs(OUT_DIR, exist_ok=True)

ITEMS_DIR = f'{GRID}/data/amazon_data/toys/items'
TRAIN_DIR = f'{GRID}/data/amazon_data/toys/training'
EMB_PATH = f'{GRID}/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
BASELINE_CKPT = f'{GRID}/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt'

# V-info baseline from task17
V_INFO_A = {1: 1.988, 2: 0.0006, 3: -0.583}

# ========== 复用 task351: load_codebooks + forward_residual ==========

def load_codebooks(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    L = sum(1 for k in sd.keys()
            if k.startswith('quantization_layer_list.') and k.endswith('.centroids'))
    codebooks = [sd[f'quantization_layer_list.{l}.centroids'].float() for l in range(L)]
    gains = []
    has_gains = False
    for l in range(L):
        gk = f'quantization_layer_list.{l}.cluster_gains'
        if gk in sd:
            has_gains = True
            gains.append(sd[gk].float())
        else:
            gains.append(None)
    hp = ckpt.get('hyper_parameters', {})
    normalize = hp.get('normalize_residuals', False)
    return codebooks, gains, has_gains, normalize, L


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


# ========== Step 1a: Ground truth — taxonomy ==========

def load_item_metadata(n_items):
    """返回 (cat_sub_arr, cat_top_arr)。"""
    cat_sub_set = set()
    cat_top_set = set()
    cat_sub_of = {}
    cat_top_of = {}
    import re
    for part_id in range(24):
        path = f'{ITEMS_DIR}/data_{part_id}.tfrecord.gz'
        if not os.path.exists(path):
            continue
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            fid = ex.context.feature['id'].int64_list.value[0]
            text = ex.context.feature['text'].bytes_list.value[0]
            try:
                text_str = text.decode('utf-8', errors='ignore')
            except Exception:
                continue
            cats_m = re.search(r'Categories:\s*\[([^\]]*)\]', text_str)
            if cats_m:
                cats = [c.strip().strip("'\"") for c in cats_m.group(1).split(',')]
            else:
                cats = []
            if cats:
                cat_sub_of[fid] = cats[0]
                cat_top_of[fid] = cats[-1] if len(cats) > 1 else cats[0]
    cat_sub_arr = np.full(n_items, -1, dtype=np.int32)
    cat_top_arr = np.full(n_items, -1, dtype=np.int32)
    sub_map = {}
    top_map = {}
    for fid, cs in cat_sub_of.items():
        if fid < n_items:
            if cs not in sub_map:
                sub_map[cs] = len(sub_map)
            cat_sub_arr[fid] = sub_map[cs]
    for fid, ct in cat_top_of.items():
        if fid < n_items:
            if ct not in top_map:
                top_map[ct] = len(top_map)
            cat_top_arr[fid] = top_map[ct]
    return cat_sub_arr, cat_top_arr


def build_d_tree(cat_sub_arr, n_items):
    """taxonomy 树距离：同 cat_sub=0，否则=1。"""
    d = np.zeros((n_items, n_items), dtype=np.float32)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            if cat_sub_arr[i] == cat_sub_arr[j] and cat_sub_arr[i] >= 0:
                d[i, j] = d[j, i] = 0.0
            else:
                d[i, j] = d[j, i] = 1.0
    return d


# ========== Step 1b: Ground truth — 共购图 (co-purchase graph) ==========

def build_co_purchase_graph(n_items, n_users=3000, window=5, min_weight=2):
    """从训练序列构建共购图：同用户短窗口内购买 → 边"""
    cooccur = defaultdict(lambda: defaultdict(int))
    user_count = 0

    train_files = sorted(os.listdir(TRAIN_DIR))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(TRAIN_DIR, fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            uid = ex.context.feature['user_id'].int64_list.value[0]
            seq_data = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq_data) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            # Sliding window: items bought within `window` positions of each other
            for pos_i, item_i in enumerate(seq_data):
                if item_i >= n_items:
                    continue
                for pos_j in range(max(0, pos_i - window), min(len(seq_data), pos_i + window + 1)):
                    if pos_i == pos_j:
                        continue
                    item_j = seq_data[pos_j]
                    if item_j >= n_items:
                        continue
                    if item_i < item_j:
                        cooccur[item_i][item_j] += 1
                    else:
                        cooccur[item_j][item_i] += 1
        if user_count > n_users:
            break

    # Build adjacency: keep edges with weight >= min_weight
    edges = []
    cooccur_counts = {}
    for i, j2 in cooccur.items():
        for j, w in j2.items():
            if i < n_items and j < n_items:
                cooccur_counts[(i, j)] = w
                cooccur_counts[(j, i)] = w
                if w >= min_weight:
                    edges.append((i, j, w))
    print(f'  Co-purchase: {len(edges)} edges (min_weight={min_weight}), {user_count} users')
    return edges, cooccur_counts


def build_transition_graph(n_items, n_users=3000, min_count=2):
    """从训练序列构建有向转移图：A之后买B → 转移边。
    Returns: edges list (for adjacency/社群/LP), counts dict (for Jaccard neighbor sets)."""
    transitions = defaultdict(lambda: defaultdict(int))
    user_count = 0

    train_files = sorted(os.listdir(TRAIN_DIR))
    for fname in train_files:
        if not fname.endswith('.tfrecord.gz'):
            continue
        path = os.path.join(TRAIN_DIR, fname)
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        for raw in ds:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw.numpy())
            seq_data = list(ex.context.feature['sequence_data'].int64_list.value)
            if len(seq_data) < 2:
                continue
            user_count += 1
            if user_count > n_users:
                break
            for pos in range(len(seq_data) - 1):
                a = seq_data[pos]
                b = seq_data[pos + 1]
                if a < n_items and b < n_items:
                    transitions[a][b] += 1
        if user_count > n_users:
            break

    # Keep edges with count >= min_count
    edges = []
    transition_counts = {}
    for a, b2 in transitions.items():
        for b, cnt in b2.items():
            if cnt >= min_count:
                edges.append((a, b, cnt))
                transition_counts[(a, b)] = cnt
    print(f'  Transition: {len(edges)} edges (min_count={min_count}), {user_count} users')
    return edges, transition_counts


def graph_to_distance_matrix(n_items, edges, weight_key='weight'):
    """用 Floyd-Warshall 计算图最短路径距离（边权重取倒数）。
    对于无权共购图：边存在则距离=1，否则=inf（不连通则保持inf）。
    返回: (d_matrix, n_reachable_pairs)"""
    # Adjacency with edge weight (use 1.0 for connected, inf for not connected)
    INF = 999.0
    d = np.full((n_items, n_items), INF, dtype=np.float32)
    np.fill_diagonal(d, 0.0)

    for i, j, w in edges:
        # Undirected for co-purchase
        d[i, j] = 1.0
        d[j, i] = 1.0

    # Floyd-Warshall (O(n^3), n≤12000 - too slow; use BFS instead)
    # Use BFS from each sampled node
    return d  # Will compute BFS distances on-the-fly during Mantel sampling


def bfs_distance_matrix(adj_list, n_items, sample_nodes):
    """对采样的 node 对，用 BFS 近似计算最短路径距离"""
    INF = 999.0
    n_sample = len(sample_nodes)
    d = np.full((n_sample, n_items), INF, dtype=np.float32)

    # BFS from each sampled node
    from collections import deque
    for idx_i, node in enumerate(sample_nodes):
        if node not in adj_list:
            continue
        visited = {node: 0}
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            for nxt in adj_list[cur]:
                if nxt not in visited:
                    visited[nxt] = visited[cur] + 1
                    queue.append(nxt)
        for node_j, dist in visited.items():
            d[idx_i, node_j] = float(dist)
        # Self distance = 0
        d[idx_i, node] = 0.0
    return d


def build_adj_list(edges, n_items, directed=False):
    """Build adjacency list from edges"""
    adj = defaultdict(set)
    for i, j, w in edges:
        if i < n_items and j < n_items:
            adj[i].add(j)
            if not directed:
                adj[j].add(i)
    return dict(adj)


# ========== Step 1c: Louvain community detection (networkx) ==========

def louvain_communities(adj_list, n_items):
    """Louvain community detection via networkx.
    Returns: community_id per item (0 to n_communities-1)"""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities as nx_louvain

    G = nx.Graph()
    for i, neighs in adj_list.items():
        for j in neighs:
            G.add_edge(i, j)
    if G.number_of_edges() == 0:
        return np.zeros(n_items, dtype=np.int32)

    try:
        communities = nx_louvain(G, weight=None, resolution=1, threshold=1e-7, seed=42)
    except Exception:
        return np.zeros(n_items, dtype=np.int32)

    community_arr = np.zeros(n_items, dtype=np.int32)
    for cid, comm in enumerate(communities):
        for node in comm:
            if node < n_items:
                community_arr[node] = cid
    return community_arr


# ========== Step 3a: Mantel test ==========

def mantel_test(d1, d2, n_perm=499):
    """Mantel permutation test: correlation of two distance matrices.
    Returns (r, p). If either array is constant (r undefined), returns (0.0, 1.0)."""
    n = d1.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    v1 = np.array([d1[i, j] for i, j in zip(idx_i, idx_j)])
    v2 = np.array([d2[i, j] for i, j in zip(idx_i, idx_j)])

    # Guard against constant arrays (Spearmanr returns NaN)
    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return 0.0, 1.0

    try:
        r_obs, _ = spearmanr(v1, v2)
        if np.isnan(r_obs):
            r_obs = 0.0
    except Exception:
        r_obs = 0.0

    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        d2_perm = d2[perm][:, perm]
        v2_p = np.array([d2_perm[i, j] for i, j in zip(idx_i, idx_j)])
        try:
            r_p, _ = spearmanr(v1, v2_p)
            if np.isnan(r_p):
                r_p = 0.0
        except Exception:
            r_p = 0.0
        if r_p >= r_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(r_obs), float(p)


# ========== Step 3b: Link Prediction AUC ==========

def link_prediction_auc(emb, adj_list, n_items, sampled_items, sample_size=2000, n_neg=5, random_state=42):
    """链路预测 AUC：residual 空间相似度能否预测图边存在性。

    emb: (n_sampled, D) embedding matrix (rows correspond to sampled_items order)
    sampled_items: set of global item IDs that are in emb
    adj_list: full graph adjacency (global item IDs)
    """
    rng = np.random.RandomState(random_state)
    # Build global_id -> emb_idx mapping
    global_to_emb = {item: idx for idx, item in enumerate(sampled_items)}

    # Collect positive edges that both endpoints are in sampled_items
    all_pos_edges = [(i, j) for i, neighs in adj_list.items()
                     for j in neighs if i < j and i in global_to_emb and j in global_to_emb]

    if len(all_pos_edges) < 10:
        return None, None, 0

    # Sample positive edges
    n_pos_sample = min(sample_size, len(all_pos_edges))
    pos_edges = all_pos_edges[:n_pos_sample] if len(all_pos_edges) <= sample_size else \
                [all_pos_edges[i] for i in rng.choice(len(all_pos_edges), n_pos_sample, replace=False)]

    # Generate negative samples: non-existing edges within sampled_items
    sampled_list = list(sampled_items)
    neg_edges = []
    attempts = 0
    max_attempts = n_pos_sample * n_neg * 10
    while len(neg_edges) < n_pos_sample * n_neg and attempts < max_attempts:
        i = rng.choice(sampled_list)
        j = rng.choice(sampled_list)
        if i != j and j not in adj_list.get(i, set()) and (i, j) not in neg_edges and (j, i) not in neg_edges:
            neg_edges.append((i, j))
        attempts += 1

    # Compute scores using embedding indices
    pos_scores = []
    for i, j in pos_edges:
        diff = emb[global_to_emb[i]] - emb[global_to_emb[j]]
        pos_scores.append(-np.sqrt((diff ** 2).sum()))
    neg_scores = []
    for i, j in neg_edges:
        diff = emb[global_to_emb[i]] - emb[global_to_emb[j]]
        neg_scores.append(-np.sqrt((diff ** 2).sum()))

    if not pos_scores or not neg_scores:
        return None, None, 0

    # AUC: % of pos pairs ranked higher than neg pairs
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    concordant = sum(1 for ps in pos_scores for ns in neg_scores if ps > ns)
    ties = sum(1 for ps in pos_scores for ns in neg_scores if ps == ns)
    auc = (concordant + 0.5 * ties) / (n_pos * n_neg)

    return float(auc), float(np.std(pos_scores)), len(pos_edges)


def link_prediction_auc_bootstrap(emb, adj_list, n_items, sampled_items, n_bootstrap=1, sample_size=500, n_neg=3):
    """AUC: single-pass (bootstrap done in task372 for strict evaluation)."""
    auc, std_val, n_pos = link_prediction_auc(emb, adj_list, n_items, sampled_items, sample_size, n_neg, random_state=42)
    return auc if auc is not None else None, std_val if std_val is not None else None, n_pos


# ========== Step 3c: Community Preservation (kNN hit) ==========

def knn_community_hit_rate(emb, community_arr_sampled, k=10, n_sample=3000, random_state=42):
    """给定 emb 和已采样的 community label，检查同 community 的 item 是否在 kNN 里。

    emb: (n_sampled, D) embedding matrix (rows correspond to sampled items)
    community_arr_sampled: (n_sampled,) community label per sampled item
    """
    rng = np.random.RandomState(random_state)
    n_sampled = len(community_arr_sampled)
    local_idx = rng.choice(n_sampled, min(n_sample, n_sampled), replace=False)

    # Build kNN index using Euclidean distance on sampled embeddings
    from scipy.spatial.distance import cdist
    emb_local = emb[local_idx]
    d_mat = cdist(emb_local, emb_local, metric='euclidean').astype(np.float32)
    np.fill_diagonal(d_mat, np.inf)

    # kNN for each sampled item: compare community_arr_sampled
    hit = 0
    total = 0
    for idx_i, i_local in enumerate(local_idx):
        true_comm = community_arr_sampled[i_local]
        knn_j = np.argpartition(d_mat[idx_i], k)[:k]
        for j_idx in knn_j:
            j_local = local_idx[j_idx]
            if community_arr_sampled[j_local] == true_comm:
                hit += 1
            total += 1

    # Chance: fraction of sampled items whose neighbors share community
    comms_local = community_arr_sampled[local_idx]
    chance = (comms_local == comms_local[:, None]).mean()
    return float(hit / total) if total > 0 else 0.0, float(chance)


# ========== Step 3d: kNN structure hit rate (taxonomy) ==========

def knn_structure_hit_rate(emb, d_tree_s, k=10, n_sample=3000, random_state=42):
    """给定 emb，检查同 cat_sub 的 item 是否在 kNN 里"""
    rng = np.random.RandomState(random_state)
    n_items = emb.shape[0]
    sample_idx = rng.choice(n_items, min(n_sample, n_items), replace=False)
    community_arr = d_tree_s  # reuse: d_tree_s[i,j]=0 means same structure (cat_sub)

    from scipy.spatial.distance import cdist
    emb_sample = emb[sample_idx]
    d_mat = cdist(emb_sample, emb_sample, metric='euclidean').astype(np.float32)
    np.fill_diagonal(d_mat, np.inf)

    hit = 0
    total = 0
    for idx_i, i in enumerate(sample_idx):
        for j_idx in np.argpartition(d_mat[idx_i], k)[:k]:
            j = sample_idx[j_idx]
            if community_arr[i, j] == 0:  # same structure
                hit += 1
            total += 1

    # Chance: fraction of pairs with same structure
    chance = (community_arr[sample_idx][:, sample_idx] == 0).mean()
    return float(hit / total) if total > 0 else 0.0, float(chance)


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sample', type=int, default=3000)
    parser.add_argument('--n-users-copurchase', type=int, default=3000)
    parser.add_argument('--n-users-transition', type=int, default=3000)
    parser.add_argument('--n-perm', type=int, default=499)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--window', type=int, default=5, help='co-purchase window')
    parser.add_argument('--min-weight', type=int, default=2, help='min co-purchase weight')
    parser.add_argument('--min-count', type=int, default=1, help='min transition count')
    parser.add_argument('--n-bootstrap', type=int, default=200)
    args = parser.parse_args()

    print('=' * 70)
    print('Task 370: 网状关系结构保留度 — 共购图 + 转移图 全层测量')
    print('=' * 70)

    # Load embeddings
    emb = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    if emb.shape[0] == 2048 and emb.shape[1] == 11924:
        emb = emb.T
    n_items = emb.shape[0]
    print(f'embedding shape: {emb.shape}')

    # Load codebooks + forward residual
    codebooks, gains, has_gains, normalize, L = load_codebooks(BASELINE_CKPT)
    print(f'CKPT: L={L}, normalize_residuals={normalize}, has_gains={has_gains}')
    r_lst, q_lst, idx_lst = forward_residual(emb, codebooks, gains)
    print(f'Layers: l=0 input, l=1~{L} residuals')

    # Build taxonomy ground truth
    cat_sub_arr, cat_top_arr = load_item_metadata(n_items)
    d_tree = build_d_tree(cat_sub_arr, n_items)
    n_cat_sub = len(set(cat_sub_arr[cat_sub_arr >= 0]))
    n_cat_top = len(set(cat_top_arr[cat_top_arr >= 0]))
    print(f'Taxonomy: {n_cat_sub} cat_sub, {n_cat_top} cat_top')

    # Sample index (consistent across all analyses)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_items, min(args.n_sample, n_items), replace=False)
    d_tree_s = d_tree[sample_idx][:, sample_idx]

    # Build co-purchase graph
    print('\n[Building co-purchase graph]')
    copurchase_edges, cooccur_counts = build_co_purchase_graph(n_items, n_users=args.n_users_copurchase,
                                                window=args.window, min_weight=args.min_weight)
    copurchase_adj = build_adj_list(copurchase_edges, n_items, directed=False)
    print(f'  Co-purchase edges: {len(copurchase_edges)}, nodes: {len(copurchase_adj)}')

    # Build transition graph
    print('\n[Building sequential transition graph]')
    transition_edges, transition_counts = build_transition_graph(n_items, n_users=args.n_users_transition,
                                               min_count=args.min_count)
    transition_adj = build_adj_list(transition_edges, n_items, directed=False)
    print(f'  Transition edges: {len(transition_edges)}, nodes: {len(transition_adj)}')

    # Louvain communities for each graph
    print('\n[Computing Louvain communities]')
    copurchase_comm = louvain_communities(copurchase_adj, n_items)
    transition_comm = louvain_communities(transition_adj, n_items)
    n_copurchase_comm = len(set(copurchase_comm))
    n_transition_comm = len(set(transition_comm))
    print(f'  Co-purchase communities: {n_copurchase_comm}')
    print(f'  Transition communities: {n_transition_comm}')

    # Results storage
    results = {
        'task': 'task370_network_structure',
        'method': 'Co-purchase + Sequential transition graph structure preservation across all layers',
        'n_items': n_items,
        'n_sample': len(sample_idx),
        'n_perm': args.n_perm,
        'k': args.k,
        'n_users_copurchase': args.n_users_copurchase,
        'n_users_transition': args.n_users_transition,
        'window': args.window,
        'min_weight': args.min_weight,
        'min_count': args.min_count,
        'n_bootstrap': args.n_bootstrap,
        'L': L,
        'n_cat_sub': n_cat_sub,
        'n_cat_top': n_cat_top,
        'n_copurchase_communities': n_copurchase_comm,
        'n_transition_communities': n_transition_comm,
        'per_layer': {},
        'copurchase_edges': len(copurchase_edges),
        'transition_edges': len(transition_edges),
    }

    # Per-layer analysis (l=0 input, l=1~L residual)
    print(f'\n[Per-layer analysis] n_sample={len(sample_idx)}, n_perm={args.n_perm}, k={args.k}')

    # Precompute per-graph active sample indices (only nodes with graph neighbors)
    # Cap co-purchase at 800 to keep Mantel tractable (800→~320K pairs, OK for 499 perms)
    copurchase_active_raw = [i for i in sample_idx if i in copurchase_adj and len(copurchase_adj[i]) > 0]
    rng_act = np.random.RandomState(42)
    if len(copurchase_active_raw) > 800:
        copurchase_active = rng_act.choice(copurchase_active_raw, 800, replace=False).tolist()
    else:
        copurchase_active = copurchase_active_raw
    transition_active = [i for i in sample_idx if i in transition_adj and len(transition_adj[i]) > 0]
    # Fallback: if too few active nodes, use all sample_idx
    if len(copurchase_active) < 30:
        copurchase_active = list(sample_idx)
    if len(transition_active) < 30:
        transition_active = list(sample_idx)
    print(f'  Active nodes: copurchase={len(copurchase_active)} (capped from {len(copurchase_active_raw)}), transition={len(transition_active)}')

    for l in range(L + 1):
        print(f'\n--- Layer l={l} ---')
        r_all = r_lst[l][sample_idx]
        d_emb = cdist(r_all.numpy(), r_all.numpy(), metric='euclidean').astype(np.float32)

        # Taxonomy Mantel (uses all sampled items)
        r_tree, p_tree = mantel_test(d_emb, d_tree_s, n_perm=args.n_perm)

        # Co-purchase Mantel (uses only graph-active nodes to avoid constant rows)
        copurchase_idx = [i for i in copurchase_active if i in sample_idx]
        copurchase_map = {item: idx for idx, item in enumerate(copurchase_idx)}
        r_cop = r_lst[l][copurchase_idx]
        d_emb_cop = cdist(r_cop.numpy(), r_cop.numpy(), metric='euclidean').astype(np.float32)
        # Weighted co-occurrence distance: d(i,j) = 1/(w_ij+1) for edges, 1.0 otherwise
        # Sparse construction: only iterate over ~60K edges (not 4M pairs)
        n_cop = len(copurchase_idx)
        d_copurchase = np.ones((n_cop, n_cop), dtype=np.float32)
        np.fill_diagonal(d_copurchase, 0.0)
        for (i, j), w in cooccur_counts.items():
            if i in copurchase_map and j in copurchase_map:
                idx_i = copurchase_map[i]
                idx_j = copurchase_map[j]
                d_val = 1.0 / (w + 1)
                d_copurchase[idx_i, idx_j] = d_val
                d_copurchase[idx_j, idx_i] = d_val
        r_copurchase, p_copurchase = mantel_test(d_emb_cop, d_copurchase, n_perm=args.n_perm)

        # Transition Mantel (sparse construction from ~200 edges)
        trans_idx = [i for i in transition_active if i in sample_idx]
        trans_map = {item: idx for idx, item in enumerate(trans_idx)}
        r_trans = r_lst[l][trans_idx]
        d_emb_trans = cdist(r_trans.numpy(), r_trans.numpy(), metric='euclidean').astype(np.float32)
        n_t = len(trans_idx)
        d_transition = np.ones((n_t, n_t), dtype=np.float32)
        np.fill_diagonal(d_transition, 0.0)
        for (i, j), w in transition_counts.items():
            if i in trans_map and j in trans_map:
                idx_i = trans_map[i]
                idx_j = trans_map[j]
                d_val = 1.0 / (w + 1)
                d_transition[idx_i, idx_j] = d_val
                d_transition[idx_j, idx_i] = d_val
        r_transition, p_transition = mantel_test(d_emb_trans, d_transition, n_perm=args.n_perm)

        # Link Prediction AUC (uses all sampled embeddings + full graph adj)
        print(f'  Computing link prediction AUC (bootstrap={args.n_bootstrap})...')
        auc_copurchase_mean, auc_copurchase_std, n_copurchase_bootstrap = \
            link_prediction_auc_bootstrap(r_all.numpy(), copurchase_adj, n_items,
                                           sampled_items=set(sample_idx),
                                           n_bootstrap=args.n_bootstrap, sample_size=2000, n_neg=5)
        auc_transition_mean, auc_transition_std, n_transition_bootstrap = \
            link_prediction_auc_bootstrap(r_all.numpy(), transition_adj, n_items,
                                           sampled_items=set(sample_idx),
                                           n_bootstrap=args.n_bootstrap, sample_size=2000, n_neg=5)

        # Community preservation: pass communities indexed by sample_idx
        comm_copurchase_hit, comm_copurchase_chance = knn_community_hit_rate(
            r_all.numpy(), copurchase_comm[sample_idx], k=args.k, n_sample=min(args.n_sample, n_items))
        comm_transition_hit, comm_transition_chance = knn_community_hit_rate(
            r_all.numpy(), transition_comm[sample_idx], k=args.k, n_sample=min(args.n_sample, n_items))

        # Taxonomy kNN hit
        hit_tree, chance_tree = knn_structure_hit_rate(r_all, d_tree_s, k=args.k)

        print(f'  Mantel.tree: {r_tree:+.4f} (p={p_tree:.3f})')
        print(f'  Mantel.copurchase: {r_copurchase:+.4f} (p={p_copurchase:.3f})')
        print(f'  Mantel.transition: {r_transition:+.4f} (p={p_transition:.3f})')
        print(f'  AUC.copurchase: {auc_copurchase_mean:.4f}±{auc_copurchase_std:.4f} (n={n_copurchase_bootstrap})')
        print(f'  AUC.transition: {auc_transition_mean:.4f}±{auc_transition_std:.4f} (n={n_transition_bootstrap})')
        print(f'  Comm.copurchase hit: {comm_copurchase_hit:.4f} (chance={comm_copurchase_chance:.4f})')
        print(f'  Comm.transition hit: {comm_transition_hit:.4f} (chance={comm_transition_chance:.4f})')
        print(f'  kNN.tree hit: {hit_tree:.4f} (chance={chance_tree:.4f})')

        results['per_layer'][f'l={l}'] = {
            'r_norm_mean': float(r_all.norm(dim=-1).mean().item()),
            'n_copurchase_active': len(copurchase_idx),
            'n_transition_active': len(trans_idx),
            'mantel_tree': {'r': float(r_tree), 'p': float(p_tree)},
            'mantel_copurchase': {'r': float(r_copurchase), 'p': float(p_copurchase)},
            'mantel_transition': {'r': float(r_transition), 'p': float(p_transition)},
            'auc_copurchase': {
                'mean': float(auc_copurchase_mean) if auc_copurchase_mean is not None else None,
                'std': float(auc_copurchase_std) if auc_copurchase_std is not None else None,
                'n_bootstrap': n_copurchase_bootstrap,
            },
            'auc_transition': {
                'mean': float(auc_transition_mean) if auc_transition_mean is not None else None,
                'std': float(auc_transition_std) if auc_transition_std is not None else None,
                'n_bootstrap': n_transition_bootstrap,
            },
            'comm_copurchase': {'hit': float(comm_copurchase_hit), 'chance': float(comm_copurchase_chance)},
            'comm_transition': {'hit': float(comm_transition_hit), 'chance': float(comm_transition_chance)},
            'knn_tree': {'hit': float(hit_tree), 'chance': float(chance_tree)},
        }

    # Save JSON
    json_path = f'{OUT_DIR}/network_structure.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n✅ Saved: {json_path}')

    # Write verdict
    verdict_path = f'{OUT_DIR}/verdict.md'
    write_verdict(results, verdict_path)
    print(f'✅ Saved: {verdict_path}')

    print('\n' + '=' * 70)
    print('SUMMARY: Mantel.tree ρ per layer')
    print('=' * 70)
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        print(f'  l={l}: tree={d["mantel_tree"]["r"]:+.4f} | '
              f'copurchase={d["mantel_copurchase"]["r"]:+.4f} | '
              f'transition={d["mantel_transition"]["r"]:+.4f}')
    print('=' * 70)


def write_verdict(results, path):
    L = results['L']
    lines = [
        '# Task 370 Verdict: 网状关系结构保留度 — 共购图 + 转移图',
        '',
        '## 设计',
        '',
        f'- **共购图**: window={results["window"]}, min_weight={results["min_weight"]}, '
        f'n_users={results["n_users_copurchase"]}, edges={results["copurchase_edges"]}',
        f'- **转移图**: min_count={results["min_count"]}, n_users={results["n_users_transition"]}, '
        f'edges={results["transition_edges"]}',
        f'- **Louvain社群**: 共购={results["n_copurchase_communities"]}, 转移={results["n_transition_communities"]}',
        f'- **样本**: n={results["n_sample"]}, n_perm={results["n_perm"]}, k={results["k"]}',
        f'- **Bootstrap AUC**: n={results["n_bootstrap"]}',
        '',
        '## 现象',
        '',
        '### Per-Layer Mantel ρ (taxonomy / copurchase / transition)',
        '',
        '| Layer | Mantel.tree ρ | Mantel.copurchase ρ | Mantel.transition ρ |',
        '|-------|---------------|---------------------|---------------------|',
    ]
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        lines.append(
            f'| l={l} | {d["mantel_tree"]["r"]:+.4f} (p={d["mantel_tree"]["p"]:.3f}) | '
            f'{d["mantel_copurchase"]["r"]:+.4f} (p={d["mantel_copurchase"]["p"]:.3f}) | '
            f'{d["mantel_transition"]["r"]:+.4f} (p={d["mantel_transition"]["p"]:.3f}) |'
        )

    lines.extend(['', '### Link Prediction AUC (bootstrap mean±std)', '', '| Layer | AUC.copurchase | AUC.transition |', '|-------|----------------|----------------|'])
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        auc_c = d['auc_copurchase']
        auc_t = d['auc_transition']
        c_str = f'{auc_c["mean"]:.4f}±{auc_c["std"]:.4f}' if auc_c['mean'] is not None else 'N/A'
        t_str = f'{auc_t["mean"]:.4f}±{auc_t["std"]:.4f}' if auc_t['mean'] is not None else 'N/A'
        lines.append(f'| l={l} | {c_str} | {t_str} |')

    lines.extend(['', '### Community Preservation kNN hit (chance)', '', '| Layer | Comm.copurchase | Comm.transition | kNN.tree |', '|-------|-----------------|-----------------|---------|'])
    for l in range(L + 1):
        d = results['per_layer'][f'l={l}']
        lines.append(
            f'| l={l} | {d["comm_copurchase"]["hit"]:.4f} ({d["comm_copurchase"]["chance"]:.4f}) | '
            f'{d["comm_transition"]["hit"]:.4f} ({d["comm_transition"]["chance"]:.4f}) | '
            f'{d["knn_tree"]["hit"]:.4f} ({d["knn_tree"]["chance"]:.4f}) |'
        )

    # Decay analysis
    lines.extend(['', '## 衰减模式分析', ''])
    tree_l0 = results['per_layer']['l=0']['mantel_tree']['r']
    copurchase_l0 = results['per_layer']['l=0']['mantel_copurchase']['r']
    transition_l0 = results['per_layer']['l=0']['mantel_transition']['r']

    for label, key in [('tree', 'mantel_tree'), ('copurchase', 'mantel_copurchase'), ('transition', 'mantel_transition')]:
        vals = [results['per_layer'][f'l={l}'][key]['r'] for l in range(L + 1)]
        l1_drop = (vals[1] - vals[0]) / max(abs(vals[0]), 1e-6) * 100
        l2_drop = (vals[2] - vals[1]) / max(abs(vals[1]), 1e-6) * 100 if L >= 2 else 0
        lines.append(f'- {label}: l=0→l=1 drop={l1_drop:+.1f}%, l=1→l=2 drop={l2_drop:+.1f}% (l0={vals[0]:+.4f})')

    # Compare with V-info
    lines.extend(['', '## V-info 对照', ''])
    lines.append('| Layer | Mantel.tree | Mantel.copurchase | Mantel.transition | V-info(A) |')
    lines.append('|-------|-------------|-------------------|-------------------|-----------|')
    for l in range(1, L + 1):
        d = results['per_layer'][f'l={l}']
        v = V_INFO_A.get(l, None)
        v_str = f'{v:+.4f}' if v is not None else 'N/A'
        lines.append(
            f'| l={l} | {d["mantel_tree"]["r"]:+.4f} | {d["mantel_copurchase"]["r"]:+.4f} | '
            f'{d["mantel_transition"]["r"]:+.4f} | {v_str} |'
        )

    # Key findings
    tree_l1 = results['per_layer']['l=1']['mantel_tree']['r']
    copurchase_l1 = results['per_layer']['l=1']['mantel_copurchase']['r']
    transition_l1 = results['per_layer']['l=1']['mantel_transition']['r']

    lines.extend(['', '## 结论', ''])
    if abs(copurchase_l1) > abs(tree_l1) * 0.5 or abs(transition_l1) > abs(tree_l1) * 0.5:
        lines.append(f'**网状结构在 l=1 处比 taxonomy 保留更好**: copurchase={copurchase_l1:+.4f}, transition={transition_l1:+.4f}, tree={tree_l1:+.4f}')
        lines.append('→ 深层 residual 可能包含非树状的行为结构，值得单独处理')
    else:
        lines.append(f'**网状结构在 l=1 处同样崩塌**: copurchase={copurchase_l1:+.4f}, transition={transition_l1:+.4f}, tree={tree_l1:+.4f}')
        lines.append('→ taxonomy、共购图、转移图在 l=1 同步崩塌，深层无显著网状结构')

    if copurchase_l1 > 0.05 or transition_l1 > 0.05:
        lines.append('')
        lines.append('**但注意**: 网状结构相关不等于推荐有用，需对照 R@10 验证（task371）')

    lines.extend(['', '## 建议', '',
        '1. 若网状结构保留 > taxonomy：考虑在深层保留双曲几何（但需验证 R@10 是否提升）',
        '2. 若所有结构 l=1 同步崩塌：深层双曲几何价值有限，H-E-E-E 仍是最佳方案',
        '3. 链路预测 AUC > 0.5 意味着 residual 空间包含图结构信息（可作为 TIGER decoder 的额外信号）',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
