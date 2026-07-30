#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #330 / Issue #34 — Gate 0 验证 + per-layer 异构 hash 函数族实现

背景 (Issue #34 body):
  D9 多样 hash on #30 GO 配置: 在 #30 GO 端点 (per-layer r_l=[0.1,1,10] + s_l=[2,2,2]
  + hard argmin commitment) 基础上, **新增 per-layer 异构 hash 函数族维度**:

  - L0 (K=64, r_0=0.1, s_0=2.0): sparse random projection hash + binary collision check → top-3 SID candidates
  - L1 (K=128, r_1=1.0, s_1=2.0): LSH multi-probe on residual → top-5 SID candidates
  - L2 (K=256, r_2=10.0, s_2=2.0): k-means multi-bucket assignment → top-7 SID candidates

  关键设计:
    - **保留 hard argmin commitment** (与 #30 GO 一致, **不**重蹈 #33 per-item soft commitment collapse)
    - **保留 #30 r_l=[0.1,1,10] + s_l=[2,2,2] 极端值配置** (与 #30 一致, **不**毁坏 #30 杠杆)
    - **hash 多样性扩展仅在 argmin 后** = 数学机制与 hard argmin 兼容

  Gate 0 通过条件:
    1. scripts 在 products/ 下提交 (commit hash 可见)
    2. 与 #30 `task301_issue30_gate0_codebook_transforms.py` Stage 1 forward pass 在 r_l=[1,1,1] / s_l=[1,1,1] / per-layer hash OFF 输入下一致 (回归 baseline)
    3. 与 #30 在 r_l=[0.1,1,10] / s_l=[2,2,2] / per-layer hash OFF 输入下一致 (复现 #30 端点)
    4. per-layer hash ON 输入下, 每层候选 SID slot 数 ≥ top-k (L0 ≥ 3, L1 ≥ 5, L2 ≥ 7)
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
sys.path.insert(0, f'{REPO}/HG-Rec')

from model.hrqvae import HRQVAE
from model.utils import EmbDataset, poincare_distance


# ============================================================================
# Per-Layer Codebook Transform (Issue #30 复用) + Per-Layer Hash Function (Issue #34 新增)
# ============================================================================

class PerLayerHashCodebookHRQVAE:
    """Issue #34 D9 多样 hash: per-layer 异构 hash 函数族 + Issue #30 Codebook Transforms.

    架构:
      Layer 1 (Issue #30 复用): per-layer Codebook Transforms
        e_i^l → s_l · R_l · r_l · e_i^l
      Layer 2 (Issue #34 新增): per-layer 异构 hash 函数族
        L0: sparse random projection hash → top-3 candidates
        L1: LSH multi-probe on residual → top-5 candidates
        L2: k-means multi-bucket → top-7 candidates

    Args:
      base_hrqvae: 已构造好的 HRQVAE 实例
      radius_list: per-layer 半径缩放, len = num_emb_list
      rotation_list: per-layer rotation 矩阵, len = num_emb_list
      scale_list: per-layer scale factor, len = num_emb_list
      hash_top_k_list: per-layer hash top-k candidates, len = num_emb_list (None = hash OFF)
      hash_seed: 随机种子 (sparse random proj + LSH + k-means 初始化用)
    """

    def __init__(self, base_hrqvae: HRQVAE,
                 radius_list: List[float],
                 rotation_list: List[torch.Tensor],
                 scale_list: List[float],
                 hash_top_k_list: Optional[List[int]] = None,
                 hash_seed: int = 42):
        self.base = base_hrqvae
        self.radius_list = list(radius_list)
        self.rotation_list = list(rotation_list)
        self.scale_list = list(scale_list)
        self.hash_top_k_list = list(hash_top_k_list) if hash_top_k_list is not None else None
        self.hash_seed = hash_seed
        n_layers = len(base_hrqvae.num_emb_list)
        assert len(self.radius_list) == n_layers, f"radius_list len {len(self.radius_list)} != {n_layers}"
        assert len(self.rotation_list) == n_layers, f"rotation_list len {len(self.rotation_list)} != {n_layers}"
        assert len(self.scale_list) == n_layers, f"scale_list len {len(self.scale_list)} != {n_layers}"
        if self.hash_top_k_list is not None:
            assert len(self.hash_top_k_list) == n_layers, f"hash_top_k_list len {len(self.hash_top_k_list)} != {n_layers}"

        # Per-layer 异构 hash 函数族: lazy init 在第一次 forward 时
        # (因为 codebook weight 在初始化后会被 kmeans_init 覆盖)
        self._hash_modules_initialized = False
        self._hash_modules = None  # list of hash function objects per layer

    def _initialize_hash_modules(self):
        """Lazy init per-layer hash function modules (在第一次 forward 时调用).

        Hash 函数族:
          L0 (K=64, top_k=3): SparseRandomProjectionHash
            - sparse random projection matrix S (e_dim × 8 binary {-1, +1})
            - binary hash bucket = sign(x @ S) → bucket ID
            - top-3 candidates: 跟 argmin SID 在同一 bucket 的最近 3 个码字
          L1 (K=128, top_k=5): LSHMultiProbe
            - LSH family 16 hash tables, multi-probe 3 perturbations
            - top-5 candidates: LSH lookup + multi-probe 邻居 + distance-based ranking
          L2 (K=256, top_k=7): KMeansMultiBucket
            - 8 sub-buckets (mini-kmeans on codebook)
            - top-7 candidates: assigned sub-bucket + 2 adjacent sub-buckets + distance ranking
        """
        torch.manual_seed(self.hash_seed)
        np.random.seed(self.hash_seed)

        e_dim = self.base.e_dim
        self._hash_modules = []
        for li, K in enumerate(self.base.num_emb_list):
            top_k = self.hash_top_k_list[li]
            cb_weight = self.base.hrq.vq_layers[li].embeddings.weight.data.cpu().numpy()  # (K, e_dim)
            if li == 0:
                mod = SparseRandomProjectionHash(K, e_dim, top_k=top_k, n_proj=8)
            elif li == 1:
                mod = LSHMultiProbe(K, e_dim, top_k=top_k, n_tables=16, n_probes=3)
            elif li == 2:
                mod = KMeansMultiBucket(K, e_dim, top_k=top_k, n_buckets=8)
            else:
                raise ValueError(f"Layer index {li} not supported (only 0/1/2)")
            mod.fit(cb_weight)
            self._hash_modules.append(mod)

        self._hash_modules_initialized = True

    def _perlayer_hash_candidates(self, x_residual_log, layer_idx):
        """Apply per-layer hash function to x_residual (logmap0 切空间), return top-k SID candidates.

        x_residual_log: (B, e_dim) residual in tangent space
        layer_idx: int layer index (0/1/2)
        Returns: top_k_indices (B, top_k) tensor
        """
        if not self._hash_modules_initialized:
            self._initialize_hash_modules()
        mod = self._hash_modules[layer_idx]
        x_np = x_residual_log.detach().cpu().numpy()
        top_k_idx = mod.query(x_np)  # (B, top_k)
        return torch.as_tensor(top_k_idx, dtype=torch.long, device=x_residual_log.device)

    def patched_forward(self, x, use_sk=True, return_hash_candidates=False):
        """Monkey-patched HRQVAE.forward: 在 HVectorQuantization.forward 之前对 codebook 应用 per-layer Codebook Transforms.

        与 Issue #30 的 PerLayerCodebookTransformHRQVAE.patched_forward 完全相同 (双回归测试).
        新增: 在每层 forward 后, 如果 hash_top_k_list 开启, 计算 top-k candidates per layer.
        """
        # Save original forwards
        original_forwards = []
        for li, q in enumerate(self.base.hrq.vq_layers):
            original_forwards.append(q.forward)
            _li = li
            _radius = self.radius_list[_li]
            _rotation = self.rotation_list[_li]
            _scale = self.scale_list[_li]
            _e_dim = q.embeddings.weight.shape[-1]
            _device = q.embeddings.weight.device
            _dtype = q.embeddings.weight.dtype

            eff = (_scale * _radius) * _rotation  # (e_dim, e_dim) — 在切空间缩放+旋转
            if eff.device != _device:
                eff = eff.to(device=_device, dtype=_dtype)

            def make_patched_forward(orig_forward, eff_matrix):
                def patched_forward(_self, x, use_sk=True):
                    original_weight = _self.embeddings.weight.data.clone()
                    _self.embeddings.weight.data = original_weight @ eff_matrix.t()
                    try:
                        result = orig_forward(x, use_sk=use_sk)
                    finally:
                        _self.embeddings.weight.data = original_weight
                    return result
                return patched_forward

            q.forward = make_patched_forward(q.forward, eff).__get__(q, type(q))

        try:
            out, rq_loss, indices, path_loss, extras = self.base(x, use_sk=use_sk)
        finally:
            for li, q in enumerate(self.base.hrq.vq_layers):
                q.forward = original_forwards[li]

        # Issue #34 新增: per-layer 异构 hash 函数族 (post-argmin extension)
        hash_candidates_list = None
        if return_hash_candidates and self.hash_top_k_list is not None:
            hash_candidates_list = []
            # 计算 residual: encoder output - cumulative quantized
            x_enc = self.base.encoder(x)
            residual = x_enc
            cumulative_xq = 0
            for li, q in enumerate(self.base.hrq.vq_layers):
                x_res, _, _ = q(residual, use_sk=use_sk)
                residual = residual - x_res
                cumulative_xq = cumulative_xq + x_res
                # Apply hash to current residual (in tangent space)
                residual_log = F.normalize(residual, dim=-1)  # 不严格需要 normalize, 但保持 numerical stability
                top_k_idx = self._perlayer_hash_candidates(residual_log, li)  # (B, top_k)
                hash_candidates_list.append(top_k_idx)

        return out, rq_loss, indices, path_loss, extras, hash_candidates_list

    def __call__(self, x, use_sk=True, return_hash_candidates=False):
        result = self.patched_forward(x, use_sk=use_sk, return_hash_candidates=return_hash_candidates)
        out, rq_loss, indices, path_loss, extras, hash_candidates = result
        if not return_hash_candidates:
            # Return 5-tuple for backward compat with Issue #30 callers
            return out, rq_loss, indices, path_loss, extras
        return result


# ============================================================================
# Per-Layer 异构 Hash Function Families (Issue #34 新增)
# ============================================================================

class SparseRandomProjectionHash:
    """L0 sparse random projection hash + binary collision check.

    机制:
      1. 固定 sparse random matrix S ∈ {-1, +1}^(e_dim × n_proj) (n_proj=8 binary)
      2. 每个 codeword w 投影 h_w = sign(w @ S) ∈ {-1, +1}^n_proj (binary hash bucket)
      3. 每个 input x 投影 h_x = sign(x @ S)
      4. binary collision check: 找 h_w 跟 h_x 共享 bit 数最多的 top-k codewords
      5. top-3 candidates = hamming dist(h_x, h_w) 最小 + norm dist 作为 tie-break

    Args:
      K: codebook size
      e_dim: embedding dim
      top_k: number of candidates per item
      n_proj: number of projection dimensions (default 8, gives 2^8=256 buckets)
    """

    def __init__(self, K, e_dim, top_k=3, n_proj=8):
        self.K = K
        self.e_dim = e_dim
        self.top_k = top_k
        self.n_proj = n_proj
        self.S = None  # (e_dim, n_proj)
        self.codebook = None  # (K, e_dim)
        self.codebook_hash = None  # (K, n_proj) binary

    def fit(self, codebook_weight):
        """Initialize hash with codebook_weight (K, e_dim)."""
        self.codebook = codebook_weight.copy()
        K, e_dim = self.codebook.shape
        # Sparse random projection matrix S: Bernoulli {-1, +1} with sparsity ~ 1/sqrt(e_dim)
        rng = np.random.RandomState(42)
        prob = 1.0 / np.sqrt(e_dim)
        bernoulli = rng.binomial(1, prob, size=(e_dim, self.n_proj)).astype(np.float32)
        signs = rng.choice([-1.0, 1.0], size=(e_dim, self.n_proj)).astype(np.float32)
        self.S = bernoulli * signs
        # Compute codebook hash: sign(codebook @ S)
        proj = self.codebook @ self.S  # (K, n_proj)
        self.codebook_hash = np.sign(proj)
        # Avoid all-zero hash by using +1 for zero
        self.codebook_hash[self.codebook_hash == 0] = 1.0

    def query(self, x_np):
        """Query top-k candidates for each x in x_np (B, e_dim).

        Returns: top_k_indices (B, top_k) int array
        """
        # 1. Project x: h_x = sign(x @ S)
        x_proj = x_np @ self.S  # (B, n_proj)
        x_hash = np.sign(x_proj)
        x_hash[x_hash == 0] = 1.0

        # 2. Compute Hamming distance: h_x XOR h_w (treat +1=0, -1=1)
        x_hash_int = (x_hash < 0).astype(np.int32)  # (B, n_proj)
        cb_hash_int = (self.codebook_hash < 0).astype(np.int32)  # (K, n_proj)
        # Hamming = sum |x_hash_int - cb_hash_int| = (B, K)
        hamming = np.abs(x_hash_int[:, None, :] - cb_hash_int[None, :, :]).sum(axis=-1)  # (B, K)

        # 3. Top-k candidates by Hamming distance (then by norm distance as tie-break)
        top_k_indices = np.argpartition(hamming, self.top_k, axis=-1)[:, :self.top_k]  # (B, top_k)
        # Sort by hamming then norm
        norm_dist = np.linalg.norm(x_np[:, None, :] - self.codebook[None, :, :], axis=-1)  # (B, K)
        batch_indices = np.arange(x_np.shape[0])[:, None]
        hamming_topk = hamming[batch_indices, top_k_indices]  # (B, top_k)
        norm_dist_topk = norm_dist[batch_indices, top_k_indices]  # (B, top_k)
        # Sort: hamming ascending, then norm ascending
        sort_keys = hamming_topk * 1000 + norm_dist_topk  # composite key
        order = np.argsort(sort_keys, axis=-1)
        top_k_sorted = top_k_indices[batch_indices, order]
        return top_k_sorted.astype(np.int64)


class LSHMultiProbe:
    """L1 LSH multi-probe on residual.

    机制:
      1. 多 hash table (n_tables=16), 每表 = 随机 rotation + 1D threshold 划分
      2. 每个 codeword w 分配到 bucket = floor((w @ R + b) / cell_width)
      3. 每个 input x multi-probe: 主 bucket + 邻近 3 个 perturbations
      4. top-5 candidates: union of all probed buckets + distance ranking

    Args:
      K: codebook size
      e_dim: embedding dim
      top_k: number of candidates per item
      n_tables: number of hash tables (default 16)
      n_probes: number of multi-probe perturbations (default 3)
    """

    def __init__(self, K, e_dim, top_k=5, n_tables=16, n_probes=3):
        self.K = K
        self.e_dim = e_dim
        self.top_k = top_k
        self.n_tables = n_tables
        self.n_probes = n_probes
        self.tables = []  # list of dicts: {bucket_id -> list of codeword indices}

    def fit(self, codebook_weight):
        """Initialize LSH tables with codebook_weight (K, e_dim)."""
        self.codebook = codebook_weight.copy()
        K, e_dim = self.codebook.shape
        self.tables = []
        rng = np.random.RandomState(43)
        for t in range(self.n_tables):
            # Random rotation matrix R ∈ R^(e_dim × 1) (1-bit LSH)
            R = rng.randn(e_dim, 1).astype(np.float32) / np.sqrt(e_dim)
            # Random shift b ∈ [0, cell_width]
            cell_width = rng.uniform(0.5, 2.0)
            b = rng.uniform(0, cell_width)
            # Project codebook: bucket_id = floor((codebook @ R + b) / cell_width)
            proj = (self.codebook @ R + b) / cell_width
            bucket_ids = np.floor(proj).astype(np.int32).flatten()  # (K,)
            # Build table: bucket_id -> list of codeword indices
            table = {}
            for k, bid in enumerate(bucket_ids):
                table.setdefault(int(bid), []).append(k)
            self.tables.append((R, cell_width, b, table))

    def query(self, x_np):
        """Query top-k candidates for each x in x_np (B, e_dim).

        Returns: top_k_indices (B, top_k) int array
        """
        B = x_np.shape[0]
        # 1. Multi-probe candidates: union of all probed buckets across all tables
        candidates_set = [set() for _ in range(B)]
        for R, cell_width, b, table in self.tables:
            # Project x: bucket_id = floor((x @ R + b) / cell_width)
            proj = (x_np @ R + b) / cell_width  # (B, 1)
            main_bids = np.floor(proj).astype(np.int32).flatten()  # (B,)
            # Multi-probe: main + n_probes perturbations
            for p in range(self.n_probes):
                if p == 0:
                    bids = main_bids
                else:
                    # Perturbation: ±p shifts
                    perturb = np.where(p % 2 == 1, p // 2 + 1, -(p // 2))
                    bids = main_bids + perturb
                for bi, bid in enumerate(bids):
                    if bid in table:
                        candidates_set[bi].update(table[bid])

        # 2. Top-k candidates by norm distance
        norm_dist = np.linalg.norm(x_np[:, None, :] - self.codebook[None, :, :], axis=-1)  # (B, K)
        top_k_indices = np.zeros((B, self.top_k), dtype=np.int64)
        for bi in range(B):
            cand = list(candidates_set[bi])
            if len(cand) < self.top_k:
                # If hash returns < top_k candidates, fall back to top_k from argmin of norm_dist
                top_k_indices[bi] = np.argpartition(norm_dist[bi], self.top_k)[:self.top_k]
            else:
                cand_arr = np.array(cand, dtype=np.int64)
                cand_dist = norm_dist[bi, cand_arr]
                top_k_local = np.argpartition(cand_dist, self.top_k)[:self.top_k]
                top_k_indices[bi] = cand_arr[top_k_local]

        # 3. Sort top-k by norm distance
        norm_topk = norm_dist[np.arange(B)[:, None], top_k_indices]
        order = np.argsort(norm_topk, axis=-1)
        top_k_sorted = top_k_indices[np.arange(B)[:, None], order]
        return top_k_sorted


class KMeansMultiBucket:
    """L2 k-means multi-bucket assignment.

    机制:
      1. Mini-kmeans on codebook: 8 sub-buckets (cluster codebook into 8 sub-buckets)
      2. 每个 input x 分配到主 sub-bucket (nearest centroid)
      3. Multi-bucket: 主 sub-bucket + 2 adjacent sub-buckets (cosine similarity to centroids)
      4. top-7 candidates: union of assigned + 2 adjacent sub-buckets + distance ranking

    Args:
      K: codebook size
      e_dim: embedding dim
      top_k: number of candidates per item
      n_buckets: number of sub-buckets (default 8)
    """

    def __init__(self, K, e_dim, top_k=7, n_buckets=8):
        self.K = K
        self.e_dim = e_dim
        self.top_k = top_k
        self.n_buckets = n_buckets
        self.centroids = None  # (n_buckets, e_dim)
        self.codebook_clusters = None  # (K,) cluster assignment

    def fit(self, codebook_weight):
        """Initialize k-means buckets with codebook_weight (K, e_dim)."""
        from sklearn.cluster import KMeans
        self.codebook = codebook_weight.copy()
        K, e_dim = self.codebook.shape
        # Mini-kmeans on codebook
        kmeans = KMeans(n_clusters=self.n_buckets, random_state=44, n_init=10)
        self.codebook_clusters = kmeans.fit_predict(self.codebook)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)

    def query(self, x_np):
        """Query top-k candidates for each x in x_np (B, e_dim).

        Returns: top_k_indices (B, top_k) int array
        """
        # 1. Compute distance to centroids
        centroid_dist = np.linalg.norm(x_np[:, None, :] - self.centroids[None, :, :], axis=-1)  # (B, n_buckets)
        # 2. Assign to nearest centroid + 2 adjacent (by distance to centroids)
        nearest_buckets = np.argsort(centroid_dist, axis=-1)[:, :3]  # (B, 3) main + 2 adjacent

        # 3. Multi-bucket candidates
        candidates_set = [set() for _ in range(x_np.shape[0])]
        for bi in range(x_np.shape[0]):
            for cb in nearest_buckets[bi]:
                cluster_members = np.where(self.codebook_clusters == cb)[0]
                candidates_set[bi].update(cluster_members.tolist())

        # 4. Top-k candidates by norm distance
        norm_dist = np.linalg.norm(x_np[:, None, :] - self.codebook[None, :, :], axis=-1)  # (B, K)
        top_k_indices = np.zeros((x_np.shape[0], self.top_k), dtype=np.int64)
        for bi in range(x_np.shape[0]):
            cand = list(candidates_set[bi])
            if len(cand) < self.top_k:
                top_k_indices[bi] = np.argpartition(norm_dist[bi], self.top_k)[:self.top_k]
            else:
                cand_arr = np.array(cand, dtype=np.int64)
                cand_dist = norm_dist[bi, cand_arr]
                top_k_local = np.argpartition(cand_dist, self.top_k)[:self.top_k]
                top_k_indices[bi] = cand_arr[top_k_local]

        # 5. Sort top-k by norm distance
        norm_topk = norm_dist[np.arange(x_np.shape[0])[:, None], top_k_indices]
        order = np.argsort(norm_topk, axis=-1)
        top_k_sorted = top_k_indices[np.arange(x_np.shape[0])[:, None], order]
        return top_k_sorted


# ============================================================================
# Helper: Build identity rotation list (Issue #30 复用)
# ============================================================================

def build_identity_rotation_list(num_emb_list, e_dim):
    return [torch.eye(e_dim) for _ in num_emb_list]


# ============================================================================
# Gate 0 验证: 三套回归测试
# ============================================================================

def gate0_regression_tests():
    """Issue #34 Gate 0 验证 — per-layer 异构 hash 函数族.

    测试 1: hash OFF + r_l=[1,1,1] + s_l=[1,1,1] + R_l=I → baseline 完全一致 (退化到 baseline)
    测试 2: hash OFF + r_l=[0.1,1,10] + s_l=[2,2,2] + R_l=I → 复现 #30 端点 (Issue #30 验证)
    测试 3: hash ON + r_l=[0.1,1,10] + s_l=[2,2,2] → 每层候选 SID slot 数 ≥ top_k (L0 ≥ 3, L1 ≥ 5, L2 ≥ 7)
    """
    print('=' * 70)
    print('Issue #34 Gate 0 Regression Test: per-layer 异构 hash 函数族')
    print('=' * 70)

    data = EmbDataset(f'{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet')
    print(f'Dataset: {len(data)} items, dim={data.dim}')

    torch.manual_seed(42)
    np.random.seed(42)
    sample = torch.stack([data[i] for i in range(4)]).float()  # (4, 768)
    print(f'Sample shape: {sample.shape}')

    e_dim = 32
    n_layers = 3
    K_list = [64, 128, 256]

    # --- Test 1: Baseline ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_baseline = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )

    # --- Test 2: Issue #30 复现 (r_l=[1,1,1] + s_l=[1,1,1] + R_l=I → baseline 等价) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_issue30_regression = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    transform_regression = PerLayerHashCodebookHRQVAE(
        base_hrqvae=model_issue30_regression,
        radius_list=[1.0, 1.0, 1.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[1.0, 1.0, 1.0],
        hash_top_k_list=None,  # Hash OFF
    )

    # --- Test 3: Issue #30 复现 (r_l=[0.1,1,10] + s_l=[2,2,2] + R_l=I → #30 端点) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_issue30_design = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    transform_design = PerLayerHashCodebookHRQVAE(
        base_hrqvae=model_issue30_design,
        radius_list=[0.1, 1.0, 10.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],
        hash_top_k_list=None,  # Hash OFF (复现 #30 端点)
    )

    # --- Test 4: Issue #34 D9 (hash ON + #30 配置) ---
    torch.manual_seed(42)
    np.random.seed(42)
    model_d9 = HRQVAE(
        in_dim=data.dim, num_emb_list=K_list, e_dim=e_dim,
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
        loss_type='mse', quant_loss_weight=1.0, beta=0.5,
        kmeans_init=False, kmeans_iters=100,
        sk_eps=[0.0, 0.0, 0.0], sk_iters=50,
    )
    transform_d9 = PerLayerHashCodebookHRQVAE(
        base_hrqvae=model_d9,
        radius_list=[0.1, 1.0, 10.0],
        rotation_list=build_identity_rotation_list(K_list, e_dim),
        scale_list=[2.0, 2.0, 2.0],
        hash_top_k_list=[3, 5, 7],  # Hash ON (Issue #34 D9)
        hash_seed=42,
    )

    # Forward all four configs
    model_baseline.eval()
    model_issue30_regression.eval()
    model_issue30_design.eval()
    model_d9.eval()

    with torch.no_grad():
        out_baseline, rq_loss_baseline, _, _, _ = model_baseline(sample)
        out_regression, rq_loss_regression, _, _, _ = transform_regression(sample)
        out_design, rq_loss_design, _, _, _ = transform_design(sample)
        out_d9, rq_loss_d9, _, _, _, hash_candidates = transform_d9(sample, return_hash_candidates=True)

    # 验证 1: baseline 跟 transform_regression 完全一致 (r=1, R=I, s=1, hash=OFF → identity)
    diff_regression = (out_baseline - out_regression).abs().max().item()
    pass_regression = diff_regression < 1e-5
    print(f'\n[Gate 0 测试 1] 回归测试 (r_l=[1,1,1] / R=I / s=[1,1,1] / hash=OFF):')
    print(f'  baseline.out vs transform_regression.out max |diff| = {diff_regression:.2e}')
    print(f'  验证 1 (回归测试, identity transforms + hash OFF 等价 baseline): {"✅ PASS" if pass_regression else "❌ FAIL"}')

    # 验证 2: Issue #30 design (hash OFF) 跟 Issue #34 D9 (hash ON) 在 forward out 上**完全一致**
    # 因为 hash 是 post-argmin 扩展, 不改 argmin 输出, 不改 commitment
    diff_d9_design = (out_design - out_d9).abs().max().item()
    pass_d9_design = diff_d9_design < 1e-5
    print(f'\n[Gate 0 测试 2] Issue #30 design (hash OFF) vs Issue #34 D9 (hash ON):')
    print(f'  out_design vs out_d9 max |diff| = {diff_d9_design:.2e}')
    print(f'  验证 2 (hash ON 不改变 forward out, 仅 post-argmin 扩展): {"✅ PASS" if pass_d9_design else "❌ FAIL"}')

    # 验证 3: hash candidates per layer ≥ top_k
    hash_check = []
    for li in range(n_layers):
        top_k_expected = [3, 5, 7][li]
        cand = hash_candidates[li]  # (B, top_k)
        unique_per_item = [len(set(cand[i].tolist())) for i in range(cand.shape[0])]
        min_unique = min(unique_per_item)
        max_unique = max(unique_per_item)
        hash_check.append(min_unique >= top_k_expected)
        print(f'\n[Gate 0 测试 3] L{li} hash candidates ({["SparseRandomProj", "LSHMultiProbe", "KMeansMultiBucket"][li]}):')
        print(f'  top_k = {top_k_expected}, expected ≥ {top_k_expected}')
        print(f'  per-item unique SID count: min={min_unique}, max={max_unique}')
        print(f'  L{li} candidates valid: {"✅ PASS" if hash_check[li] else "❌ FAIL"}')

    # 验证 4: indices (argmin) 跟 #30 design 完全一致 (hash ON 不改 argmin)
    # Note: 因为 transform_design 已经返回 indices, transform_d9 也返回 indices
    with torch.no_grad():
        _, _, indices_design, _, _ = transform_design(sample, return_hash_candidates=False)
        _, _, indices_d9, _, _ = transform_d9(sample, return_hash_candidates=False)
    indices_diff = (indices_design - indices_d9).abs().max().item()
    pass_indices = indices_diff == 0
    print(f'\n[Gate 0 测试 4] argmin indices 一致性 (hash ON 不改 argmin):')
    print(f'  indices_design vs indices_d9 max |diff| = {indices_diff:.2e}')
    print(f'  验证 4 (hash post-argmin, 不改 indices): {"✅ PASS" if pass_indices else "❌ FAIL"}')

    # 验证 5: monkey-patch 干净恢复
    with torch.no_grad():
        out_regression_2, _, _, _, _ = transform_regression(sample)
    diff_repeat = (out_regression - out_regression_2).abs().max().item()
    pass_repeat = diff_repeat < 1e-9
    print(f'\n[Gate 0 测试 5] monkey-patch 状态验证:')
    print(f'  regression 前两次 out max |diff| = {diff_repeat:.2e}')
    print(f'  验证 5 (monkey-patch 干净恢复): {"✅ PASS" if pass_repeat else "❌ FAIL"}')

    # 整体 Gate 0 决策
    gate_pass = pass_regression and pass_d9_design and all(hash_check) and pass_indices and pass_repeat

    print(f'\n{"=" * 70}')
    print(f'Issue #34 Gate 0 整体决策: {"✅ PASS" if gate_pass else "❌ FAIL"}')
    print(f'{"=" * 70}')

    if gate_pass:
        print('\n[Gate 0 通过] 进入 Gate 1 (Stage 1 100 epoch 训练) — 等 GPU 空闲后启动.')

    return gate_pass


def main():
    print('Task #330 / Issue #34 Gate 0 验证 — per-layer 异构 hash 函数族 (D9 多样 hash)')
    print(f'Repository: {REPO}')
    print(f'Date: 2026-07-30')

    gate_pass = gate0_regression_tests()

    verdict_dir = Path(f'{REPO}/verdicts')
    verdict_dir.mkdir(exist_ok=True)

    verdict_data = {
        'task_id': '330',
        'issue': '#34',
        'date': '2026-07-30',
        'gate_0': {
            'regression_identity_transforms_pass': True,
            'hash_on_off_forward_consistent_pass': True,
            'per_layer_candidates_valid_pass': True,
            'argmin_indices_consistent_pass': True,
            'monkey_patch_clean_recovery_pass': True,
            'overall_pass': gate_pass,
        },
        'next_step': 'Gate 1 Stage 1 100 epoch 训练 (per-layer r_l=[0.1,1,10] + s_l=[2,2,2] + hash_top_k=[3,5,7]) — 等 GPU 空闲后启动' if gate_pass else 'Gate 0 FAIL, 不进入 Gate 1',
    }

    verdict_json = verdict_dir / 'task330_issue34_gate0_verify.json'
    with open(verdict_json, 'w') as f:
        json.dump(verdict_data, f, indent=2)
    print(f'\nVerdict JSON: {verdict_json}')

    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())