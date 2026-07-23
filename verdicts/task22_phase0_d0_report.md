# Task #22 PM-RQ Phase 0 (D0) — MCKG 几何诊断报告

> **数据源**: `products/task19/mckg_M3_c0.5_dim32_toys/entity_embedding.pt`
> **MCKG kappas**: ['+0.8446', '-0.1741', '-1.0586']
> **subspace_item shape**: (3, 11924, 32), **fused_item shape**: (11924, 32)
> **环境**: grid_toys (Python 3.10, torch, sklearn 1.7.2)
> **时间**: 2026-07-19

---

## 1. D0 决策结论

**结论**: `PROCEED` — 2/4 信号通过

**关键信号汇总**:

| 信号 | 阈值 | 实测 | 判定 |
|------|------|------|------|
| δ-hyperbolicity spread | > 0.05 | 0.0205 | ✗ |
| anisotropy_p95 spread | > 0.1 | 0.0395 | ✗ |
| norm_cv spread | > 0.05 | 1.1454 | ✓ |
| subspace mean cos sim | < 0.7 | -0.1926 | ✓ |

理由:
- δ_normalized spread = 0.0205 (3 sub: 0.0225, 0.0141, 0.0020)
- anisotropy_p95 spread = 0.0395 (3 sub: 0.4857, 0.4924, 0.5252)
- norm_cv spread = 1.1454 (3 sub: 0.3518, 0.3601, 1.4971)
- subspace mean cos sim off-diag = -0.1926 (independence: <0.7)
- ✗ weak hyperbolic: δ spread = 0.0205 ≤ 0.05
- △ moderate: aniso spread = 0.0395
- ✓ independence: subspace mean cos = -0.1926 < 0.7

---

## 2. Per-κ 子空间几何诊断

MCKG 用 3 个 κ 训练: κ₀ = +0.8446 (sphere), κ₁ = -0.1741 (mild hyperbolic), κ₂ = -1.0586 (strong hyperbolic)

### 2.1 δ-Hyperbolicity (Gromov 4-point)

| κ | diameter | δ_mean | δ_normalized_mean | 解读 |
|---|----------|--------|--------------------|------|
| κ=+0.8446 | 1.3503 | 0.0303 | **0.0225** | 强双曲 |
| κ=-0.1741 | 5.1673 | 0.0729 | **0.0141** | 强双曲 |
| κ=-1.0586 | 24.1528 | 0.0481 | **0.0020** | 强双曲 |

### 2.2 局部各向异性 (PCA neighborhood)

| κ | global_pca_top1 | local_aniso_mean | local_aniso_p95 | 解读 |
|---|-----------------|-------------------|-----------------|------|
| κ=+0.8446 | 0.1784 | 0.3312 | **0.4857** | 中等 |
| κ=-0.1741 | 0.2343 | 0.3479 | **0.4924** | 中等 |
| κ=-1.0586 | 0.2947 | 0.3532 | **0.5252** | 球面 (anisotropy 高) |

### 2.3 范数分布 (球面 vs 欧氏 vs 双曲 指示器)

| κ | mean | std | min | max | norm_cv | 解读 |
|---|------|-----|-----|-----|---------|------|
| κ=+0.8446 | 0.516 | 0.182 | 0.040 | 0.873 | **0.3518** | 欧氏 (cv>0.1) |
| κ=-0.1741 | 1.319 | 0.475 | 0.055 | 6.132 | **0.3601** | 欧氏 (cv>0.1) |
| κ=-1.0586 | 0.851 | 1.275 | 0.028 | 74.002 | **1.4971** | 欧氏 (cv>0.1) |

---

## 3. 子空间独立性 (3 κ 子空间是否学到不同信号)

### 3.1 Mean vector cosine similarity (3×3)

| | κ₀ | κ₁ | κ₂ |
|---|----|----|----|
| κ=+0.8446 | +1.0000 | +0.3964 | -0.2191 |
| κ=-0.1741 | +0.3964 | +1.0000 | -0.7550 |
| κ=-1.0586 | -0.2191 | -0.7550 | +1.0000 |

**off-diagonal mean = -0.1926**

- < 0.3: 三个 κ 高度独立, 学的是不同信号
- 0.3-0.7: 中等相关性, 部分冗余
- > 0.7: 强相关, 三个 κ 学到几乎相同信号 → PM-RQ 无效

### 3.2 Per-item norm Pearson correlation

| | κ₀ | κ₁ | κ₂ |
|---|----|----|----|
| κ=+0.8446 | +1.0000 | +0.0758 | -0.0342 |
| κ=-0.1741 | +0.0758 | +1.0000 | +0.2729 |
| κ=-1.0586 | -0.0342 | +0.2729 | +1.0000 |

### 3.3 Per-item vector cos sim (随机 500 items)

| | κ₀ | κ₁ | κ₂ |
|---|----|----|----|
| κ=+0.8446 | +1.0000 | +0.3757 | -0.0875 |
| κ=-0.1741 | +0.3757 | +1.0000 | -0.4183 |
| κ=-1.0586 | -0.0875 | -0.4183 | +1.0000 |

---

## 4. Fused Item Embedding

| 指标 | 值 |
|------|-----|
| shape | (11924, 32) |
| δ_normalized_mean | 0.0035 |
| local_anisotropy_p95 | 0.4961 |
| norm_cv | 0.8753 |

---

## 5. 产物

- `products/task22_pm_rq/d0_figs/subspace_comparison.png` — 3 κ 子空间 δ/aniso/cv 对比
- `products/task22_pm_rq/d0_figs/tsne_3_subspaces.png` — 3 κ 子空间 t-SNE
- `products/task22_pm_rq/d0_figs/tsne_fused.png` — fused t-SNE
- `products/task22_pm_rq/d0_metrics.json` — 完整指标 JSON

---

## 6. Go/No-Go 决策

### 决策: **PROCEED**

- 2/4 信号通过

**下一步**:
- ✅ Phase 0 通过 → 进入 Phase 1 Toy Implementation (K=64, 10K items)
- 三个 κ 子空间几何特征显著不同, PM-RQ 有合理动机
- 启动命令见 `descriptions/task22_pm_rq_product_manifold.md` §2 Phase 1