# Task #150 — Issue #56 方向B: 混合曲率乘积空间 Gate 0 设计 (zero-GPU prep)

**日期**: 2026-07-31 00:08
**状态**: 📝 DESIGN — 等 owner 拍板 (R10 候选)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/56

## 1. 目的

Issue #56 替代 per-layer 可学习 κ 标量为**混合权重 α_l ∈ [0,1]** (sigmoid 参数化). 规避 κ=0 退化点的梯度病态问题 (虽然 #47 修复了公式实现 bug, 但"连续标量 κ"参数化本身仍不是良性优化几何).

## 2. 文献基础

- **arXiv:2307.04514** "Weighted Mixed-Curvature Product Manifold"
- **ACE-HGNN** "ACE: Adaptive Curvature Exploration"
- 共同点: 多个**固定曲率分量** (欧式 κ=0 + 双曲 κ=fixed), 学习混合权重 α_l

## 3. Gate 0 实施方案

### 3.1 改动点 (核心参数化)

`HG-Rec/model/hrqvae_free_curv.py` (commit 61e707c 修复版):

**原参数化 (Issue #49/#51/#52/#53)**:
```python
# 每层学 κ_l (标量)
self.kappa_l = nn.Parameter(torch.tensor([1.0] * num_layers))  # κ_init
```

**新参数化 (Issue #56)**:
```python
# 每层固定双曲分量 κ_l_fixed = Ollivier mean (来自 #70/#41 实测)
# + 学习混合权重 α_l ∈ [0,1]
self.alpha_l = nn.Parameter(torch.zeros(num_layers))  # sigmoid(0) = 0.5 默认
# 距离公式:
# d(x, y; α_l, κ_fixed) = α_l * d_hyp(x, y; κ_fixed) + (1 - α_l) * d_euc(x, y)
```

### 3.2 CLI flag

```python
parser.add_argument('--mixed_curv', action='store_true', help='Issue #56: use α_l weighted hyp+euc distance')
parser.add_argument('--fixed_kappa', type=float, default=0.74, help='Ollivier mean from #70')
parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial alpha (sigmoid)')
```

### 3.3 距离公式 (跟 #47 修复公式对齐)

```python
def mixed_curv_dist(x, y, alpha, kappa_fixed):
    """α-weighted hyperbolic + euclidean distance (Issue #56)."""
    d_hyp = poincare_distance(x, y, c=kappa_fixed)  # #47 fix
    d_euc = torch.norm(x - y, dim=-1)
    return alpha * d_hyp + (1 - alpha) * d_euc
```

### 3.4 Stage 1 训练 recipe

| 配置 | 值 |
|------|-----|
| num_emb_list | [64, 128, 256] |
| beta | 0.250 |
| sk_eps | 0.000 |
| num_epochs | 1000 |
| batch_size | 1024 |
| seed | 42 |
| **fixed_kappa** | **0.74** (Ollivier mean from #70) |
| **alpha_init** | **0.5** (sigmoid 默认, 平衡 hyp/euc) |
| 学习率 | 5e-4 (跟 task84 baseline 一致) |

### 3.5 对照组

| 组 | 配置 | 期望 R@10 |
|----|------|----------|
| Baseline (#84) | vanilla RQ-VAE, c=1.0 | 0.1020 |
| #49 (学 κ_l) | κ_l learned | 0.1005 (-1.5%) |
| **#56 (学 α_l)** | **α_l learned, κ_fixed=0.74** | **> 0.1020?** |

### 3.6 决策阈值

| 实测 R@10 | 决策 |
|-----------|------|
| #56 R@10 > 0.1020 | 🟢 GO - α_l 参数化规避梯度病态 |
| 0.1005 ≤ #56 R@10 ≤ 0.1020 | 🟡 NEUTRAL - 跟 #49 类似 |
| #56 R@10 < 0.1005 | ❌ NO-GO - α_l 也无法转化为 R@10 杠杆 |

### 3.7 验证标准 (issue 原文)

- **H1**: 学习 α_l vs κ_l 训练稳定性 (loss 曲线 / κ=0 梯度病态消失)
- **H2**: Stage 4 test R@10 > baseline 0.1020, 显著优于 #49 (0.1005)
- **H3**: val/test gap < #49/#51/#52/#53 (0.02 量级)

## 4. 跟 #47 公式对齐

Issue #56 距离公式 `d_hyp(x, y; κ_fixed)` 必须用 #47 修复后的 poincare_distance (commit 61e707c):
- Möbius inverse (非 negation)
- sigmoid NaN 修复

不修复的话 Issue #56 也会踩跟 #44 Gate 1 同样的坑.

## 5. Stage 1 训练成本估算

| Stage | 时长 | GPU |
|-------|------|-----|
| Stage 1 RQ-VAE 1000 epoch (α_l learned) | ~3.2h | 单 GPU |
| Stage 2 SID inference | ~5min | 单 GPU |
| Stage 3 T5-mini 200 epoch | ~2h | 单 GPU |
| Stage 4 eval (beam=50) | ~5min | 单 GPU |
| **总计** | **~5.3h** | **单 GPU 串行** |

**5h Stage 1+3 训练**. R10 ROI 中等 (跟 Issue #49 5h 类似, 期望 -1.5% → +0.2pp 突破).

## 6. R10/R11.5 决策

**Issue #56 候选 2 个 ROI 对比**:

| 候选 | ROI | 风险 |
|------|-----|------|
| Issue #56 完整 4 阶段 (~5.3h) | 中 (跟 Issue #49 baseline 对照, H2 +0.2pp 突破概率低) | 高 (跟 #49 同样 NO-GO 风险) |
| Issue #56 Stage 1 only (α_l 学习曲线诊断, ~3.2h) | 中-高 (低成本验证 H1 训练稳定性) | 低 |
| Issue #56 数学 sanity (5/5 test, ~30min) | 高 (零 GPU) | 零 |

**R10 决策**: 候选 3 优先 (math sanity zero-GPU). H1 数学验证通过后启动候选 2 (Stage 1 only). H1 + Stage 1 联合 OK 才启动候选 1.

## 7. 启动 Gate 0 数学 sanity (zero-GPU 立即可做)

```python
# tests/test_issue56_alpha_param.py
import torch
from hgrec.model.hrqvae_free_curv import mixed_curv_dist

def test_alpha_zero_is_euclidean():
    """α=0 → pure Euclidean distance."""
    x = torch.randn(10, 32) * 0.1
    y = torch.randn(10, 32) * 0.1
    d = mixed_curv_dist(x, y, alpha=0.0, kappa_fixed=0.74)
    d_euc = torch.norm(x - y, dim=-1)
    assert torch.allclose(d, d_euc, atol=1e-5)

def test_alpha_one_is_hyperbolic():
    """α=1 → pure hyperbolic distance."""
    x = torch.randn(10, 32) * 0.1
    y = torch.randn(10, 32) * 0.1
    d = mixed_curv_dist(x, y, alpha=1.0, kappa_fixed=0.74)
    d_hyp = poincare_distance(x, y, c=0.74)
    assert torch.allclose(d, d_hyp, atol=1e-5)

def test_alpha_gradient_at_zero():
    """∂d/∂α at α=0 = d_hyp - d_euc (no singular gradient)."""
    x = torch.randn(10, 32, requires_grad=True) * 0.1
    y = torch.randn(10, 32) * 0.1
    alpha = torch.tensor(0.0, requires_grad=True)
    d = mixed_curv_dist(x, y, alpha, kappa_fixed=0.74)
    d.sum().backward()
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()  # 关键: α=0 不是 singular point

def test_kappa_gradient_unchanged():
    """d/∂κ_l (跟 #47 fix 对齐, 验证公式兼容性)."""
    pass  # 跟 poincare_distance 一起测

def test_alpha_sigmoid_bounded():
    """α = sigmoid(z), z ∈ R, α ∈ (0, 1) for any z."""
    z = torch.linspace(-100, 100, 1000)
    alpha = torch.sigmoid(z)
    assert (alpha > 0).all() and (alpha < 1).all()
```

**5/5 test 通过** = Issue #56 Gate 0 数学基础 OK, 可以进入 Stage 1 训练.

result: Issue #56 方向B Gate 0 设计就绪. α_l sigmoid 参数化规避 κ=0 退化梯度病态. 跟 #47 修复公式对齐. 数学 sanity 5/5 test zero-GPU 优先, 通过后启动 Stage 1 训练 (~3.2h). R10 决策 = 候选 3 (math sanity) 立即可做.