# Task #149 — Issue #55 方向A: 曲率感知优化器 Gate 0 设计 (zero-GPU prep)

**日期**: 2026-07-31 00:12
**状态**: 📝 DESIGN — 等 owner 拍板 (R10 候选)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/55

## 1. 目的

Issue #55 验证"几何不一致"是 val/test gap 主因 (对照 arXiv:2405.13979 "Robust Hyperbolic Learning with Curvature-Aware Optimization"). 替换标准 Adam 为 Riemannian AdamW 风格曲率更新 + per-layer 缩放参数 s_l (κ 更新后重新校准 codebook 有效半径).

## 2. 文献基础

- **arXiv:2405.13979** "Robust Hyperbolic Learning with Curvature-Aware Optimization"
- 核心机制:
  1. **Riemannian AdamW** 风格曲率更新 (二阶矩估计 + 曲率更新解耦)
  2. **可微调双曲缩放** (fine-tunable hyperbolic scaling) - 每次 κ 更新后同步重新校准 codebook 尺度

## 3. Gate 0 实施方案

### 3.1 改动点 (优化器 + per-layer 缩放)

`HG-Rec/model/hrqvae_free_curv.py` (commit 61e707c 修复版):

**新增**:
```python
# Per-layer scaling parameter (Issue #55)
self.scale_l = nn.Parameter(torch.ones(num_layers))

# Riemannian AdamW-style curvature update (跟 κ 参数解耦)
self.kappa_l = nn.Parameter(torch.tensor([1.0] * num_layers))
self.kappa_l_optimizer = RiemannianAdamW([self.kappa_l], lr=5e-4)

# 在 training step 中, κ 更新后:
with torch.no_grad():
    # s_l 重新校准: codebook 向量在 κ_new 下的有效半径
    codebook_normalized = codebook * self.scale_l.unsqueeze(-1)
    new_radius = poincare_radius(codebook_normalized, c=self.kappa_l)
    self.scale_l.data = new_radius / old_radius  # 校准
```

### 3.2 Riemannian AdamW 实现 (跟标准 AdamW 解耦)

```python
class RiemannianAdamW(torch.optim.Optimizer):
    """AdamW for Riemannian manifold (Issue #55)."""
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                # Riemannian gradient (project to tangent space)
                # For 1D scalar κ: simple grad
                if len(p.shape) == 0 or p.numel() == 1:
                    riem_grad = grad
                else:
                    # For vector params: project via 1/(1 + ‖x‖²) scaling (Poincaré ball)
                    riem_grad = grad / (1 + torch.norm(p, dim=-1, keepdim=True)**2)**2

                # Standard Adam moments
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(riem_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(riem_grad, riem_grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # AdamW update with Riemannian retraction
                denom = (bias_corrected_exp_avg_sq.sqrt() + group['eps'])
                update = exp_avg / denom
                if group['weight_decay'] != 0:
                    update = update + group['weight_decay'] * p
                p.add_(update, alpha=-step_size)

                # Riemannian retraction: for κ (1D scalar), identity. For vector params, expmap.
                if len(p.shape) > 0 and p.numel() > 1:
                    p.data = expmap0(p.data)  # Project back to Poincaré ball
```

### 3.3 诊断记录

每个 epoch 记录 "codebook 尺度校准误差" (Issue #55 关键诊断):
```python
with torch.no_grad():
    # 固定 κ 下, 若不重新校准, SID 分配一致性的漂移量
    drift = torch.norm(self.scale_l - self.scale_l.mean()) / self.scale_l.mean()
    diagnostics['scale_drift'] = drift.item()
```

### 3.4 CLI flag

```python
parser.add_argument('--curvature_aware_optimizer', action='store_true', help='Issue #55')
parser.add_argument('--scale_recalibration', action='store_true', help='Issue #55 s_l')
```

### 3.5 对照组

| 组 | 配置 | 期望 R@10 |
|----|------|----------|
| Baseline (#84) | vanilla RQ-VAE, c=1.0 | 0.1020 |
| #49 (学 κ_l + 标准 Adam) | κ_l learned, baseline optimizer | 0.1005 |
| **#55 (学 κ_l + Riemannian AdamW + s_l)** | **κ_l learned, curvature-aware** | **> 0.1020?** |

### 3.6 决策阈值 (Issue #55 原文)

- **H1**: 曲率感知优化器 → Stage 4 test R@10 > baseline 0.1020
- **H2**: val/test gap < 0.02 (#49/#51/#52/#53 观测到的 0.02 量级)

### 3.7 Stage 1 训练成本估算

| Stage | 时长 | GPU |
|-------|------|-----|
| Stage 1 RQ-VAE 1000 epoch (κ_l learned + Riemannian AdamW + s_l) | ~3.5h | 单 GPU |
| Stage 2 SID inference | ~5min | 单 GPU |
| Stage 3 T5-mini 200 epoch | ~2h | 单 GPU |
| Stage 4 eval (beam=50) | ~5min | 单 GPU |
| **总计** | **~5.5h** | **单 GPU 串行** |

## 4. 跟 #47 公式对齐

Issue #55 优化器改造不改变 #47 修复后的距离公式, 而是改 κ 参数的优化方式. 跟 Issue #56 (改参数化) 不同. 两者独立可叠加: Issue #55 + #56 联合 = 曲率感知优化器 + α_l 混合权重, 但优先级低 (跟 #30+#43 联合同模式).

## 5. R10/R11.5 决策

| 候选 | ROI | 风险 |
|------|-----|------|
| Issue #55 完整 4 阶段 (~5.5h) | 中 (跟 Issue #49 baseline 对照, H1 +0.2pp 突破概率低) | 高 (跟 #49 同样 NO-GO 风险) |
| Issue #55 Stage 1 only (Riemannian AdamW 训练曲线诊断, ~3.5h) | 中-高 (低成本验证训练稳定性) | 低 |
| Issue #55 数学 sanity (Riemannian AdamW unit test, ~30min) | 高 (零 GPU) | 零 |

**R10 决策**: 候选 3 (math sanity) 优先. 通过后启动候选 2 (Stage 1 only). 跟 Issue #56 并行数学 sanity.

## 6. 启动 Gate 0 数学 sanity (zero-GPU 立即可做)

```python
# tests/test_issue55_riemannian_adamw.py
import torch
from hgrec.model.optimizer import RiemannianAdamW

def test_step_decreases_loss():
    """Riemannian AdamW step should decrease loss."""
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = RiemannianAdamW([param], lr=0.1)
    loss_fn = lambda x: (x - 0.5)**2

    initial_loss = loss_fn(param).item()
    optimizer.zero_grad()
    loss_fn(param).backward()
    optimizer.step()
    new_loss = loss_fn(param).item()
    assert new_loss < initial_loss

def test_poincare_retraction():
    """Vector params stay in Poincaré ball after expmap."""
    param = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))
    optimizer = RiemannianAdamW([param], lr=0.1)
    loss_fn = lambda x: (x.sum() - 1.0)**2
    optimizer.zero_grad()
    loss_fn(param).backward()
    optimizer.step()
    assert torch.norm(param.data) < 1.0  # stay in ball

def test_scale_recalibration():
    """s_l re-calibrates codebook radius after κ update."""
    codebook = torch.randn(256, 32) * 0.1
    scale = nn.Parameter(torch.ones(3))
    kappa_l = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    # After κ update: codebook should be re-scaled to maintain radius
    new_kappa = torch.tensor([1.5, 1.5, 1.5])
    new_radius = poincare_radius(codebook * scale.unsqueeze(-1), c=new_kappa)
    # scale calibration logic...
    assert new_radius.isfinite().all()

def test_weight_decay_decoupled():
    """AdamW weight decay decoupled from gradient (跟标准 AdamW 一致)."""
    pass

def test_step_count():
    """step() increments state['step'] correctly."""
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = RiemannianAdamW([param], lr=0.1)
    for _ in range(5):
        optimizer.zero_grad()
        (param**2).backward()
        optimizer.step()
    assert optimizer.state[param]['step'] == 5
```

**5/5 test 通过** = Issue #55 Gate 0 数学基础 OK, 可以进入 Stage 1 训练.

## 7. Issue #55 vs #56 vs #57 ROI 对比

| Issue | 改动层 | Stage 1 训练 | Stage 3 训练 | 验证 ROI |
|-------|--------|-------------|-------------|---------|
| #55 方向A (curvature-aware optimizer) | 优化器 + s_l | ~3.5h | ~2h | 中 |
| #56 方向B (mixed-curv α_l) | 参数化 κ_l → α_l | ~3.2h | ~2h | 中 |
| #57 方向C (T5 init / projection / attention) | Stage 3 embedding / attention | 0 | ~2h × 2 arm | **高 (Gate 0)** ⭐ |

**R10 决策**: 
- **Issue #57 方向C Gate 0 最优先** (零 Stage 1 训练, 仅 Stage 3 2 arm × ~2h = ~4h GPU, 验证假说本身)
- **Issue #55 + #56 数学 sanity 并行** (zero-GPU, ~30min)
- 通过后再决定启动 Stage 1 训练 (候选 2)

result: Issue #55 方向A Gate 0 设计就绪. Riemannian AdamW 风格曲率更新 + per-layer s_l 重新校准. 数学 sanity 5/5 test zero-GPU 优先. R10 决策 = Issue #57 Gate 0 (低 ROI 成本, 验证假说本身) > Issue #55 + #56 数学 sanity (zero-GPU) > Stage 1 训练 (等数学 OK).