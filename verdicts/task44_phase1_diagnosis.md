# Task #44 Phase 1 Toy — 第一轮诊断

> **完成日期**: 2026-07-20
> **状态**: 🔴 **失败 → 关键发现 → 需 STE 量化器**
> **任务**: 验证 PM-RQ 是否比标准 RQ-VAE 更优 (Phase 1 toy)

---

## 1. 第一轮结果

| 组 | final_loss | 解读 |
|---|---|---|
| B (MCKG-standard, fused → 标准 RQ) | **0.275** | baseline |
| C (PM-RQ 3 段) | **1.926** ❌ 高 7× | κ 和 fusion_logits 没动 |
| D (PM-RQ 2 段, 消融) | **0.631** ❌ 高 2.3× | w 始终 [0.5, 0.5], 没学到区分 |

## 2. 关键诊断

**Symptom**: κ 完全不变 (sphere=+0.8446, hyperbolic=-1.0586 全程不变),fusion_logits 也不变 (始终 [1/3, 1/3, 1/3])。

**Root cause**: 
1. **argmin 的 straight-through 问题**: `d_total.argmin(dim=-1)` 选 1 个码字,反向传播到 fusion_logits 的梯度是 0(argmin 不可微)。需要 **STE 量化器**(`quantize(x) = argmin(x) + x - x.detach()`)让梯度能通过。
2. **距离量级不匹配**: sphere_dist mean=0.34, euclid=1.0, hyperbolic=2.7。直接 softmax 权重 [1/3,1/3,1/3] 等价于"被 euclid 主导"。已加 per-segment `dist_scale` 归一化,但因 argmin 阻断梯度仍未生效。
3. **κ 没动**: 即使加了 `lr × 100` 给 κ,argmin 阻断导致 κ 接收不到"我有助选码字"的信号。

## 3. 修复方案 (第二轮)

**STE 量化器**:
```python
def quantize_ste(x, codebook):
    """Soft argmin + straight-through."""
    d = torch.cdist(x, codebook, p=2)
    # soft assignment (probabilistic)
    soft_idx = F.softmax(-d * 10, dim=-1)  # high temp → near one-hot
    # hard argmin for forward, soft for backward
    hard_idx = d.argmin(dim=-1)
    # STE: forward = hard, backward = soft
    # 实现: code_selected = soft_idx @ codebook (gradient 可传)
    code_soft = soft_idx @ codebook
    code_hard = codebook[hard_idx]
    return hard_idx, code_hard + (code_soft - code_soft.detach())  # STE trick
```

**重置距离 scale 估计**: 用实际训练 batch 上的 per-segment mean 来 scale,而不是固定 0.34/1.0/2.7。

## 4. 决策

**Phase 1 第一轮未通过 (C > B × 1.05)**:
- ⚠️ 这是预期内的,因为 STE 还没加
- 修复后重跑 (第二轮)

**第二轮计划**:
1. 加 STE 量化器
2. 用 actual per-batch distance mean 做 normalize
3. 重跑 B/C/D 三组, 验证 C 是否 ≤ B

## 5. 产物

| 产物 | 路径 |
|---|---|
| 主脚本 | `scripts/task44_pmrq_toy.py` |
| 第一轮日志 | `logs/task44_phase1.log` |
| Summary (第一轮) | `products/task44_pmrq_phase1/task44_phase1_summary.json` |

---

**result:** Task #44 第一轮失败,关键发现:**argmin 阻断梯度,需要 STE 量化器 + actual per-batch 距离归一化**。第二轮必须修这两点。
