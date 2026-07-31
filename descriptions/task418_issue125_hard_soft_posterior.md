# Task #418 / Issue #125 [方向B Gate1] hard SID + 软后向 mixed-distance posterior 恢复 gate gradient

**日期**: 2026-08-01
**Issue**: #125 [方向B Gate1] — hard argmin forward + per-codeword alpha 进入 soft posterior (softmax(-d_mix/τ)) backward 恢复 alpha gradient (解决 #122 grad=0)
**任务**:
1. **Hard SID forward + soft posterior backward**: 前向 hard argmin; 反向 `p_l(k|x)=softmax(-d_mix/τ)` 让 alpha 进入可微 reconstruction/commit loss
2. **保留 #122 的三分量定义** + 软后向 reconstruction/commit loss
3. **真实 batch 5-step audit**:
   - alpha 微扰改变 d_mix + posterior + soft quantized embedding + loss
   - 各层 gate gradient 有限非零
   - hard assignment 仍被记录
   - save/load round-trip 一致
4. **30 epoch 受限短训**: per-codeword main + uniform detach control, 固定 seed=42

**Gate 1 PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 每层 gate 对 loss 梯度有限非零且偏离初值
- 每层至少两分量在非平凡 codeword 子集权重 > 0.1
- hard SID / round-trip 通过
- 无 NaN/Inf

**前置**:
- 保留 L0 K64/L1 K128/L2 K256
- 三层 learnable-κ 主路 + fixed-hyperbolic/Euclidean components + 独立 alpha_l,k
- 禁止 global κ, fixed-only, pure Euclidean bypass, Gumbel/温度 sweep, 改 K, 软 SID 导出
- 区别 #122: 修复已实证的梯度断点 (soft posterior), 而非再改 gate 粒度

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 1)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 本任务核心
- Hard SID + soft posterior backward
- 30 epoch per-codeword main + uniform detach control

### Gate 2/3/4: ⏸ STOP per spec

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #122 (task415, closed) | Issue #125 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | per-codeword alpha + STE detach | **hard SID + soft posterior (softmax(-d_mix/τ)) backward** ✅ |
| **D2 实施核心** | STE detach 让 gate grad=0 | **soft posterior 让 alpha grad 通过 soft qz 流回** ✅ |
| **D3 Gate 失败机制** | argmin + STE 切断 alpha gradient | **soft posterior 让 alpha grad 通过 reconstruction 流回** ✅ |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 ✅ |

**R18 v2 强制结论**: Issue #125 跟 #122 路径**有差异** (soft posterior bridge vs STE detach). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Soft posterior 公式 | p_l(k\|x) = softmax(-d_mix/τ) | Issue spec §Gate1 1 强制 |
| τ | 1.0 (默认) | spec 没指定 |
| Hard forward | argmin(d_mix) | spec §Gate1 "导出仍使用 hard argmin" |
| Soft backward | p_l(k\|x) @ embedding | spec §Gate1 "soft-backward reconstruction/commit loss" |
| 训练 epoch | 30 main + 30 control | spec "相同有限预算" |
| GPU 分配 | GPU 1 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |