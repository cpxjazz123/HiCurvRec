# Task #417 / Issue #124 [方向A Gate1] hard 前向 + 软后向 κ gradient bridge

**日期**: 2026-08-01
**Issue**: #124 [方向A Gate1] — hard argmin forward + soft posterior (softmax(-d_hyp/τ)) backward 恢复 κ gradient (解决 #121 grad=0)
**任务**:
1. **Hard-forward / soft-backward bridge**: 前向仍输出 hard `argmin` codeword; 反向用 `p_l(k|x)=softmax(-d_hyp(κ_l,scale_l)/τ)` 的 soft quantized embedding 传递梯度
2. **κ 更新后按 #47 公式同步重标定 codebook/距离** (per spec)
3. **真实 batch 5-step audit**:
   - 硬 assignment 与 baseline 一致
   - κ 微扰改变 posterior + soft quantized embedding + loss
   - `dL/dκ_l` 三层有限非零
   - κ 后同步残差有限
4. **30 epoch 受限短训**: main (soft-backward) + control (DETACH backward, 跟 #121 一样), 固定 seed=42

**Gate 1 PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 三层 κ 梯度有限非零且离开初值
- hard SID 路径仍存在
- 同步残差在声明容差内
- 无 NaN/Inf

**前置**:
- 保留 L0 K64/L1 K128/L2 K256
- 三层独立 learnable κ
- 禁止 Gumbel 抽样, K/温度 sweep, 软 SID 导出
- 复用 #121 径向重标定公式 + κ 参数化

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 0)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 本任务核心
- Hard-forward / soft-backward bridge
- 30 epoch main + control

### Gate 2/3/4: ⏸ STOP per spec

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #121 (task414, closed) | Issue #124 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | 径向重标定 + STE detach | **hard 前向 + 软后向 bridge 恢复 κ grad** ✅ |
| **D2 实施核心** | STE detach 让 κ grad=0 | **soft posterior = softmax(-d_hyp/τ) backward** ✅ |
| **D3 Gate 失败机制** | argmin + STE 切断 κ gradient | **soft posterior 让 κ grad 通过 soft qz 流回** ✅ |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #124 跟 #121 路径**有差异** (soft posterior bridge vs STE detach). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Soft posterior 公式 | p_l(k\|x) = softmax(-d_hyp(κ_l,scale_l)/τ) | Issue spec §Gate1 1 强制 |
| τ | 1.0 (默认) | spec 没指定, 用 unit temperature |
| Hard forward | argmin(d_hyp) | spec §Gate1 "前向仍输出 hard argmin codeword" |
| Soft backward | p_l(k\|x) @ embedding (weighted sum) | spec §Gate1 "soft quantized embedding" |
| 径向重标定 | r = sqrt(κ_old/κ_new) | 复用 #121 |
| 训练 epoch | 30 main + 30 control | spec "同有限预算" |
| GPU 分配 | GPU 0 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |