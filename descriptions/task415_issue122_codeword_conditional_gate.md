# Task #415 / Issue #122 [方向B Gate1] codeword-conditional mixed-distance gate 防 L1/L2 坍缩

**日期**: 2026-08-01
**Issue**: #122 [方向B Gate1] — per-codeword alpha_l,k 替代全局 alpha, 避免单一全局权重放大 L1/L2 坍缩
**任务**:
1. **per-codeword gate**: alpha_l,k = softmax(g_l,k), K 个 codeword 各自独立的 alpha
2. **d_mix 重组**: d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k)
3. **5-step functional audit**: 微扰任一活跃 k 的 gate → d_mix/loss/assignment 改变; permute alpha 改变 score; grad 有限非零
4. **component-usage 熵审计**: 每层每 codeword alpha 熵, 不允许直接伪造 SID 均衡
5. **30 epoch 受限短训**: main (per-codeword gate) + control (shared gate), 固定 seed=42/相同预算
6. **最终 SID**: hard argmin (不允许伪造均衡)

**Gate 1 PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 每层至少两分量在非平凡 codeword 子集上权重 > 0.1
- alpha 对 d_mix/loss 梯度有限非零, 训练后变化
- round-trip 通过
- 无 NaN/Inf

**前置**:
- 保留 L0 K64/L1 K128/L2 K256
- 三层独立 learnable κ 主分量 + 固定双曲 + 固定欧氏
- 所有 alpha_l,k 都直接进入 d_mix
- 禁止 global κ/fixed-only/pure Euclidean/改 K/独立模型/温度扫描

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 1)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 本任务核心 (per-codeword gate + 5-step audit + 30 epoch)
- Per-codeword alpha_l,k (K 个 codeword 各自独立)
- Component-usage 熵审计
- 30 epoch main + control

### Gate 2/3/4: ⏸ STOP per spec
- 原因: 仅在 Gate 1 PASS 后启动 Sinkhorn / T5 / R@K eval

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #119 (task412, closed) | Issue #122 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | d_mix = Σ_j α_lj d_lj (全局) | **d_mix = Σ_j α_l,k,j d_l,j (per-codeword)** ✅ |
| **D2 实施核心** | alpha 是 layer 标量 (per-layer) | **alpha 是 per-layer per-codeword (K 个)** ✅ |
| **D3 Gate 失败机制** | 全局权重让 L1/L2 集中坍缩 | **per-codeword 让不同 codeword 学不同几何配比** ✅ |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 ✅ |

**R18 v2 强制结论**: Issue #122 跟 #119 路径**有差异** (per-codeword vs 全局). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| alpha_l,k 参数化 | alpha_l,k = softmax(g_l,k), g_l,k learnable | Issue spec §Gate1 1 强制 |
| alpha 维度 | (L, K, 3) — 每层每 codeword 3 个 component 权重 | Issue spec §Gate1 1 强制 |
| d_mix 公式 | d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k) | Issue spec §Gate1 1 强制 |
| Component-usage 熵 | 仅审计, 不约束 loss | Issue spec §Gate1 2 强制 |
| 训练 epoch | 30 main + 30 control | Issue spec "相同预算" |
| GPU 分配 | GPU 1 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |