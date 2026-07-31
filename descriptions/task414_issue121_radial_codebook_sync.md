# Task #414 / Issue #121 [方向A Gate1] κ 更新后 assignment-preserving 径向 codebook 同步传输

**日期**: 2026-08-01
**Issue**: #121 [方向A Gate1] — κ 更新后保持 assignment 的径向 codebook 同步传输 (针对 #118 早坍缩)
**任务**:
1. **per-layer optimizer-step hook**: 记录旧 κ/scale/codebook → 按 #47 公式更新 κ → 径向 codebook 传输 → 重算距离
2. **径向传输**: codebook 乘以 r = sqrt(κ_old/κ_new) (沿径向均匀缩放, Voronoi assignment-preserving)
3. **5-step functional audit**: κ update 改变 scale/距离; 传输前后 assignment + ranking 一致; grad 有限非零; 否则 FAIL 不训练
4. **30 epoch 受限短训**: main (有径向传输) + control (无径向传输但其他相同), 固定 seed=42/相同预算
5. **逐 epoch 输出**: κ/scale/codebook norm/util/max_load/entropy/loss/sync residual/ranking mismatch/NaN/Inf

**Gate 1 PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- 无 ranking/assignment mismatch
- 同步残差在预声明容差内
- κ 梯度有限非零且离开初值
- 无 NaN/Inf

**前置**:
- 保留 L0 K64/L1 K128/L2 K256
- 三层独立 learnable κ (per-component)
- 禁止 global κ/fixed-only/pure Euclidean/改 K/独立模型

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 0)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 本任务核心 (per-layer optimizer-step hook + 径向 codebook 传输)
- 5-step functional audit
- 30 epoch main + control 实证

### Gate 2/3/4: ⏸ STOP per spec
- 原因: 仅在 Gate 1 PASS 后启动 Sinkhorn / T5 / R@K eval

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #118 (task411, closed) | Issue #121 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | softplus κ + sync rescale | **径向重标定 + per-layer optimizer step hook** ✅ |
| **D2 实施核心** | scale 跟 κ 同步 | **codebook 径向重标定 + per-layer step hook** ✅ |
| **D3 Gate 失败机制** | scale 不跟 κ → distance scale 不一致 | **assignment 改变 + distance ranking 不一致** ✅ |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 ✅ |

**R18 v2 强制结论**: Issue #121 跟 #118 路径**有差异** (径向 codebook 传输 vs scale rescale). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 径向重标定公式 | r = sqrt(κ_old/κ_new) | 数学保证 Voronoi-preserving (均匀缩放) |
| optimizer step hook | per-layer per-step | Issue spec §Gate1 1 强制 |
| Control | 无径向传输但相同其他 | Issue spec §Gate1 3 强制 |
| 训练 epoch | 30 main + 30 control | Issue spec "相同有限预算" |
| GPU 分配 | GPU 0 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |