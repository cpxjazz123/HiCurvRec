# Task #409 / Issue #116 [方向B Gate1] 连续失真目标学 gate + 冻结 hard assignment

**日期**: 2026-07-31
**Issue**: #116 [方向B Gate1] — 解决 #113 gate 零梯度根因, 换连续 distortion 目标
**任务**: 
1. **Gate phase**: 冻结 codebook/assignment, 最小化连续加权 component distortion + gate entropy floor (用连续目标更新 gate)
2. **Codebook phase**: 冻结已学 gate, 恢复 hard assignment + 原 RQ-VAE loss (用 hard assignment 训练 codebook)
3. **验证**: alternating continuous-gate optimization 能否通过 Gate 1 (三层 util≥90% + max_load<5% + L1/L2 norm≥L0×0.2 + 每层≥2 component weight>0.1 + gate gradient 有限非零 + 无 NaN/Inf + round-trip 一致)

**前置**:
- #113 (task407) 已通过 schema (commit 5aed3c5, verdicts/task407_issue113_three_component_gate1_v2.md):
  - 18 keys schema PASS
  - 三 component forward PASS
  - Round-trip PASS
- #113 Gate 1 4 轮修复 (v1-v4) 全部 USAGE-KILL @ ep 5, 根因: hard argmin 让 gate 零梯度

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 alternating training): ⏸ 进行中

**两阶段交替训练** (per Issue spec §Gate1):

**Phase 1 (Gate phase)**:
- 冻结: codebook (三 component), assignment
- 优化: 每层 gate_logits (minimize continuous weighted component distortion)
- 添加: gate entropy floor (阻止 gate 退化到单 component)
- 记录: gate gradient, component distortion, gate weights 变化

**Phase 2 (Codebook phase)**:
- 冻结: 已学 gate
- 优化: codebook (三 component) + 原 RQ-VAE loss
- 恢复: hard assignment (per #113 schema)
- 记录: utilization, max_load, entropy, norm, loss

**强制不允许** (per Issue spec):
- 无界 sweep
- 超过 1 个主配置 + 1 个 control

**PASS 阈值**:
- 三层 utilization ≥ 90%
- max_load < 5%
- L1/L2 norm ≥ L0 × 0.2
- 每层 ≥ 2 component weight > 0.1
- gate gradient 有限非零 (确认 gate 离开 init)
- 无 NaN/Inf
- round-trip 一致

**实施**:
- `scripts/task409_issue116_continuous_gate_train.py` (待写, 复用 #113 schema + 两阶段交替)
- 复用 #113 `ThreeComponentHRQVAE` 类 (commits 5aed3c5)
- 加 `gate_logits` optimizer (Phase 1) + `codebook_xxx` optimizer (Phase 2)
- Phase 切换: per-epoch alternating (or per-N-step)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 必须先 PASS

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #113 (task407) | Issue #116 (本 task) |
|------|----------------------|----------------------|
| **D1 spec 摘录** | 三分量 product schema (R137 fix) | **连续 distortion 目标 + 冻结 (解 gate 零梯度)** |
| **D2 实施核心** | 单阶段 hard assignment 训练 | **两阶段交替 (gate phase + codebook phase)** |
| **D3 Gate 1 失败机制** | 4 轮 v1-v4 修复, gate stuck 在 init | **换连续目标, gate 通过 distortion 反传** |
| **D4 引用文献** | arXiv:2307.04514 mixed-curvature + ACE-HGNN | **arXiv:2307.04514 weighted-PM + ACE-HGNN** |

**R18 v2 强制结论**: Issue #116 跟 #113 路径**有差异** (两阶段交替 vs 单阶段). 必须做新实验, 不允许沿用判决.

**跟 #28 / #33 区分**: 
- #28 = per-layer Gumbel-Softmax (替换最终 hard SID assignment + 温度 sweep)
- #33 = per-item soft (STE + per-item weighted sum)
- #116 = 不替换最终 hard SID assignment (只在训练时加连续 distortion 辅助目标), 不做温度 sweep

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 复用 #113 schema | 完全复用 `ThreeComponentHRQVAE` 类 (commit 5aed3c5) | Issue spec §Framework compliance precheck: "复用 #113 的 L0 K64/L1 K128/L2 K256 三分量 schema" |
| Phase 切换频率 | per-epoch (50 epoch Gate phase + 50 epoch Codebook phase = 100 epoch total) | Issue spec §Gate1 "两阶段受限验证" |
| 连续 distortion 公式 | `loss_gate = Σ_l Σ_b w_l · d_total_l(x_b, codebook) + β·H(gate_l)` (β=0.01 entropy floor) | Issue spec §Gate1: "最小化连续加权 component distortion + gate entropy floor" |
| GPU 分配 | GPU 1 (per R7 全部空闲时 GPU 0/1/2 各跑一个 Issue) | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 关键产物 (本 task 进行中)

- **Phase 1+2 训练脚本**: `scripts/task409_issue116_continuous_gate_train.py` (待写)
- **Gate 1 train log**: `products/task409_issue116_continuous_gate/train_log.json` (待写)
- **Gate 1 verdict**: `verdicts/task409_issue116_continuous_gate_gate1_v2.md` (待写)
- **整体决策**: ⏸ 进行中

---

## 后续 (per R22 + R19 + R16)

1. **写两阶段训练脚本** (立即): 复用 #113 schema + 冻结 codebook phase + 冻结 gate phase
2. **Gate 1 训练 100 epoch** (Phase 1: 50 epoch + Phase 2: 50 epoch)
3. **Gate 1 PASS/FAIL 判定**: 7 项阈值 (util + max_load + norm + component weight + gradient + NaN/Inf + round-trip)
4. **Gate 2/3/4 立即推进** (若 Gate 1 PASS): Sinkhorn SID + T5 adapter + Stage 4 R@K
5. **commit + push** (R15 + R21 v2)
6. **close Issue #116** (R16 + R20 + R21)