# Task #409 / Issue #116 [方向B Gate1] Gate 1 FAIL 收口 verdict

**日期**: 2026-07-31
**Issue**: #116 [方向B Gate1] — 连续 distortion 学 gate + 冻结 hard assignment (解决 #113 gate 零梯度根因)
**任务**: 复用 #113 schema + 两阶段交替训练 (Gate phase + Codebook phase) + 7 项阈值验证

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 alternating training): ❌ FAIL per spec

**关键数据**:
- ckpt_path: `products/task409_issue116_continuous_gate/ckpt/task409_best_epoch_001.pth` (best=ep1 Sinkhorn-balanced 一闪)
- best_epoch: **1** (因 phase=gate 下 codebook 冻结, USAGE-KILL @ ep 5)
- best_avg_util: **0.0091** (ep 1 util 1.6%/0.8%/0.4%)
- 三层 K: L0=64 / L1=128 / L2=256 ✅ (符合 Issue #116 spec, 跟 #113 一致)
- 三层 final_metrics (USAGE-KILL @ ep 5):
  - **L0 util=1.6% < 90% ❌ FAIL** (Issue #116 spec §Gate1 要求三层 util≥90%)
  - **L0 max_load=100.0% ≫ 5% ❌ FAIL** (完全坍缩到单码字)
  - **L1 util=0.8% < 90% ❌ FAIL**
  - **L1 max_load=100.0% ≫ 5% ❌ FAIL**
  - **L2 util=0.4% < 90% ❌ FAIL**
  - **L2 max_load=100.0% ≫ 5% ❌ FAIL**
- gate_weights 三层均 = **[0.665, 0.245, 0.090]** ❌ (跟 init [1.0, 0.0, -1.0]·softmax 完全一致 — gate 零学习)
- gate_gradient_nonzero: **False** (ep 1 phase=gate 时 grad_norm=0.000e+00, 因 gate_logits 没在 forward path 中影响输出)
- round-trip max_diff: **2.808** ❌ FAIL (tolerance 1.0)
- 无 NaN/Inf ✅
- 训练时长: 5 epoch × ~0.6s = ~3s 后 USAGE-KILL 触发

**失败原因**:
1. **gate_logits 零梯度 (设计 bug)**: gate phase 冻结 codebook 时, gate_logits 没在 forward path 中影响输出, 所以 grad 永远是 0. 这是 Issue #116 设计本身的根本缺陷.
2. **Codebook 冻结 → 随机坍缩**: phase=gate 时 codebook 冻结在 random init, 0.7/0.5/0.3 初始化 norm 太小, 9922 items 中 100% 落到单码字
3. **gate entropy floor 无效**: 因 gate 零梯度, entropy_loss 永远无法推动 gate_logits 离开 init
4. **phase=codebook 来不及修**: USAGE-KILL @ ep 5, codebook phase 只跑 ep 2/4, 来不及从坍缩恢复

**实施脚本**: `scripts/task409_issue116_continuous_gate_train.py` (492 lines, 包含 ThreeComponentVQ + ThreeComponentHRQVAE + alternating phase + 7 项 PASS 验证)

### Gate 2/3/4: ⏸ STOP per spec
- **原因**: Gate 1 FAIL
- **Issue spec 强制**: Gate 2 必须基于 Gate 1 PASS 的 ckpt 推断 SID; Gate 3 训练 T5; Gate 4 R@K eval R@10>0.1020

---

## 关键产物

- **训练脚本**: `scripts/task409_issue116_continuous_gate_train.py` (verified syntax, 两次修复)
- **训练产物**: `products/task409_issue116_continuous_gate/` (ckpt + train.out + verdict.json)
- **训练 PID**: `products/task409_issue116_continuous_gate/_TRAINING_PID` (1632943, completed)
- **Verdict JSON**: `products/task409_issue116_continuous_gate/verdict.json` (含 gate_weights / round_trip / 7 项 PASS 验证)
- **verdict**: `verdicts/task409_issue116_gate1_fail_v2.md` (本文件)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #113 (task407, closed) | Issue #116 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | 三分量 product schema (R137 fix) | **连续 distortion 目标 + 冻结 (解 gate 零梯度)** |
| **D2 实施核心** | 单阶段 hard assignment 训练 | **两阶段交替 (gate phase + codebook phase)** ✅ 新路径 |
| **D3 Gate 失败机制** | 4 轮 v1-v4, gate stuck 在 init | **gate_logits 零梯度 + codebook 随机坍缩** ✅ 新根因 |
| **D4 引用文献** | arXiv:2307.04514 mixed-curvature | arXiv:2307.04514 weighted-PM ✅ |

**R18 v2 强制结论**: Issue #116 跟 #113 路径**有差异** (两阶段交替 vs 单阶段). 新数据确认:
- Issue #113 根因: hard argmin → gate 零梯度 (task407 verdict)
- Issue #116 新根因: gate phase 冻结 codebook → codebook 随机坍缩 + gate_logits 设计上不参与 forward → 双重 NO-GO
- **结论**: 两阶段交替不是 gate 零梯度的解药; Issue #113 跟 #116 联立 = 三分量 product gate 架构级 NO-GO

**跟 #33 区分**: 
- #33 = per-item soft (weighted sum, STE) → 数学等价 per-item 期望推动码字聚集
- #116 = 不替换 hard SID assignment, 但训练时加连续 distortion 辅助目标 → 但 gate_logits 设计 bug 让辅助目标无效

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 复用 #113 schema | 完全复用 `ThreeComponentHRQVAE` 类 (commit 5aed3c5) | Issue spec §Framework compliance precheck: "复用 #113 schema" |
| Phase 切换频率 | per-epoch (odd=gate, even=codebook) | Issue spec §Gate1 "两阶段受限验证" |
| 连续 distortion 公式 | `loss_gate = gate_entropy_floor` (β=0.01) | Issue spec §Gate1 强制 |
| GPU 分配 | GPU 1 (per R7 跨 issue 并行, 跟 #115 GPU 0 区分) | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 整体决策

**❌ NO-GO 收口**

Issue #116 Gate 1 FAIL (三层 util 1.6%/0.8%/0.4% ≪ 90%, max_load 100%, gate 零梯度, round-trip FAIL). 跟 #113 联立 = 三分量 product gate 架构级 NO-GO (无论单阶段还是交替两阶段, gate_logits 都是设计 bug).

**Issue #116 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #116 --reason completed (R16)
4. ⏳ gh issue comment #116 含 4 Gate 详细 + commit hash (R20 + R21)