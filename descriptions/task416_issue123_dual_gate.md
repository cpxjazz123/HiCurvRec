# Task #416 / Issue #123 [方向C Gate3] 双态 gate: zero-control 测试 + active-train 训练

**日期**: 2026-08-01
**Issue**: #123 [方向C Gate3] — 保留 zero-gate 等价但训练支路用 active gate (解决 #120 训练梯度被 -30 切断)
**任务**:
1. **双态 gate**:
   - **zero-control 单元测试**: gate=0 (保留, 验证 bit-equal control)
   - **active-train gate**: 训练期 active init, 训练后置零验证 control 等价
2. **per-layer bounded-negative κ/scale/gate** 进入 checkpoint
3. **adapter-only 30 epoch 受限短训**: 训练支路 active gate, 验证期 zero-gate
4. **训练后反事实**: zero-gate equivalence + adapter on/off + κ-shuffle + L0/L2 swap + alignment destroy + active gate 置换

**Gate 3 PASS 阈值**:
- zero-gate max logits diff ≤ 1e-5 (验证期)
- active-train 分支首步与末步三层梯度有限非零
- loss 有可测下降 (vs #120 卡 1.0)
- κ/scale/gate 均有非零更新
- κ-shuffle / SID swap / alignment destroy 皆产生可复现非零输出变化
- 无 NaN/Inf

**前置**:
- 复用 task84 ckpt + task396 SID
- L0 K64/L1 K128/L2 K256
- 三层独立 κ/scale/gate
- 禁止 global κ/fixed-only/pure Euclidean/随机 SID/独立 T5/提前 Stage4

**结果**: ⏸ 进行中 (R22 + R19 立即开工, GPU 2)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 复核 #100/#102 (per Issue spec §Gate1 "复核")
- task84 HG_Rec_best ckpt SHA256: `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e`
- L0 K64/L1 K128/L2 K256
- ✅ PASS per spec (task84 baseline 已复现)

### Gate 2 = Stage 2: Gate 1 PASS 后复核 (per Issue spec §Gate2 "复核")
- task396 SID SHA256: `9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8`
- shape `(9922, 4)`, unique `9922/9922 = 100%`
- ✅ PASS per spec

### Gate 3 = Stage 3: 本任务核心 (双态 gate)
- Zero-gate control 单元测试保留 (init=-30 → sigmoid≈1e-13)
- Active-train gate 训练 (init=0 active)
- 30 epoch adapter-only 训练
- 训练后 zero-gate 验证 control 等价

### Gate 4 = Stage 4: ⏸ STOP per spec
- 原因: 仅在 Gate 3 PASS 且有外部证据后, 按 task84 同数据/seed/beam 跑 control/adapter 统一测试
- R@10 > 0.1020 才 Target reached

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #120 (task413, closed) | Issue #123 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | 同一 gate 全程, init=-30 训练被切断 | **双态 gate: zero 测试 + active 训练** ✅ |
| **D2 实施核心** | gate 全程 -30 → 训练无效 | **训练 active gate, 验证 zero gate** ✅ |
| **D3 Gate 失败机制** | gate=-30 切断梯度 → loss 卡 1.0 | **双态让 trainability + equivalence 可兼得** ✅ |
| **D4 引用文献** | arXiv:2309.04082 | arXiv:2309.04082 ✅ |

**R18 v2 强制结论**: Issue #123 跟 #120 路径**有差异** (双态 vs 单一). 必须做新实验.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 双态 gate 实现 | zero_test_gate (init=-30) + active_train_gate (init=0) | Issue spec §Gate3 1 强制 |
| zero_test_gate 函数 | sigmoid per-position | 复用 task413 修复 |
| active_train_gate 函数 | sigmoid per-position, init=0 → 0.5 | 训练期 active |
| κ 参数化 | κ_l = -(κ_min + softplus(u_l)) | 复用 task413 |
| Adapter layers | encoder block 0 + 3 | 复用 task413 |
| Training epoch | 30 | Issue spec §Gate3 2 强制 |
| GPU 分配 | GPU 2 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |