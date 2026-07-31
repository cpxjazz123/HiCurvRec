# Task #413 / Issue #120 [方向C Gate3] 有界非零曲率 κ_l=-(κ_min+softplus(u_l)) Stage 3 adapter 预检

**日期**: 2026-07-31
**Issue**: #120 [方向C Gate3] — 有界非零负曲率参数化 + zero-gate 精确 control (针对 #117 cf2/cf3 κ→0 退化)
**任务**:
1. **复用 #117 框架**: task410 commit 7b460d9 的 Position-Conditioned Adapter + zero-gate control
2. **参数化替换**: θ_m (init=0 → κ=0 卡 0 死区) → κ_l = -(κ_min + softplus(u_l)) (有界非零, 始终负, |κ_l| ≥ κ_min)
3. **Zero-gate control 保留**: init=-30 → softmax≈1e-13, 控制路径 bit-equal 数值等价
4. **5 反事实**: on/off, κ-shuffle, L0/L2 swap, item-SID 对齐破坏, κ sign/scale sanity
5. **30 epoch 受限短训** (per Issue spec §Gate3 "固定预算")

**Gate 3 PASS 阈值** (per Issue spec):
1. Zero-gate 对 control 最大 logits diff ≤ 1e-5
2. 三层 κ 均有限、负、|κ_l| ≥ κ_min
3. 三层 κ/scale/gate 梯度均有限非零
4. κ-shuffle 与 L0/L2 swap 均造成非零且可复现的输出变化
5. alignment-destroy 仍破坏输出

**前置** (per spec §Framework compliance precheck):
- Stage 3 adapter only (不启动 Stage 1/2 重训, 复用 task84 ckpt + task396 SID)
- 三层独立 κ_l/scale_l/gate_l 进入 checkpoint
- 禁止 global κ, fixed-only, pure Euclidean bypass, 替换 RQ-VAE
- zero-gate control 必须保留 #117 数值等价

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 = Stage 1: 复用 #102 产物 (per Issue spec §Gate1 "复核")
- task84 HG_Rec_best ckpt SHA256: `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e`
- L0 K64 / L1 K128 / L2 K256 复核
- ✅ PASS per Issue spec (task84 baseline 已复现)

### Gate 2 = Stage 2: 复用 #102 产物 (per Issue spec §Gate2 "复核")
- task396 SID SHA256: `9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8`
- shape `(9922, 4)`, unique `9922/9922 = 100%`
- ✅ PASS per Issue spec

### Gate 3 = Stage 3: 本任务核心 (adapter-only 30 epoch 预检)
- 有界非零 κ_l=-(κ_min+softplus(u_l)), κ_min=0.1
- Zero-gate 验证 (init=-30, softmax≈1e-13, control 数值等价)
- 5 反事实: on/off, κ-shuffle, L0/L2 swap, item-SID 对齐破坏, κ sign/scale sanity
- 30 epoch 受限短训

### Gate 4 = Stage 4: ⏸ STOP per spec
- 原因: 仅在 Gate 3 PASS 且有外部证据后, 按 task84 同数据/seed/beam 跑 control/adapter 统一测试
- R@10 > 0.1020 才 Target reached

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #117 (task410, closed) | Issue #120 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | R137 fix θ_m+tanh, init=0 卡 0 死区 | **κ_l=-(κ_min+softplus(u_l)), 有界非零** ✅ |
| **D2 实施核心** | κ_max·tanh(θ_m) init=0 → κ=0 | **κ_min + softplus 始终负, ≥ κ_min 远离 0** ✅ |
| **D3 Gate 失败机制** | κ→0 退化, cf2/cf3 SAME | **待实证: 有界非零让 κ-shuffle 有非零差异** ✅ |
| **D4 引用文献** | arXiv:2309.04082 | arXiv:2309.04082 ✅ |

**R18 v2 强制结论**: Issue #120 跟 #117 路径**有差异** (有界非零 vs tanh 卡 0). 必须做新实验.

**关键差异 vs #117**:
- #117 κ_m = κ_max · tanh(θ_m), init θ_m=0 → κ_m=0 死区, 训练后 κ 仍≈0
- #120 κ_l = -(κ_min + softplus(u_l)), init u_l=任意 → κ_l < 0 严格负, |κ_l| ≥ κ_min=0.1 远离 0
- 训练期间 κ_l 始终有限负非零, 反事实 κ-shuffle 必有差异

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| κ 参数化 | κ_l = -(κ_min + softplus(u_l)) | Issue spec §Gate3 1 强制 |
| κ_min | 0.1 | 合理下界 (不让 κ→0, 也避免 κ→∞) |
| κ_max | 2.0 (per task410) | 跟 #117 一致, 避免重新调参 |
| u_l init | 0.0 → κ_l = -(0.1 + ln(2)) ≈ -0.793 | 软初始化, softplus(0)=ln(2) |
| Zero-gate | init=-30, softmax≈1e-13 | 跟 task410 一致, 数值等价 |
| Adapter layers | encoder block 0 + 3 | 跟 task410 一致 |
| Training epoch | 30 (受限短训) | Issue spec §Gate3 2 强制 |
| GPU 分配 | GPU 0 (per R7 全部空闲, Issue #119 占 GPU 2) | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |
| Stage 1/2/4 | ⏸ STOP per spec | Issue spec §Gate3-4 强制 |

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 description (本文件)
2. ⏳ 写 task413_issue120_bounded_kappa_adapter.py (复用 task410 adapter 框架 + 替换 κ 参数化)
3. ⏳ Gate 3 预检 + 30 epoch 训练
4. ⏳ 写 Gate 3 verdict
5. ⏳ commit + push (R15 + R21 v2)
6. ⏳ close Issue #120 (R16 + R20 + R21)