# Task #410 / Issue #117 [方向C Gate4恢复] SID 层位条件化 product-stereographic attention

**日期**: 2026-07-31
**Issue**: #117 [方向C Gate4恢复] — 撤销 #114 错误依赖, 直接在 #102 SID token position 加 adapter
**任务**: 
1. **Gate 1**: 复用 #100/#102 PASS, 核对 checkpoint + K64/K128/K256 + SID 来源
2. **Gate 2**: 复用 #102 SID, 核对 shape/checksum/item alignment/collision 0%
3. **Gate 3**: 在有限 encoder self-attention 层加入 position-conditioned product-stereographic residual (三层独立 κ/scale/gate, zero gate 等价 #111 control) + 5 组反事实
4. **Gate 4**: 完整 test evaluation (control vs adapter), R@10>0.1020 才 Target reached

**前置**:
- #102 SID (9922×4, unique 9922/9922, 三层 util 100%)
- #111 Gate 3 PASS (commit 47672ac, 5 组反事实 L1/L2 + gradient)
- 不依赖 #113 (owner 撤销 #114 错误依赖, 独立 C 自身上游)

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 复用 #100/#102 PASS): ⏸ 进行中

**强制核对** (per Issue spec §Gate1):
- #100 Stage 1 ckpt (task84 HG-Rec baseline 200 epoch best_collision_model.pth)
- 三层 K: L0 K64 / L1 K128 / L2 K256
- SID 来源: #102 (task396 SID, `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy`)

**实施**:
- 核对 SHA256: task84 Stage 1 ckpt + task396 SID

### Gate 2 (= Stage 2 SID 复用 #102): ⏸ STOP per spec
- **原因**: Gate 1 必须先核对 PASS
- **强制核对**: shape (9922, 4), checksum, item alignment, collision 0%

### Gate 3 (= Stage 3 position-conditioned adapter): ⏸ STOP per spec

**强制要求** (per Issue spec §Gate3):
- 在有限 encoder self-attention 层加入 position-conditioned product-stereographic residual
- 三层独立 κ/scale/gate (per L0/L1/L2 token position)
- Zero gate 必须与 control (#111) 数值等价 (logits max error 满足预设容差)
- 训练后做 adapter on/off、κ shuffle、L0/L2 swap、item-SID 对齐破坏
- PASS: zero-gate 等价; 训练后 adapter 稳定非零; 三层梯度有限非零; #111 SID attention 路径仍有效

**关键差异 vs #114**:
- #114 错误依赖 #113 component metadata → STOP
- #117 直接用 SID token position L0/L1/L2 条件化 κ/scale/gate (不需 #113 metadata)

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 必须先 PASS
- **强制要求**: 同 Task84 数据/seed/beam/evaluator, 报告 R@5/R@10/R@20/NDCG@5/NDCG@10/NDCG@20
- **Target reached**: adapter test R@10>0.1020
- 历史 task396b Stage 4 R@10=0.0864 (普通 T5 attention 真消费 SID, 但仍 < 0.1020)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #114 (已关闭) | Issue #117 (本 task) |
|------|---------------------|----------------------|
| **D1 spec 摘录** | 错误依赖 #113 (owner 撤销) | **撤销依赖, 直接 #102 SID token position 条件化** |
| **D2 实施核心** | (未执行 adapter) | **position-conditioned κ/scale/gate per L0/L1/L2 token** |
| **D3 Gate 失败机制** | (依赖误判, 路径错误) | **真实可执行路径, 不依赖 #113** |
| **D4 引用文献** | arXiv:2309.04082 (Curve Your Attention) | **arXiv:2309.04082 (Curve Your Attention) - 端到端学习曲率** |

**R18 v2 强制结论**: Issue #117 跟 #114 路径**有差异** (撤销依赖 #113 + 直接 SID token position 条件化). 必须做新实验, 不允许沿用判决.

**跟 #111 联立**:
- #111 已证明普通 T5 attention 真消费层级 SID (4 维度 PASS)
- #117 adapter 必须保留 #111 路径 (zero gate 等价)
- 历史 task396b Stage 4 R@10=0.0864 (普通 attention 真消费 SID 但 R@10<0.1020, 必须用 adapter 提升)

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Stage 1 上游 | task84 HG-Rec baseline (200 epoch) | #100 PASS 等价产物 (Issue #100 是 HG-Rec 复现任务, R5 baseline = task84) |
| SID | task396 SID (9922×4) | #102 PASS 等价产物 (Issue #102 = task396 Stage 2) |
| Adapter 设计 | 有限 encoder self-attention 层 (1-2 层) | Issue spec §Gate3: "在有限 encoder self-attention 层加入" |
| Position-conditioned κ/scale/gate | per L0/L1/L2 token position (3 独立组) | Issue spec §Gate3: "三层独立 κ/scale/gate" |
| Zero gate 初始化 | gate_logits init = -10 (≈ 0 weight, 等价 #111 control) | Issue spec §Gate3: "zero gate 必须与 control 数值等价" |
| GPU 分配 | GPU 2 (per R7 全部空闲时 GPU 0/1/2 各跑一个 Issue) | R7 + R19 跨 issue 并行 (Issue #115 GPU 0, #116 GPU 1) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 关键产物 (本 task 进行中)

- **Adapter 实施脚本**: `scripts/task410_issue117_position_conditioned_adapter.py` (待写)
- **Zero-gate 等价验证**: `verdicts/task410_issue117_zero_gate_equivalence.json` (待写)
- **Gate 3 训练 + 反事实**: `products/task410_issue117_adapter_train/` (待写)
- **Gate 4 Stage 4 eval**: `products/task410_issue117_stage4_eval/` (待写)
- **整体 verdict**: `verdicts/task410_issue117_position_conditioned_adapter_v2.md` (待写)
- **整体决策**: ⏸ 进行中

---

## 后续 (per R22 + R19 + R16)

1. **写 Adapter 脚本** (立即): 加载 T5 + SID token position embedding + position-conditioned κ/scale/gate
2. **Zero-gate 等价验证**: Gate 3 PASS 必须项 (跟 #111 control 比 logits max error)
3. **Adapter 训练 + 5 组反事实**: on/off, κ shuffle, L0/L2 swap, item-SID 对齐破坏
4. **Stage 4 R@K eval**: control vs adapter, report R@10 vs baseline 0.1020
5. **commit + push** (R15 + R21 v2)
6. **close Issue #117** (R16 + R20 + R21)