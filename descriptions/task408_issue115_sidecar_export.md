# Task #408 / Issue #115 [方向A Gate3恢复] 独立导出 #100 Stage 1 sidecar + T5 adapter

**日期**: 2026-07-31
**Issue**: #115 [方向A Gate3恢复] — 撤销 #112 错误依赖 (#112 Gate 1 明确复用 #100, 不应依赖 #113)
**任务**: 
1. **Gate 1**: 加载 task84 (HG-Rec baseline = #100 等价) Stage 1 ckpt + 导出每层 θ/κ/scale/codebook norm + SHA256
2. **Gate 2**: 生成与 9922×4 SID 按 item/层对齐的 sidecar (shape/dtype/checksum/variance/item alignment/NaN/Inf)
3. **Gate 3**: 在原 SID embedding 上添加三层独立 adapter/gate + 同 checkpoint 5 组反事实 (原始/全零/层间置换/#47 一致重标定/故意不一致)
4. **Gate 4**: 完整 checkpoint 训练 + test R@5/R@10/R@20/NDCG@5/NDCG@10/NDCG@20 (R@10>0.1020 才 Target reached)

**前置** (per spec §Framework compliance precheck):
- #100 Gate 1 PASS (task84 HG-Rec baseline Stage 1)
- #103 Gate 2 PASS (task84 Sinkhorn SID)
- 不依赖 #113 (owner 撤销 #112 错误依赖, 独立 A 自身上游)

**结果**: ⏸ 进行中 (R22 + R19 立即开工)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 sidecar 导出): ⏸ 进行中

**强制要求** (per Issue spec §Gate1):
- 加载 task84 HG-Rec baseline Stage 1 ckpt (200 epoch best_collision_model.pth, = #100 PASS 等价产物)
- 导出每层: θ (learnable curvature parameter), κ (effective curvature), scale, codebook norm, K 数量, checkpoint SHA256
- 三层 = L0 K64 / L1 K128 / L2 K256 (固定)
- 若 checkpoint 实际缺失或三层不符 → Gate 1 FAIL (不得转用 #113)

**实施**:
- `scripts/task408_issue115_sidecar_export.py` (待写)
- 加载: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth`
- 解析 state_dict (HG-Rec baseline `FreeCurvHRQVAE` schema)
- 输出: `verdicts/task408_issue115_sidecar.json` (含每层 θ/κ/scale/norm/SHA256)

### Gate 2 (= Stage 2 SID sidecar 对齐): ⏸ STOP per spec
- **原因**: Gate 1 必须先 PASS

### Gate 3 (= Stage 3 T5 adapter): ⏸ STOP per spec
- **原因**: Gate 1+2 必须先 PASS
- **强制要求** (per Issue spec §Gate3):
  - 在原 SID embedding 上添加三层独立 adapter/gate
  - 输入为对应层 κ/scale/norm sidecar
  - checkpoint 必须包含 adapter/gate keys, 三层梯度有限非零
  - 同 checkpoint 5 组反事实 (原始/全零/层间置换/#47 一致重标定/故意不一致)
  - PASS: 原始与破坏同步组差异稳定非零, 一致重标定满足预期不变量, 三层均参与

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec
- **原因**: Gate 1+2+3 必须先 PASS

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #112 (已关闭) | Issue #115 (本 task) |
|------|---------------------|----------------------|
| **D1 spec 摘录** | 错误依赖 #113 (owner 撤销) | **撤销依赖, 独立 #100/#103 上游** |
| **D2 实施核心** | (未执行 adapter) | **加载 task84 ckpt + 导 sidecar + T5 adapter** |
| **D3 Gate 失败机制** | (依赖误判, 路径错误) | **真实可执行路径, 不依赖 #113** |
| **D4 引用文献** | arXiv:2405.13979 | **arXiv:2405.13979 (curvature-aware)** |

**R18 v2 强制结论**: Issue #115 跟 #112 路径有差异 (撤销依赖 #113 + 独立 #100 上游). 必须做新实验, 不允许沿用判决.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Stage 1 上游 | task84 HG-Rec baseline (200 epoch) | #100 PASS 等价产物 (Issue #100 是 HG-Rec 复现任务, R5 baseline = task84) |
| Sidecar 输出 | 每层 θ, κ=κ_max·tanh(θ), scale (per-layer), codebook norm, K | Issue spec §Gate1 强制要求 |
| 5 组反事实 | 原始/全零/层间置换/#47 一致重标定/故意不一致 | Issue spec §Gate3 明确列出 |
| GPU 分配 | GPU 0 (per R7 全部空闲时优先 GPU 0) | R7 + R19 跨 issue 并行 (Issue #116 占用 GPU 1, Issue #117 占用 GPU 2) |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |

---

## 关键产物 (本 task 进行中)

- **Sidecar export 脚本**: `scripts/task408_issue115_sidecar_export.py` (待写)
- **Sidecar JSON**: `verdicts/task408_issue115_sidecar.json` (含每层 θ/κ/scale/norm/SHA256)
- **T5 adapter 训练脚本**: `scripts/task408_issue115_t5_adapter_train.py` (待写)
- **Gate 1 verdict**: `verdicts/task408_issue115_sidecar_gate1_v2.md` (待写)
- **整体决策**: ⏸ 进行中

---

## 后续 (per R22 + R19 + R16)

1. **写 Sidecar 导出脚本** (立即): 加载 task84 ckpt + enumerate state_dict + 提取 θ/κ/scale/norm
2. **Gate 1 audit**: 验证三层 schema + 三层有限非零 + round-trip
3. **T5 adapter 实施**: Gate 1+2 PASS 后, 实施 adapter + 5 组反事实
4. **Stage 3 训练 + Gate 3 PASS/FAIL**: 200 epoch T5-mini 训练 + 反事实验证
5. **Stage 4 R@K eval**: Gate 3 PASS 后, test R@K
6. **commit + push** (R15 + R21 v2): 含具体 commit hash, NO pending
7. **close Issue #115** (R16 + R20 + R21): 4 Gate 详细 comment + commit hash