# Task #407 / Issue #113 [方向B Precheck→Gate1] 真实三分量 product checkpoint schema 与防坍缩准入

**日期**: 2026-07-31
**Issue**: #113 [方向B Precheck→Gate1] — 真实三分量 product checkpoint schema + 防坍缩准入
**任务**: 
1. **Precheck**: 实现并验证真实三分量 (learnable-κ hyperbolic + fixed-curvature hyperbolic + Euclidean) product checkpoint schema + 每层独立 learnable mixing/gate, 不接受 stub / 口头合同 / base RQ-VAE 冒充 product
2. **Gate 1**: Precheck PASS 后做受限短训, 三层 utilization ≥90% + max_load <5% + L1/L2 norm ≥ L0×0.2 + 三 component 有限非零 + gate 非退化 + 无 NaN/Inf + round-trip 一致
**前置**:
- task404 (Issue #110 Gate 1 FAIL): Stage 1 ckpt 不是 mixed-curvature product, L1/L2 norm 坍缩 (0.0273/0.0416)
- task405 (Issue #109 Gate 3 FAIL): Stage 3 ckpt 不持 κ/scale metadata channel
- task406 (Issue #111 Gate 3 PASS): T5 真的消费层级 SID 几何 (token identity 维度), 瓶颈在 Stage 1→2 信息丢失
- 历史 task391 (Issue #98 audit) + task397 (Issue #101 per-component monkey-patch) 都 NO-GO 收口
**区别于历史任务**: Issue #113 spec §证据 §去重 明确要求"真实 state_dict schema + forward contribution + 可重载 checkpoint" 硬准入, **不允许 per-comp learnable-κ + mean agg 冒充** (跟 task397 不同). 三分量 = 固定 3 个 component 类型 + softmax gate, 不是 per-comp M 个独立 learnable-κ 通道
**结果**: ⏸ 进行中 (Precheck 必须先 PASS 才能进 Gate 1)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE Precheck + 短训): ⏸ 进行中

**Precheck PASS 强制要求** (per Issue #113 spec §Framework compliance precheck):
- **三层 = L0 K64 / L1 K128 / L2 K256** (固定)
- **每层 3 个 component**:
  1. **Learnable-κ hyperbolic 主路**: 保留 `FreeCurvVectorQuantization` 的 `theta_m → κ_m = κ_max·tanh(θ_m)` (R137 fix)
  2. **Fixed-curvature hyperbolic component**: 固定 κ (per layer) 的 Poincaré distance 分量, 不参与梯度
  3. **Euclidean component**: 固定 κ=0 的 L2 距离分量
- **每层独立 learnable mixing/gate**: `gate_logits_l ∈ R^3` → `softmax → w_l ∈ [0,1]^3, Σ w_l = 1`, per-layer 独立, 不能 global / 不能 constant
- **checkpoint key/shape 表**: 必须存在 6 组 key = `layers.{0,1,2}.component_{learnable,fixed,euclidean}.theta/codebook/embedding` + `layers.{0,1,2}.gate_logits`
- **forward 数据流**: `d_total = Σ_m w_m · d_κ_m(x_m, c_m)`, 三 component 都参与 forward, 不允许删除任何 component
- **save/load round-trip**: 重新加载后复算同一 batch, 输出必须一致
- **禁止**: global κ / fixed-only / 删除原主路 / gate 常量化 / 纯欧旁路

**Precheck FAIL 条件**: 任一 component 或 gate 缺失/全零/不参与 forward, 或 checkpoint 无法重载

**Precheck PASS 后 Gate 1 短训要求** (per Issue #113 spec §Gate1):
- **每 epoch 记录**:
  - 三层 utilization (L0/L1/L2)
  - max_load
  - entropy (per layer)
  - 各 component contribution (learnable-κ / fixed-curv / Euclidean)
  - gate (softmax weights)
  - codebook norm (per layer, per component)
  - loss (commitment + codebook + reconstruction)
- **best checkpoint 保存**: 必须 round-trip 一致 (重新加载后复算同一 batch, 输出 bit-equal)
- **PASS 阈值**:
  - 三层 utilization ≥ 90%
  - max_load < 5%
  - L1/L2 norm ≥ L0×0.2 (防坍缩准入)
  - 三 component 有限非零 (无 NaN/Inf)
  - gate 非退化 (≥2 个 component weight > 0.1)
  - 无 NaN/Inf
  - round-trip 一致
- **FAIL**: 任一分量缺失/全零、gate 退化单分量、L1/L2 坍缩、或 checkpoint 无法重载

**决策**: Gate 1 PASS 前禁止 Gate 2 (per spec)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- **原因**: Gate 1 必须先 PASS 才能进 Gate 2
- **Issue spec §Gate2 强制**: "仅 Gate1 PASS 后生成 9922×4 SID, 并报告 unique, collision, 三层 entropy/utilization/max_load, component/gate 对照及 item 对齐"

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- **原因**: Gate 2 必须先 PASS 才能进 Gate 3
- **Issue spec §Gate3 强制**: "仅 Gate2 PASS 后定义 component/gate/learned-κ metadata 到 T5 的接口, 并完成 on/off, shuffle, gradient 验证"

### Gate 4 (= Stage 4 R@K): ⏸ STOP per spec
- **原因**: Gate 3 必须先 PASS 才能进 Gate 4
- **Issue spec §Gate4 强制**: "仅 Gate3 PASS 后统一报告 test R@5/R@10/R@20/NDCG; R@10>0.1020 才是 Target reached"

---

## 跟历史任务路径差异 (R18 强制 4 维度对比)

| 维度 | task391 (Issue #98) | task397 (Issue #101) | **Issue #113 (本 task)** |
|------|---------------------|---------------------|--------------------------|
| **D1 spec 摘录** | audit: 定位 scale_normalization 缺失 | fix: per-comp std norm + mean agg | **真实三分量 schema + 防坍缩准入** |
| **D2 实施核心** | audit-only (无 fix) | per-comp z-score + mean aggregation | **固定 3 comp + per-layer softmax gate + round-trip** |
| **D3 Gate 1 失败机制** | synthetic agree3=0%, init agree3=4.20% | 50 epoch 仍坍缩 (NO-GO 收口) | **Precheck hard-block: 缺任一 comp/gate 即 blocked** |
| **D4 引用文献** | arXiv 2307.04514 mixed-curvature | arXiv 2307.04514 + ACE-HGNN DOI | **arXiv 2307.04514 + CrossRef 10.1109/ICDM51629.2021.00021 (ACE-HGNN)** |

**R18 v2 强制结论**: Issue #113 跟 task391/task397 路径**有差异** (固定三分量 + softmax gate vs per-comp learnable κ + mean agg). 必须做新实验, 不允许沿用判决.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Issue 选择 | #113 (Precheck→Gate1) 优先于 #112/#114 | R22 + R18: #113 是架构基座, #112/#114 依赖其 metadata source |
| Stage 1 训练时长 | 50 epoch (vs HG-Rec baseline 200 epoch) | Issue spec §Gate1 "受限短训" + task402 已证 Stage 1 30-50 epoch 足够验证 |
| checkpoint schema 命名 | `layers.{l}.component_{learnable,fixed,euclidean}.{theta,codebook}` + `layers.{l}.gate_logits` | Issue spec §Framework compliance precheck 强制要求 |
| Gate 1 FAIL 早停 | USAGE-KILL @ ep30 (L0<20%) OR 任一 component NaN/Inf OR gate 退化单 weight>0.99 | Issue spec §Gate1 PASS 阈值 + 历史经验 (task178/task180/task231/task242/topic295 Phase 0 mode collapse 模式) |
| GPU 分配 | GPU 1 (task403 GPU 0 已释放, 4 卡全空闲) | R7: 不抢已占卡, 全部空闲时 GPU 1 优先 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务用 genrec_env |

---

## 关键产物 (本 task 进行中)

- **Precheck 脚本**: `scripts/task407_issue113_three_component_precheck.py` (待写)
- **Stage 1 训练脚本**: `scripts/task407_issue113_three_component_stage1_train.py` (待写)
- **checkpoint schema doc**: `docs/task407_three_component_schema.md` (待写)
- **Gate 1 verdict**: `verdicts/task407_issue113_three_component_product_gate1_v2.md` (待写)
- **整体决策**: ⏸ 进行中

---

## 后续 (per R22 + R19 + R16 + R10)

1. **写 Precheck 脚本** (立即): 实现 ThreeComponentVQ + ThreeComponentHRQVAE wrapper, 验证 schema + forward + round-trip
2. **Precheck 验证**: python3 -m py_compile + 跑 50 step 测试 + key/shape 审计
3. **Stage 1 训练** (Precheck PASS 后): 50 epoch, GPU 1, 立即 launch
4. **Gate 1 verdict**: PASS/FAIL per spec 阈值, 4 Gate 详细内容 + 关键数据
5. **commit + push** (R15 + R21 v2): 含具体 commit hash, NO pending
6. **close Issue #113** (R16 + R20 + R21): 4 Gate 详细 comment + commit hash
7. **若 Precheck/Gate 1 PASS → 立即推进 Issue #112 (task425) + Issue #114 (task427)**
8. **若 Precheck/Gate 1 FAIL → 落 NO-GO verdict + close issue + 联立 task391/task397 推论架构不可行**