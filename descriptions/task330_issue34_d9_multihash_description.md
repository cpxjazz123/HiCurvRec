# Task #330 — Issue #34 D9 多样 hash on #30 GO 配置

**日期**: 2026-07-30
**状态**: 🔄 PENDING (Issue #34 OPEN, R10 backlog 候选)
**Stage**: Stage 0 — Description 注册 + 5-Gate 协议设计
**Anchor**:
- Issue #30 GO endpoint R@10=0.1022 (+0.2% vs baseline 0.1020, 唯一 GO 实证)
- HG-Rec Task #84 baseline R@10=0.1020
- Issue #34 GitHub issue spec (2026-07-29 owner R11.5 自启, D9 多样 hash = owner 暗示唯一剩余方向)

## Issue #34 核心方向 (D9 多样 hash)

按 Issue #34 完整 spec: 在 #30 GO 端点 (per-layer 异构 r_l=[0.1,1,10] + s_l=[2,2,2] + hard argmin commitment) 基础上, **新增 per-layer 异构 hash 函数族维度**:

- **L0 (K=64)**: sparse random projection hash + binary collision check → top-3 SID candidates per item
- **L1 (K=128)**: locality-sensitive hashing (LSH) on residual + multi-probe → top-5 SID candidates per item
- **L2 (K=256)**: k-means bucket hash + multi-bucket assignment → top-7 SID candidates per item

**关键设计选择**:
- 硬 argmin commitment 保留 (与 #30 GO 一致, 不重蹈 #33 per-item soft commitment collapse)
- 新增维度 = hash 函数族多样性 (per-layer 异构)
- 保留 baseline 多码字几何 (#30 GO 真杠杆)

**5-Gate 协议**:
1. **Gate 0**: 实现 per-layer 异构 hash 函数族训练代码 (双回归测试)
2. **Gate 1**: Stage 1 训练 100 epoch (GPU) — L0/L1/L2 util ≥ 90% + collision ≤ 0.25 + norm 健康区 ‖x‖_E ∈ [0.7, 0.95] + hash candidates 有效性
3. **Gate 2**: Stage 2 Sinkhorn 推断 (cheap, 5 iter) — 4-digit SID unique ≥ 9500
4. **Gate 3**: Stage 3 T5-mini 200 epoch 训练 (GPU)
5. **Gate 4**: Stage 4 Test eval (GPU) — Test R@10 > 0.1022 (双重对照 #84 baseline 0.1020 + #30 端点 0.1022)

**硬停止**: 任何 Gate 失败 → STOP, 不跨 Gate 取数.

## R10 推进条件

- Issue #34 OPEN (2026-07-29 owner 暗示 D9 多样 hash 是唯一剩余方向)
- Issue #33 closure comment 锁定 "per-item expected-loss commitment = collapse, 禁试" → 本 issue 用 hard argmin + hash 多样性 (与 #33 数学机制不同)
- task328 R-Drop alpha sweep 当前 GPU 0/2/3 running (待落地), task327 K=256 Issue #30 synergy 当前 GPU 1 Stage 3 (待落地 ~13:50)
- 落地顺序: task328 → task327 → task330 (R10 兜底, 等 GPU 全空闲后再启动)

## R9 compliance

- 本任务编号 #330 = max(329) + 1 ✅
- Issue #34 (D9 多样 hash) 是 owner R11.5 自启 issue, 本 description 锁定 task #330 为执行入口

## 决策阈值 (R11.5 transparency)

- ✅ **GO**: Gate 4 Test R@10 > 0.1022 (突破 #30 端点 + baseline 双锁) = D9 多样 hash 在 #30 GO 端点上贡献真实 R@10 增益
- ❌ **NO-GO**: Gate 4 Test R@10 ≤ 0.1022 (不突破 #30 端点) = D9 多样 hash 候选路径 NO-GO, 关闭本 issue
- 任一 Gate 0/1/2/3 FAIL → 立即硬停止, 不跨 Gate 取数

## Critical caveats (R11.5 transparent)

1. **R7 GPU 占用约束**: task330 必须等 GPU 0/1/2/3 全空闲 (~14:00 后) 才启动
2. **Issue #33 collapse 警示**: 本 issue 保留 hard argmin commitment, 不引入 expected-loss commitment (per owner #33 closure)
3. **Issue #32 灾难警示**: 本 issue 保留 #30 r_l=[0.1,1,10] + s_l=[2,2,2] 极端值配置, 不走温和 r_l+s_l
4. **Gate 0 双回归测试**: 必须证明 per-layer hash OFF 时 = #30 端点 (R@10=0.1022), 不能跨过
5. **arXiv 检索**: Issue #34 已跑 1 query × 10 hits 全跑题, 无外部候选对照
6. **资源估计**: Gate 0 + Gate 1 + Gate 2 + Gate 3 + Gate 4 总估 5-6 hr GPU (Stage 1 100 epoch ~30 min + Sinkhorn 5 min + T5-mini 200 epoch ~90 min + Stage 4 eval 5 min + 回归测试 1 hr)

## Reference verdicts (跟当前 backbone 一致)

- Issue #30 GO endpoint R@10=0.1022 (task301 Stage 4 唯一 GO 实证)
- HG-Rec Task #84 baseline R@10=0.1020
- task328 R-Drop alpha sweep (running, Issue #38 Layer 2)
- task327 K=256 + Issue #30 synergy (running, GPU 1 Stage 3 ~30 min)
- task329 Issue #40 Gate 0 FAIL (task194_k0256 0.1053 anchor 不可信)
- Issue #32 closure: 灾难 NO-GO (r_l=[0.5,1,2] + s_l=[1,1,1] + c_k range 双轴温和毁坏 #30 杠杆)
- Issue #33 closure: NO-GO (per-item soft-assign collapse, expected-loss commitment 禁试)