# Task #313 — Issue #35 r_l=[0.1,1,10] (Issue #30 GO) + s_l=[1,1,1] identity 验证 r_l 是不是真杠杆

**日期**: 2026-07-30
**状态**: 启动 (Issue #35 OPEN → task313)
**目的**: 在 #30 GO 配置基础上, 把 s_l 改为 identity (1, 1, 1), 保留 r_l=[0.1,1,10] (Issue #30 GO). 验证 r_l 单独是不是 R@10 杠杆 (vs s_l 极端值 + r_l 协同).

**R11.5 自主决策**: Issue #35 16-candidate sweep 信息量高但 GPU 重. R11.5 简化决策: 跑 2 个最有信息量的 orthogonal single candidates:
- **task312 (GPU 0, 已启动)** = r_l=[1,1,1] identity + s_l=[2,2,2] (测 s_l 边际效应)
- **task313 (GPU 2, 本任务启动)** = r_l=[0.1,1,10] (Issue #30 GO) + s_l=[1,1,1] identity (测 r_l 边际效应)

两者 orthogonal, 联合诊断 Issue #30 GO +0.2pp marginal 的真杠杆.

**关联**:
- Issue #35 (owner 指示 #1, 后续 cron tick 4 方向 ablation)
- Issue #30 (r_l=[0.1,1,10] + s_l=[2,2,2] R@10=0.1022 marginal GO)
- Issue #32 (r_l=[0.5,1,2] + s_l=[1,1,1] R@10=0.000121, 灾难 NO-GO)
- task312 (orthogonal 反向, 测 s_l)
- task304 (D6 ablation 3-arm)
- Task #287 κ-decouple Phase A (K frozen=0) + #313 = 两条独立杠杆线

**5-Gate 协议**:
- Gate 0: 配置 + ckpt 路径 + GPU 空闲检查
- Gate 1: Stage 1 100 epoch 训练 (跟 #30 一致)
- Gate 2: Sinkhorn 5 iter + 4-digit dedup
- Gate 3: T5-mini Stage 3 200 epoch 训练
- Gate 4: Stage 4 eval (beam=50, K14 最优)

**GPU**: 2 (R7 空闲, task309 Stage 3 用 GPU 1, task312 Stage 3 用 GPU 0)
**预期**: ~3.5h 总 (Stage 1 1-2h + Stage 3 90min + Stage 4 36s)
**关键决策点 (R11.3 透明)**:
- 选 r_l=[0.1,1,10] (Issue #30 GO) 因为 (a) 是 #30 GO 配置的核心 (b) 隔离 s_l 边际效应
- 选 s_l=[1,1,1] identity 因为 (a) baseline 不缩放 (b) 对比 task312 r_l identity
- 如 r_l=[0.1,1,10] + s_l=[1,1,1] R@10 ≈ 0.1045 → r_l 是真杠杆
- 如 r_l=[0.1,1,10] + s_l=[1,1,1] R@10 ≪ 0.1045 → s_l 极端值是必要条件, r_l alone 不够

**跟 Issue #32 区分**:
- Issue #32 = r_l=[0.5,1,2] + s_l=[1,1,1] + c_k range → 灾难 NO-GO R@10=0.000121
- task313 = r_l=[0.1,1,10] + s_l=[1,1,1] (no c_k range) → 单独测 r_l 边际
- 如果 task313 也 NO-GO → r_l extreme + s_l identity 不够, 必须 r_l+s_l 协同 (Issue #30 唯一 GO)
- 如果 task313 GO → r_l alone 是杠杆 (vs task312 测 s_l alone)