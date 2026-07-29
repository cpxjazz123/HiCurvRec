# Task #312 — Issue #35 r_l=[1,1,1] identity + s_l=[2,2,2] 验证 s_l 是不是真杠杆

**日期**: 2026-07-30
**状态**: 启动 (Issue #35 OPEN → task312)
**目的**: 在 #30 GO 配置基础上, 把 r_l 改为 identity (1, 1, 1), 保留 s_l=[2,2,2]. 验证 s_l 单独是不是 R@10 杠杆 (vs r_l 极端值 + s_l 协同).

**R11.5 自主决策**: Issue #35 16-candidate sweep 信息量高但 GPU 重 (16 × Stage 1 = 8 GPU-h). R11.5 简化决策: 跑 1 个最有信息量的 single candidate = **r_l=[1,1,1] identity + s_l=[2,2,2]**, 验证 s_l 是不是真杠杆.

**关联**:
- Issue #35 (owner 指示 #1)
- Issue #30 (r_l=[0.1, 1, 10] + s_l=[2,2,2] R@10=0.1022)
- Issue #32 (r_l=[0.5, 1, 2] + s_l=[1,1,1] R@10=0.000121)
- task304 (D6 ablation 3-arm)

**5-Gate 协议**:
- Gate 0: 配置 + ckpt 路径 + GPU 空闲检查
- Gate 1: Stage 1 100 epoch 训练 (跟 #30 一致)
- Gate 2: Sinkhorn 5 iter + 4-digit dedup
- Gate 3: T5-mini Stage 3 200 epoch 训练
- Gate 4: Stage 4 eval (beam=50, K14 最优)

**GPU**: 0 (R7 空闲, task309 Stage 3 用 GPU 1)
**预期**: ~3-4h 总 (Stage 1 1-2h + Stage 3 90min + Stage 4 36s)
**关键决策点 (R11.3 透明)**:
- 选 r_l=[1,1,1] 因为 (a) identity baseline = 0 baseline r_l 杠杆 (b) s_l=[2,2,2] 保留 = 隔离 s_l 边际效应
- 如 r_l=[1,1,1] + s_l=[2,2,2] R@10 ≈ 0.1022 → s_l 是真杠杆
- 如 r_l=[1,1,1] + s_l=[2,2,2] R@10 ≪ 0.1022 → r_l 极端值是必要条件, s_l alone 不够
