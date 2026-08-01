# Task #456 / Issue #163 [方向B Gate4] 加权混合曲率适配正式单 seed run (seed=46, GPU 1)

## 任务背景

Issue #163 (方向B Gate4) 2026-08-01 owner 派发. Issue #160 Gate 3 PASS (commit 66210fa). Issue spec 强制:
- 200 epoch Stage 3 full training + Stage 4 R@K 双复跑
- **混合权重稳定策略必备** (simplex 归一化 / 有界 softplus / clip / 正则化, 防止 #160 α 0.12→15.41 斜率外推 e^40 量级)
- 完整六项: R@5/10/20 + NDCG@5/10/20 (baseline R@10=0.1020)
- 两次独立正式 Stage 3 run (不同 seed, 不同 ckpt SHA256)
- 不得把三分量折叠成静态欧氏特征, 不得更换 SID / 数据切分 / evaluator / baseline / T5 冻结边界 / 三层独立 κ 来源

## R18 4 维度对比 (vs Task #452 Issue #160 10 epoch Gate 3 sanity)

| 维度 | Task #452 (#160 Gate 3 sanity) | Task #456 (#163 Gate 4 run 1) |
|------|--------------------------------|-------------------------------|
| **D1 spec** | 10 epoch sanity + 混合权重无界 | 200 epoch full + 混合权重有界 (softmax + L2 reg) |
| **D2 实施** | α unbounded softplus | α bounded softplus + 混合权重 simplex L2 reg |
| **D3 Gate 3 失败机制** | 混合权重外推 e^40 量级, 200 epoch 数值风险 | 混合权重 softmax 归一化 + L2 reg 抑制 |
| **D4 引用** | Issue #160 spec Gate 3 sanity | Issue #163 spec Gate 4 强制混合权重稳定 + 双复跑 |

→ **4 维度全部不一致**, R18 实验强制.

## 实施目标 (Issue #163 spec Gate 4)

1. **种子独立性**: seed=46 (vs Task #457 seed=47)
2. **GPU 隔离**: GPU 2 (跟 Task #454 run 1 共占 GPU 2 排队: Task #454 完成后立即启动)
3. **混合权重稳定策略 (spec 强制)**:
   - **方法选择**: curvature_embed 输出 → softmax 归一化 → 每层混合权重 ∈ [0, 1] 且和=1 (simplex) + L2 reg
   - **混合权重 clamp**: curvature_meta softmax → 每层 [w_κ, w_α, w_β, w_γ] simplex
   - **混合权重 reg**: loss += 1e-3 * Σ (logits^2) (L2 抑制极端值)
4. **R137 TRITON_CACHE_DIR**: per-task 隔离 (`/home/wlia0047/.triton/cache_task456`)
5. **Stage 3 训练**: 200 epoch full
6. **Stage 4 R@K eval**: 200 epoch 训练完立即跑双复跑 run 1
7. **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt

## 关键路径 (跟 Task #452 锚定一致)

- 复用 Task #452 `WeightedMixedCurvatureConditioner` (但加 softmax 归一化 + L2 reg)
- 锚定 Task #84 ckpt + Musical_Instruments 数据集 (SHA 同 Task #450)

## 防错机制

- **R7 GPU 排队**: GPU 2 等 Task #454 完成后释放; GPU 3 等 Task #455 完成后释放
- **R12 ckpt / R15 push / R20 4 Gate / R21 v2 / R16 close / R137**: 跟 Task #454 同

## 8 件套 + Stage 4 产物

跟 Task #454 同, 但 seed=46 / GPU 2 排队 / task456 / cache_task456 + 混合权重稳定策略.

## Gate 4 决策阈值

- **GO**: run 1 R@10 > 0.1020 + 六项指标齐全
- **NO-GO**: run 1 R@10 ≤ 0.1020 或六项指标缺失

## R11.5 自主决策

- **run 1 启动**: 等 GPU 2 释放 (Task #454 完成后立即启动, R19 跨 issue 并行)
- **混合权重稳定策略**: simplex softmax + L2 reg (跟 #162 spec 同模式, R11.5 兜底)
- **并行 run 2 (Task #457 GPU 3)**: 等 GPU 3 释放 (Task #455 完成后立即启动)
- **Stage 4 双复跑**: 两份 run 都完成后立即跑
- **Issue #163 close**: 双复跑完成后立即 R20+R21 v2+R16 闭环