# Task #454 / Issue #162 [方向A Gate4] κ同步尺度适配正式双复跑 run 1 (seed=44, GPU 2)

## 任务背景

Issue #162 (方向A Gate4) 2026-08-01 owner 派发. Issue #159 Gate 3 PASS (commit 66210fa). Issue spec 强制:
- 200 epoch Stage 3 full training + Stage 4 R@K 双复跑
- **α 稳定策略必备** (有界 softplus / clip / regularization, 防止 #159 α 0.12→18.75 斜率外推 e^50 量级)
- 完整六项: R@5/10/20 + NDCG@5/10/20 (baseline R@10=0.1020)
- 两次独立正式 Stage 3 run (不同 seed, 不同 ckpt SHA256)
- 不得更换 SID / 数据切分 / evaluator / baseline / T5 冻结边界 / 三层独立 κ 来源

## R18 4 维度对比 (vs Task #451 Issue #159 10 epoch Gate 3 sanity)

| 维度 | Task #451 (#159 Gate 3 sanity) | Task #454 (#162 Gate 4 run 1) |
|------|--------------------------------|-------------------------------|
| **D1 spec** | 10 epoch sanity + α 无界 | 200 epoch full + α 有界 (clamp ≤ 1) |
| **D2 实施** | α unbounded softplus | α bounded softplus + alpha_reg_loss |
| **D3 Gate 3 失败机制** | α 外推 e^50 量级, 200 epoch 数值风险 | α clamp 到 [0, 1] 避免外推 |
| **D4 引用** | Issue #159 spec Gate 3 sanity | Issue #162 spec Gate 4 强制 α 稳定 + 双复跑 |

→ **4 维度全部不一致**, R18 实验强制 (新代码 + 新训练 + 新验证) 必备.

## 实施目标 (Issue #162 spec Gate 4)

1. **种子独立性**: seed=44 (跟 Task #455 seed=45, Task #450 seed=42, Task #453 seed=43 都不同)
2. **GPU 隔离**: GPU 2 (Task #450/#453 占 GPU 0/1, Task #455 占 GPU 3)
3. **α 稳定策略 (spec 强制)**: 
   - **方法选择**: α bounded softplus + L2 regularization (跟 Issue #162 spec "可审计"要求匹配)
   - **α clamp**: α = min(F.softplus(alpha_logit), 1.0) (有界 ≤ 1)
   - **α reg**: loss += 1e-3 * (alpha_logit ** 2) (L2 抑制增长)
4. **R137 TRITON_CACHE_DIR**: per-task 隔离
5. **Stage 3 训练**: 200 epoch full
6. **Stage 4 R@K eval**: 200 epoch 训练完立即跑双复跑 run 1
7. **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt

## 关键路径 (跟 Task #451 / #450 锚定一致)

- 复用 Task #451 `KappaScaleConditioner` + `HG_Rec_with_KappaScaleAdapter` (但加 α clamp + reg)
- 锚定 Task #84 ckpt: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (SHA256 56d046db...)
- 锚定 Musical_Instruments 数据集 (SHA 同 Task #450/#451)

## 防错机制

- **R7 GPU**: 启动 GPU 2 (Task #450/#453 占 GPU 0/1, Task #455 占 GPU 3)
- **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新
- **R15 push**: 训练结束立即 commit + push
- **R20 4 Gate 详细**: commit message + Issue #162 comment 每个 Gate ≥3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: 写完 4 Gate 详细 comment + close issue with --reason completed
- **R137**: TRITON_CACHE_DIR=~/.triton/cache_task454

## 8 件套 + Stage 4 产物

1. `config.json` — seed=44, GPU 2, α clamp + reg 配置
2. `adapter_init_proof.json` — α=0 → max diff=0
3. `layernorm_unfreeze_proof.json` — 训练参数清单
4. `gradient_proof.json` — 首步后 conditioner + LN 梯度 nonzero
5. `train_trace_200ep.json` — 200 epoch 训练 trace
6. `verdict.json` — gate3_decision PASS (α clamp 验证 + L2 reg 验证)
7. `adapter_200ep.pt` — R12 ckpt
8. `log` — `logs/task454_issue162_gate4_run1_seed44.log`
9. `eval_run1.json` — Stage 4 R@K eval 六项指标
10. `stage4_verdict.json` — run 1 GO/NO-GO 决策 (跟 baseline 0.1020 对比)

## Gate 4 决策阈值

- **GO**: run 1 R@10 > 0.1020 + 六项指标齐全
- **NO-GO**: run 1 R@10 ≤ 0.1020 或六项指标缺失

## R11.5 自主决策

- **run 1 启动**: 立即 (Issue #162 spec 强制双复跑, GPU 2 空闲)
- **α 稳定策略**: bounded softplus + L2 reg (R11.5 兜底: 有界可审计, 兼容 Issue #162 spec "simplex归一化/有界softplus/clip/正则化" 任一)
- **并行 run 2 (Task #455 GPU 3)**: 立即 (R19 跨 issue 并行)
- **并行 Task #450/#453 (方向C 双复跑 GPU 0/1)**: 已在跑 (R7 + R19)
- **Stage 4 双复跑 run 1 + run 2**: 两份 run 都完成后立即跑 + 跟 baseline 对比
- **Issue #162 close**: 双复跑完成后立即 R20+R21 v2+R16 闭环