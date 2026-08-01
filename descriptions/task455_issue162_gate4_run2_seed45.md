# Task #455 / Issue #162 [方向A Gate4] κ同步尺度适配正式双复跑 run 2 (seed=45, GPU 3)

## 任务背景

Issue #162 (方向A Gate4) 第二份独立 run. 跟 Task #454 (run 1 seed=44 GPU 2) 并行.

## R18 4 维度对比 (vs Task #454 run 1 seed=44 GPU 2)

| 维度 | Task #454 (run 1, seed=44, GPU 2) | Task #455 (run 2, seed=45, GPU 3) |
|------|-----------------------------------|-----------------------------------|
| **D1 spec** | seed=44, GPU 2, 200 epoch full + α 有界 | seed=45, GPU 3, 200 epoch full + α 有界 (独立初始化) |
| **D2 实施** | adapter init = seed 44 | adapter init = seed 45 (不同初始权重) |
| **D3 Gate 4 决策** | 双复跑 run 1, Stage 4 eval 后产出 eval_run1.json | 双复跑 run 2, Stage 4 eval 后产出 eval_run2.json |
| **D4 引用** | Issue #162 spec Gate 4 = 两份独立 run | Issue #162 spec Gate 4 = 两份独立 run |

→ **4 维度全部不一致**, R18 实验强制.

## 实施目标 (Issue #162 spec Gate 4 run 2)

1. **种子独立性**: seed=45 (vs Task #454 seed=44)
2. **GPU 隔离**: GPU 3 (Task #454 占 GPU 2)
3. **α 稳定策略**: 同 Task #454 (bounded softplus + L2 reg)
4. **R137 TRITON_CACHE_DIR**: per-task 隔离 (`/home/wlia0047/.triton/cache_task455`)
5. **Stage 3 训练**: 200 epoch full
6. **Stage 4 R@K eval**: 200 epoch 训练完立即跑 run 2 eval
7. **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt, SHA256 跟 run 1 完全不同

## 关键路径

- 跟 Task #454 同 (codebook_size=[64,128,256,1], max_len=20, batch_size=32, d_model=128, lr_conditioner=1e-3, lr_layernorm=1e-4, alpha_init=0.0, α bounded softplus + L2 reg)
- 复用 Task #451 `KappaScaleConditioner` (但加 α clamp + reg)
- 锚定 Task #84 ckpt + Musical_Instruments 数据集 (SHA 同 Task #450/#451/#454)

## 防错机制

- **R7 GPU**: 启动 GPU 3 (Task #454 占 GPU 2)
- **R12 ckpt / R15 push / R20 4 Gate / R21 v2 / R16 close / R137**: 跟 Task #454 同

## 8 件套 + Stage 4 产物

跟 Task #454 同, 但 seed=45 / GPU 3 / task455 / cache_task455.

## Gate 4 决策阈值

- **GO**: 双复跑 (run 1 + run 2) 平均 R@10 > 0.1020 + 六项指标齐全
- **NO-GO**: 任一 run R@10 ≤ 0.1020 或六项指标缺失

## R11.5 自主决策

- **run 2 启动**: 立即 (Issue #162 spec 强制双复跑, GPU 3 空闲, 跟 Task #454 并行)
- **α 稳定策略**: 跟 Task #454 一致 (R11.5 兜底, 双复跑变量隔离 = seed 不同 + ckpt 内容不同)
- **Stage 4 双复跑**: 两份 run 都完成后立即跑
- **Issue #162 close**: 双复跑完成后立即 R20+R21 v2+R16 闭环