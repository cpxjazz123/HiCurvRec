# Task #453 / Issue #161 [方向C Gate4] 第二份独立 200 epoch Stage 3 run (seed=43, GPU 1) + 双复跑六项指标

## 任务背景

Issue #161 (方向C Gate4) 2026-08-01 owner 派发. Issue spec 强制:
- 外部产生**两次独立正式run**, 记录 seed / 配置 / ckpt SHA256 / 训练 eval 日志 / commit
- 对 Task #84 HG-Rec baseline 同一 Musical_Instruments 测试协议报告完整六项指标: R@5/10/20 + NDCG@5/10/20 (baseline R@10=0.1020)
- 真实 test R@10 > 0.1020 且六项指标齐全 = `[TARGET REACHED]`; 否则 Gate 4 FAIL/PENDING
- 不得更换 SID / 数据切分 / evaluator / baseline; 不得使用 10 epoch sanity 结果冒充最终指标

## R18 4 维度对比 (vs Task #450 Issue #150 seed=42 run 1)

| 维度 | Task #450 (run 1, seed=42, GPU 0) | Task #453 (run 2, seed=43, GPU 1) |
|------|-----------------------------------|-----------------------------------|
| **D1 spec** | seed=42, GPU 0, 200 epoch full Stage 3 | seed=43, GPU 1, 200 epoch full Stage 3 (独立初始化) |
| **D2 实施** | adapter 初始化 = seed 42 torch.manual_seed + nn init | adapter 初始化 = seed 43 torch.manual_seed + nn init (不同初始权重) |
| **D3 Gate 4 决策** | 双复跑 run 1, Stage 4 eval 后产出 eval_run1.json | 双复跑 run 2, Stage 4 eval 后产出 eval_run2.json |
| **D4 引用** | Issue #161 spec Gate 4 = 两份独立 run | Issue #161 spec Gate 4 = 两份独立 run |

→ **4 维度全部不一致 (D2: 不同 seed 初始化 → 不同 ckpt 内容)**, R18 实验强制 (新代码 + 新训练 + 新验证) 必备.

## 实施目标 (Issue #161 spec Gate 4)

1. **种子独立性**: seed=43 (vs Task #450 seed=42), torch.manual_seed + numpy.random.seed + adapter nn.init 都用 seed 43
2. **GPU 隔离**: GPU 1 (Task #450 占 GPU 0, Issue #161 spec 要求"独立" run)
3. **R137 TRITON_CACHE_DIR**: per-task 隔离 (`/home/wlia0047/.triton/cache_task453`)
4. **Stage 3 训练**: 200 epoch full (跟 Task #84 anchor 等同 epoch)
5. **Stage 4 R@K eval**: 200 epoch 训练完立即跑双复跑 (run 2 单跑 + 跟 Task #450 run 1 对比)
6. **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新 ckpt, SHA256 跟 Task #450 run 1 完全不同 (种子差异)

## 关键路径

- 复用 Task #450 锚定信息: codebook_size=[64,128,256,1], max_len=20, batch_size=32, d_model=128, lr_conditioner=1e-3, lr_layernorm=1e-4, alpha_init=0.0
- 复用 Task #440 wrapper: `HG_Rec_with_ZeroCenteredLayerNormAdapter`
- 锚定 Task #84 ckpt: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth` (SHA256 56d046db...)
- 锚定 Musical_Instruments 数据集: SID_NPY / TRAIN_PARQUET / TEST_PARQUET (SHA 同 Task #450)

## 8 件套 + Stage 4 双复跑产物

1. `config.json` — seed=43, GPU 1, 配置 + 关键路径
2. `adapter_init_proof.json` — α=0 → max diff=0 strict identity
3. `layernorm_unfreeze_proof.json` — 训练参数清单
4. `gradient_proof.json` — 首步后 conditioner + LN 梯度 nonzero
5. `train_trace_200ep.json` — 200 epoch 训练 trace (epoch / loss / cond_grad / ln_grad / α)
6. `verdict.json` — gate3_decision PASS
7. `adapter_200ep.pt` — R12 ckpt (删旧 + 存新)
8. `log` — `logs/task453_issue161_gate4_run2_seed43.log`
9. `eval_run2.json` — Stage 4 R@K eval 完整六项 (R@5/10/20 + NDCG@5/10/20)
10. `stage4_verdict.json` — 双复跑 GO/NO-GO 决策 (跟 Task #450 run 1 对比)

## 防错机制

- **R7 GPU**: 启动 GPU 1 (Task #450 占 GPU 0), GPU 2/3 空闲
- **R12 ckpt**: 训练结束 → 删旧 adapter_200ep.pt → 存新
- **R15 push**: 训练结束立即 commit + push
- **R20 4 Gate 详细**: commit message + Issue #161 comment 每个 Gate ≥3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: 写完 4 Gate 详细 comment + close issue with --reason completed
- **R137**: TRITON_CACHE_DIR=~/.triton/cache_task453

## Gate 4 决策阈值

- **GO**: 双复跑 (run 1 + run 2) 平均 R@10 > 0.1020 + 六项指标齐全
- **NO-GO**: 任一 run R@10 ≤ 0.1020 或六项指标缺失

## Code 复用

- 复制 Task #450 脚本 + 改 SEED=43 + DEVICE=cuda:1 + PRODUCT_DIR=task453 + LOG_PATH=task453
- 复用 Task #440 wrapper 直接 import (`HG_Rec_with_ZeroCenteredLayerNormAdapter`)
- Stage 4 eval 复用 Task #441 / Task #450 Stage 4 eval 模式 (`GenRecDataset mode='evaluation'`)

## R11.5 自主决策

- **第二份 run 启动**: 立即 (Issue #161 spec 强制两份独立 run, Task #450 run 1 已经在跑)
- **GPU 1**: 启动 (Task #450 占用 GPU 0, R7 严禁抢卡)
- **Stage 4 双复跑**: 两份 run 都完成后立即跑 (跟 Task #450 同步)
- **Issue #161 close**: 双复跑完成后立即 R20+R21 v2+R16 闭环