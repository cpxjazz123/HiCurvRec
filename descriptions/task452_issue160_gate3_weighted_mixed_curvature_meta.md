# Task #452 / Issue #160 [方向B Gate3] 加权混合曲率元数据的T5逐层几何适配

## 任务背景

Issue #160 (方向B Gate3) 2026-08-01 02:15 owner 派发. Issue #158 Gate 2 PASS (commit fa0b455). 本任务进入 Gate 3 Stage 3, 把 #158 的 [κ_l, alpha_l, beta_l, gamma_l] 三分量加权混合曲率元数据注入 T5.

## R18 4 维度对比 (vs task440 Issue #150 Gate 3)

| 维度 | task440 #150 (通用零中心残差) | task452 #160 (三分量元数据注入) |
|------|-------------------------------|--------------------------------|
| **D1 spec** | 通用 zero-centered bounded-linear residual (无三分量) | L0/L1/L2 各自 [κ_l, alpha_l, beta_l, gamma_l] 加权混合曲率 → 零中心几何残差 |
| **D2 实施** | conditioner 输入 SID 4-dim + κ 3-dim, 不区分 L0/L1/L2 | conditioner 输入 [κ_l, alpha_l, beta_l, gamma_l] per layer (12 维 per sample) |
| **D3 Gate 3 失败机制** | 静态权重的三分量折叠 (跟 #158 spec 反例) | 把三分量折叠成单一欧氏特征 (vs #158 保留三分量) |
| **D4 引用** | arXiv:2309.04082 + 内部 spec | arXiv:2307.04514 (数据驱动加权混合曲率产品空间) + #158 实施 |

→ **4 维度全部不一致**, R18 实验强制 (新代码 + 新训练 + 新验证) 必备.

## 实施目标 (Issue #160 spec)

1. **Conditioner**: per-layer [κ_l, alpha_l, beta_l, gamma_l] → 零中心混合几何残差 → 注入 SID token
2. **冻结**: T5 主干 + SID / checkpoint hash (跟 #158 / task84 anchor 一致)
3. **解冻**: 三分量元数据 conditioner + T5 input LayerNorm
4. **预检**: 残差系数=0 → max diff=0; 首步后 κ/混合权重编码路径 conditioner + LayerNorm 均有有限非零 gradient/delta
5. **训练**: 10 epoch 短训 (Gate 3 spec); 5 个连续 epoch 梯度 < 阈值 视为 FAIL
6. **save/load**: missing=0/unexpected=0, forward 一致, SID hash 跟 #158 一致

## 8 件套

1. `config.json` — config + 关键路径
2. `adapter_init_proof.json` — 残差系数=0 strict identity
3. `layernorm_unfreeze_proof.json` — 训练参数清单
4. `gradient_proof.json` — 首步后梯度 nonzero
5. `train_trace_200ep.json` — 10 epoch 训练 trace
6. `verdict.json` — gate3_decision
7. `adapter.pt` — R12 ckpt (删旧 + 存新)
8. `log` — `logs/task452_issue160_gate3_weighted_mixed_curvature_meta.log`

## Gate 4 决策

Gate 3 PASS 后才进入 Gate 4. Gate 4 = 200 epoch Stage 3 + Stage 4 R@K 双复跑 (跟 Issue #150 / #159 同 spec).
- **Gate 4 启动**: 等 owner 拍板 (R11.5 critical: 200 epoch GPU 训练 + 几小时)
- **Task #450 Issue #150 已经在跑 200 epoch long train**, Issue #159 / #160 走同轨迹 (10 epoch Gate 3 → 等 owner 拍板 200 epoch Gate 4)

## 防错机制

- **R12 ckpt 强制**: 训练结束 → 删旧 adapter.pt → 存新
- **R15 push**: 训练结束立即 commit + push
- **R20 4 Gate 详细**: commit message + Issue #160 comment 每个 Gate ≥3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: 写完 4 Gate 详细 comment + close issue with --reason completed
- **R7 GPU**: 启动 GPU 2 (GPU 0 被 Task #450 占用, GPU 1 被 Task #451 占用)
- **R137**: TRITON_CACHE_DIR=~/.triton/cache_task452

## Code 复用

- 复用 task440 wrapper + train loop 框架
- 不同点: conditioner 输入 [κ_l, alpha_l, beta_l, gamma_l] 12 维 per layer
- 复用 task440 EOFs: load_t5_state_dict, get_t5_config, verify_sid_token_range
- 复用 task84 anchor: SID_NPY, T5_CKPT, test.parquet (SHA 同 task451)

## R11.5 自主决策

- **10 epoch Gate 3 启动**: 立即 (Issue #160 spec 强制 10 epoch, 短训)
- **R@K Stage 4 (200 epoch)**: 等 owner 拍板 (R11.4 critical)
- **GPU 2**: 启动 (Task #450 占用 GPU 0, Task #451 占用 GPU 1)
- **并行 Issue #159 GPU 1**: 立即 (R19 跨 issue 并行)
