# Task #451 / Issue #159 [方向A Gate3] κ同步尺度元数据的T5零中心几何适配

## 任务背景

Issue #159 (方向A Gate3) 2026-08-01 02:15 owner 派发. Issue #157 Gate 2 PASS (commit 206ebb5). 本任务进入 Gate 3 Stage 3, 把 #157 的 κ 同步重校准元数据 (L0/L1/L2 κ + 同步代码本尺度) 注入 T5. 

## R18 4 维度对比 (vs task440 Issue #150 Gate 3)

| 维度 | task440 #150 (通用零中心残差) | task451 #159 (κ尺度元数据注入) |
|------|-------------------------------|--------------------------------|
| **D1 spec** | 通用 zero-centered bounded-linear residual (无 κ 输入) | 每层 κ + 同步代码本尺度元数据 → 有界零中心残差注入 SID token |
| **D2 实施** | conditioner 输入 SID 4-dim + κ 3-dim, 不区分 L0/L1/L2 | conditioner 输入 [κ_l, scale_l, sid_meta] per layer |
| **D3 Gate 3 失败机制** | sigmoid 饱和 + 全 T5 冻结 (Issue #147 反例) | 不显式注入 κ 元数据 (Stage 3 看不到 κ 状态) |
| **D4 引用** | arXiv:2309.04082 + 内部 spec | arXiv:2405.13979 (曲率感知同步元数据) + #150 接口参照 |

→ **4 维度全部不一致**, R18 实验强制 (新代码 + 新训练 + 新验证) 必备.

## 实施目标 (Issue #159 spec)

1. **Conditioner**: κ-scale 元数据 → 有界零中心残差 → 注入 SID token 表示
2. **冻结**: T5 主干 + SID/checkpoint hash (跟 #157 / task84 anchor 一致)
3. **解冻**: κ-scale conditioner + T5 input LayerNorm
4. **预检**: 残差系数=0 → max diff=0; 首步后有限非零 gradient
5. **训练**: 10 epoch 短训 (Gate 3 spec); 5 个连续 epoch 梯度 < 阈值 视为 FAIL
6. **save/load**: missing=0/unexpected=0, forward 一致, SID hash 跟 #157 一致

## 8 件套

1. `config.json` — config + 关键路径
2. `adapter_init_proof.json` — 残差系数=0 strict identity
3. `layernorm_unfreeze_proof.json` — 训练参数清单
4. `gradient_proof.json` — 首步后梯度 nonzero
5. `train_trace_200ep.json` — 10 epoch 训练 trace
6. `verdict.json` — gate3_decision
7. `adapter.pt` — R12 ckpt (删旧 + 存新)
8. `log` — `logs/task451_issue159_gate3_kappa_scale_meta.log`

## Gate 4 决策

Gate 3 PASS 后才进入 Gate 4. Gate 4 = 200 epoch Stage 3 + Stage 4 R@K 双复跑 (跟 Issue #150 同 spec). 
- **Gate 4 启动**: 等 owner 拍板 (R11.5 critical: 200 epoch GPU 训练 + 几小时)
- **Task #450 Issue #150 已经在跑 200 epoch long train**, Issue #159 / #160 走同轨迹 (10 epoch Gate 3 → 等 owner 拍板 200 epoch Gate 4)

## 防错机制

- **R12 ckpt 强制**: 训练结束 → 删旧 adapter.pt → 存新
- **R15 push**: 训练结束立即 commit + push
- **R20 4 Gate 详细**: commit message + Issue #159 comment 每个 Gate ≥3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: 写完 4 Gate 详细 comment + close issue with --reason completed
- **R7 GPU**: 启动 GPU 1 (GPU 0 被 Task #450 占用), GPU 1/2/3 空闲
- **R137**: TRITON_CACHE_DIR=~/.triton/cache_task451

## Code 复用

- 复用 task440 wrapper + train loop 框架 (跟 #150 同)
- 不同点: 新增 κ + scale 元数据输入 conditioner
- 复用 task440 EOFs: load_t5_state_dict, get_t5_config, verify_sid_token_range
- 复用 task84 anchor: SID_NPY (SHA256 2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a), T5_CKPT (SHA256 56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e), test.parquet (SHA256 70f00f3a3ac4cf39088e4ba7c516104ecf1d6fddf14b19e255c8f89229e55899)

## R11.5 自主决策

- **10 epoch Gate 3 启动**: 立即 (Issue #159 spec 强制 10 epoch, 短训)
- **R@K Stage 4 (200 epoch)**: 等 owner 拍板 (R11.4 critical)
- **GPU 1**: 启动 (Task #450 占用 GPU 0, R7 严禁抢卡)
- **并行 Issue #160 GPU 2**: 立即 (R19 跨 issue 并行)
