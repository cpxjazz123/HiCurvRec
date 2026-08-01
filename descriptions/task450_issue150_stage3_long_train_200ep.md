# Task #450 / Issue #150 方向C Gate3+Gate4 long train (200 epoch Stage 3 + 双复跑 Stage 4)

## 任务背景

Owner 2026-08-01 拍板 "do long train" — 立即启动 Issue #150 Stage 3 全量 200 epoch 长训 + Gate 4 Stage 4 R@K 双复跑. Issue #150 Stage 3 10 epoch 短训已 PASS (commit 6ca0abb/379847d/b238f48). Gate 4 200 epoch Stage 3 训练 + 双复跑 R@10 > 0.1020 是 Target reached 条件.

## R18 4 维度对比 (vs task441 仅 10 epoch Stage 4 sanity)

| 维度 | task441 (短训) | task450 (本任务长训) |
|------|----------------|----------------------|
| **D1 spec** | 10 epoch Stage 3 short-train Stage 4 sanity check | 200 epoch Stage 3 full training + Gate 4 双复跑 Stage 4 R@K |
| **D2 实施** | 复用 task440 10 epoch ckpt, 推 R@10 期望 << 0.1020 | 重训 200 epoch (Task #84 anchor 等同 epoch), 推 R@10 与 0.1020 对比 |
| **D3 Gate 失败机制** | 短训 R@K 无意义, 是 sanity, 不是 decision metric | 200 epoch full train + 双复跑后 R@10 > 0.1020 = Target reached |
| **D4 引用文献** | Issue #150 spec, 10 epoch protocol | Issue #150 spec Gate 4 强制 200 epoch protocol + 双复跑 |

→ **4 维度全部不一致**, R18 实验强制 (新代码 + 新训练 + 新推断) 已规划.

## 实施 8 件套

1. `config.json` — 200 epoch Stage 3 + 双复跑 Stage 4 配置
2. `long_train_log.json` — 200 epoch 全程 cond_grad/ln_grad/loss/α logit/SID hash 监控
3. `adapter_200ep.pt` — 长训 ckpt (R12 强制, 删旧 + 存新)
4. `train_trace_200ep.json` — 200 epoch 训练 trace (per-epoch metrics)
5. `eval_run1.json` — Stage 4 R@K 双复跑第一轮 (R@5/10/20, NDCG@5/10/20)
6. `eval_run2.json` — Stage 4 R@K 双复跑第二轮 (跟 run1 验证一致性)
7. `verdict.json` — gate4_decision (PASS/FAIL) + 6 指标 + 双复跑对比
8. `log` — `logs/task450_issue150_stage3_long_train_200ep.log`

## 决策指标 (Issue #150 spec Gate 4)

- **Target reached**: test R@10 > 0.1020 (HG-Rec Task #84 baseline) + 双复跑 R@10 一致
- **GO 收口**: R@10 > 0.1020 + 双复跑差距 < 0.005
- **PARTIAL** (架构修复成功但 R@10 未达 baseline): 仍 GO 收口, 论文 §6.7.x 记录 zero-centered LN + bounded-linear 路径作为 viable 修复
- **NO-GO**: 200 epoch 训练失败 (loss 不收敛 / NaN / forward 不一致)

## 关键风险

- **GPU 时间**: 200 epoch × 4120 batch/epoch ÷ 4 batch/sec ≈ 200×1030s ≈ 5.7h (single L40S)
- **R12 ckpt 强制**: 训练结束 → 删旧 + 存新
- **Gate 3 spec 10 epoch gradient 不衰减 这条是否在 200 epoch 持续**: 可能后期 α logit 增长突破 bounded-linear 区间, 需全程监控
- **memory accumulating**: 200 epoch log 文件会很大, 必须 log rotate

## 防错机制

- **R12 删旧存新**: 训练前 ls adapter_200ep.pt, 如存在先 rm -f
- **R15 push**: 训练结束立即 commit + push (R21 v2 后回填 hash)
- **R20 4 Gate 详细**: commit message + issue comment 每个 Gate 至少 3-5 行
- **R21 v2**: commit hash 落地后立即回填, comment 不含 pending/TBD/TODO
- **R16 close**: 训练 + eval 完整跑完 + verdict 落盘 + 4 Gate 详细 comment + close issue

## 与之前 task441 关系

- task441 = 10 epoch Stage 4 sanity check (R@K 推 << 0.1020, 验证 Issue #150 路径 sanity)
- task450 = 200 epoch Stage 3 full train + Stage 4 双复跑 (Issue #150 spec Gate 4 强制)
- task450 是 Issue #150 真正决策 metric, 短训 task441 仅是 sanity engineering check
- R18 4 维度全部不一致, 不复用 task441 数据

## R11.5 自主决策

- **200 epoch 启动**: 立即 (owner 拍板 "do long train")
- **双复跑**: Issue #150 spec 强制, 立即执行
- **Target reached**: owner 拍板 R@10 > 0.1020 (跟 task84 anchor 等同)
- **PARTIAL**: 架构修复成功但 R@10 ≤ 0.1020, 写作 paper §6.7.x 收起

## R7 GPU 分配

- 启动 GPU 0 (全空闲, 46068 MiB available)
- TRITON_CACHE_DIR=~/.triton/cache_task450 (per R137 fix)
- 监控 GPU 0 util/memory 防止溢出
- 不用 GPU 1/2/3 (留给可能的新 issue)
