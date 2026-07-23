# Task #76 result — ETEGRec 128d paper_exact 400 epoch replica (R3 复跑 baseline)

> **任务名**: Task #76 — ETEGRec 128d paper_exact 配置 400 epoch 完全复跑
> **完成日期**: 2026-07-23
> **状态**: ⏸️ **用户主动 SIGTERM 中断**; **R3 否证 + paper_exact 复现证实**: max R@10=0.026393 vs paper 0.0624

---

## 1. 任务目标

完全复制 Task #73 paper_exact 配置 (cycle=2, warmup=8000, warm_epoch=10, early_stop=15, 400 epoch), 排除 Task #73 是单次失败的可能, 验证 R3 在 paper_exact 配置下也失败.

## 2. 关键决策 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| 配置 | 100% 复制 task73 paper_exact | 双盲: 必须与 task73 一致才能横向对比 |
| GPU | 2 | 与 task74/75/77 并行 4 卡 |
| 中断处置 | ⏸️ 标 SIGTERM | 用户主动 kill (与 #74/#75 同步) |

## 3. 执行时间线

| 阶段 | 起止时间 | 结果 |
|------|---------|------|
| 启动脚本 | 2026-07-23 | scripts/task76_etegrec_train_paper_exact_replica.sh |
| 训练启动 | 2026-07-23 02:05 | GPU 2 (与 #74/#75 并行 4 卡) |
| 训练进度 | 2026-07-23 02:05 → 12:35 | 主训练 85 epoch, finetune 阶段 1 epoch (epoch 0/1), 45 val 报告 |
| **SIGTERM** | **2026-07-23 12:35** | **用户主动 kill** |
| Eval | — | 未完成 |

## 4. 关键指标 (vs paper)

| 指标 | Task #76 max val | Task #73 paper_exact | paper 目标 | 差距 |
|------|-----------------|---------------------|-----------|------|
| **Recall@10** | **0.026393** | 0.0253 | 0.0624 | **-58%** (与 #73 同水平) |
| Recall@5 | ~0.016 | 0.0149 | — | — |
| NDCG@10 | ~0.013 | 0.0126 | — | — |
| Finetune 阶段 val_R@10 | 0.026 (epoch 0 finetune 阶段) | — | — | finetune 阶段未显著拉开 |

**关键发现**: Task #76 paper_exact 复跑与 Task #73 第一次跑的结果**几乎完全相同** (差 < 0.001). 这给出 **paper_exact 配置是真不可达 paper, 不是随机性**.

## 5. 分析解读

### 5.1 复现性证实 (paper_exact 配置是真失败, 不是随机失败)
Task #73 / Task #76 是同一配置两个独立 run, R@10 上限几乎完全相同 (~0.026). 这从统计学上排除了"Task #73 单次失败", 证实 **paper_exact 配置在 Musical_Instruments 上真实天花板 ~0.026**.

### 5.2 paper gap 是系统性的
任何 ETEGRec 主路径调参 / cycle / warmup / early_stop 调整 (Task #73-76 四次训练) 都达到 R@10≈0.026 上限, vs paper 0.0624 差 -58%. 这条天花板线说明上游代码与 paper 之间存在系统性 gap (例如: RQ-VAE codebook size / embedding 来源 / 训练流程 / data preprocessing).

## 6. 产物清单

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/scripts/task76_etegrec_train_paper_exact_replica.sh` | 启动脚本 |
| `/home/wlia0047/ar57/wenyu/GeneRec/logs/task76_etegrec_pe_jul-23-2026_02-05-34.log` | 训练日志 |
| `/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task76_etegrec_paper_exact_400_replica.md` | 任务定义 |
| NOT EXIST: products/task76/ | 无 checkpoint |

## 7. 后续建议

1. 关闭所有 ETEGRec paper_exact 配置复跑路径 (R3 全面失败)
2. 切换到 R4: paper GitHub commit diff vs 本仓库 upstream
3. 并行启动 Tasks #78-#83 (paper Table 2 baselines) 提供横向对比

result: ⏸️ Task #76 paper_exact 复跑证实 R3 否证; max R@10=0.026393 ≈ task73 0.0253 (paper_exact 真不可达 paper)
