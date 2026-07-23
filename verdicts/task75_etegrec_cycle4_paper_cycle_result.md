# Task #75 result — ETEGRec 128d cycle=4 paper-cycle (R3 备选 cycle 验证)

> **任务名**: Task #75 — ETEGRec 128d cycle=4 paper-cycle 备选周期验证
> **完成日期**: 2026-07-23
> **状态**: ⏸️ **用户主动 SIGTERM 中断**; **R3 cycle=4 也否证** (max R@10=0.026393 vs task74 0.026289, 几乎完全相同)

---

## 1. 任务目标

承接 Task #74 R3 否证期间, 并行验证 paper Section 4.1.5 备选 cycle=4 (vs cycle=2). 假设: cycle 越大, ID/REC 切换频率越低, 训练稳定性越好, R@10 越高.

## 2. 关键决策 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据集/emb/code_length | 与 #74 完全相同 | 仅变量: cycle=2 → cycle=4 |
| early_stop/warmup | warmup=0/warm_epoch=1/early_stop=30 (paper-cycle) | 与 #74 一致 |
| lr_rec/lr_id | 0.005/0.0001 (与 #74 一致) | 排除 LR 干扰 |

## 3. 执行时间线

| 阶段 | 起止时间 | 结果 |
|------|---------|------|
| 启动脚本创建 | 2026-07-23 (parallel to #74) | scripts/task75_etegrec_train_cycle4.sh |
| 训练启动 | 2026-07-23 02:05 | GPU 1 (与 #74 GPU 0 并行) |
| 训练进度 | 2026-07-23 02:05 → 12:35 | 跑 73 epoch, 37+ val 报告 |
| **SIGTERM** | **2026-07-23 12:35** | **用户主动 kill**, epoch 73 val R@10=0.020161 |
| Eval | — | 未完成, inference 跳过 |

## 4. 关键指标 (vs paper)

| 指标 | Task #75 max val | Task #74 max val | paper 目标 | 差距 |
|------|-----------------|-----------------|-----------|------|
| **Recall@10** | **0.026393** | 0.026289 | 0.0624 | **-58%** |
| Recall@5 | 0.016191-0.018 (epoch 65-71 范围) | 0.016034-0.016905 | — | — |
| NDCG@10 | 0.012042-0.013964 | 0.012021-0.013549 | — | — |

**关键发现**: Task #75 cycle=4 与 Task #74 cycle=2 的 R@10 上限**几乎完全相同** (差 < 0.0001), 强烈排除 cycle 维度作为根因. 真实瓶颈仍在别处.

## 5. 分析解读

### 5.1 cycle 不是瓶颈
两个 cycle 配置达到几乎完全相同的 R@10 上限 (~0.0264), 任何 cycle={2,4} 都不是瓶颈来源.

### 5.2 与 Task #73/74/76 一致
四个 ETEGRec 训练实验 (task73 paper_exact, task74 paper-cycle, task75 cycle=4, task76 paper_exact replica) 都卡在 R@10≈0.026. 这条天花板线说明 ETEGRec upstream 代码与 paper 报告之间存在系统性 gap, 单纯调参无法突破.

## 6. 产物清单

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/scripts/task75_etegrec_train_cycle4.sh` | 启动脚本 |
| `/home/wlia0047/ar57/wenyu/GeneRec/logs/task75_etegrec_c4_jul-23-2026_02-05-33.log` | 训练日志 |
| `/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task75_etegrec_cycle4_paper_cycle.md` | 任务定义 |
| NOT EXIST: products/task75/ | 无 checkpoint (SIGTERM 前未保存) |

## 7. 后续建议 (与 #74 同步)

1. R4 调查方向: paper GitHub 最新 commit diff vs 本仓库 upstream
2. 优先 backbone 升级路径 (LETTER t5-base) 已证明可行 (Task #77 LETTER-TIGER R@10=0.0997)
3. 关闭所有 ETEGRec 当前路径, 切换到 paper Table 2 baseline 横向对比 (Tasks #78-#83)

result: ⏸️ Task #75 cycle=4 中断未达 paper; max R@10=0.026393 (≈ task74 cycle=2, R3 cycle 维度全面否证)
