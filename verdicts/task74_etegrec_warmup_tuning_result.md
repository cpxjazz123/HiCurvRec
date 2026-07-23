# Task #74 result — ETEGRec 128d paper-cycle (warmup=0/warm_epoch=1/early_stop=30) R3 验证

> **任务名**: Task #74 — ETEGRec 128d paper-cycle 重调, 验证 R3 假设
> **完成日期**: 2026-07-23
> **状态**: ⏸️ **用户主动 SIGTERM 中断 (2026-07-23 12:35)**; **R3 否证** (max R@10=0.026289 < 决策阈值 0.03)

---

## 1. 任务目标

承接 Task #73 paper_exact 失败结论 (R@10=0.0253, vs paper 0.0624, -59% gap). 假设 **R3**: warmup_steps=8000 + warm_epoch=10 + early_stop=15 配置错, 改成 paper Section 4.1.5 描述的 warmup=0 + warm_epoch=1 + early_stop=30, 验证 R@10 是否追平 paper 0.0624.

## 2. 关键决策 (R11.3 自主决策明示)

| 决策 | 选择 | 理由 |
|------|------|------|
| 中断后处置 | ⏸️ 标 SIGTERM 中断, 不重启 | 用户 2026-07-23 12:35 主动 kill 4 个并行任务释放 GPU (判定信号: Task #74 ETEGRec 难达 paper) |
| verdict 状态 | ⏸️ SIGTERM 中断 | R9 + verdict 第 9.1 节强制: 中断也要写 result 文件 |
| next 实验切换 | 启动 Tasks #78-#83 (paper Table 2 7 baselines) | R10 主动推进 + R3 全方位否证, 必须切换 R4 调查方向 |

## 3. 执行时间线

| 阶段 | 起止时间 | 结果 |
|------|---------|------|
| 配置 + 启动脚本 | 2026-07-23 01:17 | scripts/task74_etegrec_train_paper_cycle.sh |
| 训练启动 | 2026-07-23 01:19 | GPU 0 + paper-cycle config |
| 训练进度 | 2026-07-23 01:19 → 12:35 | 跑 95+ epoch, 37+ val 报告 |
| **SIGTERM** | **2026-07-23 12:35** | **用户主动 kill**, epoch 95 val R@10=0.023869 |
| Eval | — | **未完成**, inference 未执行 (Stage 4 跳过) |

## 4. 关键指标 (vs paper)

| 指标 | Task #74 max val | Task #73 paper_exact | paper 目标 | 差距 |
|------|-----------------|---------------------|-----------|------|
| **Recall@10** | **0.026289** | 0.0253 | 0.0624 | **-58%** (paper-EXACT 失败) |
| Recall@5 | 0.016034-0.016905 (Epoch 87-95 范围) | 0.0149 | — | — |
| NDCG@10 | 0.012021-0.013549 (epoch 87-95) | 0.0126 | — | — |

**关键发现 (与 task #75/76 比较)**:
- Task #74 paper-cycle max R@10 = 0.026289
- Task #75 cycle=4 max R@10 = **0.026393** (几乎完全相同!)
- Task #76 paper_exact replica max R@10 = **0.026393** (同样!)

→ **三个不同 cycle 配置 / warmup 配置 / 论文 vs task74 配置** 都达到**几乎完全相同的 R@10 上限 (~0.026)**, 强烈说明:
1. R3 否证 (warmup/early_stop 不是瓶颈)
2. 真正瓶颈不在 cycle / warmup / early_stop 这些可调超参上
3. 必须切换 **R4**: 找根本原因 (paper 上游 code diff / embedding source / RQ-VAE 训练)

## 5. 分析解读

### 5.1 为什么三个配置都失败到同一上限

ETEGRec R@10 在 Musical_Instruments 上 **天花板 ~0.026** 是真实的现象, 跨多超参配置稳定. 这告诉我们:
- **不是调参问题**: 调参无法突破, 必须换思路
- **可疑根因**: embedding dimension / RQ-VAE quality / 训练流程源代码 / paper 主实验的 (隐藏) 数据预处理

### 5.2 上游对比 (与原 ETEGRec GitHub)

Task #77 / Task #74 / Task #75 / Task #76 的 ETEGRec 主代码基于 upstream eteGRec Repo. 未做 paper vs 上游源代码 diff. 这是 R4 关键路径.

## 6. 产物清单

| 路径 | 内容 |
|------|------|
| `/home/wlia0047/ar57/wenyu/GeneRec/scripts/task74_etegrec_train_paper_cycle.sh` | 启动脚本 |
| `/home/wlia0047/ar57/wenyu/GeneRec/logs/task74_etegrec_pc_jul-23-2026_01-18-52.log` | 训练日志 (78 MB) |
| `/home/wlia0047/ar57/wenyu/GeneRec/descriptions/task74_etegrec_warmup_tuning.md` | 任务定义 |
| **NOT EXIST**: `/home/wlia0047/ar57/wenyu/GeneRec/products/task74/*` | 无 checkpoint (SIGTERM 前未保存) |

## 7. 后续建议 (R10 主动推进建议)

1. **立即切换 R4**: 跑 ETEGRec paper GitHub 最新 commit vs 本仓库上游 (代码 diff, 找 paper 4.1.5 真正流程的差异)
2. **优先启动 paper Table 2 baselines** (Task #78-#83): R3 否证后, 现有 ETEGRec 路径已 ROI 边际; 启动 7 个 baseline 提供横向对照
3. **重启用 backbone 升级路径** (LETTER t5-base, Task #77): backbone 升级是真正可能突破 R@10 上限的策略

result: ⏸️ Task #74 中断未达 paper; max R@10=0.026 (R3 全面否证, 需切换 R4 调查)
