---
type: status
status: "FAIL"
created: 2026-08-02
tags:
  - loop-tick
up: "[[index]]"
---
# Idle Status — 2026-08-02 Tick

**触发**: R10 v2 (0 open issue + §16 空 + 无 owner 派工) + R16 (loop 定时 tick)
**Git**: 05d81c6 (R28/R29) → 59299ca (Issue #4 close) → 2cec35f (Issue #5 close) → d6bbdcc (housekeeping)
**Open issues**: 0
**GPU**: 4× L40S 全部空闲 (util 0%, mem 0 MiB)

## Stale _TRAINING_PID 清单 (历史训练已终止, PID 文件残留)

| 路径 | PID | 状态 |
|---|---|---|
| taskB/stage3/taskB_stage3_mixed_curv_recontinue/_TRAINING_PID | 426717 | STALE (no process) |
| taskA/stage3/taskA_stage3_kappa_scale_recontinue/_TRAINING_PID | 426245 | STALE (no process) |
| taskB/_history/task469_issue176_stage2_mixing_kappa_vq_loss_forward_path_fix/_TRAINING_PID | 5067 | STALE (no process) |
| taskA/_history/task468_issue175_stage2_kappa_vq_loss_forward_path_fix/_TRAINING_PID | 4165386 | STALE (no process) |
| taskA/_history/task175/_TRAINING_PID | 292001 | STALE (no process) |
| taskA/_history/task174/_TRAINING_PID | 214618 | STALE (no process) |

**评估**: 不影响当前 idle 状态, 但 _history/ 下的 PID 文件历史归档层 (housekeeping 已删除前几轮, 但 task174/175/468/469 还在). 建议下轮 housekeeping 收尾时一并清理.

## 4 Gate 状态 (R17)

| Gate | 状态 | 来源 |
|---|---|---|
| Gate 1 RQ-VAE | ✅ PASS | gate1_evidence.json (继承, Issue #4/#5 期间未变) |
| Gate 2 per-layer | ✅ PASS | 继承 |
| Gate 3 协议+early stop | ✅ PASS | Issue #4/#5 本轮补齐 |
| Gate 4 R@K | ⚠️ PARTIAL → FAIL | Issue #4 R@10=0.0389, Issue #5 R@10=0.0395, baseline 0.1020 (R@10 数字是 paper §6.7.9 结论, 评估函数已可审计) |

## 待 owner 派工候选 (3 个, per R11.5)

1. **Step 5 单 seed Task84 全 eval** (Issue #4/#5 闭环后) — 用修复后函数跑全 test set 验证 R@10=0.0389 数学上一致
2. **Issue #30 + #43 联合 (Task #344)** — 6h Stage 1 重训, 期望 +2.5-3% (paper §6.7.9 ROI 评估)
3. **T5-mini → T5-small 容量扩展** — 4-5h 训练, 期望 +1-2%
4. **接受 R@10=0.10425 ceiling, 转写 paper 收尾** (per verdicts/index.md §G 唯一突破 #43 HypPre)

## 下个 tick 触发

cron job 8c3722eb (5m, session-only, 7d auto-expire)
