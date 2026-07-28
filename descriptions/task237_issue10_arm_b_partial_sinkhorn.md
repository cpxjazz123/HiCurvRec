# Task #237 — Issue #10 Gate 1 Arm B: partial Sinkhorn (3-arm 因果曲线)

## 来源
- GitHub Issue #10 §Experimental design (2026-07-28): Arm B = vanilla + 部分 Sinkhorn
- Issue #10 Gate 0 (Task #236) 已 PASS: 锁定 collision_rate = 1 - uniqueness 权威定义
- Issue #10 Gate 1 = 唯一新增的 Stage 3 训练 (~44min)

## 背景
| Arm | Stage 2 Sinkhorn | collision | uniqueness | Stage 4 R@10 | 来源 |
|-----|------------------|-----------|------------|--------------|------|
| A (vanilla) | max_iters=0 | 0.99 | 1% | 0.1020 | task223 / task225 §5 |
| **B (Arm B, 本任务)** | **max_iters=10** | **TBD** | **TBD** | **TBD** | **新跑** |
| C (vanilla + Sinkhorn) | max_iters=30 | 0.05 | 95% | 0.1058 | task225 §5 / phonism |

## 配置
| 项 | 值 |
|----|-----|
| Stage 1 ckpt | task84 baseline best_loss_model.pth (vanilla, 不重训) |
| Stage 2 sinkhorn | max_iters=10 (vs Arm A=0, vs Arm C=30) |
| Stage 3 | T5-mini 9.18M, 200 epoch / early_stop=20 (跟 task233 一致) |
| Stage 4 | task233 eval template (R12 强制 ckpt + heartbeat) |
| GPU | 0 (4 卡全空闲) |
| Dataset | Musical_Instruments (5-core, 9922 items) |
| Seed | 42 |

## 决策阈值 (Issue #10)
| Gate | 指标 | 阈值 | 含义 |
|------|------|------|------|
| Gate 1 | B vs A collision 差 | ≥ 15pp | 三个不同水平 |
| Gate 1 | B vs C collision 差 | ≥ 15pp | 三个不同水平 |
| Gate 1 | util vs collision 变动 | util ≤ collision (否则 confounded) | Sinkhorn 单变量 |
| Gate 2 | 单调下降 (C ≥ B ≥ A) | collision 反向 → 有效杠杆 | `<=12%` bar 保留 |
| Gate 2 | B < A (U 形) | collision 不是杠杆 | Issue #9 Gate 1 collision ≤ 0.3706 降级为诊断量 |
| Gate 2 | B > C | Stage 2 后处理即可超最优 | bar 上移到 B |

## 步骤
1. ✅ 写 launcher `scripts/task237_issue10_arm_b_partial_sinkhorn.sh`
2. ⏳ 启动 Stage 2 partial Sinkhorn (max_iters=10) on GPU 0
3. ⏳ Stage 3 T5-mini 200 epoch early_stop (R12 ckpt + heartbeat)
4. ⏳ Stage 4 eval + Task #209 slice
5. ⏳ 写 verdict, 三臂表每个 recall 数字旁写明 epoch + collision (统一口径)
6. ⏳ commit + Issue #10 comment + close

## 产物
- products/task237/hrqvae_partial_sinkhorn/best_collision_model.pth (Stage 1 用 #84)
- products/task237/Instruments_t5_rqvae_task237_armB.npy (Stage 2 SID)
- products/task237/t5mini_armB_rerun/.../HG_Rec_best.pth (Stage 3 R12 ckpt)
- products/task237/phase3_slice_eval.json (Stage 4 slice)
- verdicts/task237_issue10_arm_b_result.md
- scripts/task237_issue10_arm_b_partial_sinkhorn.sh

## 依赖
- 依赖 verdicts/task236_collision_metric_unification_result.md (Gate 0 PASS)
- 跟 verdicts/task233_dual_v5_stage3_rerun_validation_result.md 共享 Stage 3 R12 协议
- 跟 verdicts/task209_phase3_slice_result.md 共享 slice protocol

## Status
Plan 文档 + launcher 已写. 启动执行需 ~50min wall-clock, 跨多个 loop tick. 等 R10 推进到 Issue #10 Gate 1 启动.
