# Task #233 — Issue #8 Validation: dual_v5 Stage 3 rerun (collision-vs-budget confound)

## 来源
- GitHub Issue #8 (2026-07-28): [Validation] Collision-to-R@10 effect is unidentified: dual_v5 -10.3% is confounded by Stage 3 truncation (ep 93/200)
- Issue #8 质疑 task200 dual_v5 R@10=0.0915 (-10.3%) 因果链: Stage 3 训练静默死亡于 epoch 93/200 (46.5%), 评估用的是中间 ckpt
- task209 Phase 3 slice 显示 checkpoint immaturity 单独造成 Head R@10 -31.6% (0.1794 → 0.1228), 3x magnitude 大于 -10.3%

## 背景
| 事件 | 时间 | 备注 |
|------|------|------|
| task200 dual_v5 Stage 3 训练 | 2026-07-26 02:51 | silent death at ep 93/200 |
| task200 Stage 4 eval | 2026-07-26 03:10 | R@10=0.0915 (用 ep93 mid ckpt) |
| Issue #8 创建 | 2026-07-28 13:03 | 质疑 -10.3% 因果链 |

## 核心想法
Issue #8 提议 Phase 1: 重跑 Stage 3 用 same dual_v5 SID, 跑 200 epoch (or early stop), 加 epoch-level logging + heartbeat 防 silent death 重现. Stage 4 eval + Task #209 slice.

## 配置
| 项 | 值 |
|----|-----|
| SID 输入 | products/task200/dual_arm_C_v5/.../best_collision_model.pth (ep14, collision 0.8387) |
| T5 model | t5-mini 9.18M (跟 task144 一致) |
| Stage 3 配置 | 200 epoch / early_stop=20 (跟 task144 + task200 一致) |
| 改进 | epoch-level logging + heartbeat print per 50 step + early_stop counter |
| GPU | 0 (4 GPU 全空闲) |
| Dataset | Musical_Instruments (5-core, 9922 items) |
| Seed | 42 |

## 决策阈值 (Issue #8 提议)
| Stage 4 R@10 | 含义 | 后续 |
|--------------|------|------|
| >= 0.1020 | -10.3% 是 truncation artifact, 撤销 collision sensitivity claim | Issue #6/#7 <=12% bar 保留 |
| <= 0.0950 | collision sensitivity 真实 | <=12% bar 需重新校准, Issue #6/#7 Phase 1 暂停 |
| 0.0950 < R@10 < 0.1020 | partial | 报告两效应, 不关闭任一方向 |

## 产物
- products/task233/dual_v5_stage3_rerun/best_loss_model.pth (R12: 强制存)
- products/task233/dual_v5_stage3_rerun/_TRAINING_PID (R12.2)
- logs/task233/stage3_rerun.out
- verdicts/task233_dual_v5_stage3_rerun_validation_result.md (含 budget-vs-R@10 table)

## 步骤 (Issue #8 §Procedure)
1. 加 epoch-level logging (early_stop counter + heartbeat per 50 step) — 修 task144 Stage 3 模板
2. 重跑 Stage 3 on same dual_v5 SID, 200 epoch / early_stop=20
3. Stage 4 eval via task144_stage4_eval.py template
4. Task #209 Head/Body/Tail slice on completed dual_v5 ckpt
5. 写 verdict with budget-vs-R@10 table

## 依赖
- 直接 follow verdicts/task200_dual_v5_stage3_4_result.md (Issue #8 引用源)
- 跟 verdicts/task209_phase3_slice_result.md 共享 slice protocol
- 跟 Issue #6 (Task #231) + Issue #7 (Task #232) 共享 <=12% collision bar 解释

## Status
Not yet run. Issue #8 Phase 1 提议. Phase 2 (3-arm converged collision design) R11.4 关键决策, 等 Phase 1 结果再决定.
