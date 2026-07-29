# Task #283 / D5 — dead_revive frequency 验证 L0 ≥ 90% (Task #282 NO-GO 推论)

## 背景

Task #282 (task270 A1 欧氏 MSE+β=0 NO-GO) 揭示:
- β 不是 L0 ≥ 90% 杠杆 (commit loss 是稳定剂非天花板)
- β=0 → 1.6% mode collapse, β=0.5 → 73.44% (Issue #17 闭环锚点 task253)
- A2 (curriculum β) 前提崩塌

L0 ≥ 90% 需要**单 β 之外**的机制. 候选:
- dead_revive frequency (anti_collapse hook, hrqvae_trainer.py 行 264)
- Sinkhorn curriculum (issue #10 closed, 0pp 分离失效)
- kmeans 重 init (Phase 0 fix 引入 mode collapse, 反向)

本任务 (D5) 验证 dead_revive frequency. Phase 0B (用户 2026-07-27) 已实现 dead_revive 在 evaluate 阶段跑. 已知:
- task263 verifier 测 post-revive L0=100% (Issue #17 Gate 1)
- task253 best_collision pre-revive L0=73.44% (Issue #17 锁定量)
- 即 pre-revive 是真实 L0 utilization, post-revive 是 "被人工救活" 的 utilization

**关键未知**: step2 monitor 打印的 utilization 是 pre-revive 还是 post-revive. 如是 pre-revive, dead_revive frequency (eval_step) 几乎不影响 step2 monitor 数字; 如是 post-revive, eval_step=1 应该把 L0 拉到 100%.

## 任务范围

### 单变量: `eval_step`
- D5-A: eval_step=5 (默认, 与 task253 baseline 同)
- D5-B: eval_step=1 (高频, 每个 epoch revive)
- D5-C: eval_step=10 (低频, 对照)

### 受控因素
- 不改 num_emb_list=[64,128,256], e_dim=36, angular_dim=4, radial_dim=32
- product_manifold=True, kmeans_init=True, seed=42
- loss_type=poincare, beta=0.5
- Musical_Instruments 5-core (9922 items / 24772 test examples)
- anti_collapse='dead_revive', 50 epoch

### 通过条件 (任一 PASS)
- pre-revive L0 ≥ 90% (≥58/64 unique) at ep30+
- L1 / L2 ≥ 80%

### 失败模式 (R11.3)
- pre-revive L0 < 90% → dead_revive frequency 不是 L0 杠杆 (Issue #17 evidence 已是 73.44% baseline, 高频也无法改变 encoder 重新饿死码字的速率)
- post-revive L0 = 100% (always) → 数字看着漂亮但不可解释

## 风险评估
| 风险 | 缓解 |
|------|------|
| GPU 0/1 task279 训练被干扰 | D5 用 GPU 2 单独, < 90 sec 估 Stage 1 速度 |
| 重跑 Stage 3 浪费 | 全部在 Stage 1 验证, 不入 Stage 2/3/4 |
| 跟 task270 撞 recipe | task270 description 已闭环, D5 是新方向 |

## 物理产物

```
descriptions/task283_d5_dead_revive_frequency_design.md  (本文件)
scripts/task283_d5_dead_revive_frequency.sh  (3 频率 launcher, 单 GPU)
verdicts/task283_d5_dead_revive_frequency_result.md
products/task283/A_eval5/  (D5A 默认频率)
products/task283/B_eval1/  (D5B 高频)
products/task283/C_eval10/ (D5C 低频)
```

## 后续 (按 R10 + R11.5)

- 若 D5 任一频率 PASS → 进 Stage 2/3/4 闭环 (~1.5h GPU)
- 若 D5 全 NO-GO → 闭环 task283 + 在 §17 加 "dead_revive frequency 非 L0 ≥ 90% 杠杆"
- 候选 4: Sinkhorn curriculum / kmeans 重 init 需新建 description

## ROUTE
- 来源: Task #268 backlog high ROI decision §4 候选 3 (L0 utilization 新方向)
- 替代 task270 A2 (已被 #282 否定)
- 跟 Issue #18 (口径闭环) 正交 — 不改 §6.7.4 阈值, 只探索杠杆

result: Task #283 / D5 dead_revive frequency 设计 + launcher 写完. 等 D5A (eval_step=5, 默认频率) 在 GPU 2 跑 Stage 1. 通过条件 L0 ≥ 90% pre-revive at ep30+. 失败回退 D5B/D5C 或 NO-GO 闭环.
