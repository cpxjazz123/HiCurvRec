# Task #220 — Stage 1 训练 逃法一 (Per-Codeword κ-Stereographic)

## 来源
- Task #218 Phase 0 判据检查: L0=28.90% TOO_STRONG / L1=67.15% ✅ OPEN / L2=75.69% ✅ OPEN
- 用户 2026-07-26 提议: "直接进 Stage 1, 并保留提前中止 (epoch 20/50 查利用率和碰撞率)"
- 收窄 c_k_min/c_k_max 范围从 [0.5, 20] → [0.5, 5.0] 来缓解 L0 过强 (TOO_STRONG) 问题

## 核心想法
攻命题前提 (b)+(d): 全层共享曲率 c 改为每个码字有独立曲率 c_k.
`score_k = (1/√c_k) · arccosh(1 + 2c_k‖z−e_k‖²/[(1−c_k‖z‖²)(1−c_k‖e_k‖²)])`
- 用户验证: ratio 4.6×→2.3× 随 δ 变化 (非恒等 rescaling)
- 期望: L0 因为 ρ 较小, c_k_min 收窄到 0.5 时不会过于激烈

## 配置
| 项 | 值 |
|----|-----|
| assignment_mode | per_codeword_kappa |
| c_k_min | 0.5 |
| c_k_max | 5.0 |
| c_k_seed | 42 |
| num_emb_list | 64 128 256 |
| e_dim | 36 |
| angular_dim | 4 |
| radial_dim | 32 |
| layers | 512 256 128 36 |
| loss_type | poincare |
| beta | 0.5 |
| epochs | 200 (Phase 1 探索用, 不跑满 1000) |
| batch_size | 1024 |
| lr | 1e-3 |
| kmeans_init | True |
| kmeans_iters | 1000 |
| sk_eps | 0.0 0.0 0.0 |
| sk_iters | 50 |
| GPU | 0 |

## 提前中止规则 (用户硬要求)
- epoch 20 evaluate 后 → 跑 9922 items 通过训练好的 model 算 L0 utilization (unique_indices / 64)
  - utilization < 30% → kill + 写 verdict (TOO_STRONG 复现)
  - collision_rate > 95% → kill + 写 verdict (坍缩)
  - 其他 → 继续
- epoch 50 evaluate 后 → 同样检查
  - utilization < 50% → kill (利用不足)
  - collision_rate > 80% → kill (坍缩)
  - 其他 → 继续跑到 200 epoch

## 产物
- products/task220/hrqvae_pck/best_loss_model.pth (R12: 强制存, 每 epoch 评估完落盘)
- products/task220/hrqvae_pck/epoch_X_collision_Y_model.pth (R12: 强制存, 每 epoch 落盘, 旧 ckpt 删除)
- products/task220/_TRAINING_PID (R12.2)
- logs/task220/pck_stage1_train.out

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| L0 utilization @ epoch 50 | ≥ 50% | < 30% (直接终止) |
| collision_rate @ epoch 50 | ≤ 50% | > 80% (直接终止) |
| 训练 loss 收敛 (epoch 100 vs 200) | < 5% 改善 | 持续发散 |

## 下一步
若 Stage 1 通过阈值 → Stage 2 (SID 推断) + Stage 3 (T5-mini 训练) + Stage 4 (test eval)
对照: HG-Rec baseline R@10=0.1020 (Task #84)