# Task #221 — Stage 1 训练 逃法二 (Gromov Product)

## 来源
- Task #219 Phase 0 判据检查: L0=79.65% ✅ OPEN / L1=50.21% TOO_STRONG / L2=40.48% TOO_STRONG
- 用户 2026-07-26 提议: "直接进 Stage 1, 并保留提前中止 (epoch 20/50 查利用率和碰撞率)"
- Task #218 vs #219 互补: Gromov 在 L0 (最浅层, 码字半径分布最大) 打开, Per-Codeword κ 在 L1/L2 打开

## 核心想法
攻命题前提 (a): 用共同祖先深度代替点对点距离.
`score_k = ½ [d(0,z) + d(0,e_k) − d(z, e_k)] ∝ ρ_k − d(z, e_k)` (argmax, 符号翻转)
- 数学等价: argmax(ρ_k - d) ≡ argmax(Gromov product) 100% across all 3 layers
- 物理意义: z 与 e_k 在同一分支叶子的深度差 = 它们的相似度

## 配置
| 项 | 值 |
|----|-----|
| assignment_mode | gromov |
| num_emb_list | 64 128 256 |
| e_dim | 36 |
| angular_dim | 4 |
| radial_dim | 32 |
| layers | 512 256 128 36 |
| loss_type | poincare |
| beta | 0.5 |
| epochs | 200 |
| batch_size | 1024 |
| lr | 1e-3 |
| kmeans_init | True |
| kmeans_iters | 1000 |
| sk_eps | 0.0 0.0 0.0 |
| sk_iters | 50 |
| GPU | 1 |

## 提前中止规则 (用户硬要求)
- epoch 20 evaluate 后 → 跑 9922 items 算 L0 utilization
  - utilization < 30% → kill + 写 verdict
  - collision_rate > 95% → kill + 写 verdict
  - 其他 → 继续
- epoch 50 evaluate 后 → 同样检查
  - utilization < 50% → kill
  - collision_rate > 80% → kill
  - 其他 → 继续跑到 200 epoch

## 产物
- products/task221/hrqvae_gromov/best_loss_model.pth (R12)
- products/task221/hrqvae_gromov/epoch_X_collision_Y_model.pth (R12)
- products/task221/_TRAINING_PID (R12.2)
- logs/task221/gromov_stage1_train.out

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| L0 utilization @ epoch 50 | ≥ 50% | < 30% (直接终止) |
| collision_rate @ epoch 50 | ≤ 50% | > 80% (直接终止) |
| 训练 loss 收敛 | < 5% 改善 (epoch 100 vs 200) | 持续发散 |

## 互补性总览 (跟 Task #220 对比)
| 层 | Per-Codeword κ (#220) | Gromov (#221) |
|----|----------------------|---------------|
| L0 (K=64) | 28.90% TOO_STRONG | **79.65% ✅ OPEN** |
| L1 (K=128) | **67.15% ✅ OPEN** | 50.21% TOO_STRONG |
| L2 (K=256) | **75.69% ✅ OPEN** | 40.48% TOO_STRONG |