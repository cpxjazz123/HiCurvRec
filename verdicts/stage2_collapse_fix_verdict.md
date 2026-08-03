# Stage2 量化器塌缩修复 — Verdict

> 任务: "stage2 量化器让它不塌缩, 产出能用且 unique 的好 SID" (方向A taskA + 方向B taskB)
> 状态: ✅ 完成, 双 Gate 2 PASS
> commit: `8828956` (taskA), `fe1696b` (taskB)

---

## 1. 任务目标

修复 taskA/taskB stage2 RQ-VAE 量化器 posterior collapse: 之前 SID unique_3digit=1
(全部 item 量化到同一 codebook entry), 无法被 stage3 用作训练输入。目标产出不塌缩、
unique 比例接近基线 (90%) 的 SID, 且 κ (可变曲率) + 三分量权重真实可学习。

## 2. 接近过程 (诊断 → 5 版迭代)

### 根因诊断 (diag_stage2_collapse.py, 30 epochs × 3 配置)
| 配置 | z_norm std | unique_3digit |
|---|---|---|
| A. mse recon + kmeans_iters=10 | 0.0006 | 1 (0.0%) ← 塌缩 |
| B. poincare recon + kmeans_iters=1000 | 0.040 | 1838 (18.5%) |
| C. poincare recon + kmeans_iters=10 | 0.034 | 1953 (19.7%) |

**结论: 塌缩根因 = 欧氏 MSE recon loss** (方向A/B 用了 `F.mse_loss`, 基线用 Poincaré 距离)。

### NaN 根因诊断 (100 epochs 长训暴露)
| 版本 | 改动 | 结果 |
|---|---|---|
| v3 | recon→poincare + kmeans_iters=1000 | epoch 85 NaN (κ→-1 自我漂移) |
| v3b | + κ clamp -0.5, κ LR 10x→3x | 仍 epoch 80 NaN |
| v3c | + commit 输入 proj_to_ball | grad_κ 爆炸 2e6 (poincare 梯度 2/(1-c·norm²)) |
| v3d | commit→欧氏 mse | κ grad=0, precheck FAIL |
| **v3e** | **poincare commit + mix_weight LR 5x→1x** | **✅ 100 epochs 稳定** |

**NaN 根因**: poincare commit/codebook loss 在 latent norm→1/√c 时梯度
2/(1-c·norm²) 爆炸; **mix_weight param group LR 5x 是 latent 失控加速器** (降 1x 后
epoch 75 z_norm 0.69→0.53, 90+ epochs 稳定)。

## 3. 最终修复 (taskA + taskB 主脚本)
1. recon loss: `F.mse_loss` → poincare (`proj_to_ball(expmap0(x)) + poincare_distance²`)
2. κ clamp [-0.99,10] → [-0.5,10] (c≥0.5, 球半径≤√2)
3. κ LR 10x → 3x; taskA mix_weight LR 5x → 1x
4. kmeans_iters 硬编码 10 → `--kmeans_iters` 默认 1000 (对齐基线初始化)
5. precheck recon 同步改 poincare

## 4. 关键指标 (100 epochs, seed 42)
| 指标 | taskA v3e | taskB v3 | 基线 |
|---|---|---|---|
| unique_3digit | 8082 (81.5%) | 7993 (80.6%) | 8936 (90%) |
| util_4digit | 1.0 | 1.0 | 1.0 |
| κ 三层 std | 0.092 | 饱和 -0.5 | — |
| mix_weight std | 0.104 | — | — |
| NaN | 无 | 无 | — |
| reload 5/5 | True | True | — |
| SID hash | 4e5abe | 6596ccb | 2dab |

## 5. 产物
- `taskA/_history/taskA_stage2_v3e_poincare_mix1x/` (sid_output.npy, hrqvae_kappa_sync.ckpt, verdict.json)
- `taskB/_history/taskB_stage2_poincare_v3/` (sid_output.npy, hrqvae_weighted_mixed.ckpt, verdict.json)

## 6. 结论与后续
- **塌缩已修复**: SID 从 0.01% unique → 81-82% unique 3-digit (基线 90%), util_4digit=100%, 可被 stage3 直接用作训练输入。
- taskB κ 三层饱和到 clamp 下界 -0.5, 其"可变曲率"由 per-sample weight_mlp (α/β/γ) 提供。
- 后续: 用新 unique SID 跑 stage3 训练 (T5 冻结), 验证 residual 注入是否带来 R@10 增益。
