# Task #221 — Stage 1 训练 逃法二 (Gromov Product) 结果

## 结论
**🔴 NO-GO — Gromov 攻前提 (a) 在训练阶段失败, 全程坍缩.**

| 层 | 训练过程 collision | ckpt 利用率 (best_loss ep124) | 状态 |
|----|-------------------|-----------------------------|------|
| L0 (K=64) | 0.999-0.9999 (全程) | 3.12% (2/64) | 完全坍缩 |
| L1 (K=128) | 0.999-0.9999 (全程) | 1.56% (2/128) | 完全坍缩 |
| L2 (K=256) | 0.999-0.9999 (全程) | 0.78% (2/256) | 完全坍缩 |
| unique SID | 0.0008 (3-7 / 9922) | 0.03% | 几乎所有 item 同 SID |

## 时间线
| epoch | collision_rate |
|-------|---------------|
| 4 | 0.9999 |
| 9 | 0.9993 |
| 14-199 | 0.9991-0.9999 (稳定在 99%+) |

**没有 escape 信号** — 训练动力学把 Gromov product 推到跟 shared κ 一样的坍缩吸引子.

## 产物
- `products/task221/hrqvae_gromov/Jul-26-2026_22-54-05_.../best_loss_model.pth` (epoch 124, 坍缩)
- `products/task221/hrqvae_gromov/Jul-26-2026_22-54-05_.../best_collision_model.pth` (最早 epoch, 同样坍缩)
- `logs/task221/gromov_stage1_train.out` (1.5 分钟跑完 200 epoch)

## 配置 (Task #221 launcher)
```
assignment_mode=gromov (argmax Gromov product = ρ_k - d(z, e_k), 攻前提 a)
num_emb_list=[64, 128, 256], e_dim=36, angular_dim=4, radial_dim=32
loss_type=poincare, beta=0.5, epochs=200, batch_size=1024, lr=1e-3
product_manifold=True, kmeans_init=True, kmeans_iters=1000
sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50
```

## 失败模式分析
- **Phase 0 OPEN (Task #219)**: Gromov L0 79.65% OPEN, 但训练后**所有层都被拉回坍缩**
- 数学等价 `argmax(ρ_k - d) ≡ argmax(Gromov)` (100% across all 3 layers) — 但这个等价是**离线计算**, 训练时跟 reconstruction loss + Sinkhorn 一起作用, 把 Gromov dynamics 也拉到坍缩吸引子
- 跟 Task #220 对比: Per-Codeword κ 在训练早期产生 escape 信号, Gromov 在训练早期**没有**产生 escape 信号

**结论**: Gromov product 攻前提 (a) 仅在 Phase 0 离线检查 (参数固定) 时有效, 训练动力学仍把 distance-based assignment 拉回坍缩.

## 决策
- **🔴 NO-GO Phase 2**: Gromov 逃法终止, 不进入下游 Stage 2/3/4
- Project escape 路径**只剩 Task #220 Per-Codeword κ** 一个候选

## 互补性总览 (Task #218 vs #219 vs #220 vs #221)
| 层 | Per-Codeword κ (offline Phase 0) | Per-Codeword κ (train #220) | Gromov (offline Phase 0) | Gromov (train #221) |
|----|--------------------------------|---------------------------|-------------------------|---------------------|
| L0 | TOO_STRONG 28.90% | 20.31% (escape 持续 ep 14-34) | OPEN 79.65% | 3.12% 全程坍缩 |
| L1 | OPEN 67.15% | 96.09% (escape 持续 ep 14-34) | TOO_STRONG 50.21% | 1.56% 全程坍缩 |
| L2 | OPEN 75.69% | 93.75% (escape 持续 ep 14-34) | TOO_STRONG 40.48% | 0.78% 全程坍缩 |

**核心信号**: Per-Codeword κ 是**唯一**能在训练中产生 escape 的方向. Gromov 在离线检查时 L0 79.65% OPEN 是"参数固定"假象, 训练动力学把它压回坍缩.

## 相关任务
- #219: Phase 0 判据检查 (Gromov L0 OPEN, 但训练后坍缩)
- #220: 平行 Stage 1 (Per-Codeword κ, escape 真实发生)
- #222 (待登记): Task #220 Phase 2 早停复现 healthy ckpt