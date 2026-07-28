# Task #220 — Stage 1 训练 逃法一 (Per-Codeword κ) 结果

## 结论
**🟢 PARTIAL SUCCESS — 首次在 HG-Rec 流水线观察到真正的 codebook escape, 但 epoch 35+ 又坍缩.**

| 层 | Healthy ckpt 利用率 (epoch 29) | vs baseline |
|----|------------------------------|-------------|
| L0 (K=64) | **20.31%** (13/64) | baseline 1.56% (1/64) — 提升 13× |
| L1 (K=128) | **96.09%** (123/128) | baseline ~4% — 提升 24× |
| L2 (K=256) | **93.75%** (240/256) | baseline ~6% — 提升 16× |
| unique SID | 5055 / 9922 (50.93%) | baseline ~3% — 提升 17× |

**关键洞察**: Per-Codeword κ 在 epoch 14-34 实现了真实 escape (collision 0.99→0.38), 但训练动力学在 epoch 35+ 把 codebook 拉回坍缩.

## 时间线 (epoch vs collision_rate)
| epoch | collision | L0_util | L1_util | L2_util | unique_sid | 状态 |
|-------|-----------|---------|---------|---------|-----------|------|
| 4 | 0.9999 | - | - | - | - | 起始坍缩 |
| 9 | 0.9879 | - | - | - | - | escape 启动 |
| 14 | 0.8717 | - | - | - | - | escape 持续 |
| 19 | 0.6268 | 12.5% | 66.4% | 84.0% | 3780 | escape 持续 |
| 24 | 0.4309 | 15.6% | 89.1% | 87.5% | 5523 | escape 最深 |
| **29** | **0.3835** | **20.3%** | **96.1%** | **93.8%** | **5055** | ⭐ **healthy local min** |
| 34 | 0.6273 | 15.6% | 78.9% | 94.1% | 2513 | 回到坍缩 |
| 99+ | 0.987-0.989 | (未测, ckpt 不存在) | | | | 完全坍缩 |
| 199 | 0.9886 | 4.7% | 2.3% | 1.2% | 3 | 完全坍缩 (best_loss) |

## 产物
- `products/task220/hrqvae_pck/Jul-26-2026_22-54-05_.../best_collision_model.pth` (epoch 29, ⭐ healthy)
- `products/task220/hrqvae_pck/Jul-26-2026_22-54-05_.../best_loss_model.pth` (epoch 199, 坍缩)
- `products/task220/hrqvae_pck/Jul-26-2026_22-54-05_.../epoch_29_collision_0.3835_model.pth` (R12 强制存)
- `logs/task220/pck_stage1_train.out` (完整 log, 1.5 分钟跑完 200 epoch)

## 配置 (Task #220 launcher)
```
assignment_mode=per_codeword_kappa, c_k_min=0.5, c_k_max=5.0, c_k_seed=42
num_emb_list=[64, 128, 256], e_dim=36, angular_dim=4, radial_dim=32
loss_type=poincare, beta=0.5, epochs=200, batch_size=1024, lr=1e-3
product_manifold=True, kmeans_init=True, kmeans_iters=1000
sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50
```

## 失败模式分析 — 为什么 epoch 35+ 又坍缩
观察 Phase 0 数据 (Task #178/#181/#199/#201/#203/#204) 累积规律:
- Poincaré 边界梯度饱和 + β=0.5 + 200 epoch 长训 → 码字被推到 boundary (‖x‖_E→1)
- epoch 29 时码字还在 ball 内, escape 信号清晰
- epoch 35+ 重构 loss 主导, 把 encoder → quantizer 拉到坍缩吸引子

**结论**: Per-Codeword κ 改变的是"assignment dynamics", 但不能阻止"boundary collapse dynamics". **healthy state 只能存在于训练早期, 必须早停才能捕获.**

## 决策
- **🟢 GO Phase 2**: Task #222 — 用同样配置重跑, 早停在 epoch 30, 锁定 healthy ckpt
- 复现后下游 Stage 2 SID 推断 + Stage 3 T5-mini 训练 + Stage 4 eval
- 对照: HG-Rec baseline R@10=0.1020 (Task #84)

## R12 验收
- ✅ best_loss ckpt 落盘
- ✅ best_collision ckpt 落盘
- ✅ epoch_29_collision_0.3835_model.pth 落盘 (R12 强制存)
- ✅ training PID 文件 (_TRAINING_PID) 写过 (launcher 退出后保留)

## 历史意义
这是 2026-07-26 用户提议"攻前提 a/b/d" 后的**第一个真实 escape 信号**:
- 12 方向 (Task #191/192/193/196/203/204/205/209/211/212/213/214) 全部 NO-GO (boundary collapse 持续)
- Task #218 (Phase 0): Per-Codeword κ 67-76% OPEN signal
- **Task #220**: Phase 1 Stage 1 真实复现 Phase 0 OPEN signal, healthy ckpt 锁定
- 下游待验证: T5-mini 训练是否在 healthy SID 基础上击败 baseline R@10=0.1020

## 相关任务
- #218: Phase 0 判据检查 (L0 TOO_STRONG / L1+L2 OPEN)
- #219: 平行 Phase 0 (Gromov L0 OPEN / L1+L2 TOO_STRONG)
- #221: 平行 Stage 1 (Gromov, 完全坍缩 NO-GO)
- #222 (待登记): Phase 2 早停 30 epoch 复现 healthy ckpt