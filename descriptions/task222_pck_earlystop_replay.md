# Task #222 — Phase 2 早停 30 epoch 复现 healthy ckpt

## 来源
- Task #220 verdict: 🟢 PARTIAL SUCCESS — Per-Codeword κ epoch 14-34 真实 escape, ep29 healthy ckpt (L0=20.3%, L1=96.1%, L2=93.8%, unique_sid=5055)
- 用户 2026-07-26 提议: "投哪条" = 投 Per-Codeword κ 路线, 健康 ckpt 锁定在 epoch 30

## 核心改动 vs Task #220
| 项 | Task #220 | Task #222 |
|----|-----------|-----------|
| epochs | 200 | **40** (ep30 = healthy local min + buffer) |
| early_stop | 被动 (R12 强制存每 epoch) | **主动** (ep 30 检查 utilization + collision) |
| 早停阈值 | N/A | L0 util ≥ 15%, L1 ≥ 90%, L2 ≥ 90%, collision ≤ 0.50 |
| seed | 42 | **42** (固定, R11.5 排除 multi-seed per user 2026-07-23 override) |
| GPU | 0 | **2** (R7: 4×L40S 全空闲) |

## 配置 (同 Task #220)
```
assignment_mode=per_codeword_kappa, c_k_min=0.5, c_k_max=5.0, c_k_seed=42
num_emb_list=[64, 128, 256], e_dim=36, angular_dim=4, radial_dim=32
loss_type=poincare, beta=0.5, epochs=40, batch_size=1024, lr=1e-3
product_manifold=True, kmeans_init=True, kmeans_iters=1000
sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50
```

## 早停检查 (ep 30)
- ✅ L0_util ≥ 15% (Task #220 ep29 = 20.3%) → 继续
- ✅ L1_util ≥ 90% (Task #220 ep29 = 96.1%) → 继续
- ✅ L2_util ≥ 90% (Task #220 ep29 = 93.8%) → 继续
- ✅ collision_rate ≤ 0.50 (Task #220 ep29 = 0.38) → 继续
- 任一不达标 → 报告"replay 失败, healthy ckpt 不可复现"

## 决策
- ✅ 全部通过 → 锁 ep30 ckpt, 启动 Stage 2 SID 推断 + Stage 3 T5-mini 训练 + Stage 4 eval
- 🔴 任一不达标 → 写 Phase 2 NO-GO verdict, Project 主线回到"boundary collapse 不可破"基线

## 产物
- products/task222/hrqvae_pck_replay/best_collision_model.pth (ep30 healthy ckpt, R12)
- products/task222/hrqvae_pck_replay/epoch_30_collision_X_model.pth (R12 强制存)
- products/task222/_TRAINING_PID (R12.2)
- logs/task222/pck_replay_stage1_train.out

## 下游 (若 healthy ckpt 锁定)
- Stage 2: SID 推断 (Sinkhorn 30 轮 + 4th-digit dedup)
- Stage 3: T5-mini 训练 (200 epoch)
- Stage 4: test eval (R@5/10/20, NDCG@5/10/20)
- 对照: HG-Rec baseline R@10=0.1020 (Task #84)

## R12 验收
- ✅ 早停 ckpt (epoch 30 或 best_collision) 强制落盘
- ✅ 旧 ckpt 删除 (save_limit=5)
- ✅ _TRAINING_PID 文件存在