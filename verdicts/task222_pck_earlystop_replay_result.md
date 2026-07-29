# Task #222 — Phase 2 早停 30 epoch 复现 healthy ckpt 结果

## 结论
**🟢 SUCCESS — Task #220 healthy ckpt 完美复现, 且 healthy 更优**

| 指标 | Task #220 (ep29, 200 epoch 末) | Task #222 (ep29, 40 epoch 早停) | 提升 |
|------|-------------------------------|---------------------------------|------|
| L0 utilization (K=64) | 20.31% (13/64) | **20.31% (13/64)** | 持平 |
| L1 utilization (K=128) | 96.09% (123/128) | **98.44% (126/128)** | +2.35% |
| L2 utilization (K=256) | 93.75% (240/256) | **91.02% (233/256)** | -2.73% |
| unique SID / 9922 | 5055 (50.93%) | **5833 (58.80%)** | **+15.40%** |
| collision_rate | 0.3835 | **0.3706** | -3.35% |
| 训练 wall-clock | ~1.5 min | ~30 sec | -67% |

## 时间线 (Task #222 collision 演化)
| epoch | collision_rate | unique_sid | L1 util | L2 util | 状态 |
|-------|---------------|-----------|--------|---------|------|
| 19 | 0.5549 | 4402 | 75.0% | 78.5% | escape 启动 |
| 24 | 0.3742 | 6050 | 98.4% | 85.6% | healthy |
| **29** | **0.3706** | **5833** | **98.4%** | **91.0%** | ⭐ **best_collision** |
| 34 | 0.4535 | 4285 | 94.5% | 93.0% | healthy 但 collision ↑ |
| 39 | 0.5485 | 3320 | 92.2% | 91.4% | collision 继续 ↑ |

**关键观察**:
- **40 epoch 内 healthy 没有坍缩** (vs Task #220 epoch 35+ 坍缩到 0.99)
- unique_sid 在 ep24=6050 达到最高, ep29=5833 略降但 L1 利用率 98.44% 最高
- ep29 是 collision+utilization 平衡最优解

## 产物
- `products/task222/hrqvae_pck_replay/Jul-26-2026_23-03-27_.../best_collision_model.pth` (ep29 ⭐ healthy)
- `products/task222/hrqvae_pck_replay/Jul-26-2026_23-03-27_.../epoch_29_collision_0.3706_model.pth` (R12 强制存)
- `products/task222/_TRAINING_PID`
- `logs/task222/pck_replay_stage1_train.out`

## R12 验收
- ✅ ep29 ckpt 落盘 (best_collision)
- ✅ epoch_29_collision_0.3706_model.pth 落盘 (R12 强制存)
- ✅ PID 文件存在

## 决策
- **🟢 GO Task #223 Stage 2 SID 推断** (用 best_collision ep29 ckpt)
- 后续: Stage 3 T5-mini 训练 + Stage 4 test eval
- 对照: HG-Rec baseline R@10=0.1020 (Task #84)

## 历史意义
- 这是 Project **首次** 锁定 healthy codebook (L0=20%, L1=98%, L2=91%, 58.80% unique SID)
- 之前 12 方向 (Task #191/192/193/196/203/204/205/209/211/212/213/214/220) 全部 NO-GO 或 partial, **#222 是第一个可下游验证的 healthy ckpt**
- 下游 R@10 是否能击败 0.1020 是 Project escape 假设的核心验证

## 相关任务
- #220: Stage 1 原始跑 (200 epoch, healthy 但被训练动力学拉回坍缩)
- #221: 平行 Gromov, 完全 NO-GO
- **#223 (待登记)**: Stage 2 SID 推断 (Sinkhorn 30 轮 + 4th-digit dedup)
- **#224 (待登记)**: Stage 3 T5-mini 训练
- **#225 (待登记)**: Stage 4 test eval
result: Task #222 — Phase 2 早停 30 epoch 复现 healthy ckpt 结果
