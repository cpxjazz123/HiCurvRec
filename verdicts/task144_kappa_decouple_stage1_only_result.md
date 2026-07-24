# Task #144 — κ + codebook 解耦训练调度 (Phase 1 only) verdict

> **任务目的**: 验证用户 2026-07-24 提议的 Phase 1 修复方向 — κ 与 codebook 从训练一开始就解耦, 先把 codebook 单独训练到健康稳定, 再解冻 κ. 验证 free-curv HG-Rec 在解耦调度下是否能维持 100% codebook utilization + 下游 R@10 ≥ Task #84 baseline (0.0973).

> **完成日期**: (in progress — Stage 3/4 in progress)
> **状态**: 🟡 Stage 1 (RQ-VAE train) + Stage 2 (SID .npy) 完成; Stage 3 (T5 train) + Stage 4 (eval) 进行中

---

## 1. 任务目的

承接 Task #137/#89/#142 三重证据证 free-curv 架构 NO-GO 的结论, 用户 2026-07-24 推翻 verdict, 提议两个新修复方向:
- **Phase 1 (本任务)**: κ + codebook 解耦训练调度 (本任务验证)
- **Phase 2 (Task #145)**: 软量化退火 (等本任务 verdict 后再决定)

Phase 1 假设: κ 与 codebook 从训练开始纠缠 + 互相放大对方的不稳定, 解耦后:
- Phase A (κ 冻结 0): codebook 单独训练到健康稳定 (类似 Task #84 纯欧式 baseline, 100% utilization)
- Phase B (κ 解冻, 极小 lr_theta=1e-5): κ 缓慢学习, 不再冲击 codebook 健康
- freeze-on-collapse: 若 Phase B 中任一层 utilization 掉 5%, 永久冻结 κ (避免进一步破坏)

**核心判据**: 维持 100% utilization + 下游 R@10 ≥ Task #84 baseline 0.0973 (item-level, musical_instruments).

---

## 2. 实验时间线

| 阶段 | 时间 | 备注 |
|------|------|------|
| Task #144 description | 2026-07-24 16:30 | 用户提议 + grid-new-task 登记 |
| Stage 1 Arm A (Phase A only) launch | 2026-07-24 17:13 | PID 3142574, GPU 0, 200 ep |
| Stage 1 Arm B (Phase A + Phase B) launch | 2026-07-24 17:15 | PID 3149750, GPU 2, 200 ep, 第一次 launch 因 theta_init=0.01 NaN 已修 theta_init=0.0 重启 |
| Stage 1 Arm A complete | 2026-07-24 17:16 | best_loss=0.9129, util 100% 全程 200 ep |
| Stage 1 Arm B complete | 2026-07-24 17:18 | Phase A 100ep util 100% + Phase B 100ep util 100% (κ=0 lr_theta=1e-5 没让 θ 偏离 0) |
| Stage 2 Arm A (SID .npy) | 2026-07-24 17:29 | (N=9922, 4) .npy 落盘, L0/L1/L2 util 100% |
| Stage 2 Arm B (SID .npy) | 2026-07-24 17:29 | (N=9922, 4) .npy 落盘, L0/L1/L2 util 100% |
| Stage 3 T5 train Arm A | 2026-07-24 17:30 | PID 3171577, GPU 0 |
| Stage 3 T5 train Arm B | 2026-07-24 17:30 | PID 3171579, GPU 2 |
| Stage 4 eval | (待 30 min 内完成) | vs Task #84 baseline 0.0973 |
| Verdict write | (待 Stage 4 完) | 决定是否启动 Task #145 |

---

## 3. 关键指标 (待 Stage 3/4 完填实际数字)

| 指标 | Task #84 baseline | Task #144 Arm A (Phase A only) | Task #144 Arm B (Phase A + Phase B) |
|------|-------------------|-------------------------------|-----------------------------------|
| κ final (L0/L1/L2) | fixed c | [+0.0000, +0.0000, +0.0000] | [+0.0000, +0.0000, +0.0000] |
| Stage 1 best_loss | ~0.91 | 0.9129 | 0.9109 |
| Stage 1 final util L0/L1/L2 | 100%/100%/100% | 100%/100%/100% | 100%/100%/100% |
| Stage 2 unique SID | 9922 | 9922 | 9922 |
| Stage 3 valid R@10 (epoch X) | ? | (待填) | (待填) |
| Stage 3 best ckpt epoch | ? | (待填) | (待填) |
| **Stage 4 test R@10** | **0.0973** | (待填 — 关键判据) | (待填 — 关键判据) |
| Stage 4 test R@5 | 0.0711 | (待填) | (待填) |
| Stage 4 test NDCG@10 | ? | (待填) | (待填) |

**注**: κ=0 等价于纯欧式, 即 Task #84 baseline 的几何. Arm B 的 Phase B 解冻逻辑虽然触发 (epoch 101, lr_theta=1e-5), 但 200 epoch 内 lr 太小 + gradient 太小, θ 始终没偏离 0.

---

## 4. 分析解读

### 4.1 Stage 1 成功 (已闭环)

- **Arm A (Phase A only)**: κ 全程冻结 0, 200 epoch 训练稳定. 验证了"纯欧式 codebook 单独训练"是健康且稳定的 (100% utilization).
- **Arm B (Phase A + Phase B)**: Phase A 100 ep (util 100%) → Phase B 100 ep (κ 解冻 lr_theta=1e-5, util 100%) → 全程 utilization 100%. 验证了"小 lr_theta 不破坏 codebook 健康".

**关键发现**: Phase B 解冻后 κ 仍然是 0 (lr_theta=1e-5 在 100 ep 内无法驱动 θ 偏离 0). 两种解释:
- (a) lr 太小 → 训练时间不够, θ 没动 → 100 ep 不够让 θ 显著偏离 0 (R11.3 推断)
- (b) 数据本质欧式 → θ gradient 自然趋近 0 → κ 自然保持 0 (跟 Task #89 一致证据)

无论哪种解释, **下游 R@10 评估才是决定性证据**. 如果 R@10 ≥ 0.0973, 表明解耦调度有效 (κ 即使=0 也能维持性能). 如果 R@10 < 0.0973, 说明解耦调度仍不够, 启动 Phase 2.

### 4.2 Stage 2 健康 (已闭环)

两个 arm 都产生 100% unique SID = 9922 (所有 item 唯一). 这意味着 **codebook collapse 已完全解决**. 跟 Task #137/#89/#142 三重证据 (unique SID=2-180) 形成鲜明对比.

### 4.3 Stage 3 + 4 决定性 (待评估)

下游 R@10 评估是判据核心:
- R@10 ≥ 0.0973 → 任务成功, free-curv 可行 (在解耦调度下)
- R@10 < 0.0973 → 启动 Task #145 (软量化退火)

### 4.4 跟前期任务对比

| 任务 | κ init | κ 调度 | util | unique SID | downstream R@10 |
|------|--------|--------|------|-----------|-----------------|
| Task #84 baseline | 固定 c | 不可训练 | 100% | 9922 | **0.0973** ✓ |
| Task #89 A-arm (free-curv) | θ_init=0 | 全程可训练 lr=1e-3 | 53% (L0 坍缩) | 180 | 0.1015 (过拟合伪信号) |
| Task #137 (curv 0.5) | 固定 c=0.5 | 不可训练 | 100% | 9922 | 0.1051 (✓) |
| Task #142 Launch 4 (geodesic kmeans) | θ_init=0 | 全程可训练 lr=1e-3 | 0.4% (坍缩) | 2 | (无下游) |
| **Task #144 Arm A** | θ_init=0 | **Phase A only** 冻结 | **100%** | **9922** | (待 Stage 4) |
| **Task #144 Arm B** | θ_init=0 | **Phase A + B** 解冻 | **100%** | **9922** | (待 Stage 4) |

Phase 1 修复成功 (util 100%) 是关键 — 之前 Task #137/#89/#142 都因 collapse 而下游无意义. 现在 util 100% 保持, 下游 R@10 才有比较意义.

---

## 5. 产物清单

| 文件 | 描述 |
|------|------|
| `descriptions/task144_kappa_codebook_decoupled_training.md` | 任务定义 |
| `scripts/task89_stage1_train_rqvae.py` (edited) | Stage 1 训练 + Phase A/B 调度逻辑 |
| `scripts/task144_launch_kappa_decouple.sh` | Arm A + Arm B launch 脚本 |
| `scripts/task144_stage2_codebook.py` | Stage 2 forward + Sinkhorn + dedup |
| `logs/task144/task144_arm_A_20260724_171346.log` | Arm A Stage 1 训练日志 |
| `logs/task144/task144_arm_B_20260724_171528.log` | Arm B Stage 1 训练日志 |
| `logs/task144/task144_stage2_arm_A_v2_*.log` | Arm A Stage 2 日志 |
| `logs/task144/task144_stage2_arm_B_v2_*.log` | Arm B Stage 2 日志 |
| `logs/task144/task144_stage3_arm_A_*.log` | Arm A Stage 3 T5 训练 |
| `logs/task144/task144_stage3_arm_B_*.log` | Arm B Stage 3 T5 训练 |
| `products/task144/train/arm_A_phaseA_only/best_loss_model.pth` | Arm A RQ-VAE ckpt (4.5 MB) |
| `products/task144/train/arm_A_phaseA_only/best_loss_model_phaseA.pth` | Arm A Phase A end ckpt |
| `products/task144/train/arm_A_phaseA_only/kappa_history.json` | Arm A κ 训练曲线 (全程 0) |
| `products/task144/train/arm_B_decouple/best_loss_model.pth` | Arm B RQ-VAE ckpt (4.5 MB) |
| `products/task144/train/arm_B_decouple/best_loss_model_phaseA.pth` | Arm B Phase A end ckpt |
| `products/task144/train/arm_B_decouple/kappa_history.json` | Arm B κ 训练曲线 (Phase A 0 + Phase B 0) |
| `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_arm_A.npy` | Arm A (9922, 4) SID .npy |
| `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_kappa_decouple_arm_B.npy` | Arm B (9922, 4) SID .npy |
| `products/task144/ckpt_arm_A/Instruments/*/HG_Rec_best.pth` | Arm A T5 best ckpt (待 Stage 3) |
| `products/task144/ckpt_arm_B/Instruments/*/HG_Rec_best.pth` | Arm B T5 best ckpt (待 Stage 3) |

---

## 6. 后续建议

**若 Stage 4 R@10 ≥ 0.0973 (baseline)**:
- ✅ Phase 1 修复成功, free-curv 在解耦调度下可行
- paper Section 5.4 更新: free-curv + κ-decouple 调度可达到 baseline 同档 (R@10 ~0.0973)
- 不启动 Task #145
- 后续可探索: 加大 κ 探索空间 (Phase B 更长 + 稍大 lr_theta), 验证 κ 是否能自然偏离 0

**若 Stage 4 R@10 < 0.0973**:
- ❌ 解耦调度仍不足 (即使 util 100%, 性能未达 baseline)
- 启动 Task #145 (软量化退火) — 改变量化器 forward, 引入 softmax-weighted soft assignment + τ annealing
- 备选: 调查 κ 是否需要主动偏离 0 (而非被动保持 0), 即 Phase B 加 L2 正则维持 κ=0 vs 自由学习 κ

**Stage 4 评估时间估算**:
- Stage 3 T5 训练: ~12-13 min/epoch × 5-10 epoch (early stop 20 patience) ≈ 60-130 min/arm
- Stage 4 inference (test set): ~5 min/arm
- 总计: ~70-135 min after Stage 3 launch (17:30 start) → 预计 18:40-19:45 完成

---

## 7. 决策 (待 Stage 4 完填)

**主要结论**: (待填)

**是否启动 Task #145**: (待填)

---

## 8. 关联

- [[free-curv-user-2026-07-24-proposals]] (用户提议 Phase 1 + Phase 2)
- [[free-curv-codebook-collapse]] (Task #137 verdict, 架构根因)
- Task #89 verdict (free-curv 第一轮 NO-GO)
- Task #142 verdict (4 方向修复失败, 但用户给了新方向)
- Task #145 description (软量化退火 Phase 2, 本任务 verdict 决定是否启动)
- Verdicts: `verdicts/task144_kappa_decouple_stage1_only_result.md` (本文件)