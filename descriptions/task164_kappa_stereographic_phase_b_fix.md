# Task #164 — κ-Stereographic Phase A/B 修复 (200 epoch) + 完整 Stage 1-4

> **任务目的**: 用 Phase A/B 训练机制 (冻结 θ_m → 慢速解冻 lr_theta=1e-5) 修复 κ-Stereographic 距离公式下的 codebook collapse,Stage 4 R@10 > baseline (Task #88 c=[0.5,0.5,0.5] R@10=0.1051 或 phonism 0.1058)

> **完成日期**: 2026-07-25
> **状态**: 🟡 Phase A/B 训练待启动

---

## 1. 背景

**前置结论**:
- Task #89 / Task #162 FreeCurv + κ-Stereographic 公式:codebook 完全坍塌 (util < 5%)
- Task #118 c=[2,2,2] (κ=2 固定) = 100% 利用率 (decisive evidence!)
- Task #144 试过 Phase A (kappa_freeze_epochs=200) 用 **旧 3-branch torch.where** 公式 → Stage 1 util 100% 但 Stage 4 R@10=0.0973 (与基线齐平,未超越)
- **Task #164 (ad-hoc)**:κ-Stereographic 公式 + θ_init=[0,0,0] + 无 Phase A/B → 100 epoch 内全坍塌到 κ=-1.99 (真实测距辐射)

**核心假设 R1**:Phase A (冻结 θ_m 使 κ≡0) + 训练完成后再以极低 lr_theta=1e-5 解冻,既能让 codebook 在欧式度量下学到健康分布,又能避免 θ_m 训练梯度的"双向正反馈"循环,最终训练出的 codebook + 微调 κ 在保持几何一致性的同时被下游 Stage 3 T5 利用。

---

## 2. 实验设计

**Phase P0 (已完成,作 Phase A/B 修复的对照组)**: task164 launcher (`task164_kappa_stereographic_short_test.sh`)
- θ_init=[0,0,0], κ_max=2.0, **无 kappa_freeze_epochs** (自由学习)
- 结果:L0/L1/L2 κ = [-1.99, -1.98, -1.97] 全跑到 -κ_max (kappa_history.json)
- util 跌至 0.39%~3.12%

**Phase P1 (本任务核心):Phase A (100 ep 冻结) → Phase B (100 ep 解冻 lr_theta=1e-5)**
**变量**: 加 `--kappa_freeze_epochs=100` (Phase A), 保留 `--lr_theta=1e-5` (Phase B 解冻),其余 Recipe 与 #164 P0 完全一致
**保持不变**:
- M=1, κ_max=2.0, theta_init_list=0 0 0
- β=0.5, num_emb=32/64/256, e_dim=32, layers=512/256/128
- sk_epsilons=0.0 0.0 0.0 (与 P0 一致,先看 Phase A/B 主效应)
- dead_code_reset_every=0, kmeans_init enabled
- seed=42

**启动命令 Stage 1** (`task164_part2_kappa_decouple.sh`):
```bash
CUDA_VISIBLE_DEVICES=0 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-5 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 2.0 \
    --kappa_freeze_epochs 100 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --dead_code_reset_every 0 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --log_interval 20 \
    --save_every 50 \
    --TRITON_CACHE_DIR /home/wlia0047/.triton/cache_task164_p1
```

**Stage 2 (SID inference)** — 跑完 Stage 1 直接 fork HG-Rec Stage 2 utils,产出 `_t5_rqvae_phase_b.npy`
**Stage 3 (T5-small 60M)** — `task84_hgrec_stage3_train.py` with code_path=`_t5_rqvae_phase_b.npy`, vocab_size=1025
**Stage 4 (eval)** — `task84_hgrec_stage4_eval.sh` fork, R@10 / R@5 / NDCG@5 / NDCG@10

---

## 3. 决策触发 (vs baseline)

| 指标条件 | 结果区间 | 决策 |
|----------|----------|------|
| Stage 1 util L0/L1/L2 都 ≥ 80% | 健康 codebook,继续下游 | ✅ 假设 R1 第一部分验证 |
| Stage 1 κ_m 始终保持在 \|κ_m\| < 0.3 (解冻后) | Phase B 未触发反向坍塌 | ✅ Phase B 修复路径成立 |
| Stage 4 R@10 > 0.1058 (phonism) | **任务成功 — 超越最强 baseline** | 🎯 目标达成, 写 verdict, 触发 paper 更新 |
| Stage 4 R@10 ∈ [0.0973, 0.1058] | 与 baseline 齐平 | ⚠️ 用 κ-stereographic 未拿到增益, 写 Phase B 修复 verdict (锁定贡献) |
| Stage 1 util 仍 < 50% 或 κ 退回 ±κ_max | Phase A/B 不足以修复 | ❌ NO-GO,需走 Sinkhorn 增强或阈值收紧路径 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 P1 (200 epoch) | ~25 min (P0 135s/200ep 基础上 κ 解冻只 100 ep,稍慢) |
| Stage 2 SID inference | ~30 s |
| Stage 3 T5-small 60M (200 epoch) | ~15 min |
| Stage 4 eval | ~1 min |
| 总计 | **~45 min** (4 阶段串行,GPU 0) |

---

## 5. 风险与缓解

**风险 1**: 即使 Phase A (κ=0 冻结 100 epoch),codebook 仍学不到 100% util
- 缓解: 利用率 ≥ 80% 即视为成功(< 100% 也比当前 0.39% ~ 3.12% 量级好三个数量级)

**风险 2**: Phase B 解冻时 κ_m 反弹到 ±κ_max
- 缓解: lr_theta=1e-5 已远低于 codebook lr=1e-3,gradient 量级被压制,Task #144 已验证此策略有效

**风险 3**: Stage 3 T5-small 60M 在 Phase B codebook 训练下仍 underperform
- 缓解: 准备 fallback 用 T5-base (与 #157 同口径) 作二次评估

---

## 6. 完成度跟踪

- [x] P0 自由 κ 训练 (已确认坍塌,作对照)
- [ ] P1 Stage 1 (Phase A/B 训练) 启动
- [ ] P1 Stage 1 完成 (util ≥ 80% 解冻后稳定)
- [ ] P1 Stage 2 (SID tensor)
- [ ] P1 Stage 3 (T5-small 60M)
- [ ] P1 Stage 4 (R@10 / NDCG 评估)
- [ ] verdict 写入 verdicts/task164_kappa_stereographic_phase_b_result.md
