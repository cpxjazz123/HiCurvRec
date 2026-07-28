# Task #167 — κ-Stereographic × T5-base 220M ablation (third capacity point)

> **任务目的**: 复用 Task #164 Phase B κ-Stereographic SID, 训练 T5-base 220M (更大容量, 12 enc + 12 dec, d_model=768), 完成 ablation across capacities 三档 (T5-mini 9.18M / T5-small 60M / T5-base 220M), 验证 κ-Stereographic 是否在不同 T5 容量下都提升模型.
>
> **完成日期**: (in progress)
> **状态**: 🟢 在跑 (GPU 2)

---

## 1. 背景

承接 Task #164 (κ-Stereographic + Phase A/B) + Task #165 (#165 v3 T5-small 60M) + Task #166 (T5-mini 9.18M):

用户 2026-07-25 02:35 追问 "220m also need" → 启动第三档 capacity ablation.

**假设 R1**: κ-Stereographic 是容量无关的 inductive bias, 三个 T5 容量 (9.18M / 60M / 220M) 都能从 κ-Stereographic SID 受益
**假设 R2**: 如果三档 R@10 都 > 普通 SID capacity-matched baseline (Task #161 T5-mini 普通 0.1012 / Task #84 HG-Rec c111 0.1020 / Task #157 T5-base 普通 0.0950), 则 κ-Stereographic 是 common improvement
**假设 R3**: 大模型 (220M) 在小数据集 (Musical_Instruments 9922 items) 上过拟合严重 (Task #157 已证), κ-Stereographic 可能加剧过拟合

---

## 2. 实验设计

**变量**: T5 容量 (60M → 220M, +267%) + SID 类型 (code_default → κ-Stereographic Phase B)
**保持不变**:
- Stage 1 ckpt: Task #164 Phase B (κ-Stereographic, util 90/100/97%)
- Stage 2 SID: Instruments_t5_rqvae_phase_b_kappa_decouple.npy (9922 unique)
- Stage 3 训练: 200 epoch + early_stop=30, lr=1e-4, batch_size=256, seed=42, beam_size=20

**Stage 3 T5-base config** (与 #157 一致):
- num_layers=12, num_decoder_layers=12, d_model=768, d_ff=3072, num_heads=12, d_kv=64 (12×64=768 strict, ~220M params)

**启动命令**: `bash scripts/task167_t5base_kappa_decouple_stage3.sh` (GPU 2)
**Stage 4 auto**: `bash scripts/task167_t5base_kappa_decouple_stage4_eval.sh` (R88 daemon)

---

## 3. 决策触发 (vs baseline + capacity ablation 对照)

| 条件 | test R@10 | 解读 |
|------|-----------|------|
| R@10 > 0.1058 baseline + > #157 T5-base 普通 0.0950 | ✅ κ-Stereographic 帮助大模型 | 三档通用 |
| R@10 ≈ #157 ±10% | ⚠️ κ-Stereographic 对 T5-base 无效 | 容量上限不可救 |
| R@10 < #157 (即 < 0.0950) | ⛔ κ-Stereographic 伤害大模型 | 过拟合加剧 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (T5-base 220M) | ~3-4 hour (类比 #157 ~3 hour) |
| Stage 4 推断 | ~5 min |
| R@10 评估 | ~1 min |
| **总计** | ~3-4 hour |

---

## 5. 风险与缓解

**风险 1**: GPU 2 与其他任务冲突 → 当前空闲 (Task #165 GPU 0 + Task #166 GPU 1 已占)
**风险 2**: T5-base 220M 在小数据集过拟合 → early_stop=30 (vs #157 的 20) 留更多 buffer
**风险 3**: 训练崩溃 → R12 save_strategy 强制每 epoch 保存 + R88 daemon 监控

---

## 6. 完成度跟踪

- [x] scripts/task167_t5base_kappa_decouple_stage3.sh 创建 (chmod +x)
- [x] scripts/task167_t5base_kappa_decouple_stage4_eval.sh 创建 (chmod +x)
- [x] R88 daemon (_STAGE4_TRIGGER.sh) 配置
- [x] products/task167/_TRAINING_PID 写入
- [x] GPU 2 启动 Stage 3 训练
- [ ] Stage 3 训练完成 (200 epoch 或 early_stop)
- [ ] Stage 4 自动 eval 完成
- [ ] verdicts/task167_t5base_kappa_decouple_result.md 写完
- [ ] paper.md §6.2 新增 ablation across capacities 表