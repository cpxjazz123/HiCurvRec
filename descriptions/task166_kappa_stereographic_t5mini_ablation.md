# Task #166 — κ-Stereographic × T5-mini 9.18M ablation (across capacities)

> **任务目的**: 复用 Task #164 Phase B κ-Stereographic SID, 训练 T5-mini 9.18M (更小容量), 验证 κ-Stereographic 是否在不同 T5 容量下都提升模型.
>
> **完成日期**: (in progress)
> **状态**: 🟢 在跑

---

## 1. 背景

承接 Task #164 (κ-Stereographic + Phase A/B κ-decouple 修复 codebook collapse) + Task #165 (T5-small 60M 200 epoch 长训):
- κ-Stereographic SID 在 T5-small 60M 上 val R@10=0.1177 (12 epoch early-kill, +11.2% over baseline)
- 测试 R@10 = 0.0964 (-8.9% vs baseline, val/test gap ≈18%)

用户 2026-07-25 02:30 提问: "is κ-Stereographic a common way to improve the model in different T5?"

**假设 R1**: κ-Stereographic 是容量无关的 inductive bias, 小 T5 (T5-mini 9.18M) 也能从 κ-Stereographic SID 受益
**假设 R2**: 如果 T5-mini R@10 > T5-mini 普通 SID R@10=0.1012 (+3.5%), 则 κ-Stereographic 是 common improvement
**假设 R3**: 如果 T5-mini R@10 ≤ 普通 SID baseline, 则 κ-Stereographic 对小模型没有帮助 (T5 容量上限才生效)

---

## 2. 实验设计

**变量**: T5 容量 (60M → 9.18M, -85% 参数) + SID 类型 (code_default → κ-Stereographic Phase B)
**保持不变**:
- Stage 1 ckpt: Task #164 Phase B (κ-Stereographic, util 90/100/97%)
- Stage 2 SID: Instruments_t5_rqvae_phase_b_kappa_decouple.npy (9922 unique)
- Stage 3 训练: 200 epoch + early_stop=20, lr=1e-4, batch_size=256, seed=42, beam_size=20
- Stage 4 eval: T5-mini config + test.parquet

**Stage 3 T5-mini config** (与 #161 一致, 4 enc + 4 dec, d_model=256, d_ff=1024, 4 heads × d_kv=64, ~8.5M params):
- d_model=256, num_layers=4, num_decoder_layers=4, d_ff=1024, num_heads=4, d_kv=64

**启动命令**: `bash scripts/task166_t5mini_kappa_decouple_stage3.sh` (GPU 1)
**Stage 4 auto**: `bash scripts/task166_t5mini_kappa_decouple_stage4_eval.sh` (R88 daemon)

---

## 3. 决策触发 (vs baseline + ablation 对照)

| 条件 | test R@10 | 解读 |
|------|-----------|------|
| R@10 > 0.1058 baseline + R@10 > 0.1012 (#161) | ✅ κ-Stereographic common improvement | 跨容量有效 |
| R@10 ≈ 0.1012 (#161) ±5% | ⚠️ κ-Stereographic 对 T5-mini 无效 | 需更大容量才生效 |
| R@10 < 0.1012 (#161) | ⛔ κ-Stereographic 伤害小模型 | 反向 evidence |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (T5-mini 9.18M) | ~20-30 min (类比 #161 31min) |
| Stage 4 推断 | ~3 min |
| R@10 评估 | ~1 min |
| **总计** | ~30 min |

---

## 5. 风险与缓解

**风险 1**: GPU 1 与其他任务冲突 → nvidia-smi 当前 GPU 1 空闲 (Task #165 v3 在 GPU 0)
**风险 2**: T5-mini 容量不足 → 即使训 200 epoch 也可能 R@10 < #161, 但这本身是 evidence
**风险 3**: val/test gap 仍存 → early_stop=20 + 充分训练应能收窄

---

## 6. 完成度跟踪

- [x] scripts/task166_t5mini_kappa_decouple_stage3.sh 创建 (chmod +x)
- [x] scripts/task166_t5mini_kappa_decouple_stage4_eval.sh 创建 (chmod +x)
- [x] R88 daemon (_STAGE4_TRIGGER.sh) 配置
- [x] products/task166/_TRAINING_PID 写入
- [x] GPU 1 启动 Stage 3 训练
- [ ] Stage 3 训练完成 (200 epoch 或 early_stop)
- [ ] Stage 4 自动 eval 完成
- [ ] verdicts/task166_t5mini_kappa_decouple_result.md 写完
- [ ] paper.md §6.2/§5.7 更新 ablation 表
