# Task #168 — κ-Stereographic × T5-5.5M ablation (4th capacity point)

> **任务目的**: 复用 Task #164 Phase B κ-Stereographic SID, 训练更小 T5-5.5M (5+5 layers, d_model=192, d_ff=1024, 3 heads × d_kv=64), 完成 ablation across capacities 四档 (T5-5.5M / T5-mini 9.18M / T5-small 60M / T5-base 220M), 验证 κ-Stereographic 是否在非常小模型上也保持 improvement.
>
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #164 (κ-Stereographic + Phase A/B) + Task #165 (T5-small 60M) + Task #166 (T5-mini 9.18M) + Task #167 (T5-base 220M):

用户 2026-07-25 03:08 追问 "5.5m also need to run with same way" → 启动第四档 capacity ablation (向下探索极小模型).

**假设 R1**: κ-Stereographic 是容量无关的 inductive bias, 极小 T5 (5.5M) 也能从 κ-Stereographic SID 受益
**假设 R2**: 如果 4 档 R@10 都 > 普通 SID capacity-matched baseline, 则 κ-Stereographic 是 common improvement (跟 capacity 无关)
**假设 R3**: 极小模型 (5.5M) 在小数据集 (Musical_Instruments 9922 items) 上可能欠拟合, κ-Stereographic 几何 inductive bias 可能仍有效 (因为几何信号来自数据本身而非模型容量)

---

## 2. 实验设计

**变量**: T5 容量 (T5-mini 9.18M → T5-5.5M, -40% params, 更小)
**保持不变**:
- Stage 1 ckpt: Task #164 Phase B (κ-Stereographic, util 90/100/97%)
- Stage 2 SID: Instruments_t5_rqvae_phase_b_kappa_decouple.npy (9922 unique)
- Stage 3 训练: 200 epoch + early_stop=20, lr=1e-4, batch_size=256, seed=42, beam_size=20

**Stage 3 T5-5.5M config**:
- num_layers=5, num_decoder_layers=5 (比 T5-mini 多 1 层, 但更窄)
- d_model=192, d_ff=1024 (比 T5-mini d_model=256 小 25%, d_ff 持平)
- num_heads=3, d_kv=64 (3×64=192 strict, 估算 ~5.5M params)
- 比 T5-mini 9.18M 小约 40%, 但保持 5 层深度 → 容量瓶颈在 width 而非 depth

**启动命令**: `bash scripts/task168_t5_5p5m_kappa_decouple_stage3.sh` (GPU 3, R7: 空闲)
**Stage 4 auto**: `bash scripts/task168_t5_5p5m_kappa_decouple_stage4_eval.sh` (R88 daemon)

---

## 3. 决策触发 (vs baseline + 4 档 capacity ablation 对照)

| 条件 | test R@10 | 解读 |
|------|-----------|------|
| R@10 > 0.1058 baseline + 4 档 val 都 ≥ baseline | ✅ κ-Stereographic 容量无关 | 4 档都 common improvement |
| R@10 < 0.1058 baseline | ⚠️ 极小模型欠拟合 | κ-Stereographic 几何信号在小模型上不够 |
| R@10 ≈ 0.1058 ±5% | ⚠️ 持平 baseline | κ-Stereographic 对极小模型 neutral |

**对照表 (四档 capacity-matched control)**:

| 容量 | 普通 SID baseline | κ-Stereographic SID (本次) |
|------|-------------------|----------------------------|
| T5-5.5M (#168) | 无现成 baseline, 需估 | Task #168 |
| T5-mini 9.18M | #161 = 0.1012 | Task #166 |
| T5-small 60M | #84 c111 = 0.1020 | Task #165 v3 |
| T5-base 220M | #157 = 0.0950 | Task #167 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (T5-5.5M) | ~30-45 min (类比 #161 ~1 hour 减半, 因为 params -40%) |
| Stage 4 推断 | ~3 min |
| R@10 评估 | ~1 min |
| **总计** | ~35-50 min |

---

## 5. 风险与缓解

**风险 1**: GPU 3 跟其他任务冲突 → 当前空闲 (Task #165 GPU 0 / Task #167 GPU 2 已占, GPU 1 跑 #166 Stage 4)
**风险 2**: 极小模型欠拟合 → early_stop=20 (跟 #161 一致) 留足够 buffer
**风险 3**: 训练崩溃 → R12 save_strategy 强制每 epoch 保存 + R88 daemon 监控
**风险 4**: T5-5.5M 在 small batch 上 GPU 利用率低 → 估算 ~30-45 min 已包含低利用率因素

---

## 6. R11.3 决策记录

- **决策 1**: config 选择 5+5 layers (而非 4+4) + d_model=192 (而非 256), 平衡 width/depth → 估算 ~5.5M params. 备选 4+4 layers d_model=224 d_ff=896 (估算 ~5.0M, 偏小). 选定理由: 5+5 layers 略增加深度有助于学习抽象语义, d_model=192 是 3 heads × d_kv=64 的自然选择.
- **决策 2**: GPU 3 启动 (R7: 完全空闲), 不抢 GPU 0/1/2 已用卡.

---

## 7. 完成度跟踪

- [x] R11.3 决策 (config 选择)
- [x] scripts/task168_t5_5p5m_kappa_decouple_stage3.sh 创建 (chmod +x)
- [x] scripts/task168_t5_5p5m_kappa_decouple_stage4_eval.sh 创建 (chmod +x)
- [x] R88 daemon 配置
- [x] products/task168/_TRAINING_PID 写入
- [x] GPU 3 启动 Stage 3 训练
- [ ] Stage 3 训练完成 (200 epoch 或 early_stop)
- [ ] Stage 4 自动 eval 完成
- [ ] verdicts/task168_t5_5p5m_kappa_metrics.json 落盘
- [ ] verdicts/task168_t5_5p5m_kappa_decouple_result.md 写完
- [ ] paper.md §6.2 ablation across capacities 四档表 更新
- [ ] R8 §16 cleanup