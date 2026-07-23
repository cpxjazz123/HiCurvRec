# Task #74 — ETEGRec 128d warmup_steps + warm_epoch 重调 (paper 4.1.5 对齐)

> **任务目的**: 按 paper Section 4.1.5 实际训练方式 (cycle 内 ID 先, REC 后; 无 step-level warmup; patience 更大) 重跑 ETEGRec 128d, 验证 R@10 是否追平 paper 0.0624

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #73 paper_exact 失败结论 (R@10=0.0253, vs paper 0.0624, -59% gap). Task #59 已验证 R1 (embedding source) ❌ 否证 (896d fused emb 反而更差).

**用户指示假设 R3**: warmup_steps + early_stop 配置错.

**Paper Section 4.1.5 关键描述 (verbatim)**:
> "The number of epochs per cycle 𝐶 is tuned in {2, 4}. The training process begins with training the item tokenizer for 1 epoch, followed by training the generative recommender for 𝐶-1 epochs. **This process is repeated until convergence based on validation performance.**"
> "The learning rates for the generative recommender and item tokenizer are tuned within the ranges of {5e-3, 3e-3, 1e-3} and {5e-4, 1e-4, 5e-5}, respectively."

**当前 config vs paper**:
| Config | 当前 (Task #73 paper_exact) | Paper |
|--------|------------------------|-------|
| cycle | 2 | 2 or 4 |
| warmup_steps | 8000 (step-level linear warmup lr) | 无 step-level warmup |
| warm_epoch | 10 (固定 10 epoch ID-only) | 1 (cycle 内 ID epoch) |
| early_stop | 15 | "until val convergence" (无具体数值) |
| epochs | 400 | 未指定 |
| lr_rec | 0.005 | {5e-3, 3e-3, 1e-3} |
| lr_id | 0.0001 | {5e-4, 1e-4, 5e-5} |

**假设 R3.1**: 去掉 step-level warmup_steps + 改 warm_epoch=1 → cycle 内 ID/REC 自然交替, 与 paper 一致 → 改善 cycle 训练稳定性
**假设 R3.2**: early_stop=30 (更大 patience) → 让 cycle V-shape val 自然演进不被打断

---

## 2. 实验设计

**变量**:
- warmup_steps: 8000 → **0** (no step-level warmup)
- warm_epoch: 10 → **1** (cycle 内 ID epoch, 与 paper 一致)
- early_stop: 15 → **30** (2x patience)

**保持不变**:
- cycle=2, lr_rec=0.005, lr_id=0.0001
- semantic_emb_path: Musical_Instruments_emb_128.npy (沿用 128d SASRec-only)
- rqvae_path: ./dataset/Musical_Instruments/256-256-256-128.rqvae.pth
- batch_size=128 + gradient_accumulation_steps=4 (effective 512)
- code_length=4, layers=[256, 128], e_dim=128
- seed=2025

**启动命令** (基于 Task #73 paper_exact 脚本, 仅改 config):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec
# 修改 config/musical_instruments.yaml:
#   warmup_steps: 0
#   warm_epoch: 1
#   early_stop: 30
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task74_etegrec_train_paper_cycle.sh
```

---

## 3. 决策触发 (vs paper 0.0624)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| R@10 ≥ 0.0624 | ✅ 完全对齐 paper | 任务完成, 写 verdict 总结 paper-exact 复现成功 |
| 0.05 ≤ R@10 < 0.0624 | 🟡 partial (-20%) | 任务完成, 但记录剩余 gap (可能源自 LR / RQ-VAE 配置) |
| 0.03 ≤ R@10 < 0.05 | ⚠️ 部分改善 | 任务继续, 考虑 cycle=4 试一遍 |
| R@10 < 0.03 | ❌ 无改善 | 切换调查方向, R3 否证, 启动 R4 (paper 上游 code diff) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| ETEGRec 训练 (200 epoch, cycle=2) | ~4-6 h |
| 推断 + R@10 评估 | ~10 min |
| 写 verdict | ~10 min |
| **总计** | **~4-6 h** |

---

## 5. 风险与缓解

**风险 1**: 去掉 warmup_steps=8000 后 lr cosine schedule 直接起步, REC lr=0.005 起步可能让模型震荡
→ **缓解**: 用 lr_id 起始值 5e-5 (paper 最小区间), 不让 ID lr 过大

**风险 2**: cycle=2 + warm_epoch=1 意味着每 epoch 切换 ID/REC, val 在 cycle 内 V-shape 严重
→ **缓解**: early_stop=30 (容忍更长的 val 震荡)

**风险 3**: paper "until val convergence" 没具体 patience, 我们 30 epoch 仍可能不够
→ **缓解**: 监控 val_loss 趋势, 若 epoch 50 仍下降则继续

**风险 4**: 200 epoch 上限 (vs 当前 400) 可能不够收敛
→ **缓解**: 监控 val_R@10 峰值, 若 epoch 150 仍未达 0.05, 提前 verdict 失败

---

## 6. 完成度跟踪

- [ ] Step 1: 修改 config/musical_instruments.yaml (warmup_steps=0, warm_epoch=1, early_stop=30)
- [ ] Step 2: 启动 ETEGRec 训练 (GPU 0, batch_size=128, ~4-6h)
- [ ] Step 3: 监控 val_R@10 趋势 (每 5 epoch 报一次)
- [ ] Step 4: 训练完成 → Stage 4 推断 (~10 min)
- [ ] Step 5: 写 `verdicts/task74_result.md` 含 result: 行
- [ ] Step 6: 更新 §16 + 同步 Task #73 verdict (若成功)