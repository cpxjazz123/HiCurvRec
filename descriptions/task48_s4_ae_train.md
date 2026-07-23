# Task #48 — 战线一 S4 AE 训练 (纯重构目标)

> **任务目的**: 用纯重构 (MSE) 训 embedding, 作为 QMP 矩阵的对照源之一 (与 S1 MCKG margin ranking 对比)
> **完成日期**: 2026-07-20
> **状态**: ✅ 已完成

---

## 1. 背景

承接 Task #99 MCKG margin ranking 训出的 norm 长尾问题. 用 Toy 文本 (Title+Brand+Categories+Price) 训纯 AE, 对照训练目标对 embedding 度量结构的影响.

## 2. 实验设计

Encoder: vocab_size(4096) → 256 → 256 → 64
Decoder: 64 → 256 → 256 → vocab_size
Loss: MSE(x, decoder(encoder(x))), 5000 steps, Adam lr=1e-3.

## 3. 决策触发

| norm stats | 判定 |
|------------|------|
| ρ_max ≈ 1.0 (L2 归一) | ✅ norm 健康, 可作 QMP 对照 |

## 4. 预算

~3 min (5000 steps, batch=512, AE 极小)

## 5. 结果 (回填)

- final_recon = 0.05 (典型 AE 收敛)
- norm: mean=1.000, max=1.000 (L2 归一后)
- **S4 AE 是 QMP 矩阵中 NP@10 最佳 (0.80)**.

## 6. 产物

`scripts/task156_s4_ae_train.py` + `products/task156_s4_ae/{entity_embedding.pt, task156_summary.json}`
