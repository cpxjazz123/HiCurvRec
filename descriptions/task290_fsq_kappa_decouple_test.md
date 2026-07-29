# Task #290 — FSQ + κ-decouple 测试 (3-way 并行 #1, 2026-07-29)

## 来源

用户 2026-07-29 明示: "三个并行试一下" (FSQ + EMA + Restoration). 选 FSQ 是因为它**完全消除可学码本**, 100% util by construction (Mentzer et al. 2023). 测它能不能跟 κ-decouple 兼容.

## 核心假设

- FSQ 没有 argmin, 没有 Poincaré distance saturation, 推到极端 κ 时不会卡 boundary → 应该 100% L0 by construction
- κ learnable 在 FSQ 架构下意义有限 (FSQ 是 scalar round, 没有码本), 但测一下能不能跟 κ-decouple Phase A/B 兼容
- 预期 R@10: 跟 κ-decouple K=64 中性 (≈ baseline) — FSQ 改造的是 Stage 1, 跟 Stage 3/4 的 T5 训练耦合可能仍受 κ-decouple 跳崖影响

## 实验设计

| 阶段 | 内容 | 估计 GPU 时间 |
|------|------|---------------|
| Stage 1 | FSQQuantizer + κ-decouple Phase A (100 ep κ frozen) + Phase B (100 ep κ unfreeze lr=1e-5) | 1.5-2 hr (cuda:0) |
| Stage 2 | Sinkhorn 30 iter + 4th-digit dedup | 5 min |
| Stage 3 | T5-mini 200 epoch | 30-40 min |
| Stage 4 | R@10 / R@5 / R@20 / NDCG 评估 | 2 min |
| **总计** | | **~2.5 hr** |

## 改动方案 (R11.4 critical, dry-run)

- **新文件**: `HG-Rec/model/fsq_quantizer.py` — FSQ forward (每维独立 round + straight-through) + κ-decouple curriculum
- **CLI flag**: `--quantizer fsq` (新增, 替代 `--quantizer rqvae`)
- **不改 src/**: FreeCurvHRQVAE 等保留
- **唯一 src/ 改动**: `HG-Rec/train_hrqvae.py` 加 `--quantizer fsq` argparse 分支, 指向新 FSQQuantizer

## 决策点 (R11.3 自主决策透明)

- **选 K=64**: 跟 κ-decouple K=64 中性最接近 baseline (避免 K ≥ 128 跳崖混淆 FSQ 效应)
- **FSQ levels = [8,5,5,5]**: 给 1000 effective codes (跟 K=64 baseline 同一量级)
- **保留 κ-decouple**: 测"FSQ + κ-decouple" 联合效应, 不是 FSQ 单测

## 评估指标

- Stage 1 L0/L1/L2 utilization (HRQVAE stage1 monitor)
- Stage 4 R@10 vs baseline 0.1020
- 跟 task144 K=64 κ-decouple R@10=0.1026/0.1017 对比 (FSQ 加成是多少)
