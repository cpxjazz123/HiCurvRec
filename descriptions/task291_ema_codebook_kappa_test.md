# Task #291 — EMA codebook + κ learnable 测试 (3-way 并行 #2, 2026-07-29)

## 来源

用户 2026-07-29 明示: "三个并行试一下". EMA codebook update (VQ-VAE-2 Razavi et al. 2019) 用 EMA 替代 gradient 更新码本. **可能切断 κ+codebook 反馈循环** (codebook 不通过 gradient 推, 而通过 EMA 平滑, κ 推到极端时码本不会被推走).

## 核心假设

- baseline codebook 通过 ∂L/∂codebook 推到 dominant 位置 → feedback loop → 崩溃
- EMA 用 `codebook_new = decay * codebook_old + (1-decay) * encoder_output_avg`
- EMA 切断了 gradient 反向传到 codebook, 只通过 EMA 平滑更新
- κ 仍然 learnable (跟 baseline 一样), 但 codebook 不再被 κ 直接推走 → 可能 κ 推到极端但码本不动 → L0 健康

## 实验设计

| 阶段 | 内容 | 估计 GPU 时间 |
|------|------|---------------|
| Stage 1 | FreeCurvHRQVAE + EMA codebook update (decay=0.99, 跟 VQ-VAE-2 一致) + κ learnable | 1.5-2 hr (cuda:1) |
| Stage 2 | Sinkhorn 30 iter + 4th-digit dedup | 5 min |
| Stage 3 | T5-mini 200 epoch | 30-40 min |
| Stage 4 | R@10 评估 | 2 min |
| **总计** | | **~2.5 hr** |

## 改动方案 (R11.4 critical, dry-run)

- **改 `HG-Rec/model/hrqvae_free_curv.py`** line 285-360 forward: 在训练时加 EMA codebook update 步骤
- **改 `HG-Rec/train_hrqvae.py`** argparse: 加 `--codebook_ema_decay` (default 0.99)
- **不改 trainer 主体**: 只在 quantizer forward 里加 EMA, optimizer 不动 (codebook 不参与 gradient descent, 但仍参与 EMA)

## 决策点 (R11.3 自主决策透明)

- **decay=0.99**: VQ-VAE-2 默认值, 论文推荐
- **K=64**: 跟 baseline 一致, 避免 K 维度混淆
- **κ learnable**: 跟 baseline 一样, 不冻结, 不 curriculum — 测"EMA 是不是真的能切断 feedback loop"

## 评估指标

- Stage 1 L0/L1/L2 utilization (FSQ → baseline 利用率, 应该跟 #288 baseline L0=73.44% 对比)
- Stage 4 R@10 vs baseline 0.1020
- 关键观察: κ 终态是不是仍然推到极端 (±κ_max=2) 但 L0 仍然健康 → 证明 EMA 切断了 feedback loop
