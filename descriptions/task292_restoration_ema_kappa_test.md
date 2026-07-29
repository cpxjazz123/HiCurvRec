# Task #292 — Restoration (EMA + dead code revival) + κ 测试 (3-way 并行 #3, 2026-07-29)

## 来源

用户 2026-07-29 明示: "三个并行试一下". Restoration (Yu et al. 2022) = EMA + 自适应 dead code revival. 比单纯 EMA 更复杂, 应该比 EMA 更稳定.

## 核心假设

- EMA 切断了 codebook gradient 更新 (Task #291 测)
- 但如果 codebook 已经死 (L0 < 90%), EMA 不能复活死码字 (死码字没有 data 平均)
- Restoration 加自适应 dead code revival: 当 codeword usage < threshold, 用 batch 中随机 latent 替换 → 强制保持 L0 高
- 这是 EMA + dead_revive 的组合, 应该**最稳定的 codebook 更新范式**

## 实验设计

| 阶段 | 内容 | 估计 GPU 时间 |
|------|------|---------------|
| Stage 1 | FreeCurvHRQVAE + EMA (decay=0.99) + dead code revival (threshold=1, replace_ratio=0.1, frequency=5 epoch) + κ learnable | 1.5-2 hr (cuda:2) |
| Stage 2 | Sinkhorn 30 iter + 4th-digit dedup | 5 min |
| Stage 3 | T5-mini 200 epoch | 30-40 min |
| Stage 4 | R@10 评估 | 2 min |
| **总计** | | **~2.5 hr** |

## 改动方案 (R11.4 critical, dry-run)

- **改 `HG-Rec/model/hrqvae_free_curv.py`** line 285-360 forward: EMA update + dead_code_reset 修复 (目前 hrqvae_trainer.py:271 latent_gravy=empty 让 hook no-op, 需要传真 batch_data)
- **改 `HG-Rec/train_hrqvae.py`** argparse: 加 `--codebook_ema_decay 0.99` + `--dead_revive_threshold 1` + `--dead_revive_ratio 0.1`
- **跟 Task #283 区别**: Task #283 dead_revive 但没有 EMA, hook no-op 锁死 baseline L0=70.3%. Task #292 修复 hook (传真 batch_data) + 加 EMA 双重机制

## 决策点 (R11.3 自主决策透明)

- **decay=0.99, threshold=1, ratio=0.1**: VQ-VAE-2 + Restoration 论文默认
- **K=64**: 跟 #291 同一 K, 跨实验可比
- **κ learnable**: 跟 baseline 一样, 不冻结 — 测 Restoration 是否真的比单纯 EMA 更稳定

## 评估指标

- Stage 1 L0/L1/L2 utilization (应该 ≥ baseline 73.44%, 因为 dead_revive 真 init point 让码字复用)
- Stage 4 R@10 vs baseline 0.1020
- 关键对比: Task #291 vs Task #292 L0/R@10 差异 → Restoration 是否比 EMA 显著更好
