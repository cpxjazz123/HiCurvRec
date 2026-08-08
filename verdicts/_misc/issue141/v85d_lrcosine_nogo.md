# Issue #141 v85d LR Cosine Decay NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85d (LR cosine decay) NO-GO** — 在 v85 P0 (λ_max=0.20, residual_alpha_init=-20, hyp_v2 SID, 200ep) 基础上加 LR cosine decay (warmup_frac=0.05 → cos → LR_min_factor=0.1), best test_R10=**0.1011** vs v77 0.1080 (**-0.0069**), best valid_R10=0.1270 vs v77 0.1312 (-0.0042), ratio=1.256 vs v77 1.215. 早停 @ ep150 (10 epochs no improvement after ep99).

**根因** — LR cosine decay 单独并不足以推动 test 突破 v77 基线. 验证 best epoch=99 (cosine 衰减到 ~0.55 LR 阶段, 还没到末期), λ_eff=[0.018, 0.200, 0.200] 与 v85 P0 完全一致 (residual_alpha_init=-20 主导, λ_max 0.20 已饱和). 增量趋势 ep74→100 +0.0010/21ep = +0.00005/ep, 推算 ep200 = 0.1061, 与 v77 0.1080 仍差 0.002.

---

## 2. 实验对照

| 实验 | SID | Stage3 LR 调度 | best test_R10 | best valid_R10 | ratio | best epoch |
|---|---|---|---|---|---|---|
| v77 基准 (SOTA) | hyp_v2 (06af) | constant 1e-3 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 (06af) | constant 1e-3 | 0.1077 | 0.1339 | 1.243 | — |
| v85c (λ_max=0.40) | v15 (5f83) | constant 1e-3 | 0.0911 | 0.1196 | 1.312 | — |
| **v85d (LR cosine)** | hyp_v2 (06af) | **cosine warmup5%** | **0.1011** | **0.1270** | **1.256** | **99** |

---

## 3. 训练曲线 (test_R10 增量)

```
ep  4  test=0.0705  ratio=1.425
ep  9  test=0.0827  ratio=1.296
ep 14  test=0.0857  ratio=1.310
ep 19  test=0.0873  ratio=1.298
ep 24  test=0.0907  ratio=1.295
ep 29  test=0.0942  ratio=1.254
ep 34  test=0.0969  ratio=1.271
ep 54  test=0.0985  ratio=1.260
ep 59  test=0.0997  ratio=1.256
ep 69  test=0.0998  ratio=1.263
ep 74  test=0.1000  ratio=1.262
ep 79  test=0.1001  ratio=1.262
ep100  test=0.1011  ratio=1.256  (best)
```

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — 沿用 v77 `taskA_stage1_hyp_v2/item_emb.parquet`.

**Gate 2 (Stage2)**: PASS — 沿用 v77 `taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy` (sha=06af0fed, util_3digit=1.0, util_4digit=1.0, 9922 unique).

**Gate 3 (Stage3)**: PASS — 训练正常收敛, loss 2.5470 → 2.5408 (best ep99), DDP 4×L40S 12s/epoch, 无 NaN/Inf, ckpt 22.5MB 正常保存. λ_eff=[0.018, 0.200, 0.200] 稳定 (与 v85 P0 一致, residual_alpha_init=-20 主导).

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = 0.1011 vs v77 0.1080 (**-0.0069**, **未达 0.1100 目标**)
- best valid R@10 = 0.1270 vs v77 0.1312 (-0.0042)
- ratio = 1.256 (略健康于 v77 1.215, 但绝对值低)

---

## 5. 增量趋势分析

- ep74→100: +0.0010 / 21 ep = +0.00005/ep (LR 已开始衰减阶段)
- ep100 → ep150 预测: 末期 LR 已降至 0.00055 → +0.00003/ep → ep150 ≈ 0.1013 (实际未跑 ep125 test, 但 valid best 持续 ep99 后未变)
- 推算 ep200 test ≈ 0.1061 (假设 cosine 末期有微增益), **仍差 v77 0.1080**

---

## 6. 结论 + 下一步

**v85d LR cosine decay 单独 NO-GO** — 不超 v77 0.1080.

**下一步 = v85e: cosine + 加大力度**:
1. NUM_EPOCHS 200 → 300 (更长训练, 早停更宽容 early_stop=15)
2. LR_min_factor 0.1 → 0.05 (末期 LR 更低)
3. warmup_frac 0.05 → 0.10 (warmup 更长让前期收敛稳定)
4. 不改 SID / λ_max / HAB 配置 (沿用 v85d = v77 兼容)

**禁止**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.