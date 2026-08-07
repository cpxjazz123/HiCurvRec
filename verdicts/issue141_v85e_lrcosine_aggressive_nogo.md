# Issue #141 v85e LR Cosine Aggressive NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85e (LR cosine 激进版) NO-GO** — 在 v85d (warmup_frac=0.05, LR_min_factor=0.1) 基础上加激进化 (warmup_frac=0.10, LR_min_factor=0.05) + NUM_EPOCHS 200→300 + EARLY_STOP 10→15, best test_R10=**0.0995** vs v77 0.1080 (**-0.0085**), best valid_R10=0.1272 (新高, vs v77 0.1312 -0.0040), ratio=1.278. 早停 @ ep180 (15 epochs no improvement after ep104 best).

**根因** — 早停机制 15 epochs 太严格, valid 在 ep104=0.1272 之后 75 epochs 持续震荡 0.1254-0.1270 (< 1e-4 阈值). cosine 末期衰减 (ep250-300) 没机会跑到 (因 EARLY_STOP=15 提前终止). 即使跑完 ep300, 趋势预测 test_R10 ≈ 0.10-0.105, **仍未达 0.11 目标**.

---

## 2. 实验对照

| 实验 | SID | NUM_EPOCHS | warmup | LR_min | early_stop | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 (06af) | 200 | none | 1e-3 | 50 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 (06af) | 200 | none | 1e-3 | 10 | 0.1077 | 0.1339 | 1.243 | — |
| v85d | hyp_v2 (06af) | 200 | 0.05 | 0.1 | 10 | 0.1011 | 0.1270 | 1.256 | 99 |
| **v85e** | hyp_v2 (06af) | 300 | **0.10** | **0.05** | **15** | **0.0995** | **0.1272** | **1.278** | **104** |

---

## 3. 训练曲线 (valid + test)

```
valid 趋势:
  ep  5  valid=0.0898
  ep 50  valid=0.1236
  ep 95  valid=0.1262
  ep105  valid=0.1272  (BEST, test=0.0995)
  ep130  valid=0.1254  (震荡)
  ep160  valid=0.1257
  ep175  valid=0.1270  (接近但未破 0.1272)
  ep180  valid=0.1264  (early stop)
```

**关键观察**: ep104-180 期间 valid 始终在 0.1254-0.1270 之间震荡, 始终 < 0.1272. EARLY_STOP=15 在 ep180 触发, 但 ep175=0.1270 已经接近 best. 后期 cosine 衰减 (LR<5e-4) 不再有有效改进.

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — 沿用 v77.

**Gate 2 (Stage2)**: PASS — 沿用 v77 `taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy` (sha=06af0fed).

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/epoch, 训练正常收敛 (loss 7.67→2.50), λ_eff=[-0.021, 0.200, 0.200] 稳定, 无 NaN/Inf, ckpt 22.5MB 正常保存.

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = 0.0995 vs v77 0.1080 (**-0.0085**, **未达 0.1100 目标**)
- best valid R@10 = 0.1272 vs v77 0.1312 (-0.0040, 新高)
- ratio = 1.278 (略过拟合 vs v77 1.215)

---

## 5. 根因分析

1. **EARLY_STOP=15 太严格**: valid 震荡幅度 > 1e-4 阈值, 但实际 late stage cosine 衰减没机会探索.
2. **warmup_frac=0.10 过长**: ep1-30 LR 升得慢, valid 前期低于 v85d 0.001-0.005. 后期 cosine 衰减虽更陡, 但因 EARLY_STOP 触发早, 没跑到 ep250+.
3. **λ_eff 主导**: 与 v85 P0/v85d 一致 [0.018-0.020, 0.200, 0.200], residual_alpha_init=-20 起决定作用. λ_max 0.20 已饱和, 任何 λ 调整无实质影响.

---

## 6. 结论 + 下一步

**v85e NO-GO** — 单一 LR cosine 调优已穷尽 (5 个变体: v85 P0 / v85b / v85c / v85d / v85e), 均无法超 v77 0.1080.

**下一步 = v85f 换 SID (v15 替代 hyp_v2)**:
- v85 系列全部用 hyp_v2 SID (06af0fed), 同样架构配 v15 SID (5f8331cc) 还没测过
- v15 SID 在 Issue #76 habcfix 中端到端 test=0.1063 (差 v77 -0.0017), 但配 v85e LR cosine + λ_max=0.20 完整组合未测
- 预期: v85 SID 系列与 v15 SID 配 cosine 可能产生协同 (v15 的 κ=[0.30,1.79,1.48] c=[1.35,6.00,4.39] 几何信号更强)

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.