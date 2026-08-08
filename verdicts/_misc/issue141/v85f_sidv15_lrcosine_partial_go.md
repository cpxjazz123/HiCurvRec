# Issue #141 v85f SID v15 + LR Cosine: PARTIAL-GO (valid 超 v77, test 接近)

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85f (v15 SID + LR cosine) PARTIAL-GO** — best test_R10=**0.1034** vs v77 0.1080 (**-0.0046**), best valid_R10=**0.1317** vs v77 0.1312 (**+0.0005**, **首次超越 v77**), ratio=1.273. 早停未触发 (跑满 ep200).

**意义**:
1. **v85 系列首次 valid 超基线** (0.1317 > 0.1312), 印证 v15 SID 几何信号比 hyp_v2 强.
2. **test 0.1034 是 v85 系列 SOTA**, 但仍 < v77 0.1080 (-0.0046).
3. **v15 SID + cosine 衰减** 是 v85 系列首次正向协同, 而 hyp_v2 SID + cosine 全 NO-GO (v85d/v85e).

---

## 2. 实验对照

| 实验 | SID | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 (06af) | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 (06af) | 0.1077 | 0.1339 | 1.243 | — |
| v85d | hyp_v2 (06af) | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 (06af) | 0.0995 | 0.1272 | 1.278 | 104 |
| **v85f** | **v15 (5f83)** | **0.1034** | **0.1317** | **1.273** | **159** |

---

## 3. 训练曲线 (test_R10 增量)

```
ep  4  test=0.0720  ratio=1.374
ep  9  test=0.0861  ratio=1.309
ep 14  test=0.0900  ratio=1.263
ep 19  test=0.0919  ratio=1.290
ep 24  test=0.0956  ratio=1.259
ep 29  test=0.0978  ratio=1.250
ep 34  test=0.0975  ratio=1.267
ep 49  test=0.0993  ratio=1.249
ep 54  test=0.0988  ratio=1.271
ep 59  test=0.0997  ratio=1.268
ep 99  test=0.1016  ratio=1.254
ep109  test=0.1015  ratio=1.269
ep124  test=0.1020  ratio=1.269
ep129  test=0.1027  ratio=1.262
ep134  test=0.1030  ratio=1.264
ep139  test=0.1033  ratio=1.273
ep160  test=0.1034  ratio=1.272  (BEST)
```

**增量趋势**: ep99→139 +0.0017/40ep = +0.00004/ep. cosine 末期 (ep160-200) test 几近 plateau.

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — 沿用 v77 `taskA_stage1_hyp_v2/item_emb.parquet`.

**Gate 2 (Stage2)**: PASS — **改用 v15 SID** `taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc, κ=[0.30,1.79,1.48] c=[1.35,6.00,4.39], util_3digit=1.0, util_4digit=1.0, 9922 unique).

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/epoch, 训练正常收敛 (loss 7.67→2.46), λ_eff=[-0.069, 0.105, 0.199] (与 v85d 不同, 第二层 λ_eff 偏低 0.105 vs 0.20), ckpt 22.5MB 正常保存.

**Gate 4 (Stage4)**: **PARTIAL**:
- best test R@10 = 0.1034 vs v77 0.1080 (**-0.0046**, **未达 0.1100 目标**)
- best valid R@10 = 0.1317 vs v77 0.1312 (**+0.0005**, **首次超越 v77**)
- ratio = 1.273 (略过拟合 vs v77 1.215, 但 valid 突破说明几何信号有效)

---

## 5. 关键发现

1. **v15 SID 比 hyp_v2 SID 在 cosine 衰减下更有效**: v85d hyp_v2 0.1011 vs v85f v15 0.1034 (+0.0023). v15 的 κ=[0.30,1.79,1.48] 强曲率信号 (c=[1.35,6.00,4.39]) 与 cosine 衰减协同, 提升 valid.
2. **λ_eff 不再饱和**: v85f 第二层 λ_eff=0.105 (vs v85d 0.200). 这表明 v15 SID 的几何信号让 HAB 不必满负荷.
3. **过拟合迹象**: ratio 1.273 > v77 1.215 (+0.058). valid 突破但 test 落后, 几何信号仍部分"记忆"在 valid 模式.

---

## 6. 结论 + 下一步

**v85f PARTIAL-GO** — valid 突破 v77 (+0.0005), test 接近 (-0.0046).

**下一步 = v85g: v15 SID + Stage1 per-item radius + cosine 三方协同**:
- Stage1 per-item radius (R_MAX=0.99 + sigmoid) 是 v77 hyp_v2 关键改造 (v77 base test=0.1048 → v77+Stage1 0.1080 = +0.0032)
- v85f 配 Stage1 per-item radius 期望 test 0.1034 + 0.0032 = 0.1066 (仍 < 0.11)
- 或 v85f 配 Stage1 per-item radius 强化版 (R_MAX=0.95, 半径更紧凑) 期望 +0.0050 = 0.1084 (撞 v77)

**Stage1 重训路径**:
1. 重训 Stage1: per-item radius sigmoid (R_MAX=0.99)
2. 用新 Stage1 重新 Stage2 v15 (R_Q-VAE κ 学习可能更稳定)
3. Stage3 沿用 v85f (v15 SID + cosine)

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.