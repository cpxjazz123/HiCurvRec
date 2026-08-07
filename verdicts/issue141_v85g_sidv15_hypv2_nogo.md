# Issue #141 v85g hyp_v2 Stage1 + v15 SID + Cosine: NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85g (hyp_v2 Stage1 + v15 Stage2 SID + LR cosine) NO-GO** — best test_R10=**0.1012** (ep150 stage4) vs v77 0.1080 (**-0.0068**), best valid_R10=**0.1286** (ep185) vs v77 0.1312 (-0.0026), ratio=1.255 (vs v77 1.215). DDP 4×L40S 12s/ep × 200ep = ~40 min 训练, 早停未触发 (跑满 ep200), best ckpt = ep185 valid=0.1286.

**根因** — hyp_v2 Stage1 (per-item radius sigmoid) + v15 SID (κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39]) 几何信号不协同. Stage1 输出半径化 vs v15 SID 强曲率期望冲突, 导致 test 落后. **有效 SID 信号是 v15 (5f8331cc)**, 但 Stage1 路径必须配 hyp_v2 (per-item radius) 才能上 0.1080+. v85g 同时切两个变量无法协同.

---

## 2. 实验对照 (Issue #141 v85 系列)

| 实验 | Stage1 | SID | LR 调度 | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | const 1e-3 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | const 1e-3 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | const 1e-3 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | cos 5%/0.1 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | cos 10%/0.05 + ES15 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 0.1034 | **0.1317** | 1.273 | 159 |
| **v85g** | **hyp_v2** | **v15 (5f83)** | **cos 5%/0.1** | **0.1012** | **0.1286** | **1.255** | **150** |

---

## 3. 训练曲线 (v85g)

```
ep   4  valid=0.1022  test=0.0710  ratio=1.439
ep  19  valid=0.1158  test=0.0871  ratio=1.329
ep  39  valid=0.1219  test=0.0962  ratio=1.267
ep  84  valid=0.1230  test=0.0989  ratio=1.243
ep  94  valid=0.1246  test=0.0991  ratio=1.258
ep 114  valid=0.1254  test=0.1000  ratio=1.254
ep 144  valid=0.1267  test=0.1007  ratio=1.258
ep 149  valid=0.1270  test=0.1012  ratio=1.255  (best test ep150 stage4)
ep 150  valid=0.1270  test=0.1012  ratio=1.255
ep 185  valid=0.1286  test=0.1011  ratio=1.272  (BEST valid)
ep 190  valid=0.1286  test=0.1011  ratio=1.272
ep 200  valid=0.1275  test=?       (no eval, ES reset)
```

**训练轨迹**: ep4-149 test 持续增长 (0.0710→0.1012, +0.0302 / 145ep), ep150-200 进入 plateau (~0.1011-0.1012, 几乎无增量). **cosine 末期 (ep150-200) 无突破**, 与 v85d 类似.

**Stage3 配置**:
- WD=0.01, dropout=0.20, label_smoothing=0.05
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- LR cosine (warmup_frac=0.05, LR_min_factor=0.1)
- 200 epochs, EARLY_STOP=10

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **hyp_v2 Stage1** (per-item radius sigmoid, R_MAX=0.99) `taskA_stage1_hyp_v2/item_emb.parquet`.

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** `taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc, κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39], util_3digit=1.0, util_4digit=1.0, 9922 unique). final_cs=[1.5752, 1.5918, 1.5923], Dbar=[0.3110, 0.1980, 0.1621].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 200ep ≈ 40 min, loss 7.67 → 2.47, ckpt 22.5MB 正常保存, λ_eff 稳定 (residual_alpha_init=-20 主导). 无 NaN/Inf.

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = **0.1012** vs v77 0.1080 (**-0.0068**, 未达 0.1100 目标)
- best valid R@10 = **0.1286** vs v77 0.1312 (-0.0026, vs v85f 0.1317 -0.0031)
- ratio = 1.255 (vs v77 1.215)

---

## 5. 根因分析

1. **hyp_v2 Stage1 + v15 SID 不协同**: v77 SOTA 用 hyp_v2 Stage1 + hyp_v2 SID (06af0fed, c=1.0 flat) 路径. v85f 改用 hyp_v2 Stage1 + v15 SID 测试 0.1034 (vs v77 -0.0046), 部分协同. 但 v85g 想叠加 cosine 进一步突破, 结果测试仅 0.1012 (-0.0068 vs v77). hyp_v2 Stage1 的 per-item radius 信号是"径向聚集", 而 v15 SID 的 κ=[0.30, 1.79, 1.48] 高曲率期望"切向展开", 二者几何信号冲突.
2. **cosine 末期饱和**: ep150-200 test 几乎 plateau, cosine LR 衰减到 ~5e-5 不再有有效信号. 与 v85d 同样结论: 单纯 LR 调度无法突破.
3. **valid/test 失衡**: ratio 1.255 > v77 1.215 (+0.040), valid 信号略强于 test, 部分过拟合.

---

## 6. 结论 + 下一步

**v85g NO-GO** — hyp_v2 Stage1 + v15 SID 组合不协同, cosine 调度无法挽救.

**v85 系列全部完成 (P0/v85c/d/e/f/g = 7 个变体), 最高仍是 v85f 0.1034 / v77 0.1080**.

**下一步 = v85h: 换路径 — Stage1 baseline (无 per-item radius) + v15 SID + cosine**:
- Stage1 baseline (sha=1a42341f, Euclidean 无 per-item radius) 提供纯净欧氏空间
- v15 SID 强曲率期望 vs 欧氏 Stage1 可能产生新协同
- 预期 test: 0.1034 (v85f) ~ 0.1066 (cosine 衰减增益 ~0.003)

**Stage1 切回路径**: 直接重用 baseline Stage1 (无需重训). Stage2 沿用 v15 SID. Stage3 沿用 v85f cosine + λ_max=0.20 配置.

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.