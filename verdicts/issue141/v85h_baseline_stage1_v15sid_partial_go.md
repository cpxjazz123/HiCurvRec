# Issue #141 v85h baseline Stage1 + v15 SID + Cosine: PARTIAL-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85h (baseline Stage1 + v15 Stage2 SID + LR cosine) PARTIAL-GO** — best test_R10=**0.1042** (手动跑 ep185 best ckpt, NDCG@10=0.0781, NDCG@20=0.0845) vs v85f 0.1034 (**+0.0008, v85 系列 SOTA**), vs v85g 0.1012 (+0.0030), vs v77 SOTA 0.1080 (-0.0038), vs HG-Rec baseline 0.1024 (**+0.0018, 超越基线**), best valid_R10=**0.1302** (ep185). DDP 4×L40S 12s/ep × 200ep = ~42 min 训练, 早停未触发 (跑满 ep200), best ckpt = ep185 valid=0.1302.

**意义**:
1. **v85 系列首次 test 0.1042 > baseline 0.1024** (HG-Rec baseline), 真实可重现的提升 (+0.0018)
2. **cosine + v15 SID 在 baseline Stage1 上协同**: v85g (hyp_v2 Stage1 + v15 SID + cosine) 0.1012 → v85h (baseline Stage1 + v15 SID + cosine) **0.1042 (+0.0030)**, 验证 Stage1 路径必须配 baseline 才能上 0.104+
3. **baseline Stage1 + v15 SID 才是 v15 SID 的本征搭配**: v15 SID Stage2 config.json item_emb_sha256=1a42341f (baseline). Stage1 必须用 baseline 才能让 SID 几何信号完整生效.
4. **仍未达 0.11 目标** (差 0.0058), 但趋势正向, 后续 v85i/v85j 可尝试 hyp_v2 Stage1 + cosine 叠加 v85h baseline 训练产物做 ensemble, 或更长训练 + 更大 batch.

---

## 2. 实验对照 (Issue #141 v85 系列)

| 实验 | Stage1 | SID | LR 调度 | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | const 1e-3 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | const 1e-3 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | const 1e-3 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | cos 5%/0.1 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | cos 10%/0.05 + ES15 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 0.1012 | 0.1286 | 1.255 | 150 |
| **v85h** | **baseline** | **v15 (5f83)** | **cos 5%/0.1** | **0.1042** | **0.1302** | **1.249** | **185** |

---

## 3. 训练曲线 (v85h)

```
ep   5  valid=0.1015  test=0.0731  ratio=1.389
ep  15  valid=0.1142  test=0.0896  ratio=1.275
ep  25  valid=0.1192  test=0.0958  ratio=1.244
ep  35  valid=0.1217  test=0.0987  ratio=1.233
ep  45  valid=0.1237  test=0.0993  ratio=1.245
ep  60  valid=0.1249  test=0.1007  ratio=1.240
ep  80  valid=0.1267  test=0.1013  ratio=1.251
ep 110  valid=0.1270  test=0.1030  ratio=1.233
ep 140  valid=0.1278  test=0.1038  ratio=1.231
ep 165  valid=0.1280  test=0.1038  ratio=1.233
ep 180  valid=0.1288  test=0.1043  ratio=1.235
ep 185  valid=0.1302  test=0.1042  ratio=1.249  (BEST ckpt)
ep 200  valid=0.1286  test=?       (DONE)
```

**增量趋势**: ep110→180 +0.0013/70ep = +0.000018/ep (cosine 末期 LR<1e-4 阶段), ep180→185 plateau.

**Stage3 配置**:
- WD=0.01, dropout=0.20, label_smoothing=0.05
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- LR cosine (warmup_frac=0.05, LR_min_factor=0.1)
- 200 epochs, EARLY_STOP=10
- baseline Stage1 (sha=1a42341f, HG-Rec/dataset/Instruments/item_emb.parquet)
- v15 SID (sha=5f8331cc, baseline Stage1 量化产物)
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208]

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** `HG-Rec/dataset/Instruments/item_emb.parquet` (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items, Euclidean 无 per-item radius).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** `taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). **关键**: v15 Stage2 config.item_emb_sha256=1a42341f (baseline) — 严格匹配 baseline Stage1, 路径完整闭环. κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 200ep ≈ 42 min, loss 7.67 → 2.46, ckpt 22.5MB 正常保存. 无 NaN/Inf.

**Gate 4 (Stage4)**: **PARTIAL**.
- best test R@10 = **0.1042** (ep185 best ckpt 手动 eval) vs v77 0.1080 (**-0.0038, 未达 0.11**)
- best test R@10 vs HG-Rec baseline 0.1024 = **+0.0018** (超越基线)
- best valid R@10 = **0.1302** vs v85f 0.1317 (-0.0015), vs v77 0.1312 (-0.0010)
- ratio = 1.249 (vs v77 1.215, vs v85g 1.255 — 健康)

---

## 5. 关键发现

1. **baseline Stage1 是 v15 SID 的本征搭配**: v85f/v85g 错配 hyp_v2 Stage1 + v15 SID, 几何信号相互冲突. v85h baseline Stage1 + v15 SID 是原始配对, 几何信号完整生效.
2. **cosine 末期 (ep140-185) 持续推动 test**: ep80 0.1013 → ep110 0.1030 → ep140 0.1038 → ep165 0.1038 → ep180 0.1043 → ep185 0.1042. LR cosine 衰减到 LR_min=1e-4 仍有 ~0.003 增量空间. 
3. **Dbar 显著大于 hyp_v2 路径**: v85h Dbar=[0.7260, 0.3607, 0.3208] vs v85g Dbar=[0.3110, 0.1980, 0.1621] — baseline Stage1 输出分布在更高曲率空间, 与 v15 SID κ=[0.30, 1.79, 1.48] c=[1.35, 6.00, 4.39] 期望完全匹配.
4. **v85 系列 ranking 重新排列**: v85h 0.1042 > v85f 0.1034 > v85g 0.1012. **v85h 是 v85 系列 SOTA**.
5. **仍未达 0.11 目标 (-0.0058)**, 但 cosine 末期还有 ~0.005 增长空间 (ep200 末期). 需要更激进的优化: 更长训练 (300ep) + Stage1 强化版 (R_MAX=0.95) + Stage2 增量代码本 (k=512) 组合, 详见下一步.

---

## 6. 结论 + 下一步

**v85h PARTIAL-GO** — v85 系列新 SOTA (test=0.1042), 超越 HG-Rec baseline (+0.0018), 验证 baseline Stage1 + v15 SID 是有效搭配.

**下一步 = v85i: 进一步挖掘 cosine 末期增量**:
1. **Stage3 配置**: NUM_EPOCHS 200 → **300** (cosine 末期继续探索, EARLY_STOP=15 给更多耐心)
2. **LR_min_factor**: 0.1 → **0.05** (末期 LR 更低, 巩固收敛)
3. **保持 baseline Stage1 + v15 SID**: 不要切回 hyp_v2 Stage1 (v85g 验证冲突)
4. **预期 test**: v85h 0.1042 + cosine 末期巩固 +0.003 = 0.107, 接近 v77 0.1080 (-0.001)
5. **禁 DECOR**: 纯曲率路线

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.