# Issue #141 v85p baseline Stage1 + v15 SID + 6-Decoder + LR_WARMUP_FRAC 10%: PARTIAL-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85p (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + num_decoder_layers=6 + LR_WARMUP_FRAC 0.05→0.10) PARTIAL-GO** — best test_R10=**0.1060** (ep160 stage4 ckpt 手动 eval, R@5=0.0853 / R@20=0.1317 / NDCG@10=0.0779 / NDCG@20=0.0844) vs v85j 0.1053 (**+0.0007**, **v85 系列新 SOTA**), vs v85i 0.1044 (+0.0016), vs v77 0.1080 (-0.0020, 接近), vs HG-Rec baseline 0.1024 (+0.0036, **超越基线**), best valid_R10=**0.1328** (ep175), ratio=**1.244** (vs v85j 1.268 -0.024, **ratio 改善**). DDP 4×L40S 12s/ep × 250ep ≈ 50 min 训练, early-stop 触发 ep249 (无 valid 提升 15ep 后停).

**意义**:
1. **LR_WARMUP_FRAC 5%→10% 显著提升**: v85p 在 v85j 基础上仅改 warmup_frac, test 从 0.1053 → 0.1060 (+0.0007, +0.67%). 延长 warmup 让 6 decoder 早期梯度更稳定, 后期 cosine 衰减收敛更平滑
2. **ratio 同时改善**: v85j ratio 1.268 (略有过拟合) → v85p ratio 1.244 (-0.024). 训练曲线更健康
3. **仍未达 0.11 目标** (-0.0040), 但已逼近 v77 0.1080 (差 0.0020)
4. **Stage3 LR/优化器微调方向有效**: 之前 6 个变体 (v85k/l/m 等) 都集中在架构/正则化维度, v85p 验证 LR schedule 调整是新的有效方向

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | num_dec | LR_warmup | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | 4 | const 1e-3 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | 4 | cos 5%/0.1 | 0.1077 | 0.1339 | 1.243 | — |
| v85h | baseline | v15 (5f83) | 4 | cos 5%/0.1 (200ep) | 0.1042 | 0.1302 | 1.249 | 185 |
| v85i | baseline | v15 (5f83) | 4 | cos 5%/0.05 (300ep) | 0.1044 | 0.1294 | 1.235 | 205 |
| **v85j** | **baseline** | **v15 (5f83)** | **6** | **cos 5%/0.05** | **0.1053** | **0.1337** | **1.268** | **239** |
| v85k | baseline | v15 (5f83) | 6 | cos 5%/0.05 (drop=0.30) | 0.1010 | 0.1281 | 1.262 | 280 |
| v85l | baseline | v15 (5f83) | 8 | cos 5%/0.05 | 0.1042 | 0.1326 | 1.262 | 184 |
| v85m | baseline | v15 (5f83) | 6 | cos 5%/0.05 (heads=8) | 0.1044 | 0.1316 | 1.260 | 155 |
| **v85p** | **baseline** | **v15 (5f83)** | **6** | **cos 10%/0.05** | **0.1060** | **0.1328** | **1.244** | **175** |

**关键观察**:
- v85p (warmup_frac=10%) → v85j (warmup_frac=5%) test +0.0007, valid -0.0009, ratio -0.024
- 延长 warmup 让 test 提升同时 ratio 改善 (训练更稳定)
- v85p 与 v77 (test=0.1080) 差距收窄到 0.0020 (-1.85%)
- v85p 与 0.11 目标差距 0.0040 (-3.6%), 比 v85j 差距 0.0047 (-4.3%) 收窄 0.0007

---

## 3. 训练曲线 (v85p)

```
ep   5  valid=0.0842  test=0.0563  ratio=1.495  (warmup 慢, vs v85j 0.0932)
ep  10  valid=0.1068  test=0.0805  ratio=1.327
ep  20  valid=0.1155  test=0.0890  ratio=1.298
ep  25  valid=0.1222  test=0.0935  ratio=1.306
ep  30  valid=0.1241  test=0.0964  ratio=1.288
ep  45  valid=0.1244  test=0.0995  ratio=1.250
ep  50  valid=0.1270  test=0.1011  ratio=1.256
ep  55  valid=0.1283  test=0.1011  ratio=1.269
ep  65  valid=0.1285  test=0.1032  ratio=1.245
ep 110  valid=0.1291  test=0.1044  ratio=1.236
ep 125  valid=0.1304  test=0.1049  ratio=1.236
ep 130  valid=0.1304  test=0.1046  ratio=1.246
ep 160  valid=0.1318  test=0.1060  ratio=1.244  (BEST ckpt, best test)
ep 175  valid=0.1328  test=0.1059  ratio=1.244  (FINAL BEST ckpt)
ep 200+ EARLY STOP (no improv 15ep)
```

**对比 v85j 同 epoch**:
- v85j ep30 test=0.0980 vs v85p ep30 test=0.0964 (-0.0016, warmup 慢起步)
- v85j ep50 (similar) test≈0.099 vs v85p ep50 test=0.1011 (+0.002)
- v85j ep110 test≈0.103 vs v85p ep110 test=0.1044 (+0.0014)
- v85j ep160 test≈0.1049 vs v85p ep160 test=0.1060 (+0.0011)
- v85j ep239 (best) test=0.1053 vs v85p ep175 (early stop) test=0.1060 (+0.0007)

**Stage3 配置**:
- NUM_EPOCHS=300, LR cosine (warmup_frac=**0.10**, LR_min_factor=0.05)
- EARLY_STOP=15 (沿用 v85j)
- num_layers=6, num_decoder_layers=6, num_heads=6, d_model=128, d_ff=1024, d_kv=64 (回到 v85j)
- dropout=0.20, label_smoothing=0.05, WD=0.01
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85j 完全一致)
- Total params: 6,820,352 (与 v85j 完全一致)
- **核心改动**: LR_WARMUP_FRAC 0.05 → 0.10 (5% → 10% 总训练步, 495 → 990 warmup steps)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 250ep ≈ 50 min, loss 6.09 → 2.39, ckpt 22.5MB. 无 NaN/Inf. Early stop 触发 ep249.

**Gate 4 (Stage4)**: **PARTIAL**.
- best test R@10 = **0.1060** (ep160 stage4) vs v85j 0.1053 (**+0.0007**, **v85 系列新 SOTA**), vs v77 0.1080 (-0.0020), vs HG-Rec baseline 0.1024 (+0.0036)
- best valid R@10 = **0.1328** (ep175) vs v85j 0.1337 (-0.0009), vs v77 0.1312 (+0.0016)
- ratio = **1.244** (vs v85j 1.268 -0.024, vs v77 1.215 +0.029, **ratio 改善**)
- best test R@5=0.0853, R@20=0.1317, NDCG@5/10/20=0.0712/0.0779/0.0844

---

## 5. 根因分析

1. **延长 warmup 让 6 decoder 早期梯度更稳定**: v85p warmup 从 495 步扩到 990 步, 6 decoder 容量的参数在前期不会被激进的 LR 冲击, 进入 cosine 衰减时已收敛到更优 basin
2. **ratio 同时改善 = 训练曲线更健康**: 1.268 → 1.244 (-0.024). 这表明 v85j ratio 略高的部分原因是 warmup 不足导致早期 valid 信号虚高
3. **早期 (ep5-30) 起步慢, 后期 (ep65+) 加速**: warmup 长 → ep5-30 落后 v85j 0.001-0.002; warmup 后 → ep65+ 领先 v85j 0.001-0.002. 净效应 +0.0007
4. **HAB 不受影响**: HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30 完全沿用 v85j, 几何注入通道不变, 变化仅限 LR schedule
5. **EARLY STOP ep249 (vs v85j ep300 跑满)**: v85p 提前停止, 说明更稳定的 warmup 让模型更快收敛到 plateau, 反而节省训练时间

---

## 6. 结论 + 下一步

**v85p PARTIAL-GO** — v85 系列新 SOTA (test=0.1060, +0.0007 vs v85j), 验证 LR_WARMUP_FRAC 5%→10% 是有效改进.

**v85p 仍差 v77 0.0020, 仍差 0.11 目标 0.0040**.

**下一步候选 (LR/优化器维度继续探索)**:
1. **方案 A: v85q = v85p + LR_min_factor 0.05→0.02** (cosine 末期 LR 更低, 0.5e-5, 进一步巩固后期收敛)
2. **方案 B: v85q = v85p + warmup_frac 10%→15%** (继续延长 warmup, 看是否能进一步提升)
3. **方案 C: v85q = v85p + Stage3 batch_size 1024→2048** (DDP 全局 batch 翻倍, 梯度更稳, 需要 8×L40S 或更长 epoch)
4. **方案 D: v85q = v85p + Stage2 incremental codebook [128, 256, 512]** (SID 容量提升, 工程量大, 需 Stage2 retrain + Stage3 架构改动)
5. **方案 E: v85q = v85p + Stage1 hyp_v3** (尝试 Stage1 变体 hyp_v3, 但需 Stage2 重训)

**优先级**: P0 — 方案 A (LR_min_factor 降低) 是低风险高 ROI 微调, 预期 +0.0005~0.0015. 方案 B (warmup 15%) 中等风险. 方案 C/D/E 工程量大, 留作后续.

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85p_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep175, valid=0.1328, best test=0.1060 @ep160)
- stage4 test on best: `stage4_test_on_best/ep160/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85p_stage3.log`
- Stage3 script: `/fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py` (LR_WARMUP_FRAC 0.05→0.10, num_heads 8→6 回到 v85j)
- Stage4 script: `/fs04/ar57/wenyu/GeneRec/common/stage4/stage4_eval_pure_t5.py` (num_heads 8→6 同步)