# Issue #141 v85k baseline Stage1 + v15 SID + 6-Decoder + dropout 0.30: NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85k (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + num_decoder_layers=6 + dropout 0.30) NO-GO** — best test_R10=**0.1010** (ep280 stage4 ckpt 手动 eval) vs v85j 0.1053 (**-0.0043**, 退化), vs v85i 0.1044 (-0.0034), vs HG-Rec baseline 0.1024 (-0.0014), best valid_R10=**0.1281** (ep289), ratio=1.262 (vs v85j 1.268 -0.006). DDP 4×L40S 12s/ep × 300ep ≈ 60 min 训练, 跑满 ep300.

**根因** — dropout 0.20 → 0.30 **过度正则化**, 抑制了 6 decoder layers 的容量优势. v85j (dropout 0.20) → v85k (dropout 0.30) test -0.0043 (从 0.1053 → 0.1010). 假设 "ratio 1.268 略高 → dropout 加压 ratio → test 提升" 被证伪: 高 dropout 同时压制 valid/test 信号, 没有任何相对提升.

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | num_dec | dropout | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | 4 | 0.20 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | 4 | 0.20 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | 4 | 0.20 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | 4 | 0.20 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | 4 | 0.20 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | 4 | 0.20 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | 4 | 0.20 | 0.1012 | 0.1286 | 1.255 | 150 |
| v85h | baseline | v15 (5f83) | 4 | 0.20 | 0.1042 | 0.1302 | 1.249 | 185 |
| v85i | baseline | v15 (5f83) | 4 | 0.20 | 0.1044 | 0.1294 | 1.235 | 205 |
| **v85j** | **baseline** | **v15 (5f83)** | **6** | **0.20** | **0.1053** | **0.1337** | **1.268** | **239** |
| **v85k** | **baseline** | **v15 (5f83)** | **6** | **0.30** | **0.1010** | **0.1281** | **1.262** | **280** |

**关键观察**:
- v85j (dropout 0.20) → v85k (dropout 0.30) test -0.0043, valid -0.0056: 加 dropout 同时压制 valid/test, ratio 几乎不变 (1.268 → 1.262)
- ratio 没有显著下降, 表明 v85j 的 ratio 1.268 并非过拟合主因, 而是 6 decoder 容量导致的 mild 复杂度增加

---

## 3. 训练曲线 (v85k)

```
ep   5  valid=0.0806  test=0.0522  ratio=1.542
ep  10  valid=0.1004  test=0.0708  ratio=1.417
ep  30  valid=0.1142  test=0.0894  ratio=1.277
ep  65  valid=0.1224  test=0.0970  ratio=1.261
ep 110  valid=0.1236  test=0.0990  ratio=1.249
ep 145  valid=0.1241  test=0.0987  ratio=1.257
ep 195  valid=0.1256  test=0.1001  ratio=1.255
ep 240  valid=0.1272  test=0.1000  ratio=1.271
ep 280  valid=0.1275  test=0.1010  ratio=1.262  (BEST test)
ep 289  valid=0.1278  test=0.1007  ratio=1.269
ep 300  valid=0.1275  test=?        (DONE)
```

**对比 v85j 同 epoch**:
- v85j ep65 test=0.1026 vs v85k ep65 test=0.0970 (-0.0056)
- v85j ep145 test=0.1037 vs v85k ep145 test=0.0987 (-0.0050)
- v85j ep240 test=0.1053 vs v85k ep240 test=0.1000 (-0.0053)
- v85j ep280 (final) test=0.1053 vs v85k ep280 test=0.1010 (-0.0043)

**Stage3 配置**:
- NUM_EPOCHS=300 (沿用 v85j)
- LR cosine (warmup_frac=0.05, LR_min_factor=0.05)
- EARLY_STOP=15
- num_decoder_layers=6 (沿用 v85j)
- **dropout=0.30** (vs v85j 0.20, **核心改动**)
- label_smoothing=0.05, WD=0.01
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85j 完全一致)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 300ep ≈ 60 min, loss 7.69 → 2.58, ckpt 22.5MB. 无 NaN/Inf. 早停未触发, 跑满 ep300.

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = **0.1010** (ep280 stage4) vs v85j 0.1053 (**-0.0043**, 退化), vs v85i 0.1044 (-0.0034)
- best valid R@10 = **0.1281** vs v85j 0.1337 (-0.0056), vs v85i 0.1294 (-0.0013)
- ratio = 1.262 (vs v85j 1.268 -0.006, 几乎不变)

---

## 5. 根因分析

1. **dropout 0.30 过度正则化**: v85j ratio 1.268 仅略高于 v77 1.215 (+0.053), 实际并无明显过拟合. 加 dropout 0.30 同时压制 valid/test, ratio 几乎不变 (-0.006). 假设被证伪.
2. **6 decoder 容量被 dropout 抵消**: v85j 6 decoder 比 v85i 4 decoder +0.0009, 但加 dropout 0.30 之后这部分增益被 dropout 噪声抵消, 净效果 -0.0043
3. **早期训练严重延迟**: v85k ep5 valid=0.0806 vs v85j ep5 0.0932 (-0.0126), ep30 valid=0.1142 vs v85j ep30 0.1227 (-0.0085). dropout 0.30 让模型在初期 epoch 表现差
4. **后期才追近但永远追不上**: ep200+ v85k valid ~0.126 vs v85j valid ~0.131 (-0.005), 整体差距稳定

---

## 6. 结论 + 下一步

**v85k NO-GO** — dropout 0.30 过度正则化, 抑制 6 decoder 容量优势, test 退化 -0.0043.

**v85j 仍是 v85 系列 SOTA** (test=0.1053, num_decoder_layers=6 + dropout=0.20).

**下一步 = v85l: 继续堆叠 decoder (6→8) + 保持 dropout 0.20**:
1. **Stage1 + Stage2 保持**: baseline Stage1 + v15 SID 不变
2. **Stage3 配置变化**:
   - num_decoder_layers 6 → **8** (继续堆叠, 看是否进一步提升)
   - dropout 0.20 保持 (v85k NO-GO 已验证 0.30 过强)
3. **预期 test**: v85j 0.1053 + 0.001~0.003 = 0.106~0.108 (接近 v77 0.1080)
4. **风险**: ratio 可能进一步上升 (1.268 → 1.29+), 但 v85k 验证 dropout 0.30 不是答案, 8 decoder 可能因数据量不足而过拟合

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85k_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep289, valid=0.1281)
- stage4 test on best: `stage4_test_on_best/ep280/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85k_stage3.log`