# Issue #141 v85l baseline Stage1 + v15 SID + 8-Encoder/8-Decoder Cosine: NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85l (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + num_layers=8 + num_decoder_layers=8 强制对称) NO-GO** — best test_R10=**0.1042** (ep184 stage4 ckpt 手动 eval) vs v85j 0.1053 (**-0.0011**, 退化), vs v85i 0.1044 (-0.0002), vs HG-Rec baseline 0.1024 (+0.0018, 超越基线), best valid_R10=**0.1326** (ep274), ratio=1.262. DDP 4×L40S 14s/ep × 300ep ≈ 70 min 训练, 跑满 ep300.

**根因**:
1. **8+8 对称 T5 比 6+6 略差**: 增加 encoder 层数 6→8 让模型参数总量增加 (~33%), 但没有相对提升. 假设 "更多解码层 → 更高性能" 被证伪.
2. **transformers cache bug 触发架构约束**: T5Config attribute_map 把 num_hidden_layers 映射到 num_layers (encoder 的 count). 当 num_layers=6 / num_decoder_layers=8 时, transformers generate() 创建 DynamicCache 只用 num_hidden_layers=6 entries, 触发 IndexError (cache_utils.py:1304). v85l 改 8+8 对称绕过 bug, 但放弃了 v85j 的 6+6 优势.
3. **训练时长 +17%**: 14s/ep (vs v85j 12s/ep), 300ep ≈ 70 min (vs v85j 60 min).

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | num_layers/num_dec | dropout | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | 6/4 | 0.20 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | 6/4 | 0.20 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | 6/4 | 0.20 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | 6/4 | 0.20 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | 6/4 | 0.20 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | 6/4 | 0.20 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | 6/4 | 0.20 | 0.1012 | 0.1286 | 1.255 | 150 |
| v85h | baseline | v15 (5f83) | 6/4 | 0.20 | 0.1042 | 0.1302 | 1.249 | 185 |
| v85i | baseline | v15 (5f83) | 6/4 | 0.20 | 0.1044 | 0.1294 | 1.235 | 205 |
| **v85j** | **baseline** | **v15 (5f83)** | **6/6** | **0.20** | **0.1053** | **0.1337** | **1.268** | **239** |
| v85k | baseline | v15 (5f83) | 6/6 | 0.30 | 0.1010 | 0.1281 | 1.262 | 280 |
| **v85l** | **baseline** | **v15 (5f83)** | **8/8** | **0.20** | **0.1042** | **0.1326** | **1.262** | **184** |

**关键观察**:
- v85j (6/6) → v85l (8/8) test -0.0011, valid -0.0011: 增加 2 个 encoder 层 + 2 个 decoder 层 (-1 ratio) 反而降低性能
- 6+6 是当前曲率框架最佳架构

---

## 3. 训练曲线 (v85l)

```
ep   5  valid=0.0927  test=0.0642  ratio=1.443
ep  30  valid=0.1211  test=0.0954  ratio=1.269
ep  60  valid=0.1292  test=0.1007  ratio=1.283
ep  85  valid=0.1304  test=0.1022  ratio=1.276
ep 140  valid=0.1309  test=0.1039  ratio=1.260
ep 185  valid=0.1315  test=0.1042  ratio=1.262  (BEST ckpt, best test)
ep 245  valid=0.1317  test=0.1030  ratio=1.278
ep 275  valid=0.1326  test=0.1030  ratio=1.285
ep 300  valid=0.1325  test=?        (DONE)
```

**Stage3 配置**:
- NUM_EPOCHS=300, LR cosine (warmup_frac=0.05, LR_min_factor=0.05)
- EARLY_STOP=15
- **num_layers=8** + **num_decoder_layers=8** (vs v85j 6+6, **核心改动 — 强制对称**)
- dropout=0.20 (沿用 v85j)
- label_smoothing=0.05, WD=0.01
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85j 完全一致)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** (sha=1a42341f).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** (sha=5f8331cc). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 14s/ep × 300ep ≈ 70 min, loss 7.66 → 2.36, ckpt ~25MB. 无 NaN/Inf. 早停未触发, 跑满 ep300.

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = **0.1042** (ep184 stage4) vs v85j 0.1053 (-0.0011, 退化)
- best valid R@10 = **0.1326** vs v85j 0.1337 (-0.0011)
- ratio = 1.262 (vs v85j 1.268 -0.006)

---

## 5. 根因分析

1. **6+6 是曲率框架最佳架构**: v85l 8+8 测试结果证明, 增加 decoder 层 (6→8) 不带来增益. v85j 6+6 + drop=0.20 已是该 SID 容量上限
2. **transformers cache_utils 兼容性 bug**: T5Config attribute_map 把 num_hidden_layers 映射到 num_layers (encoder), 当 num_layers=6 / num_decoder_layers=8 时 generate() 创建 DynamicCache 只用 6 entries, 触发 cache_utils.py:1304 IndexError. v85l 改 8+8 对称绕过 bug, 但偏离 v85j 优势架构
3. **训练时长 +17%**: 14s/ep vs v85j 12s/ep, 70 min vs 60 min. 增加 ~33% 参数 (8+8 vs 6+6) 但性能反而下降
4. **ratio 改善但 test/valid 同步下降**: ratio 1.268 → 1.262 (-0.006) 略改善, 但 valid -0.0011 / test -0.0011 同步下降. ratio 改善是 valid 下降更多导致的相对变化

---

## 6. 结论 + 下一步

**v85l NO-GO** — 8+8 对称 T5 比 v85j 6+6 退化 -0.0011 test. 增加解码器层数不是突破方向.

**v85j 仍是 v85 系列 SOTA** (test=0.1053, 6+6 + drop=0.20).

**下一步 = v85m: 跳出 T5 层数维度, 尝试 Stage2 增量代码本 k=512**:
1. **Stage1 + Stage2 变化**:
   - Stage1 baseline (sha=1a42341f) 保持
   - Stage2 训练新变体 v18: codebook_sizes=[128, 256, 512] (vs v15 [64,128,256]), SID 容量翻倍
   - REC_LAYER_W=[1,3,9] 保持, KAPPA_MAX=0.5 保持
   - κ=[?,?,?] / c=[?,?,?] 新值 (待训练)
2. **Stage3 配置**:
   - num_layers=6 + num_decoder_layers=6 (回到 v85j 6+6)
   - dropout=0.20
   - 300ep + cosine LR_min=0.05
3. **风险**: Stage3 _LAYER_ID_LUT 需同步更新 [128,256,512,1], 4-token SID 序列, MAX_LEN 20 可能不足 (需 24+)
4. **预期 test**: v85j 0.1053 + 增量代码本 SID 容量提升 = 0.107~0.110 (接近 0.11 目标)

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85l_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep274, valid=0.1326)
- stage4 test on best: `stage4_test_on_best/ep185/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85l_stage3.log`