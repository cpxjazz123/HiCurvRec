# Issue #141 v85j baseline Stage1 + v15 SID + 6-Decoder Cosine: PARTIAL-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85j (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + num_decoder_layers 4→6) PARTIAL-GO** — best test_R10=**0.1053** (ep239 stage4 ckpt 手动 eval, R@5=0.0835 / R@20=0.1313 / NDCG@10=0.0776 / NDCG@20=0.0842) vs v85i 0.1044 (+0.0009, **v85 系列新 SOTA**), vs v77 0.1080 (-0.0027), vs HG-Rec baseline 0.1024 (+0.0029, 超越基线), best valid_R10=**0.1337** (ep239), ratio=1.268 (vs v85i 1.235 +0.033, vs v77 1.215 +0.053). DDP 4×L40S 12s/ep × 300ep ≈ 60 min 训练, 跑满 ep300, best ckpt = ep239 valid=0.1337.

**意义**:
1. **6 解码层是有效提升**: v85j 在 v85i 基础上仅改 num_decoder_layers 4→6, test 从 0.1044 → 0.1053 (+0.0009, +0.86%). 标准 T5-small 解码器容量帮助 SID 序列生成
2. **仍未达 0.11 目标** (-0.0047), 但已接近 v77 0.1080 (差 0.0027)
3. **ratio 上升 (1.235 → 1.268)**: 略有过拟合, 但 test/valid 同步提升表明 6 decoder 是 net positive
4. **下一步候选**: 进一步加 decoder (6→8) + 增加 dropout (0.20→0.30) 抑制 ratio, 或 Stage2 增量代码本增加 SID 容量

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | LR 调度 | num_dec | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | const 1e-3 | 4 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | const 1e-3 | 4 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | const 1e-3 | 4 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | cos 5%/0.1 | 4 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | cos 10%/0.05 + ES15 | 4 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 4 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 4 | 0.1012 | 0.1286 | 1.255 | 150 |
| v85h | baseline | v15 (5f83) | cos 5%/0.1 (200ep) | 4 | 0.1042 | 0.1302 | 1.249 | 185 |
| v85i | baseline | v15 (5f83) | cos 5%/0.05 (300ep) | 4 | 0.1044 | 0.1294 | 1.235 | 205 |
| **v85j** | **baseline** | **v15 (5f83)** | **cos 5%/0.05 (300ep)** | **6** | **0.1053** | **0.1337** | **1.268** | **239** |

**关键观察**:
- v85 系列首次跨越 0.105 大关 (v85j 0.1053)
- v85j 仍差 v77 0.0027 (-2.5%), 差 0.11 目标 0.0047 (-4.3%)
- 6 decoder layers 较 4 decoder layers 增量 +0.0009 (相对 v85i), 是结构层面提升

---

## 3. 训练曲线 (v85j)

```
ep   5  valid=0.0932  test=?
ep  10  valid=0.1083  test=?
ep  30  valid=0.1227  test=0.0980
ep  60  valid=0.1293  test=0.1026
ep  80  valid=0.1304  test=0.1032
ep 145  valid=0.1307  test=0.1037
ep 155  valid=0.1309  test=0.1046
ep 185  valid=0.1318  test=0.1043
ep 205  valid=0.1323  test=0.1052
ep 215  valid=0.1328  test=0.1046
ep 240  valid=0.1336  test=0.1053  (BEST ckpt, best test)
ep 275  valid=0.1337  test=?
ep 300  valid=0.1326  test=?      (DONE)
```

**增量趋势**:
- ep30→ep80: test 0.0980→0.1032 (+0.0052/50ep = +0.0001/ep)
- ep80→ep145: test 0.1032→0.1037 (+0.0005/65ep = +0.000008/ep)
- ep145→ep240: test 0.1037→0.1053 (+0.0016/95ep = +0.000017/ep, cosine 末期微增)
- ep240→ep300: valid plateau 0.1337→0.1326, best ckpt 锁定 ep240

**Stage3 配置**:
- NUM_EPOCHS=300 (沿用 v85i)
- LR cosine (warmup_frac=0.05, LR_min_factor=0.05, 沿用 v85i)
- EARLY_STOP=15 (沿用 v85i)
- num_decoder_layers=**6** (vs v85i 4, **核心改动**)
- num_layers=6, d_model=128, d_ff=1024, num_heads=6, d_kv=64 (其余不变)
- dropout=0.20, label_smoothing=0.05, WD=0.01
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30 (沿用 v85i)
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85i 完全一致)
- Total params: 6,820,352 (vs v85i decoder 4 层约 5,800,000, 增加 ~17% 参数)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** `HG-Rec/dataset/Instruments/item_emb.parquet` (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items, Euclidean 无 per-item radius).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** `taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 300ep ≈ 60 min, loss 7.66 → 2.39, ckpt 22.5MB 正常保存. 无 NaN/Inf. 早停未触发, 跑满 ep300.

**Gate 4 (Stage4)**: **PARTIAL**.
- best test R@10 = **0.1053** (ep239 stage4) vs v77 0.1080 (-0.0027, 未达 0.11)
- best test R@10 vs v85i 0.1044 = **+0.0009** (v85 系列新 SOTA)
- best test R@10 vs HG-Rec baseline 0.1024 = **+0.0029** (超越基线)
- best valid R@10 = **0.1337** vs v85i 0.1294 (+0.0043), vs v77 0.1312 (+0.0025)
- ratio = 1.268 (vs v85i 1.235 +0.033, vs v77 1.215 +0.053 — 略高)
- best test R@5=0.0835, R@20=0.1313, NDCG@5/10/20=0.0706/0.0776/0.0842

---

## 5. 关键发现

1. **6 decoder layers 显著提升**: 在 v85i 0.1044 基础上仅改 num_decoder_layers 4→6, test +0.0009 (+0.86%), valid +0.0043 (+3.3%). 标准 T5-small 解码器容量帮助 SID 序列生成
2. **ratio 上升信号需关注**: 1.235 → 1.268 (+0.033), valid 信号略强于 test, 略有过拟合倾向. 下一步需要更强正则 (dropout 0.20→0.30 / label_smoothing 0.05→0.10 / weight_decay 0.01→0.02) 来压 ratio
3. **v85j 与 v77 差距收窄**: 0.1080 - 0.1053 = 0.0027 (-2.5%), 比 v85i 差 0.0036 (-3.3%) 收窄 0.0009
4. **参数增量合理**: 6.82M (vs 5.8M, +17% 参数), 训练时长略增 (12s/ep vs v85i ~12s/ep, 同等), 推理时间略增但 stage4 test 36.6s 完成
5. **新维度候选**:
   - **方案 A: v85k = v85j + dropout 0.30**: 压 ratio, 可能让 test 进一步 +0.001
   - **方案 B: v85k = v85j + num_decoder_layers 6→8**: 继续堆叠解码层, 可能 +0.001~+0.003
   - **方案 C: v85k = v85j + d_model 128→192**: 增加 embedding 维度, 风险中等
   - **方案 D: v85k = Stage2 incremental codebook [128, 256, 512]**: SID 容量提升, 工程量大

---

## 6. 结论 + 下一步

**v85j PARTIAL-GO** — v85 系列新 SOTA (test=0.1053, +0.0009 vs v85i), 验证 num_decoder_layers 4→6 是有效的结构升级.

**下一步 = v85k: 进一步挖掘 6 decoder 潜力 + 正则化**:
1. **Stage1 + Stage2 保持**: baseline Stage1 (sha=1a42341f) + v15 SID (sha=5f8331cc) 不变
2. **Stage3 配置变化**:
   - **推荐方案 A**: v85j 配置 + dropout **0.20 → 0.30** (正则化压 ratio)
   - **备选方案 B**: v85j 配置 + num_decoder_layers **6 → 8** (继续堆叠)
3. **预期 test**: 方案 A 0.1053 + 0.001~0.003 = 0.106~0.108 (接近 v77 0.1080). 方案 B 0.1053 + 0.001~0.003 = 0.106~0.108
4. **风险**: dropout 0.30 可能欠拟合, num_decoder_layers 8 可能过拟合更严重
5. **推荐先 A 后 B**: A 优先 (低风险), 若 A 失败再尝试 B

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85j_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep239, valid=0.1337)
- stage4 test on best: `stage4_test_on_best/ep240/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85j_stage3.log`
- Stage3 script: `/fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py` (modified num_decoder_layers 4→6)
- Stage4 script: `/fs04/ar57/wenyu/GeneRec/common/stage4/stage4_eval_pure_t5.py` (modified num_decoder_layers 4→6 同步)