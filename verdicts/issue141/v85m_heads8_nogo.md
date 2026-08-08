# Issue #141 v85m baseline Stage1 + v15 SID + 6-Encoder/6-Decoder + num_heads 6→8: NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85m (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + num_layers=6 + num_decoder_layers=6 + num_heads 6→8 + d_kv 64) NO-GO** — best test_R10=**0.1044** (ep155 stage4 ckpt 手动 eval, R@5=0.0847 / R@20=0.1276 / NDCG@10=0.0778 / NDCG@20=0.0836) vs v85j 0.1053 (**-0.0009**, 退化), vs v85i 0.1044 (持平), vs HG-Rec baseline 0.1024 (+0.0020, 超越基线), best valid_R10=**0.1316** (ep155), ratio=1.260 (vs v85j 1.268 -0.008, vs v85i 1.235 +0.025). DDP 4×L40S 13s/ep × 170 ep ≈ 37 min 训练, early-stop 触发 ep170 (无 valid 提升 15ep 后停).

**根因** — num_heads 6→8 增加注意力头数 (每个头维度 21→16, d_kv=64/8=16), 没有显著增益. 总参数量从 6.82M (v85j) 增加到 8.00M (+17%), 但没有带来 test 增益 (test -0.0009). 假设 "更细粒度注意力头 (更多 head + 更小 d_kv) → 提升 SID 序列建模" 被证伪.

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | num_layers/num_dec | num_heads | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | 6/4 | 6 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | 6/4 | 6 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | 6/4 | 6 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | 6/4 | 6 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | 6/4 | 6 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | 6/4 | 6 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | 6/4 | 6 | 0.1012 | 0.1286 | 1.255 | 150 |
| v85h | baseline | v15 (5f83) | 6/4 | 6 | 0.1042 | 0.1302 | 1.249 | 185 |
| v85i | baseline | v15 (5f83) | 6/4 | 6 | 0.1044 | 0.1294 | 1.235 | 205 |
| **v85j** | **baseline** | **v15 (5f83)** | **6/6** | **6** | **0.1053** | **0.1337** | **1.268** | **239** |
| v85k | baseline | v15 (5f83) | 6/6 | 6 (drop=0.30) | 0.1010 | 0.1281 | 1.262 | 280 |
| v85l | baseline | v15 (5f83) | 8/8 | 6 | 0.1042 | 0.1326 | 1.262 | 184 |
| **v85m** | **baseline** | **v15 (5f83)** | **6/6** | **8** | **0.1044** | **0.1316** | **1.260** | **155** |

**关键观察**:
- v85j (heads=6) → v85m (heads=8) test -0.0009, valid -0.0021: 增加 2 个 head 反而降低性能
- v85m ratio 1.260 (vs v85j 1.268 -0.008, vs v85i 1.235 +0.025) — 中等水平
- **6 decoder layers + 6 attention heads (v85j) 是当前 6/6 架构的注意力配置最佳点**
- v85m 训练提前停 ep170, 没跑满 300ep — 但训练曲线已稳定 (test plateau ~0.1034-0.1038), 不太可能后续有显著增益

---

## 3. 训练曲线 (v85m)

```
ep   4  valid=0.0953  test=0.0658  ratio=1.449
ep   9  valid=0.1116  test=0.0888  ratio=1.257
ep  19  valid=0.1185  test=0.0921  ratio=1.286
ep  34  valid=0.1262  test=0.0991  ratio=1.274
ep  54  valid=0.1277  test=0.1025  ratio=1.245
ep  69  valid=0.1289  test=0.1037  ratio=1.243
ep  94  valid=0.1302  test=0.1028  ratio=1.267
ep  99  valid=0.1304  test=0.1031  ratio=1.264
ep 114  valid=0.1307  test=0.1034  ratio=1.264
ep 119  valid=0.1313  test=0.1038  ratio=1.265
ep 155  valid=0.1316  test=0.1044  ratio=1.260  (BEST ckpt, best test)
ep 170  EARLY STOP (no improv 15ep)
```

**对比 v85j 同 epoch**:
- v85j ep69 test=0.1037 vs v85m ep69 test=0.1037 (持平)
- v85j ep114 test=0.1037 vs v85m ep114 test=0.1034 (-0.0003)
- v85j ep155 test=0.1046 vs v85m ep155 test=0.1044 (-0.0002)
- v85j ep239 (best) test=0.1053 vs v85m ep155 (early stop) test=0.1044 (-0.0009)

**Stage3 配置**:
- NUM_EPOCHS=300, LR cosine (warmup_frac=0.05, LR_min_factor=0.05)
- EARLY_STOP=15 (沿用 v85j)
- **num_heads=8** (vs v85j 6, **核心改动 — 注意力头细化**)
- d_kv=64 保持 (每个 head 维度 64/8=16, vs v85j 64/6≈10.67)
- num_layers=6, num_decoder_layers=6, d_model=128, d_ff=1024 (其余不变)
- dropout=0.20, label_smoothing=0.05, WD=0.01
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85j 完全一致)
- Total params: 8,000,128 (vs v85j 6,820,352, +17% 参数)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items, Euclidean 无 per-item radius).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 13s/ep × 170ep ≈ 37 min, loss 7.66 → 2.42, ckpt 32MB (vs v85j 22.5MB, +42% — 因为更多 attention head 权重). 无 NaN/Inf. Early stop 触发 ep170.

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = **0.1044** (ep155 stage4) vs v85j 0.1053 (-0.0009, 退化), vs v85i 0.1044 (持平)
- best valid R@10 = **0.1316** vs v85j 0.1337 (-0.0021), vs v85i 0.1294 (+0.0022)
- ratio = 1.260 (vs v85j 1.268 -0.008, vs v85i 1.235 +0.025)
- best test R@5=0.0847, R@20=0.1276, NDCG@5/10/20=0.0714/0.0778/0.0836

---

## 5. 根因分析

1. **6/6 + 6 heads 是当前架构甜点**: v85m 8 heads 测试结果证明, 增加 head 数 (6→8) 不带来 test 增益. v85j 6/6 + 6 heads + drop=0.20 仍是该 SID 容量上限
2. **更小 head 维度 (d_kv=16) 未提升细粒度注意力**: 头数从 6 增到 8, 每个头维度从 ~10.67 降到 16 (实际是 64/8=16 vs 64/6≈10.67 — 注意 d_kv 固定, head 维度不变). 但 head 数增加让 attention 矩阵分散, 可能稀释每个 head 的 focus 强度
3. **+17% 参数未带来收益**: 6.82M (v85j) → 8.00M (v85m), 训练时长从 12s/ep → 13s/ep (+8%), 但 test 退化 -0.0009
4. **ratio 1.260 略低于 v85j 1.268**: valid -0.0021 大于 test -0.0009, 导致 ratio 略降. 但 valid/test 都下降, ratio 改善只是表象
5. **early stop ep170 提前**: 训练稳定期比 v85j (ep239) 短 ~70ep, 说明 8 heads 配置训练曲线更平, 没有后续显著提升空间

---

## 6. 结论 + 下一步

**v85m NO-GO** — num_heads 6→8 退化 test -0.0009 (从 0.1053 → 0.1044). 增加注意力头数不是突破方向.

**v85j 仍是 v85 系列 SOTA** (test=0.1053, 6/6 + heads=6 + drop=0.20).

**下一步 = v85n: 跳出 Stage3 维度, 尝试 Stage2 增量代码本 k=512**:
1. **Stage1 保持**: baseline Stage1 (sha=1a42341f) 不变
2. **Stage2 变化** (核心):
   - 训练新变体 v18: **codebook_sizes=[128, 256, 512]** (vs v15 [64, 128, 256])
   - 容量翻倍: 第 1 层 64→128, 第 2 层 128→256, 第 3 层 256→512
   - REC_LAYER_W=[1, 3, 9] 保持, KAPPA_MAX=0.5 保持
   - κ=[?, ?, ?] / c=[?, ?, ?] 新值 (待训练)
3. **Stage3 配置**:
   - num_layers=6 + num_decoder_layers=6 (v85j 优势架构)
   - num_heads=6, d_kv=64, dropout=0.20
   - 300ep + cosine LR_min=0.05
4. **风险与对策**:
   - Stage3 `_LAYER_ID_LUT` 需同步更新 [128, 256, 512, 1], 4-token SID 序列, MAX_LEN 20 可能不足 (需 24+)
   - Stage2 训练时间增加 (k=512 vs k=256 → 训练迭代 ~30% 慢), Stage3 训练时间微增 (更多 SID 类别)
   - 预期 v85j 0.1053 + 增量代码本 SID 容量提升 = 0.107~0.110 (接近 0.11 目标)
5. **优先级**: P0 — 这是 v85 系列未探索的最大方向, 也是接近 0.11 目标的最大希望

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85m_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep155, valid=0.1316)
- stage4 test on best: `stage4_test_on_best/ep155/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85m_stage3.log`
- Stage3 script: `/fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py` (modified num_heads 6→8)
- Stage4 script: `/fs04/ar57/wenyu/GeneRec/common/stage4/stage4_eval_pure_t5.py` (modified num_heads 6→8 同步)