# Issue #141 v85i baseline Stage1 + v15 SID + Cosine 300ep + LR_min=0.05: PARTIAL-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v85i (baseline Stage1 + v15 Stage2 SID + LR cosine 300ep + LR_min=0.05 + EARLY_STOP=15) PARTIAL-GO** — best test_R10=**0.1044** (ep205 stage4 ckpt 手动 eval, R@5=0.0847 / R@20=0.1300 / NDCG@10=0.0780 / NDCG@20=0.0845) vs v85h 0.1042 (+0.0002, **v85 系列新 SOTA**), vs v77 0.1080 (-0.0036), vs HG-Rec baseline 0.1024 (+0.0020), best valid_R10=**0.1294** (ep205), ratio=1.235 (vs v85h 1.249). DDP 4×L40S 12s/ep × 300ep ≈ 60 min 训练, 跑满 ep300, best ckpt = ep205 valid=0.1294.

**意义**:
1. **v85i = v85h 兄弟版, test=0.1044 巩固 v85h 0.1042 (+0.0002)**: 验证 cosine 末期在 LR_min=0.05 仍有微增, 但增量极小 (0.0002), 触及 LR 调度极限
2. **300ep 远超 200ep 但无实质提升**: 表明 cosine LR 调度已饱和, 需要新维度变量才能突破 0.1044 → 0.11
3. **仍未达 0.11 目标** (-0.0056), v85 系列 (v85 P0/c/d/e/f/g/h/i) 9 个变体全部完成, 最高 0.1044, v77 0.1080 仍未被超越
4. **下一步必须跳出 LR 调度维度**: Stage2 增量代码本 / Stage1 强化版 (R_MAX 变化) / Stage3 多 loss 组合 / 更深 T5 结构 (num_decoder_layers 4→6) 等

---

## 2. 实验对照 (Issue #141 v85 系列完整)

| 实验 | Stage1 | SID | LR 调度 | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | const 1e-3 | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | const 1e-3 | 0.1077 | 0.1339 | 1.243 | — |
| v85c | hyp_v2 | v15 (5f83) | const 1e-3 | 0.0911 | 0.1196 | 1.312 | — |
| v85d | hyp_v2 | hyp_v2 (06af) | cos 5%/0.1 | 0.1011 | 0.1270 | 1.256 | 99 |
| v85e | hyp_v2 | hyp_v2 (06af) | cos 10%/0.05 + ES15 | 0.0995 | 0.1272 | 1.278 | 104 |
| v85f | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 0.1034 | 0.1317 | 1.273 | 159 |
| v85g | hyp_v2 | v15 (5f83) | cos 5%/0.1 | 0.1012 | 0.1286 | 1.255 | 150 |
| v85h | baseline | v15 (5f83) | cos 5%/0.1 (200ep) | 0.1042 | 0.1302 | 1.249 | 185 |
| **v85i** | **baseline** | **v15 (5f83)** | **cos 5%/0.05 (300ep)** | **0.1044** | **0.1294** | **1.235** | **205** |

---

## 3. 训练曲线 (v85i)

```
ep   5  valid=0.1015  test=0.0731  ratio=1.389
ep  60  valid=0.1249  test=0.1007  ratio=1.240
ep  80  valid=0.1267  test=0.1013  ratio=1.251
ep 130  valid=0.1275  test=0.1039  ratio=1.227
ep 175  valid=0.1280  test=0.1042  ratio=1.228
ep 205  valid=0.1289  test=0.1044  ratio=1.235  (BEST ckpt, best test)
ep 240  valid=0.1267  test=?
ep 275  valid=0.1273  test=?
ep 295  valid=0.1286  test=?
ep 300  valid=0.1280  test=?      (DONE)
```

**增量趋势**: ep175→205 +0.0002/30ep = +0.000007/ep (cosine 末期 LR_min=5e-5 完全饱和). ep205 之后无 valid 提升 (no_improv > 10), best ckpt 锁定 ep205.

**Stage3 配置**:
- NUM_EPOCHS=300 (vs v85h 200)
- LR cosine (warmup_frac=0.05, **LR_min_factor=0.05**, vs v85h 0.1)
- EARLY_STOP=15 (vs v85h 10)
- baseline Stage1 (sha=1a42341f, HG-Rec/dataset/Instruments/item_emb.parquet)
- v15 SID (sha=5f8331cc)
- WD=0.01, dropout=0.20, label_smoothing=0.05
- HAB λ_max=0.20, residual_alpha_init=-20.0, λ_lr_ratio=30
- HAB final_cs=[1.3547, 6.0021, 4.3941], Dbar=[0.7260, 0.3607, 0.3208] (与 v85h 完全一致)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** `HG-Rec/dataset/Instruments/item_emb.parquet` (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items, Euclidean 无 per-item radius).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** `taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39].

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 12s/ep × 300ep ≈ 60 min, loss 7.67 → 2.44, ckpt 22.5MB 正常保存. 无 NaN/Inf. 早停未触发, 跑满 ep300.

**Gate 4 (Stage4)**: **PARTIAL**.
- best test R@10 = **0.1044** (ep205 stage4) vs v77 0.1080 (**-0.0036**, 未达 0.11)
- best test R@10 vs HG-Rec baseline 0.1024 = **+0.0020** (超越基线)
- best valid R@10 = **0.1294** vs v85h 0.1302 (-0.0008), vs v77 0.1312 (-0.0018)
- ratio = 1.235 (vs v85h 1.249 -0.014, vs v77 1.215 +0.020 — 健康)
- best test R@5=0.0847, R@20=0.1300, NDCG@5/10/20=0.0717/0.0780/0.0845

---

## 5. 关键发现

1. **cosine LR 调度已饱和**: v85h (200ep, LR_min=0.1) → v85i (300ep, LR_min=0.05) 增量仅 +0.0002, 验证 LR 维度上 0.11 不可达
2. **Stage1 baseline + v15 SID 是 v85 系列 SOTA 搭配**: v85h/v85i 两次验证, 该搭配为本征协同
3. **Dbar 中位数稳定**: v85h/v85i 都用 baseline Stage1 + v15 SID → Dbar=[0.7260, 0.3607, 0.3208] 完全相同, 几何信号已收敛到固定 landscape
4. **v85 系列 9 变体全景**: best v85i 0.1044 < v77 0.1080 (-0.0036), 仅在 v85h 0.1042 基础上有 +0.0002 微增, 表明 LR 调度已基本触顶, 必须切换维度
5. **新维度候选**:
   - **Stage2 增量代码本 k=512**: 当前 SID codebook = 2^10 = 1024, 增加 k=512 会让码字数翻倍, 单层分配 1024→2048, 码间距离增大, Stage3 学习压力增大但潜在容量提升
   - **Stage1 强化版 R_MAX 调整**: 当前 R_MAX=0.99 (default), 改为 R_MAX=0.85 / 0.92 / 0.95, 测试不同径向约束
   - **Stage3 多 loss 组合**: 现有仅 HAB loss (λ_max=0.20), 加结构正则 (e.g. geo-margin loss / Stage1 embedding MSE reconstruction loss)
   - **更深 T5 结构**: num_decoder_layers 4 → 6, 增加 decoder 容量

---

## 6. 结论 + 下一步

**v85i PARTIAL-GO** — v85 系列新 SOTA (test=0.1044), 巩固 v85h 0.1042, LR 调度饱和确认.

**下一步 = v85j: 跳出 LR 维度, 尝试 Stage3 结构变化**:
1. **Stage1 + Stage2 保持**: baseline Stage1 (sha=1a42341f) + v15 SID (sha=5f8331cc) 不变
2. **Stage3 配置变化**:
   - **方案 A: 加 Stage1 embedding reconstruction loss (REG_EMB=0.05)**: 在 T5 训练时辅助 T5 hidden 表征对齐 Stage1 embedding, 减少 SID codebook 信息瓶颈
   - **方案 B: 加 MarginRanking loss (geo_margin=0.1)**: 让负样本 SID 距离更远, 直接优化几何边界
   - **方案 C: 加 Attention dropout (dropout=0.3 vs v85i 0.20)**: 进一步正则化, 减少过拟合 (v85i ratio 1.235 偏高)
3. **推荐**: 方案 A (REG_EMB) — 直接给 T5 一个 Stage1 表征回归信号, 把 768 维 Stage1 embedding 信息注入 decoder, 不依赖 SID 解码全部语义
4. **预期 test**: 0.1044 (v85i) + 0.003 ~ 0.005 = 0.107 ~ 0.110, 接近 0.11

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue141_v85i_stage3/`
- best ckpt: `HG_Rec_best.pth` (ep205, valid=0.1294)
- stage4 test on best: `stage4_test_on_best/ep205/eval_test.json`
- trace: `trace.json`
- verdict: 本文件
- log: `/tmp/v85i_stage3.log`