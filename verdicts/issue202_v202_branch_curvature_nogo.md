# Issue #202 v202 Branch-Aware HAB (Path B): NO-GO

日期: 2026-08-08
commit: pending

---

## 1. 总结

**v202 (baseline Stage1 + v15 Stage2 SID + 6-Decoder T5 + LR_WARMUP_FRAC=10% + Branch-Aware HAB injection (low=0.7×/mid=1.0×/high=1.3× λ_max)) NO-GO** — best test_R10=**0.0975** (ep185 stage4 ckpt 手动 eval, R@5=0.0780 / R@20=0.1194 / NDCG@10=0.0705 / NDCG@20=0.0761) vs v85p 0.1060 (**-0.0085, 退化**), vs HG-Rec baseline 0.1024 (-0.0049), best valid_R10=**0.1246** (ep185, ratio=1.279). DDP 4×L40S 13s/ep × 199ep ≈ 47 min 训练后手动 kill (用户指示提前结束, 已 no_improvement 7/15 早期无 valid 提升).

**意义**:
1. **Branch-Aware HAB 在 v15 SID 上失效**: Phase A 验证 branch heterogeneity 显著 (L0 CV(B)=0.180, L0-L1 CV(B)=0.849), 但在 Stage3 注入 per-prefix λ_max multiplier 后, 整体 test 退化 0.0085, valid 也退化 0.0072 vs v85p
2. **核心问题 — 不平衡注入破坏 HAB 训练**: high-B bucket 22 个 prefix 的 λ_max × 1.3 → 这部分 prefix 在 attention bias 上权重过大, 但其他 42 个 prefix 的梯度信号被稀释. HAB 残差学习模式 (Dbar_frozen + α·(U·V^T - Dbar_frozen)) 对 λ_max 缩放敏感, 缩放变化让 α 学到次优 anchor
3. **Path B (Stage3 改造) 路线失败**: 不动 Stage2 SID 是好约束, 但 Stage3 改造空间不足以突破. 真正 Branch Curvature 改造必须在 Stage2 (per-layer κ 分桶)
4. **v85p 仍是 v85 系列 SOTA** (test=0.1060), v77 仍是整体 SOTA (test=0.1080)

---

## 2. 实验对照 (v85 系列 + v202)

| 实验 | Stage1 | SID | Branch Curvature | best test | best valid | ratio | best epoch |
|---|---|---|---|---|---|---|---|
| v77 SOTA | hyp_v2 | hyp_v2 (06af) | ❌ | **0.1080** | 0.1312 | 1.215 | — |
| v85 P0 | hyp_v2 | hyp_v2 (06af) | ❌ | 0.1077 | 0.1339 | 1.243 | — |
| v85j (prior SOTA) | baseline | v15 (5f83) | ❌ | 0.1053 | 0.1337 | 1.268 | 239 |
| **v85p (current SOTA)** | **baseline** | **v15 (5f83)** | **❌** | **0.1060** | **0.1328** | **1.244** | **175** |
| **v202** | **baseline** | **v15 (5f83)** | **✅ (Path B)** | **0.0975** | **0.1246** | **1.279** | **185** |

**关键观察**:
- v202 vs v85p (同 SID + 同 Stage3 架构): test -0.0085, valid -0.0082
- v202 ratio 1.279 vs v85p 1.244 (+0.035) — ratio 恶化, Branch Curvature 注入让 valid/test 都同步滞后 (而非 valid 过拟合)
- v202 训练曲线: 早期 ep5 test=0.0492 vs v85p ep5=0.0563 (-0.0071) — 起步即慢, 全程未能追上

---

## 3. 训练曲线 (v202)

```
ep   5  valid=0.0732  test=0.0492  ratio=1.488  (起步 vs v85p 0.0563, -0.0071)
ep  15  valid=0.1054  test=0.0789  ratio=1.313  (vs v85p 0.0890, -0.0101)
ep  45  valid=0.1177  test=0.0911  ratio=1.293  (vs v85p 0.0995, -0.0084)
ep  55  valid=0.1193  test=0.0927  ratio=1.288  (vs v85p 0.1011, -0.0084)
ep  85  valid=0.1219  test=0.0955  ratio=1.276  (vs v85p 0.1038, -0.0083)
ep 105  valid=0.1235  test=0.0954  ratio=1.295
ep 145  valid=0.1240  test=0.0966  ratio=1.283
ep 185  valid=0.1246  test=0.0975  ratio=1.279  (BEST ckpt, best test)
ep 199  no_improv 7/15, kill -9 提前结束
```

**Stage3 配置**:
- NUM_EPOCHS=300, LR cosine (warmup_frac=0.10, LR_min_factor=0.05) — 沿用 v85p
- EARLY_STOP=15
- num_layers=6, num_decoder_layers=6, num_heads=6, d_model=128, d_ff=1024 — 沿用 v85p
- dropout=0.20, label_smoothing=0.05, WD=0.01 — 沿用 v85p
- HAB λ_max=0.20, residual_alpha_init=0.5 — 沿用 v85p
- **核心改动**: Branch-Aware HAB 注入
  - L0 prefix 64 个 → 等频分桶 (quantile 策略)
  - bucket 0 (低 B, 21 prefix): λ_max × 0.7
  - bucket 1 (中 B, 21 prefix): λ_max × 1.0
  - bucket 2 (高 B, 22 prefix): λ_max × 1.3
  - 实现: monkey-patch hab_module.get_B_geo, 给 L0 pair bias 乘以 per-source multiplier
- Stage2 不变: baseline Stage1 + v15 SID (sha=5f8331cc)
- Total params: 6,820,352 (与 v85p 完全一致, Branch Curvature 0 额外参数)

---

## 4. Gate 评估

**Gate 1 (Stage1)**: PASS — **baseline Stage1** (sha=1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc, 9922 items).

**Gate 2 (Stage2)**: PASS — **v15 Stage2 SID** (sha=5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07). κ=[0.30, 1.79, 1.48], c=[1.35, 6.00, 4.39]. **完全沿用, 未改动 Stage2**.

**Gate 3 (Stage3)**: PASS — DDP 4×L40S 13s/ep × 199ep ≈ 47 min, loss 6.27 → 2.61, ckpt 22.5MB. 无 NaN/Inf. Branch Curvature 注入成功 (日志显示 `λ_max bucket mult: low=0.70 mid=1.00 high=1.30 strategy=quantile bucket sizes=[21, 21, 22]`).

**Gate 4 (Stage4)**: **FAIL**.
- best test R@10 = **0.0975** (ep185 stage4) vs v85p 0.1060 (-0.0085, **NO-GO**), vs v85j 0.1053 (-0.0078), vs HG-Rec baseline 0.1024 (-0.0049)
- best valid R@10 = **0.1246** vs v85p 0.1328 (-0.0082)
- ratio = **1.279** vs v85p 1.244 (+0.035, 略恶化)
- best test R@5=0.0780, R@20=0.1194, NDCG@5/10/20=0.0642/0.0705/0.0761

---

## 5. 根因分析

1. **Path B (Stage3 改造) 路线本身天花板低**: Stage3 注入 per-prefix λ_max multiplier 没有改变 Stage2 码字的几何分布, 只能微调 attention bias 强度. 22 个 high-B prefix 的 λ_max × 1.3 让 HAB 残差信号偏向 high-B subtree, 但 T5 已学到的 token 注意力模式无法利用这种 prefix-specific 强度变化 (T5 编码历史 token 时并不区分 prefix bucket ID)
2. **HAB 残差学习 + per-prefix multiplier 不兼容**: residual_alpha 是 per-layer (3 个标量), 不是 per-prefix. 给 prefix × multiplier 后, 整层 λ_raw 学到的目标被扰乱 — α 学到的 anchor (Dbar_frozen) 在乘以 per-prefix multiplier 后不再是真正的 anchor, 残差学习的稳定性被破坏
3. **Branch-Aware 概念正确, 路径错误**: Phase A 验证了 v15 SID 同层 prefix 异质性 (L0 CV(B)=0.180, L0-L1 CV(B)=0.849), 这是 branch-aware 改造的必要条件. 但充分条件应在 Stage2 (per-layer per-prefix κ), 而不是 Stage3 (per-prefix λ_max)
4. **训练曲线全程低于 v85p**: ep5 起步 -0.0071, 全程 -0.008 滞后, 没有出现 ep150+ 反弹. cosine LR 主下降区间 v202 已无有效改善空间

---

## 6. 结论 + 下一步

**v202 NO-GO** — Branch-Aware HAB (Path B) 退化 test -0.0085 vs v85p. Stage3 注入 per-prefix λ_max multiplier 路线封闭.

**关键教训**:
- Phase A 的异质性证据 **必要但非充分** — 异质性存在不代表 Stage3 简单缩放能利用它
- Stage2 SID 几何分布是 "锚", Stage3 微调空间有限
- 真正 Branch Curvature 改造应回到 Stage2 (per-layer per-prefix learnable κ),但 Stage2 已有 RQ-VAE + κ sync 等复杂机制, 直接改造风险极大

**下一步候选**:
1. **方案 A: v85q = v85p + LR_min_factor 0.05→0.02** (v85 系列内继续 LR schedule 微调)
2. **方案 B: 回到 Stage2 改造 Branch Curvature** — 在 taskA_stage2.py 加 3-Expert per-layer κ (per-prefix router), 重训 Stage2. 风险大 (Stage2 重训 1000ep × ~10 min = ~3 天, 且需要 Phase A 决策是否值得)
3. **方案 C: v85s = v85p + Stage3 batch_size 1024→2048** (DDP 全局 batch 翻倍, 梯度更稳)

**优先级**: P0 — 方案 A (LR_min_factor 0.02) 是 v85p 唯一低风险微调, 预期 +0.0005~0.0010. 方案 B 改造 Stage2 风险极大, 需要先 v202/v85p Stage2 兼容性预检. 方案 C 需要更多 GPU 资源.

**严禁**任何 DECOR 机制 (--enable_prompt_former / decor_prompt_former.py). 纯曲率路线继续.

---

## 7. 文件清单

- product_dir: `/fs04/ar57/wenyu/GeneRec/taskA/_history/issue202_v202_branch_curvature/`
- best ckpt: `HG_Rec_best.pth` (ep185, valid=0.1246)
- stage4 test on best: `stage4_test_on_best/ep185/eval_test.json` (test_R10=0.0975)
- trace: `train_pure_t5.log`
- verdict: 本文件
- Stage3 script: `/fs04/ar57/wenyu/GeneRec/common/stage3/stage3_train_pure_t5.py` (Phase B Path B: 新增 --branch_curvature_* flag + build_branch_curvature_lut + install_branch_curvature_on_hab)
- SID: `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/sid_output.npy` (sha=5f8331cc, 未改动)
- commit: pending (verdict 落盘)