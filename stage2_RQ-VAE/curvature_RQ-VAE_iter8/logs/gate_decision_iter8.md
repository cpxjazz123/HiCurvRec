# iter8 Gate Decision — MGC Fixed c=1 baseline (v337 复现, R36n f)

## 机制

v318 baseline + 设置 C_CYCLIC_MIN = C_CYCLIC_MAX = 1.0, 等价于 v337 MGC fixed c=1 baseline.
锁定 c=1 让 artanh 永远在稳定域, 跳出 cyclic c[0.3, 1.0] 震荡.
其余沿用 v318 (midpoint off, Sinkhorn sk_eps=0.05).
第 93 次 R37 尝试, 第 8 次 iterN, baseline sanity check.

## Stage 2 SID Gate (R50 4 项, 100k ckpt)

| 指标 | TIGER baseline | iter8 100k | 状态 |
|------|---------------|-----------|------|
| Embedding HitRate@K=50 | 0.7165 | **0.7165** | SAME PASS |
| 3-token SID Gini | 0.0672 | **0.0522** | -22.3% PASS |
| per_layer mean Gini | 0.2574 | [0.0784/0.1268/0.1497] | -69.5% PASS |
| collision_rate | high | 1328 collision rows, ext [768,784] | PASS |
| per_layer codes | — | [254/242/246/256] | ALL HEALTHY |

**Gate 1 (SID) PASS 大胜 TIGER**: hitrate same, gini -22%, per_layer -70%.

## Stage 3 测试 (R37, 200 epoch DDP)

- 200 epochs DDP 4 卡 train, ~62 min wall-clock (started 10:18, finished 11:18)
- best_recall@10 (valid, epoch 196) = **0.059385** (vs iter6 0.060220, -1.4%)
- **test_recall@10 = 0.0552** (+3.2% vs iter6 0.0535, -0.7% vs iter5 0.0556)
- test_ndcg@10 = 0.0300

## 决策: R37 FAIL — NO-GO

iter8 MGC fixed c=1 在 SID 端大胜 TIGER (4/4 PASS), 但 stage3 test_R@10=0.0552 仍 lock ceiling 0.055-0.060.
虽然 iter8 比 iter6 (+3.2%) 改善, 但未超过 iter5 (-0.7%), 更未达 0.065 目标.

第 93 次 R37 FAIL. Stage 3 T5 ceiling 已 7 次 (iter2-iter8) 锁死 0.053-0.056, 远超
cyclic c、per-layer hetero c、midpoint、Möbius commit、RAdam、Sinkhorn 软退火、
MGC fixed c=1 七种不同机制都无法突破 0.055-0.060. **Stage 1 端任何 R36 几何/优化变更
对 stage3 T5 表征几乎透明** (memory v376/v377 多次证实).

下一步 iter9 候选:
- (a) Reverse Sinkhorn 退火 (sharp 0.05 → soft 0.5) (鼓励 entropy 探索)
- (b) Gumbel-Softmax temperature 退火 (与 Sinkhorn 不同实现)
- (c) EMA codebook + Sinkhorn sk_eps=0.05 固定
- (d) 4-layer RQ-VAE 架构变更 (depth 而非 width)
- (e) Lorentz (hyperboloid) model 替换 Poincaré ball
