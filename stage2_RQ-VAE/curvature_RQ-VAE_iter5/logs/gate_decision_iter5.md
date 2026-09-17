# iter5 Gate Decision — Mixed-curvature H × H × S (R36n g)

## 机制

Per-layer manifold 产品流形: L0/L2 = Poincaré, L1 = Sphere (新方向 R36n g).
Per-layer c hetero: [0.3..1.0, 0.5..1.5, 0.7..2.0], period [50k, 50k, 50k].
其余沿用 v318 baseline (cyclic midpoint, Sinkhorn sk_eps=0.05).

## Stage 2 SID Gate (R50 4 项)

| 指标 | TIGER baseline | iter5 | 状态 |
|------|---------------|-------|------|
| Embedding HitRate@K=50 | 0.5002 | **0.7165** | +43% PASS |
| 3-token SID Gini | 0.5310 | **0.0488** | -90.8% **历史最低** PASS |
| per_layer mean Gini | 0.4825 | [0.075, 0.153, 0.057] | -87.7% PASS |
| collision_rate | 0.4157 | 0.0152 | -96.3% PASS |

**Gate 1 PASS 极大领先**: SID 空间均匀度历史最佳, 球面层 L2 gini 0.057 极低 = 球面量化极均匀.

## Stage 3 测试 (R37)

- 200 epochs DDP 4 卡 train, 35 min wall-clock
- best_recall@10 (valid) = 0.060168 (epoch 186)
- **test_recall@10 = 0.0556** (< 目标 0.065)
- test_ndcg@10 = 0.0305

## 决策: R37 FAIL — NO-GO

iter5 是 SID 端历史最佳, 但 stage3 test_R@10=0.0556 与 iter2-iter4 (~0.055-0.060) 完全同一区间,
说明 Stage 3 T5 模型对 stage2 SID 几何变已经**完全 lock 字符级**, 纯 stage2 端几何机制
(cyclic / midpoint / Sinkhorn c-dep / per-layer hetero / mixed-curvature) 已无法突破 ~0.055 ceiling.

下一步必须**跳出"Stage 1 端纯几何变更"赛道**, 转向 Stage 0/Stage 3 协同机制 (例如 curriculum on
quantization difficulty, adaptive margin α/c, 或 stage3 端只读创新). iter6 考虑 Riemannian Adam
或 curriculum temperature.

## 提交

Commit hash: TBD (git status 时落地)