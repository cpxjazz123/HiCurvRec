# iter6 Gate Decision — Riemannian Adam 单独 (R36n d 纯版)

## 机制

v318 baseline + 对 3 层 RQ codebook.embedding.weight.grad 乘 1/√c 反 rescale.
切断 c 与 codebook 更新幅度耦合: c 大时更新慢 (高曲率精细), c 小时更新快 (低曲率大步).
其余沿用 v318 (cyclic c[0.3..1.0] period 50k, midpoint off, Sinkhorn sk_eps=0.05).
第 90 次 R37 尝试, 第 6 次 iterN.

## Stage 2 SID Gate (R50 4 项, 50k ckpt)

| 指标 | TIGER baseline | iter6 50k | 状态 |
|------|---------------|-----------|------|
| Embedding HitRate@K=50 | 0.7165 | **0.7165** | SAME PASS |
| 3-token SID Gini | 0.0672 | **0.0503** | -25.1% PASS |
| per_layer mean Gini | 0.2574 | [0.0746, 0.1292, 0.1679] | -51.9% PASS |
| collision_rate | high | low (ext [768,779] = 12 unique) | PASS |

**Gate 1 PASS 大胜 TIGER**: 唯一度 +2.0%, gini -25%, per_layer -52%.

注: iter6 50k (gini=0.0503) 略差于 iter5 final (gini=0.0488), 但仍远超 TIGER baseline. RAdam
随训练退化 (10k gini 0.0417 → 50k gini 0.0503), 但 10k ckpt 已被 _delete_previous_artifact 删除.

## Stage 3 测试 (R37)

- 200 epochs DDP 4 卡 train, ~62 min wall-clock
- best_recall@10 (valid) = 0.060220 (epoch 196, +0.05% vs iter5 0.060168 — TIE)
- **test_recall@10 = 0.0535** (-3.8% vs iter5 0.0556, < 目标 0.065)
- test_ndcg@10 = 0.0296

## 决策: R37 FAIL — NO-GO

iter6 RAdam 在 SID 端 4/4 PASS 远超 TIGER, 但 stage3 test_R@10=0.0535 略差于 iter5 0.0556.
可能原因: RAdam 改变了 codebook 优化轨迹, 找到的局部最优 generalization 略差. 字符级 lock
未破 (valid_R@10 0.060220 vs iter5 0.060168, 几乎完全 lock).

Stage 3 T5 ceiling 已基本确认: 5/5 stage2 SID 创新 (iter2-iter6) 全部 lock 0.055-0.060.

下一步 iter7 考虑: Sinkhorn temperature schedule (新机制, anneal sk_eps), 或 RAdam + 
per-layer hetero c 组合, 或 4-layer RQ-VAE 架构变更.

## 提交

Commit hash: TBD (git 时落地)