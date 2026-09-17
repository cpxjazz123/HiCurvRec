# iter7 Gate Decision — Sinkhorn sk_eps 退火 0.5→0.05 (R36 全新机制)

## 机制

v318 baseline + Sinkhorn-Knopp sk_eps 退火: 训练起点 0.5 (soft assignment, 多码字概率散布,
让 codebook 早期接触多样化数据) → 训练终点 0.05 (sharp assignment, 标准 RQ 量化). 50k 步线性.
其余沿用 v318 (cyclic c[0.3..1.0] period 50k, midpoint off).

第 92 次 R37 尝试, 第 7 次 iterN. 启用了 `_current_sk_eps` 字段 + `set_sk_eps_schedule()` API,
Step11 hook 每步更新所有层的 sk_eps.

## Stage 2 SID Gate (R50 4 项, 20k ckpt 早停)

| 指标 | TIGER baseline | iter7 20k | 状态 |
|------|---------------|-----------|------|
| Embedding HitRate@K=50 | 0.7165 | **0.7165** | SAME PASS |
| 3-token SID Gini | 0.0672 | **0.1245** | +85.3% **FAIL** |
| per_layer mean Gini | 0.2574 | [0.2342/0.3047/0.3861] | L1 PASS L2/L3 **FAIL** |
| per_layer codes | — | [72/256, 213/256, 197/256] | **L1 collapse** (72 < 218) |

**Gate 早期 FAIL — EARLY-STOP** (step=20000 时 SID gate 4 项 3 项 FAIL):
- L1 codes=72/256 严重 codebook collapse (28% utility, 远低于 85%×256=218 阈值)
- full_gini=0.1245 比 baseline 0.0672 差 85% (软 assignment 导致 SID 散布更广)
- per_layer Gini L2/L3 都超 baseline (0.3047/0.3861 vs 0.2574)

**机制失败原因分析**:
- 软 sk_eps=0.5 → Sinkhorn 不收敛到 one-hot, 多个码字共享概率质量
- Codebook 利用率低 (L1 仅 72/256), entropy 高 → SID 散度大 → R50 4 项全失败
- 与 v342 Sinkhorn-OT 失败同根: soft assignment 不适合离散 SID 生成

## Stage 3 测试

未执行 (R50 Gate FAIL 禁入 stage3).

## 决策: NO-GO

iter7 Sinkhorn 软退火在 20k 步已严重 collapse, 与 v342 (Sinkhorn-OT) 失败同根.
软 assignment 机制对 RQ-VAE 不友好 — codebook 需要硬选择才能形成稳定聚类.

下一步 iter8 候选:
- (a) 反向 sk_eps 退火 (0.05 → 0.5, sharp → soft, 鼓励 entropy 探索)
- (b) EMA codebook + Sinkhorn sk_eps 退火
- (c) 跳到 Gumbel-Softmax temperature 退火 (完全不同机制)
- (d) 4-layer RQ-VAE 架构变更
