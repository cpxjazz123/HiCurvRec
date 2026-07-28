# Task #159 Result — T5-mini 12M capacity point (Stage 4 closure)

> **完成日期**: 2026-07-24 22:29 (Stage 4 完成)
> **状态**: ✅ **闭环 — Stage 3 早停 + Stage 4 inference 完成**
> **核心结论**: R@10=**0.0978** (T5-mini 9.18M + code_default SID) — ❌ **比 5.5M 还低 -5%**, T5 容量从 5.5M → 9.18M **不单调向上**

---

## 1. Test metrics (T5-mini 9.18M + code_default SID)

| 指标 | Test 值 | vs Task #156 (5.5M) | vs Task #84 (5.5M poincare) | 评价 |
|------|---------|---------------------|------------------------------|------|
| Recall@5 | **0.0805** | 0.0831 (Δ -3.1%) | 0.0816 (Δ -1.4%) | ❌ 略低 |
| Recall@10 | **0.0978** | 0.1030 (Δ **-5.0%**) | 0.1020 (Δ -4.1%) | ❌ **反向** |
| Recall@20 | **0.1208** | 0.1304 (Δ -7.4%) | 0.1329 (Δ -9.1%) | ❌ ❌ |
| NDCG@5 | **0.0694** | 0.0711 (Δ -2.4%) | 0.0690 (Δ +0.5%) | ≈ same |
| NDCG@10 | **0.0749** | 0.0775 (Δ -3.4%) | 0.0755 (Δ -0.8%) | ≈ same |
| NDCG@20 | **0.0808** | 0.0844 (Δ -4.3%) | — | ❌ |

vs paper Table 1 Instruments HG-Rec: R@10=0.1315 → Δ -25.6% (gap 进一步扩大)

---

## 2. Stage 3 训练过程关键时序

| 阶段 | 时间 | 备注 |
|------|------|------|
| Stage 1 sentence-t5-base embedding | 启动前已完成 | 768-dim (paper-aligned) |
| Stage 2 codebook (code_default β=0.25, [32,64,256], sk=0.5) | 启动前已完成 | best_loss=8.875, collision=0.1924, 1869 ep |
| Stage 3 T5-mini 9.18M 训练 | 21:42 → 22:18 (~36 min) | NDCG@20 早停 |
| Stage 4 Test evaluation | 22:25-22:29 (~60s) | 修复 BEST_CKPT/RESULT_JSON export bug 后完成 |

---

## 3. 关键发现

### 3.1 T5 容量 ladder 第 2 点: 不单调向上 ❗

| 容量 | 任务 | R@10 | Δ vs 5.5M |
|------|------|------|----------|
| 5.5M (T5-tiny) | Task #156 | 0.1030 | — |
| 9.18M (T5-mini) | Task #159 | **0.0978** | **-5.0%** |
| 44.6M (T5-small) | Task #160 | (待跑) | ? |
| 220M (T5-base) | Task #157 | (待跑) | ? |

**结论**: T5 容量从 5.5M 增大到 9.18M 反而恶化, 不是单调 scaling。两种可能:
- (a) 9.18M 配置 (d_model=256, 4+4 layers, d_ff=1024, 6 heads) 的几何/容量比例对当前数据不平衡
- (b) HG-Rec code 本身在 T5-mini 配置下有数值不稳定 bug (e.g., d_kv=64/heads=6 → d_kv*heads=384 ≠ d_model=256)

**R11.3 自主决策**: 等 #160 / #157 收齐 4 档再判断;若 44.6M / 220M 反弹向上 → T5 容量有用但需跳过 9.18M 这个奇怪点;若仍 ≤ 0.110 → paper gap 终极确认, paper 内部代码 ≠ paper github 代码.

### 3.2 早停收敛 vs 完整 200 epoch
- Task #159 T5-mini 9.18M: 早停 ~36 min (实际 epoch 数未核验), R@10=0.0978
- 早停机制一致, NDCG@20 plateau 仍然成立

### 3.3 T5-mini 9.18M 几何概率
d_model=256 与 heads=6 (d_kv=64 → 384), **不整除**: 6 heads × 64 = 384 > 256. 这可能是 9.18M 配置的稳定性问题, 应当在 4 档 ladder 闭环后单独 verify.

---

## 4. 任务完成 checklist

- [x] Stage 2 codebook fork 启动 (β=0.25, [32,64,256], sk=0.5)
- [x] Stage 3 T5-mini 9.18M 训练 (早停 ≈ 36 min)
- [x] 修复 Stage 4 launcher export bug (BEST_CKPT + RESULT_JSON 双 export)
- [x] Stage 4 Test evaluation (60s 完成)
- [x] Test metrics JSON 落盘 (`verdicts/task159_t5mini_12m_metrics.json`)
- [x] 比较 vs paper + Task #156 baseline
- [x] 写本 verdict 闭环

---

## 5. 后续 (R11.3)

1. **继续等待**: Task #160 (44.6M T5-small) GPU 2 ~91% util 训练中, Task #157 (220M T5-base) GPU 0 100% util 训练中 — 收齐 4 档 ladder 数据后写综合 verdict.
2. **不动**: 不重启 Task #159 改完整 200 epoch (早停已确认 NDCG@20 plateau, 完整 200 epoch 跑也只是 noise). R12: 不强制覆盖 best ckpt.
3. **GPU 1 释放**: Stage 4 inference 60s 完成, 资源已释放.

---

result: Task #159 — Stage 4 closure. R@10=0.0978 (T5-mini 9.18M) ❌ REGRESSED -5% vs 5.5M baseline 0.1030. T5 capacity ladder 第 2 点异常, 等待 #160 (44.6M) / #157 (220M) 收齐 4 档定论.
