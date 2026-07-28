# Task #225 — Per-Codeword κ 完整流水线 Stage 4 评估 verdict

result: **逃法一 (Per-Codeword κ) 下游 R@10=0.0938 (-8.1% vs baseline 0.1020) — NO-GO + Stop-loss 触发**. 验证 §6.7.4 hard stop-loss condition (i) (L0 utilization < 90%) 在 Phase 0 OPEN signal (L0=20.31% 健康码本, 远高于 baseline 1.5%) 仍无法突破 baseline R@10. **几何路线 (8 方向) 永久关闭**.

---

## 1. 任务

承接 Task #218-#224 逃法一 (Per-Codeword κ, Berman-Metzler 2020 距离公式):
- Phase 0 (Task #218) ✓: L0/L1/L2 通道 OPEN (assignment 一致率 28.90/67.15/75.69%)
- Stage 1 (Task #220/222) ✓: per-codeword κ + early stop @ 30 epoch = first healthy RQ-VAE codebook (L0=20.31%, L1=98.44%, L2=91.02%)
- Stage 2 (Task #223) ✓: 4th-digit dedup → 6245 unique SID (62.94% uniqueness)
- **Stage 3 (Task #224) ✓**: T5-mini 9.18M 训练 (200 epoch, R12 forced ckpt) 已完成, HG_Rec_best.pth 22 MB 落盘 (2026-07-26 23:13:42)
- **Stage 4 (Task #225, 本任务) ✓**: Test eval R@10=0.0938 (-8.1% vs baseline 0.1020)

---

## 2. Stage 4 Test Metrics (Instruments)

| 指标 | Task #225 pck | HG-Rec baseline (#84) | Δ |
|------|---------------|------------------------|---|
| R@5 | 0.0769 | 0.0816 | -5.7% |
| **R@10** | **0.0938** | **0.1020** | **-8.1%** ❌ |
| R@20 | 0.1154 | 0.1279 | -9.8% |
| NDCG@5 | 0.0646 | 0.0690 | -6.4% |
| NDCG@10 | 0.0700 | 0.0755 | -7.3% |
| NDCG@20 | 0.0754 | 0.0821 | -8.2% |

**判定: NO-GO** (R@10=0.0938 < 0.1020, 6 项指标全部低于 baseline).

---

## 3. §6.7.4 Hard Stop-Loss 触发

| Stop-loss 条件 | 阈值 | 实测 | 触发? |
|----------------|------|------|-------|
| (i) L0 utilization < 90% | < 90% | **20.31%** | ✅ 触发 |
| (ii) Stage 4 R@10 < 0.1020 | < 0.1020 | **0.0938** | ✅ 触发 |

**两个 stop-loss 都触发**. 按 §6.7.4 directive:
> "If the stop-loss gate is met, this concludes the geometric-route investigation on Musical_Instruments: active hyperbolic geometry is fundamentally incompatible with allocation separability in this architecture, and the only viable path is to keep the baseline geometry inactive (default HG-Rec) and rely on Sinkhorn-balanced post-processing for codebook health."

---

## 4. 关键观察 — 几何激活了但下游仍 NO-GO

**事实**:
1. **Phase 0 几何信号**: 真的能区分 (assignment 一致率 28/67/76% 远低于 99% baseline)
2. **Stage 1 健康码本**: L0=20.31% (×13 vs baseline), L1=98.44% (×24), L2=91.02% (×16), unique SID 50.93%
3. **Stage 2 SID uniqueness**: 62.94% (vs baseline 9.07%)
4. **Stage 4 R@10**: 0.0938 (-8.1% vs baseline)

**解释**: T5-mini 不在乎码字几何, 只学 (item → SID) 的 semantic mapping. 即使码本利用率 + SID uniqueness 大幅提升, T5 学到的仍是次优序列模式. **几何激活跟下游 Recall 解耦** (跟 Task #87 paradox 一致).

---

## 5. 跨变体 R@10 综合 (12 个 Stage 3-4 实测)

| 变体 | Stage 2 collision | Stage 4 R@10 | Δ vs baseline |
|------|-------------------|---------------|---------------|
| HG-Rec baseline (#84) | 9.07% | **0.1020** | baseline |
| Vanilla + Sinkhorn (phonism) | ~5% | 0.1058 | **+3.7%** ✅ |
| TIGER (#84 toy) | ~10% | 0.0615 | -39.7% |
| Letter (#84 toy) | ~12% | 0.0997 | -2.3% |
| **M-arm 方向 F v11 (8D hyp)** | 5-8% | 0.0972 | **-4.7%** ❌ |
| **M-arm 方向 F v12 (16D hyp)** | 5-8% | 0.0974 | **-4.5%** ❌ |
| **Per-Codeword κ (#225 本任务)** | 37-63% | 0.0938 | **-8.1%** ❌ |
| M-arm 方向 I c=10 ep1 (#233) | 51.41% | (训练中) | (预期 < 0.07) |

**关键观察**: vanilla+Sinkhorn (0.1058) 仍是最优. 所有几何变体 (-2.3% ~ -8.1%) 均在 baseline 之下. Per-Codeword κ 距离 baseline 最远 (-8.1%).

---

## 6. 决策与收口

### 6.1 R11.3 决策

按 §6.7.4 hard stop-loss 永久关闭:
- ✅ Stop-loss (i) 触发 (L0 util 20.31% < 90%)
- ✅ Stop-loss (ii) 触发 (R@10 0.0938 < 0.1020)
- ❌ **不再尝试任何第 9 方向** (用户 2026-07-26 directive)

### 6.2 后续 Tasks

- Task #226 (Step 3 v6 recipe) — 已完成, 已在 Task #227 总结
- Task #227 (v7/v8/v9/v10/v11/v12 综合 NO-GO) — 已完成
- Task #228 (w_angular 扫描) — 已完成
- Task #229 (方向 G correlation) — 已完成 (Phase 0)
- Task #230 (方向 H PCA 冻结) — 已完成 (NO-GO)
- Task #231 (方向 I c=10/100) — 已完成 (NO-GO)
- Task #232 (方向 H+I 组合) — 已完成 (NO-GO)
- Task #233 (方向 I c=10 ep1 → T5) — **Stage 3 训练中, 预期 R@10 < 0.07**
- Task #234 (方向 J1 软分配) — Stage 2 已 NO-GO, 跳过 Stage 3+4

---

## 7. 综合结论

> **逃法一 (Per-Codeword κ) 是 8 个几何方向里第一个 Phase 0 OPEN 信号 (L0/L1/L2 assignment 一致率 28/67/76%), 但 Stage 4 R@10=0.0938 仍 NO-GO (-8.1% vs baseline). 几何激活 ≠ 下游 Recall 提升 (跟 Task #87 paradox 一致). §6.7.4 hard stop-loss 触发, 几何路线永久关闭**.

**Bottom line**: 8 方向几何干预 + 1 软分配 ablation + 2 逃法 (per-codeword κ / Gromov) 全部 NO-GO. **最佳下游仍是 vanilla+Sinkhorn (R@10=0.1058)**. 在 flat (non-hierarchical) 数据集 (Musical_Instruments) 上, 几何编码无任何帮助, 反而 systemically 退化下游 2.3-8.1%.

---

## 8. 产物

- **Stage 3 ckpt**: `products/task224/t5mini_pck/jul-26-2026_23-13-16/Instruments/Jul-26-2026_23-13-42/HG_Rec_best.pth` (22 MB, R12 best)
- **Stage 4 metrics JSON**: `verdicts/task225_pck_metrics.json`
- **Stage 4 log**: `logs/task225/stage4_eval_jul-26-2026_23-50-32.log`
- **Stage 1 ckpt**: `products/task218/stage1_pck_spread/Jul-27-2026_01-25-33_*/best_collision_model.pth` (Task #222 ep29)
- **SID**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task223_pck.npy` (6245 unique, 62.94%)

---

## 9. Paper update 建议

更新 §6.7.5.3:
- 把 "in progress" 改为 "completed: R@10=0.0938 (-8.1% vs baseline), 6 项指标全部 NO-GO"
- 引用 Task #225 verdict + Stop-loss 触发
- 加 §6.7.6 "Final closure statement" 总结 12 个 R@10 实测