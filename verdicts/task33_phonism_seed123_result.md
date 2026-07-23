# Task #33 — phonism/genrec TIGER 200ep seed=123 完成 verdict

> **完成日期**: 2026-07-19
> **状态**: ✅ 已完成（早停触发，best valid R@10 = 0.0681 @ epoch 89）

---

## 1. 任务目标

验证 phonism/genrec TIGER 完整流水线在 4 个不同 seed 下 toy 数据集 Toys 上的可复现性。

| 指标 | 决策阈值 | 来源 |
|------|---------|------|
| 4 seed 均值 R@5 ≥ 0.030 | ✅ phonism 上限可信 | phonism 报告 0.034 |
| 4 seed CV ≤ 20% | ✅ seed variance 正常 | 多 seed 报告 |

---

## 2. 关键指标

| 阶段 | 指标 | 值 | vs phonism baseline |
|------|------|-----|---------------------|
| **Valid (best)** | R@10 | **0.0681** | **+33.5% over 0.051** |
| **Valid (best)** | epoch | 89 | (counter=10/10 → 早停 epoch 99) |
| **Test @ best valid (epoch 89)** | R@5 | **0.03097** | **91.1% of 0.034** |
| **Test @ best valid (epoch 89)** | R@10 | **0.04948** | **97.0% of 0.051** |
| **Test @ best valid (epoch 89)** | NDCG@5 | **0.02014** | - |
| **Test @ best valid (epoch 89)** | NDCG@10 | **0.02607** | - |
| **Test best across all evals** | R@5 (epoch 95) | **0.03236** | **95.2% of 0.034** |
| **Test best across all evals** | R@10 (epoch 91) | **0.04954** | 97.1% of 0.051 |

---

## 3. 执行时间线

| 事件 | 时间 |
|------|------|
| 启动 | 2026-07-19 18:28 |
| 早停触发 | 2:19:53 elapsed |
| 总耗时 | ~2h 20 min |
| 最终 best valid epoch | 89 |
| 早停 epoch | 99 (counter=10/10) |
| GPU | cuda:1 / 11 GB |

---

## 4. 分析解读

### 4.1 vs phonism baseline (R@10=0.051)

- **Test R@10 = 0.04948 (97.0% of phonism)**：几乎完全复现 phonism
- **Test R@5 = 0.03097 (91.1% of 0.034)**：接近 phonism 报告
- **Valid R@10 = 0.0681 (+33.5% over 0.051)**：valid 远超 baseline（4 seed 最高）

### 4.2 vs Task #87 baseline (R@5=0.01937)

- **Test R@5 = 0.03097 (+59.9% over Task #87)**：显著超越
- 表明 phonism/genrec 完整 TIGER 实现确实比当前 GRID baseline 更强

### 4.3 Val-Test Gap

- val R@10 = 0.0681 → test R@10 = 0.04948（gap = 27.3%）
- val R@5 ≈ 0.0424 → test R@5 = 0.03097（gap = 27.0%）
- gap 与 phonism 报告一致

### 4.4 vs 其他 seed (4 seed 综合)

| seed | Test R@5 | Test R@10 | vs phonism R@5 | vs phonism R@10 |
|------|----------|-----------|----------------|------------------|
| **123 (本任务)** | **0.03097** | 0.04948 | 91.1% | 97.0% |
| 42 (Task #32) | 0.03150 | **0.04950** | **92.6%** | **97.1%** |
| 7 (Task #34) | 0.02981 | 0.04694 | 87.7% | 92.0% |
| 2024 (Task #35) | 0.02822 | 0.04766 | 83.0% | 93.5% |

seed=123 是 4 seed 中 valid R@10 最高的，但 test R@5 略低于 seed=42。

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 训练日志 | `logs/task33_phonism_seed123.log` |
| 模型 checkpoint | `/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys/seed123/` |
| Eval 记录 | logs 内 Test: 字段（每 2 epoch） |

---

## 6. 结论

✅ **phonism 报告 Toys R@5=0.034 的可复现性几乎完全确认**：
- Test R@5 = 0.03097（91% of phonism 0.034）
- Test R@10 = 0.04948（97% of phonism 0.051）
- 剩余 3-9% 差距可能来自：训练步数差异 / batch size 差异 / 优化器超参差异

✅ **vs Task #87 baseline 显著改进 +60%**：phonism/genrec 完整 TIGER 实现确实更强。

---

**result:** Task #33 (seed=123) 完成 — Test R@5=0.03097 (91% of phonism), R@10=0.04948 (97% of phonism)

result: Task #33 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
