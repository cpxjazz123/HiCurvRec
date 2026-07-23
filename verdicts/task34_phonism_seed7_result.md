# Task #34 — phonism/genrec TIGER 200ep seed=7 完成 verdict

> **完成日期**: 2026-07-19
> **状态**: ✅ 已完成（早停触发，best valid R@10 = 0.0635 @ epoch 59）

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
| **Valid (best)** | R@10 | **0.0635** | +24.5% over 0.051 |
| **Valid (best)** | epoch | 59 | (counter=10/10 → 早停 epoch 69) |
| **Test @ best valid (epoch 59)** | R@5 | **0.02981** | 87.7% of 0.034 |
| **Test @ best valid (epoch 59)** | R@10 | **0.04694** | 92.0% of 0.051 |
| **Test @ best valid (epoch 59)** | NDCG@5 | **0.01923** | - |
| **Test @ best valid (epoch 59)** | NDCG@10 | **0.02471** | - |
| **Test best across all evals** | R@5 (epoch 63) | **0.03007** | 88.5% of 0.034 |
| **Test best across all evals** | R@10 (epoch 63) | **0.04838** | 94.9% of 0.051 |

---

## 3. 执行时间线

| 事件 | 时间 |
|------|------|
| 启动 | 2026-07-19 18:28 |
| 早停触发 | 1:43:07 elapsed |
| 总耗时 | ~1h 43 min |
| 最终 best valid epoch | 59 |
| 早停 epoch | 69 (counter=10/10) |
| GPU | cuda:2 / 11 GB |

---

## 4. 分析解读

### 4.1 vs phonism baseline (R@10=0.051)

- **Test R@10 = 0.04694 (92.0% of phonism)**：接近 phonism 报告
- **Test R@5 = 0.02981 (87.7% of 0.034)**：略低于 phonism 报告
- **Valid R@10 = 0.0635 (+24.5% over 0.051)**：valid 远超 baseline

### 4.2 vs Task #87 baseline (R@5=0.01937)

- **Test R@5 = 0.02981 (+54.0% over Task #87)**：显著超越
- 表明 phonism/genrec 完整 TIGER 实现确实比当前 GRID baseline 更强

### 4.3 Val-Test Gap

- val R@10 = 0.0635 → test R@10 = 0.04694（gap = 26.1%）
- val R@5 = 0.0401 → test R@5 = 0.02981（gap = 25.7%）
- gap 与 phonism 报告一致（phonism 也有 val→test 显著 gap）

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 训练日志 | `logs/task34_phonism_seed7.log` |
| 模型 checkpoint | `/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys/seed7/` |
| Eval 记录 | logs 内 Test: 字段（每 2 epoch） |

---

## 6. 结论

✅ **phonism 报告 Toys R@5=0.034 的可复现性部分确认**：
- Test R@5 = 0.02981（88% of phonism 0.034）
- Test R@10 = 0.04694（92% of phonism 0.051）
- 剩余 8-12% 差距可能来自：训练步数差异 / batch size 差异 / 优化器超参差异

✅ **vs Task #87 baseline 显著改进 +54%**：phonism/genrec 完整 TIGER 实现确实更强。

---

**result:** Task #34 (seed=7) 完成 — Test R@5=0.02981 (88% of phonism), R@10=0.04694 (92% of phonism)

result: Task #34 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
