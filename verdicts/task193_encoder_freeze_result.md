# Task #193 — Encoder freeze 补充实验 (因果验证)

> **任务目的**: 冻结 encoder → 消除漂移源 → 验证 collision 是否停止劣化。

> **完成日期**: 2026-07-26
> **状态**: ✅ 已完成

---

## 结果

| 配置 | 最佳碰撞率 |
|------|-----------|
| Baseline (free) | 9.08% |
| Freeze encoder (epoch 60) | 8.72% |

**结论**: Encoder freeze 对 collision 无实质影响 (8.72% vs 9.08%)。Task #191 假设"encoder 漂移 → 码字滞后 → collision"被否定。HG-Rec 的碰撞稳定机制与 encoder 是否漂移无关。

**不推进 Stage 2+3+4**: 无有意义差异。

result: Task #193 — Encoder freeze 补充实验 (因果验证)
