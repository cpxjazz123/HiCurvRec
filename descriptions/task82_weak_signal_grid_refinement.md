# Task #82 — Stage 1c Metric 弱信号灵敏度 + 网格加密 (Weak Signal Sensitivity + Grid Refinement)

> **任务目的**: 验证 Task #80 Stage 1c metric 能否识别**弱双曲信号** (真实 κ ∈ {-0.1,
> -0.15, -0.2}), 排除"phonism 真实 κ 被网格分辨率 + 收缩偏差压成 0"的可能性。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #81 v3 阳性对照 (用户批评, 2026-07-23):
- v3 best κ = -0.3 ~ -0.5, 但真实 κ = -1, **存在系统性"往 0 收缩"偏差** (~30-50% shrinkage)
- 当前 Stage 1c κ grid = {0, -0.3, -0.5, -1.0, -1.5, -2.0}, 0 和 -0.3 之间是**空的**
- 如果真实最优点在 κ = -0.1 附近, 网格根本测不到, metric 只能在"0"和"-0.3 起跳"之间选, 大概率选 0

**用户的核心质疑**: "metric 测出 κ=0 时, 确实意味着欧氏最优" 这句话目前**只支持强双曲信号
不会被误判成欧氏**, 不支持**弱双曲信号也不会被误判成欧氏**. 这两个是不同强度的检验, 后者才是
phonism 真正需要回答的问题 (因为 phonism 信号可能本来就因 SINKHORN + 小维度被压弱).

---

## 2. 实验设计

### Step A — 弱信号阳性对照 (Weak Signal Positive Control)

| 合成树 | 真实 κ | 预期 best κ (如果流程无偏差) | 实际 best κ (有 shrinkage) |
|--------|--------|---------------------------|--------------------------|
| Tree-A | -0.20 | κ ≈ -0.20 | ??? (可能被压到 -0.10 ~ -0.15) |
| Tree-B | -0.15 | κ ≈ -0.15 | ??? (可能被压到 -0.05 ~ -0.10) |
| Tree-C | -0.10 | κ ≈ -0.10 | ??? (可能直接压到 0) |

**判定**:
- **A1 (灵敏度足够)**: 3/3 树 best κ 落在 (0, κ_real] 区间 (即识别弱信号方向)
- **A2 (灵敏度部分足够)**: 1-2/3 树识别弱信号
- **A3 (灵敏度不足)**: 3/3 树 best κ = 0 (弱信号被直接压成欧氏)

### Step B — Phonism 真实数据网格加密

复用 Task #80 Stage 1c v2 框架 (RGD MDS + Kruskal stress-1), 但 κ grid 加密:

```
原 grid:  {0, -0.3, -0.5, -1.0, -1.5, -2.0}
加密 grid: {0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.5, -1.0, -1.5, -2.0}
```

4 层 (L0/L1/L2/L3) × 11 κ = 44 个 stress 值, 跑 ~10-15 min (GPU).

**判定**:
- **B1 (κ=0 维持)**: 4 层 best κ 仍是 0, 加最密的 -0.05/-0.1/-0.15/-0.2/-0.25 都不优
  → phonism 真实几何匹配欧氏的结论站得住
- **B2 (best κ 落入 -0.05 ~ -0.25)**: 之前因网格空隙漏掉了真实的弱信号最优点
  → phonism 是**弱双曲**, 不是欧氏, 需要重新评估
- **B3 (best κ 仍是 -0.5 或更深)**: 加密网格不影响结果, 维持原结论

---

## 3. 决策触发

| Step A | Step B | 综合决策 |
|--------|--------|---------|
| A1 (3/3 弱信号识别) | B1 (κ=0 维持) | ✅ phonism 结论站得住, 写最终 verdict |
| A1 | B2 (落入 -0.05~-0.25) | ⚠️ phonism 是弱双曲, 需重测 + 重写结论 |
| A1 | B3 (-0.5 或更深) | ✅ 维持原结论 |
| A2 (1-2/3 弱信号识别) | B1 | 🟡 部分可信, 加更多合成配置 |
| A2 | B2 | 🟡 弱信号 + 真实数据弱双曲, 双重确认 |
| A3 (3/3 弱信号被压回 0) | B1 | ❌ metric 有 detection floor, 无法区分"真欧氏" vs "弱双曲被压成欧氏" |
| A3 | B2 | ❌ 无法判断 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Step A 合成 + 跑 (CPU) | ~5 min | - |
| Step B 4 层 × 11 κ (GPU) | ~10-15 min | GPU 1 或 3 |
| 输出 JSON + 写 verdict | ~5 min | - |
| **总计** | **~25 min** | |

---

## 5. 风险与缓解

**风险 1**: Step A 弱双曲数据生成可能失败 (κ 太接近 0, 数据点 norm 太小, 几何特征不显著).
- 缓解: step_per_level 用更大值, 让 norm 接近 0.6-0.7

**风险 2**: Step B 加密网格跑 4 层 × 11 κ, 时间可能超过预算.
- 缓解: n_subset=300, n_iter=200 (跟原 v2 一致), 单次 evaluate < 1s

**风险 3**: 如果 Step A 失败 + Step B best κ 落入弱区间, 需要更深入诊断 (可能是 metric 设计根本限制).
- 缓解: 写 verdict 明确说明 sensitivity floor, 不强行下结论

---

## 6. 完成度跟踪

- [x] descriptions/task82_weak_signal_grid_refinement.md 写入
- [ ] scripts/task82_weak_signal_positive_control.py (Step A, CPU)
- [ ] scripts/task82_phonism_grid_refinement.py (Step B, GPU)
- [ ] 跑 Step A (3 棵弱双曲树 × 11 κ)
- [ ] 跑 Step B (4 层 × 11 κ 加密网格)
- [ ] 输出 JSON
- [ ] 写 verdict verdicts/task82_*_result.md
- [ ] 更新 task80 verdict §11 (弱信号灵敏度 + 网格加密)
- [ ] 根据 A1/A2/A3 + B1/B2/B3 综合决策, 修订 phonism 结论

---

## 7. 引用

- Task #81 v3 阳性对照 (best κ 缩水偏差 ~30-50%): verdicts/task81_positive_control_v3_phonism_metric_result.md
- Task #80 Stage 1c v2 metric: scripts/task80_stage1c_kappa_distortion_v2.py
- 用户批评原文: "缩水偏差意味着什么... 结论还不能直接采纳..." (2026-07-19)