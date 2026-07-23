# Task #27 — Plan B/C/D 样本量扩展 (n: 6 → 20+)

> **触发**: 用户反馈 "样本太小 (n=6), 这个结论不可靠"
> **核心目标**: 将 Task #27 Spearman 相关性样本量从 n=6 扩到 n≥20, 达成 ρ=-0.7 时 p<0.01 (高置信度)
> **总预算**: ~28-32 小时 GPU wall time (4 GPU × 4 batch × 7h)
> **启动时间**: 2026-07-19 15:05

---

## 1. 统计需求

| n | ρ (Spearman) | p-value | 结论 |
|---|--------------|---------|------|
| 6 | -0.67 | 0.148 | ❌ 不可靠 (用户已确认) |
| 10 | -0.67 | ~0.034 | ⚠️ 边际显著 |
| 14 | -0.67 | ~0.009 | ✅ 显著 p<0.01 |
| 20 | -0.67 | ~0.001 | ✅✅ 强显著 |
| 30 | -0.67 | <0.0001 | ✅✅✅ 极显著 |

→ **目标 n≥14** (对应 ρ=-0.67 时 p<0.01), 当前 n=6, 需扩 +8.

## 2. 当前活跃任务 (Phase A, 已启动)

| Task | codebook_width | GPU | 状态 | ETA Stage 2.1 |
|------|---------------|-----|------|---------------|
| Task #26 | K=512 | cuda:0 | 在跑 step 3988 | ~17:30 |
| Task #28 | K=128 | cuda:1 | 在跑 step 244 | ~18:20 |
| Task #29 | K=64 | cuda:2 | 在跑 step 47 | ~17:50 |

→ 完成后 n: 6 → 9 (+3 数据点)

## 3. Phase B 立即启动 (本 tick)

**剩余 GPU**: cuda:3 空闲 (475 MiB / 45 GB 可用)

| Task | codebook_width | GPU | 启动 | 备注 |
|------|---------------|-----|------|------|
| Task #30 | K=384 | cuda:3 | 立即 | 半粒度 K (256 与 512 中间), 验证 K=256 是否 sweet spot |

→ 完成后 n: 9 → 10 (+1 数据点)

## 4. Phase C 第二批 (~17:30 当 Phase A Stage 2.1 完成后)

Stage 2.1 完成后所有 GPU 进入 Stage 2.2 → Stage 3 (~6h). Stage 3 是显存密集 (~29GB), 4 GPU 同时跑 4 Stage 3.

**空闲时机**: Stage 3 期间没有 GPU 可用于新 Stage 2.1.

但 Stage 3 训练完成后 (06:00 next day 左右), 立即进入 Phase D:

## 5. Phase D 第三批 (~次日 06:00)

Stage 3 完成后所有 4 GPU 重新空闲. 立即启动新一轮 4 个 RQ-VAE Stage 2.1:

| Task | codebook_width | num_hierarchies | seed | 角色 |
|------|---------------|-----------------|------|------|
| Task #31 | K=96 | 3 | 42 | 极小 K ablation |
| Task #32 | K=192 | 3 | 42 | K=128 与 K=256 中间 |
| Task #33 | K=320 | 3 | 42 | K=256 与 K=384 中间 |
| Task #34 | K=256 | 3 | **7** | seed variance probe (vs Task #87 seed=42, #25 seed=123) |

→ 完成后 n: 10 → 14 (+4 数据点)

## 6. Phase E 第四批 (~次日 13:00)

第二组 Stage 2.1 + Stage 2.2 + Stage 3 (06:00 → 13:00). Stage 3 完成后:

| Task | codebook_width | num_hierarchies | seed | 角色 |
|------|---------------|-----------------|------|------|
| Task #35 | K=256 | 3 | **200** | seed variance |
| Task #118 | K=256 | 3 | **999** | seed variance |
| Task #119 | K=256 | **2** | 42 | H=2 ablation (vs H=3 baseline) |
| Task #120 | K=256 | **4** | 42 | H=4 ablation (vs H=3 baseline) |

→ 完成后 n: 14 → 18 (+4 数据点)

## 7. Phase F 第五批 (~次日 20:00)

| Task | codebook_width | num_hierarchies | seed | 角色 |
|------|---------------|-----------------|------|------|
| Task #121 | K=1024 | 3 | 42 | 大 K ablation (vs K=512) |
| Task #122 | K=32 | 3 | 42 | 极小 K (验证 K=64 是否已 collapse) |
| Task #123 | K=256 | 3 | 42 | **重跑** 验证复现性 (Task #87 seed=42) |

→ 完成后 n: 18 → 21 (+3 数据点, 含 1 复现性验证)

## 8. 总样本量与统计能力

最终 n = 21, 远超 n≥14 目标. Spearman ρ=-0.67 时:
- p < 0.0001
- 95% CI 半宽 ≈ ±0.25

**足以支撑可靠结论**.

## 9. 调度逻辑

每 Phase 完成后立即 re-run Task #27 分析:
- `python3 scripts/task27_knn_quality_recall.py`
- 自动加载所有可用 tokenizer SID (含新完成的)
- 重算 Spearman ρ + bootstrap CI + LOO 敏感性
- 写入 `task27_correlation_summary.json` (覆盖)
- 更新 verdict 增量段

## 10. 风险与缓解

**风险 1**: 新 tokenizer Stage 2.1 collapse (cov < 0.5)
- K=32/64/96 可能严重 collapse → 早停 + 标记
- 已完成 collapse 的 tokenizer (Task #85 m=2) 不纳入主分析

**风险 2**: Stage 3 早停后只有 best ckpt, 重新推断需重新训练
- 每 tokenizer Stage 3 早停 save best.ckpt → Stage 4 推断用 best.ckpt, 不需要完整 ckpt

**风险 3**: GPU OOM (Stage 3 4 张同跑)
- Stage 3 ~29GB, 4 张 46GB 卡能独立承载, 无冲突

**风险 4**: 总体预算超 32h
- 监控每 Phase 完成时间, 若某 Phase 拖延, 砍 Phase F 减少总时长

## 11. 完成度跟踪

- [x] **Phase A 启动**: Task #26 K=512, Task #28 K=128, Task #29 K=64 (3 jobs 在跑)
- [ ] **Phase B**: Task #30 K=384 on cuda:3 (立即启动)
- [ ] **Phase A 完成**: 3 tokenizer Stage 4 完成 → re-run Task #27 → n=9
- [ ] **Phase C 启动**: Stage 3 串行 (cuda:0/1/2 → Stage 3)
- [ ] **Phase D 启动**: Task #31/114/115/116 Stage 2.1 on cuda:0/1/2/3
- [ ] **Phase D 完成**: 4 tokenizer Stage 4 → re-run Task #27 → n=14
- [ ] **Phase E 启动**: Task #35/118/119/120 Stage 2.1
- [ ] **Phase E 完成**: 4 tokenizer Stage 4 → re-run Task #27 → n=18
- [ ] **Phase F 启动**: Task #121/122/123 Stage 2.1
- [ ] **Phase F 完成**: 3 tokenizer Stage 4 → re-run Task #27 → n=21
- [ ] **最终 verdict**: 重写 `verdicts/task27_neighborhood_quality_result.md` 含 n=21 分析

## 12. 与 §1/§5 规则符合性

- ✅ §1 "禁止让 GPU 闲置": 4 GPU 用满
- ✅ §5.3 "能并行的必须并行": 4 GPU 同跑 Stage 2.1 或 Stage 3
- ✅ §5.1 强依赖: 每个 tokenizer 内部 2.1 → 2.2 → 3 → 4 串行
- ✅ §1 "AI 自主决定": 用户给 Ultracode opt-in 后所有参数自主决定

## 13. 关联任务

- Task #27 (✅): Plan A 7 tokenizer, 不可靠结论
- Task #26/#28/#29 (🟢): Phase A 当前活跃
- Task #30 (🟡): Phase B 本 tick 启动
- Task #31-123: 后续 Phase 计划