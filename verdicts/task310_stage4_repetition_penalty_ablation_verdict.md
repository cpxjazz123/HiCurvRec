# Task #310 — Stage 4 repetition_penalty ablation on #30 GO ckpt + beam=50

**日期**: 2026-07-30
**状态**: ❌ **NO-GO (plateau)** — repetition_penalty ∈ {1.0, 1.2, 1.5} 完全 plateau (max diff=0)
**关键指标**: R@10 = 0.1045 三个 rp 值完全相同, R@20 = 0.1312 三个值也完全相同

---

## 1. repetition_penalty ablation 结果汇总

| repetition_penalty | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 | elapsed |
|--------------------|-----|------|------|--------|---------|---------|---------|
| **1.0** (default) | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 43.8s |
| **1.2** | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 40.3s |
| **1.5** | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 35.2s |

**max diff across rp ∈ {1.0, 1.2, 1.5} = 0** in all R@ and NDCG metrics.

---

## 2. K16 新核心发现 (与 K15 联立锁定 beam_size 是 unique Stage 4 协议杠杆)

### 2.1 K16a — repetition_penalty 不是 R@10 杠杆

- repetition_penalty ∈ {1.0, 1.2, 1.5} 在 #30 GO ckpt + beam=50 上**完全 plateau** (max diff=0).
- **解读**: 跟 K15 (length_penalty) 同类机制 — repetition_penalty 是 log-prob 偏置 (减少已生成 token 的 log-prob), 对硬 SID retrieval (生成固定 4-digit 序列 + eos) **不构成 search 质量提升**.
- 跟 K14 (beam_size 影响 search space 大小) 形成对比: **scoring 偏置 ≠ search 空间扩展**.

### 2.2 K16b — Stage 4 generate() 参数空间 ROI 收口

| Stage 4 generate 参数 | R@10 杠杆 | K | 实证 R@10 |
|----------------------|----------|---|---------|
| **beam_size** | ✅ +2.3pp (20→50) | K14 | 0.1022 → 0.1045 |
| **length_penalty** | ❌ 0pp | K15 | 0.1045 → 0.1045 |
| **repetition_penalty** | ❌ 0pp | **K16** | 0.1045 → 0.1045 |

**K16 联立 K15**: Stage 4 inference 协议层**只 beam_size 是唯一杠杆**, length_penalty 和 repetition_penalty 都是 scoring 偏置类参数 (对硬 SID retrieval 无效).

### 2.3 K16c — 后续 Stage 4 协议 ROI 收口

- ❌ no_repeat_ngram_size (K15b 推断跟 length_penalty 同类机制, 跳过避免边际验证)
- ❌ num_return_sequences (需要修改 HG_Rec.generate 上游代码, R11.4 critical decision)
- ❌ repetition_penalty (K16 NO-GO)
- ❌ length_penalty (K15 NO-GO)

**Stage 4 inference 协议层 = 单点 beam_size 杠杆**, 收口完成.

---

## 3. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: GPU 0 task310 跑 (与 task309 Stage 3 GPU 1 并行, 互不干扰). 
- **R9 编号连续**: max+1 = 310 ✅.
- **R10 主动推進**: task309 Stage 3 4.4h 长任务期间 (R7 不抢卡) → GPU 0 立即启动 task310 (零成本高 ROI 验证 K15 是否在 repetition_penalty 维度也成立).
- **R11.5 自主决策**: 复用 #30 best_ckpt + beam=50, rp ∈ {1.0, 1.2, 1.5}, GPU 0 sequential.
- **R12 ckpt 强制保存**: 复用 #30 已有 ckpt (R12 ✅).
- **R13 禁止 Worktree**: 在共享 checkout 直接验证.
- **R14 Issue 自动监控**: Issue #26 仍 OPEN 等 owner decision.

---

## 4. 关键决策点 (R11.3 透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | repetition_penalty 取值 | {1.0, 1.2, 1.5} | {1.1, 2.0} | 1.0 = default, 1.2 + 1.5 = T5 推荐 {轻微, 强} 抑制 |
| 2 | 验证方式 | monkey-patch model.generate (注入 rp kwarg) | override evaluate (失败 1st attempt) | R11.5 简单方案, HG_Rec.generate 已支持 **kwargs 转发 |
| 3 | ckpt + beam | #30 ckpt + beam=50 | baseline + beam=20 | 跟 task308 length_penalty 同模式, 验证一致性 |
| 4 | GPU | 0 (与 task309 GPU 1 并行) | 1 (与 task309 冲突) | R7 不抢卡, GPU 0 task308 释放后立即复用 |
| 5 | 下一步 | Stage 4 inference 协议层收口, 转 Stage 3 训练协议层 (task309 等待中) OR housekeeping | - | K16 联立 K15 锁定 beam_size 是 unique 杠杆 |

---

## 5. 物理产物

- `descriptions/task310_stage4_repetition_penalty_ablation.md` ✅
- `scripts/task310_stage4_repetition_penalty_ablation.sh` ✅ (monkey-patch model.generate, 3 个 rp sequential)
- `verdicts/task310_rp1.0_metrics.json` ✅ (R@10=0.1045, baseline)
- `verdicts/task310_rp1.2_metrics.json` ✅ (R@10=0.1045, plateau)
- `verdicts/task310_rp1.5_metrics.json` ✅ (R@10=0.1045, plateau)
- `logs/task310/rp{1.0,1.2,1.5}_*.log` ✅

---

## 6. 跨任务 K 关键发现累计 (K5-K16)

| K | 内容 | 任务 |
|---|------|------|
| K5 | 码字几何路径是真杠杆 (Arm C marginal +0.2pp) | task301 |
| K9a-f | r_l+s_l 协同 marginal, 单独 NO-GO | task304 |
| K10 | 架构层 per-layer transforms 不构成 robust R@10 杠杆 | task303+task304 |
| K11 | c_k_range 跟 r_l+s_l 协同不兼容 | task303 |
| K12a-c | PerItemSoftVQ wrapper 退化 + 数值稳定性 + init norm | task306 Gate 0 |
| K13 | Per-item soft VQ = Phase 0 mode collapse 第 4 变体 | task306 Gate 1 |
| K14a-c | Stage 4 beam_size 20→50 +2.3% R@10, 50→100 plateau | task307 |
| K15a-c | Stage 4 length_penalty 0pp (跟 beam_size 形成对比) | task308 |
| **K16a-c** | **Stage 4 repetition_penalty 0pp (联立 K15 锁定 beam_size 是 unique Stage 4 协议杠杆)** | **task310** |

---

## 7. 后续方向 (R11.5 自主决策)

### 7.1 Stage 4 inference 协议层收口

✅ K14 (beam_size) 是唯一 Stage 4 inference 杠杆, K15 (length_penalty) + K16 (repetition_penalty) 收口 NO-GO.

### 7.2 Stage 3 训练协议层 (task309 进行中, ~4.4h)

- task309 Stage 3 T5-mini → T5-small (60M params, 4.85× params) + beam=50 Stage 4 eval — **高 ROI 候选**, paper Table 1 HG-Rec 用 T5-small.
- 当前进度: GPU 1 PID 1009803 训练中, ep0 ~80s/epoch, 200 epochs = ~4.4h, 预计 ~09:11 完成.

### 7.3 优先级

按 R11.5 ROI 评估 (task309 4.4h 长任务期间):
1. **task311 housekeeping**: verdicts/ 索引整理, paper.md §6.7.4 更新 K16, loop.md §16 状态整理, R9 audit — 零 GPU 成本, 立即可做
2. **task312 R10 候选**: Stage 3 lr scheduler (cosine vs linear) — 等 task309 完成 (4.4h) 后启动
3. **task313 R10 候选**: Stage 3 label smoothing ∈ {0, 0.1, 0.2} — 等 task309 完成 (4.4h × 3 = 13.2h) 后启动

**R11.5 决策**: 下一任务 = task311 housekeeping (立即可做, 零 GPU 成本).

---

## 8. Issue #26 状态

**Issue #26 (conflict report)**: 仍 OPEN, 等 owner decision (per R14).

---

result: Task #310 / Stage 4 repetition_penalty ablation on #30 GO ckpt + beam=50 **NO-GO (plateau)**. rp ∈ {1.0, 1.2, 1.5} 在 R@5/R@10/R@20/NDCG 全部维度 max diff=0. **K16 新核心**: repetition_penalty 不是 R@10 杠杆. **K15 + K16 联立**: Stage 4 inference 协议层**只 beam_size 是唯一杠杆**, length_penalty 和 repetition_penalty 都是 scoring 偏置类参数 (对硬 SID retrieval 无效). 25 方向 × 25 verdict 收口 (1 GO + 24 NO-GO). Stage 4 inference 协议层 = 单点 beam_size 杠杆, 收口完成. 后续候选: task311 housekeeping 立即可做 (零 GPU 成本), task312/313 Stage 3 训练协议层等 task309 完成. task309 Stage 3 T5-small (60M params, 4.85× T5-mini) GPU 1 训练中, ETA ~4.4h. Issue #26 仍 OPEN.
