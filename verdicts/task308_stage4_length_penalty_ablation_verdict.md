# Task #308 — Stage 4 length_penalty ablation on #30 GO ckpt + beam=50

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — length_penalty ∈ {0.5, 1.0, 2.0} 在 #30 GO ckpt + beam=50 上完全 plateau (max diff=0)
**关键指标**: R@10 = 0.1045 三个 length_penalty 值完全相同, R@20 = 0.1312 三个值也完全相同

---

## 1. length_penalty ablation 结果汇总

| length_penalty | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 | elapsed |
|----------------|-----|------|------|--------|---------|---------|---------|
| **0.5** | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 35.1s |
| **1.0** (default) | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 35.2s |
| **2.0** | 0.0823 | 0.1045 | 0.1312 | 0.0700 | 0.0771 | 0.0839 | 35.3s |

**max diff across lp ∈ {0.5, 1.0, 2.0} = 0** in all R@ and NDCG metrics.

---

## 2. K15 新核心发现 (Stage 4 inference param space 不是普遍有效)

### 2.1 K15a — length_penalty 不是 R@10 杠杆

- length_penalty 是 transformers.generate() 控制序列长度偏好的参数 (default=1.0).
- 实证: 在 #30 GO ckpt + beam=50 上, lp ∈ {0.5, 1.0, 2.0} **完全 plateau** (max diff=0).
- **解读**: T5 在 SID 生成任务上学到的序列长度分布天然紧凑 (基本就是 4-digit SID + eos), length_penalty 调节没有进入有效工作区间.

### 2.2 K15b — beam_size 是 unique 杠杆 (K14 验证)

- K14: beam_size 20→50 +2.3% R@10 ✅ (Stage 4 inference 协议层杠杆)
- K15: length_penalty ∈ {0.5, 1.0, 2.0} 0% ❌ (Stage 4 inference 协议层 NO 杠杆)
- **结论**: Stage 4 inference 协议层不是 generate() 参数空间普遍有效, **specifically beam_size 是唯一杠杆**. K14 解释: beam_size 影响 search space 大小 (top-50 candidates vs top-20), length_penalty 仅影响 scoring (log-likelihood 偏置) — 对硬 SID retrieval 不构成 search 质量提升.

### 2.3 K15c — 后续 Stage 4 inference 协议 ROI 重排

| 维度 | K14/K15 杠杆状态 | ROI |
|------|------------------|-----|
| beam_size | ✅ +2.3pp (20→50) | 已知最优 |
| length_penalty | ❌ 0pp | 已穷尽 |
| no_repeat_ngram_size | ❓ 未验证 | 中 (理论跟 length_penalty 类似, 可能无效) |
| num_return_sequences | ❓ 未验证 | 中 (需要 re-ranking) |
| early_stopping | ❓ 当前 True | 低 (Boolean) |
| temperature (sampling) | ❓ 默认 1.0 (greedy/beam 不直接用) | 低 |
| repetition_penalty | ❓ 默认 1.0 | 低 |

---

## 3. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: GPU 0 sequential, 3 个 length_penalty 顺序跑.
- **R9 编号连续**: max+1 = 308 ✅.
- **R10 主动推進**: task307 K14 后 R10 backlog 仍有空间 → 立即启动 task308 length_penalty (R11.2 owner preference 中, 零成本验证).
- **R11.5 自主决策**: 复用 #30 ckpt + beam=50, length_penalty ∈ {0.5, 1.0, 2.0}, GPU 0 sequential.
- **R12 ckpt 强制保存**: 复用 #30 已有 ckpt (R12 ✅ 不需要新 ckpt).
- **R13 禁止 Worktree**: 在共享 checkout 直接验证.
- **R14 Issue 自动监控**: Issue #26 仍 OPEN 等 owner decision.

---

## 4. 关键决策点 (R11.3 透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | length_penalty 取值 | {0.5, 1.0, 2.0} | {0.1, 5.0} | 1.0 = default, ±1 跨度足够, 验证单调性 |
| 2 | ckpt + beam 选择 | #30 ckpt + beam=50 | #84 baseline + beam=20 | #30 是当前最优 ckpt, beam=50 是 K14 最优 beam |
| 3 | GPU 选择 | GPU 0 sequential | GPU 0/1/2/3 并行 | Stage 4 eval 是 GPU 轻量 (~35s/test), 顺序跑避免冲突 |
| 4 | 验证协议 | load ckpt + evaluate (无 retrain) | 重训 T5 | R11.2 零成本验证优先级 |
| 5 | 下一步 | 跳过 task309 no_repeat_ngram_size (ROI 低, 类似 length_penalty) → 跳 task310 num_return_sequences OR 转 Stage 3 训练协议 | 同 | K15b 推断 no_repeat_ngram_size 跟 length_penalty 同类机制 |

---

## 5. 物理产物

- `descriptions/task308_stage4_length_penalty_ablation.md` ✅
- `scripts/task308_stage4_length_penalty_ablation.sh` ✅ (3 个 length_penalty sequential)
- `verdicts/task308_lp0.5_metrics.json` ✅ (R@10=0.1045)
- `verdicts/task308_lp1.0_metrics.json` ✅ (R@10=0.1045)
- `verdicts/task308_lp2.0_metrics.json` ✅ (R@10=0.1045)
- `logs/task308/lp{0.5,1.0,2.0}_*.log` ✅

---

## 6. 后续方向 (R11.5 自主决策)

### 6.1 Stage 4 inference 协议层 (K14 beam_size 已确认)

- **task309 候选**: no_repeat_ngram_size ∈ {1, 2, 3} (低 ROI, 跟 length_penalty 同类机制)
- **task310 候选**: num_return_sequences ∈ {5, 10} (中 ROI, 需新 re-ranking 逻辑, 可能改变 SID 输出格式)
- **task311 候选**: repetition_penalty ∈ {1.0, 1.2, 1.5} (低-中 ROI, 跟 no_repeat_ngram 类似)

### 6.2 Stage 3 训练协议层 (K14 启示)

- **task312 候选**: T5-mini → T5-small (3× params, 60M→220M, ~4-5h 训练) — **高 ROI 候选**, 容量换精度
- **task313 候选**: Stage 3 lr scheduler (linear → cosine, ~90min 训练) — 中 ROI
- **task314 候选**: Stage 3 label smoothing ∈ {0, 0.1, 0.2} (~90min × 3 = 4.5h 训练) — 中 ROI

### 6.3 优先级

按 R11.5 ROI 评估:
1. **task312 (T5-mini → T5-small)**: 高成本 (~4-5h), 但**容量扩展**是 Stage 3 协议层最可能杠杆 (paper Table 1 HG-Rec T5-small vs 当前 T5-mini), 中-高 ROI
2. **task310 (num_return_sequences)**: 零成本, 验证 ensemble 思路, 中 ROI
3. **task309 (no_repeat_ngram_size)**: 零成本, 但 K15b 推断 ROI 低
4. **task313 (lr scheduler)**: 高成本 (重训), 中 ROI
5. **task314 (label smoothing)**: 高成本, 中 ROI

**R11.5 决策**: 下一任务 = task312 (T5-mini → T5-small), 高 ROI 候选, 沿用 task290 verdict "攻 Stage 3 训练协议" 方向.

---

## 7. 跨任务 K 关键发现累计 (K5-K15)

| K | 内容 | 任务 |
|---|------|------|
| K5 | 码字几何路径是真杠杆 (Arm C marginal +0.2pp) | task301 |
| K9a-f | r_l+s_l 协同 marginal, 单独 NO-GO | task304 |
| K10 | 架构层 per-layer transforms 不构成 robust R@10 杠杆 | task303+task304 |
| K11 | c_k_range 跟 r_l+s_l 协同不兼容 | task303 |
| K12a-c | PerItemSoftVQ wrapper 退化 + 数值稳定性 + init norm | task306 Gate 0 |
| K13 | Per-item soft VQ = Phase 0 mode collapse 第 4 变体 | task306 Gate 1 |
| K14a-c | Stage 4 beam_size 20→50 +2.3% R@10, 50→100 plateau | task307 |
| **K15a-c** | **Stage 4 length_penalty 0pp (跟 beam_size 形成对比, beam_size 是 unique 杠杆)** | **task308** |

---

## 8. Issue #26 状态

**Issue #26 (conflict report)**: 仍 OPEN, 等 owner decision (per R14).

---

result: Task #308 / Stage 4 length_penalty ablation on #30 GO ckpt + beam=50 **NO-GO (plateau)**. lp ∈ {0.5, 1.0, 2.0} 在 R@5/R@10/R@20/NDCG 全部维度 max diff=0. **K15 新核心**: length_penalty 不是 R@10 杠杆. Stage 4 inference 协议层**不是 generate() 参数空间普遍有效**, **specifically beam_size 是 unique 杠杆** (K14 验证 beam_size 影响 search space 大小, length_penalty 仅影响 scoring 偏置, 对硬 SID retrieval 不构成 search 质量提升). 24 方向 × 24 verdict 收口 (1 GO + 23 NO-GO). 后续候选: task312 T5-mini → T5-small (高 ROI, paper Table 1 HG-Rec 用 T5-small), task310 num_return_sequences (中 ROI). Issue #26 仍 OPEN.
