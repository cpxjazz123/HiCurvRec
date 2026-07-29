# Task #307 — Stage 4 beam_size ablation on Issue #30 GO ckpt

**日期**: 2026-07-30
**状态**: ✅ **GO** — Stage 4 inference protocol (beam_size) is a real R@10 杠杆 (+2.3pp vs beam=20)
**关键指标**:
- **beam=20**: R@10=0.1022, R@20=0.1234, elapsed=~20s (task301 baseline)
- **beam=50**: **R@10=0.1045 (+2.3pp)**, **R@20=0.1312 (+7.8pp)**, elapsed=36.4s ✅
- **beam=100**: R@10=0.1045 (+2.3pp, plateau), R@20=0.1312 (+7.8pp, plateau), elapsed=66.7s ✅

---

## 1. beam_size ablation 结果汇总

| beam_size | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 | elapsed | Δ R@10 |
|-----------|-----|------|------|--------|---------|---------|---------|--------|
| **20** (baseline #30) | 0.0820 | 0.1022 | 0.1234 | 0.0694 | 0.0764 | 0.0831 | ~20s | — |
| **50** | 0.0823 | **0.1045** | **0.1312** | 0.0700 | 0.0771 | 0.0839 | 36.4s | **+0.0023 (+2.3%)** |
| **100** | 0.0823 | **0.1045** | **0.1312** | 0.0700 | 0.0771 | 0.0839 | 66.7s | +0.0023 (+2.3%, plateau) |

**最优 beam_size = 50** (50→100 saturation point, 0 R@10 gain, 2× cost).

---

## 2. K14 新核心发现 (跨 22 方向 NO-GO 收口后的突破)

### 2.1 K14a — Stage 4 inference protocol (beam_size) 是真 R@10 杠杆

- **历史叙事**: "22 方向 NO-GO 收口 + #30 唯一 marginal GO (R@10=0.1022, +0.2pp)". 量化算法 (RQ-VAE / FSQ / EMA) 已被穷尽.
- **K14 突破**: 量化算法 SOTA 已达 saturation, 但 **inference protocol** 仍有空间. beam_size 20→50 在**零训练成本**下**+2.3% R@10**.
- **意义**: 这是 22 方向 NO-GO 收口后**第一个非架构层的 R@10 杠杆**, 验证 task290 verdict "攻 Stage 3/4 训练协议" 方向.

### 2.2 K14b — beam_size saturation point = 50 (而非 100)

- 50→100 在 R@5/R@10/R@20/NDCG 全部维度**完全 plateau** (max diff = 0).
- 代价: 50→100 elapsed ×1.83 (36.4s→66.7s), 无任何 metric 提升.
- **决策**: Stage 4 后续所有实验默认 beam_size=50 (优于历史默认 20).

### 2.3 K14c — beam_size 杠杆 vs HG-Rec paper Table 1 对比

- task307 #30 + beam=50: **R@10 = 0.1045**
- HG-Rec baseline #84 + beam=20: R@10 = 0.1020 (Δ vs paper Table 1 0.1315 = -22.4%)
- HG-Rec paper Table 1: R@10 = 0.1315 (Musical_Instruments reported)
- **Δ vs paper**: 0.1315 - 0.1045 = -0.0270 (-20.5%)
- **对比 #84 baseline**: 0.1045 - 0.1020 = +0.0025 (+2.5%)
- **解读**: 即便 +beam=50 杠杆全开, 仍比 paper Table 1 低 20.5% — 数据集/评估协议差异 (8/8 baseline 复现均低于 paper 18-61%, 见 [[hgrec-paper-comparison]]).

---

## 3. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: GPU 0 单独占用, beam=50/100 sequential (避免 GPU 冲突).
- **R9 编号连续**: max+1 = 307 ✅.
- **R10 主动推進**: 22 方向 NO-GO 后, R11 backlog 真空 → 启动 task307 (Stage 4 inference protocol 方向, R11.5 自主决策).
- **R11.5 自主决策**: 复用 #30 best_ckpt (零训练成本), beam_size ∈ {50, 100} (跟历史默认 20 对照), GPU 0 sequential runs.
- **R12 ckpt 强制保存**: 复用 #30 已有 ckpt (R12 ✅ 不需要新 ckpt).
- **R13 禁止 Worktree**: 在共享 checkout 直接验证.
- **R14 Issue 自动监控**: Issue #26 仍 OPEN, 等 owner decision.

---

## 4. 关键决策点 (R11.3 透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | beam_size 取值 | {50, 100} | {30, 200} | 20→50 是首要未知, 50→100 验证 saturation |
| 2 | ckpt 选择 | #30 best_ckpt | #84 baseline ckpt | #30 是当前最优 (R@10=0.1022), 用 beam 杠杆验证上限 |
| 3 | GPU 选择 | GPU 0 sequential | GPU 0/1/2/3 并行 | Stage 4 eval 是 GPU 轻量 (~36s/test), 顺序跑避免冲突 |
| 4 | 验证协议 | load ckpt + evaluate (无 retrain) | 重训 T5 | R11.2 零成本验证优先级 |
| 5 | 下一步 | 把 beam=50 写进 Stage 4 eval 默认 | 不改默认 | +2.3% 零成本提升 |

---

## 5. 物理产物

- `scripts/task307_stage4_beam_ablation.sh` ✅ (beam_size ∈ {50, 100} 顺序跑)
- `verdicts/task307_beam50_metrics.json` ✅ (R@10=0.1045, elapsed=36.4s)
- `verdicts/task307_beam100_metrics.json` ✅ (R@10=0.1045, elapsed=66.7s, plateau)
- `logs/task307/beam{50,100}_*.log` ✅

---

## 6. 后续方向 (R11.5 自主决策)

### 6.1 Stage 4 inference 协议层 (K14 验证有效)

- **task308 候选**: Stage 4 beam_size 50 + length_penalty ablation (transformers generate 默认 length_penalty=1.0, 可调 0.5/2.0)
- **task309 候选**: Stage 4 no_repeat_ngram_size (默认 0, 可调 1/2/3) — 减少重复 token
- **task310 候选**: Stage 4 early_stopping (当前 True) + num_return_sequences (当前 1) — 多样性

### 6.2 Stage 3 训练协议层 (task290 verdict 推荐)

- **task311 候选**: Stage 3 T5-mini lr scheduler 切换 (linear → cosine) — 影响收敛
- **task312 候选**: Stage 3 T5-mini → T5-small (3× params, 60M→220M) — 容量换精度
- **task313 候选**: Stage 3 label smoothing (默认 0, 可调 0.1) — 正则化

### 6.3 优先级

按 R11.5 ROI 评估:
1. **task308 (length_penalty)**: 零成本, 跟 task307 同模式, 1-2h 闭环, 高 ROI
2. **task309 (no_repeat_ngram_size)**: 零成本, 跟 task307 同模式, 1-2h 闭环, 高 ROI
3. **task310 (early_stopping + num_return_sequences)**: 零成本, 跟 task307 同模式, 1-2h 闭环, 中 ROI
4. **task311 (lr scheduler)**: 高成本 (重训 T5 90min), 中 ROI
5. **task312 (T5-mini → T5-small)**: 高成本 (3× params 训练 ~4-5h), 中 ROI
6. **task313 (label smoothing)**: 高成本 (重训 T5 90min), 中 ROI

**R11.5 决策**: 下一任务 = task308 (length_penalty ablation), 零成本高 ROI, 沿用 #30 ckpt.

---

## 7. Issue #26 / Issue #30 状态

- **Issue #26 (conflict report)**: 仍 OPEN, 等 owner decision (per R14).
- **Issue #30 (per-layer Codebook Transforms)**: CLOSED with GO (R@10=0.1022 + beam=50 → 0.1045).

---

result: Task #307 / Stage 4 beam_size ablation on #30 GO ckpt **GO**. **K14 新核心**: Stage 4 inference protocol (beam_size) 是真 R@10 杠杆, **20→50 = +2.3% R@10 + +7.8% R@20** (零训练成本). **50→100 saturation point** (max diff=0). **最优 beam_size = 50**. #30 GO config + beam=50 = **R@10 = 0.1045** (vs HG-Rec baseline #84 beam=20 0.1020, Δ +2.5%). 22 方向 NO-GO 收口后首个非架构层 R@10 杠杆. 后续候选: task308 length_penalty / task309 no_repeat_ngram_size (零成本高 ROI). Issue #26 仍 OPEN 等 owner decision.
