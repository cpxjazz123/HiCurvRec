# Task #316 — Issue #38 Arm δ1 (Stage 4 frequency prior rerank)

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — alpha=0.10 跟 baseline 完全一致, alpha≥0.30 NDCG 暴跌 (parallel session 创建, R11.5 跟进)
**Stage**: Stage 4 rerank hook (无 retraining, 即时验证)

## 背景

Issue #38 task314 description 4-arm 设计 (修订) Arm δ 立即可跑 (10 min/eval).
Arm δ1 = Frequency prior — P(item) ∝ log(freq) 调整 logit.

## 实现 (parallel session 创建, R11.5 follow-up 写 verdict)
- ckpt: task301 Issue #30 GO (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2])
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- Beam size: 50
- GPU 0
- log_freq from `Instruments_item_log_freq.npy` (pre-computed from train.history)
- Rerank hook: combined_score = beam_index * (1-alpha) - log_freq_score * alpha (lower better)

## Results (refer to verdict file)
| alpha | R@10 | Verdict |
|-------|------|---------|
| 0.00 (baseline) | 0.1041 | anchor |
| 0.10 | 0.1042 | in-sigma |
| 0.30 | 0.1039 | NDCG crash |
| 0.50 | 0.1027 | NDCG crash |
| 1.00 | 0.0388 | -62.7% |

## Verdict
**Arm δ1 frequency prior rerank NO-GO** — see `verdicts/task316_arm_delta1_freq_rerank_verdict.md` for full analysis.

## 关联
- Issue #38 (Stage 3/4 训练协议改造, OPEN)
- Task #314 (Issue #38 description, Arm δ 立即可跑)
- Task #317 (Arm δ2 embedding centroid rerank, 同步 NO-GO)
- Task #319 (Arm δ3 sequence diversity rerank, 同步 NO-GO)
- [[issue30-codebook-transforms-gate1-gate2-pass-gate3-training]]
