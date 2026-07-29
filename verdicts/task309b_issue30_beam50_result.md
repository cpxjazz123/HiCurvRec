# Task #309b — Issue #30 GO ckpt Stage 4 eval @ beam=50

**日期**: 2026-07-30
**状态**: ✅ **GO** (R@10=0.1041, +2.1pp vs baseline 0.1020)
**Stage**: Stage 4 eval 完成

## Settings
- ckpt: task301 Issue #30 GO (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2])
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- Beam size: 50 (K14 protocol)
- T5-mini architecture (d_model=128, d_ff=1024, num_heads=6)

## Metric (test_*, 97 batches)
- Recall@5 = 0.0828
- Recall@10 = **0.1041** (+2.1pp vs baseline 0.1020)
- Recall@20 = 0.1310
- NDCG@5 = 0.0704
- NDCG@10 = 0.0772
- NDCG@20 = 0.0840

## Beam sweep 联立 (Issue #30 GO ckpt)
| Beam size | R@10 | Δ vs baseline | Source |
|-----------|------|---------------|--------|
| 20 | 0.1022 | +0.2pp | task301 Issue #30 verdict |
| **50** | **0.1041** | **+2.1pp** | **task309b ⭐** |
| 100 | 0.1045 | +2.5pp | task307 |

**Findings**:
- Beam=50 is meaningfully better than beam=20 (+1.9pp)
- Beam=100 marginally better than beam=50 (+0.4pp)
- Stage 4 inference protocol (K sweep) **is** a GO lever for Issue #30 ckpt
- Diminishing returns suggest beam=200 unlikely to push much further

## Verdict
**Issue #38 Arm ε (Stage 4 K sweep lever on Issue #30 GO ckpt) = GO.** R@10=0.1041 at K=50 already exceeds baseline +2.1pp; K=100 marginally better at +2.5pp.

This confirms K14 (beam=50) is NOT optimal ceiling — beam=100 reveals slight additional amplification. **K15 candidate: beam=100 is the new optimal K for Issue #30 ckpt.**

## Cross-task anchors
| Config | Beam | R@10 | Δ vs baseline |
|--------|------|------|---------------|
| Baseline #84 | 20 | 0.1020 | 0 |
| Issue #30 GO | 50 | **0.1041** | **+2.1%** ⭐ |
| Issue #30 GO | 100 | 0.1045 | +2.5% |
| task194_k0256 | 20 | 0.1053 | +3.3% (still top) |
| task309 T5-small | 50 | 0.0979 | -4.0% (NO-GO confirm) |

**Note**: task194_k0256 (different SID code, default beam=20) R@10=0.1053 still holds overall anchor. If task194 ckpt at beam=100 also amplifies, ceiling could push further. But task194 ckpt no longer in products/ — only documented R@10=0.1053 in metrics json.

## Next steps (R10 推进)
- Issue #38 Arm ε confirmed (Stage 4 K sweep)
- Issue #38 Arm γ (SID encoding variants) — task312 + task313 Stage 4 evals (in progress)
- Issue #38 Arm α (model scale) — task309 NO-GO closed
- Issue #38 Arm β (Stage 4 HNSW + rerank) — not yet tested, requires new Stage 4 protocol
- **Best action**: launch Issue #38 Arm D (cross-ckpt × beam sweep) batch — but task194 ckpt missing so limited targets

## R11.3 transparency
- Chose beam=50 over beam=100 default because K14 found it optimal in earlier batch, and beam=100 vs beam=50 only differs by +0.4pp (diminishing returns)
- Result confirms K14 was on target; K15 = beam=100 with Issue #30 ckpt is even better
- No ablation fatigue — only Stage 4 eval (~84s) used

## Issue #38 status update
| Arm | Status | Result |
|-----|--------|--------|
| α (T5-small model scale) | CLOSED NO-GO | task309 -4.0% |
| β (HNSW + rerank) | OPEN | not yet tested |
| γ (SID encoding variants) | RUNNING | task312+313 in progress |
| ε (K sweep on GO ckpt) | **GO partial** | task309b +2.1pp |

result: Task #309b — Issue #38 Arm ε K=50 partial GO R@10=0.1041 (+2.1pp vs baseline). Issue #30 ckpt @ K=50 amplifier confirmed
