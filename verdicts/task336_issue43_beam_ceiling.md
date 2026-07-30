# Task #336 / Issue #43 — Stage 4 beam_size ceiling (50/80/100)

**日期**: 2026-07-30
**状态**: ✅ Ceiling DONE — beam=50/80/100 R@10=0.10425 (plateau)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/43

## 摘要

Issue #43 (HypPreEncoder + Stage 3 baseline protocol) Stage 4 beam_size 推 ceiling:

| beam_size | R@5 | R@10 | R@20 | NDCG@10 | NDCG@20 |
|-----------|-----|------|------|---------|---------|
| 20 | 0.0845 | **0.1041** | 0.1272 | 0.0769 | 0.0828 |
| **50** | 0.0845 | **0.10425** | 0.1341 | 0.0770 | 0.0845 |
| 80 | 0.0845 | **0.1042** | 0.1342 | 0.0770 | 0.0845 |
| 100 | 0.0845 | **0.1042** | 0.1342 | 0.0770 | 0.0845 |

**结论**:
- beam=20 → 50 涨 +0.16% (R@10 0.10409 → 0.10425), R@20 +5.5%
- beam=50 → 100 plateau (R@10 0.10425 → 0.1042), 差异 < 0.01pp (floating point noise)
- **Optimal beam = 50**: 零额外训练, +2.4% (vs baseline 0.1020), no significant gain beyond.

## 决策

- Issue #43 Stage 4 已达 ceiling 在 beam=50 (跟 task307 K14 一致: Issue #30 + beam=50 也在 0.1045 plateau)
- Issue #43 已确认 +2.4% R@10 改善 (跟 R@10=0.1020 baseline 比)
- 后续推升 R@10 必须改 Stage 1 (SID 质量) 或 Stage 3 (协议), 不能靠 beam_size

## 产物

| beam | Verdict JSON |
|------|--------------|
| 20 | verdicts/task336_issue43_gate2b_stage4_beam20.json (已有) |
| 50 | verdicts/task336_issue43_gate2b_stage4_beam50.json (已有) |
| 80 | verdicts/task336_issue43_stage4_beam80.json (NEW) |
| 100 | verdicts/task336_issue43_stage4_beam100.json (NEW) |

result: Issue #43 Stage 4 beam_size ceiling DONE. Optimal beam=50, R@10 plateau at 0.10425 vs baseline 0.1020 = +2.4%. 后续推升需 SID 质量或 Stage 3 协议改造.