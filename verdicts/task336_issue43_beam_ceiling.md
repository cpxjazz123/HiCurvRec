# Task #336 / Issue #43 — Stage 4 beam_size ceiling (5/10/20/50/80/100)

**日期**: 2026-07-30
**状态**: ✅ Ceiling DONE — beam=50 saturation point, +2.4% R@10 vs baseline 0.1020
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/43

## 摘要

Issue #43 (HypPreEncoder + Stage 3 baseline protocol) Stage 4 beam_size 推 ceiling, 6 个 beam_size 综合:

| beam_size | R@10 | Δ vs prev beam | Δ vs baseline 0.1020 |
|-----------|------|----------------|-----------------------|
| 5 | 0.0767 | — | -25.0% |
| 10 | 0.0974 | +27.0% | -4.5% |
| 20 | 0.1041 | +6.9% | +2.1% |
| **50** | **0.10425** | +0.16% | **+2.4%** (optimal) |
| 80 | 0.10425 | 0% | +2.4% |
| 100 | 0.10425 | 0% | +2.4% |

**结论**:
- beam=5 → 10 → 20 大幅增长 (saturation search space filling)
- beam=20 → 50 微涨 +0.16% (R@10 0.10409 → 0.10425), R@20 +5.5%
- **beam=50 完全 plateau** (50 → 80 → 100 R@10 全在 0.10425, 差异 < 0.01pp floating point noise)
- **Optimal beam = 50**: 零额外训练, +2.4% (vs baseline 0.1020), no significant gain beyond.

## 决策

- Issue #43 Stage 4 已达 ceiling 在 beam=50 (跟 task307 K14 一致: Issue #30 + beam=50 也在 0.1045 plateau)
- Issue #43 已确认 +2.4% R@10 改善 (跟 R@10=0.1020 baseline 比)
- 后续推升 R@10 必须改 Stage 1 (SID 质量) 或 Stage 3 (协议), 不能靠 beam_size

## 产物

| beam | Verdict JSON |
|------|--------------|
| 5 | verdicts/task336_issue43_stage4_beam5.json (NEW 2026-07-30) |
| 10 | verdicts/task336_issue43_stage4_beam10.json (NEW 2026-07-30) |
| 20 | verdicts/task336_issue43_gate2b_stage4_beam20.json (已有) |
| 50 | verdicts/task336_issue43_gate2b_stage4_beam50.json (已有) |
| 80 | verdicts/task336_issue43_stage4_beam80.json (NEW 2026-07-30) |
| 100 | verdicts/task336_issue43_stage4_beam100.json (NEW 2026-07-30) |

result: Issue #43 Stage 4 beam_size ceiling DONE 6-point curve. Saturation at beam=50 R@10=0.10425 (+2.4% vs baseline 0.1020). 后续推升 R@10 必须改 SID 质量或 Stage 3 协议, 不能靠 beam_size.