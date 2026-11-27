# Issue #269 verdict

Scale-calibrated triplet curvature: distinguish local geometry from global distance
rescaling and validate hard SID.

## Headline

The #268 failure was a **global-scale artefact**. After one weighted-least-squares scale
`S* = 0.06828367045803868`, a per-triple curvature is identifiable on an independent
reference set: it beats one global curvature by **4.84 %** mean held-out J (centre bootstrap
95 % CI `[4.19 %, 5.54 %]`) with reference-perturbation Spearman **0.947**, and the full
28,391,948-row cache is no longer boundary-dominated (17.7 % boundary, 2.5 % ties, κ mean
`0.0288`, positive `60.7 %`). The calibrated candidate trained 3000 epochs (single card,
owner-authorised) and passed the SID gate with catalogue 3-code collision **0.1563**
(< 0.40). Stage3 result: **PENDING**.

## 1. Does scale calibration explain the #268 degradation?

Yes. #268 fitted in the chart `w = Z/(2S)` with `S = max||Z_ref|| = 0.5955`, where
`max||w|| = 0.5` and the legal negative curvature is bounded by `|k| < 1/0.25 = 4`; 99.78 %
of the fits hit the `-0.125` floor and one global Euclidean scale beat the whole family
(`J 0.1746` vs `0.2537`). At the calibrated `S*` the coordinate norm is `4.3605`, the legal
grid becomes `[-0.05, 1.0]`, and the same fit is informative: mean held-out J `0.173216`
vs global κ `0.182597` vs `κ=0` `0.181664`. The `S*` is reproducible from the closed form
`c* = Σ(t1+a_bc)/Σ(t2+a_bc²)` to `2.86e-7` relative error (`analysis/GATE.json`).

## 2. Does the local curvature provide extra information under an independent reference?

Yes at the gate: 4.84 % mean held-out improvement over the best single global κ,
`[4.19 %, 5.54 %]` bootstrap CI, Spearman `0.947` between refits on sets A and B, and 97.5 %
of full-data rows improve (`analysis/GATE.json`, cache manifest). But it is **partial**: a
free per-row scale still reaches J `0.148579` on the held-out set, below the fitted κ
`0.173216`, so curvature is a local-shape correction on top of a global ruler, not a complete
geometric model (`H269-3`).

## 3. Does the hard q / exported SID improve?

- **Training**: 3000 epochs, `1213.8 s`, mean `0.392 s/epoch` (single card, global batch 4096),
  final recon `9.54e-5` vs the #265 parent's `1.23e-4`.
- **Hard q's A.1 curvature proxy** (20k rows, independent references): Pearson `0.334`,
  Spearman `0.353`, sign accuracy `0.684`; the frozen `Z_ref` on the same rows is
  `0.093 / 0.084 / 0.548`. Training roughly quadruples the ranking correlation.
- **Distance distortion** `J` at the trained q is `0.288`, worse than `Z_ref`'s `~0.17`; the
  A.1 proxy and the raw distance distortion are different objectives.
- **Independent fixed Euclidean / equal-weight three-expert** full evaluation (28,391,948 rows,
  all shards): q Pearson `0.0273`, Spearman `-0.1110`, sign `0.3632`, MAE `2.0168`,
  RMSE `2.4698`; z Pearson `0.0245`, Spearman `-0.1189`, sign `0.3620`, MAE `2.0151`.
  The #265 parent under the same fixed metric was q Pearson `-0.0331`, Spearman `-0.0934`,
  sign `0.3323`, MAE `2.2460`. The fixed metric does not measure the trained κ geometry, and
  neither arm is good under it; ours is slightly better on Pearson/MAE/sign, slightly worse on
  Spearman.
- **SID**: catalogue 3-code collision `0.1563` (< 0.40), L3 triples `20,744`, L2 prefixes
  `6,246`, refinement gain `2.849` bits (parent `1.611`), L1 hit rate `0.0836`, L1 hard ER
  `5.72` (parent `8.66`). Collision is worse than the parent's `0.041` but inside the gate.
- **Graphlet fidelity**: triangle leaves vs open-wedge leaves at L1 `0.0895` vs `0.0274`
  (3.27x), comparable to frozen TIGER (`0.0990` vs `0.0240`); AF3 strata recorded.

## 4. Does recommendation improve?

`test_R@10` = **PENDING** (Stage3 single-card run `TIGER_calibrated_kappa_rq`, 150 epochs,
seed 42, beam 20). Comparison points: frozen TIGER `0.05520639286895663`, #265 parent
`0.052594926791900976`, target `> 0.065`.

## Acceptance table

| ID | Issue requirement | Evidence | Verdict |
|---|---|---|---|
| C1 | Recheck distance, row ids, Z_ref, references, mask, shortest path, target; directed vs symmetric | `CONTRACT.md` §1-2; `gate.py` | PASS |
| C2 | Freeze #265 encoder; deterministic split; independent references; one global S*; analytic 1/S; 0.068284 reproducible, not hard-coded | `analysis/GATE.json`: `scale_reproduced=0.06828365095335823`, rel. err `2.86e-7` | PASS |
| C3 | Freeze S*; coordinate map, κ=0 scale, legal domain; no illegal range widening | `CONTRACT.md` §2-3; cache manifest `k_min=-0.05`, `max_coord_norm=4.3605` | PASS |
| C4 | Same scale/sample: κ=0 vs one global κ vs per-triple; scale-invariant shape error; offline only | `GATE.json`: `0.181664 / 0.182597 / 0.173216 / 0.148579` | PASS |
| C5 | Frozen ranges/references/weights/tolerance; report held-out gain, quantiles, boundary/flat, sign, within-centre variance, structure strata, reference stability | `gate.py` constants; `GATE.json`; `PER_ROW.npz`; cache manifest statistics | PASS |
| C6 | Success: mean held-out improvement ≥1 %, bootstrap lower > 0, Spearman ≥0.5; report boundary; no post-hoc change | `GATE.json`: `0.048371`, `[0.041905,0.055416]`, `0.946566`, boundary `0.1543/0.2773` | PASS |
| C7 | Full shards, all legal triples, S*, κ, row ids, distortions, diagnostics, hashes, runtime, cache verification | `kappa_cache_s0p068284/` (28,391,948 rows, 115.3 s, 325 MB) + manifest | PASS |
| C8 | κ=0 degeneration if the gate fails; one candidate only | Gate passed -> per-triple arm only; no κ=0 arm trained | PASS |
| C9 | Point-value objective, fixed κ/S* on every distance, L=L_RQVAE+λL_curv, hard-q supervision, frozen scale/κ/targets | candidate wiring `w=q/(2S*)`; `STAGE2_REPORT.json` curvature calibration; scale regression test | PASS |
| C10 | κ=0 limit, signs, boundaries, coincident, reference agreement, index, frozen-scale test, gradient route, SID export | 121 candidate tests pass; probe 60 epochs at batch 4096; smoke entry PASS; `NOT_APPLICABLE` behavior graph | PASS |
| C11 | 3000 epoch, seed42, fixed batch; reconstruction/quantisation/collision/rank/usage/prefix/ER/graphlet/own and fixed metrics/time | `STAGE2_REPORT.json`, layer record, `OWN_KAPPA.json`, `CURVATURE_FULL_*`, `LOCAL_STRUCTURE.json` | PASS |
| C12 | One Stage3 150 epoch; freeze candidate, final test once; independent Recall/NDCG; compare TIGER/parent/0.065; disclose single-card and test-reuse risk | Stage3 run + `replay.py` | PENDING |
| C13 | Answer the four questions | this file | PENDING (Q4) |
| C14 | DIAGNOSIS/HYPOTHESES/CONTRACT, manifests, logs, code/tests, reports, R2 four-piece, VERDICT, push | this directory + `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/` + R2 | PENDING |
| C15 | Negative results honest; no unrun item as PASS; no end-to-end claim before legal Stage2/Stage3; 0.065 is a target | this file | PENDING |

## Closure checks (skill §6.4)

PENDING.

## Knowledge increment

- The #268 "no local curvature" conclusion was a scale artefact: at `S*` the per-triple fit
  transfers across independent reference sets and is no longer boundary-dominated.
- A calibrated per-triple κ on the hard q improves the A.1 curvature ranking roughly fourfold
  relative to the frozen representation, while the raw graph-distance distortion gets worse;
  the two must be reported separately, not conflated.
- The single-card NaN/floor stall was a numerical bug (`inf` in an unselected `torch.where`
  branch and `sqrt'(0)`), not a batch-size or framework fault.
