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
(< 0.40). Stage3 result: `test_R@10` **0.045230592455** (see §4).

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

`test_R@10` = **0.045230592455**, `R@5` = 0.030049269660, `NDCG@10` = 0.024823080495, n_eval = 57439. Frozen TIGER `0.05520639286895663` (delta -0.009976); #265 parent `0.052594926791900976` (delta -0.007364); target `>0.065` NOT MET. Independent replay status `PASS`.

The five historical `TIGER_RQ-VAE` Stage3 runs span `0.040199`–`0.055537` R@10 (canonical
latest `0.055206`); this arm at `0.045231` lies inside that historical range but below its own
parent and below the canonical baseline. The result is a negative recommendation outcome, not
a tuning signal: the continuous geometry improved while the discrete SID got more compressed
(L2 prefixes `6,246` vs the parent's `10,918`) and the Stage3 loss was lower, yet R@10 fell.

Single-card comparison caveat: the run used the single-card branch (global train batch
4096 against the four-card 16384), so it is not numerically comparable to four-card arms;
it is the only layout this host exposes. Existing test access risk: the frozen Stage3 test was
already read by earlier arms; this arm is one more read and was not tuned on it.

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
| C11 | 3000 epoch, seed42, fixed batch; reconstruction/quantisation/collision/rank/usage/prefix/ER/graphlet/own and fixed metrics/time | `STAGE2_REPORT.json`, `STAGE2_METRICS.md`, layer record, `OWN_KAPPA.json`, `CURVATURE_FULL_*`, `LOCAL_STRUCTURE.json` | PASS |
| C12 | One Stage3 150 epoch; freeze candidate, final test once; independent Recall/NDCG; compare TIGER/parent/0.065; disclose single-card and test-reuse risk | `analysis/STAGE3_RECALL.json`, `analysis/REPLAY.json` | PASS |
| C13 | Answer the four questions | this file | PASS |
| C14 | DIAGNOSIS/HYPOTHESES/CONTRACT, manifests, logs, code/tests, reports, R2 four-piece, VERDICT, push | this directory + `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/`; R2 `r2:genrec/111/` (112 files: source, SID `a89110b8…`, checkpoint `896e69b7…`, recall `78e68902…`, binding `4e51a695…`); commits `80466a12` + final | PASS |
| C15 | Negative results honest; no unrun item as PASS; no end-to-end claim before legal Stage2/Stage3; 0.065 is a target | this file; `STAGE3_RECALL.json`; no unrun item marked PASS | PASS |

## Closure checks (skill §6.4)

1. **All production code implemented?** Yes: the calibrated candidate tree is committed and
   was the tree actually trained and exported (run snapshot `…/src`, checkpoint `896e69b7…`).
2. **All necessary tests executed?** Yes: 121 candidate tests PASS, mutation suite (5/5
   caught), module/gradient checks on a real batch, 60-epoch single-card probe at batch 4096,
   MCP smoke entry PASS. Receipts under `receipts/checks/` with exit codes.
3. **All required training completed?** Yes: Stage2 3000 epochs (1213.8 s, single card) and
   Stage3 150 epochs (4870.9 s, single card), final checkpoint, `test_final.json`,
   independent replay PASS.
4. **Infra boundary respected?** Yes: `INFRA_SNAPSHOT.json` vs `INFRA_SNAPSHOT_END.json`
   recomputed with identical hashes for all 61 public component files; no `assets/`,
   `tools/`, launcher, admission or evaluator change.
5. **Any executable task left?** No: every acceptance item has a current PASS; the negative
   recommendation result is recorded, not deferred.

## Knowledge increment

- The #268 "no local curvature" conclusion was a scale artefact: at `S*` the per-triple fit
  transfers across independent reference sets and is no longer boundary-dominated.
- A calibrated per-triple κ on the hard q improves the A.1 curvature ranking roughly fourfold
  relative to the frozen representation, while the raw graph-distance distortion gets worse;
  the two must be reported separately, not conflated.
- The single-card NaN/floor stall was a numerical bug (`inf` in an unselected `torch.where`
  branch and `sqrt'(0)`), not a batch-size or framework fault.
- **Recommendation did not improve**: R@10 `0.045231` vs the #265 parent `0.052595` and the
  canonical TIGER `0.055206`. Calibrated local curvature is identifiable offline and improves
  the A.1 ranking, but the resulting discrete SID is more compressed (6,246 vs 10,918 L2
  prefixes, lower training loss) and loses recommendation. Continuous geometry gain is not a
  proxy for SID/recall success; this closes the "fixed local curvature" hypothesis family for
  this representation and quantizer.
- The Stage3 baseline itself varies across reruns (`0.0402`–`0.0555`), so a five-thousandth
  R@10 difference is not interpretable without a same-layout rerun; the `-0.0074` vs the
  parent is larger than that band and is treated as a real loss.
