# Issue #277 — REPORT

Analysis-only, `INFRA_MODE=FROZEN`, no training, no Stage3. All #276 inputs are pinned to commit
`f62fd6ca`; `SID_TABLE.json` (`4c927914…`), `FOLDS.json` (`1a8fb81d…`), `ANALYSIS_276.json`
(`95d9492c…`) verified byte-equal to that commit (`AUDIT_277.json.pinned_inputs`).

## Deliverables

| item | path |
|---|---|
| frozen acceptance list | `RUN_LOG.md` |
| audit (metrics, identity, preprocessing) | `audit277.py`, `AUDIT_277.json` |
| feature contract + worked examples | `FEATURE_CONTRACT.md`, `tests/test_feature_examples.py` |
| coefficients / ablations / contributions | `explain277.py`, `COEFFICIENTS.json` |
| leak & TIGER reference audit | `LEAK_AND_REFERENCE.json` |
| frozen predictor | `PREDICTOR.json`, `predict277.py`, `tests/test_predictor.py` |
| frozen prospective predictions | `PROSPECTIVE.json` (committed `03c8e1df`, 2026-11-28 14:03:16 +1100) |
| hypotheses | `HYPOTHESES.json` |

## C1 — metric recomputation

Every stored OOF metric was recomputed from the stored prediction vectors and matches the archive
exactly (`matches_stored: true` for all 7 models × 2 queues, 1e-12 tolerance). The >0.06 confusion
table was recomputed **from the Ridge6 predictions** (the #276 script only stored PySR's):

| queue | model | tp | fp | fn | tn | precision | recall |
|---|---|---|---|---|---|---|---|
| Q_SID (13 pos) | ridge6 | 13 | 0 | 0 | 116 | 1.00 | 1.00 |
| Q_SID | PySR | 4 | 0 | 9 | 116 | 1.00 | 0.31 |
| Q_VEC (12 pos) | ridge6 | 12 | 0 | 0 | 61 | 1.00 | 1.00 |
| Q_VEC | PySR | 2 | 0 | 10 | 61 | 1.00 | 0.17 |

## C2 — identity audit (strong vs weak)

* **120 rows strong**: R2-ledger sha256 of `item_sids.json` or an inventory record binding.
* **9 rows weak**: 8 collision-equality only (`resample_s44`, `crowd_posattract_w3`,
  `resample_l1only`, `resample_s43`, `resample_fn_mask`, `crowd_couser_w3`, `resample_proj_dir`,
  `resample_proj_sym`) and 1 git-era name only (`bootstrap_code_stability__resample`).
  Collision equality is content-consistent evidence (73/73 cross-checks) but **not** a content hash;
  it cannot alone prove the SID↔Recall pairing, so these rows stay weak per the issue.
* **7 of the 9 weak rows are >0.06.** Consequently the strict subset contains only **6 of the 13**
  high-recall SIDs.
* Strict-subset OOF (stored ridge6 predictions restricted to the 120 strong rows):
  MAE 0.003188, ρ 0.552, **tp 6 / fp 0 / fn 0** (precision 1.0, recall 1.0 over the 6 covered positives).
  Weak-subset (9 rows): MAE 0.001342, ρ 0.500, 7/7 tp, 0 fp — reported for completeness, not as a
  strict result.
* Conclusion: the *13/13* statement is not fully strict; the strict result is **6/6 on the strong
  high-recall cores with 0 false positives**, and the remaining 7 positives are weak-identity.
  Both are reported; the strict number is the audited one.

## C3 — grouping, duplicates, reference

* Fold counts 27/27/26/25/24; every SID in exactly one fold; 0 duplicate groups; 75 groups carry a
  canonical mapping hash and **0 mappings span folds** (no near-duplicate leakage).
* Global code permutation leaves all six features unchanged (counts, prefix equality, entropies are
  relabeling-invariant), so permutation-equivalent SIDs cannot leak through features.
* TIGER reference: `0.0469` is a stored older report value with no matching `test_final.json`
  (protocol-era difference); `0.055206` is a Nov-23 Stage3 run **without recorded SID identity**;
  the canonical baseline SID `025ad8bd…` has label **0.051975** (primary, inventory unit) and
  `53273cb6…` 0.054719. **Same-protocol reference fixed to 0.051975** (canonical SID).

## C4 — preprocessing audit

#276's `inner_ridge_alpha` fitted `StandardScaler` on the whole outer-training block before the inner
split (inner-fold leak). Rerun with the scaler fitted inside every inner training fold:

| variant | alphas per fold | OOF MAE | OOF ρ | confusion >0.06 |
|---|---|---|---|---|
| #276 procedure (leaky inner scaler, reimplemented) | 10 / 0.01 / 0.01 / 0.1 / 0.01 | 0.003031 | 0.6228 | 13/13, fp 0 |
| corrected (scaler inside inner folds) | 1 / 0.1 / 0.01 / 0.1 / 0.01 | **0.002972** | **0.6666** | 13/13, fp 0 |
| stored #276 ridge6 (reference) | — | 0.003059 | 0.6160 | 13/13, fp 0 |

The leak did not inflate the result; the corrected variant is slightly better. Max |Δprediction|
leaky vs corrected 0.0039 (alpha choice changes per fold). No seed or fold change.

## C5 — feature definitions

`FEATURE_CONTRACT.md` gives exact formulas, denominators and masks; `l3_collision` is the duplicate
fraction `1 − unique/N` (not pairwise coincidence — demonstrated 0.6 vs 0.4 on a crafted SID);
`l1_hard_er = l1_pos_hit / l1_neg_hit` is a deterministic function of the two rates. Hand-computed
examples (ER 2.0, negative probabilities 0.5/0.25, entropies 1.585/2.585 bits, refine 1.0 bit) are
asserted in `tests/test_feature_examples.py` (4 tests, pass).

## C6 — formula and coefficients

Frozen 5 outer SID-group folds, inner alpha grid {0.01,0.1,1,10}, scaler inside inner folds.
Per-fold coefficients (standardised) and scaler are in `COEFFICIENTS.json`; NumPy recomputation of
`b + Σ w_i (x_i − mean_i)/scale_i` matches `sklearn.predict` to 0.0 (all folds).

| feature | mean standardised coef | fold std | direction |
|---|---|---|---|
| l1_pos_hit | +0.00812 | 0.00262 | higher → higher recall |
| l1_neg_hit | −0.00590 | 0.00221 | higher → lower recall |
| l1_hard_er | +0.00535 | 0.00038 | higher → higher recall (most stable) |
| l2_distinct | −0.00326 | 0.00043 | higher → lower recall |
| l3_collision | −0.00326 | 0.00133 | higher → lower recall |
| l3_refine_bits | +0.00009 | 0.00061 | ~0 |

Collinearity: corr(`l1_pos_hit`,`l1_neg_hit`) = 0.94; VIF 17.7 and 13.5; `l2_distinct`/`l3_refine_bits`
−0.82 (VIF 7.7/3.9). **The pos/neg coefficients are not independent effects** and the ER coefficient is
not causal: ER = pos/neg by construction. Ablations (same folds, corrected scaler):

| model | OOF MAE | OOF ρ |
|---|---|---|
| full 6 | **0.002972** | **0.667** |
| minus ER | 0.003993 | 0.558 |
| minus hit rates | 0.003372 | 0.445 |
| ER only | 0.003855 | 0.355 |
| best single (`l1_pos_hit`) | 0.004880 | −0.273 |
| mean / median | 0.004669 / 0.004422 | — |

SID-group bootstrap (2000): MAE 95% CI [0.002376, 0.003620]; TP 7–20 at the resampled positive counts;
**FP 0 in every resample**.

Contributions (frozen final predictor): e.g. `TIGER_r48k2_rq` (y 0.06476): prediction 0.06339 =
intercept 0.05266 + ER +0.00861 + `l3_collision` +0.00243 + `l1_neg_hit` +0.00193 − `l2_distinct`
−0.00164 − `l1_pos_hit` −0.00051 ≈ 0. Fixing ER alone would move the prediction by ~0.009 — the
dominant single term; the capacity terms act in the opposite direction on this arm.

## C7 — frozen predictor

`PREDICTOR.json`: final Ridge6 fit on the frozen 129-SID development queue, alpha 0.01 selected by the
corrected inner group CV; scaler mean/scale, standardised and raw-unit coefficients, intercept,
formula, training label/feature/table/folds hashes, domain (training min/max per feature), empirical
OOF residual quantiles (q05/q95; abs q90), and a `full_data_fit` block marked **in-sample, not a
performance estimate**. OOF reference: MAE 0.002972 / ρ 0.6666 / 13/13 (strict core 6/6).
`predict277.py` rejects missing/non-finite inputs, warns on out-of-domain values, and cross-checks the
raw-unit and standardised formulas at every call; `tests/test_predictor.py` (5 tests) passes.

## C8 — prospective predictions (frozen before labels)

`PROSPECTIVE.json` (commit `03c8e1df`, 2026-11-28 14:03:16 +1100) predicts **124 recoverable Stage2
SIDs without any public Stage3 label**, including `TIGER_pairrank_enc_rq` (SID `9f6bf673…`,
prediction **0.05319**, inside the training domain). At freeze time none of these SIDs had a Stage3
`test_final.json`; the Stage3 run in flight (started 13:00) had not produced a result. Reading their
labels later is the prospective check; no Stage3 is started by this issue.

## Limitations

1. Strict high-recall coverage is 6 SIDs; the other 7 positives are weak-identity and cannot be used
   as a strict result.
2. The frozen predictor is a 6-feature linear model on 129 SIDs with a narrow label range
   (0.0260–0.0648); the 0.065 criterion has no positives (NOT_EVALUABLE).
3. `l1_hard_er` is functionally dependent on the two hit rates; coefficients are descriptive, not
   causal, and the pos/neg pair is strongly collinear (VIF ~13–18).
4. Residuals are as large as ~0.018 on low-recall arms with good fidelity (over-prediction).
5. Bootstrap intervals are finite-sample and assume exchangeable SID groups.
