# Issue #277 — VERDICT

## Answers to the four questions

1. **What are the six metrics?** SID-only quantities from the frozen evaluator: `l1_pos_hit`
   (rate that the last-history item and the target share the L1 prefix), `l1_neg_hit` (expected
   probability that a legal uniform negative shares it), `l1_hard_er = pos/neg` (hard expected-rate
   ratio), `l2_distinct` (count of distinct L2 prefixes), `l3_collision = 1 − unique triples/N`
   (duplicate fraction; not a pairwise rate), `l3_refine_bits = H3 − H2` (bits). Exact formulas,
   denominators, masks and worked examples: `FEATURE_CONTRACT.md`.
2. **What do the coefficients do?** Standardised: `l1_pos_hit +0.0081`, `l1_neg_hit −0.0059`,
   `l1_hard_er +0.0054`, `l2_distinct −0.0033`, `l3_collision −0.0033`, `l3_refine_bits ≈ 0`.
   The ER term is the most stable (fold std 0.0004) and dominates individual predictions (e.g. +0.0086
   of the 0.0107 total deviation for `TIGER_r48k2_rq`). The pos/neg pair is collinear (r 0.94,
   VIF 13–18) and ER is their deterministic ratio — **not three independent effects, and not causal**.
3. **Does 13/13 survive the audit?** Partially. On the strictly bound subset (120 SIDs, 6 of the 13
   high-recall) the frozen Ridge6 keeps **6/6 tp, 0 fp**; the remaining 7 positives rest on collision
   equality or git-era names (weak identity) and are 7/7 tp, 0 fp but are not a strict result.
   The full 13/13 with 0 fp is reproduced from the stored predictions but must be quoted together with
   this identity split. The preprocessing-leak fix (scaler inside inner folds) slightly **improves**
   the OOF numbers (MAE 0.003031→0.002972, ρ 0.623→0.667, 13/13 unchanged).
4. **Can it predict on new SIDs, and with what limits?** Yes, as a research filter: `PREDICTOR.json`
   + `predict277.py` give a recomputable formula with scaler, coefficients, domain and an empirical
   OOF interval (MAE 95% CI [0.002376, 0.003620]; FP = 0 in all 2000 bootstrap resamples). Limits:
   strict high-recall coverage 6 SIDs; label range 0.0260–0.0648 (0.065 NOT_EVALUABLE); collinearity /
   functional dependence of the fidelity terms; residuals up to ~0.018 on low-recall/high-fidelity arms.

## C1–C10

| C | evidence | verdict |
|---|---|---|
| C1 | all OOF metrics recomputed and equal to the archive (1e-12); Ridge6 confusion 13/0/0/116 and 12/0/0/61 recomputed from its own predictions | PASS |
| C2 | 120 strong / 9 weak; 7 weak positives; strict-subset 6/6 tp 0 fp; weak sensitivity reported separately | PASS |
| C3 | folds 27/27/26/25/24, 0 duplicates, 0 mapping crossover; TIGER ref fixed to canonical SID 025ad8bd = 0.051975 (0.0469 = old report, 0.055206 = unbound Nov-23 run) | PASS |
| C4 | leaky vs corrected OOF tables; no seed/fold change; corrected is slightly better | PASS |
| C5 | exact definitions + worked examples; hand values asserted (4 tests) | PASS |
| C6 | per-fold scaler/alpha/coefs, NumPy check 0.0; correlations/VIF; 10 ablations; contributions; bootstrap; 13/13; 0.065 NOT_EVALUABLE | PASS |
| C7 | PREDICTOR.json frozen (alpha 0.01, hashes, domain, interval); predictor tests 5/5 | PASS |
| C8 | 124 prospective predictions committed `03c8e1df` at 14:03:16 +1100 before any label; no Stage3 started | PASS |
| C9 | AUDIT, FEATURE_CONTRACT, COEFFICIENTS, PREDICTOR, PROSPECTIVE, tests, REPORT/VERDICT, infra drift | PASS |
| C10 | four answers above | PASS |

## Hypotheses

* **H9 SUPPORTED**: the 13/13 threshold result is reproduced from the Ridge6 predictions; it survives
  on the strict core 6/6 but strict coverage is 6 of 13 (weak-identity positives documented).
* **H10 SUPPORTED**: the preprocessing leak did not inflate the result — the corrected variant is
  marginally better (0.002972/0.667 vs 0.003031/0.623).
* **H11 UNTESTED (prospective)**: the frozen predictor's prediction for `TIGER_pairrank_enc_rq`
  (0.05319, below the 0.06 threshold) will be checked when its Stage3 label lands; frozen before the
  label existed.

## Closure checklist (§6.4)

1. Production code: `audit277.py`, `explain277.py`, `predict277.py` are the executed code. **YES**
2. Tests executed: `test_feature_examples.py` 4/4, `test_predictor.py` 5/5 with printed output. **YES**
3. Training: none required (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra: FROZEN; snapshot comparison added/removed/changed = 0. **YES**
5. Executable work left in scope: none. **NO**
