**Result: analysis complete — descriptive positives, predictive gate negative; #277 retained.**

Full traversal / joint-assignment analysis over the 28,391,948-row triplet asset (all valid; every
row's 6-bit directed code recomputed from `visible_forward`, `canonical_code` = stored class, 0
mismatches), 36 role classes with Orbit roles, 253 SIDs × 3 cumulative prefix levels.

- **The joint 5-partition has exactly one free parameter beyond the pairwise marginals:**
  ALL = P(a=m=b); AM = p_am−ALL, MB = p_mb−ALL, AB = p_ab−ALL (CONTRACT §3, hand example
  D1={ALL,ALL,NONE,NONE} vs D2={ALL,AM,MB,AB}: same marginals 0.5, ALL 0.5 vs 0.25; 7 deterministic
  tests pass, including refinement-only transitions and 22 feasible LCP triples).
- **Descriptive:** at k=1 ALL is 1.4–2.0× the uniform permutation null for chain/wedge (0.063 vs
  0.045) and ≈1.9× for triangles (0.086 vs 0.045); at k=2/3 the observed ALL is *below* the null
  (chain −0.0011/−0.0063). LCP and transitions are exact (only the 12 allowed transition cells occur).
  Null models verified: 320 empirical-permutation tests, 316 within 3σ, median |z| 0.74.
- **Matched marginals (509 chain SID pairs with |Δ|≤0.005 on all three rates):** ALL vs recall
  Spearman **+0.191 [0.045, 0.325]** (weak positive; sign agreement 0.56). Matched-SID contrasts
  (146 pairs): all3_chain_lift +0.357 [0.043, 0.598]; the role contrasts (AM−MB, AMB−AB) and the
  triangle/reciprocal contrasts have CIs covering 0. Residual associations vs #277: all_ab_triangle
  −0.306 (BH q=0.004), all3_chain_lift +0.242 (q=0.026) — descriptive only.
- **Predictive gate (pre-registered):** baseline #277 corrected Ridge6 reproduced exactly
  (0.0029719 / ρ 0.6666); inner-fold-only selection of one pre-frozen joint feature (all1_chain
  3 folds, am_mb_chain 2 folds) gives OOF 0.0028218 / 0.7156, but the paired SID-cluster bootstrap
  MAE(A)−MAE(new) 95 % CI is **[−1.57e−4, +4.57e−4]** — contains 0 → **criterion fails**. The fixed
  `all1_chain` arm (0.0026990 / 0.7665) also fails ([−1.06e−4, +6.35e−4]); the lift variants are
  significantly worse. Secondary metrics improve (ρ, reversals, fp/fn 13/0/0, strict 120), but the
  frozen rule requires the CI gate: **#277 kept, no new formula/coefficients, no new predictions,
  no Stage3.** 0 rows >0.065 → 0.065 NOT_EVALUABLE.
- **Population-2 diagnostic:** the same-level hit is mechanically implied by the partition
  (ALL/AB ⇒ hit), so the hit-difference is a tautology check; the nontrivial composition: among
  unseen hit pairs with intermediates the intermediate shares the anchor/target L1 code in ≈36 % and
  differs (AB) in ≈64 %. The validation ALL1 rate is ρ=0.980/0.981 with `l1_pos_hit` — redundant with
  the existing feature, no independent signal. #274 R2: `NOT_AVAILABLE` (SID not in the #276 table).
- **Knowledge:** the joint ALL at k=1 is the best joint candidate (point estimate MAE −9.1 %, ρ +0.10)
  but not statistically resolvable at 129 SIDs; redundancy and support are *not* the limiting factors.
  The direction remains open (X_tested ≠ X_allowed); no infeasibility claim is made.

Evidence: `results/research/issue283/{CONTRACT,DIAGNOSIS,REPORT,VERDICT}.md`,
`JOIN_SUMMARY.json`, `JOIN_TABLE.part1/2.json.gz` (sha256 in `STATS_MANIFEST.json`),
`MATCHED.json`, `MODEL_COMPARISON.json`, `PERM_CHECK.json`, `VALIDATION_*.json.gz`,
`ROWS_MANIFEST.json`, `tests/test_joint283.py`. Infra drift: 0 (start vs close snapshot).

Commit: 801048ca8153fd765e38b045a4c0578cb6343a0c (main).
