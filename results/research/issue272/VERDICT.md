# Issue #272 verdict — round 1

## 1. Question and frozen criteria

Can an **active** global-distance constraint (`J` at the #269 level) plus the frozen kappa_t local
objective deliver J <= 0.338336, local MAE <= 1.893506 / Spearman > 0.015851 / sign > 0.432453,
L2 >= 10,918 with effective rank >= 8, and reconstruction <= 1.05 x the same-criterion #265 value,
without running Stage3?

## 2. Reference inventory (one ruler, `ruler_inventory.json`)

| arm | full J | 1-hop | 2-hop | L2 | rank | local MAE | local Spearman | sign | recon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TIGER | 0.739635 | 15.1579 | 2.6745 | 7,374 | 11.479 | 7.3092 | 0.015851 | 0.432453 | n/a |
| #265 | 0.189476 | 1.1267 | 0.1843 | 10,918 | 3.049 | 2.2460 | -0.071041 | 0.332350 | 8.18e-5 |
| #269 | 0.338336 | 0.7603 | 0.2649 | 6,246 | 4.335 | 1.9973 | -0.070083 | 0.363704 | 9.54e-5 |
| #271 | 0.351361 | 0.6420 | 0.2605 | 6,096 | 4.501 | 1.8935 | -0.005483 | 0.392646 | 9.04e-5 |
| #272 round 1 | **0.256476** | **1.2448** | 0.2575 | **6,848** | **2.774** | **2.3438** | **-0.098838** | **0.348062** | **9.13e-5** |

## 3. Frozen PASS table (round 1, `TIGER_active_j_rq`, run `stage2_4dffd8b4528440b3aca334b380749f01`)

| criterion | threshold | observed | verdict |
|---|---:|---:|---|
| full J | <= 0.338336 | 0.256476 | PASS |
| 1-hop J (tolerance) | <= 0.798295 | **1.244779** | **FAIL** |
| 2-hop J (tolerance) | <= 0.278106 | 0.257521 | PASS |
| local MAE | <= 1.893506 | **2.343824** | **FAIL** |
| local Spearman | > 0.015851 | **-0.098838** | **FAIL** |
| local sign | > 0.432453 | **0.348062** | **FAIL** |
| anti-degeneracy var ratio | >= 0.5 | 0.6227 | PASS |
| anti-degeneracy slope | >= 0.1 | **-0.0361** | **FAIL** |
| L2 prefixes | >= 10,918 | **6,848** | **FAIL** |
| effective rank | >= 8 | **2.774** | **FAIL** |
| codebook occupancy | 256 each | 256/256/256 | PASS |
| L3 gain | recorded | 2.662 bits | recorded |
| collision | recorded only | 0.119779 | recorded |
| L1 prefix ER | not below TIGER | 5.872 (TIGER layer record unavailable; #269 5.722) | **FAIL vs #269** |
| conditional enrichment | not below TIGER | 1.294 bits (#269 1.637) | **FAIL vs #269** |
| reconstruction | <= 8.59e-5 (same criterion #265) | **9.13e-5** | **FAIL** |
| Stage3 | forbidden | not run | PASS (no Stage3) |

**Overall: FAIL.** Only the mean J, the 2-hop J, the anti-degeneracy variance and the codebook occupancy
pass. No criterion was relaxed.

## 4. Active-constraint evidence (C3)

- Training bound `J_MAX = 0.320`; acceptance 0.338336 (reference minus the frozen estimator offset).
- `mu > 0` for **80 of 3000 epochs (2.7%)**, first at epoch 9, max **0.021894**; the constraint acted
  briefly, then the training trajectory stayed below the bound. **The J improvement cannot be attributed
  to a sustained global gradient**; the controlled real-batch probe (`MODULE_CHECKS.json`) proves the
  mechanism can act: at the #271 checkpoint `J_est=0.351831 > 0.32`, `mu` 0.1 -> 0.10318, global gradient
  reaches encoder 0.1154 / codebooks 0.0934, decoder 0, hard-q gradient true, local/global cosine -0.086.
  A satisfied synthetic estimate (0.30) lowers mu (0.098). No existing q on the pair table satisfies 0.32
  (the closest scaled #269 q is 0.3242), so the satisfied side is a controlled estimate, stated here.

## 5. Failure diagnosis (C4)

1. **The active constraint worsens near-neighbour fidelity and the local proxy.** From #271 to #272 the
   mean J fell 0.3514 -> 0.2565, but 1-hop J rose 0.6420 -> 1.2448, local MAE 1.8935 -> 2.3438, Spearman
   -0.0055 -> -0.0988, sign 0.3926 -> 0.3481. The mean J is not a sufficient criterion; the pre-frozen
   per-hop guard catches exactly this.
2. **L2/rank frontier.** Across five arms, L2 >= 10,918 and rank >= 8 never co-occur: TIGER (lambda=0) is
   7,374/11.48; #269 6,246/4.34; #271 6,096/4.50; the round-1 tight bound 6,848/2.77; the only L2 >= 10,918
   arm (#265, 10,918) uses the **excluded Euclidean local objective** and has rank 3.05. Moving lambda from
   0 into the allowed kappa_t family moves both L2 and rank in the wrong direction; the bound moves L2
   slightly up and rank down. Within the issue's fixed mechanism this conjunction is **empirically
   infeasible**.
3. **Local MAE/Spearman/sign frontier.** The MAE <= 1.8935 and Spearman > 0.015851 combination is absent
   in all five arms; TIGER is the only arm above the Spearman/sign thresholds and has MAE 7.31. The local
   gain in #271 was partly under-dispersion (var ratio 0.335, slope 0.041); round 1 raised the variance
   (0.623) but the slope stayed negative (-0.036), so the anti-degeneracy check also fails.
4. **Reconstruction.** 9.13e-5 is above the same-criterion 1.05 x #265 (8.59e-5). The issue's quoted
   1.23e-4 is a different metric; the contract froze the same-criterion value and the candidate fails it.

## 6. Infeasibility statement and minimal alternatives (issue §7 / skill §7)

The frozen conjunction cannot be satisfied inside the issue's mechanism boundary:

- L2 >= 10,918 requires an objective/representation outside the fixed kappa_t family (the only observed
  arm with that L2 is #265's Euclidean supervision, rank 3.05, also excluded by the local criteria);
- rank >= 8 requires the lambda=0 endpoint (TIGER), whose L2 is 7,374 and whose local MAE is 7.31;
- the active J bound lowers the mean J but worsens 1-hop J and the local proxy, and the local
  MAE/Spearman/sign conjunction is absent from every observed arm.

Minimal alternatives for the owner (none applied here):

A. Allow a different local objective or an explicit rank/capacity term (new mechanism; requires a new
   issue and the §121 scope gate).
B. Keep the mechanism and re-register thresholds on the observed frontier (e.g., L2 >= 6,800 with
   rank >= 2.5; local MAE/Spearman/sign at the #271 level; 1-hop J <= 1.25), explicitly not a #272 PASS.
C. Accept a Stage2 candidate with mean J + collision only, documented as not meeting the frozen criteria.

No threshold was relaxed, no Stage3 was started, and no recall was produced.

## 7. Close checks

1. Production code implemented: yes (round-1 candidate, one primary change: bound 0.7819 -> 0.320).
2. Necessary tests executed: yes (16 candidate tests; module probe; inventory; SID/local/geometry).
3. Required training completed: yes (3000 epochs, seed 42, single card, batch 4096; run above).
4. Infra boundary: `INFRA_SNAPSHOT_START.json` vs `INFRA_SNAPSHOT_END.json` (to be written); no public
   tool/asset changed. The leftover #271 Stage3 was stopped by explicit user instruction.
5. Executable tasks remain: **not within the frozen mechanism**; the contract is infeasible as specified.
   The issue is **not** closed as COMPLETED; per the skill it remains OPEN for the owner's decision on
   the minimal alternative.
