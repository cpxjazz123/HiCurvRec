# Issue #271 verdict

## 1. Question

Does local per-triple sectional-curvature supervision improve the discrete SID and recommendation when
the global behavioural-distance field is held at the frozen TIGER level (`J <= J_max`)?

## 2. Frozen thresholds (calibrated before training)

| quantity | value | evidence |
|---|---:|---|
| `J_TIGER` (full graph, common ruler) | 0.739634552674212 | `geometry/threshold.json` |
| `eps` (2x bootstrap SE of the fixed-pair estimator) | 0.04230598283915425 | same |
| `J_max` | 0.7819405355133663 | same |
| independent local A.1 of TIGER | MAE 7.309217, Spearman 0.015851, sign 0.432453, abs-better 0.198139 | `assets/geometry/TIGER_RQ-VAE.json` |

## 3. Results

### 3.1 Global constraint (P1)

Formal run `stage2_cc9a31a9325b439f98c8b1779aaa32aa` (single card, 3000 epochs, seed 42, batch 4096;
disclosed as not fully comparable to four-card history): exact full-graph J = **0.351359**
(299,402,685 pairs) <= `J_max` 0.781941 -> **PASS**. `mu` stayed **0** for all 3000 epochs: the
constraint was never violated, so the safety bound is non-binding (the final step estimate was 0.3354).
The per-stratum means: 1-hop 0.642, 2-hop 0.260, 3-hop 0.323, 4-5 0.412, >=6 0.580.

**Ruler dependence (important).** The issue cites J 0.183 -> 0.326 from #270, which used the
three-expert ruler. Under the present Euclidean common ruler (the kappa->0 member of the frozen
#269 map, factor 2 verified), TIGER is 0.739635 and #269 is **0.338336** (1-hop 0.760, 2-hop 0.265,
3-hop 0.311, 4-5 0.394, >=6 0.561). Thus the issue's premise "the local gain came with global
degradation" does not hold under this ruler; the candidate's constraint is a safety bound, and the
experiment is a local-supervision test with that bound, not a tight intervention.

### 3.2 Local fidelity (P2)

On the independent frozen #270 common ruler (SCALE 0.084334, full 28.4M triplet table, not the
training kappa ruler): candidate **MAE 1.893506** vs TIGER 7.309217 -> **PASS on the pre-registered
error criterion**. Ranking/sign: candidate Spearman **-0.005483** vs TIGER 0.015851, sign accuracy
**0.392646** vs 0.432453, abs-better 0.354228 vs 0.198139. The error is the lowest of the three
arms (#265 2.246, #269 1.997), the sign is better than #265/#269 but still below TIGER, and the
Spearman is slightly negative. Mixed result: the mean error improved; the rank/sign proxy did not.

### 3.3 SID structure (P3)

`collision 0.152113 < 0.40`; 20,847 unique three-code tuples; **L2 groups 6,097** vs #269 6,246
(no #269-style compression; 2,703 of #269's L2 groups split, no groups merged in this direction);
L2/L1 23.82; L3/L2 3.42; L3 refinement gain 2.883 bits; all three codebooks use 256/256; quant error
0.00939 (below the 0.035 diagnostic band, warning only); effective rank 4.50. This is **not** the
#269 over-merging pattern; P3's structural condition holds.

### 3.4 Stage3 and recommendation (P4)

Pending the 150-epoch Stage3 run (single card, seed 42, beam 20) and the single final test.

## 4. Interpretation

Pending Stage3. The Stage2 evidence already separates the local/global channels: the constraint is
satisfied with a wide margin while `mu=0`, so this run does not test a binding dual; it tests the
#269 local objective under a preserved global distance field. The local error improves, the SID
does not compress, and the rank/sign proxy is mixed. The Stage3 result determines whether the
downstream channel reacts; no attribution to local curvature alone is possible without a matched
global-constraint-only intervention.

## 5. Close checks

Pending final Stage3, R2, infra snapshot and commit.
