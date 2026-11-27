# Issue #274 verdict — round 1: curvature-selected relative pair ranking

## Scope

Parent = the frozen TIGER baseline; the single change is the two-pair distance ranking
`L_geo = mean max(0, ||q_B - q_C|| - ||q_D - q_E|| + m)` on the final hard `q = sum_l c_l`, with the
positive/negative pairs selected **offline** from the real graph's sectional curvature and frozen in
`results/research/issue274/pairs_index.npz` (sha256
`a3395e5446c4681279a2351881a8c38e80155459c4419cae846a2a76ba9f7469`). The curvature never enters the
loss, the margin, a sample's weight, the distance function, the code selection or the model inputs.

## The mechanism as implemented

* the positive pair: the closed-triangle triplet with the highest raw graph curvature inside the
  centre's comparable group;
* the negative pair: the open-wedge triplet with the lowest raw graph curvature inside the same group;
* the margin `m = 0.2529585361480713`, frozen in advance from the frozen TIGER's training-pair
  distance scale;
* the weight `lambda` frozen by the per-module gradient budget (0.02 of the reconstruction gradient)
  on the real batches, with a 200-step ramp;
* the gradient of `L_geo` reaches the three codebooks through the embedding lookup; the encoder is
  not in the loss's graph (the RQ selection is an argmin), so the loss shapes the discrete code
  geometry and cannot compete with the reconstruction on the latent.

## The calibration frozen on the real batches

| quantity | value |
|---|---|
| `lambda` (the pair weight) | 1.2112073947212332e-06 |
| the budget | 0.02 |
| the binding ratio | 6.0560369736061655e-05 |
| the median ratio | 6.41524168246569e-05 |
| the reconstruction/pair gradient cosine | 0.03512789623965412 |
| the measured batches | 18 |
| the margin | 0.2529585361480713 |
| the groups per step | 256 |
| the ramp steps | 200 |

## The frozen acceptance list

`results/research/issue274/baselines.json` (C1-C10, all thresholds pre-registered before the run).

## The measured criteria (round 1, run `stage2_debbfe6a8e714c12aea4dbc7d2c236e7`)

The full table is `results/research/issue274/round1_pass_table.json`.

| criterion | value | threshold | verdict |
|---|---|---|---|
| C1 the hold-out ranking accuracy | **0.627937** (the centre macro 0.618102; the mean distance gap 0.016530) | > 0.538605 (macro 0.529198; gap 0.012160) | **PASS** |
| C2 the fixed-margin satisfaction | **0.004849** (macro 0.006373) | > 0.081499 (macro 0.079290) | **FAIL** |
| C3 the L2 distinct prefixes | **7,929** | >= 10,918 | **FAIL** |
| C4 the effective rank | **15.606206** | >= 8 | **PASS** |
| C5 the prefix discrimination | the LCP(pos) > LCP(neg) **0.127378** (threshold 0.136889, FAIL); the mean LCP difference **0.123461** (threshold 0.116747, PASS); the positive pair's L1 share **0.131294** (threshold 0.141179, FAIL) | — | **FAIL** (2 of 3) |
| C6 the behavioural neighbourhood fidelity | **0.081965** | >= 0.079318 | **PASS** |
| C7 the global common-scale J | **0.226056** | <= 0.338336 | **PASS** |
| C8 the 1-hop / 2-hop J | **2.805521 / 0.442659** | <= 0.838210 / <= 0.292011 | **FAIL** (both) |
| C9 the reconstruction | **6.70e-05** | <= 8.589273e-05 | **PASS** |
| C10 the SID identity | the SID and the checkpoint hashed and recorded | — | **PASS** |

`all_pass = false`. Six criteria pass, four fail; the issue's close condition (all of them) is **not**
met.

### Diagnostics (not criteria)

| diagnostic | value | reference |
|---|---|---|
| the pair training accuracy | 0.592 | the start 0.43 |
| the prediction variance ratio | **2.242** | the #269 0.42 (collapse), the TIGER 28.7 |
| the prediction slope | 0.080 | the #269 0.019, the TIGER 0.041, the #272 r3 0.111 |
| the curvature MAE / Spearman / sign / within-centre ordering | 2.568 / 0.003 / 0.370 / 0.104 | no curvature supervision in this mechanism |
| the collision | 0.155 | recorded only |
| the mean L2 per L1 | 30.97 | the TIGER 29.61, the #265 42.65 |
| the mean L3 per L2 | 2.62 | the TIGER 3.33, the #265 2.25 |

## What round 1 establishes

1. **The relative ordering works.** The hold-out accuracy rises from the TIGER's 0.539 to 0.628
   (+0.089, the centre macro +0.089, the mean gap 0.0122 -> 0.0165), on pairs the mechanism never
   trained on. This is the first #269-#274 arm to improve the C1 at the frozen caliber.
2. **The dispersion collapse is avoided.** The prediction variance ratio is 2.24 against the #269's
   0.42 and the slope 0.080 against the #269's 0.019 — the code-geometry-only supervision does not
   reproduce the representation degradation that motivated the issue, and the reconstruction stays
   at 6.70e-05 (well inside the 8.589e-05 bound).
3. **The scale shrinks and the gap tail flattens.** The global q distance scale halves (the mean
   positive distance 0.286 -> 0.139) and the positive/negative gap standard deviation halves
   (0.173 -> 0.090). At the pre-frozen absolute margin (0.253) the candidate's satisfaction rate
   collapses to 0.005 against the TIGER's 0.082; a margin sweep shows the crossover at
   m ~ 0.03-0.04 (at m = 0.02 the candidate 0.510 > the TIGER 0.485), so the C2 failure is the
   **tail shape and the scale**, not the ordering.
4. **The SID capacity is not lifted.** The L2 distinct prefixes reach 7,929 (the TIGER 7,374, the
   #265 10,918); the conditional diversity (the mean L2 per L1) is 30.97 against the #265's 42.65,
   while the codebook effective ranks are *higher* than the #265's ([7.46, 21.63, 25.74] against
   [2.93, 5.00, 4.98]). The capacity gap is the level-1 -> level-2 co-variation, not the codeword
   rank. The SID's effective rank itself is 15.61 (the C4 passes).
5. **The 1-hop/2-hop relations degrade against the #269 thresholds.** The full common-scale J passes
   (0.226 <= 0.338) but the 1-hop (2.81) and the 2-hop (0.443) strata are far above the frozen
   thresholds (0.838 / 0.292), which are the #269 values plus 5 %; the #269 is the only arm in
   #269-#273 that met them, and it did so with the pointwise curvature supervision this issue
   forbids.
6. **The curvature fidelity diagnostics remain weak** (the MAE 2.57, the Spearman 0.003) — by
   construction: this mechanism never regresses a curvature value, and the issue's acceptance treats
   these as diagnostics only.


## Round 2 (the encoder channel)

Run `stage2_818150011991495cbeebd225909b1f91`; the full table is
`results/research/issue274/round2_pass_table.json`. The single primary change: the pair term's
gradient also reaches the encoder through the reference implementation's identity path
(`q = sum_l c_l + (z - z.detach())`), and the per-module gradient budget (0.02, lambda
`6.509635430931112e-06`, the reconstruction/pair cosine 0.0284) covers the encoder and the three
codebooks.

| criterion | round 1 | round 2 | threshold | verdict |
|---|---|---|---|---|
| C1 the hold-out ranking accuracy | 0.627937 | **0.806416** (the macro 0.764684, the gap 0.250012) | > 0.538605 | **PASS** |
| C2 the fixed-margin satisfaction | 0.004849 | **0.468482** (the macro 0.387052) | > 0.081499 | **PASS** |
| C3 the L2 distinct prefixes | 7,929 | 7,923 | >= 10,918 | **FAIL** |
| C4 the effective rank | 15.606 | 12.350 | >= 8 | **PASS** |
| C5 the prefix discrimination | 2 of 3 fail | 2 of 3 fail (the LCP 0.118053, the LCP difference 0.116934, the L1 share 0.121037) | — | **FAIL** |
| C6 the behavioural neighbourhood fidelity | 0.081965 | 0.079249 | >= 0.079318 | **FAIL** (by 0.00007) |
| C7 the full J | 0.226056 | 0.303676 | <= 0.338336 | **PASS** |
| C8 the 1-hop / 2-hop J | 2.805521 / 0.442659 | 9.656970 / 1.097662 | <= 0.838210 / <= 0.292011 | **FAIL** |
| C9 the reconstruction | 6.70e-05 | 6.70e-05 | <= 8.589273e-05 | **PASS** |
| C10 the SID identity | pass | pass | — | **PASS** |

`all_pass = false` (six pass, four fail); the issue's close condition is still not met.

### The trade-off the two rounds expose

The encoder channel **fixes the C2's root cause** (the scale no longer halves: the mean positive
distance 0.2146 against the negative 0.4646, a 0.25 gap against the round-1's 0.0165) and lifts the
C1 to 0.806, the best of the whole #269-#274 family by a wide margin. The same channel **degrades
the graph-distance relations**: the 1-hop J goes from 2.81 (round 1) to 9.66, and the 2-hop from
0.443 to 1.098, against the TIGER's 15.16 / 2.674. The codebooks-only variant (round 1) kept the
relations at 2.81 / 0.443 — still far from the #269's 0.760 / 0.264 — but could not hold the scale.

The two criteria that remain out of reach both carry thresholds from the **other supervision
families**: the C3's 10,918 is the #265's value, attained with the Euclidean local-curvature
supervision; the C8's 0.838 / 0.292 are the #269's values plus 5 %, and the #269 is the only arm in
#269-#273 that met them, with the pointwise curvature supervision this issue's fixed mechanism
forbids. The three probes and the two full runs bound this mechanism's frontier: the L2 stays at
about 7,920-7,930 in every configuration (the probe's transient 9,058 at a 100-step fine-tune does
not survive a full training), and the 1-hop J is traded against the ranking accuracy.

## Round 3 (the sampling / matching-coverage channel)

Run `stage2_6fe2faf7fc204330b129dfa4900ae641`, the dense 1-closed/1-open index (38,871 groups
against 26,583). Full table: `results/research/issue274/round3_pass_table.json`.

| criterion | round 1 | round 2 | round 3 | threshold | verdict |
|---|---|---|---|---|---|
| C1 the hold-out ranking accuracy | 0.6279 | 0.8064 | **0.8034** | > 0.5386 | PASS |
| C2 the fixed-margin satisfaction | 0.0048 | 0.4685 | **0.4642** | > 0.0815 | PASS |
| C3 the L2 distinct prefixes | 7,929 | 7,923 | 7,911 | >= 10,918 | FAIL |
| C4 the effective rank | 15.61 | 12.35 | 11.20 | >= 8 | PASS |
| C5 the prefix discrimination | 2/3 fail | 2/3 fail | 2/3 fail (0.1175 / 0.1169 / 0.1195) | — | FAIL |
| C6 the behavioural neighbourhood fidelity | 0.08197 | 0.07925 | **0.08097** | >= 0.07932 | PASS |
| C7 the full J | 0.2261 | 0.3037 | 0.3152 | <= 0.3383 | PASS |
| C8 the 1-hop / 2-hop J | 2.806 / 0.443 | 9.657 / 1.098 | 9.567 / 1.158 | <= 0.838 / <= 0.292 | FAIL |
| C9 the reconstruction | 6.70e-05 | 6.70e-05 | 6.71e-05 | <= 8.589e-05 | PASS |
| C10 the SID identity | pass | pass | pass | — | PASS |

`all_pass = false`: seven pass, three fail (C3, C5, C8).

### The cross-round structure

Three structurally different levers now agree:

* **C3 is terminal at ~7,910-7,930** — the codebook channel, the encoder channel and the
  matching-coverage channel all end at the same L2 count and the same collision band
  (0.071 at epoch 50 -> 0.136 at 750 -> **0.1543** final, against round 2's 0.1573). The
  collision rises monotonically with training in every configuration; the capacity ceiling is not a
  sampling artifact. The threshold 10,918 belongs to the #265 Euclidean local-curvature supervision.
* **C1/C2/C4/C7/C9 pass in every encoder-channel round** (0.803-0.806 accuracy, 0.464-0.468
  margin, J 0.304-0.315 on a 0.338 budget): the curvature-selected ranking itself is a solved,
  reproducible signal.
* **C8 fails in the encoder-channel rounds** (9.57/1.16 vs the codebook-only round's 2.81/0.44 and
  the TIGER's 15.16/2.67) and its threshold belongs to the #269 pointwise-curvature supervision:
  the pair-ranking channel trades the graph-distance strata for the ranking accuracy.
* **C5 fails 2/3 in every round** (the mechanism's SID prefix structure sits slightly below the
  frozen TIGER values); **C6 recovered** in round 3 to 0.08097 above the 0.079318 floor.
