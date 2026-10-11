# Semantic ID prefix vs behaviour relation — corrected analysis

Offline only. No Stage 2 or Stage 3 retraining, no checkpoint and no Semantic ID was modified.

- Models: A = TIGER Euclidean baseline (R@10 0.053674), B = Exp1a behaviour contrastive (R@10 0.063633)
- SID levels analysed: L1, L1+L2, L1+L2+L3. The fourth stored token is the Collision Extension Token and is excluded.
- Seed 20261011, 2000 bootstrap resamples, minimum cell 30.


## Reproduction

```bash

cd /home/wlia0047/ar57/wenyu/GeneRec && \
OMP_NUM_THREADS=8 /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/analyze_prefix_behavior.py

```


## Prefix structure

| model | level | prefixes | mean group | singleton prefix-group % | singleton item % | max group |
|---|---|---|---|---|---|---|
| A_tiger | 1 | 247 | 99.54 | 3.2389% | 0.0325% | 859 |
| A_tiger | 2 | 7414 | 3.32 | 52.0502% | 15.6953% | 111 |
| A_tiger | 3 | 20223 | 1.22 | 88.2016% | 72.5465% | 42 |
| B_exp1a | 1 | 241 | 102.02 | 0.4149% | 0.0041% | 253 |
| B_exp1a | 2 | 16145 | 1.52 | 70.8455% | 46.5205% | 25 |
| B_exp1a | 3 | 23695 | 1.04 | 97.0373% | 93.5169% | 10 |

- A_tiger: distinct 3-token prefixes 20223, collision 17.7492%, largest group 42
- B_exp1a: distinct 3-token prefixes 23695, collision 3.6279%, largest group 10

## Relation: train_transition

Behaviour positives: 24474 pairs. Negatives: 24474 (distinct 12995), bin fallbacks 0, invariant violations none.

Negative semantics: an item with NO OBSERVED relation of this kind to the anchor. Absence of an observed edge is not evidence of irrelevance, so every negative means 'unobserved', never 'unrelated'.


| level | A P(share\|pos) | B P(share\|pos) | A P(share\|neg) | B P(share\|neg) | A ER | B ER | A sampled precision | B sampled precision |
|---|---|---|---|---|---|---|---|---|
| L1 | 0.0958 | 0.1424 | 0.01316 | 0.00515 | 7.31 | 27.58 | 0.8792 | 0.9651 |
| L2 | 0.0131 | 0.0130 | 0.00041 | 0.00004 | 32.23 | 326.65 | 0.9698 | 0.9969 |
| L3 | 0.0032 | 0.0020 | 0.00000 | 0.00000 | n/a | n/a | 1.0000 | 1.0000 |


Between-model paired tests (same items, same positives, same negatives, same split; CI by cluster bootstrap over anchor items, p by paired Wilcoxon):

| level | delta_pos B-A [CI] | p | delta_neg B-A [CI] | p | DiD [CI] | p |
|---|---|---|---|---|---|---|
| L1 | 0.04666 [0.04196, 0.05136] | 1.24e-83 | -0.00801 [-0.00956, -0.00633] | 1.15e-23 | 0.05467 [0.04977, 0.05970] | 2.45e-103 |
| L2 | -0.00016 [-0.00196, 0.00155] | 0.857 | -0.00037 [-0.00065, -0.00012] | 0.0117 | 0.00020 [-0.00164, 0.00200] | 0.824 |
| L3 | -0.00127 [-0.00208, -0.00045] | 0.00204 | 0.00000 [0.00000, 0.00000] | 1 | -0.00127 [-0.00204, -0.00049] | 0.00204 |

Conditional retention (own condition, and the paired comparison on the pairs where BOTH models share the previous level):

| condition | side | n_common | A retain | B retain | B-A |
|---|---|---|---|---|---|
| L2_given_L1 | pos | 1178 | 0.1910 | 0.1562 | -0.03480 |
| L2_given_L1 | neg | 33 | 0.0303 | 0.0000 | -0.03030 |
| L3_given_L2 | pos | 72 | 0.4583 | 0.3194 | -0.13889 |
| L3_given_L2 | neg | 0 | n/a | n/a | n/a |

Matched group-size strata (positive and negative in the SAME size bucket):

| level | bucket | n | A pos | A neg | B pos | B neg |
|---|---|---|---|---|---|---|
| L1 | 25-99 | 648 | 0.0602 | 0.00772 | 0.0758 | 0.00567 |
| L1 | 100+ | 15405 | 0.1087 | 0.01701 | 0.2013 | 0.00756 |
| L2 | 1 | 621 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L2 | 2-4 | 1673 | 0.0078 | 0.00000 | 0.0109 | 0.00025 |
| L2 | 5-9 | 1117 | 0.0081 | 0.00000 | 0.0643 | 0.00000 |
| L2 | 10-24 | 1551 | 0.0187 | 0.00000 | 0.1000 | 0.00000 |
| L3 | 1 | 13339 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L3 | 2-4 | 937 | 0.0064 | 0.00000 | 0.0522 | 0.00000 |


## Relation: test_transition

Behaviour positives: 16828 pairs. Negatives: 16828 (distinct 10259), bin fallbacks 1, invariant violations none.

Negative semantics: an item with NO OBSERVED relation of this kind to the anchor. Absence of an observed edge is not evidence of irrelevance, so every negative means 'unobserved', never 'unrelated'.


| level | A P(share\|pos) | B P(share\|pos) | A P(share\|neg) | B P(share\|neg) | A ER | B ER | A sampled precision | B sampled precision |
|---|---|---|---|---|---|---|---|---|
| L1 | 0.0972 | 0.1004 | 0.01379 | 0.00636 | 7.03 | 15.80 | 0.8757 | 0.9405 |
| L2 | 0.0121 | 0.0063 | 0.00071 | 0.00000 | 17.13 | n/a | 0.9442 | 1.0000 |
| L3 | 0.0031 | 0.0008 | 0.00000 | 0.00000 | n/a | n/a | 1.0000 | 1.0000 |


Between-model paired tests (same items, same positives, same negatives, same split; CI by cluster bootstrap over anchor items, p by paired Wilcoxon):

| level | delta_pos B-A [CI] | p | delta_neg B-A [CI] | p | DiD [CI] | p |
|---|---|---|---|---|---|---|
| L1 | 0.00327 [-0.00208, 0.00814] | 0.212 | -0.00743 [-0.00933, -0.00553] | 4.78e-14 | 0.01070 [0.00547, 0.01593] | 0.000126 |
| L2 | -0.00576 [-0.00761, -0.00386] | 1.67e-09 | -0.00071 [-0.00113, -0.00036] | 0.000488 | -0.00505 [-0.00689, -0.00321] | 2.43e-07 |
| L3 | -0.00226 [-0.00321, -0.00142] | 9.31e-07 | 0.00000 [0.00000, 0.00000] | 1 | -0.00226 [-0.00315, -0.00137] | 9.31e-07 |

Conditional retention (own condition, and the paired comparison on the pairs where BOTH models share the previous level):

| condition | side | n_common | A retain | B retain | B-A |
|---|---|---|---|---|---|
| L2_given_L1 | pos | 693 | 0.1833 | 0.1082 | -0.07504 |
| L2_given_L1 | neg | 32 | 0.0625 | 0.0000 | -0.06250 |
| L3_given_L2 | pos | 25 | n/a | n/a | n/a |
| L3_given_L2 | neg | 0 | n/a | n/a | n/a |

Matched group-size strata (positive and negative in the SAME size bucket):

| level | bucket | n | A pos | A neg | B pos | B neg |
|---|---|---|---|---|---|---|
| L1 | 25-99 | 356 | 0.0393 | 0.00843 | 0.0468 | 0.00638 |
| L1 | 100+ | 11617 | 0.1067 | 0.01575 | 0.1364 | 0.01001 |
| L2 | 1 | 362 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L2 | 2-4 | 979 | 0.0051 | 0.00000 | 0.0060 | 0.00000 |
| L2 | 5-9 | 769 | 0.0052 | 0.00000 | 0.0111 | 0.00000 |
| L3 | 1 | 8698 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L3 | 2-4 | 735 | 0.0041 | 0.00000 | 0.0154 | 0.00000 |


## Relation: common_succ

Behaviour positives: 24130 pairs. Negatives: 24130 (distinct 12072), bin fallbacks 0, invariant violations none.

Negative semantics: an item with NO OBSERVED relation of this kind to the anchor. Absence of an observed edge is not evidence of irrelevance, so every negative means 'unobserved', never 'unrelated'.


| level | A P(share\|pos) | B P(share\|pos) | A P(share\|neg) | B P(share\|neg) | A ER | B ER | A sampled precision | B sampled precision |
|---|---|---|---|---|---|---|---|---|
| L1 | 0.0680 | 0.0730 | 0.01077 | 0.00406 | 6.32 | 17.96 | 0.8632 | 0.9473 |
| L2 | 0.0111 | 0.0061 | 0.00021 | 0.00008 | 53.28 | 71.93 | 0.9816 | 0.9866 |
| L3 | 0.0038 | 0.0014 | 0.00008 | 0.00000 | 46.24 | n/a | 0.9785 | 1.0000 |


Between-model paired tests (same items, same positives, same negatives, same split; CI by cluster bootstrap over anchor items, p by paired Wilcoxon):

| level | delta_pos B-A [CI] | p | delta_neg B-A [CI] | p | DiD [CI] | p |
|---|---|---|---|---|---|---|
| L1 | 0.00501 [0.00153, 0.00858] | 0.00499 | -0.00671 [-0.00812, -0.00535] | 8.51e-21 | 0.01173 [0.00792, 0.01546] | 1.07e-09 |
| L2 | -0.00497 [-0.00642, -0.00352] | 2.59e-12 | -0.00012 [-0.00033, 0.00008] | 0.453 | -0.00485 [-0.00626, -0.00348] | 1.54e-11 |
| L3 | -0.00236 [-0.00315, -0.00162] | 3.41e-09 | -0.00008 [-0.00021, 0.00000] | 0.5 | -0.00228 [-0.00311, -0.00149] | 1.67e-08 |

Conditional retention (own condition, and the paired comparison on the pairs where BOTH models share the previous level):

| condition | side | n_common | A retain | B retain | B-A |
|---|---|---|---|---|---|
| L2_given_L1 | pos | 772 | 0.2604 | 0.1412 | -0.11917 |
| L2_given_L1 | neg | 29 | n/a | n/a | n/a |
| L3_given_L2 | pos | 60 | 0.5167 | 0.4000 | -0.11667 |
| L3_given_L2 | neg | 0 | n/a | n/a | n/a |

Matched group-size strata (positive and negative in the SAME size bucket):

| level | bucket | n | A pos | A neg | B pos | B neg |
|---|---|---|---|---|---|---|
| L1 | 25-99 | 445 | 0.0494 | 0.00225 | 0.0416 | 0.00351 |
| L1 | 100+ | 16523 | 0.0732 | 0.01325 | 0.1045 | 0.00587 |
| L2 | 1 | 482 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L2 | 2-4 | 1376 | 0.0029 | 0.00000 | 0.0064 | 0.00000 |
| L2 | 5-9 | 1076 | 0.0065 | 0.00000 | 0.0135 | 0.00000 |
| L2 | 10-24 | 1715 | 0.0216 | 0.00000 | 0.0000 | 0.00000 |
| L3 | 1 | 12547 | 0.0000 | 0.00000 | 0.0000 | 0.00000 |
| L3 | 2-4 | 959 | 0.0094 | 0.00000 | 0.0172 | 0.00000 |


## Group-size distribution of behaviour positives

| level | model | bucket | n_pos | % of positives | n_neg | % of negatives |
|---|---|---|---|---|---|---|
| L1 | A_tiger | 1 | 6 | 0.0357% | 4 | 0.0238% |
| L1 | A_tiger | 2-4 | 18 | 0.1070% | 13 | 0.0773% |
| L1 | A_tiger | 5-9 | 73 | 0.4338% | 73 | 0.4338% |
| L1 | A_tiger | 10-24 | 460 | 2.7335% | 487 | 2.8940% |
| L1 | A_tiger | 25-99 | 2292 | 13.6202% | 2339 | 13.8995% |
| L1 | A_tiger | 100+ | 13979 | 83.0699% | 13912 | 82.6717% |
| L1 | B_exp1a | 1 | 1 | 0.0059% | 0 | 0.0000% |
| L1 | B_exp1a | 25-99 | 7247 | 43.0651% | 7280 | 43.2612% |
| L1 | B_exp1a | 100+ | 9580 | 56.9289% | 9548 | 56.7388% |
| L2 | A_tiger | 1 | 2272 | 13.5013% | 2322 | 13.7984% |
| L2 | A_tiger | 2-4 | 4002 | 23.7818% | 3981 | 23.6570% |
| L2 | A_tiger | 5-9 | 3482 | 20.6917% | 3679 | 21.8624% |
| L2 | A_tiger | 10-24 | 4539 | 26.9729% | 4451 | 26.4500% |
| L2 | A_tiger | 25-99 | 2401 | 14.2679% | 2271 | 13.4954% |
| L2 | A_tiger | 100+ | 132 | 0.7844% | 124 | 0.7369% |
| L2 | B_exp1a | 1 | 8066 | 47.9320% | 7803 | 46.3691% |
| L2 | B_exp1a | 2-4 | 6788 | 40.3375% | 7104 | 42.2154% |
| L2 | B_exp1a | 5-9 | 1343 | 7.9807% | 1309 | 7.7787% |
| L2 | B_exp1a | 10-24 | 622 | 3.6962% | 606 | 3.6011% |
| L2 | B_exp1a | 25-99 | 9 | 0.0535% | 6 | 0.0357% |
| L3 | A_tiger | 1 | 12046 | 71.5831% | 12110 | 71.9634% |
| L3 | A_tiger | 2-4 | 3350 | 19.9073% | 3467 | 20.6026% |
| L3 | A_tiger | 5-9 | 892 | 5.3007% | 842 | 5.0036% |
| L3 | A_tiger | 10-24 | 464 | 2.7573% | 356 | 2.1155% |
| L3 | A_tiger | 25-99 | 76 | 0.4516% | 53 | 0.3150% |
| L3 | B_exp1a | 1 | 15693 | 93.2553% | 15689 | 93.2315% |
| L3 | B_exp1a | 2-4 | 1061 | 6.3050% | 1063 | 6.3169% |
| L3 | B_exp1a | 5-9 | 67 | 0.3981% | 67 | 0.3981% |
| L3 | B_exp1a | 10-24 | 7 | 0.0416% | 9 | 0.0535% |


## Bug fixes applied in this version

Two defects were fixed and the earlier results kept in `prefix_behavior_analysis_corrected_v1.json` / `.md` for traceability.

**1. Singleton item fraction.** `items_in_singleton_groups_frac` divided by the number of PREFIXES, because `(sizes == 1).mean()` is indexed by prefix. It therefore reproduced `singleton_prefix_group_frac` exactly and the two columns were identical in v1. The item-side denominator is n_items.


| model | level | singleton prefix groups (v1=v2) | of prefixes | items in singleton groups (v1 WRONG) | of items (v2) |
|---|---|---|---|---|---|
| A_tiger | 1 | 8 | 3.2389% | 3.2389% | 0.0325% |
| A_tiger | 2 | 3859 | 52.0502% | 52.0502% | 15.6953% |
| A_tiger | 3 | 17837 | 88.2016% | 88.2016% | 72.5465% |
| B_exp1a | 1 | 1 | 0.4149% | 0.4149% | 0.0041% |
| B_exp1a | 2 | 11438 | 70.8455% | 70.8455% | 46.5205% |
| B_exp1a | 3 | 22993 | 97.0373% | 97.0373% | 93.5169% |

**2. Conditional retention of the negative side.** Inside the `for side, others in (("pos", positives), ("neg", negatives))` loop the common-set retention read `positives` on both branches, so the negative row reported the POSITIVE retention under the negative-conditioned mask. It now reads `others`.


| relation | condition | side | n_common | A v1 | A v2 | changed |
|---|---|---|---|---|---|---|
| train_transition | L2_given_L1 | pos | 1178 | 0.1910 | 0.1910 | no |
| train_transition | L2_given_L1 | neg | 33 | 0.0303 | 0.0303 | no |
| train_transition | L3_given_L2 | pos | 72 | 0.4583 | 0.4583 | no |
| train_transition | L3_given_L2 | neg | 0 | n/a | n/a | no |
| test_transition | L2_given_L1 | pos | 693 | 0.1833 | 0.1833 | no |
| test_transition | L2_given_L1 | neg | 32 | 0.0000 | 0.0625 | **yes** |
| test_transition | L3_given_L2 | pos | 25 | n/a | n/a | no |
| test_transition | L3_given_L2 | neg | 0 | n/a | n/a | no |
| common_succ | L2_given_L1 | pos | 772 | 0.2604 | 0.2604 | no |
| common_succ | L2_given_L1 | neg | 29 | n/a | n/a | no |
| common_succ | L3_given_L2 | pos | 60 | 0.5167 | 0.5167 | no |
| common_succ | L3_given_L2 | neg | 0 | n/a | n/a | no |

Everything else is unchanged: prefix counts, mean and maximum group sizes, the collision rate, all positive and negative sharing rates, all enrichment ratios, the sampled precision, the group-size distributions and every between-model test were verified bit-identical between v1 and v2.


## Conclusions

### Q1. Is the L1 behaviour-selectivity gain statistically significant?
Yes for the SELECTIVITY, but not for the positive-side retention. On the independent test relation the difference-in-differences is **+0.01070** [+0.00547, +0.01593], paired Wilcoxon p = 0.000126, so B is more selective at L1. That gain decomposes asymmetrically:
- negative side: -0.00743 [-0.00933, -0.00553], p = 4.78e-14 — **significant**
- positive side: +0.00327 [-0.00208, +0.00814], p = 0.212 — **NOT significant**
So on data the model never trained on, B's L1 advantage comes almost entirely from pushing unrelated items apart, not from gathering more behaviour-related pairs. On the train relation the positive-side gain is large and significant (+0.04666, p = 1.24e-83), which is expected because that is the objective; on common_successor, which never entered any objective, it is small but significant (+0.00501, p = 0.00499).

### Q2. Does B really scatter more genuinely behaviour-related items at L2 and L3, and does it survive the common condition?
**Yes, and it survives the paired conditional comparison — but only at L2; at L3 the independent-test estimate is not obtainable.**
- Unconditional on the test relation: L2 -0.00576 [-0.00761, -0.00386], p = 1.67e-09; L3 -0.00226 [-0.00321, -0.00142], p = 9.31e-07. Both are negative, i.e. B keeps fewer behaviour pairs together.
- On the pairs where BOTH models already share L1 (n = 693), B retains 0.1082 against A's 0.1833, a paired gap of -0.07504. The direction holds, and this is a genuine paired comparison on identical pairs.
- One level deeper, on pairs where both models share L1+L2, the test relation leaves n = 25 common pairs, below the minimum cell of 30, so **the conditional L3 comparison is not estimable on the independent test relation**. It is reported, not inferred, on the other two relations (train n = 72, common_successor n = 60), both with B below A.

The corrected negative side changes how this is read. On the pairs where both models already share L1 and the NEGATIVE item also shares L1 (n = 32), A retains 0.0625 of those negatives while B retains 0.0000. At L2, B therefore splits behaviour-related pairs faster than A AND pushes unrelated pairs apart faster than A: the cost is paid in behaviour retention, the benefit in negative discrimination. v1 reported A's value here as 0.0000 because that branch read the positives, which inverted the sign of this comparison. At L3 the negative common set is empty for all three relations, so nothing can be said about negative discrimination there.

### Q3. Is B's lower collision rate just finer partitioning, and can prefix group size be separated out?
Yes, and group size does account for a large part of it. B has 16145 L2 prefixes against A's 7414 (2.18x) and 23695 against 20223 at L3, while L1 is NOT finer (241 against 247 prefixes). B's L1 is instead far more uniform: largest group 253 against A's 859. Collision falls from 17.7492% to 3.6279%.

Where the collision reduction actually sits, using the corrected ITEM-side singleton fraction: the share of items sitting alone in their prefix group is

| level | A items in singleton groups | B items in singleton groups | difference |
|---|---|---|---|
| level1 | 0.0325% | 0.0041% | -0.0285% |
| level2 | 15.6953% | 46.5205% | 30.8252% |
| level3 | 72.5465% | 93.5169% | 20.9704% |

So the improvement is carried by **L2 and L3, and L2 carries more of it**. L1 contributes essentially nothing: only 0.0041% of B's items sit alone in their L1 prefix against 0.0325% for A, a difference of only -0.0285%. The collision number is a full-triple quantity, and it falls because B hands out far more unique L2 codes, not because it separates anything at L1.

Group size can be separated, and separating it changes the reading of L2 and L3. At L2, the share of behaviour positives sitting in a SINGLETON prefix group is 47.9320% for B against 13.5013% for A; at L3 it is 93.2553% against 71.5831%. An item in a singleton group cannot share its prefix with anything, including its own behaviour sibling, so a large part of B's lower raw sharing rate is placement, not discrimination.

Within strata where the positive and the negative sit in groups of the SAME size bucket, B's positive sharing is at least A's, reversing the naive comparison:

| L2 bucket | n | A P(share\|pos) | B P(share\|pos) |
|---|---|---|---|
| 1 | 362 | 0.00000 | 0.00000 |
| 2-4 | 979 | 0.00511 | 0.00596 |
| 5-9 | 769 | 0.00520 | 0.01111 |

That matched view is itself limited and must not be read as the population answer: it covers only 21.9575% of pairs for A and 39.6898% for B at L2, because the two models put different pairs into matching buckets. The honest statement is therefore two-sided: B places far more behaviour positives into singleton groups, which lowers its raw sharing rate mechanically, AND on the genuinely paired given-shared-L1 comparison B does split behaviour pairs faster. Both are measured, and neither alone is the whole story.

### Identifiability and limitations
- `sampled_precision_given_share` is a property of the balanced positive/negative sample. The population P(behaviour-related | shared prefix) is **not identified** here: it needs the positive prior and the negative sampling probability, neither of which is estimated. It is not reported.
- Every negative means NO OBSERVED relation, never proven irrelevance.
- Test edges are used only to build negatives for evaluation; they enter no model and no training.
- Enrichment ratios are left undefined, never zero-filled, wherever the negative sharing rate is zero.
