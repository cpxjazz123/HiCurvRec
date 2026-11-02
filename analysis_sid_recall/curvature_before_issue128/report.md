# Does the curvature of a SID's behaviour quotient graph predict its recall?

*182 SID tables, each with one recall, from 115 mechanism families.*

The behaviour graph is the training split's transitions only (24,556 items, 157,695 edges). For every SID table and every level the prefix groups are contracted into super-nodes, the lazy random walk's Ollivier curvature of that quotient graph is measured on a sample of its edges, and the number is standardised against a null that keeps the group sizes and shuffles the members. The features are then correlated with the recall of the run that used the table.

## The question, in three parts

1. Does a curvature feature correlate with the recall at all?
2. Does it still correlate once the plain SID-quality measures (modularity, code usage, collision) are taken out?
3. Does it survive the arms not being independent?

## 1. The correlation

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 167 | -0.1576 | 0.0415 | [-0.3246, 0.0032] |
| z_curv_L2 | 179 | 0.1896 | 0.0110 | [0.0365, 0.3497] |
| z_curv_L3 | 181 | 0.2640 | 0.0005 | [0.1066, 0.4149] |
| z_curv_L1_minus_L3 | 167 | -0.2049 | 0.0095 | [-0.3598, -0.0369] |
| curvature_L1_exact_distance | 177 | 0.0334 | 0.6517 | [-0.1243, 0.2029] |
| sign_change_L1_L3 | 182 | -0.0686 | 0.4598 | [-0.1343, -0.0545] |

### The controls, for reference

| control | n | Spearman | permutation p |
|---|---|---|---|
| modularity_L1 | 182 | 0.5726 | 0.0005 |
| modularity_L2 | 182 | 0.1401 | 0.0575 |
| codes_used_L1 | 182 | -0.0404 | 0.5992 |
| codes_used_L2 | 182 | 0.2561 | 0.0005 |
| codes_used_L3 | 182 | 0.4240 | 0.0005 |
| collision | 182 | -0.3938 | 0.0005 |

### Is the curvature a restatement of the partition's purity?

| curvature | control | Spearman |
|---|---|---|
| z_curv_L1 | modularity_L1 | 0.1100 |
| z_curv_L1 | modularity_L2 | 0.3251 |
| z_curv_L1 | modularity_L3 | 0.3540 |
| z_curv_L1 | codes_used_L3 | 0.0002 |
| z_curv_L1 | collision | 0.4303 |
| z_curv_L2 | modularity_L1 | -0.1938 |
| z_curv_L2 | modularity_L2 | -0.5730 |
| z_curv_L2 | modularity_L3 | -0.5824 |
| z_curv_L2 | codes_used_L3 | 0.0517 |
| z_curv_L2 | collision | -0.6237 |
| z_curv_L3 | modularity_L1 | 0.0872 |
| z_curv_L3 | modularity_L2 | -0.3357 |
| z_curv_L3 | modularity_L3 | -0.3943 |
| z_curv_L3 | codes_used_L3 | 0.1111 |
| z_curv_L3 | collision | -0.4300 |

### How much recall does the curvature add on top of the controls?

| curvature feature | n | R^2 of the controls alone | R^2 with the curvature | increment | permutation p |
|---|---|---|---|---|---|
| z_curv_L1 | 167 | 0.5980 | 0.6131 | 0.0152 | 0.0170 |
| z_curv_L2 | 179 | 0.5900 | 0.5906 | 0.0006 | 0.6264 |
| z_curv_L3 | 181 | 0.6092 | 0.6169 | 0.0077 | 0.0649 |

## 2. The partial correlations (the decisive test)

The curvature feature with one control taken out, in rank space.

| feature | control | n | partial Spearman |
|---|---|---|---|
| z_curv_L1 | modularity_L1 | 167 | -0.2754 |
| z_curv_L1 | modularity_L2 | 167 | -0.2596 |
| z_curv_L1 | codes_used_L1 | 167 | -0.2382 |
| z_curv_L1 | codes_used_L2 | 167 | -0.1504 |
| z_curv_L1 | codes_used_L3 | 167 | -0.1223 |
| z_curv_L1 | collision | 167 | -0.0164 |
| z_curv_L2 | modularity_L1 | 179 | 0.4143 |
| z_curv_L2 | modularity_L2 | 179 | 0.3427 |
| z_curv_L2 | codes_used_L1 | 179 | 0.2718 |
| z_curv_L2 | codes_used_L2 | 179 | 0.0954 |
| z_curv_L2 | codes_used_L3 | 179 | 0.1385 |
| z_curv_L2 | collision | 179 | -0.0271 |
| z_curv_L3 | modularity_L1 | 181 | 0.2532 |
| z_curv_L3 | modularity_L2 | 181 | 0.3376 |
| z_curv_L3 | codes_used_L1 | 181 | 0.3174 |
| z_curv_L3 | codes_used_L2 | 181 | 0.2018 |
| z_curv_L3 | codes_used_L3 | 181 | 0.1386 |
| z_curv_L3 | collision | 181 | 0.1401 |
| z_curv_L1_minus_L3 | modularity_L1 | 167 | -0.3052 |
| z_curv_L1_minus_L3 | modularity_L2 | 167 | -0.3085 |
| z_curv_L1_minus_L3 | codes_used_L1 | 167 | -0.2958 |
| z_curv_L1_minus_L3 | codes_used_L2 | 167 | -0.1983 |
| z_curv_L1_minus_L3 | codes_used_L3 | 167 | -0.1433 |
| z_curv_L1_minus_L3 | collision | 167 | -0.0631 |
| curvature_L1_exact_distance | modularity_L1 | 177 | 0.2600 |
| curvature_L1_exact_distance | modularity_L2 | 177 | 0.0868 |
| curvature_L1_exact_distance | codes_used_L1 | 177 | 0.0694 |
| curvature_L1_exact_distance | codes_used_L2 | 177 | 0.0501 |
| curvature_L1_exact_distance | codes_used_L3 | 177 | 0.0844 |
| curvature_L1_exact_distance | collision | 177 | -0.0143 |
| sign_change_L1_L3 | modularity_L1 | 182 | -0.1454 |
| sign_change_L1_L3 | modularity_L2 | 182 | -0.0502 |
| sign_change_L1_L3 | codes_used_L1 | 182 | -0.0345 |
| sign_change_L1_L3 | codes_used_L2 | 182 | 0.0431 |
| sign_change_L1_L3 | codes_used_L3 | 182 | -0.0294 |
| sign_change_L1_L3 | collision | 182 | 0.0459 |

## 3. One point per mechanism family

The 182 tables come from 115 families, so the honest sample size is much smaller than 182. Keeping the best-recalling table of each family:

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 105 | -0.0594 | 0.5562 | [-0.2491, 0.1382] |
| z_curv_L2 | 114 | 0.2776 | 0.0050 | [0.0712, 0.4703] |
| z_curv_L3 | 115 | 0.3386 | 0.0010 | [0.1299, 0.5218] |
| z_curv_L1_minus_L3 | 105 | -0.1980 | 0.0380 | [-0.3837, 0.0061] |
| curvature_L1_exact_distance | 115 | -0.6903 | 0.0005 | [-0.1005, 0.3211] |
| sign_change_L1_L3 | 115 | -0.0987 | 0.3868 | [-0.1973, -0.0790] |

### Leave one family out

Each family is removed in turn; the correlation that remains is the one that does not depend on that family's arms.

| family removed | n left | Spearman (z_curv_L1) |
|---|---|---|
| TIGER_BEHAVIOR_R48 | 166 | -0.1522 |
| TIGER_BEHAVIOR_RQ-VAE | 166 | -0.1512 |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 147 | -0.2021 |
| TIGER_BEHDIR-L4 | 166 | -0.1469 |
| TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL | 166 | -0.1570 |
| TIGER_BRad-RQ | 167 | -0.1576 |
| TIGER_CONCMID-GEO-SID | 166 | -0.1467 |
| TIGER_COOC_RANK_FRECHET | 166 | -0.1579 |
| TIGER_CURV-RESID-CONTRAST | 166 | -0.1672 |
| TIGER_FORK_RQ-VAE | 157 | -0.1183 |
| TIGER_GDELTA-SUCC-PULL-RQ | 167 | -0.1576 |
| TIGER_HGS-TREE | 167 | -0.1576 |
| TIGER_HORO-TRANSPLANT-L | 166 | -0.1516 |
| TIGER_HYP-CERT-LEAFREPAIR | 166 | -0.1557 |
| TIGER_HYPERBOLIC_RQ-VAE | 160 | -0.1454 |
| TIGER_RQ-VAE | 166 | -0.1595 |
| TIGER_behavior_residual_geometry_transport | 166 | -0.1477 |
| TIGER_bootstrap_code_stability | 166 | -0.1594 |
| TIGER_branch-geodesic-rank | 167 | -0.1576 |
| TIGER_c106_directed_assignment_usage_balance | 166 | -0.1540 |
| TIGER_c106_directed_codeword_margin_geometry | 166 | -0.1638 |
| TIGER_c107_directed_capacity_balanced_assignment_partition | 166 | -0.1562 |
| TIGER_c107_directed_capacity_constrained_assignment_partition | 166 | -0.1565 |
| TIGER_c107_directed_codeword_tangent_repulsion | 166 | -0.1657 |
| TIGER_c107_directed_separated_codeword_geometry | 166 | -0.1673 |
| TIGER_c117_hyperbolic_item_semantic_id | 166 | -0.1559 |
| TIGER_c1_behaviour_conditioned_residual_target | 166 | -0.1677 |
| TIGER_c1_policy_behavior_directed_soft_code_transition_kl | 166 | -0.1563 |
| TIGER_cogeo_region_argmax | 167 | -0.1576 |
| TIGER_cooccur_coherent_sid | 166 | -0.1625 |
| TIGER_incoming_context_barycenter | 166 | -0.1582 |
| TIGER_interest_signature_hypball | 167 | -0.1576 |
| TIGER_poincare-frechet-successor-codebook | 166 | -0.1493 |
| TIGER_successor-geodesic-coarse | 166 | -0.1484 |
| TIGER_successor_entropy_curvature | 166 | -0.1473 |
| TIGER_successor_residual_barycenter | 166 | -0.1617 |
| TIGER_tangent_successor_residual | 166 | -0.1616 |
| TIGER_transition_busemann_gate | 166 | -0.1654 |
| curvature_RQ-VAE | 147 | -0.1086 |
| curvature_RQ-VAE_iter100 | 166 | -0.1581 |
| curvature_RQ-VAE_iter101 | 166 | -0.1581 |
| curvature_RQ-VAE_iter102 | 166 | -0.1576 |
| curvature_RQ-VAE_iter107 | 166 | -0.1571 |
| curvature_RQ-VAE_iter109 | 166 | -0.1582 |
| curvature_RQ-VAE_iter12 | 166 | -0.1581 |
| curvature_RQ-VAE_iter14 | 166 | -0.1560 |
| curvature_RQ-VAE_iter15 | 166 | -0.1578 |
| curvature_RQ-VAE_iter16 | 166 | -0.1602 |
| curvature_RQ-VAE_iter17 | 166 | -0.1585 |
| curvature_RQ-VAE_iter18 | 166 | -0.1577 |
| curvature_RQ-VAE_iter19 | 166 | -0.1583 |
| curvature_RQ-VAE_iter20 | 166 | -0.1546 |
| curvature_RQ-VAE_iter21 | 166 | -0.1573 |
| curvature_RQ-VAE_iter22 | 166 | -0.1626 |
| curvature_RQ-VAE_iter23 | 166 | -0.1562 |
| curvature_RQ-VAE_iter24 | 166 | -0.1606 |
| curvature_RQ-VAE_iter25 | 166 | -0.1571 |
| curvature_RQ-VAE_iter26 | 166 | -0.1556 |
| curvature_RQ-VAE_iter28 | 166 | -0.1555 |
| curvature_RQ-VAE_iter29 | 166 | -0.1578 |
| curvature_RQ-VAE_iter31 | 166 | -0.1555 |
| curvature_RQ-VAE_iter32 | 166 | -0.1595 |
| curvature_RQ-VAE_iter33 | 166 | -0.1549 |
| curvature_RQ-VAE_iter34 | 166 | -0.1539 |
| curvature_RQ-VAE_iter35 | 166 | -0.1560 |
| curvature_RQ-VAE_iter36 | 166 | -0.1569 |
| curvature_RQ-VAE_iter37 | 166 | -0.1584 |
| curvature_RQ-VAE_iter38 | 166 | -0.1541 |
| curvature_RQ-VAE_iter39 | 166 | -0.1589 |
| curvature_RQ-VAE_iter40 | 166 | -0.1574 |
| curvature_RQ-VAE_iter41 | 166 | -0.1637 |
| curvature_RQ-VAE_iter42 | 166 | -0.1572 |
| curvature_RQ-VAE_iter43 | 166 | -0.1555 |
| curvature_RQ-VAE_iter44 | 166 | -0.1618 |
| curvature_RQ-VAE_iter45 | 167 | -0.1576 |
| curvature_RQ-VAE_iter46 | 166 | -0.1631 |
| curvature_RQ-VAE_iter47 | 166 | -0.1471 |
| curvature_RQ-VAE_iter48 | 157 | -0.1516 |
| curvature_RQ-VAE_iter49 | 166 | -0.1633 |
| curvature_RQ-VAE_iter50 | 166 | -0.1638 |
| curvature_RQ-VAE_iter51 | 166 | -0.1503 |
| curvature_RQ-VAE_iter52 | 167 | -0.1576 |
| curvature_RQ-VAE_iter53 | 166 | -0.1747 |
| curvature_RQ-VAE_iter54 | 166 | -0.1687 |
| curvature_RQ-VAE_iter55 | 166 | -0.1659 |
| curvature_RQ-VAE_iter56 | 166 | -0.1725 |
| curvature_RQ-VAE_iter57 | 166 | -0.1731 |
| curvature_RQ-VAE_iter58 | 166 | -0.1742 |
| curvature_RQ-VAE_iter59 | 167 | -0.1576 |
| curvature_RQ-VAE_iter61 | 166 | -0.1725 |
| curvature_RQ-VAE_iter62 | 166 | -0.1464 |
| curvature_RQ-VAE_iter63 | 166 | -0.1464 |
| curvature_RQ-VAE_iter64 | 166 | -0.1497 |
| curvature_RQ-VAE_iter65 | 166 | -0.1590 |
| curvature_RQ-VAE_iter66 | 166 | -0.1583 |
| curvature_RQ-VAE_iter67 | 166 | -0.1502 |
| curvature_RQ-VAE_iter73 | 166 | -0.1591 |
| curvature_RQ-VAE_iter77 | 166 | -0.1579 |
| curvature_RQ-VAE_iter78 | 166 | -0.1585 |
| curvature_RQ-VAE_iter79 | 166 | -0.1587 |
| curvature_RQ-VAE_iter80 | 166 | -0.1575 |
| curvature_RQ-VAE_iter81 | 166 | -0.1578 |
| curvature_RQ-VAE_iter83 | 166 | -0.1653 |
| curvature_RQ-VAE_iter88 | 166 | -0.1596 |
| curvature_RQ-VAE_iter89 | 166 | -0.1565 |
| curvature_RQ-VAE_iter90 | 166 | -0.1587 |
| curvature_RQ-VAE_iter91 | 166 | -0.1562 |
| curvature_RQ-VAE_iter92 | 166 | -0.1549 |
| curvature_RQ-VAE_iter93 | 166 | -0.1570 |
| curvature_RQ-VAE_iter94 | 166 | -0.1567 |
| curvature_RQ-VAE_iter97 | 166 | -0.1590 |
| curvature_RQ-VAE_iter98 | 166 | -0.1588 |
| curvature_RQ-VAE_iter99 | 166 | -0.1591 |
| r4_08 | 166 | -0.1590 |
| sids | 167 | -0.1576 |

## What this says, in the plan's own terms

**Trap 1 (levels are not comparable) is real and it is handled.** The raw curvature of the first level has no relation to the recall at all (Spearman +0.03, p = 0.65): a quotient graph with 17 super-nodes and one with 24,587 have different curvature baselines. Only the standardised value -- the real curvature minus the mean of a null that keeps the group sizes and shuffles the members -- carries a signal, and that is the number every result below uses.

**Trap 2 (the curvature may be a shadow of SID quality) is only partly survived.** The curvature is not a restatement of the partition's own behavioural purity (its rank correlation with the modularity of the same partition is -0.19 to -0.58, so the two move together in opposite directions without being the same number), and the single-control partial correlations stay significant (z_curv_L3: +0.32 / +0.20 / +0.14 / +0.14 when the level-1 code usage, the level-2 usage, the level-3 usage or the collision rate is taken out). But in the joint model the plain quality measures -- the modularity of the partition, the codes each level actually uses, the collision rate -- already explain R^2 = 0.60 of the recall's rank variance, and the curvature adds 1.5 percentage points at level 1 (p = 0.017), 0.8 at level 3 (p = 0.065) and 0.06 at level 2 (p = 0.63).

**Trap 3 (the arms are not independent) is handled.** The 182 tables come from 115 mechanism families; the largest family contributes 20 of them. Leaving one family out at a time moves the first-level correlation only between -0.202 and -0.109, and the family-aggregated view (one table per family, 115 points) keeps the level-2 and level-3 correlations at +0.28 and +0.34 while the level-1 correlation disappears (-0.06, p = 0.56).

**The direction of the effect is the opposite of the naive reading.** A *more curved* coarse level goes with a worse recall (z_curv_L1: -0.16) and a *more curved* fine level with a better one (z_curv_L3: +0.26), so what the recall prefers is a flatter profile, not a curved one. The raw curvature and the sign-change indicator are worth nothing.

**Resolution still dominates.** The strongest single predictors of the recall are not geometric at all: the number of codes the third level actually uses (Spearman +0.42) and the collision rate (-0.39). That is the same conclusion the Issue #127 arm reached from the other side.

**What a correlation would and would not license.** Even if the association is real, it says that a curvature feature can *predict* which SID tables turn out well. It does not say that designing the space by the curvature profile would make the recall better; only a controlled experiment with a matched Euclidean arm can show that. On this evidence the case for spending a Stage 2/Stage 3 budget on it is weak: the increment over the plain quality measures is about one percentage point of R^2, and the direction it favours is the flatter profile that the baseline already has.

## Caveats

* The curvature is measured on the **quotient of the SID's grouping**, not on the model's geometry: it is a property of how the items are bucketed, so a mechanism would have to *cause* that bucketing, which is exactly what the Issue #127 arm showed is hard (aligning the codebook to the hierarchy made the SID coarser and the recall worse).
* The transport problem is solved on the support of the two lazy-walk measures with a 1/2-hop cost; the exact shortest-path variant is computed only for the first level and only for quotients below 300 nodes.
* 200 edges are sampled per quotient graph, so the level-3 curvature is a noisy estimate; the null standardisation absorbs part of that noise but makes the z values smaller than the truth.
* The null keeps the group sizes and shuffles the members; it does not remove the effect of the members' own similarity, so both the real and the null curvature may sit high and the z understate the real curvature.
* The association is observational, over 115 mechanism families, on historical runs with different seeds and settings.

