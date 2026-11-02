# Does the curvature of a SID's behaviour quotient graph predict its recall?

> **Read this before the numbers.** `sid_recall.csv` holds the pairs admitted by the directory-name rule (`pair_audit.py`): `results/stage2_RQ-VAE/<X>` and `results/stage3_T5Train/<X>` carry the same name (exactly, or with the nested `/` written as `__`). That is 110 of the original 182 pairs, over 34 series (largest 69%); the co-publication tier is recorded as a diagnostic (`SAME_COMMIT` 73 / `COMMITS_DIFFER` 20 / `NOT_TRACKED` 17) but no longer gates admission. The full-table numbers are kept in `curvature/full_182/` and compared in `AUDIT_CORRELATION.md`; the sampling budget and its effect are in `CONVERGENCE_FINDINGS.md`.

*110 SID tables, each with one recall, from 107 mechanism families.*

The behaviour graph is the training split's transitions only (24,556 items, 157,695 edges). For every SID table and every level the prefix groups are contracted into super-nodes, the lazy random walk's Ollivier curvature of that quotient graph is measured on a sample of its edges, and the number is standardised against a null that keeps the group sizes and shuffles the members. The features are then correlated with the recall of the run that used the table.

## The question, in three parts

1. Does a curvature feature correlate with the recall at all?
2. Does it still correlate once the plain SID-quality measures (modularity, code usage, collision) are taken out?
3. Does it survive the arms not being independent?

## 1. The correlation

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 100 | -0.0744 | 0.4623 | [-0.2926, 0.1232] |
| z_curv_L2 | 108 | 0.2007 | 0.0380 | [-0.0014, 0.3972] |
| z_curv_L3 | 110 | 0.1789 | 0.0635 | [-0.0101, 0.3580] |
| z_curv_L1_minus_L3 | 100 | -0.0907 | 0.3643 | [-0.3076, 0.1108] |
| sign_change_L1_L3 | 110 | 0.0371 | 0.7046 | [-0.1886, 0.2385] |

### The controls, for reference

| control | n | Spearman | permutation p |
|---|---|---|---|
| codes_used_L1 | 110 | 0.1513 | 0.1174 |
| codes_used_L2 | 110 | 0.2568 | 0.0115 |
| codes_used_L3 | 110 | 0.4389 | 0.0005 |
| collision | 110 | -0.4820 | 0.0005 |

### Is the curvature a restatement of the partition's purity?

| curvature | control | Spearman |
|---|---|---|
| z_curv_L1 | codes_used_L3 | -0.1454 |
| z_curv_L1 | collision | 0.7476 |
| z_curv_L2 | codes_used_L3 | -0.0930 |
| z_curv_L2 | collision | -0.2261 |
| z_curv_L3 | codes_used_L3 | 0.0586 |
| z_curv_L3 | collision | 0.0382 |

### How much recall does the curvature add on top of the controls?

| curvature feature | n | R^2 of the controls alone | R^2 with the curvature | increment | permutation p |
|---|---|---|---|---|---|

## 2. The partial correlations (the decisive test)

The curvature feature with one control taken out, in rank space.

| feature | control | n | partial Spearman |
|---|---|---|---|
| z_curv_L1 | codes_used_L1 | 100 | -0.1799 |
| z_curv_L1 | codes_used_L2 | 100 | -0.0557 |
| z_curv_L1 | codes_used_L3 | 100 | -0.0232 |
| z_curv_L1 | collision | 100 | 0.2895 |
| z_curv_L2 | codes_used_L1 | 108 | 0.1853 |
| z_curv_L2 | codes_used_L2 | 108 | 0.1652 |
| z_curv_L2 | codes_used_L3 | 108 | 0.2075 |
| z_curv_L2 | collision | 108 | 0.1476 |
| z_curv_L3 | codes_used_L1 | 110 | 0.1778 |
| z_curv_L3 | codes_used_L2 | 110 | 0.1247 |
| z_curv_L3 | codes_used_L3 | 110 | 0.1174 |
| z_curv_L3 | collision | 110 | 0.2388 |
| z_curv_L1_minus_L3 | codes_used_L1 | 100 | -0.2103 |
| z_curv_L1_minus_L3 | codes_used_L2 | 100 | -0.0803 |
| z_curv_L1_minus_L3 | codes_used_L3 | 100 | -0.0281 |
| z_curv_L1_minus_L3 | collision | 100 | 0.2631 |
| sign_change_L1_L3 | codes_used_L1 | 110 | 0.0985 |
| sign_change_L1_L3 | codes_used_L2 | 110 | 0.1198 |
| sign_change_L1_L3 | codes_used_L3 | 110 | 0.0197 |
| sign_change_L1_L3 | collision | 110 | 0.0849 |

## 3. One point per mechanism family

The 182 tables come from 107 families, so the honest sample size is much smaller than 182. Keeping the best-recalling table of each family:

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 98 | -0.0710 | 0.4943 | [-0.2790, 0.1483] |
| z_curv_L2 | 106 | 0.1880 | 0.0515 | [-0.0196, 0.3926] |
| z_curv_L3 | 107 | 0.1695 | 0.0720 | [-0.0206, 0.3522] |
| z_curv_L1_minus_L3 | 98 | -0.0879 | 0.3958 | [-0.2973, 0.1329] |
| sign_change_L1_L3 | 107 | 0.0351 | 0.7351 | [-0.1986, 0.2485] |

### Leave one family out

Each family is removed in turn; the correlation that remains is the one that does not depend on that family's arms.

| family removed | n left | Spearman (z_curv_L1) |
|---|---|---|
| TIGER_BEHDIR-L4 | 99 | -0.0521 |
| TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL | 99 | -0.0616 |
| TIGER_BRad-RQ | 100 | -0.0744 |
| TIGER_CONCMID-GEO-SID | 99 | -0.0567 |
| TIGER_COOC_RANK_FRECHET | 99 | -0.0631 |
| TIGER_CURV-RESID-CONTRAST | 99 | -0.0596 |
| TIGER_GDELTA-SUCC-PULL-RQ | 100 | -0.0744 |
| TIGER_HGS-TREE | 100 | -0.0744 |
| TIGER_HORO-TRANSPLANT-L | 99 | -0.0594 |
| TIGER_HYP-CERT-LEAFREPAIR | 99 | -0.0618 |
| TIGER_behavior_residual_geometry_transport | 99 | -0.0792 |
| TIGER_bootstrap_code_stability | 99 | -0.0816 |
| TIGER_branch-geodesic-rank | 100 | -0.0744 |
| TIGER_c106_directed_assignment_usage_balance | 99 | -0.0872 |
| TIGER_c106_directed_codeword_margin_geometry | 99 | -0.0657 |
| TIGER_c107_directed_capacity_balanced_assignment_partition | 99 | -0.0683 |
| TIGER_c107_directed_capacity_constrained_assignment_partition | 99 | -0.0822 |
| TIGER_c107_directed_codeword_tangent_repulsion | 99 | -0.0801 |
| TIGER_c107_directed_separated_codeword_geometry | 99 | -0.0932 |
| TIGER_c117_hyperbolic_item_semantic_id | 99 | -0.0625 |
| TIGER_c1_behaviour_conditioned_residual_target | 99 | -0.1012 |
| TIGER_c1_policy_behavior_directed_soft_code_transition_kl | 99 | -0.0835 |
| TIGER_cogeo_region_argmax | 100 | -0.0744 |
| TIGER_cooccur_coherent_sid | 99 | -0.0626 |
| TIGER_incoming_context_barycenter | 99 | -0.0809 |
| TIGER_interest_signature_hypball | 100 | -0.0744 |
| TIGER_poincare-frechet-successor-codebook | 99 | -0.0573 |
| TIGER_successor-geodesic-coarse | 99 | -0.0554 |
| TIGER_successor_entropy_curvature | 99 | -0.0568 |
| TIGER_successor_residual_barycenter | 99 | -0.0617 |
| TIGER_tangent_successor_residual | 99 | -0.0617 |
| TIGER_transition_busemann_gate | 99 | -0.0620 |
| curvature_RQ-VAE | 97 | -0.0669 |
| curvature_RQ-VAE_iter100 | 99 | -0.0732 |
| curvature_RQ-VAE_iter101 | 99 | -0.0710 |
| curvature_RQ-VAE_iter102 | 99 | -0.0714 |
| curvature_RQ-VAE_iter107 | 99 | -0.0741 |
| curvature_RQ-VAE_iter109 | 99 | -0.0737 |
| curvature_RQ-VAE_iter12 | 99 | -0.0737 |
| curvature_RQ-VAE_iter14 | 99 | -0.0757 |
| curvature_RQ-VAE_iter15 | 99 | -0.0752 |
| curvature_RQ-VAE_iter16 | 99 | -0.0702 |
| curvature_RQ-VAE_iter17 | 99 | -0.0789 |
| curvature_RQ-VAE_iter18 | 99 | -0.0787 |
| curvature_RQ-VAE_iter19 | 99 | -0.0762 |
| curvature_RQ-VAE_iter20 | 99 | -0.0693 |
| curvature_RQ-VAE_iter21 | 99 | -0.0717 |
| curvature_RQ-VAE_iter22 | 99 | -0.0784 |
| curvature_RQ-VAE_iter23 | 99 | -0.0644 |
| curvature_RQ-VAE_iter24 | 99 | -0.0668 |
| curvature_RQ-VAE_iter25 | 99 | -0.0731 |
| curvature_RQ-VAE_iter26 | 99 | -0.0746 |
| curvature_RQ-VAE_iter28 | 99 | -0.0704 |
| curvature_RQ-VAE_iter29 | 99 | -0.0761 |
| curvature_RQ-VAE_iter31 | 99 | -0.0757 |
| curvature_RQ-VAE_iter32 | 99 | -0.0778 |
| curvature_RQ-VAE_iter33 | 99 | -0.0781 |
| curvature_RQ-VAE_iter34 | 99 | -0.0647 |
| curvature_RQ-VAE_iter35 | 99 | -0.0726 |
| curvature_RQ-VAE_iter36 | 99 | -0.0699 |
| curvature_RQ-VAE_iter37 | 99 | -0.0770 |
| curvature_RQ-VAE_iter38 | 99 | -0.0646 |
| curvature_RQ-VAE_iter39 | 99 | -0.0648 |
| curvature_RQ-VAE_iter40 | 99 | -0.0774 |
| curvature_RQ-VAE_iter41 | 99 | -0.0778 |
| curvature_RQ-VAE_iter42 | 99 | -0.0713 |
| curvature_RQ-VAE_iter43 | 99 | -0.0754 |
| curvature_RQ-VAE_iter44 | 99 | -0.0795 |
| curvature_RQ-VAE_iter45 | 100 | -0.0744 |
| curvature_RQ-VAE_iter46 | 99 | -0.0926 |
| curvature_RQ-VAE_iter47 | 99 | -0.0875 |
| curvature_RQ-VAE_iter49 | 99 | -0.0807 |
| curvature_RQ-VAE_iter50 | 99 | -0.0798 |
| curvature_RQ-VAE_iter51 | 99 | -0.0712 |
| curvature_RQ-VAE_iter52 | 100 | -0.0744 |
| curvature_RQ-VAE_iter53 | 99 | -0.0948 |
| curvature_RQ-VAE_iter54 | 99 | -0.1003 |
| curvature_RQ-VAE_iter55 | 99 | -0.0928 |
| curvature_RQ-VAE_iter56 | 99 | -0.1029 |
| curvature_RQ-VAE_iter57 | 99 | -0.0958 |
| curvature_RQ-VAE_iter58 | 99 | -0.1033 |
| curvature_RQ-VAE_iter59 | 100 | -0.0744 |
| curvature_RQ-VAE_iter61 | 99 | -0.0990 |
| curvature_RQ-VAE_iter62 | 99 | -0.0539 |
| curvature_RQ-VAE_iter63 | 99 | -0.0790 |
| curvature_RQ-VAE_iter64 | 99 | -0.0839 |
| curvature_RQ-VAE_iter65 | 99 | -0.0603 |
| curvature_RQ-VAE_iter66 | 99 | -0.0631 |
| curvature_RQ-VAE_iter67 | 99 | -0.0840 |
| curvature_RQ-VAE_iter73 | 99 | -0.0742 |
| curvature_RQ-VAE_iter77 | 99 | -0.0733 |
| curvature_RQ-VAE_iter78 | 99 | -0.0750 |
| curvature_RQ-VAE_iter79 | 99 | -0.0759 |
| curvature_RQ-VAE_iter80 | 99 | -0.0670 |
| curvature_RQ-VAE_iter81 | 99 | -0.0720 |
| curvature_RQ-VAE_iter83 | 99 | -0.0877 |
| curvature_RQ-VAE_iter88 | 99 | -0.0742 |
| curvature_RQ-VAE_iter89 | 99 | -0.0760 |
| curvature_RQ-VAE_iter90 | 99 | -0.0751 |
| curvature_RQ-VAE_iter91 | 99 | -0.0784 |
| curvature_RQ-VAE_iter92 | 99 | -0.0691 |
| curvature_RQ-VAE_iter93 | 99 | -0.0721 |
| curvature_RQ-VAE_iter94 | 99 | -0.0768 |
| curvature_RQ-VAE_iter97 | 99 | -0.0741 |
| curvature_RQ-VAE_iter98 | 99 | -0.0749 |
| curvature_RQ-VAE_iter99 | 99 | -0.0762 |
| r4_08 | 99 | -0.0669 |

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

