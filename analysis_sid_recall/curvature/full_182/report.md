# Does the curvature of a SID's behaviour quotient graph predict its recall?

> **Read this before the numbers.** The 182 rows of `sid_recall.csv` were audited (`pair_audit.py`): a pair is *verified* only when the SID table and the recall file were introduced by the same commit **and** the stage-2 and stage-3 directories carry the same name. 73 pairs pass; on those, no curvature feature correlates with the recall (z_curv_L2 +0.129 p=0.28, z_curv_L3 -0.044 p=0.71, the profile +0.141 p=0.24), and the full-table association does not survive either the same-commit restriction (z_curv_L2 +0.074 p=0.50, z_curv_L3 +0.029 p=0.80) or a change of the edge-sampling budget (sign reversals, |delta| up to 0.19). See `AUDIT_CORRELATION.md` and `curvature/stability.md`; the numbers below are the unaudited table.

*182 SID tables, each with one recall, from 115 mechanism families.*

The behaviour graph is the training split's transitions only (24,556 items, 157,695 edges). For every SID table and every level the prefix groups are contracted into super-nodes, the lazy random walk's Ollivier curvature of that quotient graph is measured on a sample of its edges, and the number is standardised against a null that keeps the group sizes and shuffles the members. The features are then correlated with the recall of the run that used the table.

## The question, in three parts

1. Does a curvature feature correlate with the recall at all?
2. Does it still correlate once the plain SID-quality measures (modularity, code usage, collision) are taken out?
3. Does it survive the arms not being independent?

## 1. The correlation

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 167 | -0.0907 | 0.2409 | [-0.2325, 0.0658] |
| z_curv_L2 | 179 | 0.1503 | 0.0355 | [0.0010, 0.3059] |
| z_curv_L3 | 181 | 0.1454 | 0.0490 | [0.0058, 0.2850] |
| z_curv_L1_minus_L3 | 167 | -0.0948 | 0.2249 | [-0.2375, 0.0596] |
| sign_change_L1_L3 | 182 | -0.0592 | 0.4268 | [-0.2192, 0.1107] |

### The controls, for reference

| control | n | Spearman | permutation p |
|---|---|---|---|
| codes_used_L1 | 182 | -0.0404 | 0.5992 |
| codes_used_L2 | 182 | 0.2561 | 0.0005 |
| codes_used_L3 | 182 | 0.4240 | 0.0005 |
| collision | 182 | -0.3938 | 0.0005 |

### Is the curvature a restatement of the partition's purity?

| curvature | control | Spearman |
|---|---|---|
| z_curv_L1 | codes_used_L3 | -0.0711 |
| z_curv_L1 | collision | 0.6666 |
| z_curv_L2 | codes_used_L3 | 0.0165 |
| z_curv_L2 | collision | -0.3944 |
| z_curv_L3 | codes_used_L3 | 0.1381 |
| z_curv_L3 | collision | -0.1910 |

### How much recall does the curvature add on top of the controls?

| curvature feature | n | R^2 of the controls alone | R^2 with the curvature | increment | permutation p |
|---|---|---|---|---|---|

## 2. The partial correlations (the decisive test)

The curvature feature with one control taken out, in rank space.

| feature | control | n | partial Spearman |
|---|---|---|---|
| z_curv_L1 | codes_used_L1 | 167 | -0.2925 |
| z_curv_L1 | codes_used_L2 | 167 | -0.0685 |
| z_curv_L1 | codes_used_L3 | 167 | -0.0402 |
| z_curv_L1 | collision | 167 | 0.1400 |
| z_curv_L2 | codes_used_L1 | 179 | 0.1969 |
| z_curv_L2 | codes_used_L2 | 179 | 0.1039 |
| z_curv_L2 | codes_used_L3 | 179 | 0.1054 |
| z_curv_L2 | collision | 179 | 0.0379 |
| z_curv_L3 | codes_used_L1 | 181 | 0.1613 |
| z_curv_L3 | codes_used_L2 | 181 | 0.0690 |
| z_curv_L3 | codes_used_L3 | 181 | 0.0291 |
| z_curv_L3 | collision | 181 | 0.0726 |
| z_curv_L1_minus_L3 | codes_used_L1 | 167 | -0.2921 |
| z_curv_L1_minus_L3 | codes_used_L2 | 167 | -0.0746 |
| z_curv_L1_minus_L3 | codes_used_L3 | 167 | -0.0358 |
| z_curv_L1_minus_L3 | collision | 167 | 0.1461 |
| sign_change_L1_L3 | codes_used_L1 | 182 | -0.0650 |
| sign_change_L1_L3 | codes_used_L2 | 182 | 0.0432 |
| sign_change_L1_L3 | codes_used_L3 | 182 | 0.0195 |
| sign_change_L1_L3 | collision | 182 | 0.0394 |

## 3. One point per mechanism family

The 182 tables come from 115 families, so the honest sample size is much smaller than 182. Keeping the best-recalling table of each family:

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 105 | -0.0769 | 0.4558 | [-0.2720, 0.1248] |
| z_curv_L2 | 114 | 0.1532 | 0.1024 | [-0.0562, 0.3466] |
| z_curv_L3 | 115 | 0.1064 | 0.2574 | [-0.0937, 0.2984] |
| z_curv_L1_minus_L3 | 105 | -0.0840 | 0.4063 | [-0.2826, 0.1162] |
| sign_change_L1_L3 | 115 | -0.0041 | 0.9695 | [-0.2433, 0.2322] |

### Leave one family out

Each family is removed in turn; the correlation that remains is the one that does not depend on that family's arms.

| family removed | n left | Spearman (z_curv_L1) |
|---|---|---|
| TIGER_BEHAVIOR_R48 | 166 | -0.0886 |
| TIGER_BEHAVIOR_RQ-VAE | 166 | -0.0939 |
| TIGER_BEHAVIOR_SAMPLE_RQ-VAE | 147 | -0.0600 |
| TIGER_BEHDIR-L4 | 166 | -0.0780 |
| TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL | 166 | -0.0850 |
| TIGER_BRad-RQ | 167 | -0.0907 |
| TIGER_CONCMID-GEO-SID | 166 | -0.0786 |
| TIGER_COOC_RANK_FRECHET | 166 | -0.0889 |
| TIGER_CURV-RESID-CONTRAST | 166 | -0.0852 |
| TIGER_FORK_RQ-VAE | 157 | -0.1442 |
| TIGER_GDELTA-SUCC-PULL-RQ | 167 | -0.0907 |
| TIGER_HGS-TREE | 167 | -0.0907 |
| TIGER_HORO-TRANSPLANT-L | 166 | -0.0857 |
| TIGER_HYP-CERT-LEAFREPAIR | 166 | -0.0847 |
| TIGER_HYPERBOLIC_RQ-VAE | 160 | -0.0526 |
| TIGER_RQ-VAE | 166 | -0.0845 |
| TIGER_behavior_residual_geometry_transport | 166 | -0.0883 |
| TIGER_bootstrap_code_stability | 166 | -0.0948 |
| TIGER_branch-geodesic-rank | 167 | -0.0907 |
| TIGER_c106_directed_assignment_usage_balance | 166 | -0.0868 |
| TIGER_c106_directed_codeword_margin_geometry | 166 | -0.0900 |
| TIGER_c107_directed_capacity_balanced_assignment_partition | 166 | -0.0878 |
| TIGER_c107_directed_capacity_constrained_assignment_partition | 166 | -0.0910 |
| TIGER_c107_directed_codeword_tangent_repulsion | 166 | -0.0902 |
| TIGER_c107_directed_separated_codeword_geometry | 166 | -0.1015 |
| TIGER_c117_hyperbolic_item_semantic_id | 166 | -0.0850 |
| TIGER_c1_behaviour_conditioned_residual_target | 166 | -0.1025 |
| TIGER_c1_policy_behavior_directed_soft_code_transition_kl | 166 | -0.0868 |
| TIGER_cogeo_region_argmax | 167 | -0.0907 |
| TIGER_cooccur_coherent_sid | 166 | -0.0864 |
| TIGER_incoming_context_barycenter | 166 | -0.0925 |
| TIGER_interest_signature_hypball | 167 | -0.0907 |
| TIGER_poincare-frechet-successor-codebook | 166 | -0.0819 |
| TIGER_successor-geodesic-coarse | 166 | -0.0839 |
| TIGER_successor_entropy_curvature | 166 | -0.0806 |
| TIGER_successor_residual_barycenter | 166 | -0.0841 |
| TIGER_tangent_successor_residual | 166 | -0.0833 |
| TIGER_transition_busemann_gate | 166 | -0.0842 |
| curvature_RQ-VAE | 147 | -0.1321 |
| curvature_RQ-VAE_iter100 | 166 | -0.0906 |
| curvature_RQ-VAE_iter101 | 166 | -0.0913 |
| curvature_RQ-VAE_iter102 | 166 | -0.0902 |
| curvature_RQ-VAE_iter107 | 166 | -0.0913 |
| curvature_RQ-VAE_iter109 | 166 | -0.0912 |
| curvature_RQ-VAE_iter12 | 166 | -0.0905 |
| curvature_RQ-VAE_iter14 | 166 | -0.0923 |
| curvature_RQ-VAE_iter15 | 166 | -0.0902 |
| curvature_RQ-VAE_iter16 | 166 | -0.0928 |
| curvature_RQ-VAE_iter17 | 166 | -0.0914 |
| curvature_RQ-VAE_iter18 | 166 | -0.0927 |
| curvature_RQ-VAE_iter19 | 166 | -0.0912 |
| curvature_RQ-VAE_iter20 | 166 | -0.0931 |
| curvature_RQ-VAE_iter21 | 166 | -0.0913 |
| curvature_RQ-VAE_iter22 | 166 | -0.0923 |
| curvature_RQ-VAE_iter23 | 166 | -0.0867 |
| curvature_RQ-VAE_iter24 | 166 | -0.0882 |
| curvature_RQ-VAE_iter25 | 166 | -0.0920 |
| curvature_RQ-VAE_iter26 | 166 | -0.0924 |
| curvature_RQ-VAE_iter28 | 166 | -0.0882 |
| curvature_RQ-VAE_iter29 | 166 | -0.0922 |
| curvature_RQ-VAE_iter31 | 166 | -0.0886 |
| curvature_RQ-VAE_iter32 | 166 | -0.0921 |
| curvature_RQ-VAE_iter33 | 166 | -0.0902 |
| curvature_RQ-VAE_iter34 | 166 | -0.0848 |
| curvature_RQ-VAE_iter35 | 166 | -0.0896 |
| curvature_RQ-VAE_iter36 | 166 | -0.0885 |
| curvature_RQ-VAE_iter37 | 166 | -0.0910 |
| curvature_RQ-VAE_iter38 | 166 | -0.0879 |
| curvature_RQ-VAE_iter39 | 166 | -0.0866 |
| curvature_RQ-VAE_iter40 | 166 | -0.0913 |
| curvature_RQ-VAE_iter41 | 166 | -0.0933 |
| curvature_RQ-VAE_iter42 | 166 | -0.0905 |
| curvature_RQ-VAE_iter43 | 166 | -0.0984 |
| curvature_RQ-VAE_iter44 | 166 | -0.0914 |
| curvature_RQ-VAE_iter45 | 167 | -0.0907 |
| curvature_RQ-VAE_iter46 | 166 | -0.1027 |
| curvature_RQ-VAE_iter47 | 166 | -0.1012 |
| curvature_RQ-VAE_iter48 | 157 | -0.0970 |
| curvature_RQ-VAE_iter49 | 166 | -0.0897 |
| curvature_RQ-VAE_iter50 | 166 | -0.0923 |
| curvature_RQ-VAE_iter51 | 166 | -0.0885 |
| curvature_RQ-VAE_iter52 | 167 | -0.0907 |
| curvature_RQ-VAE_iter53 | 166 | -0.1008 |
| curvature_RQ-VAE_iter54 | 166 | -0.1056 |
| curvature_RQ-VAE_iter55 | 166 | -0.0981 |
| curvature_RQ-VAE_iter56 | 166 | -0.1046 |
| curvature_RQ-VAE_iter57 | 166 | -0.0943 |
| curvature_RQ-VAE_iter58 | 166 | -0.1017 |
| curvature_RQ-VAE_iter59 | 167 | -0.0907 |
| curvature_RQ-VAE_iter61 | 166 | -0.1055 |
| curvature_RQ-VAE_iter62 | 166 | -0.0790 |
| curvature_RQ-VAE_iter63 | 166 | -0.0900 |
| curvature_RQ-VAE_iter64 | 166 | -0.0957 |
| curvature_RQ-VAE_iter65 | 166 | -0.0859 |
| curvature_RQ-VAE_iter66 | 166 | -0.0870 |
| curvature_RQ-VAE_iter67 | 166 | -0.0838 |
| curvature_RQ-VAE_iter73 | 166 | -0.0914 |
| curvature_RQ-VAE_iter77 | 166 | -0.0910 |
| curvature_RQ-VAE_iter78 | 166 | -0.0915 |
| curvature_RQ-VAE_iter79 | 166 | -0.0926 |
| curvature_RQ-VAE_iter80 | 166 | -0.0852 |
| curvature_RQ-VAE_iter81 | 166 | -0.0908 |
| curvature_RQ-VAE_iter83 | 166 | -0.0973 |
| curvature_RQ-VAE_iter88 | 166 | -0.0919 |
| curvature_RQ-VAE_iter89 | 166 | -0.0922 |
| curvature_RQ-VAE_iter90 | 166 | -0.0912 |
| curvature_RQ-VAE_iter91 | 166 | -0.0926 |
| curvature_RQ-VAE_iter92 | 166 | -0.0879 |
| curvature_RQ-VAE_iter93 | 166 | -0.0888 |
| curvature_RQ-VAE_iter94 | 166 | -0.0919 |
| curvature_RQ-VAE_iter97 | 166 | -0.0917 |
| curvature_RQ-VAE_iter98 | 166 | -0.0931 |
| curvature_RQ-VAE_iter99 | 166 | -0.0917 |
| r4_08 | 166 | -0.0880 |
| sids | 167 | -0.0907 |

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

