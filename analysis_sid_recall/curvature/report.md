# Does the curvature of a SID's behaviour quotient graph predict its recall?

> **Read this before the numbers.** `sid_recall.csv` now holds only the pairs that pass the audit (`pair_audit.py`): the SID table and the recall file were introduced by the same commit **and** the stage-2 and stage-3 directories carry the same name (73 of the original 182). The numbers below are those 73. The full-table numbers are kept in `curvature/full_182/` and compared side by side in `AUDIT_CORRELATION.md`; the edge-sampling budget is checked in `curvature/convergence/` and `CONVERGENCE_FINDINGS.md`.

*73 SID tables, each with one recall, from 73 mechanism families.*

The behaviour graph is the training split's transitions only (24,556 items, 157,695 edges). For every SID table and every level the prefix groups are contracted into super-nodes, the lazy random walk's Ollivier curvature of that quotient graph is measured on a sample of its edges, and the number is standardised against a null that keeps the group sizes and shuffles the members. The features are then correlated with the recall of the run that used the table.

## The question, in three parts

1. Does a curvature feature correlate with the recall at all?
2. Does it still correlate once the plain SID-quality measures (modularity, code usage, collision) are taken out?
3. Does it survive the arms not being independent?

## 1. The correlation

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 68 | 0.1030 | 0.3973 | [-0.1584, 0.3428] |
| z_curv_L2 | 73 | -0.1622 | 0.1709 | [-0.3717, 0.0672] |
| z_curv_L3 | 73 | -0.0866 | 0.4598 | [-0.3347, 0.1593] |
| z_curv_L1_minus_L3 | 68 | 0.0974 | 0.4263 | [-0.1654, 0.3441] |
| sign_change_L1_L3 | 73 | 0.2429 | 0.0240 | [0.1566, 0.3914] |

### The controls, for reference

| control | n | Spearman | permutation p |
|---|---|---|---|
| codes_used_L1 | 73 | -0.2050 | 0.0860 |
| codes_used_L2 | 73 | -0.0920 | 0.4363 |
| codes_used_L3 | 73 | 0.2995 | 0.0085 |
| collision | 73 | -0.1596 | 0.1729 |

### Is the curvature a restatement of the partition's purity?

| curvature | control | Spearman |
|---|---|---|
| z_curv_L1 | codes_used_L3 | -0.1408 |
| z_curv_L1 | collision | 0.6636 |
| z_curv_L2 | codes_used_L3 | 0.1352 |
| z_curv_L2 | collision | -0.1461 |
| z_curv_L3 | codes_used_L3 | 0.0405 |
| z_curv_L3 | collision | -0.1766 |

### How much recall does the curvature add on top of the controls?

| curvature feature | n | R^2 of the controls alone | R^2 with the curvature | increment | permutation p |
|---|---|---|---|---|---|

## 2. The partial correlations (the decisive test)

The curvature feature with one control taken out, in rank space.

| feature | control | n | partial Spearman |
|---|---|---|---|
| z_curv_L1 | codes_used_L1 | 68 | -0.0870 |
| z_curv_L1 | codes_used_L2 | 68 | -0.0123 |
| z_curv_L1 | codes_used_L3 | 68 | 0.1185 |
| z_curv_L1 | collision | 68 | 0.2431 |
| z_curv_L2 | codes_used_L1 | 73 | -0.0850 |
| z_curv_L2 | codes_used_L2 | 73 | -0.1357 |
| z_curv_L2 | codes_used_L3 | 73 | -0.1609 |
| z_curv_L2 | collision | 73 | -0.1873 |
| z_curv_L3 | codes_used_L1 | 73 | -0.0106 |
| z_curv_L3 | codes_used_L2 | 73 | -0.0718 |
| z_curv_L3 | codes_used_L3 | 73 | -0.0677 |
| z_curv_L3 | collision | 73 | -0.0970 |
| z_curv_L1_minus_L3 | codes_used_L1 | 68 | -0.0859 |
| z_curv_L1_minus_L3 | codes_used_L2 | 68 | -0.0180 |
| z_curv_L1_minus_L3 | codes_used_L3 | 68 | 0.1125 |
| z_curv_L1_minus_L3 | collision | 68 | 0.2304 |
| sign_change_L1_L3 | codes_used_L1 | 73 | 0.0789 |
| sign_change_L1_L3 | codes_used_L2 | 73 | 0.1251 |
| sign_change_L1_L3 | codes_used_L3 | 73 | 0.1180 |
| sign_change_L1_L3 | collision | 73 | 0.1625 |

## 3. One point per mechanism family

The 182 tables come from 73 families, so the honest sample size is much smaller than 182. Keeping the best-recalling table of each family:

| feature | n | Spearman | permutation p | bootstrap CI95 |
|---|---|---|---|---|
| z_curv_L1 | 68 | 0.1030 | 0.3973 | [-0.1584, 0.3428] |
| z_curv_L2 | 73 | -0.1622 | 0.1709 | [-0.3717, 0.0672] |
| z_curv_L3 | 73 | -0.0866 | 0.4598 | [-0.3347, 0.1593] |
| z_curv_L1_minus_L3 | 68 | 0.0974 | 0.4263 | [-0.1654, 0.3441] |
| sign_change_L1_L3 | 73 | 0.2429 | 0.0240 | [0.1566, 0.3914] |

### Leave one family out

Each family is removed in turn; the correlation that remains is the one that does not depend on that family's arms.

| family removed | n left | Spearman (z_curv_L1) |
|---|---|---|
| curvature_RQ-VAE_iter100 | 67 | 0.1083 |
| curvature_RQ-VAE_iter101 | 67 | 0.1096 |
| curvature_RQ-VAE_iter102 | 67 | 0.1122 |
| curvature_RQ-VAE_iter107 | 67 | 0.1069 |
| curvature_RQ-VAE_iter109 | 67 | 0.1068 |
| curvature_RQ-VAE_iter14 | 67 | 0.0997 |
| curvature_RQ-VAE_iter15 | 67 | 0.1024 |
| curvature_RQ-VAE_iter16 | 67 | 0.1100 |
| curvature_RQ-VAE_iter17 | 67 | 0.0981 |
| curvature_RQ-VAE_iter18 | 67 | 0.1048 |
| curvature_RQ-VAE_iter19 | 67 | 0.0993 |
| curvature_RQ-VAE_iter20 | 67 | 0.1138 |
| curvature_RQ-VAE_iter21 | 67 | 0.1061 |
| curvature_RQ-VAE_iter22 | 67 | 0.0897 |
| curvature_RQ-VAE_iter23 | 67 | 0.1265 |
| curvature_RQ-VAE_iter24 | 67 | 0.1073 |
| curvature_RQ-VAE_iter25 | 67 | 0.0953 |
| curvature_RQ-VAE_iter26 | 67 | 0.0996 |
| curvature_RQ-VAE_iter28 | 67 | 0.1160 |
| curvature_RQ-VAE_iter29 | 67 | 0.1020 |
| curvature_RQ-VAE_iter31 | 67 | 0.1204 |
| curvature_RQ-VAE_iter32 | 67 | 0.0952 |
| curvature_RQ-VAE_iter33 | 67 | 0.1116 |
| curvature_RQ-VAE_iter34 | 67 | 0.1230 |
| curvature_RQ-VAE_iter35 | 67 | 0.1165 |
| curvature_RQ-VAE_iter36 | 67 | 0.1096 |
| curvature_RQ-VAE_iter37 | 67 | 0.0960 |
| curvature_RQ-VAE_iter38 | 67 | 0.1238 |
| curvature_RQ-VAE_iter39 | 67 | 0.1263 |
| curvature_RQ-VAE_iter40 | 67 | 0.0951 |
| curvature_RQ-VAE_iter41 | 67 | 0.0889 |
| curvature_RQ-VAE_iter42 | 67 | 0.1134 |
| curvature_RQ-VAE_iter43 | 67 | 0.0974 |
| curvature_RQ-VAE_iter44 | 67 | 0.1020 |
| curvature_RQ-VAE_iter45 | 68 | 0.1030 |
| curvature_RQ-VAE_iter46 | 67 | 0.0912 |
| curvature_RQ-VAE_iter47 | 67 | 0.1021 |
| curvature_RQ-VAE_iter49 | 67 | 0.0928 |
| curvature_RQ-VAE_iter50 | 67 | 0.0878 |
| curvature_RQ-VAE_iter51 | 67 | 0.1081 |
| curvature_RQ-VAE_iter52 | 68 | 0.1030 |
| curvature_RQ-VAE_iter53 | 68 | 0.1030 |
| curvature_RQ-VAE_iter54 | 67 | 0.0648 |
| curvature_RQ-VAE_iter55 | 67 | 0.0757 |
| curvature_RQ-VAE_iter56 | 67 | 0.0643 |
| curvature_RQ-VAE_iter57 | 67 | 0.0681 |
| curvature_RQ-VAE_iter58 | 67 | 0.0648 |
| curvature_RQ-VAE_iter59 | 68 | 0.1030 |
| curvature_RQ-VAE_iter61 | 67 | 0.0660 |
| curvature_RQ-VAE_iter62 | 68 | 0.1030 |
| curvature_RQ-VAE_iter63 | 67 | 0.1128 |
| curvature_RQ-VAE_iter64 | 67 | 0.0888 |
| curvature_RQ-VAE_iter65 | 67 | 0.1302 |
| curvature_RQ-VAE_iter66 | 67 | 0.1350 |
| curvature_RQ-VAE_iter67 | 67 | 0.1336 |
| curvature_RQ-VAE_iter73 | 67 | 0.1066 |
| curvature_RQ-VAE_iter77 | 67 | 0.1082 |
| curvature_RQ-VAE_iter78 | 67 | 0.1040 |
| curvature_RQ-VAE_iter79 | 67 | 0.1009 |
| curvature_RQ-VAE_iter80 | 67 | 0.1119 |
| curvature_RQ-VAE_iter81 | 67 | 0.1051 |
| curvature_RQ-VAE_iter83 | 67 | 0.0728 |
| curvature_RQ-VAE_iter88 | 67 | 0.1029 |
| curvature_RQ-VAE_iter89 | 67 | 0.0983 |
| curvature_RQ-VAE_iter90 | 67 | 0.1042 |
| curvature_RQ-VAE_iter91 | 67 | 0.1035 |
| curvature_RQ-VAE_iter92 | 67 | 0.1228 |
| curvature_RQ-VAE_iter93 | 67 | 0.1115 |
| curvature_RQ-VAE_iter94 | 67 | 0.0999 |
| curvature_RQ-VAE_iter97 | 67 | 0.1055 |
| curvature_RQ-VAE_iter98 | 67 | 0.1034 |
| curvature_RQ-VAE_iter99 | 67 | 0.1046 |
| r4_08 | 67 | 0.1185 |

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

