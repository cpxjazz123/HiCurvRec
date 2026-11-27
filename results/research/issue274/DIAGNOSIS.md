# Issue #274 — diagnosis (round 1)

## 1. The phenomenon (what failed, in numbers)

Every arm of the #269–#273 family that **regresses a curvature value onto the representation**
shrinks the representation's dispersion and loses SID capacity, while the curvature-fidelity metrics
stay poor. The frozen records:

| arm | curvature supervision | variance ratio | slope | Spearman | sign | L1 hard ER | L2 prefixes | rank | collision | recon |
|---|---|---|---|---|---|---|---|---|---|---|
| TIGER (frozen baseline) | none | 28.7 | 0.041 | 0.0159 | 0.4315 | 7.31 | 7,374 | 11.5 | 0.0656 | — |
| #269 calibrated kappa | pointwise Huber on `kappa_t` | 0.42 | 0.019 | −0.0701 | 0.3483 | 2.34 | 6,848 | 2.77 | 0.1203 | ≤ 8.589e-05 |
| #272 round 3 (codebook budget) | pointwise, codebook side at 1 % | 1.32 | 0.111 | 0.0798 | 0.4049 | 2.46 | 7,463 | 2.21 | 0.1337 | 7.709e-05 |
| #273 same-centre difference | pointwise + same-centre difference | — | — | NOT_RUN | NOT_RUN | NOT_RUN | 5,966 | NOT_RUN | 0.1419 | 9.793e-05 |

Evidence paths: `results/research/issue272/VERDICT.md` (the round-3 table and the arm matrix),
`results/research/issue273/VERDICT.md` and `results/research/issue273/round1_sid_structure.json`
(the #273's measured criteria), `results/stage2_RQ-VAE/TIGER_RQ-VAE/stage2_metrics/` (the TIGER's
frozen layer metrics).

The two observations that define the phenomenon:
1. the **dispersion collapse** (the variance ratio 28.7 → 0.42, the slope 0.041 → 0.019): the
   prediction's spread collapses towards the mean;
2. the **capacity loss** (the L2 7,374 → 6,848 → 7,463 → 5,966, the rank 11.5 → 2.77 → 2.21): the
   discrete code space loses prefixes and effective rank.
Both happen while the curvature-fidelity metrics stay at or below the TIGER's, i.e. the value
supervision does not even buy its own metric.

## 2. The candidate root causes and the competing explanations

| id | candidate cause | competing explanation | status |
|---|---|---|---|
| RC1 | the curvature **value** is the wrong objective: a graph-theoretic average is regressed onto the SID's geometry, and the regression target's scale (a mean over a ~24k-node component) is nearly constant, so the loss mostly pushes a bias and the dispersion collapses | the curvature target is informative and only the *weight* was wrong (the #272 budget hypothesis) | RC1 and the "weight" explanation are both consistent with the table; the #272 round-3 showed that even a 1 % codebook budget leaves the L2 at 7,463 and the rank at 2.21, so **the weight alone does not restore the capacity** |
| RC2 | the supervision reaches the **encoder latent** through the STE and competes with the reconstruction (the #272 round-2's 293× jump collapsed the assignment outright) | the loss is harmless on the latent and the collapse was a scheduling artefact | the #272 round-2 and round-3 evidence supports the competition: the collapse appeared exactly when the local term's share of the codebook gradient was large and disappeared when it was budgeted |
| RC3 | the supervision is **targeted at the representation** at all; what the SID needs is a *relative* relation, and the value regression spends the capacity on the wrong axis | the relative relation carries no extra information beyond the graph adjacency, so no curvature-based supervision can help (the null) | the null is the live competitor: the #274 round-1 offline check (below) shows the curvature's within-class ordering **does** track an independent tightness signal, so the null is weakened but not excluded |

## 3. The low-cost verification of the root cause (no training)

The issue's section 1 requires the real graph's curvature and the closed/open relation to be
re-verified before any mechanism design. The measurements:

* `results/research/issue274/triplet_survey.json`: the full table has 28,391,948 rows, all valid,
  13 directed structure classes; the class separates the closed triangles (`bc_dist_sq == 1`) from
  the open wedges (`bc_dist_sq == 4`) exactly; the per-class curvature medians are +0.29…+0.49 for
  the closed classes and −0.09…−1.04 for the open ones.
* `results/research/issue274/class_direction_map.json`: the centre-edge direction patterns of each
  class, derived from the visible training graph (2-3 patterns per class; the b–c edge is absent in
  every open class and present in every closed class).
* `results/research/issue274/pairs_manifest.json`: the full scan of the legal triplets produced
  79,185 comparability groups and 26,583 selected groups (21,221 training, 5,362 hold-out) over
  10,782 centres; the exclusions (39,905 without a closed candidate, 406 without an open candidate,
  12,288 below the minimum support, 3 tied at the extremum) are recorded.
* `results/research/issue274/offline_diagnostics.json`:
  * **the curvature is not redundant with the closed/open label**: inside one centre and one
    structure class, the curvature's ordering tracks the outer pair's common-neighbour count with a
    mean Spearman of **+0.874** (the median +0.937, 98.9 % of the 2,482 matched groups positive) —
    an independent higher-order tightness signal;
  * **the raw curvature is not comparable across the two populations**: the selected closed
    representative's curvature has the median +0.697 and the selected open representative's +3.802
    (the open wedges carry the larger `g(b,c)^2/4` term), so a cross-population curvature
    comparison would invert the relation (only 0.9 % of the groups have the closed above the
    open). The frozen rule therefore selects the extremal representative *inside* each population;
  * **the residual correlation**: 28.4 % of the training groups have an item shared between the
    positive and the negative pair; 11,122 distinct items are used (the mean participation 7.63,
    the p99 88.8), which is why the hold-out is split by **centre** and the residual correlation is
    reported rather than assumed absent.

## 4. What the round-1 mechanism does about the cause

`L_geo` never regresses a curvature value: the curvature only chooses which closed triangle and
which open wedge form a pair (offline, frozen), and the loss ranks two *pair distances*. The loss
acts on the final hard `q = sum_l c_l`, so its gradient reaches the codebooks and **not** the encoder
latent (the RQ selection is an argmin, so no gradient path exists). This removes RC1's value
regression and RC2's competition with the reconstruction, and it tests RC3 by construction: if the
relative relation carries no usable signal, the hold-out ranking will not move (H274-4).

## 5. The still-uncertain parts

* H274-6 stays INCONCLUSIVE: the within-class correlation shows the curvature carries extra
  information, but the missing measurement is whether the **selected extremal** representative is
  closer to the centre's other items than a random representative of the same class. That
  comparison is cheap and will be added in the next round's diagnostics.
* The #269's variance ratio / slope are the degradation signature; this round's diagnostics must
  report the candidate's variance ratio and slope to show whether the code-geometry-only
  supervision avoids that collapse.
* The hold-out's residual correlation (28.4 % of the groups share an item) is not eliminated, only
  measured; the acceptance compares the candidate and the TIGER on the *same* pairs, so the leakage
  affects both arms equally.

## 6. The six admission questions (before the formal training)

1. **What exactly does this round solve?** The representation degradation and the capacity loss of
   the value-supervised family: the round replaces a curvature-value regression with a relative
   two-pair ranking whose gradient touches only the discrete code geometry, and it measures the
   result on the hold-out ranking and on the SID capacity.
2. **What evidence says this is the cause of the previous failure?** The frozen arm matrix above:
   every value-supervised arm shows the dispersion collapse (the variance ratio 28.7 → 0.42) and the
   capacity loss (the L2 7,374 → 6,848/7,463/5,966) together, and the one arm that changed the
   *budget* rather than the *target* (#272 round 3) recovered the ordering but not the capacity.
3. **Why can the new mechanism act on that cause?** The curvature no longer enters the loss at all
   (so there is no value regression to shrink the dispersion), and the gradient path of a hard-q pair
   distance reaches only the codebooks (so the loss cannot compete with the reconstruction on the
   encoder latent).
4. **What is substantively different from the failed mechanisms?** The failed family regressed a
   per-triplet scalar (pointwise `kappa_t`, or its within-centre difference) onto the latent; this
   round supervises a **difference of two pair distances** on the discrete code geometry, with the
   curvature demoted to an offline selection key. The supervision's target type changes from a
   scalar to a comparison, and the parameter set it touches changes from the latent to the codebooks.
5. **Which metrics decide whether the problem is solved?** C1 (the hold-out ranking accuracy
   > 0.538605) and C2 (the margin satisfaction > 0.081499) against the frozen TIGER, plus the
   non-regression criteria C3–C10 (the L2 ≥ 10,918, the rank ≥ 8, the reconstruction ≤ 8.589273e-05,
   the J ≤ 0.3383359, the 1-hop/2-hop tolerances, the prefix discrimination and the behavioural
   neighbourhood fidelity). The thresholds are frozen in `baselines.json`.
6. **If this round still fails, what new knowledge does it produce?** The failure separates the
   competing explanations: (a) the ranking moved but the capacity regressed → the margin/λ budget
   must be re-derived against the reconstruction's gradient (the issue allows changing the margin,
   λ, the budget and the schedule); (b) nothing moved → the curvature-selected pairs carry no usable
   signal for the SID, and the *matching rule* (not the mechanism) becomes the suspect; (c) the
   ranking improved and the capacity held but the curvature fidelity fell → the relative ordering
   and the curvature value are genuinely different objectives, which the issue allows us to report
   (it asks explicitly whether the distance improved while the curvature fidelity dropped).
