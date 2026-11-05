# Does the behaviour curvature order the SID's prefix sharing?

The design under test asks the SID hierarchy to respect a relative relation
about behaviour: two items whose behaviour neighbourhoods overlap heavily (a
local cohesion: many shared neighbours, many triangles) should share a COARSER
SID prefix than a pair that is connected but whose surroundings lean to
different local structures (a local bridge).

Measured with `curvature_prefix_probe.py` on the frozen validation pairs
(52,855 anchors x 1 positive + 10 negatives, seed 42), the behaviour graph
built from the **training split only**, and the overlap statistic |N(A) ∩ N(B)|
- the triangles through the edge, which is the design's own definition of a
cohesion edge.

## 1. The signal exists, and it is heavy-tailed

| quantity | value |
|---|---|
| median triangles, anchor–positive | **6** |
| median triangles, anchor–negative | **0** |
| positives with at least one triangle | **54.8 %** |
| negatives with at least one triangle | **40.3 %** |
| anchors whose positive beats the mean of its negatives | 36.7 % |

The median separates the true next item from the negatives cleanly; the mean
does not, because the distribution is extremely heavy-tailed: the **mean** overlap
of an L1-sharing pair is 626–969 triangles against a median of 6–11. A mechanism
that puts the raw overlap count into a loss would be driven by a handful of hub
items, so the statistic has to be **bounded or ranked** before it can be used.

## 2. The headroom: the SID does not follow the curvature order

| arm | L1-sharing pairs: median triangles | not sharing L1 | does the SID group cohesive pairs? |
|---|---|---|---|
| `TIGER_RQ-VAE` (baseline) | 8 | 6 | weakly yes |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` (0.0637) | **11** | 6 | **yes, more strongly** |
| `TIGER_pmi-geomargin-l1-rq` (closed, collapsed) | 8 | 5 | barely (and 38.6 % of all pairs share L1) |

The share of positive pairs that share the first level, by overlap tertile:

| arm | low | middle | high |
|---|---|---|---|
| baseline | 8.7 % | 6.9 % | 8.9 % (not monotone) |
| resample | 7.3 % | 7.4 % | 8.3 % (weakly monotone) |
| pmi-geomargin | 38.6 % | 38.8 % | 41.6 % (monotone, level inflated by the collapse) |

And the paired test: in 2,410 anchors a **negative** has more shared neighbours
than the true next item, and in **94–95 %** of those the negative shares the
same or a coarser prefix with the anchor (77 % for the collapsed arm).

⇒ The SID today mostly puts **bridges** together with the anchor rather than
the cohesive partners, even though the behaviour-supervised arm (resample) is
already better at it (11 vs 6 triangles). The relation the design asks for is
**not** what the SID encodes today, so there is real headroom - and the arm that
already spends its budget on behaviour supervision is the closest to the target.

## 3. What the numbers require of the mechanism

1. **The statistic must be bounded or ranked.** Raw triangle counts have a mean
   of ~700 against a median of ~6; a loss on the raw count is a loss on hubs.
2. **The relation is a ranking, not a threshold.** The design asks for a
   relative order (A shares with B more than with C), and the median, not the
   mean, is what separates the classes.
3. **The deeper levels barely participate.** The pairs that share L1+L2 are
   ~0.3 % and L3 ~0.1 %; the mechanism's effect will be read at L1.
4. **The Sinkhorn balance must stay** (CLAUDE.md 19h). The curvature enters the
   cost matrix or as an extra term; it never replaces the balanced assignment -
   the collapsed arm of round_20261105_144909_130 is what happens when it does.
5. **A matched Euclidean control is mandatory.** The repository's own audit
   (issue #128) withdrew the curvature-recall association as an artefact of a
   one-direction graph, and the metric rounds found the hyperbolic increment over
   a near-flat control at +0.021 [-0.031, +0.076] (P = 0.20). The geometry-off
   switch is that control, and it has to be reported with the run.
6. **The behaviour side is train-only.** The graph, the curvature and the
   cohorts come from the training split; no validation or test interaction is
   read (the frozen validation pairs are used here only to MEASURE, never to
   train).

## Reproduce

    python3 analysis_sid_recall/curvature_prefix_probe.py --anchors 4000 \
        --arms TIGER_RQ-VAE TIGER_BEHAVIOR_SAMPLE_RQ-VAE TIGER_pmi-geomargin-l1-rq

## 4. The level-1 partition is largely blind to the cohesion rank

The mechanism of round_20261106_020541_888 puts a bounded cohesion rank
`rho_i` (the mid-rank of the item's local triangle cohesion) into the level-1
cost with a learned per-code stratum `s_k = sigmoid(a_k)`, so it assumes the
codes can be ORDERED by that rank. `cohesion_rank_probe.py` measures how much of
the rank's variance the level-1 partition already explains (`eta^2`):

| arm | level-1 codes | eta^2 | lowest code's median rank | highest code's median rank |
|---|---|---|---|---|
| `TIGER_RQ-VAE` (baseline) | 247 | **0.039** | 0.170 | 0.882 |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` (0.0637) | 241 | **0.132** | 0.170 | 0.880 |
| `TIGER_pmi-geomargin-l1-rq` (collapsed) | 42 | 0.012 | 0.170 | 0.958 |

The codes DO span the rank range (the lowest code sits at rank 0.17, the highest
near 0.88-0.96 in every arm), but the partition explains only 1-13 % of the
rank's variance: it is largely blind to the cohesion rank. And the arm that
explains the most - the behaviour-supervised `resample`, the best recall in the
repository at 0.0637 - is the one that also groups the cohesive pairs best (11
vs 6 triangles). The stratification the mechanism adds is therefore both absent
today and aligned with the one arm that works.

Reproduce:

    python3 analysis_sid_recall/cohesion_rank_probe.py
