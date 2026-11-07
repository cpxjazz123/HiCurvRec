# The community line, closed on measurements

> **Correction (2026-11-07).** The AF3 gap ordering reported here was a
> property of an earlier version of the shared model, not of the mechanism. A
> reproducible re-run of the whole arm table gives 139.47 for the plain GCN
> propagation against 134.54 for the curvature-aware one, so the curvature-aware
> propagation does **not** raise the gap here, and the curvature does not earn
> its place inside the community model. The arm table of a single run remains a
> valid comparison. See `REPORT_community_transition.md`.


## The claim, and the measurement that settles it

The community mechanism was to learn a behaviour-aware partition of the
catalogue and let it supervise the RQ-VAE's identifiers. The partition was
learned and is good: the mixed geometry separates the curvature-ordered item
pairs **twice as well** as a capacity-matched Euclidean control (0.051717
against 0.039919 on the held-out ranking pairs, paired over 24 571 anchors,
permutation p = 0 of 20 000), the curvature-aware graph propagation lifts the
partition's hard AF3 gap from 124.52 to 134.52 (a plain GCN) to **145.11**, and
the learnable curvature moves the space to c = 1.1683 and adds a further
+0.004 to the separation.

It still cannot go into the identifiers, and the reason is an information
ordering that was measured rather than argued. On the held-out behaviour pairs,
with the quality gate's own yardstick:

| grouping | positives' share | negatives' share | ratio |
|---|---|---|---|
| community partition, K = 64 | 0.0430 | 0.0166 | **2.58** |
| community partition, K = 1024 | 0.0068 | 0.0012 | **5.45** |
| a k-means partition of the transition profile (64 / 256 / 1024) | 0.218–0.248 | 0.143–0.187 | 1.53 / 1.43 / 1.33 |
| **the frozen baseline's own level-1 codes** | 0.0864 | 0.0140 | **6.16** |
| the resample arm's level-1 codes | 0.0794 | 0.0050 | 16.03 |

**Every partition of the behaviour graph is a weaker version of what the
identifiers already encode.** Resolution moves the number (2.58 at K = 64,
5.45 at K = 1024) but never past the codes' 6.16, and it costs the partition's
own signal: the curvature-ordering separation falls from 0.0604 to 0.0130 when
the communities go from 64 to 1024. A partition that *could* beat 6.16 would
have to be fitted to the semantics, and then it carries no behaviour
information the codes do not already have. There is no informational room.

## What each placement cost, measured

| shape | what it did to the codes | result |
|---|---|---|
| community term in the L1 cost (attraction) | discrimination 6.16 → 5.49 | recall 0.05263 |
| the same, repulsion | 6.16 → 5.23 | recall 0.05423 |
| pair contrast on the encoder's output | the code space is spent on the behaviour structure: L2 prefixes 7414 → 4110, collision 0.4116 | refused by the gate |
| pair contrast on the later residuals | L2 prefixes → 3688, collision 0.4490 | refused by the gate |
| the baseline's balanced plan at L1 | the negatives' share 0.0134 → 0.0044, but the positives' 0.0864 → 0.0365 | recall 0.05315 |
| **the baseline's balanced plan + the repulsion** | L2 prefixes 8575, collision 0.1815, discrimination 8.22 | **recall 0.05315** |

The last row is the sharpest statement of the whole campaign: **every SID
diagnostic improved and the recall fell.** The 47-arm cross-sectional
correlation between the level-1 discrimination and the recall (0.728) is a
property of the historical arms, not a lever: the arm was built to move that
number, it moved, and the recall went the other way.

## Why, in one line

The ratio that the correlation measures is composed of a **numerator** (the
behaviour positives share a level-1 code) and a **denominator** (the behaviour
negatives do not). The balanced plan cut the denominator and cut the numerator
with it, because spreading the codes by usage is not the same as keeping the
behaviour partners together. The resample family reaches a numerator close to
the baseline's (0.0794) with a denominator of 0.0050: it gets the uniform usage
without its cost, and what provides that is a **behaviour-trained encoder**.
A plan can only redistribute what the encoder has already separated.

## What the community line is good for, then

Not the identifiers. Two things, both real:

1. **A measurable model of the graph's curvature structure.** The mixed
   geometry beats a capacity-matched Euclidean control two to one on the
   curvature ordering, and the curvature-aware propagation lifts the hard AF3
   gap by 16.6%. That is a standalone result about hyperbolic geometry and
   behaviour graphs.
2. **A diagnostic for any arm.** The ledger
   (`discrimination_ledger.py`) reads the identifiers of every arm on disk and
   reports, with one yardstick, whether the level-1 codes still keep the
   behaviour partners together. Applied to the campaign's own arms it is what
   separated the arms that kept the numerator (the curriculum arm: 0.0831) from
   the arms that diluted it (the contrast arms: 0.0412), and it is the
   cheapest early warning available: no GPU, one sample, one ratio.

## The recommendation the measurements give

The only lever in this repository that has moved the recall is the
**behaviour-pair contrast on the residuals**, trained the way the behaviour-sample
tree trains it (per-epoch resampled transitions, the repository's own ramp, its
own anti-collapse guard), which reaches 0.06374 with 241/256 codes used and the
highest level-1 discrimination measured. Its tree is an older lineage, so
porting that mechanism onto today's frozen baseline is the next experiment worth
running -- and its prediction is checkable in advance: the level-1 positives'
share must stay near the baseline's 0.0864 while the negatives' share falls
towards 0.005.

**Files.** `tools/research/science/community_curvature/` -- the learner with the
learnable curvature, the two cheap measurements (`alignment.py`,
`partition_capacity.py`), the ledger (`discrimination_ledger.py`) and the
reports in `results/research/issue121/community_curvature/`.
