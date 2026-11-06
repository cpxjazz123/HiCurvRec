# The community-guided level-1 cost: the coarse level improves, the recall does not

**Question.** Use the community information the standalone experiment learned —
a soft community per item, derived from the training split's behaviour graph
and its AF3 curvature — as supervision for the RQ-VAE's level-1 SID training,
and see what it does.

**Answer.** It improves exactly the level it was injected into and it does not
reach the recommender. The level-1 Hard ER falls from 6.47 to **5.49** and the
share of held-out positives that land on the same level-1 code rises from
0.0864 to **0.0989** (the highest of every arm so far), with the level-1
codebook fully used (256/256 codes, the largest holding 3.50% of the items
against the baseline's 3.49%). The fine levels pay for it (level-2 prefixes
7414 → 6280, three-level collision 0.2745 → 0.3259, still under the owner's
35% gate), and the final **`test_recall@10` is 0.05263 against the frozen
baseline's 0.05554**: the SID-level gain does not survive the hand-off to the
recommender, and it is 0.013 below the 0.065 the programme is aiming at.

## The mechanism

* each level-1 code carries a **community profile**: the mean of the tangent
  vectors of the items it received, accumulated over the epoch and summed
  across the four ranks, then mapped back into the ball with the contract's
  origin exponential;
* the level-1 cost gains the **community term**: the contract's Poincaré
  distance between the item's community point and the code's profile, plus the
  free branch's Euclidean distance in the same way, standardised per item;
* the **assignment rule is the baseline's** and is untouched — the term enters
  the cost matrix the rule consumes;
* the term is **detached** (the community representation is frozen, the profile
  is a statistic of past assignments), so it moves the assignment and not the
  gradient of the losses;
* the **weight is measured, not chosen**: the term is standardised per item so
  its spread is one, and the weight is the baseline cost's own per-item spread,
  read once.

The community representation itself is the previous step's output (the
curvature-aware graph convolution arm, the best hard AF3 gap of the five
community arms) exported as a frozen artifact, so this arm inherits that
mechanism rather than re-learning it.

## Admission

The live admission is the framework's machine check, and all ten checks pass:
the frozen constants and the launcher, a real one-epoch entry run that exports
24587 four-token identifiers, the four-rank probe, the assignment-rule check
(the balanced assignment stays the rule and the mechanism's score enters it as
the cost), the geometry contract, the encoder gradient (8 of 8 tensors), and
the reference agreement rebuilt from the SymPy statements (values 2.2e-16,
gradients 8.9e-16, gradcheck on both sides, coupling surface moves the
computation by 2.05).

The checks found two real defects, both fixed before any GPU time:

1. the declared surface computed `((-x ⊕ u)) ⊕ u` instead of the contract's
   `(-x) ⊕ u` — the SymPy statements were rewritten;
2. the supervisor was constructed with four positional arguments against a
   signature whose fourth parameter is the device — the arguments are named now.

## Stage 2 (3000 epochs, four cards, frozen budget, one run)

Measured with `record_single_sid`, recorded **before** any Stage 3 result
existed (`stage3_results_used: false`).

| arm | L1 Hard ER | L2 prefixes | L3 triples | L3 collision | L1 positive hit | L1 codes used |
|---|---|---|---|---|---|---|
| frozen baseline | 6.47 | 7414 | 20223 | 0.2745 | 0.0864 | 247/256, max 3.49% |
| curvature-stratum-sinkhorn-v1 | 6.36 | 955 | 9510 | 0.7713 | 0.0290 | — |
| curvature-embedding-rq (Plan A) | 4.26 | 14021 | 24333 | 0.0201 | 0.0357 | — |
| curriculum-behaviour-rq | 6.68 | 7439 | 20516 | 0.2600 | 0.0831 | 249/256, max 2.65% |
| **community-l1-rq** | **5.49** | 6280 | 19244 | 0.3259 | **0.0989** | **256/256, max 3.50%** |

There is no collapse, which was this mechanism's real risk: the community term
could have made a few codes attractive for a whole community and folded the
catalogue onto them. All 256 level-1 codes are used and the largest holds
3.50% of the items, level with the baseline. The safeguards did that work: the
balanced assignment rule is the baseline's, the profile lags one epoch, and
the weight is the cost's own spread rather than a free constant.

The coarse level and the fine levels move in opposite directions, which is the
trade the earlier rounds predicted: pushing behaviour into the coarse
assignment buys coarse-level separability and costs fine-level resolution.

## Stage 3 (150 epochs, four cards, frozen protocol, one run)

| arm | test_R@10 | test_R@5 |
|---|---|---|
| frozen baseline | **0.05554** | 0.03743 |
| curriculum-behaviour-rq | 0.05481 | 0.03755 |
| **community-l1-rq** | **0.05263** | 0.03548 |
| curvature-embedding-rq (Plan A) | 0.04897 | 0.03209 |
| threshold the programme is aiming at | 0.065 | — |
| resample reference (historical best) | 0.06374 | — |

The recall is **0.0029 below the baseline** and 0.0124 below the threshold. Per
the experiment policy this is a negative result: one run, no reseed, no tuning,
recorded as it stands.

## What this says about the direction

The community information is real — it is the only signal in this campaign that
has ever improved the coarse level's positive-hit rate (0.0864 → 0.0989) — and
it is measurable at the SID level. It does not translate into recall.

Two readings are consistent with the numbers and this experiment cannot
separate them:

1. **The coarse level is not the bottleneck.** The recall is set by what the
   generative model can do with the whole prefix; making level 1 more
   behaviour-consistent while losing level-2 prefixes (7414 → 6280) may simply
   move resolution from a place that mattered to a place that did not.
2. **The hand-off is lossy.** The community term supervises the *cost* of an
   assignment, not the decoder's conditional distribution; a sharper level-1
   partition does not by itself make the next item more predictable from the
   prefix.

Either way, the next thing to try is not another way to sharpen a level. The
evidence across five arms — sinkhorn collapse, Plan A's fine-level blow-up, the
curriculum's baseline-level plateau, node-to-cluster, and this one — is that
SID-level diagnostics and Stage-3 recall have not moved together in any arm so
far. The arms that improved a SID diagnostic did not improve the recall, and
the one arm that came closest to the threshold (resample, 0.06374) did so by
changing *which pairs the SID construction sees per epoch*, not by changing a
metric.

## Files

* `stage2_RQ-VAE/TIGER_community-l1-rq/` — the candidate tree (mechanism,
  tests, reference contract)
* `results/stage2_RQ-VAE/TIGER_community-l1-rq/` — `item_sids.json`, the
  layered record `stage2_metrics/9e00e58d….json`, the metrics stream
* `results/stage3_T5Train/TIGER_community-l1-rq/logs/Amazon_2023_Instruments/Nov-07-2026_01-57-10/`
  — `test_final.json`, the metrics stream, the four test shards
* `tools/research/science/community_curvature/export_representation.py` — the
  exporter that turned the community model into this arm's frozen input
