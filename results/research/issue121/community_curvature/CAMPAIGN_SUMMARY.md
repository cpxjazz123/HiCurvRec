# The community mechanism, campaign summary: five shapes, none of them the baseline

## The table

| arm | L1 Hard ER | L2 prefixes | L3 collision | L1 positive hit | **test_R@10** |
|---|---|---|---|---|---|
| curriculum-behaviour-rq | 6.68 | 7439 | 0.2600 | 0.0831 | 0.05481 |
| **TIGER_RQ-VAE (frozen baseline)** | 6.47 | 7414 | 0.2745 | 0.0864 | **0.05554** |
| community-l1-rq (attraction, cost term) | 5.49 | 6280 | 0.3259 | **0.0989** | 0.05263 |
| community-repulse-l1-rq (repulsion, cost term) | 5.23 | 6318 | 0.3340 | **0.0996** | 0.05423 |
| curvature-embedding-rq (Plan A) | 4.26 | 14021 | 0.0201 | 0.0357 | 0.04897 |
| community-pairs-guarded-rq (contrast + guard) | 3.03 | 6575 | 0.2686 | 0.0412 | 0.04328 |
| *reference*: resample (the campaign's best) | 16.39 | 16145 | — | — | **0.06374** |
| the threshold the programme aims at | | | | | 0.065 |

Six arms of this campaign, one outcome: **no arm that used the community
information beat the frozen baseline's recall**, and the gap is 0.0013 to 0.0123
below it. The two community arms that produced the campaign's best *coarse*
numbers — a positive hit rate of 0.0989 and 0.0996 against the baseline's
0.0864 — produced recalls of 0.05263 and 0.05423.

## The five shapes, and what each one cost

| shape | where the community acts | collision | R@10 |
|---|---|---|---|
| cost at level 1, attraction | the assignment's cost matrix | 0.3259 (passes) | 0.05263 |
| cost at level 1, repulsion | the same cost matrix, opposite sign | 0.3340 (passes) | 0.05423 |
| pair contrast on the latents, unbounded | the encoder's latents | 0.42–0.58 (refused) | not run |
| pair contrast, gradient-bounded | the encoder's latents | 0.38–0.49 (refused) | not run |
| pair contrast + the repository's guard | the encoder's latents | 0.1693 (passes) | 0.04328 |

Plus the standalone community model, which never touched the RQ-VAE: the mixed
geometry separates the curvature-ordered pairs twice as well as its
capacity-matched Euclidean control on held-out interactions (p = 0 of 20 000),
so the community partition itself is real and measurable. It simply does not
pay anywhere in the identifier.

## What the five shapes establish

1. **The community signal is real at the coarse level and useless for the
   recall.** Every version of it that reached the codes improved the coarse
   level's behaviour hit rate (0.0864 → 0.0989 / 0.0996) and none of them
   improved the recall. The recall follows the behaviour structure that the
   *fine* levels carry, not the coarse level's score.
2. **A grouping force on the codes costs prefix diversity.** The contrast
   shapes collapsed the code space (collision 0.42–0.58) until the
   repository's own guard was added; with it the collision fell to 0.1693 and
   the recall was still the worst of the campaign (0.04328).
3. **The direction is wrong, and the extremes show it.** Ordering the arms by
   the level-1 hard error rate — the arms whose coarse codes discriminate
   behaviour pairs best — the recall falls: 3.03 → 0.04328, 4.26 → 0.04897,
   and the arm with the *least* discriminative coarse codes in the campaign
   (resample, 16.39, with 16 145 level-2 prefixes) has the best recall
   (0.06374). In the middle of the range (5.2–6.7) the differences are
   0.001–0.003 and the ordering is not monotone, so the middle of that table
   is noise rather than signal; the extremes are the informative part.

## What the campaign's one near-threshold arm did differently

Resample reached 0.06374 without touching any of these quantities. It has the
**least discriminative coarse level** of everything measured (16.39 against the
baseline's 6.47), the **most level-2 prefixes** (16 145 against 7 414), and it
changed *which pairs the encoder was contrasted on every epoch* rather than
any metric. The prescription that falls out of the table is therefore not a
better signal but a different shape: **a coarse first level that does not try
to predict the behaviour, with the discrimination pushed into the fine
levels**.

## What is closed and what is not

Closed, on evidence: the community partition as a force on the identifiers, in
every shape — attraction, repulsion, contrast, contrast with a guard. Five
arms, five measurements, one outcome, and the two extremes of the campaign's
own table say the direction is wrong.

Not closed: the community information *outside* the identifiers. It is
measurable, it is better than a capacity-matched Euclidean control at the
thing it was built for, and the two places where it can act without touching
prefix diversity are the generation side (a community prior on the decoder's
candidates, which changes no SID) and the sampling side (which pairs the
encoder is contrasted on, the lever that demonstrably moved the recall once).

**Caveat.** One run per arm, no reseeding, so every difference below about
0.003 between two arms of the same table is inside the noise the policy
forbids measuring. The findings that survive that limit are the extremes: the
best-ER arm is the worst-recall arm, the worst-ER arm is the best-recall arm,
and no arm beat the baseline.
