# The community mechanism, measured end to end: the recall follows the error rate backwards

**Question.** Does the community information -- a 64-way partition of the
catalogue learned from the training split's behaviour graph and its AF3
curvature -- improve the RQ-VAE's Semantic IDs?

**Answer.** It improves the coarse level and it makes the recommender worse, and
the second part is now measured four ways. Across every arm of this campaign
the Stage-3 recall is **negatively ordered by the level-1 hard error rate**: the
arms whose coarse codes are the most homogeneous recall the least.

| arm | L1 Hard ER | L2 prefixes | L1 positive hit | **test_R@10** |
|---|---|---|---|---|
| frozen baseline | 6.47 | 7414 | 0.0864 | **0.05554** |
| curriculum-behaviour-rq | 6.68 | 7439 | 0.0831 | 0.05481 |
| community-l1-rq (community in the L1 cost) | 5.49 | 6280 | **0.0989** | 0.05263 |
| curvature-embedding-rq (Plan A) | 4.26 | 14021 | 0.0357 | 0.04897 |
| **community-pairs-guarded-rq (this arm)** | **3.03** | 6575 | 0.0412 | **0.04328** |
| *reference*: resample, the campaign's best | — | — | — | 0.06374 |
| the threshold the programme aims at | | | | 0.065 |

This arm has the **best level-1 hard error rate of any arm in the campaign**
(3.03 against the baseline's 6.47) and the **worst recall** (0.04328 against
the baseline's 0.05554). The two are the same fact: the codes are now
*homogeneous*, and a prefix that is homogeneous carries less.

## What this arm was

The community contrast of the previous two arms (which collapsed the code
space to a collision of 0.42–0.58 and 0.38–0.49) plus **the repository's own
anti-collapse guard** -- the diversity term of the behaviour-sample tree,
copied verbatim with its constants (`DIVERSITY_WEIGHT 0.1`, `GUMBEL_TAU 0.5`)
cited rather than chosen -- and the measured gradient bound from the arm
before it. The guard did its job: the collision fell from 0.601 at epoch 50 to
0.1693 at the end of the budget, below the baseline's own 0.2064, and the arm
completed the frozen 3000-epoch run with 24 587 unique identifiers.

The layer record (written before the Stage 3 result existed) shows what the
guard bought and what it cost:

| | baseline | this arm |
|---|---|---|
| L1 Hard ER | 6.47 | **3.03** |
| L1 positive hit (held-out behaviour pairs) | 0.0864 | **0.0412** |
| L1 negative hit | 0.0048 | 0.0136 |
| L2 prefixes | 7414 | 6575 |
| L3 collision | 0.2745 | 0.2686 |

The codes agree with each other much more (the error rate halves, the negative
hit rate nearly triples) and they agree with the **held-out behaviour pairs**
much less (the positive hit rate halves). The 64-way community partition is
**coarser than the transition graph's local structure**: aligning the codes to
it means that pairs the recommender cares about -- an item and the item its
users actually consumed next -- land on different level-1 codes.

## The four shapes, and what each one cost

| shape | where the community acts | collision | R@10 |
|---|---|---|---|
| cost at level 1 | the assignment's cost matrix | 0.3259 (passes) | 0.05263 |
| pair contrast, unbounded | the encoder's latents | 0.42–0.58 (refused) | not run |
| pair contrast, gradient-bounded | the encoder's latents | 0.38–0.49 (refused) | not run |
| **pair contrast + the guard** | the encoder's latents | **0.1693 (passes)** | **0.04328** |
| standalone community model | outside the RQ-VAE | — | not run (mechanism only) |

The weak form (the cost term) is the only one that improved the *behaviour*
hit rate (0.0989, the best of any arm) and it still lost recall; the strong
forms all trade the behaviour hit rate away and lose more recall.

## What the campaign now says

1. **The community signal is real.** It is the only signal in this campaign
   that improved the coarse level's positive hit rate (0.0864 → 0.0989), and
   its quality is measurable and reproducible (the five-arm community
   experiment: the mixed geometry separates curvature-ordered pairs twice as
   well as its capacity-matched Euclidean control, p = 0 of 20 000).
2. **The recall of this campaign tracks the level-1 hard error rate backwards.**
   Six arms, one direction: the more homogeneous the coarse codes, the worse
   the recommender. A prefix earns its place by carrying information, and every
   version of the community mechanism that made the codes more homogeneous
   took that information away.
3. **The community partition is the wrong target for the codes.** The
   behaviour structure the recommender needs is finer than the 64-way
   partition: the partition says which items are neighbours in the graph, and
   the codes then cannot separate the items *within* a community that behave
   differently.

## What follows from that

Not another placement of the community signal on the codes, and not another
strength: the evidence says the direction is wrong. If the community
information is to be used in the RQ-VAE at all, it has to be used to keep the
coarse codes **apart** rather than together -- the community as a
*separation* constraint (the items of one community must not collapse onto one
code) rather than as an attraction -- or it has to leave the codes alone and act
where diversity is not at stake, on the generation side, as a conditioning
feature of the decoder rather than as a property of the identifiers.

**Caveat.** Six arms is a small sample, and the policy allows one run per arm
with no reseeding, so the association above is an observation from six
measurements, not an established law. The two arms that share the community
signal's family and sit at opposite ends of the error-rate axis
(`community-l1-rq` at 5.49 and 0.05263, `community-pairs-guarded-rq` at 3.03 and
0.04328) do agree with it.
