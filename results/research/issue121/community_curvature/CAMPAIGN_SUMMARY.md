# The community mechanism, campaign summary: five shapes, none of them the baseline

## The table

| arm | L1 Hard ER (= pos/neg) | L2 prefixes | L3 collision | **test_R@10** |
|---|---|---|---|---|
| *reference*: resample (the campaign's best) | **16.39** | 16145 | — | **0.06374** |
| curriculum-behaviour-rq | 6.68 | 7439 | 0.2600 | 0.05481 |
| **TIGER_RQ-VAE (frozen baseline)** | 6.47 | 7414 | 0.2745 | **0.05554** |
| community-l1-rq (attraction, cost term) | 5.49 | 6280 | 0.3259 | 0.05263 |
| community-repulse-l1-rq (repulsion, cost term) | 5.23 | 6318 | 0.3340 | 0.05423 |
| curvature-embedding-rq (Plan A) | 4.26 | 14021 | 0.0201 | 0.04897 |
| community-pairs-guarded-rq (contrast + guard) | 3.03 | 6575 | 0.2686 | 0.04328 |
| the threshold the programme aims at | | | | 0.065 |

The error rate is a ratio (`positives x expected negatives / negatives`), so a
**higher** rate means the identifier discriminates behaviour pairs better, and
the recall follows it. See `WHY_IT_FAILED.md` for the measurement that explains
the ordering and closes the mechanism.

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

## 2026-11-07 — the graph reconstruction round, and the finding that outranks it

**The determinism finding.** Two runs of the frozen baseline's Stage 3 — the
same code, the same seed, the same data — ended at `test_R@10` **0.055537** and
**0.040199**, a 38% relative spread. The Stage-3 trainer pins no deterministic
algorithm and the cuBLAS workspace was unset. Both launch paths now pin
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, each with a regression test. Until that fix,
every Stage-3 comparison in this campaign was comparing noise to noise; the
committed baseline number is no longer a usable reference and the comparison is
made against a fresh run under the pinned environment.

**The community transition, which the campaign had never measured.** The
campaign had only ever asked whether the next item is in the *same* community,
and measured that at chance. On the independent validation split, with the
transition fitted on the train split:

| predictor of the next community | NLL | hit@1 | hit@10 |
|---|---|---|---|
| global community frequencies | 4.1531 | 0.0196 | 0.1878 |
| community transition (current → next) | **4.0440** | **0.0459** | **0.3029** |
| the same, with the targets shuffled (control) | 4.2663 | 0.0156 | 0.1739 |
| transition along the whole prefix | 5.6058 | **0.0548** | **0.3422** |
| the identifiers' own community-level prediction | 19.4011 | 0.0464 | 0.1945 |
| the transition conditioned on the identifiers too | 11.9671 | 0.0522 | 0.2847 |

The transition carries real information (it beats the global baseline on every
metric and the shuffled control is *worse* than the global baseline, so the
signal is the relation between communities and not their sizes), and it carries
information the identifiers do not: the identifiers' own community-level
prediction reaches hit@10 0.1945 against the transition's 0.3422, and
conditioning the transition on the identifiers does not improve it.

The earlier "chance level" reading was a measurement artefact: it compared the
target against the *modal community of the whole prefix*, not the last item's
community. Against the last item's community the rate is 0.0452, 2.9× the base
rate.

**The correction to the model side.** A reproducible re-run of the whole arm
table gives 139.47 for the plain GCN propagation against 134.54 for the
curvature-aware one, so the curvature-aware propagation does **not** raise the
AF3 gap here and the earlier 145.11 was a property of an earlier version of the
shared model. The mixed geometry remains the one robust modelling gain
(A → B: +30% on the separation, and the propagation arms another +25% over B);
the AF3 curvature does not earn its place inside the community model.

**The graph reconstruction arm.** `TIGER_community_graph_recon` adds one loss:
the quantised representation must predict the behaviour neighbourhood weighted
by the community geometry, on the sum of the three codewords, with the
encoder path through the straight-through estimator and the codebook path
through the undetached codewords, ramped in at epoch 1500 and capped at 10% of
the base gradient. Ten of ten machine checks passed; the run took 600.8 s for
3000 epochs. The three questions the experiment asked:

* does the community weight move the target? **By 0.2% of its entropy**
  (target 2.3865 nats against uniform 2.3919). The specified bound `1 + s` with
  `s ≤ 1` is what makes it inert.
* does the term reach the parameters and the identifiers? **Yes on both**: the
  term's own gradient is non-zero for the encoder and the codebook at every
  logged epoch, the applied scale rises 3.1e-6 → 3.4e-5 to hold the
  contribution at a tenth, and the identifiers never freeze.
* are the behaviour neighbours better preserved? **No**: the level-1 share is
  0.0821 against the baseline's 0.0833 and the level-3 enrichment is worse. The
  hierarchy's *shape* improved (collision 0.1775 → 0.1672, largest level-1 code
  3.49% → 2.62%) with the prefix count unchanged.

The lesson for the next design is the first bullet: a community weight bounded
in `[1, 2]` over eleven neighbours cannot express a preference strong enough to
reshapes the identifiers, and the number to design against is the target's KL
from uniform, not the community model's separation.
