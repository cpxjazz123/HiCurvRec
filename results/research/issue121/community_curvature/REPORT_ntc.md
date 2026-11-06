# The node-to-cluster enhancement: what it improved, and what it did not

**Question.** Make the items not only satisfy the high/low curvature ordering but
also move closer to the right community centre, while the communities keep their
distinctness. Then look directly at the final **hard** communities: did their
AF3 gap and their cohesion improve?

**Answer.** The soft, pairwise signal improved and the hard communities did not.
The node-to-cluster terms raise the validation curvature separation by 14% over
the same model without them (0.058888 against 0.051532, paired over the 24 571
anchors, permutation p = 0 of 20 000) with a partition that stays balanced (the
largest hard community holds 1.77% of the items, 62.7 of the 64 communities are
effectively used). But the hard **AF3 gap falls slightly** (116.17 against
120.55 on the validation graph) and the hard **cohesion falls slightly**
(community transitivity 0.0705 against 0.0772). The mechanism buys a sharper
ordering of the community overlap and pays for it with a slightly worse
partition: its ranking loss ends at 0.01815 against the other model's 0.00845,
so the commitment term competes with the curvature signal for the geometry.

The mechanism is therefore **not** licensed to go into the RQ-VAE on this
evidence, and fRGCN is not licensed by a hard-structure improvement either,
because there was none. What the four attempts did deliver is a protocol that
can measure the hard structure honestly — that was the precondition for the
question to be answerable at all.

## The four attempts, and what each one exposed

The attempt log is in `attempts.json`; the first attempt is archived in
`attempt_1_unbalanced.json`.

| # | assignment | weights | what it showed |
|---|---|---|---|
| 1 | softmax on the standardised cost | 1 everywhere | the commitment term decided the model: its value was 0.55 against the ranking hinge's 0.0156, the ranking loss never moved off its initial value, and the hard partition collapsed (87.4% of the items on one community) |
| 2 | softmax on the standardised cost | per-model gradient-norm balancing | the ranking signal was rescued, but the **soft** usage constraint could not see the hard partition: every model put 82% to 99.7% of its items on one community while its soft usage stayed uniform, and the AF3 gap and the transitivity inflated for that reason alone |
| 3 | the baseline's balanced plan, five iterations | derived (ranking 1/margin, usage 1, commitment 1, separation 1/threshold) | the node-to-cluster terms raised the separation by +0.167 (p = 0) and again folded 54.7% of the items into one community: on a near-decision cost, five iterations leave the plan's columns unbalanced by tens of percent |
| 4 | the baseline's balanced plan, one hundred iterations | derived | the hard partition is balanced for every model (largest community 1.77%–1.86%, 62.7–63.8 effective communities), the hard metrics are trustworthy, and the verdict above is what the mechanism actually does |

The two failures were not the mechanism's fault alone: both were the *soft*
assignment hiding the *hard* collapse. A softmax over a standardised cost is
nearly flat over 64 centres, so the soft usage can look uniform while the
argmax concentrates on one community; and the usage constraint only sees the
soft distribution. The RQ-VAE's own assignment rule — the balanced Sinkhorn
plan on the centred cost — has the opposite property: the plan is nearly a
decision, so the soft usage and the hard usage agree and the constraint bites
the partition that is actually measured.

## The final numbers (attempt 4)

Validation graph, one run per arm, seed 42, no sweep and no ablation.

| model | separation | AF3 gap (valid) | community transitivity | group transitivity | largest hard community |
|---|---|---|---|---|---|
| A — free 128 (control) | 0.040659 | 121.88 | 0.0710 | 1.0000 | 1.86% |
| B — free 64 + hyperbolic 64 | 0.051532 | 120.55 | 0.0772 | 1.0000 | 1.82% |
| C — B + node-to-cluster | 0.058888 | 116.17 | 0.0705 | 1.0000 | 1.77% |

Paired over the anchors: B − A = +0.010873 [+0.009038, +0.012736], p = 0;
C − B = +0.007356 [+0.004011, +0.010610], p = 0. Random-label null on the
validation graph: AF3 gap mean 1.47 (p95 9.14), community transitivity mean
0.0308 (p95 0.0356), group transitivity 1.0 — the last one is saturated at
K = 64 and carries no information.

The AF3 gap of every arm is already 80 times the random-label null, and it is
the **curvature ranking** that produces it: the ranking loss asks exactly for
the high-curvature pairs to share a community and the low-curvature pairs not
to. The node-to-cluster term adds tightness around the centres — no
curvature-aligned information — and it competes with the ranking signal for
the same geometry. That is why the pairwise separation rises while the hard
partition does not.

## What this does to the earlier claim

The first experiment (commit `bf7c3970`) reported the mixed geometry beating
the Euclidean control by 2.13× (0.015825 against 0.007448). That measurement
was taken under the flat-softmax protocol, and its hard partitions were
balanced (largest community 7.19% and 3.73%), so the claim itself stands; but
the *magnitude* is protocol-dependent, because a flatter assignment gives the
soft overlap more room to separate. Under the final protocol the same
comparison reads **+27%** (0.051532 against 0.040659, p = 0 of 20 000, 11 271
of 24 571 anchors improved). The direction is stable; the size is not a
constant.

## Where the plan goes from here

* The **node-to-cluster** mechanism as implemented does not improve the hard
  communities. It can be kept as a knob (the terms exist, they are derived and
  switchable) but it must not be claimed as a hard-structure improvement, and
  it must not be carried into the RQ-VAE on this evidence.
* The **balanced-plan protocol** is the thing to keep: without it the hard
  metrics cannot be read at all, and the earlier soft-softmax readings of the
  AF3 gap and the transitivity were one collapse away from being meaningless.
* **fRGCN** (the neighbour structure participating in the embedding) is the
  next candidate the plan names, and it is the right one: the curvature
  ranking already produces the gap, so what is missing is information the
  neighbourhood carries and the embedding does not. It should be tried on the
  balanced-plan protocol, with the hard AF3 gap and the hard cohesion as the
  judgement and the largest hard community as the guard.
* Learnable curvature and the MLP curvature-consistency loss stay where the
  plan put them: later.
