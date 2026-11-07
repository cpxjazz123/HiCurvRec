# The curvature-aware graph convolution: the neighbourhood is where the curvature signal lives

> **Correction (2026-11-07).** The AF3 gap ordering reported here was a
> property of an earlier version of the shared model, not of the mechanism. A
> reproducible re-run of the whole arm table gives 139.47 for the plain GCN
> propagation against 134.54 for the curvature-aware one, so the curvature-aware
> propagation does **not** raise the gap here, and the curvature does not earn
> its place inside the community model. The arm table of a single run remains a
> valid comparison. See `REPORT_community_transition.md`.


**Question.** Let the item representation carry the behaviour graph's
neighbourhood, not only the item's own semantics, and check the hard
communities with the protocol that is now validated: the RQ-VAE's own balanced
plan, the hard AF3 gap, the hard cohesion, the separation, the balance guard
and the random-label null.

**Answer.** The neighbourhood is worth a lot on the curvature metric and nothing
on the cohesion metric. The hard AF3 gap rises from 124.52 (no graph) to
134.52 (a plain GCN propagation) to **145.11** (the curvature-aware
propagation), i.e. **+16.6% over the arm without a graph and a further +7.9%
over the plain neighbourhood**, and the curvature ranking loss drops from
0.00898 to 0.00291 — the neighbourhood makes the curvature ordering
substantially easier to satisfy, which is what one would expect if the
curvature signal of the graph is carried by the neighbourhood itself. But the
hard community cohesion does **not** improve (0.0661 → 0.0652 → 0.0638), and
the soft separation is highest for the *plain* GCN arm (0.066025 against
0.060842).

Under the rule that was fixed before the run — the arm has to improve the hard
AF3 gap, the hard cohesion, keep the partition balanced and not lose the
separation, and the curvature weighting has to beat the plain neighbourhood on
its own — **the mechanism passes on the AF3 gap and fails on the cohesion**:
it is a partial result, and the report says so.

## The five arms

One run each, seed 42, the same budget (60 epochs over 24 505 training
anchors, batch 1024, Adam 1e-3), no sweep, no ablation, no second seed.

| arm | separation (valid) | **AF3 gap (valid)** | community transitivity | group transitivity | largest hard community | effective hard communities | ranking loss (last) |
|---|---|---|---|---|---|---|---|
| A — free 128 (Euclidean control) | 0.039919 | 118.10 | 0.0687 | 1.0000 | 1.87% | 63.71 | 0.00686 |
| B — free 64 + hyperbolic 64 | 0.051717 | 124.52 | 0.0661 | 1.0000 | 1.75% | 63.88 | 0.00898 |
| C — B + node-to-cluster | 0.059904 | 109.81 | 0.0720 | 1.0000 | 1.77% | 62.69 | 0.02005 |
| D — B + plain GCN propagation | **0.066025** | 134.52 | 0.0652 | 1.0000 | 1.89% | 63.73 | 0.00415 |
| E — B + curvature-aware propagation | 0.060842 | **145.11** | 0.0638 | 1.0000 | 2.03% | 63.63 | **0.00291** |

Random-label null on the validation graph: AF3 gap mean 1.47 (p95 9.14),
community transitivity mean 0.0308 (p95 0.0356), group transitivity 1.0. The
group transitivity is saturated at K = 64 — a random assignment reads 1.0 as
well — so it carries no information and is not quoted as evidence.

Paired over the 24 571 validation anchors (permutation p over 20 000 sign
flips):

| comparison | difference | 95% CI | anchors improved |
|---|---|---|---|
| B − A (mixed geometry vs Euclidean) | +0.011798 | [+0.009791, +0.013855] | 11 342 / 24 571 |
| C − B (node-to-cluster) | +0.008187 | [+0.004736, +0.011560] | 8 845 / 24 571 |
| D − B (plain neighbourhood) | +0.014308 | [+0.012117, +0.016469] | 14 795 / 24 571 |
| E − B (curvature-aware neighbourhood) | +0.009125 | [+0.007014, +0.011344] | 15 321 / 24 571 |
| E − D (curvature weighting vs plain) | −0.005183 | [−0.007056, −0.003271] | 12 617 / 24 571 |

## What the curvature weighting does, and where it trades

The propagation is one layer, on the train split's behaviour graph only. The
edge weight is fixed, because the curvature of an edge is a property of the
graph and not of the model: the AF3 curvature of the edge is standardised over
the graph's edges and turned into a multiplicative weight on the GCN's
symmetric normalisation, `alpha_ij ~ exp(z_ij) / sqrt(deg_i deg_j)`, and the row
is normalised so every node's message is a weighted average of its neighbours.
The strength of the propagation is a learnable scalar, initialised at one. The
hyperbolic branch is propagated in the tangent space and mapped into the ball
afterwards, so the geometry is applied to the propagated representation.

The weighting does exactly what it says on the curvature metric: **E − D is
+10.59 on the hard AF3 gap** (145.11 against 134.52), while the plain
neighbourhood's advantage is in the *soft* separation (D − E = +0.005183, p =
0). In other words the curvature-aware weighting buys curvature alignment at the
cost of some of the soft ordering, and the arm with the most curvature-aligned
partition is not the arm with the sharpest soft ordering. That trade-off is
visible in the training traces too: the curvature-aware arm ends with the
lowest ranking loss of all five (0.00291), i.e. it satisfies the curvature
ordering most easily.

The plain GCN arm (D) is the control that makes the last statement possible:
without it, "E is better" would only have meant "a graph is better".

## What is not proved

* The **cohesion does not improve**. The community transitivity is 0.0661 for
  the arm without a graph, 0.0652 with the plain propagation and 0.0638 with
  the curvature-aware one: the partitions are equally tight, or marginally
  looser, while the AF3 gap grows. So the mechanism reorganises *which* edges
  fall inside a community (the curvature-aligned ones) without making the
  communities internally denser.
* The **node-to-cluster arm (C) remains a negative result** and is unchanged
  by this round: its hard AF3 gap is 109.81, below the arm without the terms
  (124.52), even though its separation is the second highest.
* One seed, one run per arm. The gap differences (124.52 → 134.52 → 145.11)
  are single measurements; the paired separations carry a permutation p of 0,
  the gap differences do not have such a test and are reported as they are.
* Nothing here touched the quantizer, the Semantic IDs or Stage 3. The AF3 gap
  is a property of the *community partition*, and the claim that a better
  partition produces a better recommender is still untested.

## Where this leaves the plan

The fRGCN direction has legs, and the evidence points at a specific place: the
neighbourhood is where the curvature signal lives, and injecting it makes the
curvature ordering dramatically easier to satisfy (the ranking loss drops by a
factor of three) and raises the hard AF3 gap by 16.6%. The curvature-aware
weighting is what moves the curvature metric; the plain neighbourhood is what
moves the soft ordering.

The next thing worth doing is therefore not another geometric variant but the
step the plan had queued: turn the community probabilities into L1
supervision, on the partition this round produced — where the AF3 gap is
145.11 instead of 124.52 — and let the Stage-3 recall decide. Learnable
curvature and the MLP curvature-consistency loss stay where the plan put them.

## Files

* `report.json` — every number, with the protocol and the data census
* `attempts.json`, `attempt_1_unbalanced.json` — the earlier rounds' record
* `REPORT.md` — the first experiment (mixed geometry)
* `REPORT_ntc.md` — the node-to-cluster round, including the four attempts that
  produced the validated protocol
* `pi_<arm>.npy` — the soft assignments (not in git; regenerable by rerunning)
