# Experiment 1 — curvature-guided soft community learning

**Question.** Without touching the RQ-VAE quantizer, can a mixed free +
hyperbolic encoder learn item communities that agree with the AF3 curvature of
the behaviour graph better than a pure Euclidean encoder of the same capacity?

**Answer.** Yes, on the metric the experiment was designed around, and by a
wide margin: the validation curvature separation is **0.015825** for the mixed
model against **0.007448** for the Euclidean control — 2.13×, with disjoint
bootstrap intervals and a paired permutation p of 0/20000. The mixed model also
keeps the communities more evenly used (largest community 3.73% of the items
against the control's 7.19%) and shows no sign of having memorised the training
pairs (its validation separation is marginally *higher* than its training one).

Where the mixed geometry does **not** win: the AF3 curvature gap and the
community transitivity are ties, and the group transitivity is saturated at
K = 64 and therefore uninformative. The advantage is specifically in the
**ordering** of the community overlap, not in the partition's edge structure.

## Protocol

| | |
|---|---|
| input | the frozen 768-d sentence-T5 item embeddings (24587 items), unchanged |
| model A (control) | one learnable projection, free 128-d, 64 learnable centres |
| model B (experiment) | two learnable projections of the same input, free 64-d **and** hyperbolic 64-d (Poincaré ball, c = 1 fixed), 64 learnable centres per branch |
| geometry | the frozen contract operators of the Stage-2 baseline, sha256 `46791369db959f91…` (`model/poincare_contract.py`) |
| loss | `max(0, m − S_AB + S_AC)` with `m = 1/K`, plus the usage term `1 − H(usage)/log K` |
| training | one run each, 60 epochs over 24 505 training anchors, batch 1024, Adam 1e-3, seed 42; no sweep, no second seed, no ablation |
| evaluation | the **validation split's** own behaviour graph and its own AF3 curvature: 24 584 nodes, 366 168 edges, 24 571 ranking pairs, never seen in training |

The margin is derived, not tuned: under a uniform soft assignment every pair
shares `S = 1/K`, so requiring `S_AB − S_AC ≥ 1/K` asks the high-curvature pair
to beat chance. The branch combination has no knob either — each branch's
distance row is standardised per item and the branches are averaged with equal
weight.

## Data

| | train split | validation split |
|---|---|---|
| nodes / edges | 24 556 / 315 391 | 24 584 / 366 168 |
| ranking pairs | 24 505 | 24 571 |
| mean AF3 of the high edge | −29.80 | −34.30 |
| mean AF3 of the low edge | −759.50 | −926.88 |

The graph is the frozen authority of issue #136, edge for edge, **binarised**:
the AF3 convention is defined on an unweighted graph, and a weighted one would
change both the degrees and the triangle counts.

## The judgement metric — validation curvature separation

| model | S(high) | S(low) | separation | 95% bootstrap CI |
|---|---|---|---|---|
| A — free 128 | 0.02446 | 0.01702 | **0.007448** | [0.007289, 0.007628] |
| B — free 64 + hyperbolic 64 | 0.02834 | 0.01251 | **0.015825** | [0.015533, 0.016114] |

Random pairs share 0.015664 under the control, so the control's low-curvature
pairs sit at chance while the mixed model's low-curvature pairs fall **below**
chance and its high-curvature pairs rise further above it: the mixed model
separates in both directions.

Paired over the 24 571 anchors: difference **+0.008377** [+0.008134,
+0.008621], permutation p = 0.0 (20 000 sign flips), 18 426 anchors improved.

Generalisation: model B's separation is 0.015685 on the training pairs and
0.015825 on the held-out ones — it learned a structure, not the training
sample.

## The secondary measurements

| measurement (validation graph) | model A | model B | random-label null |
|---|---|---|---|
| AF3 curvature gap | 164.32 | 163.55 | mean 1.47, p95 9.14 |
| community transitivity (mean) | 0.0776 | 0.0677 | mean 0.0308, p95 0.0356 |
| community transitivity (edge-weighted) | 0.0650 | 0.0777 | — |
| group transitivity | 0.9936 | 0.9990 | 1.0 |
| communities used | 64/64 | 64/64 | — |
| effective communities | 63.97 | 63.92 | — |
| largest community share | 7.19% | 3.73% | — |

Both models are far above the random-label null on the AF3 gap and on the
community transitivity, so the communities are real; but those two metrics do
not separate the models. The group transitivity reads 1.0 for a random
assignment as well — at K = 64 the community quotient graph is essentially
complete, so this metric carries no information here and must not be quoted as
evidence either way.

## No collapse

The usage term fell to 0.00022 (A) and 0.00044 (B) in the last epoch, all 64
communities are occupied, the effective number of communities is 63.9, and no
community holds more than 3.73% of the items. The mixed model is the more even
of the two.

## What this licenses, and what it does not

Experiment 2 (turning the community probabilities into L1 supervision) is
licensed: the mechanism was validated on held-out interactions, it beat its
capacity-matched Euclidean control decisively on the metric it was designed
around, and it did not collapse.

It is **not** yet evidence about the recommender. Nothing here touched the
quantizer, the Semantic IDs or Stage 3, and the metrics that this repository
tied to Recall@10 were either ties (AF3 gap, community transitivity) or
saturated (group transitivity). The community partition that a
recommender could use is therefore still unproven; what is proven is that the
mixed geometry orders the community overlap along the behaviour graph's
curvature better than the Euclidean control does.

## Files

* `report.json` — every number, with the protocol and the data census
* `pi_A_free_128.npy`, `pi_B_free_64_hyper_64.npy` — the soft assignments
  (24 587 × 64; not in git, the weights are regenerable by rerunning the script)
* the code: `tools/research/science/community_curvature/`, the tests in
  `tools/research/tests/test_community_curvature.py`
