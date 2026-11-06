# Experiment 1 — curvature-guided soft community learning

## The question

> Without changing the RQ-VAE quantizer, can a **mixed free + hyperbolic**
> encoder learn item communities that agree with the Augmented Forman-Ricci
> (AF3) curvature of the behaviour graph **better than a pure Euclidean encoder
> of the same capacity**?

This runs **before** any RQ-VAE change on purpose. The mechanism is judged on
its own, on data it never saw, and only a positive answer earns the next step
(turning the community probabilities into L1 supervision).

## What is learned

| | model A (control) | model B (experiment) |
|---|---|---|
| input | frozen 768-d sentence-T5 embedding | the same, unchanged |
| free space | 128-d learnable projection | 64-d learnable projection |
| hyperbolic space | none | 64-d learnable projection → Poincaré ball, c = 1 fixed |
| community centres | 64 learnable | 64 learnable per branch |
| assignment | soft, over the 64 communities | soft, over the 64 communities |
| RQ-VAE / SID / Stage 3 | untouched | untouched |

Both branches receive the **same** frozen input and learn their own
representation; the input embedding is never modified. The curvature is fixed
at c = 1 (sectional curvature −1) in this round, so the only variable is the
**geometry of the combination**, not its strength.

The hyperbolic branch uses the frozen contract operators of the Stage-2
baseline (`stage2_RQ-VAE/TIGER_RQ-VAE/model/poincare_contract.py`), imported by
path; the run report carries that file's sha256.

## How the communities are supervised

The AF3 curvature of an edge (the frozen convention of issue #136,
`analysis_sid_recall/augmented_forman_gap/af3.py`, imported and never
re-derived):

```
F_H(e)   = 4 - deg_H(u) - deg_H(v)
T_H(e)   = |N_H(u) ∩ N_H(v)|
AF3_H(e) = F_H(e) + 3 * T_H(e)
```

For every anchor item that has two incident edges of **different** curvature,
the neighbour on the highest-AF3 edge is *B* and the neighbour on the
lowest-AF3 edge is *C*. The overlap of two items' soft distributions is

```
S_ij = sum_k pi_ik * pi_jk
```

and the loss is the ranking hinge

```
L_community = max(0, m - S_AB + S_AC)
```

So a pair of items joined by a high-curvature edge has to share more community
mass than a pair joined by a low-curvature edge. An anchor whose incident
edges all carry the same curvature carries no signal and is dropped.

**Why the relative order and not the CONGREGATE Ricci loss:** Ollivier-Ricci
and augmented Forman-Ricci both describe local graph structure, but their
values and signs are not interchangeable, so the first round asks the question
the AF3 numbers can actually answer — *does a high-AF3 pair belong together
more than a low-AF3 pair?* — instead of importing a loss defined on another
curvature.

### The constants, and where they come from

| constant | value | why |
|---|---|---|
| communities K | 64 | the experiment's design |
| curvature c | 1.0 | fixed in this round on purpose |
| margin m | 1/K | under a uniform soft assignment every pair shares S = 1/K, so requiring S_AB − S_AC ≥ 1/K asks the high-curvature pair to beat chance |
| temperature | implicit | each branch's distance row is standardised per item (zero mean, unit spread) and the branches are averaged with equal weight, so no temperature or branch weight had to be chosen |
| usage weight | 1 | the usage term is `1 − H(usage)/log K`, dimensionless and bounded by 1 — the same order as the hinge |
| epochs / batch / lr | 60 / 1024 / 1e-3 | training budget defaults, not swept |

### The anti-collapse constraint

`L_usage = 1 − H(usage)/log K`, where `usage` is the mean soft assignment over
the items. It is exactly zero for a uniform usage of the 64 communities and one
when every item falls on one community, so a collapse is impossible without
having to pick a weight by trying values.

## The data

* **graph**: two items are adjacent when they follow each other inside a user
  sequence. The training graph is the frozen authority issue #136 measured
  (`analysis_sid_recall.curvature_analysis.build_behaviour_graph`), and the
  test in `tools/research/tests/test_community_curvature.py` asserts that the
  graph this experiment measures is edge-for-edge the same one.
* **train split** supplies the curvature ranking pairs the model is trained on.
* **validation split** supplies an **independent** sample: the same construction
  on the held-out interactions, with its own AF3 curvature, used only for
  evaluation. Nothing from it enters the training.

## What is measured

| measurement | question it answers |
|---|---|
| curvature separation | how much more community mass a high-AF3 pair shares than a low-AF3 pair of the same anchor (bootstrap interval over anchors) |
| community occupancy | how many of the 64 communities are used, and how concentrated the usage is |
| community transitivity | whether a community's items form a tight subgraph (induced subgraph transitivity) |
| AF3 curvature gap | within-community minus cross-community edge curvature, with the frozen `af3.gap_metrics` |
| group transitivity | the transitivity of the community quotient graph |
| random-label null | the same measurements on random assignments, so every number has the bar it must clear |

The judgement is the **validation** separation: if the mixed geometry does not
separate high-AF3 from low-AF3 pairs better than the Euclidean control on
held-out interactions, the mechanism is refuted and nothing is carried into the
RQ-VAE.

## Result (2026-11-06)

The mixed geometry **passes** the experiment's own judgement metric: the
validation curvature separation is 0.015825 against the Euclidean control's
0.007448 (2.13x, disjoint bootstrap intervals, paired permutation p = 0 of
20 000, 18 426 of 24 571 anchors improved), the validation separation is even
marginally above the training one (so it is not memorisation), and there is no
collapse (64/64 communities used, largest community 3.73% of the items against
the control's 7.19%).

It is a narrow win, and the report says so: the AF3 curvature gap and the
community transitivity are ties, and the group transitivity is saturated at
K = 64 (a random assignment reads 1.0 as well), so only the ordering of the
community overlap separates the two models. The full numbers are in
`results/research/issue121/community_curvature/REPORT.md`.

## Run

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
    tools/research/science/community_curvature/run_experiment.py
```

Outputs land in `results/research/issue121/community_curvature/`:
`report.json` (every number), `pi_<model>.npy` (the soft assignments) and
`REPORT.md` (the readable summary).

Tests:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
    tools/research/tests/test_community_curvature.py -q
```
