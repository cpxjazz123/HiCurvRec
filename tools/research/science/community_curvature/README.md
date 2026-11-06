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
| assignment | the baseline's balanced plan on the centred cost | the same |
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
| ranking weight | 1/margin = K | the hinge lives on the scale of its margin, so it is divided by that margin to become O(1) |
| usage weight | 1 | the usage term is `1 − H(usage)/log K`, dimensionless and bounded by 1 |
| commitment weight | 1 | the commitment lives on the scale of the frozen branch distances |
| separation weight | 1/threshold | the hinge lives on the scale of the threshold it is measured against |
| Sinkhorn iterations | 100 | a solver setting, not a tuned quantity: with a near-decision cost, five iterations leave the plan's columns unbalanced by tens of percent, and an unbalanced plan lets a node-to-cluster term fold the batch onto one community |
| epochs / batch / lr | 60 / 1024 / 1e-3 | training budget defaults, not swept |

### The node-to-cluster terms (optional, the third arm)

The third arm adds two terms on top of the same losses:

* **commitment** — `sum_k pi_ik * d(i, k)`, the distance an item keeps to the
  centre its soft assignment leans towards. It needs an **absolute** distance:
  the standardised distance of the assignment cannot carry it, because
  standardising per item rescales the row and the term would never ask an
  item to move closer. The absolute distance is measured in units of the
  branch's scale at initialisation, frozen once from the first batch.
* **separation** — every centre keeps at least the distance to its nearest
  neighbour that the centres had at initialisation. The hinge is zero once
  they are that far apart, so the term cannot be won by pushing the centres to
  infinity.

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
| hard partition balance | the largest hard community's share and the effective number of hard communities — the guard that says whether the hard metrics mean anything |

The judgement is the **validation** separation: if the mixed geometry does not
separate high-AF3 from low-AF3 pairs better than the Euclidean control on
held-out interactions, the mechanism is refuted and nothing is carried into the
RQ-VAE.

## Results

Two reports, both in `results/research/issue121/community_curvature/`:

* `REPORT.md` — the first experiment: the mixed geometry separates the
  curvature-ordered pairs better than the Euclidean control. Under the
  protocol of that run the separation was 0.015825 against 0.007448; under the
  final protocol (below) the same comparison reads 0.051532 against 0.040659.
  The direction is stable, the size is protocol-dependent, and the report says
  which numbers belong to which protocol.
* `REPORT_ntc.md` — the node-to-cluster enhancement: it improves the soft
  separation by 14% (p = 0) with a balanced partition, and it does **not**
  improve the hard communities' AF3 gap (116.17 against 120.55) or their
  cohesion (0.0705 against 0.0772). `attempts.json` records the four
  configurations that got there, and why the first three were not readable.
* `REPORT_frgcn.md` — the curvature-aware graph convolution: the neighbourhood
  raises the hard AF3 gap from 124.52 (no graph) to 134.52 (a plain GCN
  propagation) to **145.11** (curvature-aware), and the curvature ranking loss
  drops from 0.00898 to 0.00291; the hard cohesion does not move (0.0661 →
  0.0652 → 0.0638), so the mechanism passes on the gap and fails on the
  cohesion. The plain GCN arm is the control that separates "a graph helps"
  from "the curvature weighting helps".

The final protocol assigns items with **the RQ-VAE's own rule** — the balanced
Sinkhorn plan on the centred cost, run to convergence — because a softmax
over a standardised cost is too flat to see a collapse: the soft usage can
look uniform while the hard partition has folded onto one community. The test
suite asserts that this module's plan and the baseline quantizer's plan are the
same algorithm.

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
