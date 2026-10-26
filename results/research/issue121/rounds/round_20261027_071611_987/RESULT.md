# round_20261027_071611_987 — hyperbolic x behaviour, the offset design

## What the objective asked for and what the round produced

The objective carried the exact offset form

    d_i = (b_i - x_i) - 0.5 (r_i - x_i),   z_i = x_i + w * gamma_i * d_i,
    code_i = argmax_k -|| z_i - v_k ||^2

with `b_i` the Poincare Frechet barycentre of the item's training predecessors' first-level
codepoints, `r_i` the same for its co-consumption hard negative, `gamma_i` the incoming-context
concentration, and `w` about 0.1. The round produced three proposals, selected
`transition-geodesic-prior` (weighted score above threshold, Copeland tie-break), implemented it
on a fresh copy of `TIGER_RQ-VAE`, and verified it after five revisions.

The review's findings, all real and all fixed:

* the Poincare denominator used `(1 - ||z||^2)` while the proposal declared curvature c = 0.1,
  so the geodesic was not the declared one;
* one pass returned `NEEDS_FIX` without a stated finding (process artefact, re-run);
* the connected path used predecessor-region masses and a geodesic kernel but not the proposed
  region transition matrix; the repair added it.

## Stage 2: RESOURCE_BLOCKED

The trainer logged **one** epoch, with a training-loop time of 0.475 s, and then spent more
than fifty minutes without logging a second one: the per-epoch rebuild of the predecessor-region
transition matrix costs about twenty minutes, so the frozen 3000-epoch budget extrapolates past
a thousand hours. The job was stopped and recorded as `RESOURCE_BLOCKED`, which keeps the idea
and moves it to the optimisation route rather than failing the mechanism.

The optimisation route is concrete: the region masses depend only on the frozen train-only
predecessor lists and the codebook, so they can be built once at startup (the codebook moves,
but the *assignments* can be refreshed every N epochs instead of every epoch), exactly as the
earlier arm's barycentres were.

## Status of the four iterations

| iteration | design | measured outcome |
|---|---|---|
| 1 | shared context replaced the assignment | 3000 epochs, Stage 3 **0.058497** (baseline 0.055537), but the partition collapsed: largest L1 bucket 0.985, `cooccur_overlap_L3` 0.0584 against 0.0756 |
| 2 | context as a bounded modulation of the Euclidean assignment | smoke-verified, 14 review revisions, not trained |
| 3 | Sinkhorn-balanced assignment, full-strength context | 3000 epochs, largest L1 bucket 0.693, `cooccur_overlap_L3` 0.0571, Stage 3 not spent (gate not met) |
| 4 | context as an offset on the item's own latent | verified, **RESOURCE_BLOCKED** at the per-epoch region-transition rebuild |

The one scientific constant across the iterations: every design that let the *shared* context
decide the code collapsed the partition, and every collapsed partition measured a coherence
below the plain baseline, while iteration 1 still reached a Stage 3 above the baseline — the
coherence metric is a population-level prior, not a sufficient statistic.

## Iteration 5: the same candidate, made trainable and measured

Three defects kept the verified candidate from ever completing a second epoch, and all three
were found only by running it on four ranks:

1. the mechanism summed the predecessor-region mixture into a **scalar per row** and then
   subtracted it from the `(N, 256)` distance matrix — a shape error at epoch 2 and, had it
   broadcast, a constant shift of every candidate of a row. It is now the per-code prior
   `distances -= 0.1 * (mass @ transition_logp)`;
2. the per-step per-row loop with a `bincount` call per item cost about 50 ms per step; it is
   now one gather plus one `scatter_add` over a frozen padded predecessor tensor;
3. the per-epoch tuple loop over `outgoing` ran ~2.8M dict lookups because the train split
   repeats every user's history at several prefix lengths; the unique pairs are now built once
   and the matrix is assembled with `np.add.at`.

Measured on four ranks after the fixes: **3000 epochs in 525 s, 0.14 s per epoch, exit 0** —
faster than the plain baseline's 0.15 s.

### The SID it produces

| quantity | this arm | baseline | iteration 1 (collapsed) | iteration 3 (balanced) |
|---|---|---|---|---|
| L1 codes | 170 | 247 | 116 | 104 |
| largest L1 bucket | **0.166** | ~0.02 | 0.985 | 0.693 |
| L2 prefixes | 2,928 | — | 572 | 853 |
| L3 triples | 13,325 | ~20,000 | 6,460 | 8,545 |
| extension alphabet | 117 | 42 | 136 | 117 |
| `cooccur_overlap_L3` | **0.0596** | **0.0756** | 0.0584 | 0.0571 |
| `adjacency_L3` | 0.0217 | 0.0506 | 0.0212 | 0.0192 |

The collapse is essentially gone (0.166 against 0.985 and 0.693), so the offset design did
what it was meant to do structurally — and the **target metric is still below the plain
baseline**. The gate was not met, so no Stage 3 budget was spent.

### The synthesis the five iterations give

Every mechanism tried here injects the **directed** behaviour structure — predecessors,
successors, transition masses, co-consumption hard negatives — into the assignment. The metric
that predicts recall, `cooccur_overlap_L3`, is **symmetric**: the mean co-consumption cosine of
item pairs that share a prefix. The two are not the same signal: grouping items by *who leads
into them* moves the symmetric co-consumption coherence **down**, consistently and reproducibly
(0.0584, 0.0571, 0.0596 against 0.0756), no matter whether the context replaces the assignment,
modulates it, is balanced, or is an offset.

The one mechanism in this repository that *did* raise the metric was symmetric by construction
(it grouped items by the cosine of their co-consumption rows) and it raised the metric
0.0756 -> 0.0786-0.0820 while moving Stage 3 by +0.0004. So the next lever is not another
way to inject the directed signal: it is to keep the **symmetric** grouping as the objective
and use the directed signal only where it cannot distort the partition (as a tie-break between
equally coherent codes, or as a per-epoch re-ordering of the assignment candidates), or to
accept that this baseline family is near the ceiling of what the coherence metric can express
and aim at the Stage 3 number through a partition-level search instead.
