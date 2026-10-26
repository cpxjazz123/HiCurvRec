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
