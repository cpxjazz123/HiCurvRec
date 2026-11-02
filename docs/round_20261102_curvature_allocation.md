
> **Correction (Issue #128).** The curvature numbers this round quotes were
> measured on a one-direction (upper-triangular) behaviour graph with a bounded
> ground metric. With the undirected graphs and the exact metric the association
> between the standardised curvature and the recall collapses
> (level-1 minus level-3: -0.184 before, -0.047 after; every feature is now
> statistically null), and the profile of the frozen baseline's prefixes changes
> from `[-2.32, -1.80, -0.86]` to `[-7.49, +18.75, +3.33]`. The round's Stage 3
> measurements are unaffected; the *interpretation* of the curvature is
> withdrawn. See `docs/issue128_status.md` and
> `analysis_sid_recall/curvature_before_issue128/before_after_summary.json`.

# The behaviour-graph curvature, explored with the multi-agent round framework

Round `round_20261102_031529_470` (issue 121), issue #127's open question taken
through the framework: **can the behaviour graph's curvature be spent on the
SID's geometry?**

The short answer is no, and the interesting part is that the mechanism did
exactly what the analysis predicted it should do and still made the SID worse.

## The round

| phase | what happened |
|---|---|
| ideation (5 independent agents) | four proposals, all `IN_SOPE`: `nullcurv-hinge-rq`, `CURVALLOC-LORENTZ-RQ`, `ORC-CELLCHART`, `CURVPROF-TRANSPORT-RQ`. Every one of them spends the **standardised** quotient curvature, never the raw one. |
| evidence probe | the train-only behaviour statistics, a small-tensor curved check and a verified literature pass, frozen per proposal |
| scoring (3 independent scorers, ranking only) | `CURVALLOC-LORENTZ-RQ` selected (total rank 5, mean rank 1.67); the other three are kept as backups under the one-mechanism cap |
| baseline chain | the last completed implementation, `TIGER_BRad-RQ` (its SID is already collapsed: 12 / 60 / 170) |
| implementation (the main agent writes it) | `model/curvature_allocation.py`, the per-level geometry in the quantizer, the 200-epoch refresh, 15 new tests; the tree's suite: 99 tests, 0 failures |
| Stage 2 | 4 GPUs, 3000 epochs, after three repairs (below) |
| Stage 3 | frozen recipe, `n_eval = 57439` |

## The mechanism, as it ran

The training split's co-occurrence graph (24,556 items, 157,695 edges) is
contracted into each level's prefix groups; the Ollivier curvature of the lazy
random walk on that quotient is standardised against a null that keeps the group
sizes and shuffles the members; the mean-zero, clipped, EMA-smoothed deviation
sets each level's curvature `c_l = exp(-delta_l)` and code radius
`rho_l = rho0 exp(delta_l/2)`.

At the final refresh the mechanism had measured and spent:

```text
kappa_hat = [  0.00,  -8.96, -25.55]     delta = [ 0.7022,  0.2977, -0.9998]
curvature = [ 0.4955,  0.7426,   2.7178]  radius = [0.0381, 0.0312,  0.0163]
```

That is the direction the correlation analysis favoured: a flatter coarse level
and a more curved fine level. The mechanism is not a no-op, and it is not the
wrong sign.

## The result

| arm | SID (L1 / L2 / L3) | collision | test_recall@10 |
|---|---|---|---|
| frozen baseline `TIGER_RQ-VAE` | 247 / 7,414 / 20,223 | 0.177 | 0.0553 |
| inherited baseline `TIGER_BRad-RQ` | 12 / 60 / 170 | 0.991 | 0.0446 |
| **this arm** | **2 / 13 / 56** | **0.998** | **0.041888** |
| target | — | — | > 0.065 |

The valid L1 positive / negative hit rates are 0.982 / 0.980: with two first-level
codes the first token carries no information at all.

So the curvature allocation made the SID **coarser than the baseline it inherits**
(170 → 56 distinct triples) and the recall **worse** (0.0446 → 0.0419), and it
is 36% below the target.

## What this says about the correlation

The analysis behind the round (`analysis_sid_recall/curvature/report.md`) found
that the standardised curvature profile tracks the recall (the level-1 minus
level-3 deviation: Spearman -0.20, p = 0.0095 over 115 mechanism families), and
it said so itself: *the association is observational; it does not say that
designing the space by the curvature profile would make the recall better*.

This round is the second half of that sentence, run end to end. The correlation
picked the right **sign** for the geometry, the mechanism spent it, and the
outcome was a worse SID. The reason is visible in the numbers: the profile is
read off a quotient graph whose density dominates the raw curvature, and
spending it on the geometry shrinks the code radius (`rho_l` fell to 0.016–0.038)
until the codebook's own resolution collapses with it.

**A curvature feature can predict which SID tables turn out well and still be
useless as a design rule.** The prediction is about tables that exist; the rule
is about tables you have to build.

## The three repairs, and what they say about the framework

1. **CPU broadcast.** The level geometry and the measured radius were broadcast
   on the CPU; the NCCL backend has no CPU backend, so every rank died in the
   first epoch. The single-process smoke cannot see this, because it never
   broadcasts.
2. **Transport support.** The exact transport problem on the real quotient has a
   far larger support than the proposal's "9+9 cells" assumption; the first
   refresh spent fourteen minutes. The support is now capped at 48 cells and the
   measures are renormalised over it; the refresh costs about four seconds.
3. **Deadlock.** The refresh exports the current prefixes on rank 0 alone and
   then broadcasts the geometry. The export's all-reduce over the used codes was
   not skipped, so rank 0 waited for the other ranks in the all-reduce while they
   waited for rank 0 in the broadcast. The four-rank probe of the framework
   exercises the model and not the trainer's main, and the single-process smoke
   exercises neither, so **no existing check could see it**. It is now pinned by
   a four-rank regression test with the same shape as the refresh.

The general lesson for the framework: a mechanism whose control flow differs
between rank 0 and the other ranks needs a check that runs the trainer's main
under `torchrun`, not a probe of the model.
