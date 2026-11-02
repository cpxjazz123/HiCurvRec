# Issue #128 — the curvature evidence was an artefact of a one-direction graph

The bug report is correct on both counts, and the correction is decisive: the
association the curvature analysis reported **does not survive the fix**.

## The two defects

1. **The behaviour graph and its prefix quotients held one orientation per
   edge.** `build_behaviour_graph` canonicalised every co-occurring pair as
   `(min, max)` and inserted only that sparse entry, and the quotient did the
   same with the group ids. The walk, the row degrees and the hop costs were
   then computed on an **upper-triangular** graph, so the curvature carried an
   item-id artefact. Fixed by `symmetrise()` — `A.maximum(A.T)` with the
   diagonal cleared — applied to the item graph and to every quotient, in one
   place (`tools/research/science/quotient_curvature.py`) that both code paths
   import or copy.
2. **The ground metric was a bounded approximation.** Two cells that are not
   adjacent were priced at 2 hops although the shortest path can be longer, and
   the Stage 2 module used a source-dependent fallback. The estimator now
   carries the exact unweighted shortest path, and the bounded form that a
   182-arm sweep can afford is **verified against the exact one** on a sample,
   with the numbers travelling with every result.

## The evidence, before and after

| quantity | before the fix | after the fix |
|---|---|---|
| level-1 curvature vs Recall@10 (Spearman) | −0.130 | **−0.075** |
| level-2 curvature vs Recall@10 | +0.169 | **+0.026** |
| level-3 curvature vs Recall@10 | +0.219 | **−0.077** |
| the profile (level-1 minus level-3) | −0.184 | **−0.047** |
| arms with a positive level-3 deviation | 156 / 168 | **73 / 168** |

The regenerated report (`analysis_sid_recall/curvature/report.md`) now reads:

```text
z_curv_L1            n=167  Spearman -0.061  permutation p 0.43  CI [-0.207, +0.093]
z_curv_L2            n=179  Spearman +0.051  permutation p 0.50  CI [-0.083, +0.198]
z_curv_L3            n=181  Spearman -0.049  permutation p 0.51  CI [-0.206, +0.108]
z_curv_L1_minus_L3   n=167  Spearman -0.038  permutation p 0.63  CI [-0.192, +0.117]
sign_change_L1_L3    n=182  Spearman +0.098  permutation p 0.19  CI [-0.058, +0.242]
```

**Every curvature feature is statistically null.** The earlier claim — that the
standardised quotient curvature tracks the recall at about −0.20 — was an
artefact of the one-direction graph and the bounded metric. It is withdrawn.

The correction also changes the *profile* the mechanisms spend. On the frozen
baseline's prefixes the corrected measurement gives

```text
level 0 (247 prefixes):    kappa 0.268   null 0.350   std 0.011   kappa_hat -7.49
level 1 (7407 prefixes):   kappa 0.015   null -0.007  std 0.001   kappa_hat +18.75
level 2 (20201 prefixes):  kappa -0.215  null -0.247  std 0.010   kappa_hat +3.33
```

against the pre-fix `[-2.32, -1.80, -0.86]`. After the ±1 clip the deviation is
`[-1, +1, -1]`, so the level weights a mechanism derives from it are
`[0, 0.5, 0]` — the pull sits on the **second** level, not the third.

## The second null

The issue asks for two nulls because they answer different questions. Both are
now measured for the reference SID:

| level | size-preserving SID shuffle | degree-preserving graph rewiring |
|---|---|---|
| 0 | kappa_hat −7.49 (null mean 0.350, std 0.011) | −3.91 |
| 1 | kappa_hat +18.75 (null mean −0.007, std 0.001) | +2.24 |
| 2 | kappa_hat +3.33 (null mean −0.247, std 0.010) | +0.31 |

The amplified magnitudes (z ≈ 19) are a property of the null's tiny spread
(std ≈ 0.001 at level 1): a small systematic excess over a very tight null is a
large z. That is why the sweep reports the raw curvature and the z together.

## The bounded metric's verification

The sweep uses the 1/2-hop cost; the exact metric costs ten seconds per edge on
a dense quotient. The bound is verified per level against the exact transport:

```text
level 0: mean |exact - bounded| 0.0000
level 1: mean |exact - bounded| 0.0048
level 2: mean |exact - bounded| 0.0139
```

## Regression tests

`tools/research/tests/test_quotient_curvature.py` — 10 tests:

* the item graph and the quotient are symmetric with no self-loops, and the
  neighbour sets are the co-occurrence ones;
* a missing `-1` label is excluded and produces no negative index;
* the same edges under a permuted id order have the same per-edge curvature
  (the invariant that the one-direction construction violated);
* the shortest path is exact in both directions (A→D = D→A = 3 on a 4-node
  path) and **the middle edge of that path has curvature exactly 0** under the
  0.5-lazy walk with the exact transport;
* the transport cost is symmetric, nonnegative and zero on the diagonal, and a
  disconnected support fails closed instead of returning a zero;
* a degenerate null is reported `NOT_MEASURED`, never a silent zero.

## The pair table behind the numbers (added after the correction)

The 182 (SID, recall) rows of `sid_recall.csv` are not 182 published pairs. An
audit (`analysis_sid_recall/pair_audit.py`) applied two mechanical criteria — the
SID table and the recall file must be introduced by the same git commit, and the
`results/stage2_RQ-VAE/<X>` and `results/stage3_T5Train/<X>` directories must
carry the same name — and **73 pairs pass**. Restricting the statistics to them:

| feature | all 182 | same commit (86) | verified (73) |
|---|---|---|---|
| `z_curv_L2` | +0.150 (p 0.035) | +0.074 (p 0.50) | +0.129 (p 0.28) |
| `z_curv_L3` | +0.145 (p 0.049) | +0.029 (p 0.80) | −0.044 (p 0.71) |
| profile (`L1−L3`) | −0.095 (p 0.23) | +0.051 (p 0.64) | +0.141 (p 0.24) |

So the only p ≈ 0.05 associations belong to the unaudited table, and they
disappear under either criterion. Combined with the sampling-budget instability
(sign reversals, |Δρ| up to 0.19), the conclusion of this issue stands and
sharpens: **no curvature–recall association can be claimed in either
direction.** Details: `analysis_sid_recall/PAIR_AUDIT_FINDINGS.md`.

## What this does not change

The downstream Recall of the arms is untouched: it was measured, and no
mechanism's Stage 3 result depends on these numbers. What changes is the
*interpretation* of the curvature, and the profile the next mechanism spends —
which is now the corrected one.

## Artefacts

`analysis_sid_recall/curvature_before_issue128/` (the pre-fix evidence, kept),
`analysis_sid_recall/curvature/` (the corrected evidence and the regenerated
report), `analysis_sid_recall/curvature_before_issue128/before_after_summary.json`,
`analysis_sid_recall/curvature_before_issue128/before_after_per_arm.csv`,
`tools/research/science/quotient_curvature.py` (the authority),
`tools/research/science/curvature_profile.py` (the profile generator with both
nulls), seeds and every estimator parameter are recorded in the JSON.
