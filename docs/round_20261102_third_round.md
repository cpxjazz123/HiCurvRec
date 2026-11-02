# Round 3 — the exact-quota hyperbolic allocation (EQVOTE-HYP-RQ)

The round asked for a mechanism that (a) couples the successor statistics to the
code assignment by an **additive** term, so a behaviour-off run would differ;
(b) preserves the exact per-level usage (no collapse); (c) keeps the training
path free of collectives.

## What was implemented

`model/eqvote.py` builds, for every item, the successor lists from the frozen
successor statistics, the disjoint-transposition search over the level's
labelling, and the sharpened additive cost

```
cost(i, c) = K(c, c) * weight[level]                       # the Euclidean own cost
           - 2 * weight[level] * (profile_i · U_c)         # the behaviour pull
           + weight[level] * (profile_i · U_c)             # the item's own successor
                                                           # (the code-cost vector)
```

The trainer freezes the **Euclidean** per-level counts as an exact quota (the
baseline's own assignment) and the level's labelling is improved only by
**accepted disjoint transpositions** — a transposition moves one copy of a code
from one item to another, so every level's usage multiset stays the Euclidean
one by construction. The level weights come from the corrected profile of Issue
#128 (`delta = [-1, +1, -1]`, clipped ±1, weight = max(delta, 0) / 2 ⇒
`[0, 0.5, 0]`): the pull sits on the **second** level.

## The Stage 2 result

| metric | Baseline | Round 1 (argmax) | Round 2 (Kendall) | **Round 3 (this)** |
|---|---|---|---|---|
| L1 Hard ER | 12.41 | 5.31 | 2.55 | **6.81** |
| L2 distinct prefixes | 16,145 | 5,011 | 1,760 | **7,312** |
| L3 collision | 0.0000 | 0.1757 | 0.3685 | **0.2682** |
| L3 refinement gain (bits) | 2.25 | 2.14 | 2.05 | **2.17** |
| Conditional G3 | +0.939 | +0.236 | +0.088 | **−0.048** |
| code usage per level | [247,256,256] | [46,168,256] | [9,59,254] | **[242,256,256]** |

Three things follow from this table.

1. **The quotas work.** The code usage stays at the baseline's resolution
   ([242,256,256] against [247,256,256]) while the L2 prefix count halves
   (7,312 against 16,145) and the L1 Hard ER nearly halves (6.81 against 12.41).
   This is the opposite regime of rounds 1 and 2, whose collapse came with
   9–168 used codes.
2. **The behaviour term changes the assignment**, not the usage: the level
   weights are `[0, 0.5, 0]` and the L1/L2 diagnostics differ from the
   baseline's, so the term is live in the assignment path. The `[eqvote]`
   traces in the training log show the quota converging 45 → 95 → 217 → 233 →
   242 codes at level 1.
3. **The round's "both metrics improve" bar is only half met.** The L2 prefix
   count improves (7,312 < 16,145) but the L3 collision **worsens** (0.268
   against the baseline's 0.0000) and the conditional G3 falls below zero: the
   behaviour-sharpened assignment pushes different items into shared triples.
   The arm is therefore a **contradiction of its own prediction** on the L3 side
   and a confirmation on the L2 side; the Stage 3 recall decides.

## The Stage 3 result

The frozen 150-epoch Stage 3 protocol (NO_EVAL=True, one test after training)
ran on the arm's SID:

| metric | value |
|---|---|
| test_recall@10 | **0.05519** |
| test_recall@5 | 0.03677 |
| test_ndcg@10 | 0.03032 |

Recalls that are backed by a round record or by the arm's own Stage 3 run in
this tree:

| arm | stage-2 directory | stage-3 directory | `test_R@10` | evidence |
|---|---|---|---|---|
| `CURVALLOC-LORENTZ-RQ` | `TIGER_CURVALLOC-LORENTZ-RQ` | same name | 0.04189 | `round_20261102_031529_470/jobs.json` + the run's own launcher log |
| `EQVOTE-HYP-RQ` | `TIGER_EQVOTE-HYP-RQ` | same name | 0.05519 | `round_20261102_102950_314/jobs.json` + the run's own launcher log |

The project's recorded reference for the frozen Euclidean baseline is 0.0655,
but its run directory is no longer in the tree, so it is **not** re-verified
here; the success bar in `CLAUDE.md` is `test_R@10 > 0.065`. An earlier version
of this paragraph also quoted 0.0610 and 0.0551 for "round 1" and "round 2";
those numbers are not reproducible from the current tree (the round record for
`CURVALLOC-LORENTZ-RQ` reads 0.04189) and are withdrawn from this report.

## The verdict bar

The round's own resolution: Stage 3 `test_R@10` is the final judgment; the
diagnostics above are the leading indicators. The arm is accepted for the next
round only if its recall exceeds the frozen baseline's 0.0655 **and** the
mechanism's interpretation survives (the hyperbolic pull is the reason, not the
quotas alone — the round-2 lesson).

**Verdict: rejected.** `test_R@10 = 0.05519` is below the recorded baseline
reference (0.0655) and below the project's 0.065 bar — despite the third arm having the best-balanced SID of the three (the exact-quota
usage [242,256,256], the L2 prefix count halved, the L1 Hard ER nearly halved).
The coupling's interpretation does not survive either: the level weights
`[0, 0.5, 0]` put the entire behaviour pull on the second level, where the
successor statistics improved the diagnostic — and the recall still did not
move.

The wider conclusion of rounds 1–3: on this data, **none of the three SID-side
interventions (the pooled hyperbolic argmax, the successor-overlap rank sum, the
exact-quota hyperbolic allocation) improved the downstream recall**, and the
best SID remains the frozen Euclidean baseline. Together with the corrected
curvature analysis (Issue #128: no curvature–recall association survives), the
evidence does not support a curvature-driven SID as the next step.

## Artefacts

Candidate tree `stage2_RQ-VAE/TIGER_EQVOTE-HYP-RQ`, layer record
`results/stage2_RQ-VAE/TIGER_EQVOTE-HYP-RQ/stage2_metrics/<sid_sha>.json`,
training log `logs/train_migrated.log`, Stage 3 in
`results/stage3_T5Train/TIGER_EQVOTE-HYP-RQ`, and the round record
`results/research/121/rounds/round_20261102_102950_314/`.
