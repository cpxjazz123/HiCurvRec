# What to read when the run lands (the approved candidate, one cold Stage-2)

The run needs the operator's authorisation. When it finishes, the numbers below are the ones
that decide whether the candidate lives or dies — read them **in this order**, because the first
one is the one the ER cannot fix.

## 0. The L0 tail — read this FIRST, before the ER

The candidate's whole reason for existing is that the level-0 partition changes *without* the
behaviour signal coarsening it. The archived control (seed 42) is the comparison:

| quantity | the control (archived) | what would say the mechanism works | what would say it failed |
|:---|---:|---|---|
| L0 largest cell | **237** | ≤ 237 (the tail does not grow) | > 300 (the partition is coarsening) |
| L0 distinct codes used | **245** of 256 | ≥ 240 | < 230 (the tail is thinning) |
| L0 Gini | **0.2162** | ≤ 0.22 | > 0.30 |
| L1→L2 branches | **65.91** | ≥ 65 | materially lower |

Why first: the cohort's 14 gate-clearing arms all raised the ER and none improved downstream, and
the worst of them grew the L0 largest cell from 237 to 374 (Case A) or 382 (Case E). A candidate
whose tail grows has reproduced the campaign's central failure — a better ER and a worse
downstream — and the ER would hide it.

## 1. The ER, with BOTH shares

| quantity | the control | the gate |
|:---|---:|---:|
| Hard-L1 ER | 16.334505666275888 | **> 16.38953942232631** |
| numerator positive share | 0.07908428720083246 | record it |
| denominator negative share | 0.004841547630309337 | record it |

The shares are not optional: an ER that rises because the denominator fell is a different claim
from one that rises because the positives collide more, and the cohort's ER gains came from both.

## 2. The D2, as a diagnostic with its support

The conditional level-2 discrimination with its support and a `low_support` flag. In the cohort it
was null or thin for every arm, so it is a diagnostic, never a veto.

## 3. The G_H1 probe, and the occupancy it must be read at

The probe is a measurement on the arm's own export (`sids_for_hgrec.npy` ✓ — the probe refuses the
JSON export ✓). Report `G_H1` **together with** the L0 occupancy: the cohort's highest G_H1
(0.2895) came from an arm whose level-0 partition had collapsed to **two codes** ✓, so a G_H1
without the occupancy beside it is uninterpretable.

## 4. The deeper tree, because that is where the next family looks

| quantity | the control (measured on 4000 items) | what to look for |
|:---|---:|---|
| the fraction of (L1,L2) cells with a single L3 code | **0.9208** (92.08%) | a candidate that enriches the third token must move this |
| the mean L3 codes per (L1,L2) cell | 1.0969 | above 1.10 would be the first movement |
| the within-cell spread over the level-3 codebook's spacing | 0.7895 (Euclidean) | 1.4251 under a Möbius residual chain — the ratio the geometry moves |
| the level-wise residual removal | 61.1% / 26.1% / 20.2% | the chain's information budget |

These are the baselines the next family is measured against; they are not part of this run's
pass criterion.

## 5. The pass criterion, exactly one predicate

**Stage-3 runs only if the ER gate is cleared, and only then.** The scheduler enforces it: the
Stage-3 node depends on an `er_gate` node whose requirement is `hard_l1_er > 16.38953942232631`,
and the gate refuses to close without a measured value that clears it (a missing value, a failing
value and a value exactly at the threshold are all refused ✓).

The scientific success predicate is one function over the SAME arm: ER strictly above the
threshold **and** Recall@10 strictly above 0.06406796775709883 **and** NDCG@10 strictly above
0.03530651701940677, with complete audited artifacts. An ER-only win is not a win, and a
Stage-3-only win is not a win.

## 6. What is forbidden afterwards

No second run of this mechanism under any name, no second seed, no curvature or weight sweep, no
baseline or control re-run, and no Stage-3 before the ER gate. A failed arm is recorded and the
portfolio moves on.
