# Boundary causal audit (issue #79) — verdict: INCONCLUSIVE, control parity not established

This audit was asked to identify which gradient updates cross L1 quantization boundaries and
destroy SID discrimination, by branching controlled interventions off the SAME frozen pre-step
state of each arm. It does not deliver that identification, and the reason is stated rather
than worked around: **the audit's own machinery does not reproduce the independent training
run**, so every branch contrast it produced is inadmissible.

## What was built and verified

- `tools/stage2_boundary_causal_audit.py`: quantiser-boundary primitives, complete frozen
  state capture/restore (weights, buffers, codebooks, optimizer moments, scheduler, RNG),
  the CTRL / FULL / NO-DEEP / BASE-ONLY branches, and the adjudication rules.
- Interventions remove a loss contribution BEFORE the arm's own gradient assembly and let the
  arm recompute. No branch subtracts a gradient after clipping, so the PCGrad assembly in R4-08
  and the soft-surrogate VJP in R6-04 are traversed as they are, not approximated.
- `tests/test_stage2_boundary_causal_audit.py`: 36 analytic checks, exit 0 — two-codebook
  boundaries with known answers, z-only vs c-only vs joint motion, exact ties, the interaction
  term, switch counting and margin quantiles, usage and collision, and parity with the
  trainer's native `get_indices`.

## The control branch failed

The audit's frozen origin must reproduce the independent probe v2 run on the same arm, protocol
and seed at the fork steps. It does not:

| arm | fork step | audit L1 used | probe L1 used | audit 3-level collision | probe 3-level collision |
|:--|--:|--:|--:|--:|--:|
| `r4_08` | 96 | 130 | 131 | 0.698540 | 0.698499 |
| `r4_08` | 160 | 118 | 116 | 0.447269 | 0.439135 |
| `r6_02` | 96 | 116 | 110 | 0.790580 | 0.791394 |
| `r6_02` | 160 | 75 | 65 | 0.727864 | 0.744174 |
| `r6_03` | 96 | 115 | 112 | 0.812503 | 0.816448 |
| `r6_03` | 160 | 72 | 72 | 0.767519 | 0.766584 |
| `r6_04` | 96 | 140 | 128 | 0.913003 | 0.916094 |
| `r6_04` | 160 | 62 | 54 | 0.916297 | 0.916460 |

The deviation is far larger than floating-point noise — a handful of L1 codes, and collision
differences in the third decimal — so it is a different model state, not rounding. The audit's
measurement forwards, state capture/restore and branch stepping therefore perturb the run
they are supposed to observe.

Per this issue's own instruction — *"If this reproduction fails, report `invalid`; no causal
claim or misleading charts"* — the verdict for all four arms is **inconclusive** and
`no_causal_claim = true`. The measured FULL-versus-NO-DEEP numbers are retained in each
`report.json` under `comparison`, but they are not evidence and must not be read as such.

## What is and is not established

- **Established:** L2/L3 behaviour gradients can be removed at the loss level and the arm
  recomputes its own gradient correctly for that altered objective; the boundary primitives
  are exact and agree with native inference.
- **Not established:** whether those gradients cause L1 boundary crossings, because the
  measurement cannot yet be shown not to change the run it measures.
- **Unchanged:** the question in this issue stands. R4-08's healthy 314.7x gradient ratio is
  not explained by magnitude alone, and no claim is made about what does explain it.

## The smallest falsifiable next step

The blocking defect is mechanical, not scientific: the audit must be shown not to perturb the
run before any intervention is worth measuring again. Concretely, the next attempt should
branch WITHOUT advancing the training loop — replay each intervention from the frozen state
on the captured minibatch through the arm's own loss, clipping and optimizer path — so that
the continuing trajectory is untouched by construction rather than restored after the fact.
Until control parity holds, no R7 mechanism should be designed on the strength of a branch
contrast from this harness.

## Integrity

- No production artifact changed: `production_artifacts_unchanged` is true for every arm, and
  the frozen held-out pair artifact was hashed before and after and never read as data.
- No 3000-epoch training and no Stage3 run were launched.
- The frozen Stage3 Hard-L1 ER threshold `> 16.38953942232631` is untouched.
