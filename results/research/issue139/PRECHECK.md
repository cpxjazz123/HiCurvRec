# Issue #139 — measured pre-check and the state of the five-arm design

> **Historical evidence only — RECHECK_REQUIRED (2026-11-07 repair).**
> This document predates the repaired graph mapping, structural cache, feasible
> region, shared calibration and complete outer-loop sweep. Its runnability and
> no-training statements below are historical, not the current state. The old
> JSON feasibility conclusion and prefix-level counts are invalid for the new
> code. See `REPAIR_REVIEW.md` and the updated `precheck.json`; no formal
> admission or real-data resource conclusion is granted by the CPU repair tests.

Written 2026-11-07, on `m3g110`, against the frozen data of the repository.
Everything below is measured, not estimated; the machine-readable form is
`precheck.json` next to this file.

## 1. What is implemented and verified

| Surface | State | Evidence |
|---|---|---|
| Native mixed codeword space (E16 × H16, one joint index, c=1) | implemented | `stage2_RQ-VAE/TIGER_mixed_rq/model/mixed_rq.py` |
| Joint cost `C = 2[α‖rE−eE‖² + (1−α) d_c(rH,eH)²/4]` | verified against the arc-cosh closed form at c=1 and c=0, for α ∈ {¼,½,¾} | `tests/test_mixed_rq.py` |
| `dC/dα = 2(E² − d_c²/4)` | measured numerically, matches to 1e-9 | same |
| Native residual, right-nested readout | left-cancellation and right-nested composition both hold | same |
| The α-weight reverses the order of two disagreeing codes (the required counterexample) | E cost [1,4], curved cost [4,1] → α=¼ picks code 2, α=¾ picks code 1 | same |
| c=0 arm reproduces a plain 32-dimensional nearest-code search | token-for-token identical | same |
| AF3 target `κ = 4 − d_i − d_j + 3T` | computed on the real graph (315,391 edges) | `model/af3_structure.py` |
| Bernoulli surrogate `μ̂ = p(2 − Σp_ik − Σp_jk + 3Σp_ik p_jk)` | **equals the exact expectation over all 2^6 subgraphs** of a four-node graph; binary p recovers κ exactly | `tests/test_af3_structure.py` |
| Frozen readout `p = σ(b − δ/τ)` with the intercept that matches the edge density | calibrates to `mean p = |E|/|S|` exactly (0.3328 = 0.3328) | same |

The two oracle errors found while writing these tests are recorded in the test
docstrings: the arc-sinh form of the ball distance is wrong by a normalisation,
and subtracting one from each degree gives the *conditional* form of the
surrogate rather than the definition of κ.

## 2. The measured cost of one outer-loop evaluation

| Quantity | Value |
|---|---|
| Training graph | 315,391 edges, 24,556 nodes |
| κ inter-quartile range (the Huber scale `s_κ`) | 239 |
| Support `S` | 947,197 pairs: 315,391 edges + 631,806 unobserved (316,415 semantic candidates + 315,391 degree-stratified) |
| Support build | 9.4 s |
| Readout calibration | 0.4 s |
| **One full `D_AF3` evaluation** | **12.2 s**, peak RSS 5.9 GB |

## 3. The budget verdict

The issue's own rule: the outer loop may not cost more than F0's full training
fit on the same hardware. The frozen fit measures **570 s** (3000 epochs, four
cards).

| Arm | Outer-loop evaluations | Measured cost | Verdict |
|---|---|---|---|
| F0 | none (α fixed at ½) | 0 | within budget |
| G | 4 snapshots × 3 candidates = 12 | ~2.5 min | within budget |
| A | none (α fixed at ½, AF3 enters as the existing auxiliary loss) | 0 | within budget |
| L | 4 snapshots × ~1,000 prefixes × 3 candidates ≈ 12,000 | ~40 h | **RESOURCE_BLOCKED**, 64× over |
| E | same as L | ~40 h | **RESOURCE_BLOCKED**, 64× over |

So three of the five arms are runnable today and two are not. The blocker is
specific and mechanical: the per-prefix loop needs the exact structural
distortion at roughly 50 ms per evaluation, and the current exact implementation
takes 12.2 s. The issue anticipates exactly this and offers two ways out —
build the exact incremental cache (accepting only pairs whose endpoints, or
whose triangle terms, can have changed, validated against the full recompute) or
revise the specification. Both are real work, and neither is a reason to reduce
the support, skip the cross-prefix terms or shorten the epochs: the issue
forbids all three.

## 4. What has *not* been done

- The five arms have **not** been trained. No Stage 2 run of this design exists.
- `math_spec.json` and `reference_contract.json` have not been frozen yet; the
  tests above are the evidence so far, not the frozen specification.
- The trainer in `TIGER_mixed_rq` still quantizes with the baseline's
  `RQLayer`: the mixed stack is implemented and tested as a module but is not
  wired into `train_rqvae.py`, and the α outer loop is not scheduled.

## 5. Approval record

The owner approved the #139 five-arm, Stage2-only, one-seed exception in this
session (2026-11-07); the exception is recorded in `CLAUDE.md` §19g. The
approval covers the five arms this issue lists and nothing else.

