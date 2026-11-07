# Issue #139 — final report (2026-11-07)

**Status: NOT complete.** Two of the five arms were trained to the full frozen
budget; the other three were not, because the work was stopped on the owner's
instruction. What is here is the mechanism, its verification, the two completed
arms, and the evidence that says what the mechanism does and does not do.

No claim in this report is a Stage 3 result: this issue is Stage 2 only, and no
arm of it has produced a recall number.

## 1. What was built

A candidate tree per arm, `stage2_RQ-VAE/TIGER_mixed_rq_{F0,G,L,E,A}`, each a
copy of the frozen baseline with one mechanism added and one constant changed:

| arm | space | weight policy | what it answers |
|---|---|---|---|
| F0 | E16 × H16, c=1 | fixed at ½ | what the mixed space alone does |
| G | E16 × H16, c=1 | one weight per level, chosen by the outer loop | whether a global geometry weight helps |
| L | E16 × H16, c=1 | one weight per prefix, chosen by the outer loop | whether local structure needs different weights |
| E | E16 × E16, c=0 | the same local outer loop | whether anything is gained beyond two weighted Euclidean blocks |
| A | E16 × H16, c=1 | fixed at ½, plus the existing AF3 auxiliary loss | whether choosing the metric beats adding the loss |

The mechanism, in one paragraph: every token is one **joint** codeword, 16
Euclidean coordinates and 16 tangent coordinates that the canonical Poincaré
contract maps into the ball, so a scalar weight still decides between two codes
whose halves disagree. The cost is
`C = 2[α‖rᴱ−eᴱ‖² + (1−α)d_c(rᴴ,eᴴ)²/4]`; the residual is native (a
subtraction on one half, a Möbius addition on the other); the readout is the
Euclidean sum and the right-nested ball sum; the assignment rule is the
baseline's, with this cost in place of the Euclidean one. The outer loop reads
the AF3 target `κ = 4 − d_i − d_j + 3T` of the training graph, scores the hard
SIDs it would produce under each candidate weight through a frozen shared
readout, and accepts a weight only when the structural distortion falls and the
feasible region holds.

## 2. What was verified

| surface | evidence |
|---|---|
| the cost equals the arc-cosh closed form at c=1 and c=0, for α ∈ {¼,½,¾} | `tests/test_mixed_rq.py` |
| `dC/dα = 2(E² − d²/4)` | measured numerically, 1e-9 |
| the weight reverses the order of two codes whose halves disagree (the issue's required counterexample) | α=¼ picks code 2, α=¾ picks code 1 |
| both halves compose back to the encoded point | left cancellation and right nesting |
| the c=0 arm reproduces a plain 32-dimensional residual search | token for token |
| the Bernoulli surrogate of the AF3 target equals the exact expectation over every subgraph of a four-node graph; binary p recovers κ exactly | `tests/test_af3_structure.py` |
| a local update of the structural distortion equals a full recomputation, pair by pair | `tests/test_af3_engine.py`, 1e-12 |
| the mechanism against an independently rebuilt reference | values 1.4e-17, gradients 1.1e-16, gradcheck on both sides |
| the machine admission | 11 of 11 checks pass on F0, G, E and A; L's entry check passes but its local path now fails closed (below) |
| the reviewed regression suites | 115 tests plus 24 subtests pass in each of the six trees (base, F0, G, L, E, A), from the review commit `3ee9374c` |

A later review (`3ee9374c`, on the remote) repaired the incremental degree
dependencies, the shared feasibility definitions, the TRAIN node mapping, the
deterministic support, the immutability of the F0 calibration, the alpha
inheritance, rollback and checkpoint state, and the rank-zero codebook
initialisation; it also removed the wall-clock prefix truncation this work had
introduced, as an unapproved policy, and made the explicit local entries fail
closed rather than invent a pruning rule. That review's state is what this
report describes.

Writing those tests found two errors in my own mathematics, both recorded in
the test docstrings: the arc-sinh form of the ball distance is wrong by a
normalisation (the arc-cosh form is the right one), and subtracting one from
each degree in the surrogate gives the *conditional* expectation rather than
the definition of κ.

## 3. The two completed arms

| arm | epochs | wall | collision | L1 hard ER | L1 pos hit | L1 neg hit | L2 prefixes | L3 triples | L3 collision | valid SID | outer-loop rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 3000 | 546 s | 0.6307 | 3.691 | **0.1571** | 0.0426 | 2,096 | 9,081 | 0.7672 | **False** | — |
| G | 3000 | 729 s | **0.5809** | 3.786 | 0.1472 | 0.0389 | 2,160 | 10,304 | **0.7321** | **False** | 1 accepted at 1500, 0 at 2000/2500/3000 |
| reference (frozen resample) | — | — | — | 16.390 | 0.0794 | 0.0048 | 16,145 | — | — | — | — |

The shared readout was frozen once, from F0 at epoch 1500: τ = 0.70150,
intercept = −0.08502, and the mean predicted edge probability equals the true
edge density to seven digits (0.33297294 against 0.33297297).

**The outer loop of G, in full:** round 0 at epoch 1500 accepted one weight
change and the distortion moved from 0.26226 by less than the fifth decimal;
rounds 1, 2 and 3 at epochs 2000, 2500 and 3000 accepted nothing, every
candidate being either infeasible or unable to change a single code. The
mechanism's leverage is therefore real but almost exhausted after the first
round.

## 4. What the evidence says

**The mixed space changes the behaviour separation and hurts the resolution.**
F0's first-level positive hit rate is 0.1571 against the reference's 0.0794 —
twice as good — and its hard ER is 3.69 against 16.39. But its third level
collides at 0.767 (the reference: 0.065) and its identifiers come out
**technically invalid** against the owner's learned collision gate of 0.35.
Both arms are therefore refused by that gate, and neither could have been
carried into Stage 3 in any case.

**The AF3-selected weight did not improve the behaviour separation.** G is
better than F0 on collision (0.5809 against 0.6307), on the third-level
collision (0.7321 against 0.7672) and on the number of second-level prefixes
(2,160 against 2,096), and worse on the first-level positive hit rate
(0.1472 against 0.1571) and on the hard ER (3.786 against 3.691). By the issue's
own classification this is **STRUCTURE_ONLY_OR_TRADEOFF at best, and
NO_VERIFIED_GAIN on the behaviour metric**: the geometry weight buys a little
resolution and costs the behaviour separation it was supposed to improve.

**Why the weight has so little to work with.** Near the origin the canonical
distance is twice the Euclidean one, so the two halves of the cost are nearly
proportional and the weight cannot separate the codes. That the weight is not
completely inert was measured rather than assumed: at the scale of the frozen
embeddings, changing it from ¼ to ¾ changes the hard codes of 12,909 of 24,556
items. But the same near-isometry is why the *deep* levels collapse: the mixed
space at this scale is close to a re-weighting of one Euclidean space, not a
different geometry.

**A limitation of the design, measured.** With the issue's activation rule (a
prefix needs at least 32 items and 64 observed and 64 unobserved probes), the
second level has 34 qualifying prefixes and the **third level has none** on a
real SID of this codebook shape. The third-level table would therefore fall
back to its nearest ancestor, so no arm of this design can have exercised a
full three-level local fit. That is recorded rather than glossed, and it is one
reason the L arm was worth running anyway: it is the only arm that could have
shown a local advantage, and its absence of qualifying prefixes means it could
not.

## 5. What was not done

- **L, E and A were not trained.** All three trees exist and are wired; the L
  dispatch was cancelled before its smoke finished. The issue's DoD ("five arms,
  each one complete 3000-epoch run with an exported hard SID") is therefore
  **not met**, and this report does not claim it.
- **The L arm cannot be dispatched as it stands.** The reviewed code rejects
  explicit local entries that it cannot serve, pending a fallback-transition
  specification that has been reviewed and approved. Until that exists, the arm
  that could have shown a local advantage is the one that fails closed.
- The wall-clock ceiling on one outer-loop round that this work had added was
  removed by the review as an unapproved policy: the round now processes every
  qualifying prefix, and the cost of that is a measurement the L arm has not
  yet paid.
- The differential check across three independent derivations was not run.
- The frozen-snapshot diagnostics of §3.1 (Forman instead of AF3, shuffled AF3,
  degree and triangle errors separately, and the shuffled local table) were not
  run.
- No arm entered Stage 3, so nothing here says anything about recall.
- The `TIGER_af3_hierarchy` arm's identifiers were overwritten by a run that
  used the wrong arm name; they were moved to `/tmp/af3h_quarantine/` and that
  arm has to be re-run to restore them. No Stage 3 result consumed them.
- The `TIGER_af3_triplet` arm, which belongs to an earlier issue, finished its
  Stage 3 at **test_recall@10 = 0.0551**, below the 0.065 line. It is recorded
  here only so that nothing about this session is lost: it is not part of this
  issue.

## 6. Where the evidence lives

- `results/research/issue139/precheck.json` — the measured graph, support, readout
  calibration, the distortion cost, the incremental engine's timings, the
  active-prefix counts and the budget verdict.
- `results/research/issue139/arms_summary.json` and `arms_table.md` — this table,
  regenerated by `collect_arms.py` from what the runs wrote.
- `results/research/issue139/reference_check_open.md` — the two defects the
  reference check found, with their numbers.
- `results/stage2_RQ-VAE/TIGER_mixed_rq_{F0,G}/` — the two runs' artifacts and
  their layered SID records.
- The candidate trees carry their own `math_spec.json` (frozen) and
  `reference_contract.json`.
