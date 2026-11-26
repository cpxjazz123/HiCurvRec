# Issue #267 contract (initial freeze)

The final mechanism contract is frozen in `CONTRACT_MECHANISM.md` after the root-cause
diagnosis. This file fixes the issue-level boundaries before implementation.

## Parent and non-negotiables

Parent: #265 R2 109, `TIGER_euclid_local_curve_supervision_rq`.
Frozen: 768-d semantic input, 32-d bottleneck, 3×256 codebooks, collision suffix,
Euclidean residual selection, parent floor/last-level Sinkhorn, three fixed curvature
experts {-1,0,+1}, center-local mixed gate distance, training-side curvature targets,
sampler protocol and Stage3 protocol. Frozen token/GNN assets are reused; no online
GNN forward/backward/optimizer.

The single new mechanism is a differentiable relative ranking term over same-centre
valid triples. No InfoNCE, edge prediction, prefix supervision, extra baseline or seed.

## Root-cause gate

Implementation starts only after `DIAGNOSIS.md` records: full q and z curvature
recomputation on 109, competing explanations, low-cost fixed-q/gate evidence, and
answers to the six training-admission questions. The ranking term must target the
diagnosed cause; a different loss without this evidence is not authorized.

## Score contract

`SCORE_CONTRACT.md` freezes the three-indicator post-hoc formula before Stage2 output;
the new arm is scored before Stage3/Recall is read. S is diagnostic only, not an
optimization target or an admission gate.

## Acceptance

C1–C9 in `RUN_LOG.md`. Negative ranking/Stage3 results are valid if the implementation,
protocol and recomputation are correct. Public Infra and historical assets must have
empty start/end hash diff.
