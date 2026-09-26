# Mechanism Pool — FCCR-1 Active Scope

This file is subordinate to `.claude/skills/curvature-rqvae-iter/SKILL.md`.

## Active research contract

Current contract: **FCCR-1 — Fixed Closed-Form Curvature**.

The active question is:

[
(B_l, m_l^{raw}) ightarrow f(cdot) ightarrow c_l
]

with all (c_l) computed before Stage2 and fixed for the entire run.

## Active candidate family

Only the following are active unless the user explicitly changes the contract:

1. **Alternative bounded closed-form mappings**
   - Change only the mathematical mapping (f(B_l,m_l^{raw})).
   - Curvature remains fixed, non-trainable, and time-invariant.
   - No cyclic schedule, curvature regularization, optimizer-side curvature learning, or auxiliary curvature-learning loss.

2. **Input-ablation variants of the same closed-form family**
   - branching-only;
   - raw-residual-only;
   - branching + raw residual.
   - These are allowed only as clean one-factor ablations under the same protocol.

3. **Sensitivity/robustness checks of a validated mapping**
   - Small bounded changes to a preregistered coefficient or transform.
   - Only after the base mapping receives a clean FCCR-1 run.

## Deferred families

These are not active candidates under FCCR-1:

- learnable layer curvature;
- cyclic or scheduled curvature;
- curvature regularization;
- curvature-conditioned LR / AdamW beta2;
- curvature-dependent Sinkhorn redesign;
- curvature-dependent behavior-loss redesign;
- Riemannian optimizer as a new mechanism;
- manifold replacement;
- mixed/product manifolds;
- Stage1 embedding changes;
- Stage3 trainer changes.

They may be reopened only if the user explicitly changes the research contract.

## Historical experiments

Historical iterations remain useful as evidence, but they must not be automatically inherited as mechanism parents.

Important observations:

- learnable/cyclic curvature can erase a closed-form prior;
- stronger Stage2 geometry proxies do not guarantee Stage3 improvement;
- optimizer-side curvature changes showed relative promise but are outside FCCR-1;
- Sinkhorn/behavior mechanism stacking created attribution ambiguity;
- Iter16 did not cleanly test branching + raw residual because it used the wrong residual semantics;
- Iter25 used correct raw residual values but implemented them as a learnable prior, so it was contract-invalid for the fixed-curvature hypothesis.

## Anti-pattern

Do not rotate through this pool simply because something has not yet been tried. Candidate selection starts from the active contract and unresolved scientific question.
