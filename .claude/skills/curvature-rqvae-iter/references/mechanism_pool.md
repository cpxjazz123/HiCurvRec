# Mechanism Pool — FCCR-1 Active Scope

This file is subordinate to `.claude/skills/curvature-rqvae-iter/SKILL.md`.

## Active research contract

Current contract: **FCCR-1 — Fixed Closed-Form Curvature**.

The active question is:

[
(B_l, m_l^{raw}) → f(·) → c_l
]

with all (c_l) computed before Stage2 and fixed for the entire run.

## Current implementation state

The working tree is a single directory, `stage2_RQ-VAE/curvature_RQ-VAE`,
holding a fixed-curvature Poincare RQ-VAE with
`LAYER_CURVATURES = (1.0, 1.0, 1.0)`.

Its measured weakness is the open question: at the encoder's natural operating
radius the ball geometry is a no-op (see the ledger), yet forcing a larger
radius is also known to fail. A candidate mechanism must address that tension
directly rather than re-testing either side of it.

## Active candidate family

Only the following are active unless the user explicitly changes the contract:

1. **Alternative bounded closed-form mappings**
   - Change only the mathematical mapping (f(B_l, m_l^{raw})).
   - Curvature remains fixed, non-trainable, and time-invariant.
   - No cyclic schedule, curvature regularization, optimizer-side curvature
     learning, or auxiliary curvature-learning loss.

2. **Per-layer curvature selection**
   - Choose different fixed (c_0, c_1, c_2) from a closed form of the
     pre-Stage2 statistics.
   - Still closed-form, non-trainable, and time-invariant, so it stays inside
     FCCR-1 while letting the per-layer budget act where it matters.

3. **Quantization-term shaping that does not touch curvature**
   - Reshape how the fixed-curvature distance enters the codebook or
     commitment term, leaving (c_l) untouched.
   - Admissible because the curvature contract is unchanged.

4. **Sensitivity/robustness checks of a validated mapping**
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
- manifold replacement (including switching to Lorentz or Möbius);
- mixed/product manifolds;
- Stage1 embedding changes;
- Stage3 trainer changes;
- forcing the latent onto a fixed large radius, which was tried and reverted.

They may be reopened only if the user explicitly changes the research contract.

## Historical experiments

Historical iterations remain useful as evidence, but they must not be
automatically inherited as mechanism parents.

Important observations:

- learnable/cyclic curvature can erase a closed-form prior;
- stronger Stage2 geometry proxies do not guarantee Stage3 improvement;
- optimizer-side curvature changes showed relative promise but are outside
  FCCR-1;
- Sinkhorn/behavior mechanism stacking created attribution ambiguity;
- some iterations tested the intended mechanism with the wrong residual
  semantics, so their null results are not evidence about the mechanism.

## Anti-pattern

Do not rotate through this pool simply because something has not yet been
tried. Candidate selection starts from the active contract and the unresolved
scientific question.
