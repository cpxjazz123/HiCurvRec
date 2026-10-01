# Mechanism Pool — FCCR-1 Active Scope

This file is subordinate to `.claude/skills/curvature-rqvae-iter/SKILL.md`.

## Active contract

Current contract: **FCCR-1 — Fixed Closed-Form Curvature**.

The active question is:

[
(B_l, m_l^{raw}) → f(·) → c_l
]

with all (c_l) computed before Stage2 and fixed for the entire run.

## Current implementation

The working tree is a single directory, `stage2_RQ-VAE/curvature_RQ-VAE`,
holding a fixed-curvature Poincare RQ-VAE with
`LAYER_CURVATURES = (1.0, 1.0, 1.0)`.

## Open question

At the encoder's natural operating radius the ball geometry does essentially
nothing: measured on real encoder outputs, the expmap is linear to 1e-4, and
`d_poincare / d_euclidean` is a constant 2.0003 with std 0.0003, so every
curvature yields the same assignment. Enlarging the operating point to make
the expmap nonlinear is a dead end: forcing the latent onto a fixed radius
raised collision from 0.316 to 0.853 and had to be reverted, because the small
natural radius keeps the quantizer locally Euclidean where Sinkhorn is
well-conditioned.

So curvature is currently inert, and the obvious way to activate it destroys
the conditioning. A candidate must resolve that tension by some other route.

## Active candidate family

Only the following are active unless the user explicitly changes the contract:

1. **Alternative bounded closed-form mappings**
   - Change only the mathematical mapping (f(B_l, m_l^{raw})).
   - Curvature stays fixed, non-trainable, and time-invariant.
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

## Constraints learned from reverted work

These are properties of the current pipeline, not a history to re-read:

- A learnable per-layer scale absorbs a closed-form prior instead of
  expressing it.
- Better Stage2 proxies (Gini, collision, entropy) do not imply downstream
  gains; the decision is made on Stage3 alone.
- A small natural latent radius is load-bearing. Do not force a fixed one.
- Stacking two mechanisms at once destroys attribution. One change per
  condition.
- A mechanism tested with the wrong residual semantics yields a null result
  that says nothing about the mechanism.

## Deferred families

Not active under FCCR-1:

- learnable layer curvature;
- cyclic or scheduled curvature;
- curvature regularization;
- curvature-conditioned LR / AdamW beta2;
- curvature-dependent Sinkhorn redesign;
- curvature-dependent behavior-loss redesign;
- Riemannian optimizer as a new mechanism;
- manifold replacement, including switching to Lorentz or Möbius;
- mixed/product manifolds;
- Stage1 embedding changes;
- Stage3 trainer changes;
- forcing the latent onto a fixed large radius.

They may be reopened only if the user explicitly changes the research contract.

## Anti-pattern

Do not rotate through this pool simply because something has not yet been
tried. Candidate selection starts from the active contract and the open
question above.
