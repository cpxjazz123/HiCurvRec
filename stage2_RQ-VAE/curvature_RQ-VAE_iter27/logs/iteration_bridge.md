# Iteration bridge — Iter27 → Iter28

## Status

**Iter27 ended as `ITERATION_ABORTED_INFEASIBLE`** at MVG round 1
(S08_MVG), before any Stage2 GPU training.  See
`logs/iteration_abort_iter27.md` for the canonical abort artifact.

## Why iter27 was aborted

The user-specified mechanism (post-`optimizer.step()` hyperbolic
codebook trust-region with `τ_l = D_l / 2`) is **geometrically defined
but operationally inactive** under the iter18 ckpt regime:

| `trust_frac` | avg clip_frac (L0/L1/L2) |
|---:|---:|
| 0.5 (registered) | 0.000 / 0.000 / 0.000 |
| 0.1 | 0.000 / 0.000 / 0.000 |
| 0.05 | 0.000 / 0.008 / 0.065 |
| 0.02 | 0.072 / 0.445 / 0.914 |

The per-step AdamW displacement is `~0.01`, the registered trust radius
is `~0.06–0.28`.  Activating the mechanism requires tightening the
registered constant, which violates SKILL.md §2.9 "registered-spec
immutability after hypothesis lock".

## Carryover to iter28

- **Parent**: iter18 (`R@10 = 0.05988962203380978`).
- **Active contract**: CAO-1 (continues from iter27).
- **Mechanism code reused**: `Quantize.apply_hyperbolic_trust_region`,
  `RqVae.snapshot_codebooks`, `RqVae.apply_hyperbolic_trust_region`,
  and the trainer post-step wiring block (kept verbatim in
  `stage2_RQ-VAE/curvature_RQ-VAE_iter27/`, may be copied or imported
  by iter28).
- **Mechanism constant to re-register**: `trust_radius_fraction ∈
  {0.05, 0.02, 0.01}` (chosen by iter28's own S00–S08 deliberation).
- **Curvature schedule**: iter18's cyclic learnable `c_l(t)` continues.

## Forbidden in iter28 (carried forward)

- Do NOT submit another fixed closed-form curvature experiment (FCCR-1).
- Do NOT silently inherit iter27's `trust_radius_fraction = 0.5`.
- Do NOT modify Stage1 embeddings or Stage3 trainer.
- Do NOT stack a second new mechanism on top of the trust-region.

## Post iter26 → iter27 (aborted)

### Evidence

- iter26 used FCCR-1 fixed `c_l = [0.6145, 0.5333, 0.3814]`; contract
  met; geometry aligned; `R@10 = 0.057017` (NO-GO).
- iter11/iter18/iter25/iter24 all above iter26 → bottleneck is in
  **how codebook parameters move under AdamW given any `c`**, not in
  `c` itself.
- iter27's hyperbolic codebook trust-region (CAO-1) is geometrically
  correct but operationally inactive at the user-fixed `τ_l = D_l / 2`.

### Forbidden next directions (carried into iter28)

- Do NOT submit another fixed closed-form curvature experiment (FCCR-1).
- Do NOT submit another `c → AdamW hyperparameter` iteration.
- Do NOT modify Stage1/Stage3.
- Do NOT inherit iter27's mechanism with `trust_radius_fraction = 0.5`.