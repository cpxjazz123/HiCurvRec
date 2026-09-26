# hypothesis_iter28 (CAO-1 SREMA)

## A. Research question

**Can replacing AdamW's Euclidean codebook update with a Sinkhorn-weighted Riemannian EMA (SREMA) — computed entirely on the Poincaré ball — improve downstream Stage3 R@10 over iter18?**

## B. Exact mechanism

For every `Quantize` layer `l`, after the (non-codebook) AdamW step has run:

1. From the most recent forward(), retrieve the cached Sinkhorn assignment matrix `A ∈ R^{B×K}` and query residual `x ∈ R^{B×D}`.
2. Read cyclic-learnable curvature `c_l(t)` (inherited from iter18 unchanged).
3. Map codewords and residuals to the Poincaré ball:
   - `p_j_old = exp_0^{c_l}(w_j_old)` for each codeword `j`.
   - `z_i = exp_0^{c_l}(x_i)` for each residual `i`.
4. Compute per-codeword geodesic log from `p_j_old` to each `z_i` on the ball:
   - `log_{p_j_old}(z_i) = (2 / (1 - c_l ‖p_j_old‖²)) · atanh(√c_l · ‖(-p_j_old) ⊕_{c_l} z_i‖) · ((-p_j_old) ⊕_{c_l} z_i / ‖(-p_j_old) ⊕_{c_l} z_i‖)`.
5. Sinkhorn-weighted geodesic direction:
   - `ξ_j = Σ_i A_ij · log_{p_j_old}(z_i) / (Σ_i A_ij + ε)`.
6. Manifold EMA step:
   - `p_j_new = exp_{p_j_old}^{c_l}((1 - β) · ξ_j)`.
7. Write back to `embedding.weight` (tangent parameter):
   - `w_j_new = log_0^{c_l}(p_j_new)`.

The implementation projects the new tangent through `exp_0 → log_0` round-trip at the very end (`w_new + step → exp_0(w_new + step) → log_0(...)`); this preserves the ball constraint and produces a valid tangent parameter for the next forward's `exp_0^{c_l}(embedding.weight)`.

β is fixed at `0.99` for the entire run (registered constant).

## C. Numerical substitution (pre-Stage2)

Using iter18's warm-start as the initial codebook and a hypothetical batch of 640 residuals with `c_l(t) = 0.5`:

- `w_j_old` ranges in `‖w‖ ≈ 1.0–2.0`; `p_j_old` in `‖p‖ ≈ 0.5–0.8` (well inside `1/√c = 1.414`).
- Sinkhorn `A_ij` is `≈ 1/K = 0.0039` for a uniform assignment; `A.argmax() ≈ 1.0` for peaked assignment.
- For a peaked assignment on residual `i = i0` and codeword `j = j0`, the displacement `ξ_{j0}` is approximately `log_{p_{j0}}(z_{i0})` (a single dominant direction).
- `(1 - β) · ξ = 0.01 · ξ`; with `‖ξ‖ ≈ 0.05` (typical for `‖log‖` on a `‖p‖ = 0.5` codeword), `‖(1-β)·ξ‖ ≈ 5e-4`.
- After `exp_0 → log_0`, the new codeword is at distance `‖Δ‖ ≈ 5e-4` from the old one.  Over 100k steps with EMA this accumulates to `1 - 0.99^100000 ≈ 1.0` of any reachable direction — **the codebook is free to move** if the data wants it to.

The mechanism is not a "safety bound" (unlike iter27's trust-region): it is the actual codebook optimizer, run on the manifold with the correct geometry.

## D. Direct effects (must hold before Stage2 GPU training)

- **DE-1 — codebook excluded from AdamW**: optimizer's parameter groups must not include any `layers.X.embedding.weight`.  Verified by custom iter28 check in `build_curvature_conditioned_adamw`.
- **DE-2 — SREMA produces finite output**: `srema_update()` returns finite `median_update`, `max_update`, `frac_updated` for every layer; `saturation_max ∈ [0, 1)` (no ball-boundary divergence).
- **DE-3 — codebook measurably moves**: `median_update > 1e-9` for at least one layer per 1000-step window (after warm-start), and `frac_updated > 0` for that layer.
- **DE-4 — mechanism is active vs OFF**: same-seed 200-step ON (SREMA) vs OFF (AdamW-codebook) counterfactual: `|L_on − L_off| > 1e-6` AND codebook max-abs-diff `> 1e-7` per layer.
- **DE-5 — 5-step gradient path**: total loss requires grad; non-codebook parameters receive finite nonzero gradients; codebook gradients are not required (SREMA doesn't use them).
- **DE-6 — curvature unchanged**: cyclic-learnable curvature schedule from iter18 is byte-identical; `get_c()` snapshots at step 0/25k/50k/100k match iter18's trajectory.

## E. Downstream rationale

```
branching / residual structure (loaded by iter18 mechanism)
  → c_l(t) (iter18 cyclic learnable curvature, unchanged)
    → Sinkhorn assignment A_ij (iter18 ε ∝ c/c_max, unchanged)
      → SREMA: ξ_j = Sinkhorn-weighted geodesic log direction
        → manifold-EMA: p_j_new = exp_{p_j_old}((1-β) · ξ_j)
          → codebook moves ON the Poincaré ball, not through tangent + AdamW
            → Semantic IDs reflect the actual Riemannian geometry
              → Stage3 HG-Rec reads the resulting 4-token SIDs
                → R@10 reflects whether "codebook lives on the ball" helps
```

This mechanism is a **mechanism**, not a hyperparameter sweep: the only registered constant is `β = 0.99` (one dimensionless number).  Everything else (curvature, Sinkhorn ε, warm-start, training schedule, Stage3 config) is inherited verbatim.

## F. Falsification

The mechanism is falsified if:

- **F-1 (inactive)**: after warm-start, every 1000-step window has `median_update < 1e-12` for all layers → codebook never moves → iter28 ≡ iter18.
- **F-2 (destructive saturation)**: `saturation_max ≥ 1` for any layer for the entire run → codeword escapes the Poincaré ball.
- **F-3 (downstream regression)**: completed Stage3 `R@10` is below iter18's `0.0599` by `> 0.003` (a single-step miss within noise band is `ACTIVE_NEUTRAL`, not a falsification).
- **F-4 (geometry collapse)**: `full_gini < 0.04` and per-layer usage `< 0.10` for all three layers → SREMA broke assignment geometry.

A successful iter28 (`ACTIVE_POSITIVE + PROMOTION_PASS`) requires a completed Stage3 run with `R@10 > 0.065` AND F-1/F-2/F-4 not violated.  Per SKILL.md §2.10, if any F-x is violated at MVG/Stage2, the iteration is aborted autonomously — no user decision needed.

## G. Relation to iter27 (aborted)

iter27 was a "limit AdamW step" mechanism; iter28 is "remove AdamW from codebook entirely".  iter27 failed F-1 under the registered constant; iter28 succeeds F-1 by construction (the codebook is updated by SREMA, period).  iter28's failure modes are F-2 (escape) and F-4 (collapse), neither of which iter27 had.

## H. Falsifiability scope

This iter tests **one** registered mechanism (SREMA) under **one** registered constant (`β = 0.99`).  If Stage3 is near-parity with iter18, the conclusion is `ACTIVE_NEUTRAL` — a single seed is not a discovery; replication is required before any causal claim.