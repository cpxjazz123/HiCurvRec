# failure_attribution_iter28

## Decision: **NO-GO** (R@10 = 0.0565 < iter18's 0.0599 by 0.0034)

- iter28 introduced the **Sinkhorn-weighted Riemannian EMA codebook update** (SREMA): codebook `embedding.weight` was excluded from AdamW and updated by a manifold-EMA step after every optimizer step, using the existing iter8 Sinkhorn assignment matrix and the cyclic learnable curvature `c_l(t)` from iter18.
- Mechanism was **active and geometrically correct** (MVG PASS: `frac_updated = 1.000`, `median_update ≈ 1e-3`, 200-step ON/OFF `loss_delta = 0.30`).
- Stage3 result: `R@10 = 0.0565`, below iter18's `0.0599` by `0.0034` (above noise band ~0.003).
- iter28 also underperforms iter26 (FCCR-1 R@10=0.0570) and iter25 (R@10=0.0588).  Even the iter26 fixed-curvature experiment produced better SIDs than iter28's SREMA.

## Per-mechanism diagnosis

- **Codebook excluded from AdamW (canonical, one-factor deviation)**: verified by iter28's custom check (`build_curvature_conditioned_adamw`).
- **Codebook moved by SREMA**: verified by MVG (every codeword updates every step, `saturation_max < 0.22`).
- **200-step ON/OFF trajectory divergence**: codebook max-abs-diff per layer = `[0.37, 0.18, 0.13]`.  **The codebook's evolution trajectory is genuinely different** from iter18's AdamW-only trajectory.
- **Stage2 SID statistics**: `full_gini = 0.0625` (vs iter18 `0.0642`, slightly **better**), `H(L1|L0) = 5.5462` (vs iter18 `5.5907`, slightly **worse**).  The descriptive Stage2 proxies do **not** predict downstream R@10 — per SKILL.md §11.4 this is expected.

## What the iter28 data does NOT support

- iter28 does **not** falsify the CAO-1 contract.
- iter28 does **not** falsify the iter18 baseline (iter18 still the canonical baseline).
- iter28 does **not** mean "SREMA is broken" — it ran correctly; the **interaction with iter18's other hyperparameters** is what produced a worse R@10.

## Hypothesized explanations (NOT proven)

1. **Codebook moves too fast**: SREMA's `frac_updated = 1.000` means every codeword shifts every step.  iter18's AdamW rarely moves most codewords (only the few with large gradients).  SREMA may be **over-smoothing** the codebook by spreading small updates across all 256 codewords per layer.
2. **EMA momentum mismatch**: `β = 0.99` per-step contribution is `(1-β) = 0.01`.  iter18's AdamW uses effective momentum from `β₂ = 0.999` (≈ factor `1/(1-β₂) ≈ 1000`).  SREMA's effective momentum is **100× smaller** than AdamW's, meaning the codebook tracks gradients too tightly and is pulled into every local optimum.
3. **Sinkhorn-EMA coupling breaks optimization dynamics**: the Sinkhorn assignment matrix `A` is itself updated by the Sinkhorn iteration in `forward()`, but with `detach()` (no gradient).  SREMA then uses a stale `A` to compute the EMA direction.  In contrast, AdamW uses the *fresh* gradient of the codebook loss through the assignment.
4. **Curvature changes in wrong phase**: SREMA reads `c_l(t)` which oscillates from 0.05 to 1.5 every 100k steps.  When `c_l ≈ 0.05` (low curvature), `1/√c ≈ 4.5` and codeword positions are near the boundary; small SREMA steps translate to large ball-position changes.  iter18's AdamW uses Euclidean space where the boundary doesn't exist.

## Carryover to iter29

- iter28's parent is **iter18** (NOT iter27 which was ABORTED).
- The "codebook lives on the ball" hypothesis is **partially refuted** at iter18's hyperparameters: SREMA is active but downstream R@10 falls.
- Recommended next directions (in priority order):
  1. **iter29: Direct Riemannian Adam Codebook** (user's pre-suggested direction) with explicit `exp_p` retraction and parallel-transport — keep the same CAO-1 contract but use a *true* Riemannian optimizer that AdamW matches in effective momentum (β₂ = 0.999 equivalent), not a manifold-EMA at β=0.99.
  2. **iter30: revert to iter18 and pivot** — try changing the *encoder* (e.g. add a Sinkhorn-ε schedule, increase the behavior-loss weight, change `c_cyclic_min/max/period`) rather than the codebook optimizer.
  3. **iter31 (if both above fail)**: replicate iter28 with a different seed; single-seed results below noise band should not be over-interpreted.

The iter27 (trust-region) and iter28 (SREMA) results jointly suggest: in this Stage2 protocol, **constraining or replacing the codebook's optimizer is not productive**.  Future iterations should target other axes (encoder, Sinkhorn ε, behavior loss, cyclic curvature range).