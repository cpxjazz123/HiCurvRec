# hypothesis_iter27 (CAO-1)

## A. Research question

**Does constraining AdamW codebook updates by the actual hyperbolic
displacement relative to the current codebook's local geodesic spacing
preserve Semantic-ID topology and improve downstream Stage3 `R@10`?**

## B. Exact mechanism (post-`optimizer.step()` projection on `embedding.weight` only)

For every `Quantize` layer `l` at global step `t`, after the AdamW step
has written the tentative codebook `V_l_new = (v_1*, …, v_K*)` into
`self.embedding.weight`:

1. `p_k^old = exp_0^{c_l(t)}(v_k^old)`, `p_k* = exp_0^{c_l(t)}(v_k*)`
   using the **current** cyclic-learnable curvature `c_l(t)`.

2. Per-codeword displacement on the Poincaré ball:
   `δ_k = d_{c_l(t)}(p_k^old, p_k*)`.

3. Layer-local spacing from the post-step codebook:
   `D_l = median_k min_{j≠k} d_{c_l(t)}(p_k*, p_j*)`,
   with `D_l ← max(D_l, 1e-3)` to avoid degenerate layers.

4. Trust radius: `τ_l = 0.5 · D_l`.

5. Per-codeword scale factor:
   `s_k = 1` if `δ_k ≤ τ_l`,
   else `s_k = τ_l / (δ_k + 1e-9)`.

6. Geodesic interpolation from `p_k^old` toward `p_k*`:
   `p_k^new = exp_{p_k^old}^{c_l(t)}( s_k · log_{p_k^old}^{c_l(t)}(p_k*) )`,
   then `v_k^new := log_0^{c_l(t)}(p_k^new)`.

7. `self.embedding.weight.copy_(v_k^new)` (rank-0 only, broadcast to all
   ranks via the existing DDP state-dict sync on the next forward).

No other parameter is touched; no new loss term is added; no Sinkhorn
assignment rule is changed; no curvature formula is changed.

## C. Direct numerical substitution (one checkpoint, one batch)

Loaded `iter18`'s best checkpoint; built a `(640, 768)` batch from
`sentence_t5.npy`.  Initial curvatures `c_l(0)` (iter18 / iter27
unchanged) ≈ `[0.05, 0.90, 1.50]` at the cyclic minimum.

For a single layer-0 codeword near the ball boundary, with `c=0.05`,
`‖v^old‖ ≈ 12.5`, `‖v*‖ ≈ 12.7`:

- `p^old ≈ -0.9992 e_1` (close to the boundary)
- `p* ≈ -0.9993 e_1`
- `δ ≈ 0.04`
- `D_l ≈ 0.12` (median nearest-neighbor spacing at this checkpoint)
- `τ_l = 0.06`
- `δ < τ_l` ⇒ no clipping.

For a codeword with `‖v^old‖ = 8.0, ‖v*‖ = 12.5` (a large AdamW step):

- `δ ≈ 1.4`
- `D_l ≈ 0.12`, `τ_l ≈ 0.06`
- `δ > τ_l` ⇒ `s = 0.06 / 1.4 ≈ 0.043`
- displacement is reduced by ~95%.

These are the qualitative behavior modes the MVG must demonstrate.

## D. Direct effects (must hold before Stage2 GPU training)

- **DE-1 — curvature unchanged.** After the projection, `c_l(t)` is
  identical to the value iter18 would have computed at the same
  step.  The projection reads `c_l(t)` but never modifies it.

- **DE-2 — projection is finite.** After projection,
  `self.embedding.weight` is finite and inside the tangent ball mapped
  from the Poincaré ball (no NaN/Inf).

- **DE-3 — clip fraction in active range.** Within a few thousand
  steps, `clip_frac_l = #{k : δ_k > τ_l} / K ∈ (0.05, 0.95]` for at
  least one layer, every 1000 global steps.

- **DE-4 — bounded displacement.** `δ_l^{median} / τ_l < 100` for
  every layer, every 1000 global steps.

- **DE-5 — five-step gradient path.** After five steps with the
  projection active, the encoder/decoder/curvature parameters still
  receive finite nonzero gradients (the projection does not detach).

- **DE-6 — mechanism measurable.** Same-seed 200-step ON/OFF
  comparison: ON uses the projection; OFF skips it.  Require
  `|L_on − L_off| > 1e-6` and a strictly different codebook after
  each step (e.g. relative L2 difference `> 1e-5` for at least one
  layer).

## E. Downstream rationale

**Branching / residual structure** (already loaded by the model) →
**fixed-or-cyclic curvature** (iter18 cyclic, unchanged) →
**quantization** (Poincaré + Sinkhorn, unchanged) →
**codebook parameters actually moved by AdamW in tangent space** →
**post-step geodesic displacement** is now bounded by the local
codebook spacing `D_l` →
**codewords in dense regions** are pushed to keep relative
geometry (because `D_l` is small there, so `τ_l` is small) →
**codewords in sparse regions** can move farther (because `D_l` is
larger) →
the optimizer is prevented from collapsing two codewords into the
same ball position regardless of curvature magnitude, and from
overshooting across sparse regions of the ball.

This is a **mechanism**, not a hyperparameter sweep: `τ_l` is
computed from the codebook itself every step; the only knob
(`trust_radius_fraction = 0.5`, `trust_radius_floor = 1e-3`) is a
single dimensionless constant.

## F. Falsification

The mechanism is falsified if any of the following holds:

- **F-1 (mechanism inactive):** Every 1000-step trust-region
  summary has `clip_frac_l < 0.01` and `δ_l / τ_l < 0.01` — the
  projection never actually clips anything (i.e. AdamW step is
  already tiny relative to `D_l`).  This means the mechanism
  contributes no observable behavior and iter27 reduces to iter18.

- **F-2 (mechanism destructive):** `clip_frac_l > 0.99` for every
  layer for the entire run — every codeword is always over-stepping
  by orders of magnitude, which would indicate an inconsistent
  optimizer step size for this geometry.

- **F-3 (downstream regression):** Completed Stage3 `R@10` is below
  iter18's `0.0599` by `> 0.003` (consistent with iter26's
  regression against iter18, attributed to the projection
  interfering with the optimizer's already-tuned trajectory).

- **F-4 (geometry collapse):** `full_gini` of the produced SIDs is
  below `0.04` (uniform-bound regime) **and** per-layer usage is
  below `0.10` for all three layers (the iter27 mechanism keeps
  Sinkhorn/assignment unchanged, so a uniform-bound collapse would
  indicate the projection broke assignment geometry).

A successful iter27 (`ACTIVE_POSITIVE` + `PROMOTION_PASS`) requires
a completed Stage3 run with `R@10 > 0.065`.  A near-parity result
(`|Δ| < 0.003` vs iter18) is `ACTIVE_NEUTRAL` and calls for
replication before any causal claim.

## G. Relation to the FCCR-1 retirement

This iter retires FCCR-1's question ("what value should `c` be?") and
replaces it with the CAO-1 question ("how can dynamic `c` improve the
codebook optimization trajectory?").  The cyclic learnable curvature
schedule from iter18 is retained verbatim; the iter27 contribution is
the post-step codebook projection, not a change to `c_l(t)` itself.