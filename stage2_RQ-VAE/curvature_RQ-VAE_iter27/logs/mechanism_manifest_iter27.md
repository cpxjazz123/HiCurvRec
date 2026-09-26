# mechanism_manifest_iter27 (CAO-1)

## 1. Formula inputs

| Symbol | Semantic meaning | Source | Source iter | Raw value | Transformation | Transformed value | Range |
|---|---|---|---|---|---|---|---|
| `c_l(t)` | cyclic-learnable per-layer curvature (inherited from iter18) | `modules/quantize.py::get_c()` | iter18 | `[0.05, 0.90, 1.50]` at min; scaled by `c_layer_scale` and cyclic phase | `log_ratio * ((u_layer + g_t)/2)` inside `exp` | varies in `[0.05, 1.5]` per layer per step | finite > 0 |
| `v_k^old` | codebook tangent vector before AdamW step | `Quantize.embedding.weight.detach().clone()` | live | float32 `[K, embed_dim]` | none | tangent | finite |
| `v_k*` | codebook tangent vector after AdamW step | `Quantize.embedding.weight` post-`optimizer.step()` | live | float32 `[K, embed_dim]` | none | tangent | finite |
| `p_k^old`, `p_k*` | Poincaré-ball positions of `v_k^old`, `v_k*` | `_expmap0_t` from `modules/hyperbolic.py` | this iter | `float32 [K, embed_dim]` | `tanh(√c·‖v‖)/(√c·‖v‖) · v`, projected inside ball | inside Poincaré ball | `‖p‖ < 1/√c` |
| `δ_k` | per-codeword hyperbolic displacement | `_poincare_distance_t` | this iter | float scalar per `k` | `(2/√c) · atanh(√c · ‖(-p^old)⊕p*‖)` | non-negative scalar | `≥ 0` |
| `D_l` | layer-local geodesic spacing (median nearest-neighbor) | `_poincare_distance_t` pairwise | this iter | float scalar | `median_k min_{j≠k} d_c(p_k*, p_j*)`; floored at `1e-3` | non-negative scalar | `≥ 1e-3` |
| `τ_l` | trust radius | `0.5 · max(1e-3, D_l)` | this iter | float scalar | `τ_l = 0.5 · D_l` | non-negative scalar | `≥ 5e-4` |
| `s_k` | per-codeword scale factor | `min(1, τ_l / (δ_k + 1e-9))` | this iter | float scalar per `k` | elementwise | `[0, 1]` | `[0, 1]` |

## 2. Semantic / provenance notes

- The mechanism does **not** use `normalized_layer_scale` or any
  learnable `c_layer_scale` as a mechanism input; `c_layer_scale` is
  inherited from iter18 unchanged and is consumed via `get_c()`.  The
  mechanism's CAO-1 contract forbids adding a new trainable parameter.
- `raw_residual_median` and `behavior_branching` (FCCR-1 inputs) are
  **not** inputs to iter27.  They remain descriptive metrics only.
- The mechanism is **independent** of the assignment rule: Sinkhorn
  ε modulation from iter8 is retained and unchanged.
- `D_l` is computed from the post-step codebook, not from a fixed
  reference; this means `τ_l` adapts to the actual codebook geometry
  each step.

## 3. Hard-gate checklist (CAO-1)

- [x] **hyperbolic** term in manifest: present.
- [x] **trust region** term in manifest: present.
- [x] **codebook** term in manifest: present.
- [x] **poincaré** term in manifest: present.
- [x] **geodesic** term in manifest: present.
- [x] Mechanism touches `layer.embedding.weight` only (verified by
  AST inspection in `preflight_cao1.py`).
- [x] No new `nn.Parameter` introduced (preflight_cao1.py AST check).
- [x] No new auxiliary loss term introduced (preflight_cao1.py AST check).
- [x] No new assignment rule introduced (preflight_cao1.py AST check).

## 4. Constants exposed at the trainer level

```python
ITER27_TRUST_RADIUS_FRAC = 0.5    # τ_l = frac · median nearest-neighbor spacing
ITER27_TRUST_RADIUS_MIN  = 1e-3   # floor for degenerate layers
ITER27_TRUST_LOG_EVERY   = 1000   # global steps between summary prints
```

These are hard-coded in `curvature_RQ-VAE.py`; the bridge disallows
tuning them via CLI or env vars (CLAUDE.md §1).