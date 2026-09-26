# mechanism_manifest_iter28 (CAO-1 SREMA)

## 1. Formula inputs

| Symbol | Semantic meaning | Source | Source iter | Raw value | Transformation | Transformed value | Range |
|---|---|---|---|---|---|---|---|
| `c_l(t)` | cyclic-learnable per-layer curvature (inherited) | `Quantize.get_c()` | iter18 | varies | iter18 cyclic schedule | varies in `[0.05, 1.5]` | finite > 0 |
| `x ∈ R^{B×D}` | query residual (= layer input) | `Quantize.forward(x)` | this iter | tangent `‖x‖ ≈ 0.5–2` | none | tangent | finite |
| `A ∈ R^{B×K}` | Sinkhorn assignment matrix | `_sinkhorn_algorithm` | iter8 | softmax over `-d/ε` | ε ∝ c/c_max | `Σ_i A_ij ≈ 1/K` | `[0, 1]` |
| `w_old ∈ R^{K×D}` | codebook tangent parameter | `Quantize.embedding.weight` | this iter | float32 | none | tangent | finite |
| `p_old = exp_0^{c_l}(w_old)` | codebook Poincaré ball position | `_expmap0_t` | this iter | `‖p‖ < 1/√c` | `tanh(√c·‖w‖) / (√c·‖w‖) · w` | inside ball | `‖p‖ < 1/√c` |
| `z = exp_0^{c_l}(x)` | residual ball position | `_expmap0_t` | this iter | `‖z‖ < 1/√c` | same | inside ball | finite |
| `log_{p_old}(z)` | geodesic log at basepoint `p_old` | explicit closed-form | this iter | vector `K×B×D` | `2/(1-c‖p‖²) · atanh(√c·‖(-p)⊕_c z‖) · direction` | tangent vector at `p_old` | finite |
| `ξ ∈ R^{K×D}` | weighted geodesic direction | `Σ_i A_ij · log_{p_old}(z_i) / (Σ_i A_ij + ε)` | this iter | float32 | Sinkhorn-weighted sum | finite | finite |
| `β` | EMA momentum | `ITER28_SREMA_BETA` | this iter | scalar | none | `0.99` | `[0, 1]` |
| `p_new = exp_{p_old}((1-β)·ξ)` | new ball position | this iter | scalar step | tangent step | exp + projection | inside ball | `‖p‖ < 1/√c` |
| `w_new = log_0^{c_l}(p_new)` | new tangent parameter | this iter | `K×D` | none | log_0 | tangent | finite |

## 2. Semantic / provenance notes

- `c_l(t)` is **not** a new mechanism input; it is the iter18 cyclic-learnable curvature schedule consumed unchanged by SREMA via `Quantize.get_c()`.
- `A` is the iter8 Sinkhorn assignment (ε ∝ c/c_max from iter8); not modified.
- `w_old` is the iter18 codebook parameter (warm-started from iter8's checkpoint); not modified by AdamW in iter28.
- `log_{p_old}(z)` is a NEW computation (iter28); it is the closed-form geodesic log at an arbitrary basepoint on the Poincaré ball.  The hyperbolic module exposes `_logmap0_t` (origin basepoint only) and `_mobius_add_t`; iter28 inlines the closed-form `log_p` because no module function exists yet.
- The manifold EMA step `exp_{p_old}((1-β)·ξ)` is approximated by **tangent-space interpolation** `w_new = w_old + (1-β)·ξ` followed by `exp_0 → log_0` projection (preserves ball constraint and stays inside the ball).  This is first-order-correct; a true `exp_{p_old}` would require parallel transport `PT_{0→p_old}` first.  Tradeoff: simpler, faster, equivalent up to first order.

## 3. Hard-gate checklist (CAO-1)

- [x] **hyperbolic** term: present.
- [x] **trust region / SREMA** term: present (`srema_update`).
- [x] **codebook** term: present (`embedding.weight`).
- [x] **poincaré** term: present.
- [x] **geodesic** term: present (`log_{p_old}`).
- [x] Mechanism touches `layer.embedding.weight` only.
- [x] No new `nn.Parameter` introduced (preflight_cao1 AST check).
- [x] No new auxiliary loss term introduced (preflight_cao1 AST check).
- [x] No new assignment rule introduced (iter8 Sinkhorn is unchanged).

## 4. Constants exposed at the trainer level

```python
ITER28_SREMA_BETA = 0.99    # EMA momentum; per-step contribution = 1 - β
ITER28_SREMA_LOG_EVERY = 1000   # global steps between SREMA summary prints
```

Hard-coded in `curvature_RQ-VAE.py`.  The bridge disallows CLI/env override (CLAUDE.md §1).