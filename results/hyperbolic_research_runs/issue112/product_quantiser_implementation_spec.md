# Implementation spec: the product-geodesic level-0 quantiser (path B)

The one approved candidate left (`hyp_product_l1_quantiser` v2, digest `b5bcfa8a…`; the pre-check
that cleared it: `product_quantiser_precheck.json`). This note is the executable spec, so the
implementation is one step, not a re-design.

## 1. What changes in the model, and what does not

The level-0 ASSIGNMENT changes; the rest of the quantiser stays the host's.

* the level-0 codebook keeps 256 entries of 32 tangent coordinates, split into the semantic half
  `A_j = a_j[:16]` and the behaviour half `B_j = a_j[16:]` (the proposal's fixed split);
* the assignment minimises `d_{c_s}^2(x_s, exp_0^{c_s}(A_j)) + d_{c_b}^2(x_b, exp_0^{c_b}(B_j))`
  with `c_s = 1`, `c_b = 2` and ONE common tangent scale `s = 1.4937731…` (measured, see
  `../subspace_111/latent_scale.json`);
* the quantised vector the layer returns is the codeword's own tangent coordinates, exactly as the
  host does — so the residual chain, the decode, the commitment loss and the reconstruction are
  untouched;
* therefore the reconstruction's level-0 contribution is `a_j` (the stored tangent values), which is
  what the host already reconstructs from.

**Nothing else in the model changes.** `model/` stays byte-identical to the host, as every arm in
this repository does; the change is a patch of the level-0 layer's `_distances` / `_indices`, the
same mechanism `install_nearest_assignment` uses (the arm's `scripts/hyp_losses.py`).

## 2. The patch (the arm's `scripts/hyp_losses.py`)

```python
def install_product_assignment(layer, c_s: float, c_b: float, scale: float) -> None:
    """The level-0 assignment minimises a sum of two factor geodesic distances."""
    original_distances = layer._distances
    original_indices = layer._indices
    layer._hyp_assignment_active = False

    def distances(self, latent):
        if not self._hyp_assignment_active:
            return original_distances(latent)
        half = latent.shape[1] // 2
        x_s = expmap0(scale * latent[:, :half], c_s)
        x_b = expmap0(scale * latent[:, half:], c_b)
        codes = self.get_code_embs()
        a_s = expmap0(codes[:, :half], c_s)
        a_b = expmap0(codes[:, half:], c_b)
        return (pair_poincare_distance(x_s, a_s, c_s) ** 2
                + pair_poincare_distance(x_b, a_b, c_b) ** 2)

    def indices(self, distance_matrix, infer_use_sk):
        if not self._hyp_assignment_active:
            return original_indices(distance_matrix, infer_use_sk)
        return distance_matrix.argmin(dim=-1)

    layer._distances = MethodType(distances, layer)
    layer._indices = MethodType(indices, layer)
```

The `pair_poincare_distance` the arm already has takes two point sets and returns the pairwise
matrix — confirm the argument order against `scripts/hyp_losses.py` before using it, and add a
regression test that the patched assignment equals a direct computation of the formula.

## 3. The CGV contract shape for a mechanism that changes the FORWARD

This is the part the R4-08/R10 families never needed, and the contract module has no template for
it, so it is written out here:

* `mechanism`: `type: "quantiser_metric"`, `executor`: a function that **returns the mechanism's
  own term** — for a path-B assignment change that term is `None` (the mechanism contributes
  nothing additively; its effect is inside the base loss's graph), and `executor_style:
  "returns_term"` so the adapter does not treat it as a legacy gradient-assembling executor;
* `forward_contract.declared_effect`: `"changes_forward"` — the assignment IS the forward's change,
  and the forward gate must see it;
* `gradient_contract`: `target: "codebook"`, `mode: "invariants"`, `loss_terms` the host's three,
  `expected_paths` the host's (base nonzero, behaviour zero), `rule: "project_l1_against_base"` with
  `reference_term: "base_loss"` — because the arm is the R4-08 host and keeps the host's routing;
* `mechanism.host_routing`: `"_r4_08_apply_projection"` — the arm's real assembly is base + the
  host's projection + the mechanism's assignment, and the declared assembly must say so (F5);
* `runtime_contract.observe`: `mechanism_calls` with the target the trainer's own symbol.

The G2/G3 gates then measure the base path reaching the codebook **through the changed
assignment** ✓ — and the *mechanism-specific* check that the assignment really changed is the
pre-check's own comparison (the assignment change rate against the host's), which becomes a
mechanism test rather than a gate.

## 4. The checks the implementation must satisfy

| # | check | where it lives |
|:--:|:---|:---|
| 1 | the patched assignment equals a direct computation of the product formula | a new test in the arm's tree |
| 2 | the assignment change rate against the host's Euclidean assignment is the measured 0.1935 (within 1e-3) | the mechanism test `p1_joint_assignment_not_inert` |
| 3 | the joint assignment agrees with neither factor alone for ≥ 1% of a real batch (the declared falsifier) | `p1_joint_assignment_not_inert` |
| 4 | the codebook's gradient reaches the codebook and is finite and nonzero | the CGV's G2/G3 + the mechanism test `p1_nonzero_gradient` |
| 5 | the four ranks hold bitwise identical parameters after a step, and the pooled codebook gradient matches the independent pooling | the CGV's G5 (the repaired one) |
| 6 | the domain guard: a point with `c_g‖x_g‖² ≥ 1 − margin` is skipped and counted | the mechanism's own guard (F3) |
| 7 | the exported SID has three tokens per item, each in [0, 255] | the mechanism test `p1_export_shape` |

## 5. What would make this candidate fail, before any GPU time

* check 2 or 3 failing on a real batch ⇒ the split is a rotation and the family is closed;
* the trained L0 tail (largest cell, distinct codes) worsening against the control's ⇒ the
  mechanism's own reason for existing (a finer L0 partition) is gone;
* the mechanism's gradient reaching the encoder ⇒ the factor mask is not in place and the content
  partition can be coarsened by the behaviour term, which is exactly the cohort's failure mode.
