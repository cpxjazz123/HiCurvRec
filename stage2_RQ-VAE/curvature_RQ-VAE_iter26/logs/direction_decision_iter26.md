# Iter26 Direction Decision

## Context (from iteration bridge)

- Iter25 ran the user-accepted bounded branching-plus-raw-residual layer-scale
  prior `[0.990000, 0.126233, 0.058186]` anchored to the existing learnable
  scales. MVG four-layer PASS; Stage2 100k steps completed; Stage3 test_R@10
  = 0.058828 (n_eval=57439). Adoption threshold 0.065 not met; Iter18 remains
  the best baseline at 0.059890.
- Iter25 sits slightly above Iter18 on every descriptive Stage2 metric
  (full_gini 0.0637 vs 0.0642, per_layer_mean 0.2002 vs 0.2028, collision
  6.6092% vs 6.6539%, unique 22962 vs 22951, H(L1|L0) 5.5874 vs 5.5907)
  but downstream regressed by 0.001062. The bridge labels this a class-3
  "geometry improved, downstream did not eat" pattern.
- The Iter25 bridge hypothesis: "When behavior branching and the raw residual
  medians (not the Iter1 calibration) are correctly fed into the closed-form
  mapping, the resulting fixed per-layer curvatures should improve downstream
  recall." Iter25 violated this hypothesis because the prior was anchored to a
  learnable scale and the closed-form formula was not actually evaluated.

## Candidate scorecard

Iter26 is a one-factor replication of Iter16 with the residual vector corrected
from the Iter1 calibration `[0.001, 0.932889, 1.0]` to the raw residual medians
`[1.0, 0.10941, 0.09331]`. There is no candidate selection; the "candidates" are
"reuse Iter16 with raw residual" (recommended) vs. "introduce another mechanism"
(forbidden by the bridge).

| Criterion | Iter16-fixed-with-raw-residual | Any-other-mechanism |
|---|---|---|
| (a) Explicit fix path for Iter25 regression | PASS — Iter25's `c` was allowed to drift via a learnable scale; Iter26 freezes it from a corrected closed-form mapping. One-factor diff isolates the residual-input correction. | FAIL — bridge explicitly forbids another behavior-loss change and another per-layer curvature novelty. |
| (b) 2023+ empirical evidence | PASS (adjacent) — Yang 2023 layer-wise κ; Welz 2025 δ-hyperbolicity init; Poincaré ResNet 2023 norm preservation; Kratsios 2023 tree capacity. None evaluates this exact RVQ closed-form. | n/a |
| (c) Novelty vs mechanisms tried | PASS — the exact "raw residual → closed-form → fixed `c`" combination is new; the formula is Iter16's, but the input vector was never tried. | FAIL — Iter25 already exhausted the learnable-scale variant of this family. |
| (d) Stage1-only curvature ceiling risk | PASS — Stage1 untouched. The bridge-mandated ceiling risk (Stage1-only curvature) is documented in `iteration_bridge.md` but does not apply to this one-factor replication. | n/a |
| (e) Gap-closing relevance (mandatory) | PASS — directly tests the bridge's stated research question: "when raw residual is correctly input to the closed-form, does fixed `c` improve downstream recall?" Any other candidate fails (e). | FAIL — cannot explain how it fixes the bridge hypothesis. |
| (f) Curvature relevance / keyword compliance | PASS — fixed per-layer κ_ℓ via closed-form mapping from data-derived branching and residual scales. | n/a |

## Recommendation

**Implement Iter26 = Iter16's exact closed-form implementation with the residual
input vector replaced by raw residual medians `[1.0, 0.10941, 0.09331]`.** This is
not a new mechanism — it is a one-factor replication. The cloned source must come
from `curvature_RQ-VAE_iter16/`, not from Iter25.

Concretely:

- `MECHANISM_NAME = "iter26_closed_form_with_raw_residual"` (new ID; keeps the
  closed-form fingerprint and disambiguates from Iter16's `iter16_closed_form_*`).
- `residual_norm` field in `scripts/computed_behavior_branching.json` becomes
  `[1.0, 0.10941, 0.09331]` (raw residual medians). The `branching`,
  `c_base=0.5`, `alpha=0.2`, `m_min=0.09331`, `closed_form_c_l` are
  recomputed from the corrected input and locked at
  `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`.
- No learnable `c_layer_scale`, no cyclic `c(t)`, no curvature regularization.
  The fixed curvature buffer is loaded once at module construction and never
  touched by the optimizer.
- MVG layer-1 must check `get_c()` at `step ∈ {0, 25_000, 50_000, 100_000}` and
  verify each layer matches the locked values within `1e-6`. Any drift is
  `CONTRACT FAIL` and Stage2 must not launch.
- Stage3 wrapper redirects `RQVAE_VARIANT = "iter26_closed_form_with_raw_residual"`
  and writes logs/ckpt to `results/stage3_T5Train/curvature_RQ-VAE_iter26/`.

## Rationale

1. **One-factor diff**: Iter25 vs Iter26 differs only in whether `c` is learnable
   vs fixed-from-corrected-input. If Iter26 still does not clear 0.065, the
   closed-form mapping is not the binding lever; that conclusion will be reached
   with much less confound than Iter25's learnable-scale artifact.
2. **Reproducibility**: the corrected `closed_form_c_l` values are exact to 1e-6
   and are written into the input JSON, not derived at runtime. Any audit can
   re-derive them from `branching` + `residual_norm_raw`.
3. **Anti-pattern compliance**: no model capacity change, no input pipeline change,
   no Stage3 trainer change, no novelty stacked on a previous variant. This is
   the cleanest possible Iter16 replication.

## Risks

- R1 (highest) — Persistent ceiling. Iter25 (class 3) and Iter16 (regression
  unknown; test_final.json missing) both stalled under the Stage3 trainer.
  Iter26 may also regress despite the corrected residual input.
- R2 — Iter16's downstream `test_R@10` is unavailable in the registry, so
  Iter26's comparison set is reduced to Iter11 / Iter18 / Iter25 / Iter24. The
  delta interpretation will require re-running Iter16's downstream evaluation
  if Iter26's `test_R@10` is near Iter18, to attribute the effect cleanly.
- R3 — Stage2 SID geometry is descriptive only; Iter26 may improve geometry
  without improving downstream recall. Adoption gate is Stage3 `test_R@10`,
  not Stage2 metrics.
- R4 — If Iter26 reaches the [0.060, 0.065) band but not 0.065, the ceiling
  hypothesis strengthens but is not closed. Iter27 may need a different
  Stage2 lever or move outside Stage2.
- R5 — Hard contract: if the curvature drift check at any of the four
  mandated steps detects a change > 1e-6, the run is aborted before
  Stage2 launches. This is intentionally stricter than the bridge's
  "Stage2 may run to MAX_GLOBAL_STEPS even if geometry looks off" rule,
  because the closed-form fixed-`c` mechanism only makes sense if `c`
  is genuinely fixed for the entire run.