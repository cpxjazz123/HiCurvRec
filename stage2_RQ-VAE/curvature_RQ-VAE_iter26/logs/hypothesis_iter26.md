# Iter26 Falsifiable Hypothesis

## Mechanism

Iter26 reuses Iter16's exact closed-form fixed curvature implementation verbatim
except for the `residual_norm` input vector:

```text
m_raw      = [1.0, 0.10941, 0.09331]
branching  = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
c_base     = 0.5
alpha      = 0.2

s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))
z_l = (s_l - mean(s_l)) / (std(s_l) + 1e-12)
c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)
```

Computed values, locked into `scripts/computed_behavior_branching.json`:

```text
m_min = 0.09331
s_l = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z_l = [1.03129493309937,   0.3223997103378134,  -1.3536946434371857]
c_l = [0.6145357379232853, 0.5333020920777128,  0.3814078098431606]
```

`c_l` is fixed at construction time (registered as a non-trainable buffer, exactly
as Iter16). There is no learnable `c_layer_scale`, no cyclic `c(t)`, no curvature
regularization. `MECHANISM_NAME = "iter26_closed_form_with_raw_residual"`. Warm-start
excludes only `layers.{0,1,2}._fixed_c` so the buffers keep the locked values.

## Direct effects — mechanism activation (MANDATORY before Stage2)

All of the following are deterministic acceptance checks, not SID-quality gates.
A failure in any one is `CONTRACT FAIL` and Stage2 must not launch.

1. `c_layer_norm` field is replaced by `_fixed_c` buffer; `c_layer_scale` parameter
   is absent; `c_cyclic_*` constants are unused.
2. `requires_grad` of `c_layer_scale` (if present) is False; the optimizer's
   parameter list does not contain any curvature buffer.
3. `get_c()` at `step ∈ {0, 25_000, 50_000, 100_000}` returns
   `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` within
   `1e-6` absolute error on every layer. Any deviation is `CONTRACT FAIL`.
4. `total_loss.requires_grad` and `total_loss.grad_fn` are both true; all four
   loss components (reconstruction, RQ-VAE/commitment, behavior, curvature
   regularization) are finite. `curvature_regularization` is identically zero
   (fixed c → reg is zero by construction).
5. After five optimizer steps the encoder/codebook parameters update by
   relative magnitude > 1e-7. The curvature buffers do not change.
6. With identical seed, batch, and warm-start weights, a 200-step ON vs OFF
   arm (OFF = the canonical Iter11 fallback prior `[0.001, 0.932889, 1.0]`)
   produces finite terminal losses and `abs(L_on - L_off) > 1e-6`. The
   `c_l` value at step 200 must still equal the locked values to `1e-6`.

## Proxy hypothesis (descriptive only)

Iter26 has no direct downstream gain guarantee because Iter16's downstream
`test_R@10` is missing from the registry. The proxy direction is:

- The 3-token SID collision rate after 100k steps should not exceed the
  ceiling that caused Iter25's regression. Compare against the Iter18 baseline
  (6.6539%) and Iter25 (6.6092%) descriptively only.
- L0 utilization should remain high (>= 0.98 on every layer across 100k
  steps).
- H(L1|L0) is reported by the existing `sid_quality` collector; expected to
  match Iter16's fingerprint within ±0.2 bits.

These proxies do not gate the run. They cannot promote this iteration; Stage3
`test_R@10 > 0.065` is the only adoption criterion.

## Downstream criterion

Complete Stage3 with the exported Iter26 SID JSON. Report `test_R@10` (and
`valid_ndcg@20` for completeness) and compare to:

| Comparator | test_R@10 |
|---|---:|
| canonical Iter11 | 0.0597677536 |
| best completed Iter18 | 0.059890 |
| completed Iter25 | 0.058828 |
| completed Iter24 | 0.056199 |

The adoption target is `test_R@10 > 0.065`; the headline comparison is against
canonical Iter11 + best completed Iter18 (the two strongest completed baselines).
If Iter26 lands strictly above `(Iter11 + Iter18) / 2 = 0.0598` and into the
`[0.060, 0.065)` band, the closed-form mapping is a partial fix and warrants
parameter sweeps. If Iter26 regresses like Iter25, the closed-form mapping is
not the binding lever and Iter27 should move outside Stage2.

## Confounds

- The corrected `c_l` values are derived from the raw residual medians
  `[1.0, 0.10941, 0.09331]` and the canonical branching `[19.3249, 1.46057,
  1.01481]`. Any future input re-measurement that changes either vector
  changes the output.
- The MVG layer-1 contract requires `c_l` to be locked across the entire run.
  This is intentional. A mechanism that allows even a 1e-6 drift would no
  longer be "fixed closed-form curvature" and would be a different
  experiment.
- Iter26 is a one-factor replication of Iter16. Iter16's downstream
  `test_R@10` is not in the registry; this iteration does not produce a
  clean Iter16 → Iter26 delta. Iter27 (if needed) should re-run Iter16
  Stage3 before introducing any other novelty.
- Stage3 trainer, Stage1 embeddings, and the SID export pipeline are
  unchanged from Iter25. Any change in those components is a different
  experiment.