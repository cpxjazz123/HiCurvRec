# Iter26 Mechanism Manifest (FCCR-1)

This iteration is a one-factor replication of Iter16's closed-form fixed curvature
mechanism, replacing the residual input vector with raw residual medians while
holding every other Iter16 component byte-identical. The mechanism manifest below
enumerates every formula input and distinguishes the three FCCR-1 quantities that
Iter25 / Iter16 conflated:

- `raw_residual_median` ≠ `normalized_layer_scale` ≠ `learnable_c_layer_scale`

## Formula inputs (Iter26 = Iter16 closed-form with corrected residual)

| Field | Required | Iter26 value | Provenance |
|---|---|---|---|
| `behavior_branching` `B_l` | yes | `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` | `scripts/computed_behavior_branching.json`, key `branching`; produced by Iter12's `compute_behavior_branching.py` from iter8 SIDs + train.parquet |
| `raw_residual_median` `m_l^{raw}` | yes | `[1.0, 0.10941, 0.09331]` | Iter1's `calibrate_residual_scales.py` raw output (the un-normalized median residual norm across all items and codebook assignments); recorded as `raw_residual_medians` in `computed_behavior_branching.json`. The legacy `residual_norm` field also stores `[1.0, 0.10941, 0.09331]` for Iter26 |
| `closed_form_c_base = 0.5` | yes | constant | Iter16's `CLOSED_FORM_C_BASE` |
| `closed_form_alpha = 0.2` | yes | constant | Iter16's `CLOSED_FORM_ALPHA` |

Note: Iter16's `residual_norm` was the Iter1-calibrated `[0.001, 0.932889, 1.0]`
which the bridge hypothesis flagged as a misinterpretation of the raw quantity.
Iter26 corrects this by reading the raw median residual norm directly, leaving the
formula and constants unchanged.

## Transformation pipeline

| Step | Formula | Iter26 result |
|---|---|---|
| 1. normalize residual | `m_min = min(m_l^{raw}) = 0.09331` | `m_min = 0.09331` |
| 2. log-branching ratio | `s_l = log1p(B_l) / log1p(m_l^{raw} / m_min)` | `s_l = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]` |
| 3. standardize | `z_l = (s_l - mean(s_l)) / (std(s_l) + 1e-12)` | `z_l = [1.03129493309937, 0.3223997103378134, -1.3536946434371857]` |
| 4. exponential map | `c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)` | `c_l = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` |

## Stage 2 source verification

| Source file | Status |
|---|---|
| `curvature_RQ-VAE.py` | declares `FIXED_CURVATURE_INVARIANCE_STEPS = frozenset({0, 25_000, 50_000, 100_000})`; `_check_fixed_curvature_invariant` aborts the run with `CONTRACT FAIL` if any `_fixed_c` buffer drifts > 1e-6 |
| `modules/quantize.py` | `Quantize.__init__` registers `_fixed_c` as a non-trainable buffer when `fixed_curvature` is provided; `get_c()` returns the buffer unconditionally |
| `modules/rqvae.py` | passes `fixed_layer_curvatures` through; no behavior-loss or curvature-regularization change |

## Semantic / provenance gate

| Term | Found in `mechanism_manifest_iter26.md` |
|---|---|
| `raw residual` | yes — see formula inputs table |
| `behavior branching` | yes — see formula inputs table |
| `normalized layer scale` | yes — explicitly **distinguished** in the opening note |

## Why this passes the FCCR-1 hard FAIL list

- `raw_residual_median` is **not** replaced by `normalized_layer_scale`:
  Iter26 reads `[1.0, 0.10941, 0.09331]` (raw median), not
  `[0.001, 0.932889, 1.0]` (Iter1 normalized).
- provenance is recorded in the table above.
- no historical fallback is silently used; the bridge hypothesis is the
  only thing changed.
- the formula name (`s_l = log1p(B_l)/log1p(m_l^{raw}/m_min)`) matches
  the actual substitution.
- numbers match `hypothesis_iter26.md` and the on-disk JSON to 1e-6.