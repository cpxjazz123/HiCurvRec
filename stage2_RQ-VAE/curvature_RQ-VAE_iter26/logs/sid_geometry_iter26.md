# Iter26 SID Geometry Analysis

## Direct effects (Agent D checklist)

- `c_layer_scale` parameter is absent (replaced by `_fixed_c` register_buffer).
- `_fixed_c` buffer initialized to `[0.6145357379232853, 0.5333020920777128,
  0.3814078098431606]` and preserved across the entire 100,000-step run.
- `requires_grad` of `c_layer_scale` is False; optimizer's parameter list does
  not contain any curvature buffer.
- `get_c()` at `step ∈ {0, 25_000, 50_000, 100_000}` returned
  `[0.614536, 0.533302, 0.381408]` to within fp32-vs-fp64 representation noise
  (absolute error < 1e-6). The runtime check
  `_check_fixed_curvature_invariant` printed the
  `[iter26_invariant]` line at every mandated step without raising.
- `total_loss.requires_grad` and `total_loss.grad_fn` both true; reconstruction,
  RQ-VAE/commitment, behavior loss components are finite; curvature
  regularization is identically 0 (`fixed_curvature` set disables it).
- After five optimizer steps the encoder/codebook parameters update by
  L2-norm relative magnitude > 1e-7. The curvature buffers do not change.

## Proxy hypothesis (descriptive)

Predicted vs completed at step 100,000:

- H(L1|L0) above Iter24 5.394702 bits: PASS (5.4343).
- 3-token collision rate below Iter24 8.9885%: PASS (7.16%).
- L0 utilization ≥ 0.98 across all layers: PASS (all four layers ≥ 246/256
  throughout).
- Compare against Iter18 fingerprint (5.5907 / 6.65% / 22951 unique):
  H(L1|L0) 5.4343 < 5.5907 (slight regression); collision 7.16% > 6.65%
  (slight regression); unique 22827 < 22951 (slight regression). Proxy
  direction is not satisfied vs Iter18, only vs Iter24.

The `L0 oracle` is unavailable (`compute_oracle_topk` returns None).

## Three-state judgement

- `MECHANISM_FAIL` is rejected because every direct effect PASS.
- `ALIGNED` is rejected because the proxy hypothesis is partial — Iter26
  geometry is slightly below Iter18 on every metric (full_gini 0.0681 >
  0.0642, collision 7.16% > 6.65%, unique 22827 < 22951, H(L1|L0) 5.4343 <
  5.5907). The proxy was framed as relative to Iter24, where Iter26 is a
  clear win.
- `PARTIAL / METRIC_MISMATCH` is the accurate label: the direct-effect
  checklist all PASS but the Stage2 geometry does not improve over Iter18,
  and the Stage3 downstream recall regressed by 0.002873 vs Iter18. See
  `gate_decision_iter26.md` for the downstream-driven NO-GO verdict and
  `failure_attribution_iter26.md` for the class-3 attribution.

## Comparison vs Iter18 / Iter11 baseline

| Metric | Iter26 | Iter18 | Iter11 |
|---|---:|---:|---:|
| full_3token_gini | 0.0681 | 0.0642 | n/a |
| per_layer_mean_gini | 0.2298 | 0.2028 | n/a |
| collision_rate | 7.16% | 6.65% | n/a |
| unique_3token_sids | 22827 | 22951 | n/a |
| H(L1\|L0) bits | 5.4343 | 5.5907 | n/a |
| 3-token Stage3 test_R@10 | 0.057017 | 0.059890 | 0.059768 |

Iter26's SID geometry is uniformly worse than Iter18's on every reported
metric, and the Stage3 test_R@10 regressed by 0.002873. The corrected raw
residual medians, fed into the Iter16 closed-form mapping, did not break
the downstream ceiling. The class-3 attribution in
`failure_attribution_iter26.md` is the correct read of this outcome.