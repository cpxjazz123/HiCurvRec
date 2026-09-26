# Iter25 SID Geometry Analysis

## Direct effects (Agent D checklist)

- `c_layer_scale` initialized to `[0.99, 0.126, 0.058]` and
  `curvature_regularization` centered on the same values:
  PASS (verified by MVG).
- Fixed-phase layer ordering `c_0 > c_1 > c_2` at multiple
  cycle phases: PASS (`[0.269, 0.062, 0.055]` at step 0,
  `[1.475, 0.339, 0.302]` at step 50000).
- Reconstruction, RQ-VAE/commitment, behavior, and curvature
  regularization loss components all carry finite gradients:
  PASS.
- Five-step optimizer updates change each `c_layer_scale` by
  relative magnitude > 1e-7: PASS (`[0.0051, 0.0386, 0.0859]`).
- 200-step ON (Iter25 prior) vs OFF (Iter11 fallback prior) total
  loss delta > 1e-6: PASS (delta = 0.0124).

## Proxy hypothesis (descriptive)

Predicted vs completed:

- H(L1|L0) above Iter24 5.394702 bits: PASS (5.5874 at step 100k).
- H(L2|L0) above Iter24 5.742606 bits: not reported by the
  per-layer metric in the trainer log; `full_gini` is reported
  instead and is 0.0637 vs Iter24 / Iter11.
- 3-token collision below Iter24 8.9885%: PASS (6.6092% at step
  100k, vs Iter18 6.6539%).
- L0 utilization >= 0.98: PASS (all layers >= 252/256 throughout).
- L0 oracle: unavailable (`compute_oracle_topk` returns None).

## Three-state judgement

`ALIGNED` is rejected because the direct-effect checklist all
PASS but the proxy hypothesis is at best partial (no L2/L0
conditional entropy was produced; only `full_gini` and the
per-layer Gini). `MECHANISM_FAIL` is rejected because the direct
effects all PASS. `PARTIAL / METRIC_MISMATCH` is the most accurate
label here: the geometric improvement predicted by the bridge was
observed, but the Stage3 downstream gain was not. See
`gate_decision_iter25.md` for the downstream-driven NO-GO verdict
and `failure_attribution_iter25.md` for the class-3 attribution.

## Comparison vs iter18 baseline

| Metric | Iter25 | Iter18 |
|---|---:|---:|
| full_3token_gini | 0.0637 | 0.0642 |
| per_layer_mean_gini | 0.2002 | 0.2028 |
| collision_rate | 6.6092% | 6.6539% |
| unique_3token_sids | 22962 | 22951 |
| H(L1\|L0) bits | 5.5874 | 5.590675 |
| 3-token Stage3 test_R@10 | 0.058828 | 0.059890 |

Iter25 SID geometry sits slightly above Iter18 on every reported
metric. The Stage3 test_R@10 nonetheless regressed by 0.00106
relative to Iter18, which is consistent with the persistent
downstream-ceiling pattern identified in the iteration bridge.