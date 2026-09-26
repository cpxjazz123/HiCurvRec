# Iteration bridge — Iter25 → Iter26

## Mechanism execution (Iter25)

- **Implementation: PASS.** Iter25 `mvg_check_iter25.log` records `MVG PASS`; Stage2 ran 100,000 steps and exported a validated four-token SID file.
- **Activation: PASS.** `behavior_curvature_grad_norms` ≥ 1e-12 for all three learnable scales; 200-step ON/OFF total loss delta `0.0124` > 1e-6; five-step relative scale updates `[0.0051, 0.0386, 0.0859]`.
- **Geometry alignment vs Iter18 on observed proxies: PARTIAL.** Iter25 sits slightly above Iter18 on every descriptive metric (full_gini 0.0637 vs 0.0642, per_layer_mean 0.2002 vs 0.2028, collision 6.6092% vs 6.6539%, unique 22962 vs 22951, H(L1|L0) 5.5874 vs 5.5907). L0 utilization unchanged at 1.0.
- **Stage3 downstream: FAIL relative to canonical baselines.** Iter25 test_R@10 = 0.058828 (n_eval=57439); Iter11 0.059768, Iter18 0.059890, Iter24 0.056199. Adoption target 0.065 not met.

## Difference from best completed Stage3 baseline (Iter18)

| Fingerprint | Iter18 | Iter25 | Δ Iter25 − Iter18 |
|---|---:|---:|---:|
| L0 utilization | 1.0000 | 1.0000 | 0.0000 |
| H(L1\|L0), bits | 5.590675 | 5.5874 | -0.003 |
| H(L2\|L0), bits | 5.934625 | (unreported) | n/a |
| Three-token collision rate | 6.6539% | 6.6092% | -0.0447 pp |
| Test_R@10 (Stage3) | 0.0598896220 | 0.0588276258 | -0.0010620 |

`L0 oracle` is unavailable (the registered `compute_oracle_topk` returns `None`). The Stage3 metric is taken from the completed `test_final.json`.

## Dominant bottleneck

**Both Iter25 and Iter16 attempted to encode layer-curvature into Stage2 by feeding behavior branching plus raw residual medians into a closed-form formula. Iter16 used the wrong residual vector (the calibrated `[0.001, 0.932889, 1.0]` layer scale) and Iter25 kept the same wrong vector while allowing `c` to drift via a learnable scale — neither change clarified whether the closed-form mapping is the binding lever or whether one of its inputs is wrong.**

Both attempts share the same false premise: a single scalar per layer suffices to encode curvature, and that scalar should come from the same `B_l / m_l` mapping. The two failures differ only in whether `c` is fixed (Iter16) or anchored to a learnable parameter that can move under training (Iter25). With neither variant, Stage3 moves past the 0.060 ceiling.

## Forbidden next directions

- Do not modify the closed-form `s_l = log(1+B_l)/log(1+m_l/m_min)` mapping, the `c_l = c_base * exp(alpha*z_l)` formula, or the constants `c_base=0.5`, `alpha=0.2`. Iter26 must reuse Iter16's exact formula.
- Do not introduce a learnable `c_layer_scale`, cyclic `c(t)`, curvature regularization, or any other "training-aware" curvature modulation in Iter26. The fixed closed-form is the only allowed curvature knob.
- Do not change `MECHANISM_NAME` to anything that the Stage3 trainer cannot recognize via the `item_sids.json` lookup chain. The wrapper must redirect `RQVAE_VARIANT` to a fresh string but keep `_resolve_code_file`'s fallback chain matching `curvature_RQ-VAE_iter26/item_sids.json`.
- Do not modify Stage1 embeddings, Stage3 trainer, or the SID export pipeline beyond what is necessary to route outputs to `results/stage2_RQ-VAE/curvature_RQ-VAE_iter26/`.
- Do not run Iter26 with the Iter16 `m = [0.001, 0.932889, 1.0]` calibration. Iter26 must use the raw residual medians `[1.0, 0.10941, 0.09331]` (the un-normalized input to Iter1's `calibrate_residual_scales.py`, the same values that anchored Iter25's prior).

## Falsifiable Iter26 objective

Use Iter16's exact closed-form implementation but feed it the raw residual medians as the `m` input instead of the normalized calibration:

```text
m_raw = [1.0, 0.10941, 0.09331]
B    = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
c_base = 0.5
alpha  = 0.2

s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))
z_l = (s_l - mean(s_l)) / (std(s_l) + 1e-12)
c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)
```

The exact values written to `scripts/computed_behavior_branching.json` are:

```text
s_l = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z_l = [1.03129493309937,   0.3223997103378134,  -1.3536946434371857]
c_l = [0.6145357379232853, 0.5333020920777128,  0.3814078098431606]
```

These three `c_l` values must remain constant across the entire 100,000-step run. The MVG layer-1 must check `get_c()` at `step ∈ {0, 25_000, 50_000, 100_000}` and verify each layer's curvature matches `c_l` within 1e-6. Any drift is a `CONTRACT FAIL` and Stage2 must not launch.

Adoption gate (unchanged from bridge): complete Stage3 with `test_R@10 > 0.065`. Comparators: canonical Iter11 `0.0597677536`, best completed Iter18 `0.0598896220`, completed Iter25 `0.0588276258`, Iter16's downstream test_R@10 is unavailable (`test_final.json` for Iter16 is missing from the registry, so it cannot be added as a comparator).

## The single research question Iter26 must answer

> When behavior branching and the **true** raw residual medians (not the post-calibration normalization) are fed into Iter16's closed-form formula, do the resulting fixed per-layer curvatures improve downstream recommendation recall over Iter11/Iter18/Iter25?

If `test_R@10` rises above Iter11 + Iter18 / 2 = 0.0598 and into the [0.060, 0.065) band, the closed-form mapping is a partial fix and warrants further parameter sweeps. If Iter26 regresses like Iter25, the closed-form mapping is not the binding lever and Iter27 should move outside Stage2 (per the recommendation in `failure_attribution_iter25.md`).