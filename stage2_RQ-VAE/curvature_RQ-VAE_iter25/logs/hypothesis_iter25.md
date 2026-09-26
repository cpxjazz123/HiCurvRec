# Iter25 Falsifiable Hypothesis

## Mechanism

Replace only the Iter11 applied layer-scale prior with the pre-registered bounded branching-plus-raw-residual prior:

```text
h_l = log(B_l) / max_j log(B_j)
r_l = m_l_raw / max_j m_j_raw
u_l_prior = 0.01 + 0.98 * (h_l + r_l) / 2
B = [19.32, 1.46, 1.015]
m_l_raw = [1.000, 0.10941, 0.09331]
u_l_prior = [0.990000, 0.1262333365, 0.0581856194]
```

These values initialize and anchor the existing learnable `c_layer_scale`. Keep Iter11's cyclic `c_l(t)`, behavior/commitment/Sinkhorn settings, warm-start source and exclusions, optimizer, data, dimensions, and Stage3 trainer unchanged. The model has three quantization layers; the training input dimensions are not changed.

## Direct effects — mechanism activation

All statements below are deterministic acceptance checks, not SID-quality gates:

1. Before warm-start transfer, Iter25 computes `u_l_prior` within `1e-6` of `[0.990000, 0.1262333365, 0.0581856194]`. Warm-start excludes only `layers.{0,1,2}.c_layer_scale`, so the actual initialized `c_layer_scale` values and regularization centers equal those priors within `1e-6`.
2. At a fixed curriculum step before optimizer drift, curvature is strictly ordered `c_0 > c_1 > c_2`, since the unchanged cyclic formula is monotonic in `u_l`. At step 0 (`g=0`), expected `c` is approximately `[0.269243, 0.061973, 0.055201]`; at step 50,000 (`g=1`), approximately `[1.474707, 0.339439, 0.302346]`. Every value must remain in `[0.05, 1.5]`.
3. On one warm-start checkpoint and one fixed training batch: `total_loss.requires_grad` is true and `total_loss.grad_fn` is present; reconstruction, RQ-VAE/commitment, behavior, and curvature-regularization loss components have finite gradients. The behavior/curvature route must provide nonzero gradient to the learnable layer scales; curvature-regularization gradient is checked after a small controlled perturbation from its center because its exact-center derivative is zero by definition.
4. Over five optimizer steps, each of the three `c_layer_scale` parameters changes by relative magnitude `>1e-7` from its initialized value.
5. With identical seed, checkpoint weights, and batch, compare 200 optimizer steps with the Iter25 prior ON against Iter11's applied prior `[0.001, 0.932889, 1.0]` OFF. Require finite terminal losses and `abs(L_on - L_off) > 1e-6`; also record the terminal scales and curvatures. No other mechanism or loss changes between arms.

Any failed direct check is an implementation/activation failure and forbids Stage2 training until repaired and rechecked.

## Proxy hypothesis — descriptive only

Relative to completed Iter24, the prior reallocation is predicted to increase `H(L1|L0)` above 5.394702 bits, increase `H(L2|L0)` above 5.742606 bits, and reduce the three-token collision rate below 8.9885%. L0 code utilization is predicted to remain high (at least 0.98). Compare to Iter18 reference values `5.590675`, `5.934625`, and `6.6539%`, respectively; full recovery to Iter18 is not required for the proxy hypothesis to show movement.

The `L0 oracle` is not measurable in the current collector (`compute_oracle_topk` returns `None`); do not infer or substitute an oracle result. Every Stage2 geometry value is descriptive. It cannot stop or gate the run.

## Downstream criterion

Complete Stage3 on the exported Iter25 SID JSON with the unchanged trainer. `test_R@10 > 0.065` is the only adoption target. Report comparisons against canonical Iter11 `0.0597677536`, best completed Iter18 `0.0598896220`, and completed Iter24 `0.0561987500` (`n_eval=57439` for each). Also report `valid_ndcg@20` as required by the run audit; it is not a substitute for the Stage3 test-recall adoption target.

## Confounds and interpretation

Iter11's training log records that `layer_norms.json` was absent and the run used fallback `[0.001, 0.932889, 1.0]`, not the raw-median order. Iter25 changes this applied prior mapping while introducing the branching input; this is a prior-mapping experiment, not a causal isolation of branching alone. The code clamps the learnable scale when used and regularizes it toward the prior, but does not impose momentum-restricted per-step drift. Report checkpoint scales to expose any learned departure from the prior. Do not change the behavior loss, Stage1 embeddings, or Stage3 trainer.
