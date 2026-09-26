# Iter25 Gate Decision

## Outcome

NO-GO. Iter25 does not satisfy the adoption criterion.

## Numbers

- Stage3 test_R@10: 0.05882762582914048 (n_eval=57439).
- Stage3 test_R@5: 0.03906753251275266.
- Stage3 test_ndcg@10: 0.03255499542719071.
- Adoption threshold (test_R@10): 0.065.
- Canonical Iter11 test_R@10: 0.0597677536 (delta −0.00094).
- Best completed Iter18 test_R@10: 0.0598896220 (delta −0.00106).
- Completed Iter24 test_R@10: 0.05619875 (delta +0.00263).

## Mechanism executed

- Pre-registered bounded branching-plus-raw-residual layer-scale prior
  `u_l_prior = [0.990000, 0.126233, 0.058186]` was applied via
  `_build_layer_scale_prior()` and passed as `residual_layer_norms`
  to the unchanged RqVae constructor. Warm-start excludes only
  `layers.{0,1,2}.c_layer_scale`; everything else is loaded from
  the iter8 checkpoint exactly as in Iter11.
- Cyclic curvature, behavior/commitment/Sinkhorn, optimizer, warm
  start, dimensions, and Stage3 trainer are unchanged from Iter11.
- Stage2 ran 100,000 global steps on 4-GPU DDP, 1309 s wall-clock,
  ckpt at every 10k step.
- Stage3 ran 150 epochs on 4-GPU DDP with cosine LR + warmup, ndcg
  callbacks disabled (NO_EVAL=True per current protocol); final test
  inference produced the numbers above.

## Stage2 descriptive comparison (informational)

| Metric | Iter25 | Iter18 | Delta |
|---|---:|---:|---:|
| full_3token_gini | 0.0637 | 0.0642 | -0.0005 |
| per_layer_mean_gini | 0.2002 | 0.2028 | -0.0026 |
| collision_rate | 6.6092% | 6.6539% | -0.0447 pp |
| unique_3token_sids | 22962 | 22951 | +11 |
| H(L1\|L0) bits | 5.5874 | 5.590675 | -0.0032 |

Stage2 geometry sits slightly above Iter18 on Gini, per-layer
balance, collision rate, and unique count, while H(L1|L0) is
essentially identical. HitRate@50 was intentionally not computed
(CLAUDE.md §2 removed it as a gate).

## Interpretation

The pre-registered Iter25 prior matched the directional goal set
in the bridge (reallocate per-layer scale from the Iter11 fallback
mapping `[0.001, 0.933, 1.0]` to a branching-informed
`[0.99, 0.126, 0.058]`). The MVG four-layer gate confirmed the
prior loads, the scales are non-degenerate, gradient flow and
optimizer updates touch the learnable scales, and the same-seed
200-step ON/OFF arm shows a finite total-loss delta.

Stage2 completed with all three layers at high codebook
utilization (>= 252/256 per layer across 100k steps) and the
descriptive geometry improved slightly vs Iter18 on every metric
reported by `/tmp/sid_metrics_any.py`. The H(L1|L0) gap that the
bridge attributed to the Iter24 hypothesis was effectively closed
relative to Iter18 (5.5874 vs 5.590675). Despite this, the Stage3
test recall regressed by 0.00106 vs Iter18 and 0.00094 vs
canonical Iter11, and remains 0.00617 below the 0.065 adoption
threshold.

This is the persistent downstream-ceiling pattern predicted by the
iteration bridge and the direction decision: the geometry moves in
the expected direction but the change does not translate into
downstream recall gains. The seven Class-3 entries in the failed
mechanism ledger (iter7-iter13) and the +0.0030 ceiling seen by
iter8 corroborate that the binding constraint is downstream of the
Stage2 SID geometry under the current Stage3 trainer/protocol.

## Adoption status

- test_R@10 > 0.065: NO (0.058828).
- Comparison vs canonical Iter11 0.059768: NO (-0.000940).
- Comparison vs best completed Iter18 0.059890: NO (-0.001062).
- Improvement over Iter24: YES (+0.002629) but does not meet the
  absolute threshold.

Iter25 is not adopted as the new promoted baseline. No iteration
is promoted; the prior best completed baseline remains Iter18
(0.0598896220).

## Notes for next iteration

- Iter25 should not be repeated: the bridge's gap-closing
  expectation is realized on geometry proxies and absent on downstream.
- The failed mechanism ledger does not gain a new entry because
  this is the prior-mapping experiment the bridge framed as the
  pre-registered next hypothesis; it is treated as a class 3
  ceiling instance with documented bridges and is not the same
  fingerprint as a previously failed mechanism.
- Iter26 must propose a different lever. Possible directions:
  change the input pipeline (e.g., Stage1 embedding model), or
  change the Stage3 trainer/protocol, since both are unchanged
  here and the Stage2 family has repeatedly stalled under them.
- If Iter26 still operates on Stage2 mechanics, the failure ledger
  now contains eight class-3 outcomes in the per-layer curvature
  family; the (c) stopping condition of "same mechanism tried with
  three different hyperparameter settings" is satisfied and the
  mechanism family should be retired from the candidate pool.
- Audit commit: `95465eeb4dadeac6a1520e95a735d9743ce23b5e`
