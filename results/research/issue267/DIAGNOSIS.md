# Issue #267 root-cause diagnosis

Frozen before mechanism implementation. Parent: #265 R2 109; all quantities below are
read-only diagnostics on the frozen final checkpoint/SID. No training or parameter
update was performed.

## Observation

The parent #265 has a very weak full-table curvature ordering. On the complete
28,391,948 valid triples, using the frozen equal-weight three-expert metric and
the checkpoint's tangent scale:

| space | N | MAE | RMSE | Pearson | Spearman | target mean | target var | pred mean | pred var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard q | 28,391,948 | 2.24595 | 2.66498 | -0.03310 | -0.09342 | -1.06232 | 2.87297 | 0.73624 | 0.88857 |
| encoder z | 28,391,948 | 2.24801 | 2.67032 | -0.02838 | -0.09208 | -1.06232 | 2.87297 | 0.73281 | 0.94178 |

Neither hard quantization nor the encoder representation is the sole problem: q and z
have almost the same ordering metrics. Predictions are strongly compressed (variance
ratio q `0.309`, z `0.328`) and have a large positive mean offset (`+1.79856` q,
`+1.79513` z) relative to the target mean. Sign accuracy is only `0.33235` (q) /
`0.33263` (z), with balanced accuracy ~`0.3192`.

Within each centre, ordering is also weak:

| space | centres | rows | mean Spearman | median Spearman | positive fraction | weighted mean |
|---|---:|---:|---:|---:|---:|---:|
| hard q | 23,819 | 28,391,441 | 0.09288 | 0.08695 | 0.64134 | 0.04869 |
| encoder z | 23,819 | 28,391,441 | 0.09341 | 0.08758 | 0.64197 | 0.04939 |

The overall Spearman is negative because centre offsets dominate; within a centre the
parent has a small positive ordering signal that is much smaller than the point-value
error. The target and prediction moments are in
`analysis/CURVATURE_FULL_109_{Q,Z}.json`; centre results are in
`analysis/CENTER_ORDER_109.json`.

## Low-cost gate probe

On 32 deterministic training-side sampler draws (787 rows, seed 42 protocol), the
learned gate improves over equal weights but does not repair the ordering:

| prediction | MAE | RMSE | Pearson | Spearman | pred var |
|---|---:|---:|---:|---:|---:|
| learned local gate | 2.05557 | 2.50826 | 0.14420 | 0.10452 | 1.89200 |
| equal weight | 2.43348 | 2.91255 | 0.05001 | 0.01374 | 4.53374 |

This is a sampled diagnostic, not the full-table result (`analysis/GATE_PROBE.json`).
It shows that gate scaling changes the level and the small ordering signal, but q/z
full-table identity rules out "gate only" as the complete explanation.

## Competing hypotheses

| id | explanation | status | evidence |
|---|---|---|---|
| H267-1 | Pointwise Huber/error training leaves within-centre relative order underfit: point errors can be low while order is weak. | **SUPPORTED** | `CURVATURE_FULL_109_Q.json`: MAE 2.246, Spearman -0.093; `CENTER_ORDER_109.json`: within-centre median 0.087; prediction variance ratio 0.309 |
| H267-2 | The low overall correlation is dominated by centre/scale/target-definition offsets, and q-versus-z similarity means the loss is not a quantization-only failure. | **SUPPORTED** | q/z full metrics nearly identical; mean offset +1.799; variance ratio 0.309/0.328; role/degree strata in full JSON |
| H267-3 | The gate only rescales the distance and is the primary cause; a gate intervention alone would repair the ordering. | **INCONCLUSIVE** | 32-draw gate probe improves MAE by 0.378 and Spearman by 0.091, but is sampled; q and z full ordering are almost identical. Full-table fixed-gate comparison was not run. |
| H267-4 | A same-centre differentiable relative ranking term added to the parent point-value loss will improve within-centre ordering and may transfer to recommendation. | **UNTESTED** | mechanism not yet implemented; predicted by H267-1 |
| H267-5 | Improving relative curvature order will necessarily improve Stage3 Recall@10. | **UNTESTED** | no interventional evidence; behavior link is not assumed |

## Competing interpretation

A low Pearson/Spearman can result from an arbitrary monotone offset, from target
definition mismatch, or from a genuinely uninformative representation. The full q/z
comparison and the within-centre distribution show that the representation retains a
small within-centre signal, but the point-value objective does not align its order.
The new term therefore targets **relative order within a centre**, not the absolute
level; it must not be claimed to fix the cross-centre mean offset by itself.

## Six training-admission answers

1. **Problem.** The parent hard-q curvature prediction has near-zero full-table
   Spearman (-0.093) and compressed variance/offset; within-centre ordering is only
   ~0.087 median.
2. **Evidence.** Complete 28,391,948-triple recomputation for both q and z,
   centre-stratified order, target/prediction moments and the 32-draw gate probe.
3. **Why the new mechanism acts on it.** Pairwise same-centre comparisons directly
   penalize a wrong sign of `κ_q(t)-κ_q(u)` when `κ_graph(t)-κ_graph(u)` is nonzero,
   instead of only fitting each point value.
4. **Material difference from prior failures.** The parent already has a point-value
   Huber term and local mixed geometry; this adds a separate ordering objective on
   the actual final hard q, with target-derived pair weights, not a new encoder,
   extra baseline, or reweighted point loss.
5. **Success metrics.** Full-table q (and z diagnostic) MAE/RMSE/Pearson/Spearman,
   within-centre ordering and sign/ordering concordance; then one frozen Stage3
   R@10/NDCG. The pre-Stage3 score is diagnostic only.
6. **If it fails.** It distinguishes "ranking supervision did not improve the
   representation" from "structural order improved but recommendation did not"; the
   negative result is retained without reweighting or a second seed.

## Remaining uncertainty

- The gate probe is sampled, not full-table; H267-3 remains INCONCLUSIVE.
- The target curvature proxy and equal-weight metric are frozen; this issue does not
  claim the proxy is the true generative curvature.
- All historical Stage3 test results were already observed; the new result is not a
  blind test and no test-guided weight selection is allowed.

## Scope decision

The root cause is sufficiently identified to authorize a candidate mechanism:
same-centre relative ranking on the final hard q, with the existing point-value term
retained. Details are frozen in `CONTRACT_MECHANISM.md` before code changes.
