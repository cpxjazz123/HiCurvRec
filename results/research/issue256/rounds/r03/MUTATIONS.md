# R02 mutation detection contract — inherited budget plus actual native refresh

The five new valid policy mutations are injected into the production `model.calibration.norm_budget_weight` entry. Each must fail the exact assertion `maximal module-safe coefficient must be 0.25` on hardcoded encoder(4,2), Query1(1,4), codebook0(3,3) norm budgets. Crash, import error, timeout or an unrelated exception is not counted.

| Mutation | Category | Designated assertion |
|---|---|---|
| Minimum ratio replaced with maximum | numerical/coefficient | maximal module-safe coefficient must be0.25 |
| Minimum replaced with mean | structural/reduction | same hard optimum assertion |
| Query1 constraint omitted | structural/mask | same hard optimum assertion; Query1 is the true bottleneck |
| Curvature norm denominator squared | numerical/coefficient | same hard optimum assertion |
| Coefficient sign flipped | numerical/sign | same hard optimum assertion |

Routing coverage additionally executes the inherited production selection/gate/prefix mutations in candidate `tests/test_mutations.py`:direct straight-through versus same-cost legal gradient, detached selection costs, flattened layer gates, reused final prefixes and wrong softmax axis must be caught by their specified numerical/gradient/normalization assertions. Their executable captured assertion records are written into this round's `MUTATIONS.json`/`LAYER_MUTATIONS.json`, never over old Issue255 evidence. Full inherited numerical geometry/mask mutations remain included.

The real production calibrator regression uses an actual1,024-item legal training batch and actual original+curvature losses. It rejects any unbounded coupled group and verifies maximality/no parameter update. Independent global vector averaging/cancellation and four-rank actual first-step budgets are checked in the real diagnostic, not claimed from a helper echo. Execution receipts and source identities are recorded after the tests actually run; this document is not a fabricated PASS result.


## R02 native-block refresh additions

- Off-by-one anchor51->50 must fail `actual calibration epochs must be exactly1and51`.
- Skipped later anchor must fail `actual60epoch trainer must refresh its budget at epoch51`.
- Current first-step identity is checked on the actual epoch51 batch/sample; no synthetic receipt-only mutation is counted as production coverage.
- The real60epoch/four-rank observer independently checks all17 current global budgets at1/51 and maximality. A stale lambda must fail `actual initial global coefficient is not maximal` or the explicit module budget bound, not an import/crash.

These add schedule/current-state coverage to the inherited numeric/coefficient, structure/group omission and selected-cost gradient-routing mutation families. Exact assertion receipts are recorded after execution; no preemptive PASS.
