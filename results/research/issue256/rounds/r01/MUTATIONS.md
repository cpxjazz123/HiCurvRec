# R01 mutation detection contract

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
