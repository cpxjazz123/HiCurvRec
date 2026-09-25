# Iter1 hypothesis and observed outcome

## Mechanism tested

Layer baseline curvature was set from residual-scale calibration, then combined with the existing cyclic schedule:

`c_l(t) = c_min * exp(((u_l + |sin(pi*t/T)|)/2) * log(c_max/c_min))`

Calibration from the baseline checkpoint at global step 100000 produced residual medians `[1.0, 0.10941, 0.09331]` and normalized layer factors `[0.001, 0.932889, 1.0]` (the first factor is epsilon-clamped to satisfy the constructor's strict positive-value requirement).

## Verification and Stage2 observations

MVG passed. Per-layer nonzero gradient norms were `[311.930287, 30.533537, 6.848375]`; codebook update ratios were `[6.104e-03, 5.632e-03, 4.840e-03]`. At the cyclic peak, curvatures were `[0.548027, 0.96036, 0.999953]`.

At Stage2 step 100000: full Gini `0.1464`; layer Ginis `[0.2194, 0.5211, 0.5415]`; unique SIDs `20642/24587`; L0-L1 unique pairs `10412`; `H(L1|L0)=4.9125`; descriptive HR@50 `0.9911` (not a gate). Baseline run values available in the contemporaneous log: full Gini `0.1512`; layer Ginis `[0.1529, 0.4934, 0.5228]`; unique SIDs `20489`; L0-L1 pairs `10249`; `H(L1|L0)=4.8088`; HR@50 `0.9914`.

## Stage3 result and decision

Stage3 completed 150 epochs and final test evaluation over 57439 examples. `test_recall@10=0.05522380264280367`, `test_recall@5=0.03638642734030885`, `test_ndcg@10=0.029818537549432242`, `test_ndcg@5=0.02378050997806901`. The predefined adoption threshold `test_recall@10 > 0.065` was not met; do not promote this variant.

For comparison, the recorded TIGER baseline `test_recall@10=0.055049704904333294`, and repaired-curvature baseline `test_recall@10=0.0537613816396525`. The iter1 result is +0.0001741 versus TIGER and +0.0014624 versus repaired curvature; this small difference is below the strict adoption target and is not evidence of a robust improvement.

No external literature validation was performed. Stage3 trainer changes used to route this experiment have been reverted; the run outputs remain under `results/stage3_T5Train/unknown_variant/` because the absolute custom SID path did not match the variant-name lookup.
