# Iter31 gate decision — S12 canonical

```text
RUN_VALIDITY=VALID_CORRECTED_STAGE3_RUN
MECHANISM_STATUS=ACTIVE_NEUTRAL
PROMOTION_STATUS=PROMOTION_FAIL
PROMOTION_METRIC=test_R@10
PROMOTION_THRESHOLD_STRICT=> 0.065
OBSERVED_R@10=0.05659917477671965
MARGIN_TO_THRESHOLD=-0.008400825223280353
```

## Hard-gate validity

Root `CLAUDE.md` §5.1 (SHA-256 `5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2`) requires 150 completed Stage3 epochs without metric-triggered early stop and a final test. The corrected Iter31 run's primary `training_metrics.jsonl` (`results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/training_metrics.jsonl`, SHA-256 `0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd`) records epochs 1–150 once and in order, zero early-stop events, a single epoch-150 final test and `n_eval=57439`. Its final JSON SHA-256 is `d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957`; `logs/stage3_outcome_iter31.md` SHA-256 `66a68886246d668ed1079112d6b106614e0c5241b800ba73993be15ab69ce894` records process exit 0 and acceptance. S11 round-2 Judge SHA-256 `673aa677692275ec12e43d16227ddc54373f1ad55fd9b613652d73852156ca2e` authorizes this single corrected replacement; the incomplete first attempt is excluded.

Historical abort reconciliation: `logs/iteration_abort_iter31.md` SHA-256 `24e18d2752de439791eccfd3d243ee07decee9b3c9328993e1e2c97790848712` is preserved as contemporaneous round-4 history. Later S08 round-5 records invalidate the earlier incomplete provenance for the current gate and pass the repaired S08; the prelaunch S09 gate `logs/stage2_preflight_iter31.md` SHA-256 `bfc8d96b76a7664a1f7874ef8a2d9735dc55eb1fb11a26995f81a317cf75a0c1` explicitly records `DELIBERATION_GATE_PASS`, S08 round 5 `REPAIR_PASS`, and S09 round 2 `REPAIR_PASS`. `logs/stage2_completion_iter31.md` SHA-256 `7face65d5498946019503ac59e526d2e029fe12c74065e29495d896ae540fecf` verifies Stage2 completion. This later primary evidence resolves the old abort for the current path; it does not rewrite or deny that historical record.

## Promotion test and mechanism distinction

The protocol-compatible canonical comparator is Iter29 `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json` (SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`), with `n_eval=57439` and a primary 150-epoch training record (SHA-256 `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`). Candidate minus comparator deltas are R@5 −0.0014450112293041342; R@10 −0.0026114660770556603; NDCG@5 −0.0009673958223110901; NDCG@10 −0.0013285239299029965. These are observed point differences from one run per configuration, not a causal estimate or noise estimate.

S08/S10 activation and fixed-contract evidence is in `logs/deliberation/S08_MVG/round_5/judge.md` (SHA-256 `b3030748270d3711198d0d22c9a901f700073be1e55c20bd4832208bf913c8dc`) and `logs/sid_geometry_iter31.md` (SHA-256 `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2`): HRA Step6 changed decoder-facing output on its registered batch, and FCCR-1 remained fixed/non-trainable/time-invariant. Therefore the mechanism is active; the modest ambiguous one-run decrement is classified `ACTIVE_NEUTRAL` under the skill's small-delta rule, with no invented noise level. This is separate from promotion.

**Gate outcome:** `PROMOTION_FAIL`, because `0.05659917477671965` is below the strict threshold `0.065` by `0.008400825223280353`. Promotion failure does not itself make the active mechanism invalid, inactive, or causally negative. All citations and the selected statuses are consistent with `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md` (`ACCEPT_B`).
