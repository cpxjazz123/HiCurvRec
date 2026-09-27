# Iter31 failure attribution — S12 canonical

```text
CLASSIFICATION_STAGE=S12_RESULT_CLASSIFICATION
RUN_VALIDITY=VALID_CORRECTED_STAGE3_RUN
MECHANISM_STATUS=ACTIVE_NEUTRAL
PROMOTION_STATUS=PROMOTION_FAIL
```

## Classification

Iter31 is valid for result classification using only the corrected Stage3 run at `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/`. Its `training_metrics.jsonl` SHA-256 is `0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd`; `test_final.json` SHA-256 is `d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957`. Primary run metrics contain ordered training epochs 1–150, no early-stop event, one epoch-150 final test, and `n_eval=57439`. Final test: R@5=0.03809258517731855; R@10=0.05659917477671965; NDCG@5=0.025285381077977752; NDCG@10=0.031247955601888432.

The canonical Iter29 comparator is `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json` (SHA-256 `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`) and its training metrics SHA-256 is `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`. Its primary records show 150 ordered epochs, final epoch-150 test, and the same `n_eval=57439`. Deltas (Iter31−Iter29): R@5 −0.0014450112293041342; R@10 −0.0026114660770556603; NDCG@5 −0.0009673958223110901; NDCG@10 −0.0013285239299029965. All four measured point differences are lower, but one locked-seed comparison provides no run-noise estimate and cannot establish a causal mechanism effect. Applying the workflow's small/ambiguous-delta rule, classify the active mechanism as `ACTIVE_NEUTRAL` (near-parity/ambiguous observed decrement), not `ACTIVE_NEGATIVE`. This does not claim the difference is known to be noise-sized or statistically insignificant, and does not generalize beyond these runs.

Stage2 mechanism evidence is `logs/sid_geometry_iter31.md` SHA-256 `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2` and S08 round-5 Judge `logs/deliberation/S08_MVG/round_5/judge.md` SHA-256 `b3030748270d3711198d0d22c9a901f700073be1e55c20bd4832208bf913c8dc`. These record FCCR-1 fixed curvature and a finite nonzero HRA Step6-vs-Euclidean decoder-facing output difference on the registered batch; this verifies computational activity, not downstream benefit. Stage2 completion `logs/stage2_completion_iter31.md` SHA-256 `7face65d5498946019503ac59e526d2e029fe12c74065e29495d896ae540fecf` reports successful 100,000-step completion and consistent SID export.

The historical `logs/iteration_abort_iter31.md` remains untouched (SHA-256 `24e18d2752de439791eccfd3d243ee07decee9b3c9328993e1e2c97790848712`). It records the then-valid S08 round-4 abort for missing contemporaneous provenance. S08 round-5 Judge/repair records later invalidate rounds 1–4 incomplete provenance for the current gate and record `REPAIR_PASS`; `logs/stage2_preflight_iter31.md` SHA-256 `bfc8d96b76a7664a1f7874ef8a2d9735dc55eb1fb11a26995f81a317cf75a0c1` records `DELIBERATION_GATE_PASS`, explicitly listing S08 round 5 `REPAIR_PASS` and S09 round 2 `REPAIR_PASS`; Stage2 subsequently completed. This later primary gate evidence resolves the old abort for the current Stage2/Stage3 path without rewriting history. S11 round-2 Judge SHA-256 `673aa677692275ec12e43d16227ddc54373f1ad55fd9b613652d73852156ca2e` adjudicated the defective first Stage3 attempt and authorized one corrected same-iteration run. The first timestamp (`Sep-27-2026_20-15-25`) is invalid and excluded; it recorded only epochs 1–21 and early-stop.

## Attribution limits

No causal explanation of the four lower metrics is established. No run-noise magnitude is available. Stage2 descriptive SID statistics are not Stage3 gates or causal explanations. Do not run a repeat, seed replication, sweep, ablation, or root-cause investigation.

**S12_JUDGE**=`logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/judge.md`; verdict `ACCEPT_B`; confidence MEDIUM overall (high on validity/values/activation/promotion, medium on qualitative small-delta classification). Candidate A/B files are audit-only.
