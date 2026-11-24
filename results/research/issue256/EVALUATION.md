# Issue256 evaluation and isolation contract

Frozen before the first Issue256 candidate. Parent artifact identity checks are read-only; they are not supervised feature construction.

## Decision data

Reuse all registered training assets at their existing immutable paths. Recommendation validation is the complete existing `results/stage0_build_parquet/valid.parquet`, SHA256 `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9`. Its history and next-item targets are evaluated externally after a completed frozen Stage3 checkpoint, never added to Stage2 graph, curvature targets or RQ-VAE loss. No validation-dependent early stopping, seed choice or checkpoint substitution. Existing recommendation-validation results absent from a historical run stay NA.

Structure validation is the five `strict_struct_val_*` pair assets already registered in assets/manifest.json. Use direct, shared_in, shared_out, two_hop and the degree_matched control; include every registered pair and preserve masks/denominators. Validation is the only held-out structure population allowed to guide changes. Training full-graph curvature measurements are explicitly in-sample, not held-out generalization.

## Frozen ranking and population

Use the actual frozen `stage3_T5Train/train_HG-Rec.py` (SHA256 `54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690`), complete test population 57,439, four disjoint rank shards 14,360/14,360/14,360/14,359, zero-based target ranks with missing targets censored beyond the stored cutoff. Report Recall and NDCG at 5 and 10 only, logarithm base 2, micro-average over all examples. Beam20 does not imply a reported Recall@20. Preserve the trainer's decoding, item-SID namespace, seen-item filtering and candidate mapping; never infer a different metric from raw beam positions.

Frozen settings: seed42,150epochs,beam20,eval_interval5,eval_start75,early_eval_epochs[25,50,75,100,125],NO_EVAL=True,SKIP_TEST=False,per-rank train batch4096/global16384,inference batch1024. All non-path and non-data-derived configuration values must match the exact Issue253 run. Vocabulary and EOS are data-derived, not tunable: EOS=max(raw SID entry)+n_user_tokens+2; vocab=EOS+1. Recompute from each exact SID; do not force an old arm's vocabulary. Require exact first-three-code SID re-export and collision<0.4 before Stage3. No gate waiver.

External recommendation-validation inference uses the identical final checkpoint, item mapping, beam20, decoder and Recall/NDCG definitions, replacing only the split with valid.parquet. Its fixed total is measured from that immutable file. Preserve per-example identity and target rank for paired validation comparison. Never mix source/checkpoint generations.

## Test reuse disclosure

The complete fixed test has already been used by Issues252/253/254 and the operator has seen aggregate historical results. Issue255 additionally reported structure-test diagnostics. These are prior exposures, not a fresh holdout. Do not read #254 concrete test-error item lists/cases/derived features to select this issue's hypotheses. Historical aggregate scores and identity audits are allowed, but are not causal evidence and do not imply a verified TIGER Recall.

Each new formal candidate may access recommendation test only through the authorized final completed Stage3 test. Record the access, score, source/SID/checkpoint/run, purpose and subsequent selection in TEST_ACCESS.json and immutable per-round evidence. Further raw test replay for independent metric verification is recorded separately. A failed score is the issue's loop trigger; the next concrete change must be justified by training/validation diagnostics, not test sample errors. Prefer no further structure-test access until final transparent reporting.

No genuinely unvisited confirmation dataset is presently identified. Previously tested rows cannot be relabelled independent by taking a subset. Unless a new independent dataset is actually obtained, final success is a repeatedly used exploratory fixed-test threshold, not an unbiased generalization estimate.
