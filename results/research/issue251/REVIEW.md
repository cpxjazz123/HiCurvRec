# Independent review before controlled proof

`AttentionReview251`: no findings; token/axis/mask/gradient routing correct; execution still pending.

`FullRouteReview251`: four actionable findings, all require repair before accepting runtime evidence:

1. `validation_entry.source_binding` included issue250's wrapper instead of the actual issue251/framework launcher.
2. Training parquet was an input-manifest hash but not verified/recorded by checkpoint binding.
3. Trainer recomputed provenance only at final save, allowing new disk source to be mistaken for executed source.
4. Rank0-only calibration used8 items against256 codewords, advancing its rotating floor history before other ranks.

The first controlled launch was cancelled on receipt of this review. Workers exited by SIGTERM before model-stage evidence; its log is preserved in `runtime_cancelled_review/logs/controlled_validation.log`. It is not counted as a successful run or experiment.

Regression coverage: `tests/test_execution.py` and `tests/test_provenance.py`; red output in `RED_REVIEW.log`. Fixes must bind the actual launcher and input, enforce capture-before-execution/save consistency, and select a full-codebook-width calibration batch without changing frozen learning settings.
