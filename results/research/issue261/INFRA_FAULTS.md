# Infra observations / faults for issue #261

## FROZEN mode

No public framework, asset, admission, evaluator or training source was modified by this issue.
The analysis only calls existing read-only interfaces.

## Historical source/config binding

The candidate source/config hashes match the archived bindings for all eight arms.

Three historical receipts (100, 101, 102) fail `archive_audit.validate_binding` only on
`infrastructure_sources` because later public-infra commits changed files that were imported
by the 2026-11-24/25 runs:

| arm | changed historical infra paths |
|---|---|
| 100 | `tools/issue252/archive_audit.py`, `tools/issue252/training_admission.py`, `tools/research/stage2/anticollapse/floor.py`, `tools/research/stage2/anticollapse/patch_tree.py` |
| 101 | `tools/issue253/static_features.py`, `tools/research/stage2/anticollapse/floor.py`, `tools/research/stage2/anticollapse/patch_tree.py` |
| 102 | `tools/issue253/static_features.py`, `tools/research/stage2/anticollapse/floor.py`, `tools/research/stage2/anticollapse/patch_tree.py` |

Candidate source/config hashes are unchanged, the checkpoint/SID/export/archive/R2 recall hashes
match, and the same frozen R2 candidate files still load.  The drift does not change the
mechanism under test.  It is reported rather than repaired because `FROZEN` forbids public-infra
edits.  Metrics for these three arms are computed from the archived candidate source and
checkpoint; the `archive_binding` status is recorded as `FAIL`/`DRIFT` in `INVENTORY.json`.

## Missing / unavailable historical measurements

- Stage3 validation-only Recall/NDCG artifacts were not persisted for these runs; Stage3
  directories contain only test final/predictions.  This is `NA` with the directory evidence,
  not a substituted test value.
- Per-module historical gradient records (encoder/GNN/gate) were not written for the full run;
  only aggregate and level-codebook gradient norms in `training_metrics.jsonl` are available.
  Endpoint-only diagnostics are labelled as such.
- Any full-table curvature/reconstruction metric is computed with the frozen independent
  three-expert metric on the actual SID/checkpoint; a five-expert native value is not mixed in.
