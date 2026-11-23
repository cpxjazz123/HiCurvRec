# Issue 249 — infrastructure delivery

No training: the issue explicitly forbids new GNN, quantizer, Stage2 or Stage3 training.

Verification command: `bash tools/graph_infra/verify_v5.sh`.
Evidence: `VERIFY.log`, `MODULE_CHECKS.json`, `MUTATIONS.json`, `FIDELITY.json`.

| Criterion | Implementation/evidence | Result |
|---|---|---|
| C1 audit/reuse | assets/manifest.json; build_assets_v5 verifies immutable hashes | PASS |
| C2 ego batching | ego_gnn.py; 24 infrastructure tests cover masks, ID mapping, ordering and chunking | PASS |
| C3 full curvature | curvature_stream.py; 28,391,948 rows, 13 shards; exact summary agreement | PASS |
| C4 differentiable statistic | diff_sectional.py; frozen A.1 contract and independent reference | PASS |
| C5 gradient diagnostics | grad_diag.py; finite differences and chunked gradients | PASS |
| C6 hard SID evaluation | fidelity.py, quant_eval.py, heldout_eval.py; discrete/continuous reports separated | PASS |
| C7 frozen TIGER | tiger_compare.py; Stage3 table pure SID metrics recomputed; unavailable same-generation geometry NA | PASS |
| C8 persistence/API | assets/v5_gnn_geometry/, root manifest, assets/README.md, verify_v5.sh | PASS |
| C9 numerical/regression/mutation | VERIFY.log, MODULE_CHECKS.json, MUTATIONS.json | PASS |

2000 deterministic triples, one per selected centre: graph recomputation equals frozen targets exactly; hard-SID sign agreement 0.7005, continuous encoder 0.6995. Positive targets: 1776; negative: 224. Always predicting positive would achieve 0.888, so raw agreement must not be interpreted as strong fidelity. Hard-SID positive sign agreement 0.75; negative agreement 0.3080357143. Scale-aligned absolute error 2.7413730607. Alignment is sample-fitted diagnostic, not held-out performance. Raw distance distortion compares unlike scales and is not a scale-invariant quality claim.

This diagnostic uses the checkpoint-consistent re-export, not the historical Stage3 table. It neither establishes mixed-curvature superiority nor selects graphlet-dependent geometric biases. Historical Stage3 geometry remains NA.

The macro-error regression uses unequal per-item counts with opposing signed errors: item-macro MAE 5, triple-micro MAE 4. Signed means are not substituted for absolute errors.

Commit preflight exception: shared worktree contains 681 changed paths / 248517 lines, mostly unrelated historical data. Explicit guard allowance 750 paths / 300000 lines permits selective drafting only; those unrelated files are excluded.
