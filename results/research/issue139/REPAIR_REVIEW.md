# Issue #139 — code repair and verification boundary

## Status and source ownership

Status: **CPU_REPAIR_VERIFIED; FORMAL_ADMISSION_NOT_RUN; SPEC_REFREEZE_REQUIRED**.
This is a code repair, not a completed five-arm experiment and not a training
receipt. No Stage2/Stage3 job was launched, cancelled or restarted by this repair.

Work began from `9e44096b82cbdc819dbed0f4b5f28cb040202312`. During repair,
`87f2563eeeedda6fa839d61d98f0d5eee84e6682` added the five named arms and
changed MCP liveness handling. Those additions were reconciled, not overwritten
with the earlier tree. The five names/variants, mixed-model package export,
result-directory naming and A mixed auxiliary intent are preserved.
`tools/genrec-mcp/server.py` is untouched. The frozen
`TIGER_RQ-VAE` baseline is untouched.

The concurrent commit describes an F0 dispatch/restart. Its live status was not
independently checked or changed here. Its results, if any, belong to its own
source snapshot and must not be labelled as results of this repaired source.
In particular, a prior F0 calibration is not automatically reusable.

## Repairs and independently reproduced failures

| Surface | Defect reproduced before repair | Repair and regression evidence |
|---|---|---|
| AF3 incremental state | A changed endpoint's soft degree affected incident edges outside the old update set; a small path yielded cached 0.750961 versus full 0.889471 | Separate probability, degree and triangle dependency closures; bounded sparse triangle products; compare to an independent dense oracle, repeated updates and rollback |
| Feasible region | Candidate Brier used AF3 expectation instead of probability, and degree targets differed from the round start; NaN could pass | Shared class-balanced Brier and original-graph node-degree error; identical definitions at start/candidate; finite/domain validation |
| Alpha selection | Ancestor overrides disagreed with committed fallback; exceptions could leave trial cache/table state | Override-aware lookup through the same hierarchy; changed-SID dependency set; transactional rollback including failed final commit |
| Resolution accounting | Dense counts could allocate the full 256-cubed identifier space per candidate | Count occupied SID tuples only; preserve the original diagnostics |
| Quantizer gradients | Commitment detach/weighting routed gradients incorrectly; c=0 selected-code loss lost the canonical factor | Codebook term plus beta times commitment, correct stop-gradients, common canonical point cost; independent value/gradient and one-step update tests |
| Shared identifiers/state | Rank-local KMeans could diverge before residual initialization; alpha table was omitted from deployable state; final checkpoint tried tensor operations on a dict | Rank-zero initialization/broadcast before each next residual, preserve initialized prefixes; versioned extra-state and tensor/non-tensor snapshot copying; restore/export SID equality |
| Graph and readout | Compressed graph positions were treated as catalog IDs; each consumer could refit its own calibration; float32 failed strict calibration precision | Reconstruct and verify canonical TRAIN mapping; immutable F0 epoch-1500 record with graph/support/input/source identity and snapshot digest; float64 chunked calibration and readout |
| Support and outer-loop schedule | The stratified label did not enforce explicit degree quotas; ties and node counts were ambiguous; wall-clock truncation changed the sweep | Deterministic TRAIN-degree proportional quotas and largest-remainder ties; graph-row validation; no unapproved 150-second prefix cutoff |
| A integration | An unused flag did not establish active supervision; graph pools could use wrong IDs or mark observed edges/self-pairs unrelated | A adapter connects the four existing terms to the real minibatch loss, mapped TRAIN-only pools, fixed ramp and measured 0.10 encoder/codebook gradient cap |
| Concurrent A geometry | The already-squared mixed cost was squared again; a tangent residual was passed as a ball point | Softmax uses the cost once; map the tangent residual at every auxiliary level. Independent expmap/acosh oracle exposed probability errors 0.0780 and 0.6670 in the concurrent implementation, then passes for values and gradients |
| Local prefix lifecycle | Previously learned explicit entries remained active after support disappeared, despite an “inherits ancestor” log | At round start, L/E fail explicitly before selection for unsupported existing entries; reject ancestor candidates invalidating existing entries. Automatic table pruning is not silently invented |

The A auxiliary readout retains the concurrent implementation's *tangent-space
expected-codeword subtraction*. The hard quantizer still uses its native
hyperbolic residual. A's existing level-1 auxiliary kernel maps the full
expected 32-dimensional codeword into its original hyperbolic kernel; it must
not be described as the same joint E16 x H16 distance.

## Executed checks

The same test suite was run in separate processes for the base implementation
and all five named copies, avoiding Python's shared `model` import cache.
These are repeated executions of the same 115 tests, not 690 distinct tests.

| Tree | Result |
|---|---|
| TIGER_mixed_rq | 115 passed; 24 subtests passed (3.77 s) |
| TIGER_mixed_rq_F0 | 115 passed; 24 subtests passed (72.27 s) |
| TIGER_mixed_rq_G | 115 passed; 24 subtests passed (76.74 s) |
| TIGER_mixed_rq_L | 115 passed; 24 subtests passed (77.41 s) |
| TIGER_mixed_rq_E | 115 passed; 24 subtests passed (76.25 s) |
| TIGER_mixed_rq_A | 115 passed; 24 subtests passed (71.49 s) |

The five copy runs executed concurrently on CPU; those durations are not
training/resource-cost estimates. Runtime: isolated Python 3.12, PyTorch
2.15.0+cpu and pytest 9.2.0, not the prescribed HPC Stage2 environment.

The six real-`main()` CPU fixtures use eight synthetic embedding rows and
one selected epoch: F0 checkpoint/export restoration; active A at epoch 1501
including the actual recorded augmented loss; F0 epoch-1500 calibration with
matching snapshot digest; G's scheduled reuse/outer loop; and missing-F0
preflight rejection for G and A. They do not read actual TRAIN/VALID/TEST data.

Reproduction (substitute each tree; this is a CPU test command, not a launcher):

```bash
python -m pytest -q -W ignore::pytest.PytestReturnNotNoneWarning \
  stage2_RQ-VAE/TIGER_mixed_rq/tests \
  --ignore=stage2_RQ-VAE/TIGER_mixed_rq/tests/test_quantizer_ddp_regression.py
```

The common mixed-cost symbolic reference was also executed with the repository's
`tools.research.stage2.reference`: 16 samples, maximum value difference
1.3877787807814457e-17, maximum gradient difference 1.1102230246251565e-16,
both gradchecks passed, coupling check passed. This covers that declared cost
surface only, not complete mathematical correctness, the A loss or admission.

```bash
PYTHONPATH=stage2_RQ-VAE/TIGER_mixed_rq python \
  -m tools.research.stage2.reference --tree stage2_RQ-VAE/TIGER_mixed_rq
```

All six source/test trees passed Python compilation. File hashes and repeat-run
results are in `repair_verification.json`. Independent agents wrote the
regressions; the main agent made production repairs. No test is counted as a
pass merely because its implementation was inspected.

## Explicit limits and required follow-up

The attempted real two-process CPU Gloo check failed while creating its process
group with `Operation not permitted` (socket restriction). It was not retried
with a bypass. Both real-DDP tests, including the subsequently added active-A
gradient-average/update oracle, are excluded from the green CPU counts above.
The new active-A DDP test was syntax-checked only. Mocked broadcast/collective
tests do not substitute for real DDP. Four-rank GPU and actual TRAIN entry
validation remain **NOT_RUN**.

The frozen math files are preserved, not silently rewritten or stamped
approved. They require review/refreeze before another formal admission:
the commitment/codebook detach prose conflicts with the displayed loss;
the structural summations need explicit exclusion of the edge endpoints;
the 150-second truncation and proxy “no active L3” statement do not match the
Issue's complete-sweep protocol; the exact support quotas must be registered;
and A's soft assignments, tangent residual, four losses, sampling and gradient
cap are not completely registered. The local-prefix invalidation transition
also needs a reviewed policy. Current code fails closed instead of claiming
that skipping an explicit entry implements ancestor inheritance.

Historical `PRECHECK.md` and `precheck.json` are no longer current feasibility
evidence. Level-2 alpha entries use one-token prefixes (at most 256), so the old
7465-prefix number was an off-by-one-level proxy. Recount all eligible groups on
the actual F0 snapshot; remeasure full deterministic sweeps, cache construction,
guard costs, peak memory and the A active-phase overhead on the approved
hardware. Resource feasibility is **RECHECK_REQUIRED**, not FEASIBLE or a new
measured RESOURCE_BLOCKED finding.

After the specification is reviewed/refrozen, rerun the source-bound machine
checks through the existing verification/admission tools in the prescribed
Stage2 environment, including four ranks and active epochs. Do not reuse old
green receipts or silently reuse/overwrite legacy calibration. No Stage3 is
authorized by Issue #139's Stage2-only design.
