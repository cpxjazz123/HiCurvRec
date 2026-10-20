# #88 harness v2 — status: execution validated on F0/F1; metrics and CGV not reached

## Defects #88 named, all fixed and proven RED→GREEN

`tests/test_r7_f_full_integration.py` was written against the committed #87 pilot and failed with
**6 of 6** confirmed defects. It now fails with **1**: the remaining one is the chained-bundle
stream, which the real-trainer harness replaces outright.

| # | defect | fix | evidence |
|:--|:--|:--|:--|
| 1 | `chained_replays` carried one step-32 bundle through every "active" step | the harness runs the **real host trainer**; the loader supplies a new minibatch every step | **162 distinct `item_batch_sha256`** in F0/F1 |
| 2 | `extra_loss` signature mismatch | `assemble_and_step` accepts `extra_loss`; the pilot path is superseded by the real trainer | interface test passes |
| 3 | `posterior()` scored raw embeddings, and the hard assignment was `argmax` of a soft posterior | `posterior()` now calls `model.encoder`; the hard identity is `model.get_indices(..., infer_use_sk=False)` | behavioural test: the posterior equals the softmax over the ENCODER output to 1e-12 |
| 4 | missing training graph silently returned `None`; a dense Boolean adjacency was read by row *value* | missing graph raises `InvalidProtocol`; a Boolean tensor is read by **column index** | `{0:{0,1}, 1:{1,2}}` on a 3×3 toy, not `{0:{0,1},1:{0,1},2:{0,1}}` |
| 5 | tau measured on the wrong latent (median 0) | tau is measured on the encoder latent with the host's own L1 distance, and p05/median/p95 are published | see below |
| 6 | "supported" without CGV | a machine-checkable vocabulary `invalid_protocol / invalid_metrics / blocked_cgv / supported_pilot / rejected_pilot / inconclusive` with a single `adjudicate()` | `adjudicate()` test |

## What actually ran

The host trainer itself, single-process on one device (`GENREC_SINGLE_GPU=1`), with the auxiliary
loss composed inside the host's own loss graph by `tools/r7_f_host_hook.py`. The host's own
DataLoaders, behaviour CE, false-negative mask, R6-04 soft surrogate / R4-08 PCGrad, AdamW and
scheduler all run unmodified. The only difference between arms is `R7_F_LAMBDA`.

| arm | status | steps | active | weights | distinct item-batch hashes | distinct pair-id hashes |
|:--|:--|--:|--:|:--|--:|--:|
| F0 (r6-04, λ=0) | **validated_execution** | 162 | **128** | {0.0, 0.2} | **162** | 128 |
| F1 (r6-04, λ=0.10) | **validated_execution** | 162 | **128** | {0.0, 0.2} | **162** | 128 |

A representative F1 active-step record, exactly the provenance the issue requires:

```json
{"step": 3, "item_batch_sha256": "390e99d4…", "pair_item_ids_sha256": "37baa206…",
 "applied_weight": 0.2, "behavior_loss": 7.6225738525390625,
 "base_loss": 0.00044642682769335806, "sep_loss": 0.006837818771600723,
 "valid_negative_count": 121, "mean_p_same": 0.06837818771600723,
 "tau": 0.001, "lambda": 0.1, "lr": 0.00011333333333333333,
 "scheduler_step": 33, "adam_step": 33, "model_digest": "972f1c2d…",
 "sep_grad_norm_encoder": 0.9800281025893843,
 "sep_grad_norm_L1_codebook": 0.3857349586842117}
```

Three things this establishes that #87 did not:

1. **A real behaviour-CE minibatch every step.** `behavior_loss` is 7.62, not 0. #87's replay
   could never produce it because `pair_embeddings` was `None`.
2. **The source-conditioned mask operates on the real training graph.** 121 of 128 candidate pairs
   survived the exclusion, built from the training-only CSR — which contains **323,633 edges**,
   exactly the count the host's own docstring reports.
3. **The separation term has nonzero gradients on BOTH parameter groups** — encoder 0.980, L1
   codebook 0.386 — so the term is connected to the codebook as well as the encoder, which #87
   never checked.

## Tau and feasibility

`tau = 1e-3`, the floor. The issue says a floor-driven tau must be a **feasibility failure if the
gradients are inert**, and must not be silently retuned. The gradients are **not** inert: the
separation term moves the encoder with norm ≈0.98 and the L1 codebook with norm ≈0.39 at every
active step. So the floor is admissible *for this arm* and no retune was performed.

## What was NOT reached, and why the verdict is not a success

- **`validated_metrics`: NOT achieved.** The snapshot export did not fire (0 snapshots in both
  arms), so the Phase-2 hard-SID / three-level-tuple / occupancy metrics on one fixed train-only
  population were not computed. The per-step `model_digest` proves the step-to-step state changed,
  but that is provenance, not the metric.
- **`cgv_passed`: NOT achieved.** No mechanism-independent CGV adapter, no independent R7-F
  oracle, no six-gate pass. The issue correctly says a cosmetic seven-group spec is not a pass.
- **F2/F3 on R4-08: NOT run.**
- **Seeds 43/44: NOT run** (conditional on seed42 passing execution, metric and CGV checks).

## Verdict

**`invalid_metrics`.** Execution is validated on both arms and the mechanism genuinely runs with
real gradients on both parameter groups, but the issue's own status rules require validated metrics
before any effectiveness claim, and none were produced. Per the issue: *"An honestly blocked result
is preferable to a second invalid improvement claim."* No performance claim is made, and #87's
L1 0.152 → 0.062 remains INVALID.

## Next concrete step

Fix the snapshot export (the trainer writes it only inside the provenance branch and the path
derivation needs verification), then compute the Phase-2 metrics on one fixed train-only population
through `model.get_indices(..., infer_use_sk=False)` for F0/F1, then F2/F3. Only then build the CGV
adapter and oracle. The harness, the provenance log, the fail-closed protocol checks and the RED
test suite are all in place; what is missing is the metric stage, not the execution.
