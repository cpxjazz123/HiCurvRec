ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION

# Iter29 S06 Implementation Plan — Agent B

## Recommendation

Implement one shared, CPU-capable closed-form mapping function and have both the no-argument JSON runner and Stage2 loader call it. Treat the persisted `branching` array as the existing behavior-producer output corresponding to contract semantic `behavior_branching`; consume the residual input only from the exact plural JSON key `raw_residual_medians`. The absent plural key is a hard error even if `residual_norm` or another scale field is present. Preserve S03's historical-method/value-provenance-only limit. This is a mapping-only scientific delta; path isolation, provenance serialization, validation, labels, and checks are implementation hygiene.
This plan incorporates the amended canonical S05 round-2 scope: the in-process SID-metric reporting migration is authorized strictly as policy/hygiene, separate from the mapping delta.

## Canonical registration

Parent and sole direct control: iter26, parent commit `612a5a41dfe524205b6afa46370ad0dce377882`. Keep layer order `[L0,L1,L2]` and inputs exactly:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```

For each layer:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

Store these exact double-precision results/intermediates:

```text
x_l = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y_l = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u_l = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
closed_form_c_l = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

No scale normalization, clipping, retuning, fallback, schedule, or additional mechanism. The pre-existing Sinkhorn rule `effective_eps = sk_eps * (c / c_cyclic_max)` and its configured parameters stay untouched; its realized epsilon can change only as a mediated consequence of the new fixed `c`.

## Minimal file-level patch plan

| File | Change |
|---|---|
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/compute_closed_form_curvature.py` (add) | Add a no-CLI CPU runner and a reusable pure mapping function. Require an input mapping with exactly three-element `branching` and explicitly present `raw_residual_medians` arrays; never query `residual_norm`, `layer_norms`, or aliases. Reject non-arrays/wrong lengths, booleans or nonnumeric entries, NaN/Inf, `B < 0`, and `m_raw <= 0`; also reject nonfinite or unsupported computed/stored output. Compute the four formula stages once using the registered constants and return the intermediates, mapping identity, constants, and `closed_form_c_l`. The runner reads/updates the existing JSON record in place, preserving approved values and provenance; it removes only the legacy residual alias, `raw_capacity`/`branch_mid`, and iter26 mapping fields (`s_l`, `z_l`, old `closed_form_c_l`, `closed_form_c_base`, `closed_form_alpha`, `m_min`, old `closed_form_source`). It writes the new `x_l`, `y_l`, `u_l`, mapping identity/constants, and exact vector. No CLI options, environment overrides, recalibration, or call to the iter26 behavior writer. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/computed_behavior_branching.json` (update through that runner) | Retain the approved `entropy`, `branching`, `raw_residual_medians`, and original behavior source metadata. Preserve behavior branching and residual calibration as separate provenance records. In the behavior record, identify the iter12 behavior producer and its actual inputs (iter8 raw SIDs plus Stage0 `train.parquet`); explicitly record that the checkpoint path appearing in the metadata was not loaded by that producer. Add a distinct `raw_residual_provenance` record with iter10 calibration script and paired log, source label `iter1`, step `100000`, historical baseline checkpoint path, Stage1 embedding path, and the raw-median measurement meaning. State the S03 boundary: historical method/value provenance only; the old calibration checkpoint and iter8 raw SID export are unavailable at their referenced paths, historical hashes were not recorded, and this is neither current checkpoint replay nor historical byte-identity proof. Do not represent normalized `[0.001,0.932889,1.0]` scales as an alias/input, or imply one producer supplied both vectors. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py` (update) | Make `_load_closed_form_curvatures()` load the existing input JSON and the unchanged S04 `logs/mechanism_contract_iter29.json`, call the shared mapping function, and fail closed if either required input or the contract is missing/invalid. Validate input, stored `closed_form_c_l`, and contract `final_curvature_values` as three finite supported values; compare recomputed output against both stored and contract vectors with a declared tight absolute tolerance (`1e-9`, zero relative tolerance). Return only the recomputed result after both comparisons pass; never trust the stored value alone. Delete the unreferenced `_load_layer_norms()` historical fallback and obsolete `CLOSED_FORM_C_BASE`/`CLOSED_FORM_ALPHA` constants and old-map docstrings/iter16 labels. Retarget current curvature logging identifiers to iter29. At the existing checkpoint SID-reporting boundary, switch the import/call from `evaluate_sid_quality_full(sids, EMB_NPY)` to `evaluate_sid_quality(sids)`, preserving cadence, raw SID generation/writes, warning-only metric exception handling, training behavior, and final 4-token Stage3 export. Preserve model creation, intentional `[1.0] * N_LAYERS` residual-layer argument, `_fixed_c` buffer path, step checks, warm-start, input paths, loss, optimizer, Sinkhorn parameters/rule, seed, steps, and export behavior. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_config.py` (update) | Set `_CONFIG_DIR` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29`; set `MECHANISM_NAME` exactly to `iter29_bounded_rational_additive_mapping`; route `RQVAE_OUT_DIR`, `SIDS_NPY`, and `ITEM_SIDS_JSON` beneath that short iter29 result subtree. Keep the full label for mechanism identity, but never append it to the result directory. Leave all Stage0/Stage1 input paths, model/runtime/GPU/seed settings and overrides unchanged. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/run_stage3_iter29.py` (add) | Copy the iter26 wrapper pattern, changing only `CODE_PATH` to iter29 `item_sids.json`, `RQVAE_VARIANT` to the full label `iter29_bounded_rational_additive_mapping`, and `LOG_PATH`/`SAVE_PATH` to the short paths `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/` and `/ckpt/`. Keep launcher bookkeeping/dispatch pattern. Do not edit Stage3 trainer, settings, seed, evaluation, or protocol. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/mvg_check.py` (update) | Retarget the expected vector and labels; implement the complete fixed-curvature FCCR-1 checks listed below using one warm-start checkpoint and one real batch. Keep the check no-argument and do not perform full training. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/modules/sid_quality.py` (update) | Complete only the canonical S05 round-2 reporting cutover. Remove the subprocess/temp-file path `evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, and now-unused `subprocess` import. Remove `hitrate_k50` from `SidMetrics`, the NaN placeholder/metric construction, finite-value validation, formatting, and report serialization; update stale HitRate/gate claims in module/function docstrings. Retain the existing in-process SID-only `evaluate_sid_quality(sids)` and its full/per-layer Gini, unique SID count, item count, L0/L1 unique-pair count, and `H(L1|L0)` descriptions. Make the existing `should_early_stop(metrics)` body unconditionally return `(False, "")` with no type/metric validation or gate logic; do not add a caller. Preserve warning-only exception behavior at the trainer boundary and retain unrelated report APIs such as `write_quality_report` unless directly invalidated by removal of the HitRate field. |

**Conditional stale-file deletion, iter29 `scripts/` only:** remove copied `run_stage3_iter4.py`, `run_stage3_iter7.py`, `run_stage3_iter8.py`, `run_stage3_iter11.py`, `run_stage3_iter18.py`, `run_stage3_iter27.py`, `calibrate_residual_scales.py`, and `grad_check.sh` only if still unused. Delete `sid_metrics_any.py` as the explicit S05 round-2 cutover only after the active trainer caller has migrated to `evaluate_sid_quality(sids)`. Retain `grad_check.py`, `export_sids_for_stage3.py`, and the behavior JSON. Do not edit unrelated modules, Stage1, Stage3 trainer/settings, or other files; the sole newly authorized module edit is `modules/sid_quality.py`. Do not modify the canonical contract or other S00–S05 artifacts.

## MVG implementation contract

Use the exact iter8 warm-start checkpoint and one deterministic real batch. Ensure candidate and direct-control models use the same checkpoint, same batch and identical non-curvature weights; the only difference in the counterfactual is fixed `c`. Direct-control vector:

```text
iter26 = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

The no-argument `mvg_check.py` must report evidence for all FCCR-1 layers:

1. **Formula/domain:** the training loader's recomputed candidate equals the registered vector, is finite and inside the manifold-supported `[0.05,1.50]` range; three layers/order and both vectors are validated.
2. **Fixed state/optimizer:** each layer has a non-gradient `_fixed_c` buffer, no trainable curvature/`c_layer_scale` parameter exists, and no curvature buffer identity appears among optimizer parameter-group identities. Do not require curvature gradients.
3. **Invariance:** compare `get_c()` at steps `0, 25k, 50k, 100k`, in both train and eval modes, and after optimizer updates; every value remains within `1e-6` of the registered vector.
4. **Model gradient health:** from the warm-start model/batch, require finite total loss with `requires_grad` and `grad_fn`; explicitly call `loss.backward()` and verify finite, nonzero gradients reach intended trainable model parameters. Preserve/check the existing nonzero-gradient path as appropriate. Perform the existing small bounded optimizer-step exercise, confirm trainable model weights update, then confirm all fixed curvature buffers remain unchanged.
5. **Direct activation counterfactual:** build fresh candidate and iter26-control models from the same checkpoint and compare non-curvature state tensors to establish matching weights. In eval mode, feed the identical batch and require a finite difference above numerical tolerance in at least one preregistered curvature-consuming observable (prefer first-layer Poincaré distance geometry, or semantic assignment/quantization or behavior output). Emit the actual candidate/control values and delta. If the direct effect is absent, report MVG FAIL/inactivity; do not retune or rescue this iter29 mapping.

## Bounded verification plan (for the orchestrator after Judge C approves S06)

- Invoke the no-argument new mapping runner once with the approved CPU interpreter; inspect the JSON output against the exact `x/y/u/c` values above and confirm no normalized/legacy fallback field was consumed or reintroduced.
- Use a throwaway in-memory/import-level check (not a committed test and without changing the canonical input file) to verify the missing `raw_residual_medians` error even when `residual_norm` is present, plus wrong length, NaN/Inf, negative `B`, and nonpositive `m_raw` rejection.
- For the separately authorized reporting hygiene migration, run a bounded in-process smoke on a deterministic in-memory `(N,3)` SID sample: call `evaluate_sid_quality`, verify formatting/`to_dict` retain the supported SID-derived descriptions and contain no HitRate field, and verify `should_early_stop` returns exactly `(False, "")` without validating metrics. Confirm the trainer call boundary uses this evaluator while retaining its warning-only handling; no standalone test or training run.
- Run the official no-argument contract preflight and deliberation gate from the iter29 directory. Recheck the locked input hashes. Before any later launch, follow root `CLAUDE.md` and verify `RQVAE_OUT_DIR` resolves under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`.
- Run the bounded `scripts/mvg_check.py` on the one warm-start checkpoint/real batch; require explicit `loss.backward()`, optimizer exclusion/invariance evidence and matched iter26 counterfactual evidence before MVG PASS.
- No project-wide suite/build/formatter, permanent test absent a genuinely uncertain edge, Stage2/Stage3 launch, or training is part of S06. Stage2 remains unauthorized until later adjudicated gates.

## One-factor rationale, risks, and rejection criteria

The only scientific change is the deterministic per-layer mapping from unchanged branching and explicit raw residual medians to fixed `c`. JSON provenance separation, fail-closed checks, iter29 result routing, labels, the Stage3 wrapper, and the round-2-approved in-process SID reporting migration are implementation/path/policy hygiene, not added scientific mechanisms. The reporting cutover removes only the forbidden nondiscriminating HitRate subprocess path and retains descriptive SID-derived statistics; it does not change SID generation, cadence, warning handling, or final export. `sk_eps=0.05`, `iters=3`, and `effective_eps = sk_eps * (c / c_cyclic_max)` remain as inherited; a changed realized epsilon is only a consequence of `c`.

Risks: accidentally reading legacy `residual_norm` because it currently equals the raw vector; accidentally rerunning a writer that drops `raw_residual_medians`; conflating behavior metadata's iter8 checkpoint path with a checkpoint actually loaded for branching; overstating missing-historical-byte/replay evidence; accepting stale stored `closed_form_c_l` without independent recomputation; iter26 result paths/labels leaking into outputs; changing the inherited Sinkhorn rule while retargeting source; or retaining an active HitRate/subprocess reference after migrating the metric caller. Keep the warning-only SID-metric failure behavior and final SID export failure semantics unchanged. These are prevented by the strict key, provenance, contract-vector, caller-before-delete, and path requirements above.

Self-reject this plan if canonical S00–S05 or the source packet is superseded, if the approved plural raw key or required provenance record is unavailable/invalid, if implementation requires a fallback/recalibration or any non-mapping mechanism/protocol change, if the Stage3 trainer/settings must change, or if the matched-control direct signal cannot be made measurable without changing registered constants/equation. Also reject any reporting migration that changes SID generation/results, lets metric collection stop training, retains a HitRate gate/field, deletes the active helper before caller migration, or touches another iteration tree. A counterfactual activation failure means do not proceed to Stage2 under this specification; follow the S08 feasibility/abort adjudication rather than retuning.

## Autonomous next action

Submit this independent candidate only. Judge C should use the amended S05 round-2 diff/Judge plus canonical S00–S04 evidence, adjudicate A/B, then write the canonical `logs/implementation_plan_iter29.md` and S06 decision. The orchestrator applies only that judge-approved plan once, including caller migration before helper deletion; this candidate makes no source edits and authorizes no checks or GPU/training run.
