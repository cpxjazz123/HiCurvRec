ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/source_packet.md
STAGE_ID=S06_IMPLEMENTATION

# Iter29 S06 implementation plan — independent candidate B

## Decision boundary

Use only canonical S00–S05 and the S06 round-2 packet. The sole scientific delta from parent iter26 is the locked FCCR-1 mapping, applied to the same explicitly recorded inputs in `[L0,L1,L2]` order:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_raw = [1.0, 0.10941, 0.09331]
x = B/(B+2.0) = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = m_raw/(m_raw+0.1) = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = (x+y)/2 = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
c = 0.05 + 1.45*u = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Curvature remains precomputed, fixed, non-trainable, and time-invariant. `raw_residual_medians` is the only permitted residual data key; missing, malformed, or invalid values fail closed even if `residual_norm` or another tempting substitute exists. Preserve S03's historical method/value provenance limitation: no checkpoint replay, independent historical recomputation, or historical byte-identity claim. Do not rerun the legacy behavior producer. Parent/control remains iter26 with fixed vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.

The added SID reporting migration is explicitly `POLICY_HYGIENE_DIFFERENCES`, not a second scientific factor. It removes the forbidden project-local positional CLI / non-SID-dependent HitRate path and restores unconditional no-gate behavior, without changing SID/model/training values, checkpoint cadence, stopping steps, or outcomes.

## Minimal file-level patch

### 1. Add `scripts/compute_closed_form_curvature.py`

Add one no-argument, CPU-capable module/runner as the sole implementation of the registered mapping. It should locate its sibling JSON relative to `__file__`, accept no CLI arguments, and expose the same calculation/validation helper for the training loader to call (import by the existing iteration-local module path; do not duplicate the equation). Use only the locked constants `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50` and three ordered layers.

Validate exact vector shape/length 3, finite numeric elements, `B >= 0`, `m_raw > 0`, and finite intermediate/output values; reject malformed/missing keys rather than coerce, default, or fall back. Calculate/store `x_l`, `y_l`, `u_l`, and `closed_form_c_l` from explicit `behavior_branching` and `raw_residual_medians`. Require output in supported `[0.05,1.50]` range. The no-argument writer may update the JSON only after validating its required inputs and provenance; it must not attempt to recreate the historical inputs. Preserve approved provenance and existing behavior input record, and fail before writing if the plural raw key is absent. Do not introduce argparse/sys.argv/environment overrides.

### 2. Update `scripts/computed_behavior_branching.json`

Preserve the existing approved entropy/branching values, ordered `raw_residual_medians`, and behavior producer record, with provenance split so the two inputs cannot be misattributed:

- Behavior branching: keep the recorded iter12 behavior script/source metadata, raw iter8 SID and Stage0 train parquet paths/counts/definition; explicitly state the producer used raw SIDs plus train parquet and did **not** load the checkpoint merely because its path occurs in the metadata.
- Residual medians: record separately the iter10 `calibrate_residual_scales.py`/log provenance, source label `iter1`, step `100000`, historical checkpoint and Stage1 embedding paths, and measurement semantics (per-layer per-item residual L2 norm, median over items). Preserve that the source/run lack replay/byte-identity proof.

Remove semantically unsafe/obsolete `residual_norm` alias, old-map `raw_capacity`, `branch_mid`, `s_l`, `z_l`, old closed-form constants and old `closed_form_c_l` meaning before replacing it with the new mapping result. Retain `behavior_branching` (and entropy/history needed to understand it), explicit `raw_residual_medians`, and their independently stated provenance. Store the new mapping identity/constants, exact intermediates, and exact vector above. Do not claim these decimals came from a current checkpoint replay.

### 3. Update `curvature_RQ-VAE.py`

- Change the `modules.sid_quality` imports from `evaluate_sid_quality_full` to existing `evaluate_sid_quality`; at the existing checkpoint metrics boundary call `evaluate_sid_quality(sids)` (without `EMB_NPY`). Keep the same checkpoint boundary/cadence, raw 3-token SID inference and save, `format_sid_metrics` output, warning-only handling for metric-collection exceptions, training loop, and final 4-token Stage3 export/failure behavior.
- Replace `_load_closed_form_curvatures()`'s trust of stored `closed_form_c_l` with strict load-and-recompute validation through the new shared formula implementation. Require `behavior_branching` and specifically `raw_residual_medians`; never use `residual_norm`, normalized scales, hardcoded fallback or alias. Compare recomputed ordered values against both JSON's stored `closed_form_c_l` and the canonical S04 `logs/mechanism_contract_iter29.json` `final_curvature_values`; validate lengths, finite values, supported range, and numeric agreement (tight float64 tolerance for formula/JSON values; model buffer invariant tolerance remains the existing `1e-6`). Missing/malformed contract/input or any mismatch fails closed before model training. Formula implementation must not be duplicated in this file.
- Remove the unused `_load_layer_norms()` fallback helper (main already passes `[1.0]*N_LAYERS`; do not change that existing training argument). Remove obsolete `CLOSED_FORM_C_BASE`/`CLOSED_FORM_ALPHA` and only dead old mapping documentation. Retarget stale iter16/iter26 curvature labels/docstrings to iter29/FCCR-1 consistently without changing logic.
- Retain existing fixed-buffer construction, `FIXED_CURVATURE_INVARIANCE_STEPS` and tolerance, model/loss/optimizer/Sinkhorn and its inherited `effective_eps = sk_eps*(c/c_cyclic_max)` rule, inputs, warm-start, seed, steps, warning output, checkpoint/export behavior. No change to training or curvature-consuming code beyond supplying the locked vector.

### 4. Update `curvature_config.py`

Change only the iteration identity/output roots: `MECHANISM_NAME="iter29_bounded_rational_additive_mapping"`; `_CONFIG_DIR` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29`; route `RQVAE_OUT_DIR` beneath that short root (`.../out/rqvae/instruments`) and `SIDS_NPY` / `ITEM_SIDS_JSON` to the corresponding iter29 results tree. Keep `RAW_SIDS_NPY` under that iter29 output directory. Leave Stage0/Stage1 inputs, environment setup, seed/GPU/model/runtime settings unchanged. Do not place result arrays/checkpoints under the source iteration directory.

### 5. Add `scripts/run_stage3_iter29.py`

Copy the existing parent `run_stage3_iter26.py` wrapper pattern, changing only iteration-specific paths/label: use results `curvature_RQ-VAE_iter29/item_sids.json`, `trainer.RQVAE_VARIANT="iter29_bounded_rational_additive_mapping"`, and results `stage3_T5Train/curvature_RQ-VAE_iter29/logs/` and `/ckpt/`. Keep the Stage3 trainer/source/settings/seed/evaluation untouched; wrapper sets its launcher script/log as the parent does and keeps zero-argument operation.

### 6. Update `scripts/mvg_check.py`

Retarget description/module import/name and `EXPECTED_FIXED_C` to the exact iter29 vector. Keep the same iter8 warm-start checkpoint and parent architecture/settings, same fixed one-batch acquisition path and model weights (excluding legacy curvature parameters and injecting only fixed curvature). Complete FCCR-1 checks, without full training:

1. Formula: strict no-argument loaded vector matches expected values; finite and supported. Fixed buffer exists, `requires_grad=False`; no `c_layer_scale` curvature Parameter; verify fixed-curvature values are absent from optimizer parameter identities.
2. Invariance: `get_c()`/buffer remains equal at steps `0,25000,50000,100000`, after train/eval toggles, and after lightweight optimizer updates; no time dependence. Ensure the optimizer receives only model parameters and compare fixed values before/after.
3. Model gradient health: same one warm-start checkpoint and one real batch; total loss finite, requires grad, and has a grad function; intended non-curvature model parameters have finite nonzero gradients. Explicitly call `loss.backward()` (not just `autograd.grad`) and assert finite/nonzero gradients. This is model gradient verification, not a requirement that fixed curvature itself receive gradients. Keep the current zero curvature-regularization assertion; do not assert a new mechanism/loss.
4. Counterfactual: with identical checkpoint weights, exact same batch, and all settings identical except fixed-curvature vector, compare iter29 against declared iter26 vector. Require a measurable difference beyond numerical tolerance in a registered curvature-consuming output, such as quantizer assignment/geometry or curvature-dependent loss/output; report the signal and delta. A difference in curvature values alone is insufficient. No retuning or second training run.

### 7. Update `modules/sid_quality.py` (policy/hygiene only)

- Remove `subprocess`, `Path`/typing imports only if no longer needed elsewhere; remove `evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, the temp SID-file write, command construction, subprocess invocation, parser, and all related HitRate references.
- Remove `hitrate_k50` from `SidMetrics`, the NaN placeholder returned by `evaluate_sid_quality`, `should_early_stop` validation, `format_metrics`, and `write_quality_report` serialization. Keep `to_dict()` as the existing `asdict` API with only surviving fields.
- Preserve and report existing SID-derived metrics: `full_gini`, three `per_layer_gini` values, `n_unique_full`, `n_items`, `l01_unique_pairs`, and `h_l1_given_l0`. Keep their current computation and validation behavior. Correct stale comments/docstrings (including evaluator's obsolete HitRate/placeholder story and misleading gate/baseline comments) to say the metrics are SID-derived descriptions only and are not gates; do not change numerical definitions or baseline functionality unrelated to HR.
- Implement `should_early_stop` as an unconditional exact `return (False, "")`, with no argument-type, field, or metric validation and no gate branches. Its result must not depend on inputs.
- Retain `write_quality_report` and unrelated APIs if still otherwise appropriate; remove only HitRate field/report traces. Do not add a descriptive-metric selector/gate or use metrics to choose candidates.

### 8. Delete `scripts/sid_metrics_any.py` after caller migration

First land/confirm the training caller imports and invokes `evaluate_sid_quality(sids)` and `sid_quality.py` has no remaining dependency on the CLI, then delete only `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/sid_metrics_any.py`. No replacement CLI, `/tmp` copy, or edits to another iteration. This sequence is required because deleting the script before migrating the active call leaves the checkpoint path broken.

### 9. Conditional local cleanup, bounded

Only if the enumerated files still exist and are confirmed unused in iter29, remove copied stale `run_stage3_iter11/18/27/4/7/8.py`, unauthorized `calibrate_residual_scales.py`, and `grad_check.sh`. Do not remove compatible no-argument `grad_check.py` or `export_sids_for_stage3.py`. No broad cleanup, cross-iteration edits, Stage1/Stage3 trainer changes, or new permanent tests absent an identified genuinely uncertain edge.

## No-change lock

No changes to any other iteration; Stage1; Stage3 trainer/source/settings; loss definitions or weights; optimizer; inherited Sinkhorn rule or parameters; architecture; input values/identities; warm-start; random seed; training steps; final export; or evaluation protocol. In particular, no learnable/scheduled curvature, curvature regularizer, new optimizer/loss/Sinkhorn/behavior mechanism, gating, or SID-generation change. Preserve current warning-only descriptive metric exception behavior. Stage2 reaches the same `MAX_GLOBAL_STEPS` unless an actual invalid execution/contract failure occurs.

## Bounded verification plan (execute only after canonical S06 plan is applied)

1. **Formula/JSON CPU smoke in authorized implementation verification:** invoke `compute_closed_form_curvature.py` with no arguments using the approved interpreter. Inspect JSON vectors, mapping/constants identity, and x/y/u/c arrays against the exact values in this plan; require finite supported outputs, correct layer order, separated provenance, no obsolete residual alias or old mapping fields, and no unintended input regeneration. No project-wide tests/build/formatter.
2. **Fail-closed raw-key edge:** on a temporary copied payload only, delete `raw_residual_medians` while keeping a numerically equal legacy `residual_norm`; invoke the same loader/helper and assert a clear failure before returning/writing any curvature. Do not edit the canonical JSON for this case.
3. **S07 only:** after the canonical implementation and mandatory files exist, run the official no-argument `preflight_contract.py` and `deliberation_gate.py`; require their documented PASS results before advancing. These gates are not run at S06.
4. **S08 MVG only:** once preflight/adjudication allows MVG, run the bounded `mvg_check.py` on the existing one iter8 checkpoint and one real batch. Capture exact fixed vector, buffer/optimizer exclusion, time/train-eval/update invariance, finite nonzero model gradients from explicit `loss.backward()`, and measurable same-state/same-batch counterfactual signal against iter26. A failed activation check is not permission to retune; apply the registered abort/replan rules.
5. **SID-only metric smoke:** run an isolated deterministic in-memory `(N,3)` SID example (`N>=10`) through `evaluate_sid_quality`, formatting, and serialization/report helper. Assert surviving SID-derived values/fields are available, `hitrate_k50` is absent from dataclass output/printed text/report, and `should_early_stop` always returns exactly `(False, "")` even for malformed/non-metric input. Confirm training callsite is SID-only and warning-only collection errors remain isolated; final SID generation/export error semantics remain unchanged. This is policy verification, not scientific evidence.
6. **Before any later Stage2 launch:** as separately authorized downstream gates, verify locked input hashes against S01, official preflight and deliberation gate pass, and run root-required `grep RQVAE_OUT_DIR` check to establish the path under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`; Stage2 launch remains outside this S06 candidate plan and only follows its later adjudications. No full Stage2/Stage3 run here.

## Risks, rejection criteria, autonomous next action

- **Key/provenance risk:** accidental use of `residual_norm`, normalized scale, old alias, or reconstructed value silently changes semantic input; missing plural key is a hard error. Never rerun the legacy writer. Preserve separate provenance records and explicitly retain S03's replay/hash limits.
- **Mapping/contract risk:** stale `closed_form_c_l`, duplicate equation, wrong layer order, rounded/altered constants, ignored S04 values, or a fallback can make recorded code differ from registration. Recompute once via shared code, compare JSON and contract, fail closed.
- **One-factor risk:** altered Sinkhorn formula/settings, loss/optimizer, input, warm-start, seed/steps, architecture or downstream trainer invalidates this controlled test. Only the new fixed map may mediate a changed realized epsilon through the inherited formula.
- **Routing risk:** copied iter26 output paths or variant label can overwrite/mislabel results; validate configuration and wrapper paths before later execution.
- **Migration-order risk:** delete CLI before caller cutover breaks checkpoint reporting; migrate caller first, remove every HitRate/subprocess/parser/field/report artifact, then delete only iter29 CLI. Preserve checkpoint timing, SID writes, warning-only metrics failures and final 4-token export.
- **Scientific rejection:** any formula mismatch, provenance/key failure, fixed-curvature drift/trainability, missing contract match, or absent matched counterfactual effect blocks progression/indicates invalid implementation or inactive registered mechanism; do not tune inside iter29. SID reporting migration has no bearing on scientific result or candidate selection.

**AUTONOMOUS_NEXT_ACTION:** submit this independent plan for Judge C against the other fresh candidate and primary sources. If the Judge canonicalizes a plan, the orchestrator applies that plan once; then conduct only the bounded formula/key checks in the authorized verification stage, followed by the official S07 gates and one-checkpoint/one-batch S08 MVG before any later run. Do not edit source, run checks/training, or advance to GPU execution as part of this S06 candidate.