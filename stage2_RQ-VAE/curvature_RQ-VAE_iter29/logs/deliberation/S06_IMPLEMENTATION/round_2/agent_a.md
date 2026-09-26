ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/source_packet.md
STAGE_ID=S06_IMPLEMENTATION

# S06 iter29 Implementation Plan — Round 2

## Decision summary and factor accounting

Implement the registered FCCR-1 mapping using only the already approved ordered inputs, and separately perform the S05-authorized reporting-policy migration. The sole scientific factor is the mapping change from iter26's formula to the S02/S04 formula. The SID reporting change is `POLICY_HYGIENE_DIFFERENCES` only: it removes a forbidden, non-SID-dependent HR@50 subprocess while preserving in-process SID descriptions and training behavior. No source edit, test, preflight, GPU check, or training is performed by this candidate.

The numerical contract in `[L0,L1,L2]` order is:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
m_raw = [1.0, 0.10941, 0.09331]
x = B / (B + 2.0)
y = m_raw / (m_raw + 0.1)
u = (x + y) / 2
c = 0.05 + 1.45 * u
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
c = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

S03 supports historical method/value provenance only, not present checkpoint replay, independent historical recomputation, or historical byte identity. Do not elevate that evidence. The data key is explicitly plural `raw_residual_medians`; the S04 schema semantic `formula_inputs` remains exactly `['behavior_branching','raw_residual_median']`. Missing plural key is fatal even if legacy `residual_norm` exists. No fallback, normalization, old behavior-writer rerun, or recalibration.

## Minimal per-file implementation plan

### 1. Add `scripts/compute_closed_form_curvature.py`

Add a no-CLI CPU-compatible script/module, following the existing no-argument script convention. It reads `scripts/computed_behavior_branching.json` relative to `__file__` and rewrites that same JSON only after validating the complete input and successfully computing the output. Keep one authoritative formula function that the training loader can import and use; do not duplicate the equation in the consumer.

- Require exact input fields `branching` and `raw_residual_medians`; fail closed on a missing key before any write, regardless of `residual_norm` or other fields.
- Convert to numeric float64 vectors, require length exactly three, finite values, `branching >= 0`, and `raw_residual_medians > 0`. Preserve `[L0,L1,L2]` ordering; reject malformed input rather than broadcasting, clipping, sorting, or repairing it.
- Evaluate the exact constants `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; retain `x_l`, `y_l`, `u_l`, and `closed_form_c_l`. Require all results finite and within the contract-supported curvature interval. No threshold, fitting, cross-layer statistic, trainable value, or schedule.
- Preserve the input record's independent provenance sections and unchanged approved input values. Replace only obsolete derived-map metadata (`residual_norm` alias, `raw_capacity`, `branch_mid`, `s_l`, `z_l`, old `closed_form_c_base`, `closed_form_alpha`, `m_min`, and old `closed_form_source`) with mapping-specific identity/constants/intermediates/output. Do not discard the entropy/branching or source provenance. Metadata must not conflate the behavior producer with the distinct raw-residual calibration provenance or claim the named checkpoint was loaded by the behavior producer. Include the S03 replay/hash limitation in the retained/updated raw-input source record, without presenting it as newly measured data.
- Make file replacement atomic if consistent with local script style (serialize validated payload to a sibling temporary file and replace target); never leave a truncated input if validation/computation fails. The script accepts no arguments and has no environment/CLI overrides.
- Print the mapping identity and exact output/intermediates for bounded operator inspection.

### 2. Update `scripts/computed_behavior_branching.json`

Materialize the validated two-input record and new derived fields by the writer; no manual recomputation or rerunning `compute_behavior_branching.py`. Preserve exact `branching`, `raw_residual_medians`, `entropy`, behavior source/definitions/counts, and separate raw-residual provenance (iter10 calibration code/log, source label `iter1`, step 100000, historical checkpoint and Stage1 embedding paths). Remove semantically ambiguous `residual_norm` and obsolete iter26 mapping/side-field results (`raw_capacity`, `branch_mid`, `s_l`, `z_l`, old curvature vector/constants/source). Store mapping ID/constants, three intermediate arrays and exact output. Retain provenance limits: S03 is historical method/value provenance only; current checkpoint/SID replay and historical byte identity are not established. The file is generated data, not a permanent test artifact.

### 3. Update `curvature_RQ-VAE.py`

Use narrow edits only:

- Import `evaluate_sid_quality` instead of `evaluate_sid_quality_full`; at the existing checkpoint reporting call use `evaluate_sid_quality(sids)` with no embedding path. Keep imports for SID generation and formatting unchanged.
- Preserve the existing reporting checkpoint cadence, SID generation and raw `sids_raw.npy` writes, metric formatting/print location, and separate warning-only metric exception handler. Metric errors remain warnings and never stop training. Do not change the outer SID generation failure behavior or final 4-token Stage3 export.
- Replace `_load_closed_form_curvatures()`'s trust of the saved vector with strict JSON validation plus the one shared formula function from the new script/module. Require explicit `branching` and `raw_residual_medians`; never access `residual_norm` or another residual fallback. Recompute from the input values and constants, validate three finite values, supported range, and exact layer order; compare recomputation with the stored `closed_form_c_l` and S04 `mechanism_contract_iter29.json` `final_curvature_values` within a documented tight float tolerance (e.g. absolute 1e-9 for stored decimal values). Reject missing, malformed, nonfinite, out-of-range, or discrepant values. Load the contract from the iteration-local `logs` path, not a copied constant that could drift. This runtime guard supplements S07 preflight and does not reinterpret provenance.
- Remove only obsolete mapping constants `CLOSED_FORM_C_BASE`/`CLOSED_FORM_ALPHA`, the unreferenced `_load_layer_norms()` fallback helper, obsolete iter16 mapping docstrings/comments, and iter26/old-map source labels that claim to describe the live formula. Retain the configured curvature invariance steps/tolerance, all constructor/warm-start/model/loss/optimizer/Sinkhorn/train loop behavior, and fixed-buffer checks. Make current diagnostics identify iter29 mapping without renaming unrelated legacy log protocol fields unnecessarily.
- Preserve existing literal residual initialization passed to `RqVae`, including `residual_layer_norms=[1.0] * N_LAYERS`; deleting the unused fallback function must not turn this into a new input path or alter runtime values.

### 4. Update `curvature_config.py`

Change only iteration identity and Stage2 result destinations:

- `MECHANISM_NAME = 'iter29_bounded_rational_additive_mapping'`.
- `_CONFIG_DIR` is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29`.
- Retarget `RQVAE_OUT_DIR` beneath that short iter29 result subtree, and retarget `SIDS_NPY` and `ITEM_SIDS_JSON` to the same short iter29 tree. Keep `RAW_SIDS_NPY` derived from the iter29 `RQVAE_OUT_DIR` as it is currently derived.
- Keep all Stage0/Stage1 input paths and all seeds, GPU/runtime, model, worker, and environment settings unchanged. Do not introduce arguments or environment overrides. Confirm routing obeys root `CLAUDE.md` Stage2 product-path rule: no training artifacts under the code iteration subtree.

### 5. Add `scripts/run_stage3_iter29.py`

Copy the minimal wrapper shape from iter26, changing only the identity/paths: read iter29 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`, set `trainer.RQVAE_VARIANT` to the full `iter29_bounded_rational_additive_mapping`, and set logs/checkpoint paths to `results/stage3_T5Train/curvature_RQ-VAE_iter29/{logs,ckpt}/`; update launcher script/log fields through the existing trainer wrapper interface. Preserve Stage3 trainer import, launch behavior, code, settings, seed, epoch count, beam, n_eval, and evaluation. No trainer edits.

### 6. Update `scripts/mvg_check.py`

Retarget the existing one-checkpoint/one-batch MVG harness to the active iter29 training module and exact candidate vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`; preserve the locked iter8 warm-start path, model setup, one deterministic 640-example transition batch, and training settings. Extend checks as bounded FCCR-1 contract evidence:

- Formula layer: exact recomputed vector within tolerance; finite values and valid range.
- Fixed-buffer layer: each layer has `_fixed_c`, no curvature `nn.Parameter`/`c_layer_scale`, buffer is not in optimizer parameter groups, and no curvature gradient is required.
- Time/invariance layer: `get_c` and `_fixed_c` unchanged across train/eval and steps `(0, 25000, 50000, 100000)`; unchanged after optimizer updates.
- Model-gradient layer: one real forward on the selected batch; total loss finite, `requires_grad` and `grad_fn` present; use `loss.backward()` explicitly and establish finite nonzero gradients on intended trainable model parameters; do not impose gradient on fixed curvature. Confirm curvature regularization remains zero and do not add a new behavior-loss threshold as a gate beyond the inherited contract checks.
- Counterfactual layer: build two models from the same iter8 state and same batch, differing only in fixed curvature (candidate vs iter26 control `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`). Compare a preregistered curvature-consuming signal (assignment, geometry/distance, or associated output/loss) and require a measurable finite difference. Explicitly keep all non-curvature weights and settings identical. If this effect is absent, record MVG failure/inactivity; do not tune the mapping.

The MVG is not full training and not an independent science factor. Report the exact vectors, selected signal/difference, gradient health, and invariant snapshots as evidence.

### 7. Update `modules/sid_quality.py` — policy/hygiene only

Keep `build_sids_for_corpus`, the in-process `evaluate_sid_quality(sids)` calculation, and its SID-derived metrics: full Gini, three per-layer Ginis, unique full SID count, item count, L0/L1 unique pairs, and `H(L1|L0)`. Make only the bounded S05 changes:

- Remove `subprocess` and any import used only by temporary SID-file execution/parser; remove `evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, their `/tmp` write/command construction/timeout/error parsing path, and `hitrate_k50` from `SidMetrics`, `to_dict` output, NaN placeholder, validation, formatting, and `write_quality_report` serialization.
- Rewrite module/function comments to accurately say these are SID-derived descriptive metrics, no HR@50, and no outer/inner gate. Remove stale baseline-HR/gate assertions; keep non-HR SID descriptive baselines only where their existing helper/report use remains appropriate. Do not redesign or delete unrelated report API merely for cleanup; if retained, its payload must contain only retained fields and descriptive status, never HR@50.
- Implement exactly `def should_early_stop(metrics: SidMetrics) -> tuple[bool, str]: return (False, "")`. No validation, type checks, metric branches, or decision side effects in this policy helper. Preserve metric validation inside `evaluate_sid_quality` itself where it protects genuine computation.
- Keep metric formatting limited to the retained values and compatible with the existing checkpoint print boundary.

This does not touch SID production, input values, training or step count, stopping conditions (other than eliminating nonexistent descriptive-metric gates in the mandated unconditional helper), evaluation or results.

### 8. Delete `scripts/sid_metrics_any.py`

After item 3's caller/import migration is in place, delete this iter29-local script. No replacement CLI, relocation to `/tmp`, or change under another iteration. Remove only after confirming no active iter29 import/reference remains. The standalone script's deletion is authorized specifically by the S05 policy/hygiene decision.

### 9. Conditional S05-authorized stale-file removals

Inspect active iter29 references, then remove only the specifically authorized obsolete copied wrappers `run_stage3_iter4.py`, `run_stage3_iter7.py`, `run_stage3_iter8.py`, `run_stage3_iter11.py`, `run_stage3_iter18.py`, `run_stage3_iter27.py`, `calibrate_residual_scales.py`, and `grad_check.sh` if present and unused. The existing inventory shows these copied files; the current S06 contract declares them stale/unauthorized in this subtree. Retain `grad_check.py` and `export_sids_for_stage3.py`. Do not remove broad helper/API surfaces or touch other iterations.

## Locked non-changes

No edits to another iteration, Stage1 or Stage3 trainer/source, module losses, optimizer, inherited Sinkhorn rule or parameters (`effective_eps = sk_eps * (c / c_cyclic_max)`, `sk_eps=0.05`, `iters=3`), architecture, training steps, seed, checkpoint/warm-start, input identity/value/order, Stage2 reporting cadence, SID generation, raw SID writes, final export, Stage3 settings/evaluation, or protocol. No new trainable curvature, schedule, regularizer, loss, optimizer mechanism, fallback, gate, or metric-driven selection. No CLI/argparse/sys.argv/environment overrides. Do not run the old behavior writer. Exact `mechanism_contract_iter29.json` schema and values remain unchanged, including `formula_inputs`' canonical singular semantic string.

## Bounded verification after canonical plan is accepted

No permanent test is justified for the deterministic arithmetic and explicit-key guard; use short throwaway checks and the mandated stage checks only. Do not run any check, test, build, formatter, training or GPU work as the S06 candidate.

1. **Formula writer, CPU, no arguments (post-apply):** run once with the approved Python environment. Inspect the saved record's input vectors, mapping identity/constants, `x/y/u`, and `closed_form_c_l`; compare all listed values above at absolute tolerance no looser than `1e-9`. Confirm provenance fields remain separate, `raw_residual_medians` is preserved, and old `residual_norm` alias/old-map fields are absent. Do not regenerate behavior inputs.
2. **Missing-key failure (throwaway only):** invoke the shared validator against a temporary in-memory/copy payload containing valid branching and legacy `residual_norm` but no `raw_residual_medians`; require the explicit missing-key exception and prove source JSON is unchanged. Never mutate then restore the canonical JSON for this check.
3. **Routing/provenance prelaunch boundaries:** independently inspect the final code/config, require strict formula-vs-data-vs-contract comparison, confirm no fallback source access, check exact locked Stage0/Stage1 and warm-start file hashes from S01, and perform the root-required `RQVAE_OUT_DIR` path check against `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`. At S07, run the official no-argument `preflight_contract.py` and `deliberation_gate.py` only after their candidate/Judge approvals and canonical artifacts are present; require `MECHANISM_CONTRACT_PASS` and `DELIBERATION_GATE_PASS`. These are phase-authorized checks, not S06 actions.
4. **S08 single MVG (once after S07):** run the bounded `mvg_check.py` on exactly one locked warm-start checkpoint and one batch; capture exact formula output, finite loss, explicit `loss.backward()` gradients, fixed buffer/optimizer exclusion, step/mode/update invariance, and same-checkpoint/batch counterfactual against iter26. Require measurable finite difference in the selected signal. No full training or duplicate GPU run. A failure blocks progression; do not retune.
5. **SID-only behavior smoke (CPU, bounded):** on deterministic in-memory `(N,3)` SID data with at least ten rows, call `evaluate_sid_quality`, `format_metrics`, and (if retained) `write_quality_report`; observe all supported SID-derived descriptions, no `hitrate_k50`/HR field or printed HR, and no subprocess/file invocation. Call `should_early_stop` with a valid metrics object and with an invalid sentinel/object, requiring the exact `(False, "")` in both cases. This is a focused observable smoke, not a metric gate. During later Stage2 log inspection, confirm the existing cadence/fields and warning-only exception behavior remain.
6. **Stage3 wiring:** inspect wrapper constants/paths against root Stage3 path rules and the full mechanism label; do not launch Stage3 at S06.

No full pytest/build/formatter or training run. Stage2/Stage3 are authorized only after subsequent stage adjudications and official gates; no launch is part of this S06 plan execution.

## Risks, self-rejection, and autonomous next action

- **Hard rejection:** any implementation that reads `residual_norm`, normalized `layer_norms`, reconstructs/recalibrates the raw vector, silently defaults a missing plural key, or claims stronger provenance than S03 is invalid. The input must fail closed even where the alias matches numerically.
- **Formula/contract risk:** writer and consumer must call one formula implementation; avoid drift from stored metadata or the exact S04 contract values. Reject discrepancies rather than silently rewriting them at training startup. Numeric comparison tolerance must be explicit and narrow.
- **One-factor risk:** accidental differences in input order, warm-start, seed/steps, behavior loss, optimizer, Sinkhorn implementation/parameters, Stage1/Stage3 trainer/settings, or output routing confound the experiment. The inherited effective-epsilon change is allowed only as deterministic mediation of changed fixed `c`.
- **Policy migration risk:** removing too little leaves CLI/HR traces or stale invalid reports; removing too much changes SID generation/cadence/training/export or deletes unrelated report helpers. Apply the exact field/parser/caller/deletion inventory; keep exception handling warning-only and preserve all valid SID descriptions.
- **MVG risk:** don't treat fixed `c` as needing gradients; don't accept a detached total loss or no model gradient; counterfactual must isolate curvature and report a real curvature-consuming signal. No parameter retuning if inactive.
- **Artifact/routing risk:** full mechanism label and short iter29 output paths are intentionally distinct. Keep artifacts out of the code iteration directory and preserve locked upstream paths/hashes.
- **Self-rejection:** this candidate plan should be rejected if direct source review finds that it changes another conceptual variable, weakens missing-key failure, changes the required S04 contract, alters any locked protocol value, skips the counterfactual/backward requirements, omits S05's metric migration, or treats descriptive SID values as gates.

AUTONOMOUS_NEXT_ACTION=Judge C adjudicates this and the independent round-2 candidate against the updated S05 diff and primary sources, writes the canonical `logs/implementation_plan_iter29.md` plus S06 round-2 `judge.md`, and directs the orchestrator to apply the approved patch once. After that, proceed through S07 preflight and S08 single MVG under their own required approvals; do not launch Stage2/Stage3 before the later required gates and adjudications.
