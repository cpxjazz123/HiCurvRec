ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION

# Iter29 S06 implementation plan — Agent A

## Decision and scope

Implement only the accepted S02/S04 FCCR-1 mapping over the existing ordered inputs, with strict explicit `raw_residual_medians` consumption. Keep curvature precomputed, fixed, non-trainable, and time-invariant. Iter26 remains the sole direct control. Do not change model architecture, training protocol, Stage1/Stage0 inputs, loss, optimizer, Sinkhorn rule or parameters, warm-start, Stage3 trainer/settings, or any other mechanism. Do not rerun the old behavior writer, recompute historical inputs, replay an unavailable checkpoint, or imply historic byte identity.

Canonical values in `[L0,L1,L2]` order:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

For each layer: `x=B/(B+2.0)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45*u`. These constants, inputs, order, and outputs are locked; there is no clipping, cross-layer normalization, schedule, or fit/recalibration step.

## File-level patch plan

### 1. Add `stage2_RQ-VAE/curvature_RQ-VAE_iter29/scripts/compute_closed_form_curvature.py`

Add a no-CLI, CPU-capable module and no-argument runner. Expose one pure calculation/validation function, e.g. `closed_form_curvatures(behavior_branching, raw_residual_medians)`, for reuse by the training reader; importing the module must not execute the writer. Keep all mapping constants in this single implementation:

```text
B_REF=2.0, M_REF=0.1, C_MIN=0.05, C_MAX=1.50
```

The validator/calculator must reject missing fields, non-list/non-numeric values (including booleans), any vector not exactly three elements, NaN/Inf, any `B < 0`, and any `m_raw <= 0`. Do not coerce malformed strings or search alternate field names. Calculate the four arrays in layer order, reject non-finite results, and require each `c` to lie in the supported `[0.05,1.50]` interval. Do not clamp invalid inputs/results. Use Python/NumPy CPU arithmetic only; no CUDA, model loading, CLI arguments, environment overrides, or dependency on Stage1 data.

`main()` reads the adjacent `computed_behavior_branching.json`; the existing JSON field `branching` is the recorded `behavior_branching` vector, and the residual argument must be obtained by direct indexing of `payload["raw_residual_medians"]`. A missing plural key must raise before any write, even if `residual_norm` is present and numerically equal. Preserve the original recorded input/provenance information, then replace obsolete prior-map outputs and write deterministic mapping identity/constants plus `x_l`, `y_l`, `u_l`, and top-level `closed_form_c_l`. Remove the old `residual_norm` alias, `raw_capacity`, `branch_mid`, and old mapping metadata (`s_l`, `z_l`, `closed_form_c_base`, `closed_form_alpha`, `m_min`, old `closed_form_source`); retain `branching`, `entropy`, and the explicit raw-median vector.

### 2. Update `scripts/computed_behavior_branching.json`

Materialize the output of the new mapping without regenerating either historical input. Keep `entropy`, `branching`, the exact `raw_residual_medians`, and the behavior-source record; remove only the ambiguous residual alias and obsolete iter26 mapping/side-field outputs listed above. Store the new map identity and constants (`B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`), the exact intermediates, and exact `closed_form_c_l` above.

Separate the two provenance records explicitly in this JSON; do not imply a shared producer:

- Preserve the existing branching `source` metadata as the behavior-branching provenance, and state its actual inputs are iter8 `sids_raw.npy` plus Stage0 `train.parquet`. Its metadata also names the iter8 checkpoint, but the branching producer did not load that checkpoint. Retain that distinction.
- Add a separate raw-residual provenance record identifying `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py` and its paired `logs/calibrate_residual_scales.log`, source label `iter1`, recorded step `100000`, historical checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth`, and Stage1 embeddings `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`. State the S03 boundary: historical method/value provenance only; the original calibration checkpoint and iter8 raw SIDs are unavailable at their cited paths, historic hashes were not recorded, and this is not checkpoint-level replay or historic byte-identity evidence.

Do not call the copied iter26 behavior writer: it overwrites this JSON without the required plural key and depends on absent iter8 raw SIDs. Any writer touching this file must validate and preserve `raw_residual_medians`.

### 3. Update `curvature_RQ-VAE.py`

Replace `_load_closed_form_curvatures()` with a fail-closed consumer of the same pure calculation function from the new mapping module (make `scripts/` importable from this file; do not duplicate the formula). Resolve input and contract paths from `Path(__file__).resolve().parent`, independent of the current working directory.

At load time:

1. Require the JSON and contract files and directly read `branching`, `raw_residual_medians`, and stored `closed_form_c_l`; validate all vectors as exactly three finite numbers and validate formula input domains.
2. Recompute the registered equation from those input fields. Require finite, supported-range output and compare the recomputed vector to both the JSON `closed_form_c_l` and `logs/mechanism_contract_iter29.json["final_curvature_values"]` in `[L0,L1,L2]` order (absolute tolerance `1e-12`, zero relative tolerance). Reject missing/malformed/inconsistent vectors; never fall back to `residual_norm`, normalized `layer_norms`, a default vector, or stale map data. Validate stored intermediate vectors if they are materialized.
3. Return only the recomputed validated vector, which is then passed through the existing `fixed_layer_curvatures` constructor argument and registered by the existing `_fixed_c` buffers. Do not make curvature a parameter or change `get_c()`.

Remove the dead `_load_layer_norms()` fallback (it is unreferenced; `main()` currently supplies `[1.0] * N_LAYERS`), obsolete `CLOSED_FORM_C_BASE`/`CLOSED_FORM_ALPHA`, and obsolete iter16 formula comments. Retarget only current iteration curvature snapshot/invariant messages and documentation labels to iter29. At the existing checkpoint SID-reporting boundary, replace the `evaluate_sid_quality_full(sids, EMB_NPY)` import/call with `evaluate_sid_quality(sids)`; preserve checkpoint cadence, raw SID generation/writes, and warning-only handling of metric exceptions. Preserve the final 4-token export and all other model/training behavior, including `C_CYCLIC_MIN/MAX/PERIOD`, `sk_eps=0.05`, `sk_iters=3`, and the inherited Sinkhorn `effective_eps = sk_eps * (c / c_cyclic_max)` behavior. Its realized epsilon may differ solely because the new `c` differs.

### Policy/hygiene migration: `modules/sid_quality.py`

Follow the amended S05 round-2 canonical scope. Keep the existing in-process `evaluate_sid_quality(sids)` and its SID-derived descriptions (full/per-layer Gini, unique SID and item counts, L0/L1 unique pairs, `H(L1|L0)`). Remove the project-local subprocess/temp-file HitRate path: `evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, `subprocess`/parser-only imports, and all HitRate reporting. Remove `hitrate_k50` from `SidMetrics`, its NaN placeholder, validation, `format_metrics`, and serialized `write_quality_report` metrics. Update stale module/function comments and docs to state that metrics are SID-derived descriptions only, not a gate. Keep unrelated report APIs only if still appropriate.

Change `should_early_stop` to the no-argument `should_early_stop() -> tuple[bool, str]` with exactly an unconditional `(False, "")` result—no metric/type validation or gate branch. The iter29 trainer has no caller of this helper, so do not add one. Preserve the trainer's current warning-only diagnostics exception boundary and do not alter training, SID save cadence, output artifacts, or final export. This is policy hygiene only, not an experimental mechanism.

After migrating the active trainer call/import and removing the helper dependency, delete only this iter29 copy of `scripts/sid_metrics_any.py`; do not create a replacement CLI or touch other iteration trees. Preserve the final SID export failure behavior unchanged.

### 4. Update `curvature_config.py`

Change only the full iteration label and result paths:

```text
MECHANISM_NAME = "iter29_bounded_rational_additive_mapping"
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29"
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments"
SIDS_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json"
```

Keep `RQVAE_CKPT_PATH` and `RAW_SIDS_NPY` derived under that `RQVAE_OUT_DIR`; `SAVE_DIR_ROOT` and `CONFIG_PATH` continue deriving from `_CONFIG_DIR` and the full mechanism label. Leave Stage0/Stage1 input paths, seeds, GPU/runtime environment, model settings, and every other configuration unchanged. No Stage2 artifact is to be written under the source iter29 tree.

### 5. Add `scripts/run_stage3_iter29.py`

Copy the iter26 wrapper pattern with the import module name updated to iter29. Set `trainer.CODE_PATH` to the iter29 results `item_sids.json`, `trainer.RQVAE_VARIANT` to the full label `iter29_bounded_rational_additive_mapping`, and `trainer.LOG_PATH` / `trainer.SAVE_PATH` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/` and `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter29/ckpt/`. Keep the Stage3 trainer import and launcher handoff pattern; do not edit Stage3 trainer, settings, evaluation, or protocol.

### 6. Update `scripts/mvg_check.py`

Retarget the script/module labels and `EXPECTED_FIXED_C` to the iter29 vector. Preserve the exact iter8 warm-start checkpoint and one existing 640-example train-transition batch. Keep `DIRECT_CONTROL_FIXED_C=[0.6145357379232853,0.5333020920777128,0.3814078098431606]` for iter26.

Complete the FCCR-1 checks with bounded work:

- **Formula/domain:** call the production loader, require its vector to match the iter29 vector, be finite and in `[0.05,1.50]`.
- **Fixed buffer/optimizer exclusion:** every layer has `_fixed_c`, its value matches the target and `requires_grad` is false; no trainable curvature/`c_layer_scale` parameter exists; no fixed-curvature buffer id occurs in optimizer parameter groups.
- **Invariance:** test curriculum steps `0, 25_000, 50_000, 100_000` and both `train()`/`eval()` modes. Require `get_c()` and `_fixed_c` to stay equal to the target within `1e-6`.
- **Model gradient health:** on one warm-start candidate model and the same real batch, require finite `output.loss`, `requires_grad`, and `grad_fn`; explicitly call `output.loss.backward()` after clearing gradients. Require finite, nonzero gradients on intended trainable model parameters. Require curvature regularization to remain exactly zero. The current `autograd.grad`-only check is insufficient as the standalone `backward()` proof.
- **Optimizer immutability/update:** perform the current bounded five AdamW updates using Stage2’s existing `lr=1e-3`, `weight_decay=1e-4`; show the non-curvature model parameters update, and verify every `_fixed_c` remains unchanged and excluded from optimizer groups after the updates.
- **Matched direct-control counterfactual:** build candidate and control models from the same iter8 checkpoint, use the same batch, and ensure all non-curvature model state (including encoder/codebooks) is identical; only the fixed vector differs. Evaluate in deterministic `eval()` mode without training either model. Require a finite measurable difference (e.g. `max_abs_diff > 1e-7`) in the actual same-input/same-codebook Poincaré distance geometry, or in the resulting semantic assignments/curvature-consuming output. The distance comparison can use the existing hyperbolic distance path for identical encoded input and codebook with each model’s fixed `c`; it verifies geometry directly even when discrete assignments happen to coincide. Keep the parent Sinkhorn rule identical; the changed realized epsilon is only the inherited downstream consequence of changed `c`. A failure to find any registered direct-output difference is a failed activation gate, not permission to retune this mapping.

Do not launch full training or construct another mechanism; this is a one-checkpoint/one-batch MVG only.

### 7. Conditional, narrowly scoped removals

After confirming no active iter29 callsites, remove only the copied stale wrappers `scripts/run_stage3_iter4.py`, `run_stage3_iter7.py`, `run_stage3_iter8.py`, `run_stage3_iter11.py`, `run_stage3_iter18.py`, and `run_stage3_iter27.py`; each is hard-coded to a different iteration and the iter27 copy even loads iter28. Remove the copied `scripts/calibrate_residual_scales.py` (not an authorized iter29 input generator) and `scripts/grad_check.sh` (the compatible no-argument `grad_check.py` remains). Keep `scripts/grad_check.py` and `scripts/export_sids_for_stage3.py` no-argument and aligned to `curvature_config`. In addition, delete `scripts/sid_metrics_any.py` only after the approved `curvature_RQ-VAE.py` caller migration and `modules/sid_quality.py` cutover above; do not create or move a replacement CLI. These are the complete authorized removals; no other cleanup is in scope.

## Separation of scientific delta from hygiene

**Only scientific delta:** iter26’s fixed per-layer curvature map is replaced by the locked iter29 equation over the same behavior-branching and raw-residual-median vectors. The effective curvature passed to the existing quantizers is consequently different.

**Hygiene/wiring only:** the explicit-key and domain checks, deterministic stored intermediates, separate historical provenance records, runtime comparison against stored/contract vectors, removal of dead legacy map/fallback code, full mechanism label, short isolated iter29 result paths, Stage3 wrapper, approved SID-metric CLI-to-in-process migration/no-gate correction, and enumerated stale copied wrappers/helpers. These changes add no mechanism and do not alter the scientific treatment. In particular, the inherited curvature-conditioned Sinkhorn rule and its `sk_eps`/iteration settings stay byte-for-byte unchanged; only its output epsilon can vary through `c`.

## Rationale and risks

- Direct indexing of `raw_residual_medians` closes the current naming hazard: `residual_norm` is a legacy alias and normalized historical scales `[0.001,0.932889,1.0]` are not valid inputs. Absence or invalid data is a hard error.
- Reader recomputation plus comparison to both persisted result and S04 contract prevents stale/copied `closed_form_c_l` from reaching training. The fixed `_fixed_c` buffers remain non-trainable; never expect them to receive gradients.
- Provenance metadata must remain two-source and historically bounded. The behavior producer used raw iter8 SIDs and train parquet, not the checkpoint path carried in its metadata; raw medians came from the separate iter10 calibration record. No current replay or historical byte identity is established.
- The direct-control comparison isolates fixed curvature because checkpoint, batch, and every non-curvature weight/config remain matched. Changed assignments/distance/loss are mediators of the single mapping change, not justification to modify Sinkhorn.
- The iter29 reporter migration is separately authorized policy hygiene: no project-local positional CLI or HitRate output remains; the existing descriptive SID metrics and warning-only exception behavior remain, and no SID metric becomes a gate.

## Bounded verification plan (for the orchestrator after S06 adjudication; not performed by this candidate)

1. Run the new no-argument CPU mapping script once with the project’s approved Python. Confirm exact three-element inputs, all intermediates and the canonical curvature vector above, and absence of `residual_norm`/old-map outputs in the written JSON.
2. Use a throwaway in-memory/temp payload (never overwrite the canonical input) to show a missing `raw_residual_medians` raises even when `residual_norm` is present. Also exercise wrong-length, NaN/Inf, negative branching, and nonpositive residual inputs through the validator; do not add a permanent test for these deterministic checks.
3. Run the official no-argument `preflight_contract.py` and `deliberation_gate.py` only at their authorized phase. Before any Stage2 launch, recheck the exact S01 Stage0/Stage1/warm-start hashes and actual resolved input use, and run the root-required check that `RQVAE_OUT_DIR` points below `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`.
4. Exercise the checkpoint reporting path with an existing/throwaway SID array: it must call only `evaluate_sid_quality(sids)`, report the supported SID-derived descriptive fields without HitRate, and preserve warning-only failure handling. Confirm the no-argument `should_early_stop()` returns exactly `(False, "")`. This must not run a training job or project-local CLI.
5. Run the bounded one-checkpoint/one-batch `mvg_check.py`; require all fixed-buffer, invariance, explicit `loss.backward()`, optimizer-exclusion/update, and matched iter26-counterfactual outputs before MVG can pass. Root CLAUDE’s prescribed gradient-path requirement remains mandatory before actual Stage2 training.
6. Do not run project-wide suites, formatter, lint, Stage2, Stage3, or GPU training at S06. Full Stage2/Stage3 execution remains single-run and requires later Judge-approved S07–S09 gates.

No permanent test is proposed: key-presence/domain/output behavior, the no-gate contract, and reporting cutover can be demonstrated by bounded throwaway/preflight/MVG checks; no genuinely uncertain edge currently justifies retained test code.

## Self-rejection conditions

Reject this plan or its eventual patch if it duplicates the equation outside the shared mapping helper; reads `residual_norm` or normalized layer scales as fallback; fails to compare recomputed values against both JSON and contract; changes the locked input/output vector, equation/constants, or layer order; alters fixed-curvature semantics, loss, optimizer, Sinkhorn rule/parameters, seed, steps, warm-start, Stage1, or Stage3 trainer/settings; misattributes the two historical input sources or upgrades S03 to replay/hash proof; deletes the HitRate CLI before migrating its caller, drops an approved SID-derived descriptive metric or warning-only behavior, retains a HitRate report/gate surface, or changes `should_early_stop()` from unconditional `(False, "")`; routes outputs anywhere outside the short iter29 result trees; or changes the experiment after a failed direct-effect check.

## Autonomous next action

Judge C adjudicates this plan against Agent B’s independent candidate and canonical S00–S05 artifacts, including the amended S05 round-2 decision. The orchestrator writes only the selected/merged canonical `logs/implementation_plan_iter29.md`, then applies that approved patch once. Proceed to bounded preflight/MVG only in their authorized adjudicated stages; any missing key, inconsistent vector, provenance misstatement, protocol drift, or inactive registered mapping blocks Stage2 and is handled under the existing abort/replan rules without user input.
