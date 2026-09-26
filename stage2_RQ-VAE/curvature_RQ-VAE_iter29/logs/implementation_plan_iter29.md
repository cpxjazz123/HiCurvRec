# Iter29 S06 Implementation Plan — Judge C Canonical (Round 2)

```text
STAGE_ID=S06_IMPLEMENTATION
ROUND=2
VERDICT=ACCEPT_A
PARENT_ITER=iter26
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0dce377882
CANONICAL_BASELINE_ITER=iter26
EXPERIMENT_TYPE=single_factor
NEW_MECHANISM=Replace only iter26's per-layer closed-form map with the canonical iter29 bounded rational/additive map
POLICY_HYGIENE_DIFFERENCES=Replace project-local HitRate CLI reporting with SID-only descriptive metrics; remove HitRate surfaces and make should_early_stop unconditionally return (False, "")
```

## 1. Decision boundary and registered mapping

Implement only the mapping change registered in S02/S04, over the same layer-ordered inputs `[L0,L1,L2]`:

```text
JSON branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

**Physical-key distinction:** the current input JSON stores the behavior-branching vector at `branching`; this is the value whose semantic name in the S03 manifest and S04 contract is `behavior_branching`. The residual vector must be read from the explicit plural JSON key `raw_residual_medians`. The S04 `formula_inputs` values remain exactly `['behavior_branching', 'raw_residual_median']`: these are semantic formula-input names, not replacement JSON property names. Do not rename the existing `branching` property to `behavior_branching` or interpret `raw_residual_median` as permission to read a singular property.

FCCR-1 remains closed-form, computed before training, fixed in non-trainable buffers, and invariant to training step, optimizer updates, and train/eval mode. No learnable curvature, schedule, curvature regularization, new optimizer/loss, new Sinkhorn rule, or second mechanism is authorized. Iter26 remains the sole direct control. S03's provenance approval is **historical method/value provenance only**; it does not establish present checkpoint replay, independent historical recomputation, or historical input-byte identity. Never rerun the old behavior writer: it overwrites this JSON without the required raw key and depends on an unavailable iter8 raw-SID input.

## 2. File-by-file implementation scope

### 2.1 Add `scripts/compute_closed_form_curvature.py`

Add a no-argument, CPU-capable runner and one reusable pure formula helper; this helper is the sole implementation of the registered equation and is imported by the training consumer. The runner locates `scripts/computed_behavior_branching.json` relative to its own file, accepts no CLI arguments, and writes derived mapping metadata only after successful validation/calculation. Do not add `argparse`, `sys.argv`, or environment overrides.

The helper accepts the semantic branching vector and raw residual vector as arguments. The JSON runner passes `payload['branching']` and `payload['raw_residual_medians']` directly. Require both to be ordered, length-three numeric vectors with finite values; reject malformed types (including booleans), `B_l < 0`, and `m_l_raw <= 0`. Do not coerce missing values, sort, broadcast, clip, normalize, default, reconstruct, or fall back to `residual_norm`, normalized layer scales, or another alias. Compute the exact constants `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; reject non-finite intermediates or values outside the supported curvature interval. Preserve the approved inputs and provenance; print mapping identity, constants, intermediates, and output for later bounded review.

### 2.2 Update `scripts/computed_behavior_branching.json` through that runner

Preserve the exact `entropy`, `branching`, and `raw_residual_medians` values and `[L0,L1,L2]` order. Preserve the behavior producer record and make the independent provenance explicit:

- Behavior branching: retain the recorded iter12 producer and its source paths, definition, counts, and actual inputs (iter8 raw SIDs plus Stage0 `train.parquet`). State that the iter8 checkpoint path appearing in metadata was not loaded by this branching producer.
- Raw residual medians: add a separate provenance record for the iter10 `calibrate_residual_scales.py` method and contemporaneous log, source label `iter1`, recorded step `100000`, historical baseline checkpoint and Stage1 embedding paths, and the measurement definition (per-layer/per-item residual-vector L2 norm followed by median over items). Preserve S03's limitations: the historical checkpoint and referenced iter8 raw-SID export are absent; no checkpoint/SID replay, independent recomputation, or historical byte-identity claim is made.

Remove the semantically unsafe `residual_norm` alias and obsolete old-map fields `raw_capacity`, `branch_mid`, `s_l`, `z_l`, the old `closed_form_c_l` values, `closed_form_c_base`, `closed_form_alpha`, `m_min`, and `closed_form_source`. Retain the entropy/branching and both distinct provenance records. Store the iter29 mapping identity and its exact constants, `x_l`, `y_l`, `u_l`, and `closed_form_c_l` computed from the unchanged approved inputs. Do not manually regenerate the historical inputs or run either historical producer/calibrator.

### 2.3 Update `curvature_RQ-VAE.py`

Replace `_load_closed_form_curvatures()`'s current trust of the saved vector with strict validation using the shared helper. Resolve the input JSON relative to the iter29 source directory and the unchanged contract at `logs/mechanism_contract_iter29.json`; do not depend on the current working directory.

Require and directly index the physical input keys `branching` and `raw_residual_medians`; recompute `x/y/u/c` through the shared helper. Validate all input, stored intermediate/output, and contract vectors for length three, finite numeric values, and correct layer order/domain. Require the unchanged S04 FCCR-1 contract fields/flags and exact semantic `formula_inputs`; compare recomputed `x/y/u` to the stored arrays and recomputed `c` to both JSON `closed_form_c_l` and contract `final_curvature_values`. Use an explicitly documented tight formula/data/contract comparison (absolute tolerance `1e-9`, zero relative tolerance); retain the existing `1e-6` fixed-buffer invariance tolerance for model buffers. Any missing, malformed, out-of-range, or discrepant source, contract, or vector fails closed before model construction/training. Return only the recomputed, validated curvature vector; do not duplicate the formula or use a stale stored vector as the model input. Do not modify the S04 JSON/schema or its exact values.

Remove only obsolete `CLOSED_FORM_C_BASE`/`CLOSED_FORM_ALPHA`, the confirmed-unreferenced `_load_layer_norms()` fallback, and stale current-map comments/diagnostic labels that describe iter16/iter26 as the iter29 mapping. Keep the existing fixed-buffer construction, fixed-curvature checks and steps, model/loss/optimizer/Sinkhorn code, warm-start state filtering, all data loading and training behavior. In particular, keep `residual_layer_norms=[1.0] * N_LAYERS` unchanged; it is not the raw-residual formula input. Retain the inherited Sinkhorn rule `effective_eps = sk_eps * (c / c_cyclic_max)` and its parameters; any changed realized epsilon is only a mediated consequence of the new fixed `c`.

At the existing checkpoint SID-report boundary, import/call `evaluate_sid_quality(sids)` instead of `evaluate_sid_quality_full(sids, EMB_NPY)`. Preserve checkpoint cadence, SID inference and raw-SID writes, printed SID-derived metrics, warning-only metric-collection exception handling, training/stopping behavior, and separate final 4-token Stage3 export and its failure behavior.

### 2.4 Update `curvature_config.py`

Change only iteration identity and result routing:

- `MECHANISM_NAME = 'iter29_bounded_rational_additive_mapping'` (full descriptive identity).
- `_CONFIG_DIR` and every Stage2 output path resolve under `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/` (short directory name, no descriptive suffix).
- Route `RQVAE_OUT_DIR`, `RAW_SIDS_NPY`, `SIDS_NPY`, and `ITEM_SIDS_JSON` to the corresponding iter29 result subtree; keep raw SID output under the iter29 `RQVAE_OUT_DIR`.

Preserve Stage0/Stage1 inputs and paths, source iteration directory, seed, GPU/runtime, model, worker, and other configuration values. Do not place checkpoints, `.npy`, or SID JSON artifacts under the code iteration subtree.

### 2.5 Add `scripts/run_stage3_iter29.py`

Follow the iter26 wrapper's launch pattern. Change only iteration-local wiring: input `CODE_PATH` to iter29 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`; full `RQVAE_VARIANT='iter29_bounded_rational_additive_mapping'`; and `LOG_PATH`/`SAVE_PATH` to `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/` and `/ckpt/`. Preserve launcher bookkeeping/dispatch shape. Do not modify the Stage3 trainer, source, settings, seed, epochs, beam, `n_eval`, or evaluation protocol.

### 2.6 Update `scripts/mvg_check.py`

Retarget the existing no-argument MVG to the iter29 training module and expected fixed vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`. Use the locked iter8 warm-start checkpoint and one real 640-example transition batch with the existing Stage2 settings; do not perform full training. Verify the FCCR-1 requirements:

1. The recomputed fixed vector is exact within the declared tolerance, finite, and supported. Each layer has the non-trainable `_fixed_c` buffer; no curvature `nn.Parameter`/`c_layer_scale` is present; fixed curvature is excluded from optimizer parameter identities. A fixed curvature buffer is not required to receive a gradient.
2. `get_c()` and buffers are unchanged across training steps `0, 25000, 50000, 100000`, train/eval mode, and lightweight optimizer updates. Verify ordinary intended model parameters update, while fixed curvature does not.
3. On the same checkpoint/batch, total loss is finite, has `requires_grad=True` and a `grad_fn`; explicitly call `loss.backward()` and verify finite nonzero gradients on intended trainable model parameters. This is model gradient health, not curvature learning. Preserve the zero-curvature-regularization contract; do not add a new loss or metric gate.
4. Build matched candidate/control states from the same checkpoint and same batch, with identical non-curvature weights/settings and only `c` changed from the candidate vector to iter26's `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`. Require and report a measurable finite difference (beyond tolerance) in a real curvature-consuming signal such as quantizer assignment, geometry/distance, or curvature-dependent loss/output; a difference in `c` itself is insufficient. No retuning if activation fails.

This MVG establishes implementation/activation only; it does not claim a Stage3 benefit.

### 2.7 Update `modules/sid_quality.py` — `POLICY_HYGIENE_DIFFERENCES` only

Keep `build_sids_for_corpus`, the current in-process `evaluate_sid_quality(sids)` calculations, and their SID-derived descriptive outputs: full and per-layer Gini, unique full-SID count, item count, L0/L1 unique pairs, and `H(L1|L0)`. Remove `evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, its temporary SID file/command/subprocess/parser path, and subprocess-only imports. Remove `hitrate_k50` from `SidMetrics`, the NaN placeholder, metric validation, formatting, `to_dict()`/report serialization, and every remaining HitRate reference. Correct stale docs/comments about HitRate and gates; retain unrelated report/baseline helpers only where still meaningful, with no removed field or gate claim.

Preserve the existing `should_early_stop(metrics: SidMetrics)` interface, but make its body exactly an unconditional `return (False, "")` with no argument, type, or metric validation and no gate branch. Do not add a caller. This is policy hygiene only: no SID-generation or training change, no metric-based candidate selection, and no Stage2 outer/inner gate. Preserve the existing warning-only metric exception boundary and reporting cadence.

### 2.8 Delete `scripts/sid_metrics_any.py` after caller migration

Only after the iter29 training import/call and `sid_quality.py` no longer reference the CLI, delete this iter29-local positional-argument script. Do not move/create a `/tmp` replacement or edit another iteration's copy.

### 2.9 Conditional, bounded local stale-file cleanup

Only if present and confirmed unused after the replacement wrapper is added, remove these iter29 `scripts/` copies: `run_stage3_iter4.py`, `run_stage3_iter7.py`, `run_stage3_iter8.py`, `run_stage3_iter11.py`, `run_stage3_iter18.py`, `run_stage3_iter27.py`, `calibrate_residual_scales.py`, and `grad_check.sh`. The calibrator is not an authorized iter29 input generator. Retain compatible no-argument `grad_check.py` and `export_sids_for_stage3.py`. Do not remove any other helpers or make cross-iteration edits.

## 3. Locked conditions and one-factor accounting

The only scientific factor is the per-layer mapping. Keep the same values, provenance-approved input identities, order, parent, and direct control. Keep iter26's Stage2 architecture/settings (768 input, hidden `[512,256,128]`, embedding 32, three layers, codebook size 256), seed 42, 100,000 global steps, iter8 warm-start with legacy curvature state excluded, AdamW `lr=1e-3`/`weight_decay=1e-4`, behavior loss weight `0.20`/temperature `0.07`, and existing Sinkhorn rule/parameters (`sk_eps=0.05`, `iters=3`). Stage1/Stage0 paths and identities/hashes remain those in S01. Keep Stage3 trainer/source/protocol/settings unchanged (seed 42, 150 epochs, beam 20, `n_eval=57439`); only the iter29 wrapper label and iteration-local SID/results paths differ.

```text
OPTIMIZER_DIFFERENCES=None
LOSS_DIFFERENCES=None
STAGE1_DIFFERENCES=None
STAGE3_DIFFERENCES=None for trainer/source/settings/protocol; only wrapper identity and iter29 paths differ
POLICY_HYGIENE_DIFFERENCES=SID-only descriptive reporting migration; delete HitRate CLI/reporting; unconditional should_early_stop
```

Stage2 SID metrics remain descriptive-only; they must never be an outer/inner hard gate or early-stop signal. No iter28 or rejected-draft mechanism is inherited. A lower realized Sinkhorn epsilon resulting solely from the new `c` through the unchanged parent formula is a mediated mapping effect, not a second factor.

## 4. Bounded verification and stage boundaries (not performed in this adjudication)

After the canonical source patch is applied, use only bounded behavior checks: invoke the no-argument CPU formula writer and compare the persisted values/intermediates and separated provenance to the vectors above; on a throwaway/in-memory payload with `residual_norm` but no `raw_residual_medians`, require fail-closed behavior without mutating the canonical JSON; exercise the SID-only evaluator/format/report path and verify the removed HitRate field/output is absent and `should_early_stop` returns `(False, "")` even for an invalid sentinel. These are not performance gates and do not authorize training.

S07 remains a separate adjudicated static/contract stage: do not run `preflight_contract.py` or the deliberation gate until S07's own A/B/Judge authorization and required artifacts exist. S08 remains a separate adjudicated MVG stage: run the one-checkpoint/one-batch MVG only after S07 passes and S08 is independently adjudicated. Before any later Stage2 launch, recheck the S01 Stage0/Stage1/warm-start hashes and actual input routing, verify `RQVAE_OUT_DIR` under the exact iter29 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/` subtree, complete the root-required gradient-path check, and pass all later Stage2 execution gates. Stage2/Stage3 launch, build, project-wide tests, formatter, and unrelated cleanup are outside S06.

**Current Judge task status:** no implementation source is changed; no check/test/build/formatter, S07/S08 gate, GPU work, or training is run here. The canonical S06 plan is an implementation instruction only; it does not imply preflight/MVG PASS or authorize Stage2/Stage3 execution.
