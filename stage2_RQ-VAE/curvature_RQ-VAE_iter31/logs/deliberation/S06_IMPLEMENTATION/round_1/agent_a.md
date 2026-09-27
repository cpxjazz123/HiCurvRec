ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION
ROUND=1

## Verdict / recommendation

**RECOMMENDATION=ACCEPT_CONDITIONALLY** — implement exactly HRA-STEP6-1 as adjudicated in canonical S02/S04: replace only `modules/rqvae.py::RqVae._step6_sum_embeddings` model behavior. Approve the separately named `scripts/preflight_hra_step6_iter31.py` as iteration-local audit tooling in the S06 canonical plan. Resolve the Stage3 route at design level by changing only Iter31 SID/variant/results/launcher constants in the directly invoked `stage3_T5Train/train_HG-Rec.py`; do not use a wrapper as the entry point. These recommendations authorize no source changes or execution by this candidate; implementation may happen once only after Judge C selects the canonical plan.

The implementation is conditionally acceptable because its static equation is specified and the existing helper API supports it, but actual helper domain/clipping, direct output effect, gradients, current input identity, and runtime paths remain unverified and are deferred to S07–S11. Do not claim paper-equivalent HRA telescoping, runtime activation, or expected performance.

## Independence, source, and authorization

This is my independent plan, produced from the same frozen S06 packet and direct primary parent source review. I did not inspect or use the other S06 candidate. Canonical S00–S05 decisions/contracts were treated as active; Iter29 files and the current Stage3 trainer were inspected directly. I made no source edits, source copies, commands, tests, checker runs, builds, GPU use, or training launches. No user input is required.

`USER_INPUT_REQUIRED=NO`
`NO_EDIT_AUTHORIZATION=TRUE`
`NO_RUN_AUTHORIZATION=TRUE`
`NO_TRAINING_AUTHORIZATION=TRUE`

## Packet, canonical contract, and parent boundary

Frozen packet: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md`.
Canonical upstream evidence reviewed: S00–S05 Judge reports, `logs/hypothesis_iter31.md`, `logs/mechanism_contract_iter31.json`, `logs/hra_step6_contract_iter31.json`, and `logs/one_factor_diff_iter31.md`.
Parent: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/` (S01/S05 parent and single historical comparator), not an assertion that Iter29 paths/products may be reused as Iter31 outputs.

Preserve exact FCCR-1 fields, inputs `branching` and `raw_residual_medians`, values `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`, fixed buffers, and S03 provenance limits. Do not edit the skill-level shared FCCR preflight or merge HRA fields into the FCCR JSON. There is no fallback, fresh measurement, normalized-scale substitution, or historical replay/hash claim.

## Complete copy/create/modify inventory

Category labels: **A = sole scientific/model behavior delta**; **B = non-scientific Iter31 wiring**; **C = audit/verification tooling**. Copy means selective copy from Iter29 into Iter31 source tree, retaining parent source except the explicitly marked locus. The canonical Iter31 logs/contracts already exist and are not overwritten. This inventory is my intended complete code/source inventory, including unchanged copied files and explicit exclusions.

| Source/current | Iter31 target | Action and category | Exact symbols / intended content |
|---|---|---|---|
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/modules/rqvae.py` | `stage2_RQ-VAE/curvature_RQ-VAE_iter31/modules/rqvae.py` | Copy, then modify (**A**) | Only `RqVae._step6_sum_embeddings`. Keep imports `_expmap0_t`, `_logmap0_t`, `_mobius_add_t`; no forward, decoder, loss, quantizer-loop, Step4/5, or helper edits. |
| `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_config.py` | `stage2_RQ-VAE/curvature_RQ-VAE_iter31/curvature_config.py` | Copy/modify (**B**) | `_CONFIG_DIR`, `MECHANISM_NAME`, `SAVE_DIR_ROOT`, `CONFIG_PATH`, `BEST_CKPT_PATH`, `RQVAE_OUT_DIR`, `RQVAE_CKPT_PATH`, `RAW_SIDS_NPY`, `SIDS_NPY`, and `ITEM_SIDS_JSON` resolve to Iter31. Keep Stage0/Stage1 inputs and all non-identity settings exactly inherited. Use full descriptive mechanism name but short `curvature_RQ-VAE_iter31` result directory. |
| Iter29 `curvature_RQ-VAE.py` | Iter31 `curvature_RQ-VAE.py` | Copy/modify (**B**) | In `_load_closed_form_curvatures`, use `logs/mechanism_contract_iter31.json` and Iter31 wording; update Iter29 labels in docstrings/diagnostics/snapshots to Iter31 identity. Keep arithmetic, ten-field checks, expected inputs/constants, training/loss/optimizer/seed/warm-start semantics unchanged. Check all literal `iter29` strings, including `_log_curvature_snapshot`, `_check_fixed_curvature_invariant`, warm-start or mechanism diagnostics; only correct run identity, not historical provenance facts. |
| Iter29 `scripts/compute_closed_form_curvature.py` | Iter31 same relative path | Copy/modify (**B**, identity-only) | Preserve `compute_closed_form_curvature`, constants, and computation exactly. Update docstring/printed or persisted iteration identity only if it would label Iter31 output as Iter29; do not rerun the historical writer or change registered input values. |
| Iter29 `scripts/computed_behavior_branching.json` | Iter31 same relative path | Copy unchanged (**B**, frozen input) | Copy registered input bytes/values unchanged; preserve historical provenance limitation. Do not regenerate. |
| Iter29 `scripts/export_sids_for_stage3.py` | Iter31 same relative path | Copy unchanged (**B**, paths imported) | Preserve export algorithm and constants; it consumes updated Iter31 `RAW_SIDS_NPY`, `SIDS_NPY`, `ITEM_SIDS_JSON` from `curvature_config.py`. |
| Iter29 `scripts/mvg_check.py` | Iter31 same relative path | Copy/modify (**C**) | Replace alternate-curvature counterfactual with fixed-curvature same-checkpoint/same-batch HRA-output versus Euclidean-sum direct-output observation; use Iter31 training module/checkpoint identities. Preserve and run the model gradient/curvature contract checks with no curvature-gradient expectation. |
| Iter29 `scripts/grad_check.py` | Iter31 same relative path | Copy/modify (**C**, identity only) | Keep no-argument delegation to updated local `mvg_check`; correct Iter31 wording/docstring/import identity if needed. The actual required root §6 full-loss `.backward()` and intended-parameter checks remain explicitly covered; no second alternate implementation of the scientific mechanism. |
| Iter29 `modules/__init__.py` | Iter31 `modules/__init__.py` | Copy unchanged | Same module exports. |
| Iter29 `modules/encoder.py` | Iter31 `modules/encoder.py` | Copy unchanged | Encoder unchanged. |
| Iter29 `modules/hab.py` | Iter31 `modules/hab.py` | Copy unchanged | No HRA addition; inherited code only. |
| Iter29 `modules/hyperbolic.py` | Iter31 `modules/hyperbolic.py` | Copy unchanged | Reuse existing `_expmap0_t`, `_logmap0_t`, `_mobius_add_t`; preserve projection/clamp/denominator behavior. |
| Iter29 `modules/loss.py` | Iter31 `modules/loss.py` | Copy unchanged | All loss formulas/reductions unchanged. |
| Iter29 `modules/normalize.py` | Iter31 `modules/normalize.py` | Copy unchanged | Decoder normalization unchanged. |
| Iter29 `modules/quantize.py` | Iter31 `modules/quantize.py` | Copy unchanged | Quantizer/codebook/STE/fixed-curvature behavior unchanged. |
| Iter29 `modules/sid_quality.py` | Iter31 `modules/sid_quality.py` | Copy unchanged | Descriptive SID analysis only. |
| Iter29 `modules/step_checks.py` | Iter31 `modules/step_checks.py` | Copy unchanged | Existing checks unchanged. |
| Iter29 `modules/utils.py` | Iter31 `modules/utils.py` | Copy unchanged | Existing utilities unchanged. |
| Iter29 `modules/tokenizer/semids.py` | Iter31 `modules/tokenizer/semids.py` | Copy unchanged | Existing tokenizer unchanged. |
| Iter29 `data/__init__.py` | Iter31 `data/__init__.py` | Copy unchanged | Existing data module. |
| Iter29 `data/amazon.py` | Iter31 `data/amazon.py` | Copy unchanged | Existing data module. |
| Iter29 `data/instruments.py` | Iter31 `data/instruments.py` | Copy unchanged | Existing dataset implementation. |
| Iter29 `data/ml1m.py` | Iter31 `data/ml1m.py` | Copy unchanged | Existing dataset implementation. |
| Iter29 `data/ml32m.py` | Iter31 `data/ml32m.py` | Copy unchanged | Existing dataset implementation. |
| Iter29 `data/preprocessing.py` | Iter31 `data/preprocessing.py` | Copy unchanged | Existing preprocessing. |
| Iter29 `data/processed.py` | Iter31 `data/processed.py` | Copy unchanged | Existing processed-data interface. |
| Iter29 `data/schemas.py` | Iter31 `data/schemas.py` | Copy unchanged | Existing schemas, including `SeqBatch`. |
| Iter29 `data/utils.py` | Iter31 `data/utils.py` | Copy unchanged | Existing data utilities. |
| Iter29 `init/kmeans.py` | Iter31 `init/kmeans.py` | Copy unchanged | Existing initialization. |
| Iter29 `scripts/` absent file set above, all remaining source files | Corresponding Iter31 `scripts/` paths | Copy only if referenced by source and inventory-reviewed; otherwise omit | Do not introduce unlisted utility behavior. The enumerated parent scripts are compute curvature, `computed_behavior_branching.json`, MVG, grad check, Stage3 runner, and SID exporter. The historical `run_stage3_iter29.py` is deliberately omitted under the direct-launch route. |
| No Iter31 source file | `stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py` | Create (**C**, approved by this plan) | Dedicated no-argument static HRA contract/source checker. Do not modify/bypass shared FCCR checker. |
| Current `stage3_T5Train/train_HG-Rec.py` | Same current Stage3 file (not copied into Iter31 source tree) | Modify (**B**) | Hard-code Iter31 `CODE_PATH`, full descriptive `RQVAE_VARIANT`, short Iter31 `LOG_PATH`, `SAVE_PATH`, and `_LAUNCHER.log`; ensure `_LAUNCHER.script` remains direct `__file__`. Do not modify model, dataloader, training, metric, evaluator, protocol, or direct `main`/torchrun flow. |
| Iter29 `scripts/run_stage3_iter29.py` | No Iter31 copy | Omit | Wrapper is historical evidence only, is not approved route, and is unnecessary when the direct trainer itself owns the Iter31 constants. |
| Iter29 `logs/**`, including all run logs and old plans/contracts | No Iter31 source copy | Omit | Never copy old S06+ deliberation, logs, metrics, checkpoints, or historical contracts over canonical Iter31 files. |
| Iter29 `dataset/Instruments/**` / results / `out/**` | No Iter31 source copy | Omit | Do not carry generated products into source tree. |
| Any `.pth`, `.pt`, `.npy`, `item_sids.json`, cache, or `__pycache__` output from parent | No Iter31 source copy | Omit | Stage2 products only under the Iter31 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/` root. Do not copy parent run outputs. |
| Existing Iter31 `logs/mechanism_contract_iter31.json`, `logs/hra_step6_contract_iter31.json` | Same canonical files | Leave unchanged | Preserve exact independent FCCR/HRA contracts and linkage. |

The source tree may contain only implementation code/config/scripts/logs, pycache if naturally produced later, and the allowed frozen behavior input copy. Config artifact paths, including gin, checkpoint, raw SID, exported 4-token SID, and item-SID JSON, resolve only within `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`. `MECHANISM_NAME` remains descriptive (`iter31_<descriptive>`); product roots use short `curvature_RQ-VAE_iter31`. No Stage2 product is placed under the source iteration directory.

### Stage3 direct-entry route decision

**Selected design route:** change only hard-coded Stage3 iteration wiring constants in `stage3_T5Train/train_HG-Rec.py`, then invoke this file directly with no arguments as required by root §5. In particular:

- `CODE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`;
- `RQVAE_VARIANT=<full Iter31 descriptive MECHANISM_NAME>` (avoid relying on the current generic basename map, which maps to generic/unknown values);
- `LOG_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/`;
- `SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/ckpt/`;
- `_LAUNCHER['script']` remains `os.path.abspath(__file__)` (direct `train_HG-Rec.py` entry), and `_LAUNCHER['log']` is a hard-coded path below the same Iter31 Stage3 short root.

This meets the direct-entry constraint while ensuring Stage3's checkpoint, metrics/test outputs (derived from `LOG_PATH`/`SAVE_PATH`), and internal launcher log resolve to the mandated short results root. It avoids using Iter29's wrapper and does not change the model, data, evaluator, seed, epochs, beam, `n_eval`, or launch protocol. The root-prescribed outer shell redirection remains whatever §5 specifies; do not alter it or imply it is an alternative product root. Before S11, verify every Stage3 output consumer derives from these constants and inspect the direct-launch flow. S11 must independently approve the route before any Stage3 execution. If current trainer path consumers cannot be proven to resolve beneath the Iter31 root with constants alone, stop as a design blocker rather than silently introduce wrapper behavior or edit model/evaluation code.

### Required output path summary

- Stage2 root: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`.
- Stage2 expected artifacts: `out/rqvae/instruments/rqvae_best.pth`, `sids_raw.npy`, exported `dataset/Instruments/sids_for_hgrec.npy`, root `item_sids.json`; gin/config and checkpoint products stay below root.
- Stage3 root: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/`.
- Stage3 expected artifacts: `logs/_stage3_launcher.log`, training/evaluation log subtrees including `test_final.json` and `training_metrics.jsonl`, plus `ckpt/HG_Rec_best.pth` (exact nested run layout remains the trainer’s existing layout under these roots).

## Exact Step6 algorithm and tensor shape handling

Observed parent interface: `RqVaeOutput.embeddings` has shape `(n_layers, embed_dim, batch)`; Iter29 produces it by stacking per-layer `(batch, embed_dim)` code embeddings, permuting dimensions, and making contiguous. Existing `_step6_sum_embeddings` returns `(batch, embed_dim)` and `forward()` passes that directly to the unchanged decoder. Existing `rqvae.py` imports all three required geometry helpers.

Pseudocode (not a new helper/API):

```python
# embeddings: (3, D, B), selected tangent-coordinate code embeddings e_l
if embeddings.ndim != 3 or embeddings.shape[0] != 3 or embeddings.shape[1] != self.embed_dim:
    raise RuntimeError("Step6: ... shape ...")
# Keep device/dtype; curvature values come from inherited fixed layer buffers.
c0 = self.layers[0].get_c().view(1, 1)
q0_common, q1_common, q2_common = []
for l in (0, 1, 2):
    e_l = embeddings[l].transpose(0, 1)       # (B, D), tangent coordinate
    c_l = self.layers[l].get_c().view(1, 1)  # source curvature; fixed
    q_l = _expmap0_t(e_l, c_l)                # B_{c_l} ball point
    q_l_common = _expmap0_t(_logmap0_t(q_l, c_l), c0)  # B_{c0}
    ... retain in fixed layer order ...
inner = _mobius_add_t(q1_common, q2_common, c0)  # q1 ⊕ (q2)
h = _mobius_add_t(q0_common, inner, c0)          # q0 ⊕ inner; do not reassociate
z = _logmap0_t(h, c0)                            # (B, D), decoder tangent coordinates
validate z is rank 2, width embed_dim, and finite (retain current checks)
return z
```

Implementation must use explicit `q_l = exp0(c_l,e_l)` then `log0(c_l,q_l)` then `exp0(c0,·)` as specified, even where ideal-domain algebra would collapse it. Keep right-nested operation order; no reordering, associativity assumption, `log0(e_l)`, projection, or extra output transform. Handle exactly the registered three layers, or fail loudly if the expected three-layer shape is violated—do not quietly generalize this registered expression. Preserve dtype/device, batch ordering, and gradients/STE through these operations. Existing helpers project exponential outputs, clamp log arguments, and denominator-protect Möbius addition without output projection; preserve their behavior. Do not assert the pointwise exp/log identity outside its valid unclipped domain. Keep shape and finite output validation consistent with the current decoder-facing contract.

## Iter31 contracts, identities, and path-reference updates

1. Keep `logs/mechanism_contract_iter31.json` byte/schema semantics as the exact ten-field FCCR-1 contract; keep `logs/hra_step6_contract_iter31.json` separate. HRA checker must verify linkage to FCCR path/schema/vector and HRA equation, not add fields to FCCR.
2. Update `_load_closed_form_curvatures` contract filename from `mechanism_contract_iter29.json` to `mechanism_contract_iter31.json`. Keep explicit input key checks, registered vectors, formula recomputation, tolerances and all fixed-curvature invariants. Change any Iter29 run label in error strings/docs/log prefixes to Iter31 when it describes the current run; do not alter mechanism arithmetic or historical source statements.
3. Copy the registered `computed_behavior_branching.json` unchanged. Preserve `branching` and `raw_residual_medians` as named, distinct inputs and the canonical provenance limit.
4. Change path constants to Iter31 result roots. Ensure all gin/config/checkpoint/SID outputs remain under Stage2 result root, and Stage3 `CODE_PATH` consumes precisely the Iter31 root `item_sids.json`.
5. Search all copied source/config/scripts for stale Iter29 path/contract/identity strings. Historical prose may mention Iter29 as parent; only update identity strings that would misidentify current Iter31 execution. Do not alter parent result references in canonical audit files.
6. Before any future training, direct grep/equivalent check of `RQVAE_OUT_DIR` must show Iter31’s root and matching iter number as required by root rules. No CLI args or env overrides are added.

## HRA checker: approve in S06, audit/run in S07

**S06 recommendation: APPROVE creation** of `scripts/preflight_hra_step6_iter31.py` exactly as the canonical HRA contract requires. It is tooling, not an additional model mechanism. The no-argument/no-environment-override script should read the Iter31 HRA and FCCR JSONs and inspect Iter31 source to check:

- both canonical contract files exist and parse; required exact HRA `contract_version`, active transition, Iter31/parent identities, equation ID and FCCR linkage are consistent;
- FCCR contract retains the exact ten-field schema, fixed vector, fixed/non-trainable/non-time-varying flags, and formula input declaration;
- the source Step6 method implements exact source `c_l` exp/log argument roles, target `c0` exp/add/final-log roles, L0/L1/L2 ordering, and right nesting, and does not apply `log0` to tangent `e_l`;
- selected quantizer embeddings are tangent-coordinate values, not IDs or ball points; output is tangent decoder input with the existing batch×embedding shape contract;
- `forward()` still consumes Step6 at the existing decoder call and no other consumer/model method or scope has changed; there is no added d-HSTE, auxiliary loss, curvature/optimizer/Sinkhorn/data/Stage1/Stage3 change detectable within checker scope;
- implementation retains existing helper calls (no local replacements that mask helper clipping/projection semantics);
- it prints a distinct HRA pass/fail marker and records that S08 must measure shape/finiteness/domain, projection/log-clamp incidence, fixed-curvature/optimizer invariance, model gradient health, and same-checkpoint/same-batch direct effect. Static pass must explicitly disclaim runtime activation proof.

Do not pretend a simplistic source-text scan alone proves arbitrary semantic equivalence; checker must be transparent about the invariants it actually checks and fail closed on unsupported source layout. No made-up numerical activation cutoff. S07 independently audits then runs HRA checker separately from the unchanged shared FCCR preflight; both must pass independently. The deliberation gate does not substitute for either.

## Replacement MVG design (S08 only; not run now)

Do not propagate the Iter29 `CONTROL_FIXED_C` counterfactual: it changes curvature, while HRA’s registered direct effect is Step6 aggregation at unchanged FCCR curvature. Plan a single candidate model/checkpoint/batch path:

1. Load the same single locked warm-start checkpoint and the same deterministic one-batch sample once; initialize one candidate model with canonical fixed curvature and load its state once.
2. In eval/no-grad mode, call `get_semantic_ids(batch.x)` exactly once. From the returned `(3,D,B)` embeddings compute (a) registered `model._step6_sum_embeddings(output)` and (b) legacy Euclidean sum `output.embeddings.sum(dim=0).transpose(0,1)`, using identical tensor values/state/batch. Report both tensors, absolute/relative difference and finite status; do not compare different models or alter curvature/assignments. Also report input/output shapes and helper-domain observations (pre-exp norm/radius, exp projection incidence, log clamp incidence, Möbius output domain/finiteness) using the actual helper behavior without changing it. No parameter/seed/value sweep and no arbitrary acceptance threshold; Judge interprets measurable direct effect under preregistered falsifiability criteria.
3. Run the registered model’s total loss on the same batch and execute actual `loss.backward()` (root §6) after checking `requires_grad` and `grad_fn`; report finite nonzero gradients for intended encoder/quantizer/model paths and no assumption of curvature gradients. Use `torch.autograd.grad(... retain_graph=True, allow_unused=True)` on the existing loss components only if needed to satisfy root §6 component-specific checks; do not add a new loss.
4. Verify fixed curvature buffers/get_c are non-grad, absent from optimizer params, match the fixed contract, and remain unchanged across prescribed modes/steps/optimizer updates. The unchanged curvature vector and model state are held fixed; this is not the old alternate-curvature counterfactual.
5. Preserve a record with numerical observations and exact checkpoint/batch provenance; no new permanent test is justified absent a separate uncertain contract edge. Use the S08 artifact/scripts and throwaway smoke only if needed, then remove throwaway files.

S08 remains a separate adjudicated verification stage; this plan does not execute it. If runtime shows invalid geometry/nonfinite outputs, broken gradients, no meaningful intervention according to the locked criterion, destructive saturation, or an in-place repair would alter the registered mechanism, stop/abort according to the skill rather than tune HRA.

## Stage3 route rationale / no model changes

The current `stage3_T5Train/train_HG-Rec.py` has generic `CODE_PATH` and generic/variant-derived results roots, while Iter29’s wrapper overrides SID, variant, short results paths and launcher log. Root §5 requires direct no-argument execution of the trainer; root §11 requires Iter31 outputs under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`. **Choose direct trainer constants** because this is the one route satisfying both requirements without an extra entry point or dependence on a historically unapproved Iter29 wrapper. Hard-code the Iter31 SID path, descriptive variant, Stage3 results short root and launcher log in the trainer. Preserve `SEED`, model construction, data paths, model/evaluator code, train/test settings, torchrun count/port/environment, and direct call to `_launch_via_torchrun` unchanged. The full descriptive `RQVAE_VARIANT` is metadata; output root must be the exact short name, never a descriptive suffix. No CLI/env override or Stage3 model/evaluation change.

At S11, independently inspect all consumers of `LOG_PATH`, `SAVE_PATH`, `CODE_PATH`, `RQVAE_VARIANT`, `_LAUNCHER` and verify final test/metrics/checkpoint/launcher paths resolve under the required result root. If evidence uncovers any product path outside it that cannot be corrected with non-model path wiring, record a hard blocker rather than run a wrapper or silently move outputs.

## Post-apply inspection and smoke plan (not executed)

After Judge C approves and orchestrator applies once:

1. Inspect the actual Iter31 source tree and compare each copied file to Iter29, allowing only the tabled edits. Check that no `.pth`, `.pt`, `.npy`, `item_sids.json`, parent logs, results, or unreviewed file exists in the Iter31 source tree.
2. Review diff at exact symbols: only scientific delta is `_step6_sum_embeddings`; all other modifications are classified wiring/tooling. Check inherited forward/loss/Step4/5 and hyperbolic helpers untouched.
3. Search all Iter31 source/config/scripts for `iter29`, parent result paths, `mechanism_contract_iter29`, and generic output paths; distinguish legitimate parent provenance references from stale current-run identity. Inspect every config/script caller.
4. Verify Iter31 `RQVAE_OUT_DIR`, checkpoint/raw/export SID/JSON paths resolve beneath Stage2 Iter31 results root and no generated product path targets source subtree; verify direct Stage3 constants and derived outputs remain beneath Stage3 Iter31 short root.
5. S07 (only after its canonical approval): independently audit and run the unchanged FCCR preflight and the separate no-argument HRA preflight; record distinct outcomes and also the deliberation gate. Do not use static pass as activation proof.
6. S08 (only after canonical approval): one locked-checkpoint/one-batch runtime MVG as above; same fixed model/state/batch HRA Step6 tensor versus Euclidean baseline output, helper domain/clipping measurements, fixed curvature invariance/optimizer exclusion, total/component gradient checks. No multiple constants or seeds.
7. After S07/S08 and other required gates, future S09 preflight checks current data/checkpoint identities, direct path verification, `RQVAE_OUT_DIR`, deliberation gate, and root §6 actual gradient check; then Stage2 runs once only if valid. Later S10 analyzes descriptives without gates; S11 audits the direct Stage3 route and then Stage3 runs once; S12 classifies mechanism and promotion separately; S13 closes required artifacts/Git; S14 only if triggered.

This candidate performs none of the above validation or smoke operations.

## Concise patch order

1. Selectively copy the enumerated Iter29 implementation files and frozen behavior JSON into the Iter31 source tree; omit products/logs/old wrapper; leave canonical Iter31 contracts intact.
2. Update Iter31 config paths/MECHANISM_NAME, then Stage2 entry contract filename and identity-only labels.
3. Make the single `RqVae._step6_sum_embeddings` equation replacement; no adjacent refactor.
4. Rewrite copied MVG for HRA direct output and fixed-curvature gradient/invariance checks; update grad-check identity/delegation.
5. Add the dedicated HRA static preflight checker, as expressly approved by S06.
6. Update only Stage3 hard-coded SID/variant/results/launcher wiring to support direct root-prescribed trainer launch.
7. S07 and S08 execute only after their own approvals; S06 Judge approval alone never authorizes checker execution, MVG, Stage2, or Stage3.

## Risk, rollback, abort, self-rejection

- **Scientific semantics risk:** radial transfer plus helper clipping/projection is not an unconditional exp/log identity. Preserve explicit calls and measure actual incidence at S08. Do not claim heterogeneous residual telescope or universal homomorphism.
- **Contract risk:** shared FCCR preflight is FCCR-1-specific. A failure caused by that checker or a mismatch in its expected Iter31 contract blocks progression; do not bypass or edit the shared checker as an S06 workaround. If existing shared preflight cannot accept the unchanged ten-field FCCR contract under Iter31 paths, this is a blocker to resolve through canonical adjudication, not a schema expansion.
- **Wiring risk:** stale Iter29 strings, generic Stage2 paths or Stage3 map defaults could misroute products. Static path/reference inspection and later S07/S09/S11 checks are mandatory.
- **Stage3 risk:** direct trainer constants need complete consumer audit; only path/variant/launcher identity changes are allowed. If a compliant route requires model/evaluator change or a wrapper-only route, stop and report blocker.
- **Rollback:** before any execution, revert the entire Iter31 implementation patch to the canonical logs-only state if the approved patch cannot be applied exactly or changes a protected inherited factor; do not retain a partial scientific/wiring patch. Preserve the canonical deliberation evidence. This describes later orchestrator handling only; I did not mutate state.
- **Self-rejection:** reject this recommendation if direct primary source shows required curvature helpers are absent/incompatible, the decoder does not consume Step6 as assumed, exact Iter31 FCCR contract cannot be loaded without changing its ten-field schema or shared checker, Stage3 routing cannot meet direct-entry plus short-root constraints with wiring-only edits, or any second scientific change is needed. At S08, direct evidence of invalid/inactive/infeasible registered behavior invokes abort rules; do not retune.

## Remaining gate checklist

- **S07_PREFLIGHT:** independent A/B static contract/source audit; run unchanged shared FCCR preflight and separate approved HRA checker independently; preserve deliberation-gate pass; no execution unless all pass.
- **S08_MVG:** one same-checkpoint/same-batch candidate output comparison against Euclidean Step6, exact fixed curvature, helper-domain/clipping checks, model gradient health and fixed curvature invariance/optimizer exclusion. No curvature gradient; no threshold invention, sweep, or multiple seeds.
- **S09_STAGE2_EXECUTION:** independent launch/path/input identity audit; ensure canonical S00–S09 deliberation and both preflights, direct `RQVAE_OUT_DIR` path proof, registered Stage0/1/warm-start identity/consumption, project root §6 backward/component gradients, Stage2 output integrity, seed/steps/data/protocol unchanged. Run Stage2 once only after Judge approval; source tree remains product-free.
- **S10_STAGE2_ANALYSIS:** A/B analyze same run’s contract compliance, direct effect, SID/geometry metrics; proxies descriptive, not gates; preserve interpretation limits.
- **S11_STAGE3_EVALUATION:** independently approve direct `train_HG-Rec.py` route, exact Iter31 SID consumption, short Iter31 result root for test/metrics/checkpoint/internal launcher log, unchanged model/data/evaluator/protocol/seed, n_eval and metric identities. Then Stage3 once.
- **S12_RESULT_CLASSIFICATION:** separate implementation/mechanism status from promotion status; compare only protocol-compatible exact Iter29 result; strict `test_R@10 > 0.065`; avoid causal overclaim from one run.
- **S13_GIT_CLOSURE:** independently audit code + required Stage2/Stage3 products and decision/audit markdown; apply root GitHub-only main commit/push/hash protocol only after normal/abort closure criteria. No closure claim before remote hash agreement.
- **S14_GLOBAL_REVIEW:** only if three clean protocol-valid iterations or another defined trigger occurs; choose a distinct forward structural mechanism, never replication/sweep/root-cause.

`CONFIDENCE=HIGH` for the registered equation, parent Step6 consumer, FCCR schema boundary, and direct-entry wiring design based on inspected source; `MEDIUM` for full copied-file completeness until post-apply source inventory, actual checker compatibility, runtime effect/domain, current identities, and S11 route consumer audit are independently verified.