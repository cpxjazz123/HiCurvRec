ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION
ROUND=1

## Verdict / recommendation

Recommend **MERGEABLE minimal plan; no direct-evidence blocker**. The S06 implementation boundary is sufficiently specified: `_step6_sum_embeddings` is the only scientific/model-behavior delta. Proceed with Iter31 source preparation, path/identity wiring, an independently authorized Iter31-local HRA static checker, replacement of the inappropriate Iter29 curvature counterfactual in Iter31 MVG, and constants-only direct-entry Stage3 routing. The latter is a design decision supported by the root rules and source facts below, not by historical approval of the Iter29 wrapper.

This report records plans, not executed changes. The design preserves the exact registered HRA-STEP6-1 expression and FCCR-1 ten-field contract; it does not assert HRA paper equivalence, a telescope, or model activation. All runtime activation, clipping, gradient, protocol, and route checks remain future S07/S08/S09/S11 gates.

## Complete file inventory

Categories: **scientific** = sole conceptual model-behavior delta; **wiring** = non-scientific run identity/paths or unchanged source staged into Iter31; **audit** = static/runtime verification tooling. Source tree path prefix for copied files is `stage2_RQ-VAE/curvature_RQ-VAE_iter31/`. Files marked *copied unchanged* should not be edited. “Current” means the Iter29 parent source unless otherwise stated.

| Source/current | Iter31 target | Category | Exact action / symbols |
|---|---|---|---|
| Iter29 `modules/rqvae.py` | Iter31 `modules/rqvae.py` | **Scientific — sole delta** | Copy source. Change only `RqVae._step6_sum_embeddings`: implement the exact source-curvature exp → source-curvature log → common-curvature exp for each layer; perform right-nested common-`c0` Möbius aggregation; final common-`c0` log. Keep input/output validation and decoder-facing `(batch, embed_dim)` result. No edits to residual cascade, forward, decoder, losses, or imports (the three geometry helpers are already imported). |
| Iter29 `curvature_config.py` | Iter31 `curvature_config.py` | Wiring | Copy and update `_CONFIG_DIR`, `MECHANISM_NAME`, `SAVE_DIR_ROOT`, `CONFIG_PATH`, `BEST_CKPT_PATH`, `RQVAE_OUT_DIR`, `RQVAE_CKPT_PATH`, `RAW_SIDS_NPY`, `SIDS_NPY`, and `ITEM_SIDS_JSON`. Use full `MECHANISM_NAME="iter31_<descriptive>"` (recommend `iter31_hra_step6_common_reference`) and the short Iter31 roots below. Retain upstream inputs, hyperparameters, seed/environment setup, and all other config behavior. |
| Iter29 `curvature_RQ-VAE.py` | Iter31 `curvature_RQ-VAE.py` | Wiring | Copy and update the contract reference to `logs/mechanism_contract_iter31.json` and current-run-only labels/messages/docstrings containing `iter29` (e.g., FCCR input/contract missing messages, snapshot/invariant labels, Step3.5 and warm-start diagnostic labels). Keep `_load_closed_form_curvatures` field/value validation, exact ten-field schema, explicit `branching` + `raw_residual_medians`, formula, fixed curvature buffers, Iter8 warm-start location and semantics, all hyperparameters/launch code and algorithm unchanged. Iter8 is a genuine inherited warm-start identity and MUST stay `iter8`, not be renamed. |
| Iter29 `scripts/compute_closed_form_curvature.py` | Iter31 `scripts/compute_closed_form_curvature.py` | Wiring, identity-only | Copy computational source unchanged. If the Iter29-specific top-level docstring is retained, edit that identity wording to describe the copied Iter31 run; do not alter formula, constants, output schema, or input production. |
| Iter29 `scripts/computed_behavior_branching.json` | Iter31 `scripts/computed_behavior_branching.json` | Wiring, input copy unchanged | Copy byte-for-byte as the registered input. Preserve explicit `branching`, `raw_residual_medians`, values, mapping, and historical provenance limits. Do not run the old writer, remeasure, normalize, or add a fallback. |
| Iter29 `scripts/export_sids_for_stage3.py` | Iter31 `scripts/export_sids_for_stage3.py` | Wiring, copied unchanged | Copy unchanged; it imports `SIDS_NPY`, `RAW_SIDS_NPY`, and `ITEM_SIDS_JSON` from the Iter31 config, so path changes propagate without a new exporter or algorithm change. |
| Iter29 `scripts/mvg_check.py` | Iter31 `scripts/mvg_check.py` | Audit | Copy and update source/module identity and check flow. Retain one registered Iter8 warm-start checkpoint and one actual batch. Remove use of `CONTROL_FIXED_C`/alternate-curvature model as activation evidence. Compare HRA Step6 output to legacy Euclidean sum on the same model state, checkpoint, selected batch, same fixed curvature, and same quantized embeddings. Retain fixed-curvature immutability/time invariance, optimizer exclusion, finite/shape/domain and model-gradient checks where compatible. No curvature gradient is expected. |
| Iter29 `scripts/grad_check.py` | Iter31 `scripts/grad_check.py` | Audit | Copy and update docstrings/identity/module imports to Iter31; retain its role as no-argument gradient-only delegation to Iter31 MVG helpers. It must not instantiate alternate curvature as a substitute for Step6 direct-effect evidence. |
| None | Iter31 `scripts/preflight_hra_step6_iter31.py` | **Audit — create** | No-argument, no-env-override, source/contract-only checker. Implement checks enumerated below. This file is expressly approved by this S06 plan; it is not approved to execute before S07. |
| Iter29 module tree | Iter31 `modules/` | Wiring, all copied unchanged | Copy exactly: `sid_quality.py`, `quantize.py`, `__init__.py`, `encoder.py`, `hab.py`, `hyperbolic.py`, `loss.py`, `normalize.py`, `step_checks.py`, `utils.py`, `tokenizer/semids.py`. Preserve geometry helper code including projection, log clamp, denominator guard, and no Möbius output projection. |
| Iter29 data tree | Iter31 `data/` | Wiring, all copied unchanged | Copy exactly: `__init__.py`, `amazon.py`, `instruments.py`, `ml1m.py`, `ml32m.py`, `preprocessing.py`, `processed.py`, `schemas.py`, `utils.py`. |
| Iter29 `init/kmeans.py` | Iter31 `init/kmeans.py` | Wiring, copied unchanged | Copy unchanged. |
| Current `stage3_T5Train/train_HG-Rec.py` | Same root Stage3 trainer | Wiring (non-scientific) | Change only hard-coded route constants: `CODE_PATH` to the produced Iter31 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`; `RQVAE_VARIANT` to the descriptive Iter31 identity; `LOG_PATH` and `SAVE_PATH` to the short mandated Iter31 result subtree; `_LAUNCHER["log"]` to that subtree’s `_stage3_launcher.log`. Preserve `_LAUNCHER["script"] = os.path.abspath(__file__)` and all data, model, optimizer, trainer, evaluator, metrics, seeds, flags, and launcher behavior. The exact direct route is below. |

### Existing Iter31 canonical files — unchanged

`logs/mechanism_contract_iter31.json` remains the exact FCCR-1 ten-field contract and fixed vector. `logs/hra_step6_contract_iter31.json` remains the separate HRA contract. Do not edit, merge, or duplicate either contract. Existing S00–S05 canonical records remain unchanged. No new gin/config is necessary for this Iter29 source: the enumerated parent source tree has no `configs/` directory (the parent config has `CONFIG_PATH`/`SAVE_DIR_ROOT` path constants, but no enumerated source gin file). Do not invent a generated config or place training products in the source subtree.

### Copy omissions / untouched files

Copy the full Iter29 Python source inventory shown here, with only the edits explicitly listed above: `curvature_RQ-VAE.py`, `curvature_config.py`; `scripts/{compute_closed_form_curvature.py,computed_behavior_branching.json,export_sids_for_stage3.py,mvg_check.py,grad_check.py}`; `modules/{sid_quality.py,quantize.py,rqvae.py,__init__.py,encoder.py,hab.py,hyperbolic.py,loss.py,normalize.py,step_checks.py,utils.py,tokenizer/semids.py}`; `data/{__init__.py,amazon.py,instruments.py,ml1m.py,ml32m.py,preprocessing.py,processed.py,schemas.py,utils.py}`; and `init/kmeans.py`.

Do **not** copy Iter29 logs or deliberation records, Stage2/Stage3 result trees, checkpoints/`.pth`, SID `.npy` exports, `item_sids.json`, old generated/run gin files, or any old runtime logs. Do not copy `scripts/run_stage3_iter29.py`; it is historical route evidence, not an Iter31 wrapper or approval. Do not create an Iter31 wrapper. The Iter31 source subtree may contain code/config/scripts/logs and the explicitly allowed frozen JSON input, but no `.pth`, `.npy`, or `item_sids.json` products. Keep all generated Stage2 and Stage3 artifacts under their corresponding results roots. No unrelated refactor, cleanup, docs change, metric change, or source normalization is in scope.

## Step6 algorithm and tensor shapes

Observed parent interface: `quantized.embeddings` is `(n_layers, embed_dim, batch)`, assembled from per-layer `Quantize.forward` embeddings. Geometry helpers operate over the **last dimension**, so each layer must first become batch-major `(batch, embed_dim)`; directly passing `(embed_dim, batch)` to them would compute norms along batch and violate the registered point semantics. `self.layers[l].get_c()` is the fixed scalar curvature; pass it in broadcast shape `(1,1)` for batch-major vectors. Preserve layer indices 0, 1, 2 and fixed values; set `c0` to the *same live layer-0 curvature tensor/value*, not an independent parameter.

Pseudocode (with shape/finiteness assertions before returning):

```python
E = quantized.embeddings
require E.shape == (self.n_layers, self.embed_dim, batch_size)
require self.n_layers == 3  # registered HRA-STEP6-1 is exactly L0/L1/L2

c = [layer.get_c().reshape(1, 1) for layer in self.layers]
c0 = c[0]
common = []
for l in (0, 1, 2):
    e_l = E[l].transpose(0, 1)          # (batch, embed_dim), tangent coordinates
    q_l = _expmap0_t(e_l, c[l])         # B_{c_l}
    source_tangent = _logmap0_t(q_l, c[l])
    q_l_common = _expmap0_t(source_tangent, c0)  # B_{c0}
    common.append(q_l_common)

inner = _mobius_add_t(common[1], common[2], c0)
h = _mobius_add_t(common[0], inner, c0)          # exact right nesting
z = _logmap0_t(h, c0)                              # (batch, embed_dim)
require z.shape == (batch_size, self.embed_dim)
require torch.isfinite(z).all()
return z
```

This explicitly implements `q_l=exp0^{c_l}(e_l)`, then `q_l^0=exp0^{c0}(log0^{c_l}(q_l))`; do not simplify the registered source-log/target-exp path on the basis of the unclipped algebraic identity. Helpers project exponential outputs, clamp logarithm arguments, and protect Möbius denominators; these are inherited helper semantics, not implementation changes. Do not log tangent `e_l` directly, reorder layers, reassociate Möbius additions, apply a projection absent from the contract, add a final projection, add d-HSTE/losses, or change the decoder path. Keep current shape/finiteness validation and add only the exact layer/shape guard needed to prevent silently implementing a nonregistered layer count.

## FCCR and path wiring details

- `_load_closed_form_curvatures` must continue explicitly loading both `branching` and `raw_residual_medians`, recompute the registered mapping, compare the registered input vectors and exact ten-field FCCR JSON, and return the registered fixed vector. Update only the per-iteration contract path/message labels; **do not** relax its exact key/schema comparisons to accommodate HRA.
- Stage2 products resolve only under `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`. Specifically: `RQVAE_OUT_DIR` under `.../out/rqvae/instruments`; `RQVAE_CKPT_PATH=.../rqvae_best.pth`; `RAW_SIDS_NPY=.../sids_raw.npy`; `SIDS_NPY=.../dataset/Instruments/sids_for_hgrec.npy`; `ITEM_SIDS_JSON=.../item_sids.json`. `SAVE_DIR_ROOT`, `CONFIG_PATH`, and `BEST_CKPT_PATH` (and any actual Stage2 gin/checkpoint output path) must also be Iter31-isolated beneath this results root. Keep `MECHANISM_NAME` descriptive and distinct from the short product-directory name.
- The Iter31 source tree is not an output root. Retain the Iter8 warm-start path exactly as S01-locked; changing that parent/checkpoint would change the protocol. Preserve external immutable Stage0/Stage1 inputs and all model/optimizer/Sinkhorn/training constants. No CLI parameters or environment overrides are introduced.
- Before any eventual Stage2 launch, directly inspect/verify the configured `RQVAE_OUT_DIR` resolves under the exact Iter31 result root and the iteration number matches. This is a future S09 condition, not performed here.

## HRA-specific static checker (approve in this plan; execute only at S07)

Create `scripts/preflight_hra_step6_iter31.py`, a no-argument/no-environment-override static checker. Resolve the source root from `__file__`; load both canonical contracts and the model source; fail closed with a nonzero exception/exit on any mismatch. Check:

1. HRA contract version/status/Iter31 transition and the separately linked `logs/mechanism_contract_iter31.json`; FCCR contract retains exactly its required ten fields/values, fixed flags, formula inputs, and expected vector. Do not add HRA fields to FCCR or modify/bypass the repository shared FCCR preflight.
2. Exact registered HRA equation ID and specified operations in L0/L1/L2 order: tangent `e_l`; source `exp(c_l)`, source `log(c_l)`, target `exp(c0)`; inner `L1 ⊕ L2`; outer `L0 ⊕ inner`; final `log(c0)`. Verify curvature argument roles and all operations use the current helper functions. Verify shape conversion to batch-major before geometry calls and expected `(batch, embed_dim)` return / existing decoder consumer.
3. Step6 replacement scope and call-site boundary: sole replacement of the prior Euclidean aggregate in `RqVae._step6_sum_embeddings`, with `forward` still calling it as decoder input; no helper/decoder/loss/other Step4/Step5/model changes are proposed by the checker’s expected source constraints.
4. Contract’s forbidden additions are absent from the Step6 implementation plan/source scope: no d-HSTE, added auxiliary loss, altered curvature, alternate optimizer/Sinkhorn, reassociation/reordering, direct `log(e_l)`, or paper-equivalence/telescope claim. Static audit must not claim runtime activation.
5. Report required later S08 observations explicitly: output/domain/finiteness and exact shape, exp-projection/log-clamp incidence, fixed-curvature immutability and optimizer exclusion, intended model/codebook gradient health, and same-checkpoint/same-batch HRA-vs-Euclidean direct outputs. No numeric activation threshold is invented by the checker.

Use readable AST/source-structure checks where possible; narrowly bind to the exact function and contract rather than broad substring checks that can pass on comments. The static checker is an additional HRA contract check, not a substitute for unchanged FCCR preflight or S08 runtime evidence. S06 plan approval authorizes adding the checker once, but **not running it**.

## Iter31 MVG replacement and gradient verification

The Iter29 `scripts/mvg_check.py` counterfactual changes the fixed curvature vector and compares different-curvature model assignments/losses. That verifies a curvature effect, not Iter31’s registered Step6 effect, and must not be copied as the HRA activation check. Iter31 S08 design: load the one S01-approved Iter8 warm-start checkpoint and one deterministic batch once; instantiate/load the candidate with the exact registered fixed curvature; obtain the same `RqVaeOutput`/quantized embeddings once; compute `z_hra = model._step6_sum_embeddings(output)` and `z_euclidean = output.embeddings.sum(dim=0).transpose(0,1)` from those identical tensors and state. Record both outputs’ shape, finite status, norms and direct elementwise/L2 difference, and inspect helper-domain projection/log-clamp incidence. Same weights, batch, fixed `c_l`, quantizer values/assignments, mode and state are required; do not make a second model with changed curvatures or vary inputs/constants/seeds. The contract specifies no numeric difference threshold: report the observed result and let S08 adjudicate whether direct effect is demonstrably nondegenerate; do not fabricate a cutoff.

Retain FCCR-specific checks: exact values and finite fixed buffers; no curvature parameters/optimizer membership; `get_c()` invariant across train/eval and declared step probes; zero curvature regularization. For gradient health, execute the actual forward loss and `loss.backward()` on the candidate, require attached finite total loss and finite nonzero gradients through intended trainable model paths/codebooks; inspect the HRA decoder-facing path as appropriate. Root `CLAUDE.md` §6 additionally requires pre-training gradient-path checks for total loss and each relevant mechanism loss. Fixed curvature itself is not expected to receive a gradient. S08 is one checkpoint/one batch, not a sweep. S06 performs none of these executions.

## Stage3 route decision and rationale

**Select constants-only direct trainer routing; no wrapper.** Root `CLAUDE.md` §5 requires launching `stage3_T5Train/train_HG-Rec.py` directly with no arguments in the prescribed environment; §11 mandates the Iter31 short Stage3 results root. The current trainer has hard-coded generic `CODE_PATH`, generic/derived `RQVAE_VARIANT`, variant-derived `LOG_PATH`/`SAVE_PATH`, and a `_LAUNCHER` entry whose script defaults to `os.path.abspath(__file__)` but whose log defaults to a generic path. A wrapper can monkeypatch these values and then invoke the trainer, but that would make the actual entry different from the root-required direct trainer command. Iter29’s wrapper proves historical mechanics only; it is not permission to violate the direct-entry instruction.

The route is a constants-only edit in `stage3_T5Train/train_HG-Rec.py`: set the hard-coded `CODE_PATH` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`; set `RQVAE_VARIANT="iter31_hra_step6_common_reference"` (full descriptive identifier for labels/metadata); set `LOG_PATH` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/`; `SAVE_PATH` to corresponding `/ckpt/`; and `_LAUNCHER["log"]` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log` (or the precise root-level launcher filename mandated in the approved Stage3 plan; keep it under this short root). Leave `_LAUNCHER["script"]` as direct `__file__`, so the existing no-argument trainer entry forks itself. Preserve model, SID consumption, Stage0 split, optimizer, seeds, epochs/patience, beam, `n_eval`, evaluator and final test behavior exactly. Keep result paths short and do not embed the descriptive mechanism suffix in `LOG_PATH`/`SAVE_PATH`.

This resolves design-level route conflict without changing Stage3 model/evaluation. It is not historical wrapper approval. S11 must independently verify exact source identity, direct invocation, SID resolution, the unchanged effective protocol and that all products—including launcher log, training metrics, `test_final.json`, and checkpoint—land in the mandated short Iter31 results root. If inspection at S11 finds a trainer output/launcher behavior that cannot satisfy those rules with path/identity constants only, stop at that evidence-backed blocker; do not silently fall back to the Iter29 wrapper or change model/evaluator.

## Patch order (not executed)

1. Prepare the Iter31 source tree by copying the exact enumerated Iter29 source/input files and no run products; retain canonical Iter31 logs/contracts rather than copying parent logs.
2. Update `curvature_config.py` to Iter31 result roots and descriptive mechanism name.
3. Copy/update `curvature_RQ-VAE.py` contract filename and run-identity labels while preserving FCCR semantics and parent/warm-start identity.
4. Implement only `RqVae._step6_sum_embeddings` with the exact algorithm/shape behavior above.
5. Add the no-argument `scripts/preflight_hra_step6_iter31.py` as authorized static audit tooling; do not touch the shared FCCR preflight.
6. Update Iter31 `mvg_check.py` and `grad_check.py` to remove curvature-counterfactual activation logic and support the specified direct-output/gradient protocol.
7. Edit only Stage3 path/variant/launcher constants in the direct trainer for Iter31 route; do not add an Iter31 launcher wrapper.
8. After application, perform S07/S08/S11 work only in their own separately adjudicated stages; S06 approval does not authorize their execution.

## Post-apply inspection / smoke plan (not run)

No test/command/checker was run for this assignment. Once the plan is canonically adjudicated and applied, but before any GPU training:

- **S07 independent static review:** inspect the complete resulting diff against this inventory; search copied source/config/scripts for stale Iter29 *current-run* contract/output/log labels while preserving genuine Iter8 parent/warm-start references and historical JSON provenance. Resolve every route/path in code. Check all Stage2 output constants against the short results root and source tree product prohibition. Confirm no CLI/env override was introduced. Inspect exact Step6 source and prove only that method has scientific/model-behavior modification. Do not treat packet assertions as proof of applied source.
- **S07 separated no-argument checks:** with independent S07 approval only, run the unchanged shared `preflight_contract.py` and new `scripts/preflight_hra_step6_iter31.py` separately, record both results and require both to pass. `MECHANISM_CONTRACT_PASS` alone is not HRA pass. Run the deliberation gate only in its authorized later phase; it does not replace either contract checker.
- **S08 one-run MVG:** use one approved checkpoint and one batch; verify shape, finite outputs/domain, fixed `c_l` and optimizer exclusion, helper projection/log-clamp incidence, intended gradients and direct `z_hra` vs Euclidean output on identical input/model state. No parameter, seed, checkpoint, or batch sweep. Record evidence; do not infer Stage3 benefit from MVG.
- **Before Stage2 (S09):** perform root §6 actual loss/backward gradient-path checks, complete all protocol/data/checkpoint/current path/hash identity rechecks, verify `RQVAE_OUT_DIR` is under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, verify no-argument root Stage2 launch and deliberation gate. Stage2 executes only once after adjudication.
- **After Stage2 (S10/S11):** treat SID metrics as descriptive. Independently audit Stage2 SID identity and Stage3 direct-entry route; confirm unchanged Stage3 model/data/evaluator/effective protocol and exact short output roots before the one Stage3 evaluation. Record R@5/R@10/NDCG@5/NDCG@10 and `n_eval`; compare only with the canonical Iter29 result.

## Remaining gates

- **S07_PREFLIGHT:** two independent audits and Judge decision; source-vs-contract review; separate unchanged FCCR preflight and new Iter31 HRA checker; both pass; no checker edits at S07.
- **S08_MVG:** two independent design/run interpretations and Judge decision; one checkpoint/batch; FCCR immutability plus actual model gradient health; same-checkpoint/same-batch/same-state HRA-vs-Euclidean direct-output evidence; finite/domain/clipping evidence. If it is inactive under the registered spec and only a changed equation/constant could activate it, follow skill abort deliberation, no Stage2.
- **S09_STAGE2_EXECUTION:** independent launch audits and Judge; current input/checkpoint identity and output path checks; both checker results and deliberation gate; root §6 gradient checks; Stage2 runs once. No proxy-based early stop.
- **S10_STAGE2_ANALYSIS:** independent analysis of the single Stage2 output; compliance/direct effect/SID description kept separate; no quality proxy gate.
- **S11_STAGE3_EVALUATION:** independently approve direct route, actual SID input, current trainer/runtime, Stage0 evaluation data, beam/evaluator/`n_eval`, protocol-compatible Iter29 comparator, and mandated short output root; Stage3 executes once with no model/evaluator change.
- **S12_RESULT_CLASSIFICATION:** classify mechanism validity/activity separately from promotion; one result cannot establish variance or robust causality; retain strict `test_R@10 > 0.065` adoption criterion.
- **S13_GIT_CLOSURE:** independent required-artifact/path/status audit and authorized main/GitHub closure under root rules. No closure claim before required artifacts and remote hash checks.
- **S14_GLOBAL_REVIEW:** only if the skill trigger fires (three clean protocol-valid iterations or specified trigger); use a distinct forward structural mechanism, never replication/sweep/root-cause.

## Risks, rollback/abort, self-rejection

- **Shape/orientation risk:** helpers norm over the final axis; omitting transpose changes the geometry. A shape mismatch or wrong layer order is implementation invalid and blocks progression.
- **Helper clipping risk:** exp projection and log clamp may violate the pointwise exp/log simplification; keep the explicit expression and measure incidence. Do not hide saturation by changing tolerances/curvature.
- **Scientific viability risk:** near-origin or strongly projected values could make the direct effect effectively inactive. S08 evidence, not this static plan, decides. If changing the registered equation, mapping, layer order, or constants is needed to activate it, autonomously abort Iter31 rather than retune.
- **Inherited contract/provenance risk:** FCCR values are historical method/value provenance, not replay/hash proof. Preserve the frozen artifact and record this limit; a missing key or changed vector fails closed.
- **Stage3 route risk:** direct trainer constants solve the identified default-path issue only if every artifact writer uses the configured roots. S11 must verify actual resolution. No wrapper fallback or evaluator/model edit is allowed without an explicit authorized between-iteration scope change.
- **Source-tree/results confusion:** all `.pth`, `.npy`, and `item_sids.json` products must be in results roots, never under Iter31 source. Any wrong root blocks launch.
- **Rollback:** before any execution, if S07 finds an accidental extra scientific change or invalid wiring, revert only the unauthorized Iter31 change to the accepted plan/parent source while preserving canonical S00–S06 records and unrelated user work; repeat the relevant authorized audit. Do not revert unrelated files. If the registered operation is mathematically invalid, persistently nonfinite, unusable, or inactive and repair would change locked spec, use canonical `ABORT_ITERATION`, stop downstream execution, and document evidence; do not retune or relabel a scientific result.
- **Self-reject / external blocker:** reject this design if parent primary source contradicts the recorded shape or helper semantics; if approved direct Stage3 constants cannot satisfy root direct-entry/result-root requirements without model/evaluator change; if Stage2 results routing cannot isolate all products; or if the HRA checker cannot verify its linked contract/source without weakening the unchanged FCCR gate. Current reviewed evidence shows none of these as a proven blocker, but Stage3 final effective-path behavior is to be verified at S11.

CONFIDENCE=MEDIUM-HIGH for the Step6/source and output-routing design; MEDIUM for the Stage3 route pending applied-source resolution and independent S11 verification; no confidence claim on activation/performance.
USER_INPUT_REQUIRED=NO
NO_EDIT_AUTHORIZATION=TRUE
NO_COMMAND_TEST_CHECKER_GPU_OR_TRAINING_AUTHORIZATION=TRUE
