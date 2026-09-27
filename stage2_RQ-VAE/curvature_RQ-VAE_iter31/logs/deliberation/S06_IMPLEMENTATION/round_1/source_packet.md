# Iter31 S06 Implementation Design — frozen source packet

```text
STAGE_ID=S06_IMPLEMENTATION
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Canonical active inputs

Use only Judge-approved S00–S05 artifacts and their Judge reports as active instructions:

- `logs/source_snapshot_iter31.md`
- `logs/protocol_manifest_iter31.md`
- `logs/hypothesis_iter31.md`
- `logs/mechanism_manifest_iter31.md`
- `logs/mechanism_contract_iter31.json`
- `logs/hra_step6_contract_iter31.json`
- `logs/one_factor_diff_iter31.md`
- corresponding S00–S05 Judge reports

S04 activated HRA-STEP6-1 for Iter31 as a between-iteration transition from cancelled Iter30, with FCCR-1 preserved as the exact fixed-curvature substrate. S05 conditionally confirmed the only conceptual change is `_step6_sum_embeddings`, but its file inventory is explicitly non-exhaustive. Read current parent source files directly and enumerate the exact final patch inventory. Do not rely on candidate drafts as active instructions.

## S06 goal and boundary

Agent A and Agent B independently produce a complete minimal implementation plan and wiring audit for the exact S02/S04 operation. Judge C selects/merges one plan; only then may the orchestrator apply it once. Candidates/Judge do not edit code, run tests/checkers, use GPUs, or authorize execution.

The sole scientific/model behavior change is `RqVae._step6_sum_embeddings`: replace the Iter29 Euclidean sum of tangent embeddings with exactly:

```text
e_l   = selected tangent-coordinate quantizer embedding
q_l   = exp0^{c_l}(e_l) ∈ B_{c_l}
q_l^0 = exp0^{c0}(log0^{c_l}(q_l)) ∈ B_{c0}
h     = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)
z     = log0^{c0}(h)
c0    = c_0 = 1.3660953164241916
```

The method receives `RqVaeOutput.embeddings` in existing shape `(n_layers, embed_dim, batch)` and must continue returning batch×embedding-dimension tangent coordinates to the existing decoder. Preserve L0/L1/L2 order and right nesting; no `log0(e_l)`, reassociation, output projection not present in the registered expression, paper-equivalent telescope, d-HSTE, or new loss. Existing helpers in `modules/hyperbolic.py` already provide `_expmap0_t`, `_logmap0_t`, `_mobius_add_t` and their projection/clamp/denominator semantics. Do not simplify away the explicit source-log/target-exp path: clipping/projection makes the pointwise identity conditional.

Everything else is inherited: FCCR-1 exact ten-field JSON/vector and `branching` plus explicit `raw_residual_medians` path; fixed buffers; encoder, quantizer/codebooks/assignments/STE, Step4 residual update, Step5 transport, every loss/weight/reduction, optimizer/Sinkhorn/schedule, data identities/warm-start, seed/steps, decoder/reconstruction path, Stage3 model/data/evaluator/protocol. Stage2 metric summaries remain descriptive only. No performance or runtime evidence exists yet.

## Current Iter31 source workspace and parent source inventory

Current `stage2_RQ-VAE/curvature_RQ-VAE_iter31/` contains canonical `logs/` only; no model/source implementation has been copied or edited. Iter29 source tree contains these Python files (copy/use only as needed, and preserve parent code exactly except authorized loci):

```text
curvature_RQ-VAE.py
curvature_config.py
scripts/compute_closed_form_curvature.py
scripts/mvg_check.py
scripts/grad_check.py
scripts/run_stage3_iter29.py
scripts/export_sids_for_stage3.py
modules/sid_quality.py
modules/quantize.py
modules/rqvae.py
modules/__init__.py
modules/encoder.py
modules/hab.py
modules/hyperbolic.py
modules/loss.py
modules/normalize.py
modules/step_checks.py
modules/utils.py
modules/tokenizer/semids.py
data/__init__.py
data/amazon.py
data/instruments.py
data/ml1m.py
data/ml32m.py
data/preprocessing.py
data/processed.py
data/schemas.py
data/utils.py
init/kmeans.py
```

Also copy the registered input artifact `scripts/computed_behavior_branching.json` unchanged with its historical provenance limits. Do not copy Iter29 `logs/`, result products, checkpoints, NPY files, or `item_sids.json` into the Iter31 source tree. Do not create any Stage2 training product under the source tree. Root rules permit source code/config/scripts/logs/pycache/allowed input copies only there; all Stage2 outputs must be under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`.

## Required implementation-plan coverage

Each candidate must independently list exact files, symbols, intended changes, unchanged content, and a low-cost verification plan. Use primary source review and search all copied source/config/scripts for iteration-specific references; S05's file list is not exhaustive. At minimum evaluate:

1. **Copy/prepare the Iter31 source tree** from Iter29 without old run products/logs. List exact files copied and any files intentionally omitted. Preserve module/data/init source without scientific changes. Copy `scripts/computed_behavior_branching.json` as the frozen input; do not rerun the historical writer.
2. **Scientific source:** only `modules/rqvae.py::RqVae._step6_sum_embeddings` changes model behavior. Use existing imported hyperbolic helpers; obtain layer values in L0/L1/L2; explicitly apply `exp0(c_l,e_l)`, `log0(c_l,q_l)`, `exp0(c0,...)`; compute inner `L1 ⊕ L2`, then outer `L0 ⊕ inner`; finish `log0(c0,h)`. Retain output shape/finiteness validation appropriate to this unchanged decoder interface. Do not alter forward/loss/decoder/other model methods.
3. **Stage2 identity and output wiring:** `curvature_config.py` must hardcode full descriptive `MECHANISM_NAME=iter31_<descriptive>` and paths whose product roots use short `curvature_RQ-VAE_iter31` under repository `results/`. `RQVAE_OUT_DIR`, checkpoint, raw SID NPY, exported `sids_for_hgrec.npy`, and `item_sids.json` outputs must resolve only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`. The Stage2 gin/config/checkpoint locations must also be isolated there. Do not write `.pth`, `.npy`, or `item_sids.json` under the source iter subtree. The source-tree logs may hold required audit/run logs. Before eventual training, direct `functions.grep`-equivalent verification of `RQVAE_OUT_DIR` is mandatory.
4. **Stage2 entry/contract identity:** `curvature_RQ-VAE.py` path references and log labels must be iteration-correct (`mechanism_contract_iter31.json`, Iter31 source/output identities) while preserving `_load_closed_form_curvatures` semantics and the exact ten-field FCCR-1 schema. Do not merge HRA fields into FCCR JSON, modify/bypass the shared FCCR preflight, change warm-start path/hash, or retune existing settings. Search path strings such as `iter29`, including contract filename and curvature/invariant/warm-start diagnostics; update only identity labels that would otherwise misstate this run.
5. **Separate S04 HRA checker:** plan `scripts/preflight_hra_step6_iter31.py` exactly as required by `hra_step6_contract_iter31.json`. It consumes both canonical contracts and source; runs without arguments or env overrides; checks exact right-nested expression, types and curvature argument roles, only-Step6 consumer scope, no forbidden additions, and distinct linkage to FCCR contract. It records required S08 runtime observations but does not claim static code proves activation. S06 must explicitly authorize or reject checker implementation in the approved plan. If approved, the orchestrator adds it once; S07 independently audits and executes it separately from unchanged shared FCCR preflight. Never treat `MECHANISM_CONTRACT_PASS` alone as HRA pass.
6. **Gradient/MVG scripts:** directly inspect Iter29 `scripts/mvg_check.py` and `grad_check.py`. The Iter29 MVG contains an FCCR curvature counterfactual using a different fixed-curvature control; this is not the Iter31 HRA direct effect and must not be propagated as if it verifies HRA. Plan an Iter31 S08 method that uses the same checkpoint, same batch, same fixed curvature and same model state to compare registered HRA Step6 output against the legacy Euclidean sum; check actual values/domain/clipping and gradient health without adding a new scientific threshold or parameter sweep. Keep the root §6 actual `loss.backward()` checks for total loss and intended model paths before Stage2 training. No fixed-curvature gradient is expected.
7. **Stage3 route — resolve without model/evaluator change:** S01 and S05 explicitly leave routing unresolved. Current `stage3_T5Train/train_HG-Rec.py` defaults to a generic SID and output path; Iter29's `scripts/run_stage3_iter29.py` sets the Iter29 SID, variant, short results paths and launcher identity/log. Root `CLAUDE.md` §5 requires direct no-argument launch of `stage3_T5Train/train_HG-Rec.py` using the specified Python 3.10 environment; §11 requires every Iter31 Stage3 product (launcher log, metrics, test result, checkpoint) under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`. Independently assess a compliant route. If editing only path/variant/launcher constants in the direct trainer is selected, show the model, training, data, evaluator and protocol code remain unchanged and all result/launcher paths are short Iter31 paths. If a wrapper is selected, explicitly resolve its compatibility with the root's direct-entry instruction rather than assuming Iter29 precedent is sufficient. In either route, hardcode the exact Stage2 Iter31 `item_sids.json`, preserve Stage3 seed/settings/evaluator, and do not set hyperparameters through environment/CLI. Keep `MECHANISM_NAME` descriptive for variant metadata while `LOG_PATH`/`SAVE_PATH` use short result name. Stage3's detailed runtime path/eval verification remains S11; S06 must provide a concrete planned route that can satisfy root and S01, or report a hard blocker—never silently pick a generic path.
8. **Complete actual diff scope:** list every file that will be newly created or modified, including configs/scripts/JSON/contracts/preflight/Stage3 routing/MVG. Mark each as (a) the one scientific code delta, (b) non-scientific Iter31 wiring, or (c) verification/audit tooling. Mark all copied unchanged sources as unchanged. Include a hard boundary prohibiting unrelated refactors, cleanup, docs or metrics changes. 
9. **Verification plan:** specify static source inspection and precise smoke/MVG steps to run only after implementation and later S07/S08 authorization. No project-wide formatter/linter/test suite. Keep experiment confirmation to one preflight and one same-checkpoint/same-batch MVG, not multiple values or seeds. Any new permanent test must defend an observable uncertain behavior; otherwise use a throwaway smoke script and remove it after proof.

## Parent source references verified for planning

- Iter29 `modules/rqvae.py:246–317` contains inherited Step4/Step5 and the ordered quantizer loop; `:319–326` is only the current Euclidean Step6 sum; `:354–378` consumes Step6 through unchanged decoder/reconstruction/other losses.
- Iter29 `modules/rqvae.py:14–20` already imports `_expmap0_t`, `_logmap0_t`, `_mobius_add_t`; no geometry helper change is needed by the registered equation.
- Iter29 `modules/hyperbolic.py:9–28,66–73` defines exp output projection, log clamp, Möbius denominator protection, and absence of Mobius output projection.
- Iter29 `modules/quantize.py` returns tangent selected codebook embeddings/STE and persists fixed `_fixed_c` buffers.
- Iter29 `curvature_RQ-VAE.py:305–382` requires explicit `branching` and `raw_residual_medians`, recomputes the fixed values, and requires exact ten-field `mechanism_contract_iter29.json`; references to the per-iteration contract path must be updated without changing validator semantics.
- Iter29 `curvature_config.py:9–34` anchors output paths, uses `MECHANISM_NAME`, and defines `RQVAE_OUT_DIR`, `RQVAE_CKPT_PATH`, `RAW_SIDS_NPY`, `SIDS_NPY`, `ITEM_SIDS_JSON`; current Iter29 source path layout is not the required Iter31 results root.
- Current `stage3_T5Train/train_HG-Rec.py:124–159,1175–1181` shows generic SID/variant/results/launcher defaults. Iter29 `scripts/run_stage3_iter29.py:15–35` is only historical wiring evidence.
- Iter29 `scripts/mvg_check.py` checks an alternate fixed-curvature counterfactual; do not preserve that as Iter31 HRA activation evidence. `scripts/grad_check.py` delegates into that MVG.

## Authorization and downstream state

S06 Judge approval authorizes only the chosen implementation plan for one-time application; it does not authorize running the checker, MVG, Stage2, or Stage3. After patch application, S07 independently audits and executes both static preflights. S08 separately performs the one-checkpoint/one-batch geometry/direct-effect/gradient MVG. Root `CLAUDE.md` §6 requires its actual pre-training gradient-path check before any full Stage2 train. S09 then adjudicates launch and runs deliberation gate/current identities/paths; Stage2 runs once; S10/S11/S12/S13 follow the locked workflow. No Stage2 quality proxy is a gate. Stage3 remains one evaluation only, after S11 route approval. No user input or confirmation may be requested.

Each candidate returns an independent complete patch plan with exact files/symbols, change classifications, source references, unified minimal change list, Stage3 route proposal consistent with root rules or an evidence-backed hard blocker, verification plan, risks/self-rejection, confidence, `USER_INPUT_REQUIRED=NO`, and explicit no-edit/no-run authorization. Judge C writes canonical `logs/implementation_plan_iter31.md` and `judge.md`; only that plan may be applied once.
