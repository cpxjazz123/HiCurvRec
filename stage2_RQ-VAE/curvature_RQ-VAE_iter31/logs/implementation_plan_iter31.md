# Iter31 S06 Implementation Plan — Judge C canonical

```text
STAGE_ID=S06_IMPLEMENTATION
ROUND=1
VERDICT=MERGE_AB
PARENT_ITER=iter29
CANONICAL_BASELINE_ITER=iter29
EXPERIMENT_TYPE=single_factor (implementation remains gated by S07–S11)
NEW_MECHANISM=HRA-STEP6-1, common-c0 right-nested Step6 aggregation
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
```

## Decision and evidence basis

Approve the minimal implementation/wiring plan below. Agent A and Agent B independently pass the hard gates and converge on the registered HRA-STEP6-1 behavior, exact FCCR/HRA contract separation, a dedicated Iter31 HRA checker, a replacement HRA-vs-Euclidean MVG, and direct Stage3 trainer constants rather than an Iter31 wrapper. This is a MERGE_AB: retain A's exhaustive classified copy/omit inventory and explicit checker/MVG requirements; retain B's source-order and precise operation/shape account and its direct-entry route; resolve the Stage3 outer-log dispute as set out below. Neither candidate's assertions substitute for subsequent source review or runtime gates.

Primary evidence: Iter31's S04 Judge activates HRA-STEP6-1 while retaining FCCR-1 as the fixed-curvature substrate (`logs/deliberation/S04_CONTRACT/round_1/judge.md`, `CANONICAL_DECISION`, `MERGE_COMPONENTS_A/B`); the canonical HRA JSON separates the contracts and names equation `S02_ITER31_HRA_COMMON_C0_RIGHT_NESTED_V1` (`logs/hra_step6_contract_iter31.json`, `mechanism` and `inherited_curvature_contract`); the FCCR JSON remains exactly ten fields with fixed vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` (`logs/mechanism_contract_iter31.json`, lines 1–16). Iter29's parent module imports the needed helpers (`modules/rqvae.py:14–20`), builds `(n_layers, embed_dim, batch)` from ordered layer embeddings (`:290–317`), currently sums them in `_step6_sum_embeddings` (`:319–326`), and passes that result to the unchanged decoder (`:354–365`). Existing geometry helper behavior is projection in `_expmap0_t`, argument clamp in `_logmap0_t`, and denominator protection/no output projection in `_mobius_add_t` (`modules/hyperbolic.py:9–28,66–73`).

## Scope and exact model operation

The only conceptual/model-behavior delta is `stage2_RQ-VAE/curvature_RQ-VAE_iter31/modules/rqvae.py::RqVae._step6_sum_embeddings`, replacing the Iter29 Euclidean tangent sum `embeddings.sum(dim=0).transpose(0, 1)` (Iter29 `modules/rqvae.py:319–326`). The unchanged callsite remains `forward` → Step6 → decoder (`:354–365`). Do not modify the encoder, quantization, Step4 residual update, Step5 curvature transport, losses/reductions, decoder, helper implementations, or any other model behavior.

For `E = quantized.embeddings` shaped `(L,D,B)`, registered `L=3`, convert each selected tangent embedding to batch-major `(B,D)` before geometry calls because helpers reduce over the last axis:

```text
require E.ndim == 3, E.shape == (3, self.embed_dim, B)
c_l = self.layers[l].get_c().reshape(1, 1), l=0,1,2
c0 = c_0 (the live layer-0 fixed curvature; not another parameter)
for l in [0,1,2]:
    e_l = E[l].transpose(0,1)                  # (B,D), tangent coordinates
    q_l = _expmap0_t(e_l, c_l)                 # ball point in B_c_l
    q_l^0 = _expmap0_t(_logmap0_t(q_l,c_l),c0) # transferred ball point in B_c0
inner = _mobius_add_t(q_1^0, q_2^0, c0)
h = _mobius_add_t(q_0^0, inner, c0)             # preserve exact right nesting
z = _logmap0_t(h,c0)                            # (B,D), decoder tangent coordinates
require z.shape == (B,self.embed_dim) and finite(z)
return z
```

Retain dtype/device, batch order, autograd/STE, and existing validation, adding only the exact three-layer/input-shape guard required by the registered equation. Do not feed tangent `e_l` directly to `log0`; do not reorder/reassociate, omit source `exp→log→target exp`, add projection or extra output transforms, or generalize silently to another layer count. The radial exp/log simplification is valid only absent helper clipping/projection; do not claim it is unconditional. No exact HRA-paper telescope, residual inversion, or cross-curvature Möbius homomorphism is claimed. S08 measures actual helper-domain/clipping incidence and direct effect.

## FCCR-1 versus HRA contract boundary

Keep `logs/mechanism_contract_iter31.json` byte/schema semantics unchanged: exactly the inherited ten FCCR-1 fields and fixed vector; `formula_inputs` remains `behavior_branching` and `raw_residual_median`. Keep `logs/hra_step6_contract_iter31.json` as the separate HRA-STEP6-1 contract linked to FCCR-1. Never add HRA keys to FCCR JSON, loosen the loader's exact-schema check, alter/bypass the shared FCCR preflight, or create a fallback. Preserve both explicit input keys `branching` and `raw_residual_medians` and their historical provenance limits (no replay/hash proof, no fresh measurement, no normalized-layer-scale substitution). The parent `_load_closed_form_curvatures` explicitly checks those keys, recomputes the mapping, and enforces the exact ten-field FCCR schema (`curvature_RQ-VAE.py:305–382`); Iter31 may change its per-iteration contract filename and run-identity labels only.

## Complete source preparation inventory

Copy selectively from `stage2_RQ-VAE/curvature_RQ-VAE_iter29/` to the corresponding Iter31 relative paths; do not overwrite canonical Iter31 logs/contracts. Classification: **S** sole scientific delta, **W** non-scientific identity/path wiring or unchanged copied source/input, **A** audit tooling.

| Iter29 source → Iter31 target | Classification and exact action |
|---|---|
| `modules/rqvae.py` → same | **S** copy; change only `RqVae._step6_sum_embeddings` as above. |
| `curvature_config.py` → same | **W** update `_CONFIG_DIR`, `MECHANISM_NAME`, `SAVE_DIR_ROOT`, `CONFIG_PATH`, `BEST_CKPT_PATH`, `RQVAE_OUT_DIR`, `RQVAE_CKPT_PATH`, `RAW_SIDS_NPY`, `SIDS_NPY`, `ITEM_SIDS_JSON`. Set `MECHANISM_NAME="iter31_hra_step6_common_reference"`; retain all other inputs/settings. Parent path-constant pattern is `curvature_config.py:9–34`. |
| `curvature_RQ-VAE.py` → same | **W** change `_load_closed_form_curvatures` contract reference from `mechanism_contract_iter29.json` to Iter31, and current-run labels/messages/snapshot/invariant/Step3.5 diagnostics to Iter31. Preserve loader checks/arithmetic, fixed buffers, hyperparameters, launch/training behavior, and genuine Iter8 warm-start identity. Parent contract path and schema checks: `curvature_RQ-VAE.py:305–382`; snapshot identity example `:385–394`. |
| `scripts/compute_closed_form_curvature.py` → same | **W** copy computational code unchanged; adjust only Iter29 top-level run-identity wording if it labels this copied run. No recomputation or formula/value edits. |
| `scripts/computed_behavior_branching.json` → same | **W** frozen registered input copy byte-for-byte; no writer rerun/re-measurement. |
| `scripts/export_sids_for_stage3.py` → same | **W** copy unchanged; it imports the Iter31 SID path constants. |
| `scripts/mvg_check.py` → same | **A** update source/module identity and replace alternate-curvature counterfactual with the S08 same-model/state/checkpoint/batch Step6-vs-Euclidean comparison and compatible FCCR immutability/model-gradient checks. No execution in S06. |
| `scripts/grad_check.py` → same | **A** retain no-argument gradient-check delegation, update Iter31 import/identity only. No curvature-gradient expectation or duplicate alternate mechanism. |
| `modules/sid_quality.py`, `quantize.py`, `__init__.py`, `encoder.py`, `hab.py`, `hyperbolic.py`, `loss.py`, `normalize.py`, `step_checks.py`, `utils.py`, `tokenizer/semids.py` → same | **W** copy unchanged; preserve geometry helpers, quantizer, codebooks/STE, losses and other inherited behavior. |
| `data/__init__.py`, `amazon.py`, `instruments.py`, `ml1m.py`, `ml32m.py`, `preprocessing.py`, `processed.py`, `schemas.py`, `utils.py` → same | **W** copy unchanged. |
| `init/kmeans.py` → same | **W** copy unchanged. |
| none → `scripts/preflight_hra_step6_iter31.py` | **A** create the no-argument, no-environment-override HRA static checker specified below. S06 authorizes its one-time creation only. |
| current `stage3_T5Train/train_HG-Rec.py` → same current file | **W** modify only hard-coded `CODE_PATH`, `RQVAE_VARIANT`, `LOG_PATH`, `SAVE_PATH`, and `_LAUNCHER["log"]` as specified in Stage3 route. No trainer wrapper or model/evaluator/protocol change. |

**Do not copy:** `scripts/run_stage3_iter29.py` (historical wrapper only); Iter29 `logs/**`, deliberation, results, generated gin/run files, checkpoints/`.pth`/`.pt`, SID `.npy`, `item_sids.json`, caches, or generated `dataset/Instruments/**`/`out/**`. Do not copy any other unlisted parent script or utility unless a later evidence-backed canonical decision first expands this inventory. Existing Iter31 `logs/mechanism_contract_iter31.json`, `logs/hra_step6_contract_iter31.json`, and S00–S05 canonical records remain unchanged. There is no parent `configs/` source tree to duplicate. No Iter31 wrapper, unrelated refactor, cleanup, docs/metrics changes, or generated product in the source subtree.

## Stage2 output identities and path gate

Use full descriptive `MECHANISM_NAME="iter31_hra_step6_common_reference"`, but short product root `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`. All config/gin/checkpoint/SID outputs, including `SAVE_DIR_ROOT`, `CONFIG_PATH`, and `BEST_CKPT_PATH`, must stay under that root. Specifically:

- `RQVAE_OUT_DIR=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments`
- `RQVAE_CKPT_PATH=.../rqvae_best.pth`; `RAW_SIDS_NPY=.../sids_raw.npy`
- `SIDS_NPY=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy`
- `ITEM_SIDS_JSON=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`

Before any eventual Stage2 launch, S09 must directly inspect/verify `RQVAE_OUT_DIR` resolves under exactly `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/` and matches iter 31, as required by root `CLAUDE.md` §§0,10; also verify every Stage2 output consumer and ensure no `.pth`, `.npy`, or `item_sids.json` targets the source subtree. This verification is not performed or authorized here.

## Approved HRA checker scope (creation only)

Create `scripts/preflight_hra_step6_iter31.py` as a transparent, fail-closed, no-argument static source/contract checker. It must resolve its source root from `__file__`; inspect both canonical contract files plus the registered model source; verify exact HRA contract version/status/transition/equation ID and FCCR linkage; independently verify FCCR-1's exact ten-field schema/values/flags/input declaration; check exact L0/L1/L2 source-`c_l` exp/log and target-`c0` exp/add/final-log operation roles, right nesting, batch-major orientation and `(B,D)` decoder output; ensure `forward` retains the current Step6 decoder consumer and scope excludes unrelated model changes; and reject unsupported source layouts rather than passing from broad comment/string matches. It must not rewrite or bypass the shared FCCR preflight, alter either canonical contract, claim runtime activation, or invent a numerical direct-effect threshold. Report that S08 must measure finiteness/domain/shape, helper projection/log-clamp incidence, fixed-curvature/optimizer invariance, model gradient health and same-state direct output effect. S07 independently audits and runs this HRA checker separately from the unchanged FCCR preflight. Creation is authorized by this plan; running it is not.

## Replacement MVG and gradient gates (S08/S09 only)

Do not carry forward Iter29 `CONTROL_FIXED_C` alternate-curvature activation logic: that changes FCCR curvature, not HRA Step6. At S08, use exactly one approved warm-start checkpoint, one deterministic batch, one candidate state and the registered unchanged curvature vector. On identical `RqVaeOutput.embeddings`, compare `z_HRA = model._step6_sum_embeddings(output)` against `z_E = output.embeddings.sum(dim=0).transpose(0,1)`. Record exact checkpoint/batch provenance, shapes, finite status, values/norms/direct elementwise and L2 differences, and actual helper domain/projection/log-clamp incidence. Also verify exact fixed curvature values, non-grad buffers, optimizer exclusion, mode/step invariance, and finite nonzero intended model/codebook gradients; no curvature gradient is expected. No extra thresholds, parameter/seed/batch sweeps, or different-curvature control.

At pre-training S08/S09, satisfy root `CLAUDE.md` §6: check `total_loss.requires_grad` and `grad_fn`; execute actual `loss.backward()` on the registered batch; verify finite nonzero gradients on relevant trainable parameters; use `torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True)` for each relevant existing mechanism loss component where applicable; explicitly check no detach makes the changed decoder-facing path inert. Fixed curvature must not be expected to receive gradients. Static HRA checker pass is not MVG pass. If S08 shows registered mechanism invalid/inactive and repairing it requires changing equation/constants/mapping, use the skill's independent abort gate; do not tune in Iter31.

## Stage3 direct route; root §5/§11 path reconciliation

Select direct invocation of the existing `stage3_T5Train/train_HG-Rec.py` with no wrapper and no arguments, using only constants-only route wiring. Current primary source has generic SID/default output constants (`train_HG-Rec.py:121–159`), resolves absolute `CODE_PATH` as the requested input (`:215–258`), places training metrics and DDP sync files under configured `log_path` (`:827–849`), writes final `test_final.json` below `log_path` (`:1143–1154`), and writes the launcher child output through `_LAUNCHER["log"]` (`:1175–1211`). Keep the direct trainer's launcher script as `os.path.abspath(__file__)` and preserve the existing launcher invocation/protocol (`:1185–1212`).

Hard-code these Iter31 values in the trainer:

- `CODE_PATH="/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json"` exactly.
- `RQVAE_VARIANT="iter31_hra_step6_common_reference"` (full descriptive mechanism identity; do not rely on current generic basename map).
- `LOG_PATH="/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/"`.
- `SAVE_PATH="/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/ckpt/"`.
- `_LAUNCHER["log"]="/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log"` (root-level launcher file, not nested under `logs/`).
- `_LAUNCHER["script"]` stays `os.path.abspath(__file__)`.

**Outer nohup stdout/stderr path reconciliation:** root `CLAUDE.md` §5 gives an illustrative `> logs/_stage3_run.log` while running from the Stage3 source directory; §11 mandates Stage3 run outputs only beneath the Iter31 short results root. Reconcile narrowly by retaining §5's cwd (`stage3_T5Train`), absolute Python 3.10 executable, direct no-argument `train_HG-Rec.py` entry, nohup form and unchanged child launcher; direct the *outer* stdout/stderr to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log`. This is a path-only redirection adjustment to satisfy the stronger result-root mandate, not a wrapper, source cwd change, Python/entry change, CLI option, or protocol change. The internal launcher log remains the separate root-level `_stage3_launcher.log` path above. The direct command shape for eventual use is `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log 2>&1 &` from `stage3_T5Train`.

Do not edit model, data, evaluator, seed, epoch/early-stop semantics, beam, `n_eval`, metrics, training config, DDP count/port/environment or the root-prescribed launcher. This design appears wiring-feasible from the current path consumers, but it is not yet verified effective. **S11 must independently resolve every trainer/launcher output consumer, confirm all actual files—including outer log, internal launcher log, metrics, `test_final.json`, and checkpoint—under the Iter31 short results root, verify exact SID consumption and protocol, then adjudicate authorization before the single Stage3 run. Do not describe this route as verified before S11.** If S11 finds path-only wiring cannot meet the constraints, stop and record a hard blocker; never fall back silently to Iter29's wrapper.

## Post-apply gates, risks, and sequencing

This is an implementation plan, not applied code. After its canonical approval, the implementation is applied once, with a source/path identity sweep; S07 then independently audits applied source, unchanged inherited factors, FCCR loader identity, both contracts and path boundaries; and only S07 may run the approved HRA checker and unchanged shared FCCR preflight. S08 is the separately adjudicated one-checkpoint/one-batch MVG and fixed-curvature/model gradient gate. S09 independently verifies current inputs/checkpoint and all identities, directly verifies `RQVAE_OUT_DIR`, root §6 backward/component checks and all required prelaunch approvals before the one Stage2 execution. Stage2 metrics are descriptive, not gates.

Remaining required independent A/B/Judge gates: **S07_PREFLIGHT** (actual source/contract audit and both preflights), **S08_MVG** (runtime direct-effect/domain/gradient/invariance), **S09_STAGE2_EXECUTION** (one authorized Stage2 launch after all gate/path/identity checks), **S10_STAGE2_ANALYSIS** (single-run compliance/direct effect/SID analysis; proxies descriptive), **S11_STAGE3_EVALUATION** (route, SID, model/data/evaluator/protocol/output-root verification then one authorized Stage3 run), **S12_RESULT_CLASSIFICATION** (mechanism validity/activity separate from promotion, strict `test_R@10 > 0.065`), **S13_GIT_CLOSURE** (artifact and GitHub/main/remote-hash closure only after normal/abort criteria). **S14_GLOBAL_REVIEW** only if its trigger fires. S07/S08 are not authorized by this plan; no later stage is pre-approved or skipped.

Risks: tensor axis mistakes cause helper reductions over batch instead of embedding dimension; exp projection/log clamp may invalidate ideal radial identities; historical input provenance does not imply replay/hash identity; HRA could be inactive or numerically invalid; stale paths could route products outside results; direct Stage3 constants may not govern an uninspected output consumer. Do not retune registered HRA or weaken FCCR to resolve these. Any route or mechanism hard blocker is recorded at the relevant later adjudication and handled under the skill.

```text
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/implementation_plan_iter31.md; stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md
CONFIDENCE=HIGH for source-boundary/contract distinction; MEDIUM for applied wiring and eventual route/activation until S07/S08/S11
NO_UNRELATED_CHANGES=TRUE
S06_PLAN_APPROVAL=APPROVED
CHECKER_CREATION_AUTHORIZATION=APPROVED_ONCE_AS_LISTED; execution requires S07
CHECKER_MVG_STAGE2_STAGE3_EXECUTION_AUTHORIZATION=NONE
USER_INPUT_REQUIRED=NO
```