ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION
ROUND=1

# Agent A — S06 implementation design only

## Decision boundary

This is a file-by-file proposal for the orchestrator and Judge C; it is not an applied patch, preflight result, MVG result, or execution authorization. I did not modify source/configuration/results, run checks, or launch Stage2/Stage3/GPU work. S03 is PASS only for the canonical historical method/value provenance at its stated confidence (branching HIGH, raw residual MEDIUM); it does not imply recalculation, replay, historic byte identity, or runtime use.

Use iter29 only as the shared source-lineage parent. Do not compare or execute the historical iter26 tree as a second trainer. The prospective control is a fresh iter26-map run; the candidate is a fresh iter29-map run. Within each matched seed pair, the only scientific difference is the registered map and its fixed vector. Seeds vary only between pairs; map, seed, labels, and unique routes are profile metadata, never branches in model/training/reporting behavior.

## Proposed source-tree cutover

Create the iter30 implementation by copying the needed iter29 implementation tree into `stage2_RQ-VAE/curvature_RQ-VAE_iter30/`, without replacing any canonical iter30 `logs/` files. Preserve the existing iter30 root names so `.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` can run from the iter30 directory and inspect `curvature_RQ-VAE.py`, `modules/quantize.py`, and `modules/rqvae.py`.

Copy the iter29 root `curvature_RQ-VAE.py`, `curvature_config.py`, and the required `data/`, `init/`, and `modules/` Python implementation. Copy `scripts/computed_behavior_branching.json` byte-for-byte as the immutable registered data record, preserving its explicit `branching` and `raw_residual_medians` values and provenance fields. Copy the source-side no-argument helper/check/export code that remains in use, then make the specific edits below. Do not copy iter29 logs, checkpoints, generated SIDs, or result artifacts into the iter30 source subtree.

### `curvature_config.py` — shared settings plus six fixed profile records

Retain the iter29 common Stage2 input paths, environment/runtime settings, and all locked model/training settings. Change only iter identity, profile routing, profile seed/map metadata, and log paths. Hardcode an immutable six-entry profile table for exactly:

- `iter26_mapping_seed43`, `iter29_mapping_seed43`
- `iter26_mapping_seed44`, `iter29_mapping_seed44`
- `iter26_mapping_seed45`, `iter29_mapping_seed45`

Each record carries only its fixed `label`, `seed`, `arm`/map identifier, expected curvature vector, full `MECHANISM_NAME=iter30_<arm>_seed<seed>`, and unique destinations. Make the profile selection an import-time literal in each no-argument profile entrypoint; it must not inspect `sys.argv`, user environment, cwd, a mutable shared “current run” file, or a shell-injected value. The profile loader rejects every label outside the six-entry table. The profiles are data-only; no profile-specific training or metric code.

For every profile set the exact S01 routes:

```text
Stage2 root = <repo>/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/
RQVAE_OUT_DIR = <Stage2 root>/out/rqvae/instruments/
RQVAE_CKPT_PATH = <RQVAE_OUT_DIR>/rqvae_best.pth
RAW_SIDS_NPY = <RQVAE_OUT_DIR>/sids_raw.npy
SIDS_NPY = <Stage2 root>/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON = <Stage2 root>/item_sids.json
```

Keep the short result-root `curvature_RQ-VAE_iter30`; use the full arm/seed string only for `MECHANISM_NAME`/Stage3 `RQVAE_VARIANT`. Put all generated `.pth`, `.pt`, `.npy`, and `item_sids.json` files under that profile’s result root, never under the source iter30 subtree. Keep `ITEM_EMB_NPY`, `ITEM_IDS_JSON`, and `TRAIN_PARQUET` bound to the locked absolute inputs. Preserve the locked Stage2 768/[512,256,128]/32 model, 3×256 codebooks, commitment 1.0, batch 640/GPU, four ranks, 100000 global steps/checkpoint cadence 10000, AdamW 1e-3/1e-4, clip 1.0, Sinkhorn 0.05/3, behavior weight 0.20/temperature 0.07, zero curvature regularization, M2/M3 and all existing runtime/data/cache/determinism options. `PYTHONHASHSEED=42` and other inherited deterministic settings stay common; they are not the Stage2 pair seed.

The profile table must make the required `RQVAE_OUT_DIR` result-root paths visibly inspectable in `curvature_config.py` for the root rule’s pre-launch path review. Resolve and validate the active profile before importing the shared trainer constants; do not implement profile selection as an environment variable or mutate/rewrite `curvature_config.py` between runs.

### Six Stage2 no-argument profiles and common launcher

Create six tiny source entrypoints under `scripts/`, one per exact label (for example `run_stage2_iter26_mapping_seed43.py` through `run_stage2_iter29_mapping_seed45.py`). Each contains one literal profile label and delegates to one shared profile bootstrap; it accepts no parameters. The bootstrap binds that profile, imports the single root `curvature_RQ-VAE.py` trainer, and uses the existing hardcoded four-rank torchrun pattern on port 50200. Set the subprocess script path to the selected profile entrypoint, not to the generic trainer, so each child rank re-executes the same baked-in profile without a CLI argument or profile environment variable. Under `RANK`, the wrapper applies the same immutable profile and calls the shared trainer’s `main()`; otherwise it invokes the common no-argument launcher. This is the only profile-specific execution boundary: the root `curvature_RQ-VAE.py` remains the one shared implementation and preflight target, while the wrappers contain no model, training, map-calculation, or reporting logic. Invoke each profile script directly with no arguments and in the authorized serial order; preserve the required Python executable, port/GPU settings, and existing internal torchrun flags. Do not create six trainer copies, change a shared active-profile file between runs, or select profiles through CLI/environment values.

Write a distinct outer run stdout log and internal torchrun log per label (e.g. `logs/train_run_<label>.log` and `logs/train_migrated_<label>.log`). Never reuse the iter29 log names or a shared DDP/launcher file. Preserve the project’s required Python executable and serial port/GPU launcher settings; do not add command flags beyond the existing internal torchrun flags.

### `scripts/compute_closed_form_curvature.py` — one pure dual-map implementation

Replace the single-candidate writer with one shared, side-effect-free computation/validation module. It must read the same immutable JSON payload for both arms and require the exact `branching` and **plural `raw_residual_medians`** keys, ordered `[L0,L1,L2]`. Validate object/list types, length 3, numeric-not-bool values, finite values, nonnegative branching, strictly positive raw medians, and exact registered input vectors. Missing or changed `raw_residual_medians` is a hard error even if `residual_norm`, normalized scales, or cached curvature outputs are present. Never access/fallback to `residual_norm`, `[0.001,0.932889,1.0]`, another checkpoint, an inferred value, or the old iter26 consumer.

Implement both approved equations in the same helper and common validation path:

- Control `iter26_mapping`: `m_min=min(m_raw)`; `s=log1p(B)/log1p(m_raw/m_min)`; `z=(s-mean(s))/(std(s, ddof=0)+1e-12)`; `c=clip(0.5*exp(0.2*z),0.05,1.5)`. Expected vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- Candidate `iter29_mapping`: `x=B/(B+2.0)`; `y=m_raw/(m_raw+0.1)`; `u=(x+y)/2`; `c=0.05+1.45*u`, with no additional normalization/clipping. Expected vector `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.

Return formula intermediates, map id, and the selected fixed vector from one common API keyed only by the hardcoded profile’s map id. Compare all returned values to the corresponding registered intermediates/vector with the shared `FORMULA_COMPARISON_TOL=1e-9`; also check finiteness and `[0.05,1.50]` support. Remove the current helper’s behavior that rewrites `computed_behavior_branching.json` with map-specific outputs/metadata: it is input evidence, not a mutable cross-run output. Do not add a second contract JSON or put the control vector into `mechanism_contract_iter30.json`.

### `curvature_RQ-VAE.py` — common consumer, warm-start, logs, and no gate

Use the profile’s map id/vector through the same helper for all six runs. Preserve the current per-step fixed-buffer snapshots and invariance checks, but log the active map/seed/full mechanism name and both selected/intermediate values unambiguously. Never branch losses, optimizer, quantizer, sampler, runtime, SID cadence, or reporting on arm. Fixed curvature remains a buffer passed to the existing RQ-VAE constructor; it is not a Parameter and remains outside optimizer parameter groups. Preserve `CURVATURE_REG_WEIGHT=0`, fixed `get_c()`, no cyclic schedule, and all inherited M2/M3/curvature consumers.

Replace the fail-open iter8 warm-start block with a shared fail-closed loader (implemented in one small common helper, preferably `modules/warm_start.py`, and called by both trainer and checks). For every one of six profiles, immediately before the run and again in the training rank-0 load path:

1. Require the exact absolute S01 path `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, regular readable file, and SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Hash the bytes before deserialization and fail on any mismatch; record resolved path, digest, profile label, and run.
2. Load and validate a mapping containing a mapping-valued `model` state dict. Reject malformed state, non-tensor entries, unrecognized metadata structure, or incompatible target keys/shapes/dtypes. Do not catch and continue from random initialization.
3. Permit skipping only the S01-identified legacy curvature names for the three layers: `layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c`, where present. Do not use broad suffix filters. The new target model’s exact expected missing set is `{layers.0._fixed_c, layers.1._fixed_c, layers.2._fixed_c}` because the current profile installs its registered fixed values. Every other target state tensor must transfer with exact name/shape/dtype; every unrecognized source key or other missing/unexpected/mismatched target key fails closed.
4. After `load_state_dict(strict=False)`, require no unexpected keys and require the missing-key set to equal exactly those three `_fixed_c` buffers. Verify that each transferred tensor equals its checkpoint tensor after load, transferred compatible tensor count is positive (log the exact count/key count), and fixed buffers still equal the active map vector. Do not count skipped legacy curvature as transferred.
5. In DDP, communicate rank-0 load/hash success or failure before continuing so nonzero ranks cannot silently train or hang at the warm-start barrier. Do not construct/wrap a run that has not passed this check.

Preserve six distinct launcher logs and add a run-start record with profile/script/config hashes, active map/vector, seed, input identities/actual consumers, warm-start digest and positive transfer count. Keep path/run routing logs common in schema across arms.

### `modules/sid_quality.py` and checkpoint caller — common root-compliant reporting

Copy the iter29 in-process `evaluate_sid_quality(sids)` and use that same reporter/call path for all six arms. Keep only SID-derived descriptive metrics (Gini, per-layer utilization/diversity, collision/unique counts, conditional entropy and related existing descriptive fields). Keep `should_early_stop()` returning `(False, "")`; do not add gates, thresholds, or early termination. Explicitly remove/avoid iter26’s active `evaluate_sid_quality_full`/`sid_metrics_any.py` HitRate@50 subprocess; do not include HR@50 at all. Reporting failures for optional descriptive metrics must not alter training, while final raw SID/export failures remain real artifact failures. This copy/cutover is common hygiene, not an arm difference.

### `scripts/grad_check.py` — six separate imminent run gates

Make the shared gradient-check implementation profile-aware only through the same explicit profile record, then add six no-argument thin wrappers (`grad_check_<label>.py`) that each bake one literal label. For each actual Stage2 invocation, run its own wrapper immediately before launch; do not substitute one common block check or S08 MVG for these six checks. It must seed with that profile’s 43/44/45 value, load the one locked checkpoint and one batch from the locked `TransitionDataset`/train transitions and common batch semantics, build the exact model/map/profile, and apply the same SHA/tensor-transfer validation above.

Require finite scalar total loss with `requires_grad=True` and non-null `grad_fn`; execute the real `loss.backward()` path; check finite, nonzero gradients on the intended trainable model/loss paths (including the active reconstruction, quantization/commitment, and behavior losses using `autograd.grad(... retain_graph=True, allow_unused=True)` as appropriate); inspect detach hazards such as `_last_*.detach()`; verify fixed curvature buffers have `requires_grad=False`, are not named parameters or optimizer members, and receive no gradients. Curvature itself must not be expected to have nonzero gradients. Record per-profile checkpoint path/hash, seed/map/vector, batch identity/shape, loss/grad path results, and PASS/FAIL. Any failure blocks that run and requires root-policy repair/R50 before any GPU training.

### `scripts/mvg_check.py` — single common S08 counterfactual, both maps

Adapt the iter29 checker into one two-map S08 check, using the common map and warm-start helpers, one locked checkpoint and one deterministic train batch. Verify both map outputs/intermediates against registered values and fixed-buffer/time invariance at steps 0/25k/50k/100k in train/eval; curvature receives no gradient and is absent from optimizer groups. Test model gradient health while allowing only fixed buffers to lack gradients. Verify that the candidate FCCR-1 contract remains exactly the ten canonical S04 fields and candidate vector. Separately verify the control formula/vector against its registered iter26 values without adding a control contract or comparing the control vector to the candidate contract’s `final_curvature_values`.

For the activation counterfactual, construct control and candidate models from the same checkpoint with identical non-curvature state and identical input batch; verify every non-`_fixed_c` state tensor is equal. Use one identical encoded L0 input and run the same L0 `Quantize.forward` path for each fixed map. Capture the actual distance tensor after its existing optional flattening/fallback and immediately before `_center_distance_for_constraint` (e.g. a temporary checker-only wrapper around that method; do not retain/copy distances in production training). Repeat each unchanged-arm computation on the same checkpoint/batch/device/dtype. Define `M_distance=max_ij(abs(D29[i,j]-D26[i,j]))`; define repeat tolerance as the maximum within-arm max-absolute distance difference over the repeated D26 and D29 computations. PASS activation only if `M_distance > repeat_tolerance`. Report both tensors’ dtype/device, shape, `M_distance`, repeat differences/tolerance, both map outputs, and the registered source boundary. Do not use an unsupported absolute 1e-6 activation cutoff; the quantizer’s `max-min <= 1e-6` fallback is a distinct branch test. Do not use the older checker’s assignment/loss-only shortcut as the registered primary signal.

This is one common two-arm S08 counterfactual, not six per-profile gradient checks and not a training run. A no-activation failure blocks progression/activating changes; do not retune either map.

### Stage2 SID export

Keep the existing trainer’s single common `_export_4token_sids_for_stage3()` code path, collision extension, and read-back checks. Keep `RAW_SIDS_NPY`, `SIDS_NPY`, and `ITEM_SIDS_JSON` profile-specific and absolute. Do not duplicate the mapping/export algorithm in a second profile implementation. If retaining `scripts/export_sids_for_stage3.py`, make it a no-argument delegator to the common export function and active hardcoded profile only; otherwise omit the unused duplicate from the iter30 copied scripts. At launch, verify exported NPY shape/dtype/range/uniqueness and that JSON has exactly dense item keys, four tokens per item, and content matching that profile’s NPY; record JSON and NPY hashes.

## Six Stage3 wrappers; trainer is frozen

Create six small no-argument wrappers `scripts/run_stage3_<label>.py`, each with one hardcoded profile label, plus at most one shared wrapper helper. Follow `scripts/run_stage3_iter29.py`: import the existing root `stage3_T5Train/train_HG-Rec.py` by absolute path; set only the wrapper-owned Stage3 inputs/route and pair seed; set the wrapper itself as `_LAUNCHER["script"]`; preserve `nproc=4`, port 50201, the root-required internal NCCL settings, and unique launcher stdout path. For label/seed `<arm>/<seed>` set:

```text
CODE_PATH=<matching results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/item_sids.json>
RQVAE_VARIANT=iter30_<arm>_seed<seed>
SEED=<same pair seed 43|44|45>
LOG_PATH=<matching results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/logs/>
SAVE_PATH=<matching results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/ckpt/>
_LAUNCHER["log"]=<Stage3 root>/logs/_stage3_launcher.log
```

Before calling trainer/launcher, fail if the exact matching JSON is missing, non-regular, or outside the expected root; resolve `_resolve_code_file` and require its result to equal that exact JSON (never allow fallback to a stale SID). Prelaunch gate verifies the expected JSON digest and full destination inventory; the wrapper stdout log is a separate label-specific file beneath its Stage3 root. Use the same helper/template in all six wrappers. Do not edit `stage3_T5Train/train_HG-Rec.py` or its metrics/path fallback behavior; preserve locked trainer SHA `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`, architecture/optimizer/precision/data/evaluation settings, 150 configured maximum epochs with existing NO_EVAL/train-loss patience behavior, and mandatory final test. Record/disclose the existing `unknown_variant` metrics metadata caveat; do not “fix” it by touching the trainer.

## Prelaunch inventories, cleanup, and `.gitignore`

All six Stage2 destinations and all six Stage3 destinations are unique. Before each run, record hashes of that profile’s entrypoint, shared trainer, config, shared map/helper, warm-start checker, and relevant Stage3 wrapper/trainer as applicable; inventory the complete Stage2/Stage3 destination including the external SID `.npy` and JSON paths. Require expected product destinations to be absent/empty or have the explicitly recorded pre-run inventory permitted by S01; never silently overwrite a previous run. Stage3 wrapper must not start before its matching completed Stage2 export and its later independent S10/S11 authorizations.

Keep Step0 cleanup strictly limited to the active profile’s own `RQVAE_OUT_DIR` and the existing enumerated Stage2 artifacts in that directory. Do not point it at the short iter30 parent, a sibling label root, `dataset/Instruments`, `item_sids.json`, source `logs/`, or Stage3 outputs. Its patterns do not clear external `SIDS_NPY` or `ITEM_SIDS_JSON`; check these separately and fail closed on collision. Do not add broad recursive cleanup or delete preexisting unrelated files. No output path may enter the iter30 source directory.

In root `.gitignore`, add a narrow iter30 deliverable exception next to the iter29 exception: re-include `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/**` and `results/stage3_T5Train/curvature_RQ-VAE_iter30/ckpt/` plus that Stage3 root’s contents after the global `*.pth`, `*.pt`, `*.npy`, and `ckpt*` ignores. Then re-exclude `__pycache__`, the shared `dataset/Instruments/item_emb.npy`, Stage2 `_ddp_sync`, tensorboard, waits, and transient `_stage3_run*.log` under iter30 using the existing specific patterns/convention. Keep Stage3 `_stage3_launcher.log` included as required. Do not whitelist the entire `results/` tree or remove global ignores.

## Intended smoke-check sequence after canonical patch application (design only; none run here)

Before later S07/S08 gates, the orchestrator may run only narrow, non-GPU smoke checks; they must not become gates replacing the separately adjudicated preflight/MVG:

1. A throwaway no-argument CPU script imports the pure mapping helper and the copied immutable input JSON. It verifies both exact equation/intermediate/vector outputs, candidate vector equals the exact S04 contract vector, control vector equals its separate registered vector, all values are finite/in-range, and a copied payload missing `raw_residual_medians` fails even if a `residual_norm` alias is present. It must not mutate the source JSON or write result outputs.
2. A throwaway no-argument route/profile inspection loads all six entrypoint records without invoking their launcher. Assert exactly the six labels, matched seed/arm pairs, profile-specific mapping/vector, full mechanism labels, all absolute Stage2 paths below `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/`, all Stage3 routes below `results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/`, all six destinations/log names pairwise distinct, and no `argparse`/`sys.argv`/environment-based profile selector. Print a source/config/profile hash inventory; create no directories or products.
3. Inspect (do not invoke) the six Stage3 wrappers’ constants and assert all point to the corresponding absolute SID JSON, seed, full variant, logs, ckpt, launcher/stdout locations, port 50201, and locked trainer SHA. Confirm the trainer file is unchanged. Do not call `_resolve_code_file` on a missing output or let the wrapper fallback.
4. If needed, a CPU-only isolated checkpoint-transfer smoke may exercise the strict warm-start helper against the exact locked checkpoint and a constructed model without forward/backward, optimizer step, writes, or launcher; it must verify SHA, transfer count/key set, exact expected missing `_fixed_c` set, and that corrupted/malformed state fails. Do not treat it as any per-run gradient check or as MVG.

These smoke checks are after the orchestrator applies the Judge-approved patch, not part of this independent S06 design turn. Do not run preflight, MVG, per-run GPU gradient checks, Stage2, Stage3, or GPU during S06. After the smoke checks, proceed only via canonical S07 preflight and S08 MVG. Any real Stage2 execution remains blocked until all later adjudications pass; then every one of the six profiles gets its own immediately-prior input/hash/path/warm-start/output/gradient gates. No smoke result authorizes training.

## Evidence and risks

Direct inspected iter29 source already supplies a fixed-buffer quantizer, `CURVATURE_REG_WEIGHT=0`, common in-process descriptive SID quality implementation, Step0’s own-`RQVAE_OUT_DIR` cleanup boundary, trainer-inline final SID export, and a fail-open iter8 loader without checksum/completeness checks. The root preflight only inspects the root trainer, `modules/quantize.py`, and `modules/rqvae.py`, and reads the single canonical S04 contract; it cannot by itself verify the separate iter26 control. S07/S08 must explicitly check both vectors while keeping that contract candidate-only.

Highest implementation risks are accidental profile selection leakage into DDP children, using the ambiguous legacy residual alias, comparing control curvature to the candidate-only contract, treating the helper’s old JSON writer as a shared output, assuming Step0 cleans external SID paths, reusing a Stage3 output or allowing `_resolve_code_file` to find a stale SID, or editing the frozen Stage3 trainer to alter variant metadata. The proposed literal-profile wrappers, pure dual-map helper, exact output inventories, strict warm-start transfer accounting, common report/consumer path, and unchanged Stage3 trainer address these risks. If these cannot be implemented with all six independent no-argument profiles and no treatment-specific training/reporting logic, block before S07/launch rather than relaxing any locked factor.