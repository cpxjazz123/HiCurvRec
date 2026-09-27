ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S06_IMPLEMENTATION/round_1/source_packet.md
STAGE_ID=S06_IMPLEMENTATION
ROUND=1

# Agent B — Iter30 S06 implementation design

## Decision boundary

This is a complete source-patch proposal only. It is not applied source, evidence that iter30 profiles exist or pass, a preflight/MVG result, or permission to run Stage2, Stage3, or GPU work. Judge C must select/materialize the canonical implementation plan before the orchestrator edits source. S07/S08/S09 and the later per-run input, output, warm-start, gradient, and launch gates remain blocking.

Use iter29 as the common source-lineage parent; do not use its whole tree as one experiment arm. The future six runs share one trainer, input loader, model/loss/optimizer/runtime path, warm-start policy, quality-report/export implementation, and unchanged Stage3 trainer. The only scientific selector is the registered iter26 versus iter29 mapping and its resulting fixed vector. Pair seeds, descriptive variant labels, and output/log routes are protocol/routing fields, not training branches. Preserve S03's historical method/value-only limitation (branching HIGH, raw residual MEDIUM); do not describe it as replay or historical byte identity.

## Common code and mapping path

1. Copy the iter29 implementation dependency tree into `stage2_RQ-VAE/curvature_RQ-VAE_iter30/`: `curvature_RQ-VAE.py`, `curvature_config.py`, and the source files in `data/`, `init/`, `modules/`, and `scripts/` that the trainer imports. Do not copy iter29 logs, `__pycache__`, result products, checkpoints, SID arrays, or the Stage3 trainer. Preserve iter30's already-canonical S00–S05 records. Keep all generated `.pth`, `.pt`, `.npy`, and `item_sids.json` outside this source tree.
2. In the one shared `scripts/compute_closed_form_curvature.py`, retain iter29's currently validated candidate calculation and add the exact registered iter26 control calculation behind the same explicit `mapping_id` argument. Return the selected vector and its formula intermediates from one common pure calculation interface; do not fork the trainer, loss, data, or model by arm. Control: `s=log1p(B)/log1p(m/min(m))`, `z=(s-mean(s))/(std_population(s)+1e-12)` (`ddof=0`), `c26=clip(0.5*exp(0.2*z),0.05,1.5)`. Candidate: `x=B/(B+2.0)`, `y=m/(m+0.1)`, `u=(x+y)/2`, `c29=0.05+1.45*u`. Preserve the registered ordered values: `B=[19.324911558712664,1.4605688962651735,1.0148104414712726]`; `raw_residual_medians=[1.0,0.10941,0.09331]`; `c26=[0.6145357379232853,0.5333020920777128,0.3814078098431606]`; `c29=[1.3660953164241916,0.7347829661951981,0.6439958072706683]`. Do not add a third equation, tune constants, or derive a new input.
3. Keep one copied `scripts/computed_behavior_branching.json` as the common explicit registered input record. The shared loader must require the literal plural source key `raw_residual_medians` and `branching`, validate ordered length-three numeric finite values and exact registered inputs before computing either map, and reject missing/discrepant values. It must never read `residual_norm`, `RESIDUAL_NORMS`, normalized scales `[0.001,0.932889,1.0]`, another checkpoint, or inferred/recalculated replacements. The singular `raw_residual_median` in the exact S04 ten-field contract remains the semantic label only; the runtime JSON key is plural. Validate formula intermediates, finite/range bounds, map identity, and resulting vector at the common loader boundary.
4. Keep `logs/mechanism_contract_iter30.json` exactly the canonical ten-field S04 object, with the iter29 candidate vector only. Do not add control values or metadata to it. Candidate runs must match this contract. The control vector is a separate explicit assertion in the common profile/MVG code and later S07/S08 evidence; it is not a second contract. Both vectors populate the same `fixed_curvature` buffers and existing consumers. Keep `CURVATURE_REG_WEIGHT=0.0`, no curvature `Parameter`, no schedule, and time-invariant `get_c()`.
5. Modify the shared `curvature_RQ-VAE.py` only to parameterize the profile identity (seed/map/paths/logs) and use the common map loader. Preserve the complete iter29 training/data/model/optimizer/loss/quantizer path and every S01 setting: input 768; hidden `[512,256,128]`; embedding 32; three 256-entry quantizers; commitment 1.0; batch 640/GPU, four DDP ranks/global 2560; 100000 steps, 10000-step checkpoint cadence; AdamW `1e-3`/`1e-4`; clip 1.0; Sinkhorn `0.05`/3; behavior loss `0.20`/temperature `0.07`; curvature regularization zero; same M2/M3 consumers, sampling, runtime, deterministic/cache/GPU settings, and SID cadence. Preserve existing effective epsilon `sk_eps*(c/c_cyclic_max)` rather than introducing an arm-specific rule.

## Six hardcoded no-argument Stage2 routes

Use one shared trainer and a closed six-entry profile table in `curvature_config.py` (or one adjacent shared profile module). A profile contains only its fixed run label, registered map ID/vector identity, matched seed, full mechanism/variant label, absolute paths, source-profile identity, and logs. It has no CLI selector, environment selector, or injected shell variable. Use `curvature_RQ-VAE.py` as the hardcoded `iter26_mapping_seed43` no-argument profile so the root's prescribed no-argument Stage2 entrypoint remains valid; the other five routes are thin no-argument scripts under `scripts/` that select one literal table entry and call the same profile runner. The runner imports the shared trainer, applies only the selected immutable profile globals before `main()`, and points the inherited internal torchrun launcher at the selected profile script for worker re-entry. Workers reapply that same literal profile; `RANK` is used only to distinguish launcher/worker execution, never to choose the run. All runs use the inherited four-rank launcher and port 50200 serially.

| Entry/profile | Run label | Stage2 and Stage3 seed | Map | Exact Stage2 root |
|---|---|---:|---|---|
| `curvature_RQ-VAE.py` | `iter26_mapping_seed43` | 43 | iter26 control | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| `scripts/run_iter29_mapping_seed43.py` | `iter29_mapping_seed43` | 43 | iter29 candidate | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| `scripts/run_iter26_mapping_seed44.py` | `iter26_mapping_seed44` | 44 | iter26 control | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| `scripts/run_iter29_mapping_seed44.py` | `iter29_mapping_seed44` | 44 | iter29 candidate | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| `scripts/run_iter26_mapping_seed45.py` | `iter26_mapping_seed45` | 45 | iter26 control | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| `scripts/run_iter29_mapping_seed45.py` | `iter29_mapping_seed45` | 45 | iter29 candidate | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

For each route set exactly:

```text
RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments/
RQVAE_CKPT_PATH=<RQVAE_OUT_DIR>/rqvae_best.pth
RAW_SIDS_NPY=<RQVAE_OUT_DIR>/sids_raw.npy
SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON=<Stage2 root>/item_sids.json
MECHANISM_NAME=iter30_<iter26_mapping|iter29_mapping>_seed<43|44|45>
```

`curvature_config.py`'s static `RQVAE_OUT_DIR` default must itself resolve under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/` (the default route above) for the root's required config-path check. Each wrapper then sets its own table entry in memory before training; no results are written under the source tree. Do not encode training conditions by arm: the shared profile applies the selected vector once, then calls the same `main()`.

Immediately before each profile launch, a shared read-only inventory guard must validate the label/map/seed/path tuple, resolved short roots, matching per-run output destinations, relevant source/config/profile hashes, and locked input hashes. Record the future source/config/profile and input digests plus complete pre-run destination inventory in a unique `logs/run_identity_<label>.json` (the six future profile digests are not known at S06). Stage2 outer stdout logs are `logs/train_run_<label>.log`; inherited internal torchrun logs are `logs/train_migrated_<label>.log`, both beneath iter30 source `logs/`. Run labels, not timestamps or arm inference, determine all output paths.

The exact locked warm-start used by every profile is `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Rehash the actual Stage2 consumers before each invocation: Stage1 `sentence_t5.npy` (`6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`), Stage1 `item_ids.json` (`3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30` as locked in the canonical manifest), Stage0 `train.parquet` (`80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`), and the warm-start above. Check resolved paths and the call chain that actually consumes them. Do not report Stage0 `valid.parquet`, `test.parquet`, or `items.parquet`, nor declared-but-unused `ITEM_JSON_PATH`, as Stage2 runtime consumers.

## Fail-closed warm start (one common implementation)

Replace the current iter29 `exists`/warn/random-fallback block in `curvature_RQ-VAE.py` with one shared loader used by all six profiles and their gradient checks:

- Require the one exact absolute locked iter8 path, a readable regular file, and its SHA256 equal to the S01 digest immediately before loading. Missing file, digest mismatch, path mismatch, hash/read error, deserialization error, or malformed payload is fatal; there is no warning-and-random-init path.
- Require a mapping payload with the expected `model` state dictionary and tensor values. Compare source and target state keys/shapes/dtypes explicitly before transfer. Permit skipping only the S01-approved legacy curvature names `layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c` where present. Do not strip arbitrary prefixes, discard arbitrary unexpected keys, or rely on `strict=False` alone.
- Every target state tensor other than the current three `layers.{i}._fixed_c` buffers must have exactly one compatible source tensor; reject absent, unexpected, shape/dtype-incompatible, or non-finite transferred tensors. Load only those compatible tensors. After load, require `missing_keys` to equal exactly the expected current `_fixed_c` buffer keys, `unexpected_keys` empty, and transferred-compatible count positive and equal to the expected non-curvature transfer set. Assert and retain the selected profile's fixed buffers, non-trainable and absent from optimizer groups. Log the digest, compatible count, explicit legacy skips, and expected `_fixed_c` missing keys for every run.
- For DDP, make warm-start failure collective: rank zero reports success/failure before other ranks pass a barrier; on success DDP's initial module-state broadcast makes every worker start from the loaded state. On failure all ranks exit before training rather than hanging at a later barrier. The pre-launch digest/identity record and per-run positive transfer record are required; a shared pre-check alone does not replace runtime verification.

## Common SID reporting/export

Use the iter29 in-process `modules/sid_quality.py` implementation and its shared caller for both arms. It computes only SID-derived descriptive occupancy/diversity/conditional-entropy metrics and its `should_early_stop()` is permanently `(False, "")`. Remove any HR@50, embedding-neighbor subprocess, temporary SID metrics file, quality gate, or report-driven run admission/stop path. Retain the same raw three-token SID generation cadence and final TIGER-compatible four-token export for all six runs; export validation failures are data/pipeline integrity failures, not SID-quality gates. `modules/sid_quality.py` can be copied unchanged if its in-process-only implementation remains intact; remove stale comments/callers rather than add another reporter.

The common export must read the profile's exact `RAW_SIDS_NPY`, write only that profile's `SIDS_NPY` and `ITEM_SIDS_JSON`, and retain checks for expected item count, token ranges, collision extension uniqueness, JSON keys/width/sample parity, and NPY read-back parity. No exporter may silently read or write a sibling run's paths.

## Six per-run Stage2 gradient checks

Add one common checker in `scripts/grad_check.py` and six tiny no-argument entry scripts (one per exact run label) that select a literal profile entry and call that checker. Each check is performed immediately before that profile's actual Stage2 training, after the other applicable gates; it cannot be replaced by one control/candidate block check, a generic MVG, or S08. For that profile's actual fixed-curvature vector, exact input/config, locked warm-start, and one real batch, instantiate the common model on the prescribed check device, strictly transfer the locked warm-start tensors, run a forward and `total_loss.backward()`, and verify:

- `total_loss.requires_grad`, non-null `grad_fn`, finite loss and finite nonzero gradients on intended trainable model parameters; check active differentiable reconstruction, quantization/commitment, behavior, and inherited geometry paths as applicable, including `_last_*` detach hazards.
- The loss components that are expected to be differentiable retain graph and have at least one finite nonzero parameter gradient. Curvature regularization is exactly zero and is not expected to carry gradient.
- The three selected fixed buffers equal that profile's registered map vector, are buffers rather than `nn.Parameter`s, receive no gradients, and are not in the optimizer parameter groups.
- Log the profile label, seed/map/vector, checkpoint digest and positive transfer count, batch identity, loss/gradient-path outcomes, and source/config/profile hashes. The six checks must be run as six distinct checks; none is executed at S06.

## Common two-arm S08 counterfactual

Adapt the shared `scripts/mvg_check.py` for the exact S02 signal; do not retain its current loss/assignment-only activation rule as the primary measure. Use the same warm-start checkpoint, same actual batch, same encoder output, model/codebook state, evaluation mode, device, and dtype in both calculations. Build/load otherwise-identical candidate and control models and substitute only the three registered fixed-curvature buffers.

In the common `Quantize.forward` path, capture the actual L0 per-example-by-code Poincare distance tensor after any existing flattening fallback and immediately before `_center_distance_for_constraint` (which is immediately before the unchanged centering/Sinkhorn assignment path). The capture can be an S08-only wrapper around `_center_distance_for_constraint`; avoid changing training behavior or adding a production branch. Define `M_distance=max_ij(abs(D29[i,j]-D26[i,j]))`. Repeat each unchanged-arm computation on the same checkpoint/batch/device/dtype and compute the maximum same-arm repeat difference for this statistic; use the maximum arm repeat difference as the observed deterministic numerical-repeat tolerance. Count direct activation only if `M_distance` exceeds that observed tolerance. Record the two captured tensors' dtype/device, each arm's vector, `M_distance`, repeat differences, and tolerance. Do not adopt an unsupported absolute `1e-6` activation threshold; the quantizer's existing `1e-6` flattening-fallback predicate is a different statistic. S08 is one common check for both arms, not a substitute for any of the six pre-training gradient checks.

## Six Stage3 no-argument wrappers; trainer frozen

Create six wrappers under iter30 `scripts/`, each one hardcoded label, calling a common wrapper routine that dynamically imports the root Stage3 trainer and sets only route/profile globals before the existing `main()`/launcher. For every run set:

```text
CODE_PATH=<that run's absolute Stage2 root>/item_sids.json
RQVAE_VARIANT=iter30_<iter26_mapping|iter29_mapping>_seed<seed>
SEED=<matching Stage2 seed 43|44|45>
LOG_PATH=results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/logs/
SAVE_PATH=results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/ckpt/
_LAUNCHER["log"]=<same Stage3 root>/logs/_stage3_launcher.log
```

Set the wrapper itself as `_LAUNCHER["script"]` so all four workers re-enter the same no-argument wrapper. Use Stage3 port 50201 serially and preserve all trainer settings, notably 150 maximum epochs, `NO_EVAL=True` with existing train-loss patience 10, `SKIP_TEST=False`, top K `[5,10]`, beam 20, expected `n_eval=57439`, batch 4096, inference 1024, four ranks, optimizer/scheduler, determinism, precision, NCCL, data, and test behavior. Do not patch `stage3_T5Train/train_HG-Rec.py` or the current iter29 historical metadata caveat.

For each matching Stage3 launch rehash locked Stage0 `train.parquet` and `test.parquet`, verify the unchanged trainer's SHA256 is exactly `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`, verify the selected Stage2 `item_sids.json` path/hash, and write its own run identity record. Do not claim Stage0 valid is consumed while `NO_EVAL=True`; Stage3 train and final test are the actual parquet consumers. The Stage3 wrapper must fail closed if its own exact SID JSON is absent, non-absolute, resolves outside its paired Stage2 root, or differs from the file returned by the trainer's `_resolve_code_file`. Enforce this without editing the trainer by wrapping/guarding that resolver in the wrapper and rejecting any fallback result that is not the exact expected resolved path. Record a distinct wrapper stdout log beneath each run's Stage3 `logs/` (e.g. `_stage3_wrapper_stdout.log`) plus the distinct `_stage3_launcher.log`.

| Run | Exact Stage3 result root |
|---|---|
| `iter26_mapping_seed43` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| `iter29_mapping_seed43` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| `iter26_mapping_seed44` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| `iter29_mapping_seed44` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| `iter26_mapping_seed45` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| `iter29_mapping_seed45` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

Expected products remain inside these short roots, e.g. `logs/Amazon_2023_Instruments/<timestamp>/test_final.json` and matching `ckpt/Amazon_2023_Instruments/<same timestamp>/HG_Rec_best.pth`. All six route roots are distinct.

## Collision prevention, cleanup limits, and `.gitignore`

Every Stage2 profile checks before launch that its complete `RQVAE_OUT_DIR`, external per-profile `SIDS_NPY`, external `ITEM_SIDS_JSON`, source outer/internal logs, and run identity path are absent or empty and that no profile destination aliases another route. Every Stage3 wrapper checks that its whole Stage3 root/log/checkpoint/wrapper-log/identity destinations are absent or empty. Refuse a nonempty destination; never overwrite, reuse, or clean another label's output. Record the pre-run inventory. Run serially; no concurrent Stage2/Stage3 process may share port 50200/50201. The existing `check_step0_clean_output` is limited to its own `RQVAE_OUT_DIR` and runs only after the route inventory guard; it is not a substitute for inventory. Stage2 external `SIDS_NPY`/JSON outputs are not covered by Step0 cleanup. Do not remove files automatically to make a route appear clean.

Extend the root `.gitignore` only after its broad `**/*.pth`, `**/*.pt`, `**/*.npy`, `**/ckpt*/` and result checkpoint rules with narrow iter30 deliverable exceptions for:

```text
!results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/
!results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/**
!results/stage3_T5Train/curvature_RQ-VAE_iter30/
!results/stage3_T5Train/curvature_RQ-VAE_iter30/**
```

Then re-apply the existing transient exclusions scoped to these iter30 roots: shared `dataset/Instruments/item_emb.npy`, `logs/_ddp_sync/`, `logs/tensorboard/`, `logs/wait_*.log`, `logs/_stage3_run*.log`, and any `__pycache__`. This permits required Stage2/Stage3 checkpoint/SID artifacts and launcher logs to be committed while retaining the repository's common-input and transient-output rules. Do not broadly unignore other iterations or any source-tree product path.

## Files to create/copy/modify

- **Copy common iter29 source into iter30**: `curvature_RQ-VAE.py`; `curvature_config.py` as the base to update; required `data/*.py`, `init/kmeans.py`, and common `modules/*.py` (including `rqvae.py`, `quantize.py`, `sid_quality.py`, `step_checks.py`) with iter29 fixed-curvature behavior. These shared modules remain byte-for-byte/common unless a contract check exposes a necessary mechanical profile-neutral fix; do not make separate control/candidate module copies.
- **Modify** `curvature_config.py` to carry six explicit no-argument profile routes and absolute locked shared input paths; its `RQVAE_OUT_DIR` literal must pass the root short-iter path rule. Keep common deterministic/cache/GPU values fixed.
- **Modify** `curvature_RQ-VAE.py` to use shared dual-map input validation, profile-aware seed/path/map identity, collective strict warm-start, common descriptive SID reporting, and profile-specific launcher logs. Preserve the common training algorithm and default root entry as the fixed control-seed43 route.
- **Modify** `scripts/compute_closed_form_curvature.py` to add the control formula and a single explicit map-ID interface; retain candidate math/validation and explicit input-only behavior. Copy the common explicit-input `scripts/computed_behavior_branching.json`; do not regenerate it.
- **Create** one shared no-CLI profile runner, five Stage2 wrapper scripts for the non-default routes, and one Stage2 profile-specific gradient-check wrapper for each of the six labels; each wrapper binds one literal table entry. Modify `scripts/grad_check.py` into the shared checker. Keep `scripts/mvg_check.py` as one shared two-vector S08 checker with the S02 counterfactual signal.
- **Modify** `scripts/export_sids_for_stage3.py` as a common function using explicit per-profile output paths (not an arm-specific exporter); ensure the trainer and standalone helper share validation/export logic and do not read stale/sibling paths.
- **Create** one Stage3 wrapper runner and six thin no-argument per-label Stage3 wrappers under iter30 `scripts/`.
- **Modify** root `.gitignore` only with the scoped iter30 result exceptions and later transient exclusions above.
- **Keep unchanged** `stage3_T5Train/train_HG-Rec.py`, iter29 source/result trees, canonical S00–S05 iter30 artifacts, all registered S01 inputs/settings, and the ten-field S04 contract.

## Post-patch smoke design before S07/S08 (not performed in S06)

After Judge approval and the one canonical patch, use a disposable CPU-only smoke script (outside the repository; remove it after) that imports/checks the new shared code but never calls a trainer, torchrun, or Stage3 launcher. It must:

1. Recompute both mappings from the explicit saved record and assert the exact registered intermediates/vectors; assert that missing plural `raw_residual_medians`, a `residual_norm`-only payload, normalized-scale substitution, reordered/wrong input, non-finite value, unknown map ID, or altered candidate contract fails closed. Check the candidate ten-field object is unchanged and no control vector is added to it.
2. Resolve all six literal profiles and assert exact labels, matched seeds, one of the two exact map IDs/vectors, full `MECHANISM_NAME`/variant labels, required Stage2 and Stage3 short-root prefixes, unique Stage2/Stage3/SID/log paths, and Stage2 port 50200 / Stage3 port 50201. Confirm all profile entry files take no arguments and contain no environment-based profile selection. Check the static `curvature_config.py` `RQVAE_OUT_DIR` satisfies root `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/` convention.
3. Exercise output guards against temporary synthetic absent/empty/nonempty route trees; verify nonempty collision fails and no guard or cleanup deletes a file. Read/inventory the real six destinations but do not create them.
4. On CPU only, call the warm-start helper against the locked iter8 file after verifying its exact path/SHA and strict transfer count; then exercise temporary malformed payload, wrong digest, missing non-curvature key, unexpected key, and shape mismatch cases and assert each fails. This is a read-only checkpoint check, not model training or GPU work; do not bypass the locked real digest for a claimed PASS.
5. Compose (do not launch) each Stage3 wrapper's in-memory trainer settings and verify each points to its exact Stage2 JSON/seed/variant and unique short Stage3 roots, trainer source is not modified, and a missing/wrong exact JSON is rejected before invoking `_launch_via_torchrun`. Do not call the Stage3 evaluator or model.

These are throwaway implementation smoke checks, not permanent tests, preflight, MVG, Stage2 gradient checks, or authorization. After these CPU smokes, S07 must separately run the prescribed root preflight from the iter30 directory. S08 must separately perform its adjudicated common two-vector MVG. Each of the six GPU one-checkpoint/one-batch gradient checks remains immediately prior to its own actual Stage2 run under the later execution authorization. No smoke in this proposal starts Stage2/Stage3 or schedules GPU work.