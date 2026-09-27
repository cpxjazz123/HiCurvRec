# Iter30 S06 Canonical Implementation Plan

```text
STAGE_ID=S06_IMPLEMENTATION
ROUND=1
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
ITERATION=30
PARENT_STAGE=S05_ONE_FACTOR
PARENT_VERDICT=MERGE_AB
PARENT_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/one_factor_diff_iter30.md
SOURCE_PARENT=iter29 at ecd01e4712a1badd38a0338255f4b2ec7b030aff
VERDICT=MERGE_AB
STATUS=CANONICAL_PLAN_ONLY
SOURCE_PATCH_APPLIED=NO
EXECUTION_AUTHORIZATION=NO
```

## Decision and adjudication

**MERGE_AB.** Both candidates pass the scientific and protocol hard gates on the required two maps, explicit inputs, common training/reporting path, fail-closed warm-start, six independent run checks, Stage3 freeze, unique outputs, and restricted smoke scope. Neither routing design is safe as written against the literal root Stage2 entrypoint requirement: A proposes invoking six `run_stage2_*.py` files directly; B uses `curvature_RQ-VAE.py` directly for only its default seed-43 control and invokes five other wrapper names. The canonical resolution merges their profile-specific hardcoded wrappers with a fixed serial dispatcher in the root `curvature_RQ-VAE.py`. Thus the operator still starts Stage2 only with the root-required no-argument `python curvature_RQ-VAE.py` entrypoint, while each DDP job re-enters its own immutable, no-argument profile wrapper. No CLI argument, environment selector, shell-injected value, cwd selector, mutable shared profile file, or between-run source/config rewrite selects a profile.

A contributes the exact dual-map/common-loader requirements, typed explicit-input validation, DDP-safe wrapper identity, output/SID validation, six Stage3 resolver guards, and the bounded CPU-only smoke design. B contributes the especially explicit strict warm-start transfer/accounting contract and complete pre-launch inventory rules. The route dispatch resolution above is a judge integration required by root CLAUDE §3 and canonical S01 §193; it does not change the registered experiment or weaken either candidate's accepted constraints.

This plan is a design artifact only. It does not assert that future source files, profiles, checks, or outputs exist or pass.

## Authority, checked evidence, and registered invariants

Authority order: root `CLAUDE.md`; active curvature skill; canonical iter30 S00–S05 records and Judge decisions; then historical source. S00–S05 are adjudicated, with S05 a conditional prospective single-factor design PASS only. S03 supports historical method/value provenance at HIGH confidence for branching and MEDIUM for raw residuals; it does not establish checkpoint replay, historic byte identity, current recalculation, or historical runtime consumption.

Primary evidence reviewed for this decision:

- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py`, `curvature_config.py`, `modules/quantize.py`, `modules/rqvae.py`, `modules/sid_quality.py`, and relevant `scripts/` including `compute_closed_form_curvature.py` and `run_stage3_iter29.py`.
- Root `CLAUDE.md` §1 (no CLI/env parameter selection), §2 (SID metrics descriptive-only; no HR@50 gate or early stop), §3 (the required `curvature_RQ-VAE.py` no-argument Stage2 command and built-in four-rank launcher), §6 (one-checkpoint/one-batch gradients immediately before each actual Stage2 run), and §§0, 10–11, 13 (short iter30 result roots, output placement, ignore and deliverable rules).
- Canonical `logs/protocol_manifest_iter30.md`, `logs/hypothesis_iter30.md`, `logs/mechanism_manifest_iter30.md`, `logs/mechanism_contract_iter30.json`, `logs/one_factor_diff_iter30.md`, and the S00–S05 Judge records.
- Root `.gitignore`: global `**/*.pth`, `**/*.pt`, `**/*.npy`, `**/ckpt*/`, transient Stage3 launcher and result patterns, plus the later iter-specific exception convention.

Locked mappings and exact `[L0,L1,L2]` values (do not recalculate or replace the registrations):

```text
branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]

iter26 control:
  s = log1p(B) / log1p(m_raw / min(m_raw))
  z = (s - mean(s)) / (std_population(s) + 1e-12)
  c26 = clip(0.5 * exp(0.2*z), 0.05, 1.5)
  c26 = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]

iter29 candidate (the S04 contract's sole final vector):
  x = B/(B+2.0); y = m_raw/(m_raw+0.1); u=(x+y)/2
  c29 = 0.05 + 1.45*u
  c29 = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

Every future consumer requires the literal plural input key `raw_residual_medians`; it must fail closed if absent or discrepant. Never use `residual_norm`, normalized scales `[0.001,0.932889,1.0]`, another checkpoint, or inferred/recalculated substitutes. Keep `logs/mechanism_contract_iter30.json` exactly the canonical ten-field S04 candidate-only contract; the iter26 vector is a separate explicit profile/MVG assertion, not another contract field/object. Both vectors are fixed, pre-training, non-trainable and time-invariant.

Only the map/vector changes scientifically within a matched pair. Keep seed pairs 43, 44, 45; same seed for each pair's Stage2 and Stage3 runs. Preserve the common iter29 Stage2 model, data, optimizer, losses, geometry, Sinkhorn and its inherited effective epsilon, M2/M3, runtime, deterministic/cache/GPU settings, SID cadence/export, and common reporting path. In particular retain 768 input; hidden `[512,256,128]`; embedding 32; three 256-entry quantizers; commitment 1.0; batch 640/GPU, four ranks/global 2560; 100000 global steps/checkpoint cadence 10000; AdamW `1e-3`/`1e-4`; clip 1.0; configured `sk_eps=0.05` / 3 iterations; behavior weight 0.20 / temperature 0.07; zero curvature regularization. Stage2 quality metrics remain descriptive only. Keep the Stage3 trainer byte-for-byte unchanged at S01 SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`, with its existing configuration and behavior: max 150 epochs, `NO_EVAL=True` / train-loss patience 10, `SKIP_TEST=False`, top K `[5,10]`, beam 20, expected `n_eval=57439`, batch 4096, inference batch 1024, four ranks and port 50201. Preserve the historical variant-metadata caveat; do not patch the trainer.

## Canonical profile/entrypoint topology

The six and only six immutable profile labels are:

```text
iter26_mapping_seed43, iter29_mapping_seed43,
iter26_mapping_seed44, iter29_mapping_seed44,
iter26_mapping_seed45, iter29_mapping_seed45
```

Use one six-record literal profile table and six tiny no-argument route wrappers (one literal label per wrapper). The root `curvature_RQ-VAE.py` is the required operator entrypoint and a fixed serial dispatcher: it traverses the six literal wrapper paths in a fixed predeclared order, never accepts a run selector, and directs each wrapper's outer stdout/stderr to that label's unique source `logs/train_run_<label>.log`. It may keep its own required command stdout as a dispatcher log, but that shared log is not a substitute for the six per-run logs. Each wrapper resolves only its baked-in profile and starts the common trainer through the existing four-rank launcher on port 50200. Set the launcher's script path to that same wrapper file; all torchrun workers therefore re-enter the identical file with the same literal profile. The wrapper's `RANK` branch calls the shared trainer for its fixed profile; it must not infer a profile from rank, environment, cwd, shell, or a mutable cross-run setting. No profile path rewrites `curvature_config.py` or a shared selector file. The six-run dispatcher executes profiles serially, not concurrently.

Keep a static `curvature_config.py:RQVAE_OUT_DIR` default visibly under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/` to satisfy the root config-path review; it may name the first literal route. Validate all six route records against that short prefix before dispatch. The common trainer accepts the resolved immutable profile record explicitly and uses the same training/model/report/export path for both maps.

For each label `<label>` and root `<repo>/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/`, set absolute paths:

```text
RQVAE_OUT_DIR=<root>/out/rqvae/instruments/
RQVAE_CKPT_PATH=<RQVAE_OUT_DIR>/rqvae_best.pth
RAW_SIDS_NPY=<RQVAE_OUT_DIR>/sids_raw.npy
SIDS_NPY=<root>/dataset/Instruments/sids_for_hgrec.npy
ITEM_SIDS_JSON=<root>/item_sids.json
MECHANISM_NAME=iter30_<iter26_mapping|iter29_mapping>_seed<43|44|45>
```

All generated `.pth`, `.pt`, `.npy`, and `item_sids.json` products remain under that profile's prescribed external Stage2 result root, never the iter30 source subtree. Keep unique `logs/train_run_<label>.log`, `logs/train_migrated_<label>.log`, and `logs/run_identity_<label>.json` beneath iter30 source `logs/`. The controller must not create profile products in advance or overwrite any route.

## File-by-file implementation actions (after separate authorization to apply this plan)

1. **Create the iter30 common source tree from iter29.** Copy the required `curvature_RQ-VAE.py`, `curvature_config.py`, trainer-imported `data/`, `init/`, `modules/`, and required helper `scripts/` sources; copy `scripts/computed_behavior_branching.json` unchanged as input evidence. Preserve existing iter30 S00–S06 audit files. Do not copy historical logs, caches, checkpoints, SID arrays, or result products. Stage3 trainer is not copied or edited.
2. **`curvature_config.py`.** Preserve locked shared settings and absolute Stage1/Stage0 inputs; add the closed six-profile table with only label, map ID/vector identity, seed, full mechanism label, and unique absolute Stage2/Stage3/log paths. Keep the static default `RQVAE_OUT_DIR` root-compliant. No CLI/env/cwd profile discovery or mutable selector. A common validator rejects any value not exactly matching one of the six registered records.
3. **`curvature_RQ-VAE.py`.** Replace the direct single-profile `__main__` route with the fixed no-argument serial dispatcher described above while preserving the exact prescribed command entrypoint and internal hardcoded four-rank mechanism. It starts only the six fixed wrappers, each serially; for each route it completes that route's input/warm-start/path/inventory gates and invokes that label's distinct `grad_check_<label>.py` immediately before starting its wrapper/Stage2 DDP process. It records per-profile hashes and destination inventory, captures each wrapper's stdout/stderr to its own outer log, and stops dispatch on the first failed gate/child exit without relabeling or continuing a failed run. Refactor the training implementation into a shared profile consumer callable from the wrapper's rank branch. The existing worker/trainer implementation, model, loss, optimizer, schedule, sampler, SID cadence, and export remain one common path.
4. **Six Stage2 wrapper files under `scripts/`.** Each contains exactly one hardcoded label and no argument/environment selector. Each resolves its literal table entry, performs only profile-neutral/common checks, and calls the common trainer; in DDP, the launcher re-enters this same wrapper, retaining that same profile on every rank. All six ultimately invoke the same model/training/reporting implementation.
Use these exact route files (each literal route contains only its own profile label): Stage2 `scripts/run_stage2_iter26_mapping_seed43.py`, `scripts/run_stage2_iter29_mapping_seed43.py`, `scripts/run_stage2_iter26_mapping_seed44.py`, `scripts/run_stage2_iter29_mapping_seed44.py`, `scripts/run_stage2_iter26_mapping_seed45.py`, `scripts/run_stage2_iter29_mapping_seed45.py`; shared profile utilities in `scripts/profile_routes.py`. The root dispatcher launches only these six files in the registered fixed serial order; torchrun re-enters the selected same file.
5. **`scripts/compute_closed_form_curvature.py` plus immutable JSON input.** Convert the helper to a pure, shared dual-map calculation API selected only by the immutable profile's `mapping_id`; it must not rewrite the registered JSON or write per-map caches. Validate `branching` and plural `raw_residual_medians`, type/order/length/finite/range/exact registered values, map ID, all formula intermediates and exact registered output (comparison tolerance `1e-9`). Apply the specified iter26 and iter29 equations and vectors above. Reject missing/altered inputs, legacy aliases, wrong order, unknown map, nonfinite data, or wrong result. Do not add a second contract or add control values to S04 JSON.
6. **Fixed-curvature consumer files `modules/quantize.py`, `modules/rqvae.py`, and trainer.** Keep one fixed-buffer implementation for both profiles, no curvature parameter/schedule/regularization, unchanged consumers and `get_c()` behavior. Confirm the copied root trainer and module files still satisfy the actual preflight expectations, including visible `RQVAE_OUT_DIR`, fixed buffer, no trainable curvature, zero curvature regularization and invariant `get_c()`; do not treat the plan as that later preflight.
7. **Common fail-closed warm-start helper and trainer callsite.** Replace the iter29 exists/warn/random-init fallback with one shared loader used by all six profiles and each later profile-specific gradient check. Require the exact absolute S01 iter8 checkpoint and SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` immediately before load; reject path, read/hash, deserialization, payload, or state incompatibility. Accept only mapping payload with expected `model` state and tensor entries. Compare keys/shapes/dtypes explicitly; permit only S01-approved legacy curvature names `layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c` when present. Every target tensor except exactly the current three `layers.{i}._fixed_c` buffers must have one compatible source tensor; reject unexplained missing/unexpected/mismatched/nonfinite entries. After load, require missing keys equal exactly those three fixed buffers, no unexpected keys, positive compatible transfer count equal to expected non-curvature transfer set, and selected fixed buffers preserved. Log digest, count/key set, skipped legacy names, and expected missing keys. DDP must make rank-zero load success/failure collective before training; no rank proceeds or waits indefinitely after a failed load. No random-init fallback.
Implement the shared loader in `modules/warm_start.py`; both trainer and checker import it rather than maintaining separate transfer policies.
8. **Common SID quality caller/module and export.** Route both mappings through the iter29 in-process SID-derived descriptive reporter. Remove the active iter26 HR@50/embedding-neighbor subprocess and any temporary metrics-file, report-based admission, quality gate or early-stop path. Retain descriptive-only metrics and `should_early_stop() == (False, "")`. Keep one shared raw three-token generation cadence and one shared four-token collision-extension/export path. Validate the selected profile's exact raw SID source, output locations, item count, ranges, uniqueness/collision extension, dense JSON keys/four-token width, and NPY read-back parity. Export failures are integrity failures, not quality gates; no sibling path fallback.
9. **`scripts/grad_check.py` and six no-argument per-profile check routes.** Provide one common checker and six distinct hardcoded route files. For each actual future Stage2 launch, run that profile's check immediately beforehand against that exact seed, map/vector, inputs, warm start and batch. Each check does the real one-batch forward and `total_loss.backward()`, requires finite `total_loss` with `requires_grad` and non-null `grad_fn`, finite nonzero intended trainable gradients and differentiable applicable loss paths, and checks `_last_*` detach hazards. Fixed curvature must be fixed/non-trainable, excluded from optimizer groups and receive no gradient. Log profile, identities, digest/transfer count, batch, per-loss/gradient outcomes and source/config/profile hashes. These six checks are explicitly deferred; neither one common check nor S08 replaces them.
Name the six no-argument check routes exactly `scripts/grad_check_iter26_mapping_seed43.py`, `scripts/grad_check_iter29_mapping_seed43.py`, `scripts/grad_check_iter26_mapping_seed44.py`, `scripts/grad_check_iter29_mapping_seed44.py`, `scripts/grad_check_iter26_mapping_seed45.py`, and `scripts/grad_check_iter29_mapping_seed45.py`.
10. **`scripts/mvg_check.py`.** Implement the S02 counterfactual using the same checkpoint, actual batch, encoded L0 input, model/codebook state, eval mode, device and dtype. Run the unchanged L0 `Quantize.forward` path for both maps, capturing the actual per-example-by-code Poincare distance tensor after any existing flattening fallback and immediately before `_center_distance_for_constraint`, before the unchanged centering/Sinkhorn path. Define `M_distance = max_ij(abs(D29[i,j]-D26[i,j]))`. Repeat each unchanged-arm computation with identical state/batch/device/dtype; define observed repeat tolerance as the maximum within-arm max-absolute distance repeat difference. Direct activation requires `M_distance` to exceed that tolerance. Do not use loss/assignment-only proxy or an arbitrary `1e-6` threshold; that source literal is a different fallback statistic. This is deferred to S08 and must not change production training behavior.
11. **Six Stage3 no-argument wrappers and one shared wrapper helper under iter30 `scripts/`.** Each wrapper hardcodes one matching label, imports the root Stage3 trainer by the existing wrapper pattern, and changes only the selected absolute SID input, `RQVAE_VARIANT`, pair seed, and unique output/log routes. Set `_LAUNCHER["script"]` to the same wrapper for DDP worker re-entry. Before invoking its launcher, fail closed unless that exact Stage2 `item_sids.json` is a regular file inside its paired Stage2 short root and `_resolve_code_file` resolves to precisely that file; no stale fallback. Preserve the trainer SHA and every locked Stage3 setting, port 50201 serially, and distinct result/log/checkpoint/launcher paths under `results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/`. Do not edit `train_HG-Rec.py` or Stage3 metadata behavior.
Name the shared Stage3 wrapper helper `scripts/stage3_profile_runner.py` and the six literal wrappers `scripts/run_stage3_iter26_mapping_seed43.py`, `scripts/run_stage3_iter29_mapping_seed43.py`, `scripts/run_stage3_iter26_mapping_seed44.py`, `scripts/run_stage3_iter29_mapping_seed44.py`, `scripts/run_stage3_iter26_mapping_seed45.py`, and `scripts/run_stage3_iter29_mapping_seed45.py`.
12. **Root `.gitignore`.** After the global result binary/checkpoint ignores and consistent with the existing iter17–29 exception block, add only iter30 Stage2 and Stage3 deliverable-root exceptions sufficient to retain required result `.pth`, `.pt`, `.npy`, JSON, logs and Stage3 checkpoints. Reapply narrower exclusions after those exceptions for `__pycache__`, shared `dataset/Instruments/item_emb.npy`, `_ddp_sync`, tensorboard, wait logs, and transient `_stage3_run*.log`; keep required `_stage3_launcher.log` included. Do not unignore other iterations or source-tree products. Later execution/closure must confirm the exact paths with `git check-ignore -v` before committing.

## Pre-run route and collision invariants

Before every profile launch, record hashes for the actual wrapper, common trainer, config, map/input helper, relevant modules and locked consumed inputs. Rehash and confirm the S01 absolute paths and digests for Stage1 `sentence_t5.npy` (SHA256 `6490c71753952b2d8788bb7e4b52cff6a5b0b59819a2cfbd737bee7bc07a77fb`), Stage1 `item_ids.json` (`3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`), Stage0 Stage2-consumed `train.parquet` (`80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`), and locked warm start above; verify resolved actual consumers, not declarations alone. Do not call Stage0 valid/test/items Stage2 runtime inputs. For Stage3, rehash actual `train.parquet`, `test.parquet`, unchanged trainer, and exact selected JSON; do not claim valid.parquet is consumed with `NO_EVAL=True`.

Before each route, inventory the complete Stage2 root, external SID NPY/JSON, source outer/internal log and identity destinations. Require absent/empty destinations; reject aliases/collisions/nonempty routes and never overwrite or clean another profile. Existing Step0 cleanup may touch only its own profile `RQVAE_OUT_DIR` and its enumerated files, only after route checks; it does not authorize cleanup of external SID files, logs, sibling roots or Stage3. Stage3 roots receive the same absent/empty guard. The root dispatcher and child routes are serial on ports 50200/50201. Do not create placeholder output directories during S06 implementation or smoke.

## Narrow non-GPU smoke boundary (after patch, before S07; not performed now)

Only after the separately authorized canonical patch is applied, a disposable CPU-only/no-GPU smoke may exercise the pure mapping/input validator, six literal profile resolution and path uniqueness, exact Stage2 static config path, and collision guards against temporary synthetic absent/empty/nonempty directories. It may inspect Stage3 wrapper constants and verify the trainer hash without invoking the trainer. Do not launch any trainer, torchrun, Stage3 helper, preflight, MVG, model forward/backward, GPU, checkpoint-transfer check, or actual route. Do not read/write result destinations or modify the immutable input record. Remove any throwaway smoke script after use. These smokes are not permanent tests and cannot replace S07/S08 or any of the six immediate pre-run gradient checks. **No smoke, plan approval, or source patch itself authorizes Stage2/Stage3 execution.**

## Residual risks and stop conditions

The historical raw-residual provenance remains medium-confidence method/value evidence, not checkpoint replay or historic byte identity; preserve that limit and use only the explicit registered input. The S04 contract is candidate-only, so S07/S08 must separately verify the control vector without altering the contract. The root dispatcher/profile-wrapper/DDP design must be implemented and statically shown to preserve the same literal selection in every worker while the operator entry remains `curvature_RQ-VAE.py`; any need for CLI/env/cwd/shared-file selection is a blocker, not a workaround. The exact distance capture must observe the true L0 distance tensor at the S02 callsite and compare against measured same-arm repeats, not a proxy. Strict warm-start tensor compatibility and collective failure need source-level S07 review. Stage3 resolver and `.gitignore` exceptions must be proven against actual routes before any future launch/closure.

If the root entrypoint plus immutable per-profile DDP routing cannot be implemented exactly as specified without prohibited selectors or source mutation between runs, stop and return to implementation-design deliberation; do not launch. If any registered input/map/contract/settings, transfer, path, or profile check fails, fail closed and keep execution blocked; do not retune the maps or skip a profile.

## Deferred actions and next gate

Deferred explicitly: applying the source patch; S07 static/preflight adjudication; S08 MVG adjudication; all six immediately-prior per-run gradient checks; S09 launch adjudication; any Stage2, Stage3, GPU, or result-producing action. S06 approval is **not execution authorization**. After this plan is accepted, the next concrete action is to apply the canonical source patch once, then perform only the bounded non-GPU smoke above and advance to independently adjudicated S07. S08, each of the six run-specific gradient checks, S09 and all launches remain separately gated as specified.
