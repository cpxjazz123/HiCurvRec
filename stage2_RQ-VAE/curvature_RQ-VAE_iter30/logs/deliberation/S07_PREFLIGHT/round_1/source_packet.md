STAGE_ID=S07_PREFLIGHT
ROUND=1
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
ITERATION=30
PARENT_STAGE=S06_IMPLEMENTATION
PARENT_VERDICT=MERGE_AB
PARENT_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/implementation_plan_iter30.md; stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md

## Objective
Independently audit the materialized iter30 source against the canonical S00-S06 artifacts and FCCR-1 contract. Determine whether the implementation preserves both registered maps, fixed/non-trainable curvature, the approved six-route root-dispatch topology, fail-closed warm-start/input/output handling, and exact unchanged Stage3 wiring. Identify concrete source-backed blockers before running the skill-level contract preflight. A PASS is a static source audit, not authorization for MVG, per-run gradient checks, Stage2, Stage3, or GPU work.

## Canonical evidence packet
Use only canonical artifacts, this packet, and primary source files; do not use rejected candidate drafts as instructions.

- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/hypothesis_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/mechanism_manifest_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/mechanism_contract_iter30.json` (candidate-only canonical ten-field S04 contract; do not add the control map)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/one_factor_diff_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/implementation_plan_iter30.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/implementation_smoke_iter30.log`
- Primary implementation: `curvature_RQ-VAE.py`, `curvature_config.py`, `modules/quantize.py`, `modules/rqvae.py`, `modules/warm_start.py`, `modules/step_checks.py`, `scripts/compute_closed_form_curvature.py`, `scripts/computed_behavior_branching.json`, `scripts/profile_routes.py`, `scripts/grad_check.py`, all six `run_stage2_*`, `grad_check_*`, and `run_stage3_*` wrappers, `scripts/stage3_profile_runner.py`, `scripts/export_sids_for_stage3.py`, and `scripts/mvg_check.py`.
- Required static preflight implementation: `~/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`.
- Root rules are already provided in the session. Relevant rules: no CLI/env profile selector; root no-argument Stage2 command; fixed short iter30 results roots; no Stage2 quality gates; unchanged Stage3 trainer; six complete matched profiles; S06 smoke passed without executing trainers.

## Locked scientific/protocol invariants
- FCCR-1 only: both maps produce fixed, pre-training, non-trainable, time-invariant curvature. No cyclic schedule effect, curvature optimizer parameter, or nonzero curvature regularization.
- Explicit `[L0,L1,L2]` branching `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and plural `raw_residual_medians` `[1.0, 0.10941, 0.09331]`; reject missing/legacy alias inputs. Iter26 vector `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`; iter29 vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Keep the exact candidate-only S04 JSON unchanged.
- Six profiles in fixed order: iter26/iter29 pairs for seeds 43, 44, 45. Only mapping differs within each matched pair; pair Stage2 and Stage3 seed equal.
- Operator entry remains no-argument root `curvature_RQ-VAE.py`; it must dispatch all six serially through their literal Stage2 wrapper paths, run each profile's distinct gradient-check wrapper immediately before its Stage2 torchrun wrapper, and fail closed on a child failure. DDP workers re-enter their same literal wrapper; no CLI, environment, cwd, shared-file, or mutable cross-run selector.
- Stage2 products only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/`; Stage3 products only under matching `results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/`. Source subtree contains no model/array products. Each route has unique logs/identity and absent-or-empty destination guards.
- Exact iter8 warm-start path/SHA and strict transfer helper; no fallback/random initialization. Actual per-run gradient checks are still deferred and required immediately before each full Stage2 run.
- S02 MVG must later capture the actual L0 pre-centering Poincare-distance tensor and compare measured within-arm repeat tolerance; it has not run.
- Stage3 trainer file must remain byte-identical at SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; each wrapper routes exact matching Stage2 JSON and only locked variant/seed/output fields, with no fallback resolver; Stage3 protocol unchanged.

## Work restrictions
- Read-only source audit. Do not edit production source, canonical S00-S06 artifacts, or another candidate's draft.
- Do not run `preflight_contract.py`; the orchestrator runs it once only after Judge C accepts the static audit.
- Do not import/invoke either trainer, inspect/load actual checkpoints, run model/gradient checks, preflight, MVG, torchrun, Stage2/Stage3, or any GPU/result-producing action.
- Write only the assigned candidate artifact path. Do not read or message the other candidate.

## Required candidate conclusion
Give a source-backed PASS/FAIL for each hard gate: contract/preflight compatibility; exact maps/input semantics; static and DDP profile identity/serial root dispatch; strict warm-start and per-run gate placement; exact result/SID/export paths and collision guards; unchanged Stage3 protocol and resolver; material unexplained second mechanism; remaining checks deferred to S08/S09. Quote file and line evidence. State any uncertainty explicitly, identify self-rejection conditions, and recommend the concrete next authorized action without asking for user input.
