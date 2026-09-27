STAGE_ID=S07_PREFLIGHT
ROUND=2
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
ITERATION=30
PARENT_STAGE=S06_IMPLEMENTATION
PARENT_VERDICT=ACCEPT_A
PARENT_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S06_IMPLEMENTATION/round_2/judge.md; stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/implementation_smoke_iter30.log
REOPEN_REASON=S07 round 1 accepted the correct audit finding that the applied dispatcher passed a positional -u to literal wrappers; the S06 round-2 Judge-authorized command-only correction is now applied and its bounded smoke passed.

## Objective
Independently re-audit the current iter30 source against canonical S00-S06 and FCCR-1. Determine whether the applied command-only repair closes the no-argument DDP wrapper blocker and whether the complete static implementation satisfies the original S07 criteria. This is a source audit, not a preflight execution result or authorization for MVG, gradients, Stage2, Stage3, or GPU work.

## Canonical evidence
Use canonical artifacts and primary source only. Do not use round-1 A/B drafts as active instructions; round-1 `judge.md` is the adjudicated record.
- `logs/source_snapshot_iter30.md`
- `logs/protocol_manifest_iter30.md`
- `logs/hypothesis_iter30.md`
- `logs/mechanism_manifest_iter30.md`
- `logs/mechanism_contract_iter30.json` (exact candidate-only ten-field S04 contract)
- `logs/one_factor_diff_iter30.md`
- `logs/implementation_plan_iter30.md`
- `logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md`
- `logs/deliberation/S06_IMPLEMENTATION/round_2/source_packet.md`, `agent_a.md`, `judge.md`
- `logs/implementation_smoke_iter30.log`
- `logs/deliberation/S07_PREFLIGHT/round_1/judge.md`
- Primary implementation: root `curvature_RQ-VAE.py`, `curvature_config.py`, `modules/quantize.py`, `modules/rqvae.py`, `modules/warm_start.py`, `modules/step_checks.py`, `scripts/compute_closed_form_curvature.py`, `scripts/computed_behavior_branching.json`, `scripts/profile_routes.py`, all six Stage2/check/Stage3 wrappers, `scripts/stage3_profile_runner.py`, `scripts/export_sids_for_stage3.py`, and `scripts/mvg_check.py`.
- Required skill preflight implementation: `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`.

## Locked scientific and execution invariants
- FCCR-1 only: both maps produce fixed, pre-training, non-trainable and time-invariant curvature; no cyclic schedule effect, trainable curvature, or nonzero curvature regularization.
- Explicit `[L0,L1,L2]` branching `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and plural `raw_residual_medians` `[1.0, 0.10941, 0.09331]`; reject missing/legacy alias inputs. Iter26 vector `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`; iter29 vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. Keep exact candidate-only ten-field S04 JSON unchanged.
- Six profiles in fixed order: iter26/iter29 pairs for seeds 43, 44, 45. Only mapping differs within each pair; pair Stage2/Stage3 seeds match.
- Operator entry remains no-argument root `curvature_RQ-VAE.py`; root dispatches all six serially through literal Stage2 wrapper paths, each profile's distinct gradient-check route immediately precedes its Stage2 wrapper, and child failure stops dispatch. DDP workers re-enter their same literal wrapper; no CLI, environment, cwd, shared-file or mutable cross-run selector.
- Stage2 outputs only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/`; Stage3 outputs only under matching `results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/`. Source subtree contains no model/array products. Unique run logs/identity and absent-or-empty destination guards remain mandatory.
- Exact iter8 warm-start path/SHA and strict transfer helper; no fallback/random initialization. Six actual per-run gradient checks remain deferred and required immediately before each full Stage2 run.
- S02 MVG must later capture actual L0 pre-centering Poincare distance tensor and compare measured within-arm repeat tolerance; it has not run.
- Stage3 trainer SHA remains `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; exact matching Stage2 JSON, unchanged protocol, and route guards remain required.

## Round-1 blocker and authorized correction
- Canonical S07 round-1 Judge confirms the prior Stage2 `_launch_via_torchrun()` appended `-u` after `_LAUNCHER["script"]`; locked PyTorch parses that as `training_script_args` (`nargs=REMAINDER`), not an interpreter flag.
- S06 round-2 Judge authorizes removal of only that trailing list element and removal/replacement of the inaccurate adjacent comment. Current `curvature_RQ-VAE.py` states that torchrun inserts Python `-u` before the wrapper and ends the command list at `_LAUNCHER["script"]`.
- The recorded round-2 smoke is limited to AST/compile inspection of the current launcher and locked PyTorch argument-construction source. It did not invoke torchrun or execute a worker. Independently verify the current source; do not rely on the smoke log alone.

## Work restrictions
- Read-only audit. Do not edit production source or canonical S00-S06 artifacts.
- Do not run `preflight_contract.py`; the orchestrator may run it exactly once only after Judge C accepts a clean S07 static audit.
- Do not import/invoke trainers or models, load checkpoints, execute gradients/MVG, invoke torchrun, launch Stage2/Stage3, access GPUs, create results, or create route destinations.
- Do not read/message the other candidate. Write only the assigned candidate artifact.

## Required candidate conclusion
For each hard gate, give source-backed PASS/FAIL and exact file/line evidence: FCCR-1/preflight compatibility; both maps and explicit input semantics; six-profile static root dispatch and literal DDP wrapper identity including no script arguments; strict warm-start and per-run gradient placement; exact result/SID/export paths and collision guards; unchanged Stage3 trainer/protocol and exact resolver; absence of a second mechanism; preflight implementation compatibility versus actual preflight result not yet run; remaining S08/S09 work. State assumptions, self-rejection conditions, and one concrete autonomous next action. Do not claim `MECHANISM_CONTRACT_PASS` or authorize downstream work without the Judge decision and actual preflight result.

## Independence and destination
Agent A and Agent B receive this identical packet and complete independent audits in parallel. Required candidate paths:
- `logs/deliberation/S07_PREFLIGHT/round_2/agent_a.md`
- `logs/deliberation/S07_PREFLIGHT/round_2/agent_b.md`
They must not read, quote, summarize, or contact the other candidate before Judge C adjudicates.
