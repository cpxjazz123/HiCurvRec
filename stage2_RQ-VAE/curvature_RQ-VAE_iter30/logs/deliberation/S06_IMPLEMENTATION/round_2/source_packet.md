STAGE_ID=S06_IMPLEMENTATION
ROUND=2
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
ITERATION=30
PARENT_STAGE=S07_PREFLIGHT
PARENT_VERDICT=ACCEPT_B
PARENT_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S07_PREFLIGHT/round_1/judge.md
REOPEN_REASON=S07 round 1 confirmed a no-argument Stage2 DDP wrapper contract violation in the applied launcher.

## Objective
Independently design the smallest implementation repair that makes every Stage2 torchrun worker re-enter its literal profile wrapper with no unintended script arguments while preserving the approved root no-argument operator command, route identity, and unbuffered worker output. Provide a complete patch plan and bounded verification plan. Do not change the registered experiment or apply any source edit.

## Canonical evidence
- `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/implementation_plan_iter30.md` and `logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md`: canonical six-profile fixed-order dispatcher and literal no-argument Stage2 wrappers; the launcher's script path must be the matching wrapper for DDP worker re-entry.
- `logs/deliberation/S07_PREFLIGHT/round_1/judge.md`: accepted Agent B's source-backed audit; current S07 dispatch gate is FAIL/BLOCKED, and preflight/MVG/gradients/Stage2/Stage3 have not been authorized.
- `curvature_RQ-VAE.py:749-763`: current command list places `"-u"` after `_LAUNCHER["script"]`.
- Locked environment `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/lib/python3.9/site-packages/torch/distributed/run.py:630-635`: `training_script_args` is parsed as `nargs=REMAINDER`.
- The same PyTorch source at `707-718` defaults `use_env` to true; at `831-856`, Python-mode torchrun appends interpreter `-u` before the script and then extends command arguments with `training_script_args`; at `859-869`, run-path mode assigns remaining arguments into worker `sys.argv`.
- Root `CLAUDE.md` requires project scripts to use no CLI arguments and names `python curvature_RQ-VAE.py` as the Stage2 operator entrypoint. The approved experiment, map vectors, six labels/order, path roots, Stage3 trainer, and all training settings remain immutable.

## Hard constraints
- Preserve the canonical S06 root dispatcher and all six literal wrapper paths/profile bindings.
- Preserve exactly the existing torchrun executable, four-rank flags, port, serial dispatch, per-profile logging, and error handling unless the Judge finds a directly necessary command-only adjustment.
- Do not pass an accidental CLI argument to the wrapper. Preserve unbuffered Python worker execution using the behavior supported by the locked torchrun implementation.
- No CLI/environment/cwd/shared-file profile selector; no run-to-run source mutation.
- Do not change FCCR-1, either formula/vector, S04 contract, seeds, Stage2/Stage3 protocol, data, outputs, or scientific mechanisms.
- No production source edits, preflight, torchrun invocation, trainer/model/checkpoint import/load, gradients, MVG, GPU, Stage2/Stage3, or result creation during deliberation.

## Required candidate artifact
Each candidate independently reports:
1. exact source-backed diagnosis and worker argument construction;
2. a minimal, complete patch plan naming the exact file/statement and final behavior;
3. why the repair preserves the canonical no-argument root entrypoint, literal DDP wrapper identity, and unbuffered output;
4. a bounded post-patch CPU/static smoke plan within the original S06 smoke boundary, with explicit forbidden actions;
5. risks, assumptions, self-rejection criteria, and one concrete autonomous next action.

## Independence and destination
Agent A and Agent B receive this identical packet, inspect primary evidence independently, and do not communicate or read each other's draft. Candidate artifacts only:
- `logs/deliberation/S06_IMPLEMENTATION/round_2/agent_a.md`
- `logs/deliberation/S06_IMPLEMENTATION/round_2/agent_b.md`
Judge C must adjudicate against primary sources before any patch is applied. The orchestrator applies one canonical patch once, reruns only the bounded smoke, then starts a fresh S07 round-2 audit. No downstream stage may consume this packet as execution authorization.
