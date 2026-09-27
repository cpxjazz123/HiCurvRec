# Iter31 Stage2 execution record

`STAGE_ID=S09_STAGE2_EXECUTION`
`LAUNCH_AUTHORIZATION=ONE_RUN_CONDITIONAL_AFTER_ALL_PRELAUNCH_CHECKS_PASS`
`PRELAUNCH_RECORD=logs/stage2_preflight_iter31.md`
`PRELAUNCH_CONDITIONS_PASS=YES`
`SUPERVISOR=hub process Iter31Stage2 (persistent, restart=no)`
`SUPERVISOR_PID=2310974`
`TRAINING_CHILD_PID=2310975`
`SUPERVISOR_RESTARTS=0`
`CAPTURED_AT=2026-09-27T19:08:14+10:00`

## Launch

`CWD=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`
`COMMAND=nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`

The command was launched once through the process supervisor with no arguments, environment overrides, alternate interpreter, or port/worker changes. The supervisor waits for the same child process, so its exit status remains observable; the required outer shell redirection and inner trainer log are unchanged.

## Startup evidence

At capture, the supervised child was still running and the inner log reported:

- Four-rank DDP (`world_size=4`), seed 42, per-rank batch 640 / global batch 2,560, target 100,000 global steps, checkpoint interval 10,000, compile disabled.
- `Step0` cleaned zero stale artifacts from the Iter31 results-root `RQVAE_OUT_DIR`.
- Loader opened the pinned Stage1 embedding and reported shape `(24587, 768)`, 339,519 train transitions, and 24,474/24,587 active sources.
- FCCR-1 values `[1.366095, 0.734783, 0.643996]`, mechanism `iter31_hra_step6_common_reference`; step-0 invariant check passed within `1e-6`.
- Exact Iter8 checkpoint warm start loaded 11 tensors; no missing-checkpoint/random-init warning. Closed-form curvature buffers were preserved.
- First forward produced `(1280, 3)` semantic IDs; first loss and backward completed; 11 parameters had gradients; first optimizer step completed.
- First progress line: global step 320/100000, on `cuda:0`.

`logs/train_migrated.log` contains the trainer/torchrun output. `logs/train_run.log` exists as the required outer redirection target and was empty at this capture because the launcher writes worker output to the inner log. Preserve both files.

## Status at capture

`RUN_STATUS=RUNNING`
`GLOBAL_STEP_REPORTED=320/100000`
`COMPLETION=NO`
`STAGE3_AUTHORIZATION=NO`

No second run, MVG, test suite, or Stage3 process was started. Continue this same supervised run to its exit; do not retry or stop based on descriptive SID/occupancy/entropy/collision metrics.
