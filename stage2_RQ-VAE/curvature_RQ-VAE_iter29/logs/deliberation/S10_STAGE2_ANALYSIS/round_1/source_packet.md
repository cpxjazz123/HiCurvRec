# S10_STAGE2_ANALYSIS — canonical source packet

STAGE_ID=S10_STAGE2_ANALYSIS
ITER=29
ROUND=1

## Objective

Independently analyze the same completed Stage2 run and its exported SID artifacts. Establish contract/execution validity, independently recompute descriptive SID/geometry evidence from the canonical outputs, compare it with the training log using the implementation's metric definitions, state interpretation limits, and recommend the next authorized pipeline action. Do not use any Stage2 proxy as a hard quality gate; valid non-aborted candidates proceed to Stage3 under FCCR-1. Do not launch Stage3 in S10.

## Canonical primary evidence

- `logs/stage2_execution_plan_iter29.md` and `logs/deliberation/S09_STAGE2_EXECUTION/round_1/judge.md`: approved single-run command, input/output paths, no-retry and Stage3 sequencing.
- `logs/stage2_preflight_iter29.log`: prelaunch path/empty-output audit and four input hashes; this audit matched the canonical S01 hashes.
- `logs/stage2_output_integrity_iter29.log`: supervised process exit, observed readiness timeout, full-step completion, output hashes, independent artifact-integrity smoke result, and status.
- `logs/train_migrated.log`: worker DDP/startup evidence, fixed-curvature snapshots, final 100000-step checkpoint/SID/export markers, and logged descriptive metrics.
- `logs/deliberation/S08_MVG/round_2/judge.md` and `logs/mvg_check_iter29.log`: adjudicated fixed-curvature MVG evidence and its explicit limits.
- `modules/sid_quality.py:90-144`: actual Gini, unique SID, per-layer occupancy, L0/L1 pair, and conditional-entropy metric definitions.
- Stage2 result files:
  - `../../../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/rqvae_best.pth`
  - `../../../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/sids_raw.npy`
  - `../../../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`
  - `../../../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`

## Observed run facts to verify against primary files

- Exactly one authorized no-argument Stage2 process was supervised as `iter29-stage2`; it eventually exited 0 with no restart. The supervisor's TCP readiness check on port 50200 timed out at 120 seconds while the process remained running. The worker log then documented DDP rank `0/4`, continuous progress, and final completion; no restart or second launch occurred.
- The worker log reports the iter8 warm-start path, 11 tensors loaded, the locked closed-form vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`, and fixed-curvature invariant markers at 0, 25000, 50000, and 100000.
- The final worker record reports `global_step=100000`, checkpoint step 100000, raw SIDs shape `(24587,3)`, and successful 4-token NPY + JSON export with `SID_WIRING_PASS`. Searches found no `Stage3 export FAIL`, traceback, runtime, floating-point, or CUDA error markers.
- The final logged SID metrics are full Gini `0.0877`; per-layer Gini `[0.1749,0.3076,0.3610]`; 22301 unique 3-token rows among 24587 items; 12960 L0/L1 pairs; `H(L1|L0)=5.3012`; 2286 collision rows; extension IDs 768–780. These are descriptive only under `CLAUDE.md` and FCCR-1.
- The integrity smoke directly checked checkpoint `global_step=100000`, fixed-c buffers, integer shapes/dtypes, raw tokens in `[0,255]`, exact 3-token prefix equality, complete JSON index coverage, row-by-row JSON/NPY equality, and output freshness after the prelaunch audit. It emitted `STAGE2_OUTPUT_INTEGRITY_PASS`.
- S08 MVG reports a measurable quantizer counterfactual (candidate loss `2.10162615776062`, control `0.8013777732849121`, delta `1.300248384475708`, assignment fraction changed `0.4062500298023224`), but explicitly does not establish Stage2 or Stage3 efficacy. It had no batch fingerprint and did not itself rehash Stage1/Stage0 files; S09 did rehash the configured Stage2 inputs immediately before launch.
- Stage3 has not run and no Stage3 result exists yet.

## Independent work required

Each candidate must read the same source packet and primary files; independently recompute the SID metrics from `sids_raw.npy` according to the inspected implementation (including full tuple/per-layer Gini, unique counts, L0/L1 pairs, and `H(L1|L0)`), inspect collision-extension behavior from both exports, validate checkpoint/export alignment, and reconcile independent values with the captured log. Separate direct contract/execution facts from interpretation. Explicitly note the port-readiness warning and the S08 evidence limits without inflating them into a failed run. Do not infer downstream recall from Stage2 metrics or MVG. End with whether S10 gates are satisfied and the one authorized next action; no Stage3 launch or source-code change in this stage.
