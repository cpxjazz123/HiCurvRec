# Iter31 S09 Stage2 execution-plan adjudication — Judge C

STAGE_ID=S09_STAGE2_EXECUTION
ROUND=1
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Agent A independently identifies the live Iter31 result-root routes, the exact Iter8 warm-start behavior and strict=False missing-key caveat, the pinned no-argument four-rank launcher and 100,000-step/no-quality-gate policy, S08 root §6 gradient evidence, and immediate prelaunch blockers. Direct inspection corroborates its claims: curvature_RQ-VAE.py:626-665, 686-705, 827-894, 904-946; curvature_config.py:30-34; modules/step_checks.py:94-168; modules/sid_quality.py:20-28, 147-148; and logs/mvg_check_iter31.log.
EVIDENCE_FOR_B=Agent B independently reaches the same conditional-proceed recommendation and additionally makes the full external-output/overwrite inventory requirement explicit. Direct inspection confirms Stage2 writes raw SID under RQVAE_OUT_DIR and exports final SIDs to SIDS_NPY and ITEM_SIDS_JSON outside it (curvature_RQ-VAE.py:446-510, 839-845), while Step0 deletion is scoped to the listed patterns inside OUT_DIR (modules/step_checks.py:94-168). The live config places all three routes under the Iter31 results root (curvature_config.py:30-34).
PROBLEMS_A=Agent A's runtime transfer safeguards need the exact missing-key result from S08 made a blocking identity condition: the training loader itself uses strict=False and does not reject or report missing keys, and its positive transferred-tensor count is not sufficient on its own. Its output-inventory statement must enumerate the two final exports outside RQVAE_OUT_DIR and the outer/inner log overwrite targets.
PROBLEMS_B=Agent B correctly flags the strict=False limitation and all external exports, but the canonical decision must not imply the training process itself reports missing-key diagnostics. It must use S08's exact key-set validation only while the validated model/checkpoint/source identity remains unchanged, and stop if that identity cannot be re-established.

WHY_NOT_A=Not rejected. Its input, launcher, fixed-step, S08 evidence and conditional-go analysis are retained; its runtime warm-start and output-inventory cautions are tightened in the merged plan.
WHY_NOT_B=Not rejected. Its result-root-wide output inventory and explicit non-overwrite requirements are retained; the merged plan distinguishes S08's checker evidence from what the training loader actually logs.

MERGE_COMPONENTS_A=Use Agent A's direct source audit of the configured inputs, current DDP launcher, root-prescribed invocation, full-step loop, descriptive-only SID metrics, and S08's already completed §6 gradient evidence; do not rerun MVG.
MERGE_COMPONENTS_B=Use Agent B's explicit inventory of OUT_DIR cleanup plus external SIDS_NPY and ITEM_SIDS_JSON exports, and its blocking identity/no-overwrite prelaunch checks.

CANONICAL_DECISION=Conditionally authorize exactly one Iter31 Stage2 run, but NOT now. Launch is authorized only after every blocking recheck in the canonical plan is recorded PASS, including the deliberation gate, fresh input/checkpoint identities, S08 source/checkpoint identity continuity, safe output inventory, and runtime availability. Any failure blocks launch; no user input is required. This does not authorize Stage3.
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/stage2_execution_plan_iter31.md
CONFIDENCE=HIGH
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Materialize the canonical plan. Before any run, execute its immediate blocking prelaunch checks and the required deliberation_gate.py. If all pass, execute exactly the single prescribed no-argument four-rank Stage2 run and capture the specified evidence; if any check fails, do not launch and use the same-iteration operational repair/adjudication path without changing the registered mechanism or protocol.

REPLAN_CONSTRAINTS=Preserve Iter31's registered HRA-STEP6-1/FCCR-1 contract, inputs, one-factor definition, seed, batch/topology, no-argument launcher, results-only destinations, and full 100,000-step/no-quality-gate policy. No code or mechanism edits, input substitutions, alternate checkpoint, random-init fallback, second run, baseline rerun, Stage3 execution, proxy gate, sweep, replication, or root-cause study is authorized. Any unresolved identity, key-transfer, output-overwrite, runtime, or deliberation-gate failure blocks the launch and must be repaired through the applicable same-iteration operational process.

## Evidence and adjudication rationale

The frozen source packet and both completed candidates agree on a conditional single-run recommendation, but neither candidate is itself execution authorization. The live trainer imports and uses the Iter31 results-root routes; current `curvature_config.py` sets `RQVAE_OUT_DIR`, `RAW_SIDS_NPY`, `SIDS_NPY`, and `ITEM_SIDS_JSON` under the short `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/` root. The trainer writes its three-token raw SID below OUT_DIR and the final four-token NPY and item JSON to the two separate destinations outside OUT_DIR. The final exporter validates token ranges, uniqueness, JSON key/row width and sample agreement, and NPY round-trip. `SAVE_DIR_ROOT`, `CONFIG_PATH`, and `BEST_CKPT_PATH` are present in the shared config but are not consumed as Stage2 output routes by this trainer; recheck current write callsites as part of the destination inventory.

Warm-start caution: the trainer's load at `curvature_RQ-VAE.py:631-659` filters the three `c_layer_scale` and three `_fixed_c` names, calls `load_state_dict(..., strict=False)`, and raises on unexpected keys only. It does not inspect `missing`, and its transferred count alone does not prove full compatibility. S08's `scripts/mvg_check.py:_build_model` does inspect the result and hard-fails unless the missing set is exactly `{layers.0._fixed_c, layers.1._fixed_c, layers.2._fixed_c}` and unexpected is empty. It used the same Iter8 checkpoint identity and the same `RqVae` state layout (its differing `codebook_kmeans_init=False` only controls initialization; `freeze_layer_scale` defaults to the same false value explicitly passed by training). Thus the S08 check closes the structural key-compatibility concern only if the immediate launch-time checkpoint hash and the exact source/model implementation remain the S08-validated versions and training's explicit skip set remains the same. The training loader itself still does not emit missing-key diagnostics; the plan must not claim otherwise. If continuity cannot be proven, do not launch.

S08 already captured root §6 one-checkpoint/one-batch gradient evidence and MVG PASS with no optimizer step. It must not be repeated for this launch review. S07's FCCR-1 mechanism-contract PASS is also already recorded; this S09 decision makes no contract transition. Current source fixes seed 42, 100,000 global steps, batch 640 per rank, three 256-entry codebooks, periodic checkpoints every 10,000 steps, and a four-rank `torchrun --standalone` launcher on port 50200. `should_early_stop` is unconditionally false; SID quality metrics are descriptive, not gates.

The run is not authorized at this moment because immediate prelaunch identities, runtime and output state have not been re-established by this Judge. The plan is a conditional authorization, not a claim those checks already passed. No Stage2 or Stage3 process was launched during this adjudication.

## S09 evidence-only operational repair amendment

ROUND_TYPE=OPERATIONAL_REPAIR
REPAIR_AUTHORIZATION=Judge C authorizes and adjudicates an evidence-only amendment to close the active HRA/FCCR dependency identity set; no new A/B cycle is required.
LOCKED_SCIENCE_CHANGED=NO
REPAIR_SCOPE=Add ten active Step6/model/data/curvature dependency and contract identities to the canonical execution plan, require immediate prelaunch SHA-256 equality for the complete source/dependency identity union, and record the repair evidence.
REPAIR_STATUS=REPAIR_PASS
REPAIR_RECORD=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S09_STAGE2_EXECUTION/round_1/repair_record.md
VERIFICATION=Two independent read-only sha256sum batches were run in parallel for the ten added files; every returned hash matched the amended plan. Plan readback confirms the full ten-row dependency table and launch-time equality requirement. The initial source identity table is explicitly retained as coordinator-provided and still requires fresh launch-time equality.
STAGE2_LAUNCH_AUTHORIZED=NO

This repair changes no source/model/checkpoint/input/contract/output or experiment history. The original S09 `MERGE_AB` verdict remains in force, and its conditional one-run authorization is unchanged: Stage2 remains unauthorized until the deliberation gate and every immediate prelaunch condition—including rehash equality for the full combined identity set—pass. No Stage2/Stage3 process was launched.
