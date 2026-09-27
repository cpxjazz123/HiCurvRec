ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S07_PREFLIGHT/round_1/source_packet.md
STAGE_ID=S07_PREFLIGHT
ROUND=1
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45

# Scope and verdict boundary

Static audit against canonical S00–S06 artifacts, the supplied FCCR-1 contract, and the named iter30 implementation. `S07_STATIC_SOURCE_AUDIT=PASS`. This is not a preflight execution result or authorization to proceed to MVG, gradients, or launches. No trainer, checkpoint, model, gradient, preflight script, MVG, GPU, Stage2 or Stage3 run was executed. Candidate B was not read or contacted.

# Hard-gate audit

## GATE 1 — FCCR-1 fixed-buffer implementation and static preflight compatibility: PASS

`modules/quantize.py:68-73` registers `_fixed_c` as a buffer; `:98-111` shows `get_c()` returns that buffer when fixed curvature is configured and otherwise raises, without consulting curriculum step. `:128-131` only stores the step. `modules/rqvae.py:38` sets `CURVATURE_REG_WEIGHT = 0.0`; `:371-378` routes the regularizer through that zero coefficient. The root trainer defines the cyclic-range constants at `curvature_RQ-VAE.py:121-123`, but its profile-specific model construction supplies `fixed_layer_curvatures` at `:344-359` and `:456-463`; I found no reachable cyclic-curvature calculation in the inspected fixed-curvature `get_c()` path. The unchanged curvature-scaled Sinkhorn epsilon is an inherited consumer of fixed curvature, not a new rule (`modules/quantize.py:185-222`).

Static review of `preflight_contract.py:18-25, 56-77, 96-125, 145-205` shows the contract checks for FCCR-1 policy values, the approved ten-field contract schema, semantic manifest terms, curvature trainability/step dependencies, a fixed buffer, and nonzero regularization. The actual contract is the approved ten-field S04 object (`logs/mechanism_contract_iter30.json`); its semantic `formula_inputs` label is `raw_residual_median`, while the required source JSON key is the plural `raw_residual_medians`. S04 Judge evidence explicitly records this distinction (`logs/deliberation/S04_CONTRACT/round_1/judge.md:8-20`). The loader enforces the plural key and exact contract (`scripts/compute_closed_form_curvature.py:165-200`). This is consistent, not a schema mismatch.

**Not performed:** running the static preflight. Therefore `MECHANISM_CONTRACT_PASS` remains unobserved and is a blocker to MVG/Stage2; the static compatibility finding does not substitute for it.

## GATE 2 — Both registered formulas and explicit plural input: PASS

The calculator declares the registered `[L0,L1,L2]` inputs and both final vectors (`scripts/compute_closed_form_curvature.py:13-34`). The iter26 control calculation implements the registered `s`, population-standardization, and clipping formula; the iter29 candidate implements `x`, `y`, `u`, and bounded affine mapping (`:102-153`). Both validate exact ordered branching and plural residual vectors against the registrations (`:79-95`). JSON loading rejects missing `raw_residual_medians` and explicitly does not fall back to legacy aliases (`:165-183`); candidate contract values are checked against the candidate map, then either selected mapping is returned (`:184-205`). The JSON contains the explicit plural field (`scripts/computed_behavior_branching.json:12-16`).

The S06 smoke record separately reports both formulas/intermediates matched registered values within 1e-9 and a legacy-alias-only payload was rejected (`logs/implementation_smoke_iter30.log`). That is S06 evidence, not a rerun or S07 execution. Residual-provenance uncertainty remains the canonically adjudicated S03 limit: historic method/value provenance is medium confidence and does not establish checkpoint replay or historical byte identity (`logs/mechanism_manifest_iter30.md`, “§7 gate decision and limitations”; `logs/deliberation/S03_PROVENANCE/round_1/judge.md:20-27`). S03 is a limited PASS, not an open gate or permission to substitute values.

## GATE 3 — Six literal profiles, root serial dispatch, and DDP identity: PASS

`curvature_config.py:11-18, 21-76` defines the six fixed profiles in the locked pair/seed order and their distinct mapping, curvature, paths, and labels. The static `RQVAE_OUT_DIR` is visibly inside the short iter30 Stage2 result root (`curvature_config.py:79-84`). Route validation asserts exact order, mapping/vector identity, root paths, unique route destinations, and expected wrapper files (`scripts/profile_routes.py:224-294`).

The only operator entry is the no-argument root `curvature_RQ-VAE.py`; its `__main__` rejects an inherited `RANK` and calls the fixed dispatcher (`curvature_RQ-VAE.py:750-778`). `dispatch_profiles` loops in locked order, sets each literal wrapper as the torchrun script, prepares that profile, runs its distinct gradient checker, then starts its Stage2 DDP job and raises on nonzero return (`scripts/profile_routes.py:472-508`). The wrapper passes one literal label and `__file__` (`scripts/run_stage2_iter26_mapping_seed43.py:1-4`; the other five wrappers use the corresponding literal profile). DDP re-entry checks that the selected script resolves to that profile's expected wrapper and that the run is an authorized child (`scripts/profile_routes.py:453-470`). No CLI/environment/cwd/shared-file profile selector appears in this route topology.

## GATE 4 — Strict warm-start and immediately-prior per-run gradient placement: PASS (static wiring only)

The shared warm-start loader pins the exact iter8 path and digest (`modules/warm_start.py:11-25, 38-53`), requires a mapping-valued model state, rejects unexplained/missing tensors, validates shape/dtype/finiteness and transfer completeness, allows only the listed legacy curvature keys, verifies the loaded state, and confirms fixed buffers remain non-trainable and equal to the requested values (`:55-140`). Stage2 rank 0 catches warm-start failure, broadcasts failure status, and aborts all ranks (`curvature_RQ-VAE.py:467-488`). The per-run checker imports the same loader and verifies profile identity and locked input digests (`scripts/grad_check.py:19-31, 75-104`); it performs the one-batch forward/backward, requires finite connected loss/nonzero gradients, and excludes fixed curvature from optimizer parameters (`:116-203`).

Placement is explicit: after preparing each profile, the root dispatcher invokes that profile's unique no-argument gradient wrapper with `check=True` immediately before `_launch_via_torchrun()` (`scripts/profile_routes.py:472-508`). The actual six gradient checks have not run. Each remains a per-run blocker; S07 static wiring is not evidence that any passed.

## GATE 5 — Exact result/SID routes, inventory, and fail-closed export: PASS

The profile constructor routes Stage2 products to each label under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/` and Stage3 products under the matching Stage3 short root (`curvature_config.py:21-61`). Route validation rejects path collisions and checks exact roots/exports/log routes (`scripts/profile_routes.py:224-294`). The inventory rejects symlink/non-directory/nonempty roots and existing individual outputs (`:297-330`); prelaunch identity records the roots, file destinations, inputs, and hashes (`:332-423`). The child validates that identity and that all outputs remain absent before training (`:424-470`).

The exporter requires absolute distinct paths and the exact matching profile routes, rejects existing output paths, validates three-token shape/range and unique four-token extension, and verifies NPY and JSON read-back parity/dense keys (`scripts/export_sids_for_stage3.py:37-128`). Final training fails closed if raw SID creation or four-token Stage3 export fails (`curvature_RQ-VAE.py:641-718`). Stage3 independently requires the matching Stage2 checkpoint/SID exports and completion/gradient markers (`scripts/stage3_profile_runner.py:74-121`). Thus a failed export cannot silently feed a stale SID file to Stage3.

S06 smoke evidence reports synthetic absent/empty/nonempty collision guards and four-token JSON checks passed (`logs/implementation_smoke_iter30.log`). No real output roots were inspected or created in S07.

## GATE 6 — Stage3 trainer, resolver, and protocol unchanged: PASS (source wiring; hash claim remains S06 evidence)

The existing Stage3 resolver ordinarily includes legacy fallbacks (`stage3_T5Train/train_HG-Rec.py:215-256`). The iter30 wrapper requires the SID JSON inside the paired Stage2 root, checks its digest/identity/protocol/source state, and wraps the trainer resolver so only the exact absolute SID file can be selected (`scripts/stage3_profile_runner.py:225-321`). It changes only the matched profile's SID path, variant, seed, output paths, and wrapper/launcher paths. It checks the locked Stage3 settings and launcher values before launch (`:263-321`). The six Stage3 entry wrappers each bind one literal label (`scripts/run_stage3_iter26_mapping_seed43.py:1-4`; analogous wrappers provide the other labels). The source packet and S01 record lock the trainer SHA; S06 smoke log records that current trainer bytes matched `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. The trainer was not changed by the described S06 smoke, and its resolver and launch code were directly inspected; the digest statement here is specifically the recorded S06 smoke evidence, not a hash computation in this S07 audit. The existing launcher preserves `NO_EVAL=True`, `SKIP_TEST=False`, four ranks, and port 50201 as locked (`train_HG-Rec.py:56-58, 124-166, 1185-1225`; wrapper checks at `stage3_profile_runner.py:263-321`).

## GATE 7 — Unexplained second mechanism: PASS

The maps are the only registered arm difference. Common behavior loss (`modules/rqvae.py:37-39`) and fixed-curvature consumers, including the existing effective Sinkhorn epsilon (`modules/quantize.py:185-222`), are inherited and common. The source sets curvature regularization to zero, adds no trainable curvature parameter, and selects the registered profile's fixed vector. I found no additional new optimizer, auxiliary loss, schedule, or Stage3 trainer change in the inspected files.

## GATE 8 — S06 smoke versus deferred S07/S08/S09 evidence: PASS as an evidence-boundary audit

S06 smoke: the existing log records compile-only checks, both pure map calculations, profile/path checks, synthetic collision guards, and the recorded Stage3 trainer hash; it expressly records no trainer/checkpoint load, gradient check, preflight, MVG, GPU, Stage2/Stage3 run, or result-path creation (`logs/implementation_smoke_iter30.log`). This audit did not repeat that smoke.

Unperformed and still blocking: run the skill preflight once after Judge C accepts S07; conduct independently adjudicated S08 MVG, including same-batch L0 pre-centering distance counterfactual and measured repeat tolerance; perform each profile's actual gradient check immediately before its corresponding Stage2 run; obtain S09 launch authorization; and only then consider any Stage2/Stage3/GPU execution. No runtime PASS or launch authorization is claimed here.

# Blockers, consequences, and residual uncertainty

1. **Preflight command has not run** (intentionally). Consequently no observed `MECHANISM_CONTRACT_PASS`; do not advance to MVG or Stage2 until Judge C acceptance and the one authorized preflight execution.
2. **S08 MVG is unperformed.** Fixed-value immutability across optimizer updates/modes/steps and active counterfactual effect remain unproven; no Stage2 launch is allowed until adjudicated MVG PASS.
3. **Six per-run gradient checks are unperformed.** A static checker path is present, but there is no evidence that any profile has finite nonzero intended gradients; each actual Stage2 launch remains blocked until its immediate pre-run check passes.
4. **S09 and all Stage2/Stage3 launches are unperformed and unauthorized.** No result, SID artifact, evaluation, or scientific conclusion exists.
5. **Provenance limit:** S03 passed only for historical method/value lineage; raw-residual confidence is medium and does not establish historical checkpoint replay/byte identity. This is not a failed S03 gate; it remains an inference/identity limitation to retain in later claims.

No additional source-level blocker was established in this static audit. The S06 smoke log is not substituted for the unperformed S07 preflight, S08 MVG, per-run gradients, or execution.

# Self-rejection criteria

Reject this candidate if primary source evidence shows either registered equation/vector or explicit `raw_residual_medians` handling differs; any reachable cyclic/trainable curvature or nonzero curvature regularization; any profile can be silently dropped/misrouted or DDP workers lose literal wrapper identity; any warm-start/input/output failure falls back or overwrites another route; SID export failure can reach Stage3 as stale data; the Stage3 resolver can select a fallback; or Stage3 trainer/protocol changed. Also reject if any assertion above relies on S06 smoke as proof of S07/S08/runtime behavior, or if the not-yet-run preflight fails.

# Autonomous next action

After both independent candidates are complete, Judge C must adjudicate this static audit against primary source. If accepted, the orchestrator's next action is to run the skill-level `preflight_contract.py` once, without arguments, from `stage2_RQ-VAE/curvature_RQ-VAE_iter30/`, recording its actual result. On failure, keep MVG and every launch blocked; on pass, proceed only to the independently adjudicated S08 MVG. Do not run gradients, trainers, or GPU work as part of this S07 action.