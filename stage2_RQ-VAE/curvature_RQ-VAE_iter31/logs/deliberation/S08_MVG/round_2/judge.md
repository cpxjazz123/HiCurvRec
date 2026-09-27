# Iter31 S08 MVG — Round 2 Judge C Adjudication

```text
STAGE_ID=S08_MVG
ROUND=2
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Agent A recommends AUTHORIZE_ONE_MVG_RUN after independently reviewing the frozen round-two packet and repaired primary source (round_2/agent_a.md:1-15,17-25). It identifies the exact round-one repair, verifies behavior_loss as a separate component-specific autograd check with graph-derived nonzero parameter names, and confirms existing component/total backward checks remain. It also distinguishes static readiness from runtime evidence and carries the pinned-checkpoint/data-input pre-run requirements and one-run boundary (agent_a.md:18-25). Primary evidence is corroborated below.
EVIDENCE_FOR_B=Agent B independently recommends AUTHORIZE_ONE_MVG_RUN and expressly makes no runtime-success claim (round_2/agent_b.md:1-13,15-28,30-34). It reaches the same conclusion on component-specific behavior_loss gradients, retained reconstruction/quantizer and total backward checks, fixed-curvature and no-step boundary, and pre-run checkpoint/data checks. Primary evidence is corroborated below.
PROBLEMS_A=No material source-level error found. Agent A's recommendation is not execution evidence; data/checkpoint availability and the actual runtime results remain unverified and are explicit pre-run/run obligations.
PROBLEMS_B=No material source-level error found. Agent B's recommendation is not execution evidence; data/checkpoint availability and the actual runtime results remain unverified and are explicit pre-run/run obligations.

WHY_NOT_A=Not applicable to MERGE_AB: A's repair-conformance and authorization-boundary findings are accepted, but the permission below is Judge C's own bounded adjudication, not authority delegated from A.
WHY_NOT_B=Not applicable to MERGE_AB: B's source-scope and pre-run findings are accepted, but the permission below is Judge C's own bounded adjudication, not authority delegated from B.

CANONICAL_DECISION=MERGE_AB / PASS for source readiness. The exact round-one authorization was limited to scripts/mvg_check.py::_check_gradients: separately check the existing output.behavior_loss as a finite scalar with requires_grad and grad_fn, compute its own gradients through torch.autograd.grad(component, trainable model parameters, retain_graph=True, allow_unused=True), reject non-finite returned gradients, require at least one finite nonzero graph-connected parameter gradient under existing GRAD_EPSILON, and report its parameter names separately. The current implementation matches that scope: scripts/mvg_check.py:313-319 constructs the trainable named-parameter list and includes reconstruction, quantizer, and behavior_loss as distinct entries; :321-354 validates each component and differentiates each component independently, with behavior_loss scalar validation at :326-327, gradient finiteness at :332-336, nonzero graph-derived names at :337-354. For behavior_loss, expected_prefix=None makes the acceptance criterion any nonzero gradient returned by that loss-specific autograd call (:339-340); no unsupported parameter-prefix assumption is introduced. This is applicable, not an HRA-specific-loss claim: modules/rqvae.py:353-377 computes the existing behavior contrastive objective from per-layer quantized residuals, fixed-curvature distances, masked logits and cross-entropy; :393-402 includes it in actual forward total loss at BEHAVIOR_LOSS_WEIGHT=0.20; :424-429 returns it as a distinct computed component. Thus the prior Judge's concern is resolved without changing the registered loss/model mechanism.

The authorized repair preserves the other required paths: current checker :302-310 validates total-loss graph/finiteness and zero curvature regularization; :315-318 and :321-354 retain reconstruction and quantizer component checks; :356-368 still calls actual total loss.backward() and requires finite nonzero encoder/codebook gradients. The total backward does not substitute for behavior_loss's separate autograd check. S06 implementation plan requires component-wise checks for relevant existing losses and actual backward at logs/implementation_plan_iter31.md:87-91; the Round-1 Judge restricted the one repair to this precise checker scope and explicitly required fresh Round-2 adjudication before execution (S08_MVG/round_1/judge.md:19-23,31-49). The packet states grad_check.py and model/loss/HRA sources were unchanged (round_2/source_packet.md:10-19); this is consistent with the reviewed delegate/model source, and no source edits were made by Judge C.

Pre-run/source boundary is static only: mvg_check.py:16-19 pins the Iter8 rqvae_best.pth path; :35-50 selects the first BATCH_SIZE active transition records, failing if too few exist; curvature_RQ-VAE.py:87-100,140 wires configured data, seed=42, and BATCH_SIZE=640; curvature_config.py:23-26 identifies the configured Stage1 sentence_t5.npy, item_ids.json, and Stage0 train.parquet. The packet pins the expected checkpoint SHA-256 as 189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b (round_2/source_packet.md:25-32). The checker loads the fixed path but does not itself assert that digest (mvg_check.py:53-59); therefore Main must verify the exact hash externally immediately before invocation. Neither current readability nor the hash immediately before run has been established by this adjudication.

The checker source also retains the intended single-checkpoint/single-batch design, fixed-curvature checks and HRA-vs-Euclidean diagnostics (mvg_check.py:16-23,35-59,102-162,177-295,377-409). Source inspection does not establish runtime batch IDs, direct effect, actual projection/clamp/domain incidence, curvature invariance, gradient values, or MVG PASS. The static S07 PASS explicitly leaves runtime claims open (S07_PREFLIGHT/round_3/judge.md:72-78,88-100). Candidate A/B agree on readiness; their statements do not establish that runtime has occurred.

ONE_MVG_RUN_AUTHORIZED=YES, subject to ALL pre-run conditions below. This adjudication authorizes exactly one invocation, not an attempt budget:
  executable=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
  command=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 scripts/mvg_check.py
  cwd=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31
  args=NONE
Immediately before that invocation, Main MUST verify that each configured data input (Stage1 sentence_t5.npy, Stage1 item_ids.json, and Stage0 train.parquet; exact configured paths in curvature_config.py:23-26) exists and is readable, and run sha256sum on exactly /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth. The observed digest MUST equal the pinned value above. Record the checks and digest. If any input is missing/unreadable, the digest differs, or another prerequisite fails, abort without invoking the checker and do not substitute inputs/checkpoint.

For the one invocation, capture the complete raw stdout and stderr, exit status, device and batch shape, actual selected batch size/IDs and provenance, and checkpoint identity/hash. Preserve all checker output, including HRA direct-effect/domain/projection/log-clamp/Möbius diagnostics, fixed-curvature and optimizer-exclusion/invariance checks, component gradient details, and total backward results. Do not claim PASS unless the observed invocation reports success and all required checks are evidenced.

No alternate checkpoint or batch; no retry, repeat, sweep, separate grad_check.py invocation, optimizer step, checkpoint save, training loop, Stage2, Stage3, S09, or any other downstream action is authorized. The actual required loss.backward() inside the single MVG checker is permitted solely as the prescribed gradient-path verification, not as training or an optimizer update. An MVG pass alone does not authorize S09; any S09 adjudication requires its own evidence and Judge decision. The unresolved Stage3 map-derived RQVAE_VARIANT discrepancy remains a mandatory pre-S11 gate (S07_PREFLIGHT/round_3/judge.md:80-98).

CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S08_MVG/round_2/judge.md
CONFIDENCE=HIGH for source conformance and exact authorization boundary; no runtime confidence claim
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Main performs the pre-run readable-input checks and exact pinned-checkpoint sha256sum immediately before the sole authorized invocation; abort without execution on any failed prerequisite, otherwise run exactly once and record complete raw output and provenance for a separate S08 runtime adjudication. Do not advance to S09 or any later stage from this authorization alone.

REPLAN_CONSTRAINTS=No further source edit is authorized by this decision. Preserve the frozen Iter31 mechanism, losses, checkpoint, deterministic batch rule, curvature, and no-step/no-save boundary. Failure of a pre-run prerequisite means stop without substitution. Runtime failure is evidence for a separate Judge analysis and does not grant retry or downstream permission.
```

## Execution authorization state

```text
S08_SOURCE_REPAIR=PASS
S08_RUNTIME_AUTHORIZED=ONE_INVOCATION_ONLY
ONE_MVG_RUN_AUTHORIZED=YES
PRE_RUN_INPUT_READABILITY_CHECK_REQUIRED=YES
PRE_RUN_EXACT_CHECKPOINT_SHA256_REQUIRED=YES
PINNED_CHECKPOINT_SHA256=189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b
AUTHORIZED_INTERPRETER=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
AUTHORIZED_CWD=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31
AUTHORIZED_ARGUMENTS=NONE
RUNTIME_RESULTS_ESTABLISHED=NO
RETRY_OR_DUPLICATE_GRAD_CHECK_AUTHORIZED=NO
OPTIMIZER_STEP_SAVE_OR_TRAINING_AUTHORIZED=NO
S09_AUTHORIZED=NO
STAGE2_AUTHORIZED=NO
STAGE3_AUTHORIZED=NO
GPU_TRAINING_AUTHORIZED=NO
```
