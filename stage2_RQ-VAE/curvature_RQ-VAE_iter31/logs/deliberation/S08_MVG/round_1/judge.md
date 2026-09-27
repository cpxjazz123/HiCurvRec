# Iter31 S08 MVG — Round 1 Judge C Adjudication

```text
STAGE_ID=S08_MVG
ROUND=1
VERDICT=REJECT_BOTH

HARD_GATE_A=FAIL
HARD_GATE_B=FAIL

EVIDENCE_FOR_A=Agent A independently reviewed the frozen S08 packet and source, recommended HOLD_FOR_SCOPED_REPAIR, and identified the missing component-wise gradient check for the existing behavior_loss (agent_a.md:1-7,21-27). Primary source confirms BEHAVIOR_LOSS_WEIGHT=0.20 and RqVaeComputedLosses.behavior_loss (modules/rqvae.py:37,49-55), computes behavior_loss from per-layer residual contrastive distances and cross-entropy (modules/rqvae.py:353-377), and includes it in forward's total loss (modules/rqvae.py:393-402). Current scripts/mvg_check.py::_check_gradients checks only reconstruction and quantizer components with autograd.grad (scripts/mvg_check.py:313-340), then checks total loss backward (342-354); it never differentiates output.behavior_loss independently.
EVIDENCE_FOR_B=Agent B independently reviewed the same frozen packet and primary source, also recommended HOLD_FOR_SCOPED_REPAIR, and independently found the same omission (agent_b.md:1-11,19-25). Its source references and conclusions are corroborated directly by modules/rqvae.py:37,49-55,353-377,393-402 and scripts/mvg_check.py:313-354. The S08 packet required independent A/B review against the same materials (source_packet.md:23-31); neither candidate reports runtime execution or edits.
PROBLEMS_A=The candidate's proposed affected-parameter prefix is intentionally left open pending source trace; the repair must use a robust criterion tied to the actual behavior-loss autograd graph, not assume a prefix. Its source audit does not provide runtime evidence, and appropriately makes no such claim.
PROBLEMS_B=The candidate's proposed affected-parameter prefix is intentionally left open pending source trace; the repair must use a robust criterion tied to the actual behavior-loss autograd graph, not assume a prefix. Its source audit does not provide runtime evidence, and appropriately makes no such claim.

WHY_NOT_A=The recommendation to hold is evidence-supported, but the report is not itself an S08 authorization and leaves the gradient parameter boundary underspecified. The missing proof is confirmed by primary code; neither total-loss gradients nor the HRA-specific-auxiliary-loss=False flag proves behavior_loss independently differentiable and active.
WHY_NOT_B=The recommendation to hold is evidence-supported, but the report is not itself an S08 authorization and leaves the gradient parameter boundary underspecified. The missing proof is confirmed by primary code; neither total-loss gradients nor the HRA-specific-auxiliary-loss=False flag proves behavior_loss independently differentiable and active.

CANONICAL_DECISION=REJECT_BOTH for this S08 round because the current proposed MVG checker fails the applicable gradient-coverage gate. This is a checker-completeness failure, not evidence that the registered HRA mechanism is inactive or infeasible, so do not ABORT_ITERATION. Root CLAUDE.md §6 requires component-wise torch.autograd.grad(loss_item, model.parameters(), retain_graph=True, allow_unused=True) for each applicable existing mechanism loss (including contrastive); canonical S06 plan logs/implementation_plan_iter31.md:87-91 reiterates this for relevant existing components. The inherited behavior_loss is applicable despite not being HRA-specific: it is exposed as a computed loss, is calculated from quantized residuals in forward, and has nonzero coefficient 0.20 in total loss (modules/rqvae.py:37,49-55,353-377,393-402). Existing component checks cover only reconstruction and quantizer loss (scripts/mvg_check.py:313-340). Total loss.backward() and aggregate encoder/codebook gradients (342-354) cannot establish that this particular branch has a finite nonzero gradient. grad_check.py is a thin delegate to the same _check_gradients and adds no independent coverage (grad_check.py:1-5,20-27); do not edit it.

A checker-only repair is authorized, exactly once and only in scripts/mvg_check.py::_check_gradients: add output.behavior_loss as a separate component; require it to be a finite scalar with requires_grad=True and grad_fn not None; obtain its gradients with torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True); reject any non-finite returned gradient; require at least one finite nonzero gradient (using the existing GRAD_EPSILON policy) among the trainable model parameters returned from this component-specific autograd.grad, and report the nonzero parameter names under a distinct behavior_loss component entry. The parameter policy is intentionally graph-derived: autograd.grad over model.parameters() returns gradients only along the component's actual graph, so at least one finite nonzero gradient among those parameters proves a connected model-parameter path without guessing an encoder/codebook prefix. Source trace confirms behavior_loss is calculated from quantized.residuals: get_semantic_ids starts residuals at encode(x), appends each residual before quantization, then updates residual via quantizer embedding and residual/transport operations (modules/rqvae.py:282-307); the contrastive loss consumes source/candidate residuals (353-377). Do not require every parameter or a hard-coded subsystem prefix to receive gradient. Preserve reconstruction and quantizer component checks, total loss graph/finiteness checks, actual loss.backward() and existing total encoder/codebook gradient checks. No model/loss/weight/equation/batch/checkpoint/curvature change; no optimizer step, checkpoint save, training, or other checker/source edit.

The approved one-run design is otherwise bounded at source level: one hard-coded Iter8 warm-start checkpoint path (mvg_check.py:16-19,53-59), deterministic first BATCH_SIZE active transitions from the configured dataset (35-50), same quantized embeddings for HRA-vs-Euclidean comparison (201-215), fixed-curvature buffer/invariance/optimizer-exclusion checks without optimizer.step() (102-160,122-139), and explicit domain/effect diagnostics (201-295). Main's source packet records the pinned checkpoint SHA-256 and reports that it was externally verified against the expected value during setup (source_packet.md:14-16); that external hash verification is provenance evidence, not a runtime MVG result or a checker-side hash assertion. These static findings establish only that the proposed run boundary is bounded; they do not prove data availability, runtime geometry, direct effect, gradients, or successful execution. No checker, MVG, or GPU run has occurred. After the authorized edit, require fresh independent S08 round-2 Agent A and Agent B reviews and a new Judge decision before any single `python scripts/mvg_check.py` invocation. Only a separate round-2 Judge may authorize that run. S08 is FAIL / NOT AUTHORIZED now; S09, Stage2, Stage3, and all GPU/training execution remain unauthorized.

CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S08_MVG/round_1/judge.md
CONFIDENCE=HIGH
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Apply only the authorized _check_gradients behavior_loss component-check repair; then freeze the repaired source packet and obtain fresh independent S08 round-2 A/B reviews and Judge adjudication. Do not run MVG or advance to S09 before that Judge explicitly authorizes the one run.

REPLAN_CONSTRAINTS=Limit code change to scripts/mvg_check.py::_check_gradients and the behavior_loss component's own graph/finiteness/gradient reporting. Keep all registered science, loss weights, model, checkpoint, batch, curvature, existing checks, and no-step/no-save boundary unchanged. Do not edit grad_check.py, run the checker, or authorize runtime/Stage2/Stage3/GPU work in this round. Round 2 must freshly review the actual repaired source, and a separate Judge must decide execution authorization from primary evidence.
```

## Authorization state

```text
S08_SOURCE_AUDIT=FAIL (missing applicable behavior_loss component gradient check)
S08_RUNTIME_AUTHORIZED=NO
S08_REPAIR_AUTHORIZED=YES (one checker-only _check_gradients repair, exact scope above)
FRESH_S08_ROUND2_A_B_JUDGE_REQUIRED=YES
MVG_RUNS_AUTHORIZED=0
S09_AUTHORIZED=NO
STAGE2_AUTHORIZED=NO
STAGE3_AUTHORIZED=NO
GPU_OR_TRAINING_AUTHORIZED=NO
RUNTIME_RESULTS_ESTABLISHED=NO
```