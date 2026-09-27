ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT
ROUND=1
RECOMMENDATION=MERGE_NOT_APPLICABLE; independent candidate recommendation for Judge C
VERDICT=ACCEPT_TRANSITION_CONDITIONALLY
CONFIDENCE=MEDIUM
USER_INPUT_REQUIRED=NO
NO_CODE_OR_CHECKER_EDITS=TRUE
NO_IMPLEMENTATION=TRUE
NO_STAGE2_OR_STAGE3_RUN=TRUE
NO_TRAINING_AUTHORIZATION=TRUE
NO_PERFORMANCE_CLAIM=TRUE

## Recommendation and adjudication

**Recommend that Judge C formally activate the registered S02 HRA-inspired Step6 mechanism for Iter31 by a between-iteration contract transition from the user-cancelled Iter30, while preserving FCCR-1 unchanged as the curvature substrate.** Make this a new composite Iter31 experiment contract, not a mutation of FCCR-1’s curvature-mapping meaning or JSON interface. Iter30 is recorded as cancelled at the user's direction with no scientific result or Stage2/Stage3/GPU run; it is not a failed or infeasible experiment that must be continued. The skill permits a canonically adjudicated between-iteration autonomous transition, and S02 has already registered a narrow, falsifiable Step6-only change. That supports activating it at S04, provided Judge C writes the canonical artifacts and later gates pass.

This transition is conditional on a real second contract/check path. The existing FCCR-1 checker is not a valid sole HRA check. The Iter29 curvature loader requires precisely the FCCR-1 ten-field JSON shape and validates its input provenance and formula; adding HRA keys there would break its exact-field-set validation. The non-mutating route is therefore: retain that exact FCCR-1 contract as an independently checked substrate, add a separate HRA Step6 contract, then have S06/S07 adjudicate and provide a new Iter31-local HRA checker. S04 itself authorizes no source, checker, training, or evaluation changes.

The transition must not imply that the HRA paper's exact telescope applies. S02 defines a validly testable common-reference aggregation, but the Iter29 residual cascade has heterogeneous fixed curvatures and cross-layer residual transport. The paper's exact cancellation relies on paired residual subtraction and reverse nesting under one shared curvature. The cross-curvature radial transfer has a recorded non-homomorphism witness. Thus contract identity should say `HRA_INSPIRED_STEP6_COMMON_REFERENCE`, not claim paper-equivalent exact HRA telescoping.

## Proposed canonical S04 artifacts

Judge C should list both of these as canonical artifacts:

1. `logs/mechanism_contract_iter31.json` — mandatory skill filename; exact FCCR-1 schema below, preserving its loader/checker compatibility.
2. `logs/hra_step6_contract_iter31.json` — additional canonical contract for the independently activated HRA-inspired Step6 mechanism.

Do not put the HRA keys inside the FCCR JSON. The existing Iter29 loader accepts exactly ten fields (the nine fixed fields shown plus `final_curvature_values`) and rejects extras. Its active Iter31 fork/copy must continue to point at and validate the FCCR-only contract as S06 determines; S04 does not edit or wire it.

### A. `logs/mechanism_contract_iter31.json` — exact FCCR-1 substrate

```json
{
  "contract_version": "FCCR-1",
  "curvature_source": "closed_form",
  "curvature_trainable": false,
  "curvature_time_varying": false,
  "uses_cyclic_schedule": false,
  "uses_curvature_regularization": false,
  "new_curvature_conditioned_optimizer": false,
  "new_curvature_conditioned_aux_loss": false,
  "formula_inputs": ["behavior_branching", "raw_residual_median"],
  "final_curvature_values": [
    1.3660953164241916,
    0.7347829661951981,
    0.6439958072706683
  ]
}
```

Retain the precise input/value provenance qualification: S03's PASS is historical method/value provenance only; the historical raw SID source and calibration checkpoint needed for replay are unavailable at the checked paths, and there is no promoted proof of replay, independent recomputation, or historical byte identity. Keep the explicit `branching` and `raw_residual_medians` source keys in the source JSON and do not substitute normalized `[0.001, 0.932889, 1.0]`, ambiguous `residual_norm`, or a refreshed measurement. This contract represents exactly the existing fixed mapping inputs `behavior_branching` and `raw_residual_median`, not Step6.

### B. `logs/hra_step6_contract_iter31.json` — separate activated mechanism

This is a proposed schema for Judge C to adjudicate and materialize. It encodes the operation rather than asserting that it has been implemented or executed.

```json
{
  "contract_version": "HRA-STEP6-1",
  "contract_status": "ACTIVE_FOR_ITER31_AFTER_S04_TRANSITION",
  "transition": {
    "from_iteration": 30,
    "from_iteration_status": "ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE",
    "from_contract": "FCCR-1",
    "to_iteration": 31,
    "transition_scope": "ADD_STEP6_MECHANISM_PRESERVE_FCCR1_SUBSTRATE",
    "authorization_source": "CANONICAL_S04_JUDGE_ONLY"
  },
  "substrate_contract": {
    "path": "logs/mechanism_contract_iter31.json",
    "contract_version": "FCCR-1",
    "curvature_source": "closed_form",
    "curvature_trainable": false,
    "curvature_time_varying": false,
    "uses_cyclic_schedule": false,
    "uses_curvature_regularization": false,
    "new_curvature_conditioned_optimizer": false,
    "new_curvature_conditioned_aux_loss": false,
    "formula_inputs": ["behavior_branching", "raw_residual_median"],
    "curvature_order": ["L0", "L1", "L2"],
    "final_curvature_values": [
      1.3660953164241916,
      0.7347829661951981,
      0.6439958072706683
    ],
    "input_json_keys_required": ["branching", "raw_residual_medians"],
    "historical_input_replay_status": "NOT_REPLAYED_PROVENANCE_LIMITED"
  },
  "mechanism": "HRA_INSPIRED_STEP6_COMMON_REFERENCE_AGGREGATION",
  "scope": {
    "replace_only": "ITER29_STEP6_EUCLIDEAN_TANGENT_SUM",
    "unchanged": [
      "stage1_embeddings_and_data",
      "step4_residual_subtraction",
      "step5_cross_layer_residual_transport",
      "quantizer_codebooks_assignments_and_ste",
      "all_existing_losses_and_loss_weights",
      "optimizer_and_sinkhorn_settings",
      "training_schedule_and_seed",
      "decoder_and_reconstruction_loss",
      "stage3_model_data_evaluator_and_protocol"
    ],
    "forbidden_additions": [
      "d-HSTE",
      "additional_loss",
      "curvature_change_or_learning_or_schedule",
      "optimizer_or_sinkhorn_change",
      "codebook_change",
      "stage1_change",
      "stage3_change",
      "second_new_mechanism"
    ]
  },
  "equation": {
    "layer_indices": [0, 1, 2],
    "layer_order": ["L0", "L1", "L2"],
    "e_l": "tangent-coordinate quantizer embedding, including existing STE forward value during training; not a ball point or token ID",
    "q_l": "exp0^{c_l}(e_l) in B_{c_l}",
    "q_l_reference": "exp0^{c0}(log0^{c_l}(q_l)) in B_{c0}",
    "aggregation": "h = q_0^0 ⊕_{c0} (q_1^0 ⊕_{c0} q_2^0)",
    "output": "z = log0^{c0}(h) in T_0(B_{c0})",
    "reference_curvature_alias": "c0 := c_0 = 1.3660953164241916",
    "decoder_input_replaces": "z_E = e_0 + e_1 + e_2",
    "mobius_addition_associative": false,
    "reassociation_or_reordering_allowed": false,
    "apply_log0_directly_to_e_l": false
  },
  "mathematical_scope": {
    "claim": "HRA-inspired common-reference Step6 aggregation only",
    "paper_exact_telescope_claim": false,
    "cross_curvature_mobius_homomorphism_claim": false,
    "residual_inversion_or_exact_reconstruction_claim": false,
    "unclipped_exp_log_identity_is_unconditional": false,
    "reason_limit": "Paper HRA cancellation uses paired residual subtraction and reverse-nested aggregation at one shared curvature; Iter29 has heterogeneous fixed c_l and cross-layer residual transport. Radial cross-curvature transfer is not a universal Mobius-addition homomorphism."
  },
  "helper_domain_contract": {
    "source_exp_curvature": "c_l",
    "source_log_curvature": "c_l",
    "target_exp_curvature": "c0",
    "both_mobius_curvatures": "c0",
    "final_log_curvature": "c0",
    "exp_output_projection": "existing helper projects to radius (1-eps)/sqrt(c), default eps=1e-6",
    "log_input_clamp": "existing helper clamps sqrt(c)*norm argument to at most 1-1e-5",
    "runtime_projection_or_clamp_incidence": "MUST_BE_MEASURED_AT_S08",
    "finite_and_valid_ball_values": "MUST_BE_CHECKED_AT_S07_S08",
    "runtime_domain_failure_policy": "BLOCK_MVG_AND_STAGE2_UNTIL_CLASSIFIED; apply existing feasibility-abort rules if registered mechanism is invalid/inactive and cannot be repaired without changing its locked semantics"
  },
  "preflight": {
    "fccr_preflight": "REQUIRED_UNCHANGED",
    "hra_preflight": "REQUIRED_ITERATION_LOCAL_CHECKER",
    "checker_path": "scripts/preflight_hra_step6_iter31.py",
    "checker_cli_arguments": "NONE",
    "checker_environment_overrides": false,
    "checker_must_check": [
      "exact_equation_and_layer_order_against_source",
      "source_and_target_curvature_argument_roles",
      "tangent_embedding_vs_ball_point_semantics",
      "right_nested_mobius_expression_without_reassociation",
      "decoder_input_shape_and_existing_consumer_path",
      "only_step6_aggregation_is_changed",
      "no_forbidden_new_component_or_stage3_change",
      "finiteness_and_ball_domain_checks_are_defined",
      "required_S08_runtime_effect_and_gradient_evidence_is_defined"
    ],
    "fccc_pass_alone_satisfies_hra": false,
    "combined_preflight_required": true
  },
  "runtime_and_result_gates": {
    "S08_mvg_required": true,
    "direct_effect_comparison": "same_checkpoint_and_batch: registered Step6 z versus the unchanged Euclidean Step6 sum; report actual value comparison and dtype-aware numerical distinguishability; no added scientific performance threshold",
    "gradient_health": "existing model/codebook/encoder intended paths and total loss finite/nonzero as applicable; fixed curvature itself is not expected to have gradients",
    "finiteness_and_shape": "all q_l, q_l^0, h, z and decoder-facing values finite and correctly shaped; ball points valid under the implementation's declared projection/clamp semantics",
    "clip_saturation": "record projection and log-clamp incidence; destructive saturation/inactivity triggers existing feasibility evaluation, not threshold retuning",
    "stage2_allowed_only_after": [
      "S04_CANONICAL_CONTRACTS_PRESENT",
      "S05_ONE_FACTOR_PASS",
      "S06_IMPLEMENTATION_ADJUDICATION_AND_APPLICATION",
      "S07_FCCR_AND_HRA_PREFLIGHT_PASS",
      "S08_MVG_PASS",
      "S09_EXECUTION_ADJUDICATION",
      "DELIBERATION_GATE_PASS",
      "ROOT_CLAUDE_MD_SECTION_6_GRADIENT_PATH_CHECK"
    ],
    "adoption_metric": "test_R@10",
    "adoption_comparator": "Iter29 exact protocol-locked comparator specified by S01",
    "adoption_target": "test_R@10 > 0.065",
    "adoption_target_is_mechanism_activation_gate": false,
    "stage2_proxy_metrics_are_gates": false
  },
  "provenance_and_reproducibility": {
    "curvature_inputs": "historical recorded Iter29 values with S03 replay/hash limits; no fallback or refreshed values",
    "runtime_embeddings_and_maps": "must be observed on actual registered checkpoint/batch during later authorized S08 work; no runtime evidence claimed at S04",
    "training_or_performance_evidence_present": false
  }
}
```

The JSON's explicit function/expression strings are intended to be source-checked by S07's dedicated checker. They do not claim a parsed mathematical AST or prove behavior merely by existing in JSON. Exact decimal values and layer ordering must be checked against the FCCR contract and the source's actual fixed-buffer path. The HRA checker must read and validate both contracts and the code under audit, and must fail closed if either contract is absent, malformed, contradictory, or unchecked.

## Relationship between the two contracts

- FCCR-1 continues to own curvature source, mapping inputs, fixed/nontrainable/time-invariant status, and the exact `[L0,L1,L2]` curvature vector. HRA Step6 consumes that substrate; it neither alters the `c_l` values nor introduces a curvature mechanism.
- HRA-STEP6-1 owns the one new decoder-facing aggregation operation. It records the exact source/target curvature roles and order, helper projection/clamp caveats, unchanged components, and no-telescope limits.
- HRA is not inserted as new formula inputs or mapping clauses in FCCR-1. Conversely, FCCR-1 cannot be used as evidence that HRA Step6 is implemented or checked.
- The active Iter31 research contract is the conjunction `FCCR-1 substrate AND HRA-STEP6-1 mechanism`; both must pass their own checks.
- Contract status is a S04 registration decision only. It is not source/runtime conformance, MVG pass, training approval, or a performance result.

## Preflight / checker route and authorization boundary

**Directly observed:** `.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py` loads the mandatory iteration JSON, requires `contract_version=FCCR-1`, checks fixed curvature fields, exact formula-input list, three finite positive values, and statically scans for trainable curvature, step-dependent `get_c()`, fixed buffers, and nonzero curvature regularization. It has no HRA Step6 logic. The Iter29 loader independently recomputes/validates its FCCR inputs and enforces the exact ten-field contract shape. `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py` requires the S04 `mechanism_contract_iter31.json`, but its purpose is 2+1 process/canonical checking; it does not validate an HRA contract or mechanism source.

**Proposed safe path:**

1. S04 Judge writes both canonical contracts and lists both in `CANONICAL_ARTIFACT`. No code/checker changes in S04.
2. S05 adjudicates the exact one-factor boundary; S06 independently plans source changes and, if justified, the new iteration-local checker. After S06 Judge selects the plan, the orchestrator applies it once. Do not edit the installed/shared skill checker or weaken its FCCR checks.
3. S07 A/B independently audit the implemented source and both contracts. If the checker was not included in the S06 approved implementation plan, S07 may propose a narrowly scoped iteration-local checker implementation, but it must be independently adjudicated by Judge C and materialized once only after that approval; then S07 checks and runs it. No execution may occur before both HRA and FCCR checks complete. Any S07 checker work must remain within Iter31 and not alter shared preflight/gate behavior.
4. From the Iter31 directory, run the existing no-argument skill command exactly as specified by the skill (the FCCR-specific pass remains necessary), and separately run the no-argument Iter31 HRA checker. Require `MECHANISM_CONTRACT_PASS` from the former and an explicit HRA-specific pass token from the latter. Record both commands/results and source identities in `logs/preflight_contract_iter31.log`; a single FCCR pass must never be relabeled as combined pass.
5. Run the existing `deliberation_gate.py` for independent 2+1 process completeness. Its pass supplements, but does not replace, either mechanism checker. S08/MVG then independently verifies runtime behavior; S09/Stage2 remain blocked until all their own approvals and root §6 gradient checks pass.

The checker should statically enforce equation/operator order and consumer wiring, not attempt to establish scientific benefit. Runtime requirements that static inspection cannot prove flow to S08: actual tangent embedding values (including STE forward), `q_l`, `q_l^0`, `h`, `z`; correct batch-by-embedding shape; finite values and ball-domain validity; measured exp projection/log clamp incidence; fixed `c_l` invariance; same-checkpoint/same-batch contrast of registered `z` with legacy Euclidean sum; and intended gradient health through the decoder-facing path and relevant model/codebook/encoder parameters. Fixed curvature must not be required to receive gradient.

**No invented scientific threshold:** preserve only the already locked strict adoption threshold `test_R@10 > 0.065`, and do not use it as a mechanism-activation or Stage2 gate. Do not invent a numeric minimum Step6 delta, clipping fraction, or saturation threshold at S04. S07/S08 must define a deterministic, dtype-aware method to report direct output distinguishability and runtime domain/clipping observations without pretending that such a numerical cutoff was preregistered. If the S02 condition “measurable nondegenerate direct effect” cannot be given a sound, non-arbitrary operational criterion using primary-source numerical semantics before MVG, Judge C must block progression (and adjudicate `REJECT_BOTH` or abort according to whether the registered experiment can be checked), rather than making up a threshold or claiming FCCR PASS is enough. Any confirmed inactivity requiring a mechanism change is subject to skill §2.10 abort; no in-iteration retuning.

## Evidence-backed rationale: observed facts vs proposals

### Direct facts from the frozen packet, canonical artifacts, and primary source reads

- **S02 and S03 status:** S02 is Judge-approved `MERGE_AB` with the exact equation and one conceptual Step6 change. It explicitly left FCCR-1 active and the contract transition for S04. S03 is a limited provenance pass and explicitly leaves the transition/checker compatibility to S04/S07. The S00–S03 Judge reports have been read; none silently performs the contract transition.
- **Iter30 boundary:** the packet and canonical S00 Judge report characterize Iter30 as user-cancelled with no scientific result and no Stage2/Stage3/GPU run; cancellation is not mechanism infeasibility. The skill permits autonomous, canonically adjudicated between-iteration contract transitions. Iter31 is the fresh registered experiment, not a continuation or relabeling of Iter30.
- **FCCR preflight:** direct code inspection of `preflight_contract.py` shows it accepts only FCCR-1 and checks fixed curvature fields/input names/value shape/positivity plus a source scan for curvature properties. No HRA Step6 logic exists. Therefore it cannot be the sole HRA checker.
- **Loader compatibility:** Iter29 `_load_closed_form_curvatures()` reads `logs/mechanism_contract_iter29.json`, requires the exact FCCR fields and rejects any field set other than those exact ten keys; it recomputes/compares fixed curvature against explicit `branching` and `raw_residual_medians`. A composite JSON with added HRA fields is incompatible with that loader contract.
- **Deliberation-gate scope:** the gate requires `mechanism_contract_iter<N>.json` for S04, but verifies stage evidence/process, not HRA semantics. An additional canonical contract can be named by Judge C, but the gate's presence check on the mandatory file cannot validate that additional contract.
- **Iter29 quantizer and geometry:** `Quantize.forward` returns selected codebook embeddings in tangent coordinates, with training output `x + (embeddings - x).detach()` and evaluation output the embedding. Fixed curvature uses persistent `_fixed_c` buffer and fixed `get_c()` behavior. The manifold helpers perform origin exp/log and Möbius operations; exp outputs are projected at the stated radius and log clamps its atanh argument.
- **Iter29 Step6:** source `_step6_sum_embeddings` sums returned embeddings in Euclidean tangent coordinates; `forward` sends that output to the existing decoder and computes reconstruction loss on the layer-0 curvature path. Step4 residual subtraction and Step5 inter-layer curvature transport are separate unchanged paths.
- **HRA mathematical limit:** packet and canonical S00/S02 evidence distinguish paper's common-curvature paired subtraction/aggregation telescope from the heterogeneous-curvature, transported Iter29 residual path. The recorded cross-curvature witness refutes a universal homomorphism claim, not all possible operation value or model-level activation. No exact telescope or residual inversion is established.
- **Target and scope:** active Stage3 adoption target is strict `test_R@10 > 0.065`; Stage2 proxy statistics are descriptive, not gates. S02 explicitly excludes d-HSTE, other losses/optimizer/quantizer/curvature changes, and Stage1/Stage3 changes. Stage3 route conflict is still unresolved from S01 and must be resolved by the later authorized stages, not by S04.

### Proposal/inference, not observed facts

- A two-contract conjunction is the least invasive compatible representation: retain the loader-compatible FCCR contract and separately represent the Step6 mechanism.
- A new per-iteration checker is necessary to make the HRA schema operationally checked. Its file name and checks above are a proposed design; no such checker currently exists in the inspected skill scripts, and no implementation has been made.
- Whether an actual Iter31 implementation will pass static or runtime checks is unknown until S06/S07/S08. No values for actual model `e_l`, projections, clamps, `h`, `z`, direct effect, gradients, or R@10 are claimed here.

## Preflight acceptance gates and downstream conditions

- **S04:** Judge C must either adopt both JSON contracts with correct paths/values and state `USER_INPUT_REQUIRED=NO`, or reject/abort if it cannot preserve loader compatibility and an independent HRA check. The canonical contract registration is not implementation or run authorization.
- **S05:** must prove Step6 aggregation is the only conceptual delta against the declared Iter29 parent; FCCR vector, residual update/transport, optimizer/losses, Stage1, decoder protocol, and Stage3 are unchanged. Any second mechanism blocks the run.
- **S06:** must specify source and tests/checker coverage without changing the scientific equation; implementation is applied once after Judge approval. Checker authoring cannot mutate shared skill checker.
- **S07:** require both unchanged FCCR checker pass and the separate HRA checker pass. Static HRA verification must cover exact formula arguments/tree/layer ordering, tangent/ball semantics, replacement point, unchanged decoder and scope, and no forbidden components. Missing HRA check or a false pass blocks S08/Stage2.
- **S08:** verify runtime validity, fixed-curvature immutability, actual Step6 direct effect versus the Euclidean sum on same checkpoint/batch, clipping/saturation observations, shapes/finiteness, and gradients. No fixed-curvature gradient requirement. Inactivity/invalidity that would require changing registered equation or constants means abort under the skill, not retuning.
- **S09/Stage2:** only after all prior contracts/checks, deliberation, and root CLAUDE §6 gradient-path checks. Stage2 proxy metrics are never a performance gate; all valid, non-aborted runs proceed to unchanged Stage3.
- **S11/Stage3:** keep the exact S01-locked Stage3 protocol and resolve its declared route/path conflict through its own adjudication. Do not make any Stage3 model/evaluation change.
- **S12/result:** classify mechanism activation separately from promotion. Compare the exact S01 Iter29 result and strict `>0.065` adoption criterion. Neither an R@10 shortfall nor paper HRA-only ablation is itself proof of inactive mechanism.

## Hard gaps, self-rejection, and abort criteria

1. **Reject/abort if HRA cannot have an independent machine check.** If the only available route is to add HRA data into the ten-field FCCR JSON, bypass the FCCR preflight, modify it silently, or claim its FCCR pass proves HRA, the design is invalid. My proposed separate-contract/checker path avoids this, but it is only viable if later canonical stages actually implement/check it.
2. **Block if source cannot be made consistent with exact registered equation and one-factor scope.** If the actual code cannot preserve the precise L0/L1/L2 right-nested form, c-l source and c0 target roles, tangent/ball distinctions, and same decoder consumer without changing other behavior, no contract fit should be fabricated.
3. **No arbitrary activation criterion.** S02 requires a measurable nondegenerate direct effect but does not establish a numeric cutoff. Do not fabricate one. Require a judged, reproducible, dtype-aware S08 observation protocol; if no defensible operational check can be specified, do not claim MVG pass or proceed to Stage2. If direct evidence then shows inactivity that needs a spec change, abort Iter31 under the feasibility rule.
4. **Do not resolve possible failures by scientific retuning.** Do not alter curvature values, HRA equation, parent, seed, Stage3, threshold, or add d-HSTE/other components. Implementation repairs are allowed only when they restore exact registered semantics without changing the experiment.
5. **S03 historical provenance remains limited.** Do not imply raw input replay, independent measurement, current-byte hash verification, or fresh measurement. Failure to preserve explicit saved values/keys is a contract/provenance failure.
6. **No exact-telescope claim.** Any future claim that the operation exactly reconstructs the residual cascade, or that radial transfer is a gyrogroup homomorphism, exceeds available evidence and self-rejects the contract interpretation.
7. **No authorization inflation.** This candidate recommends registration only. It does not authorize checker edits, implementation, training, Stage2, Stage3, GPU work, performance interpretation, or bypass of any S05–S11 gate.

If the checker/interface route cannot be implemented and audited in a later authorized stage while leaving the shared FCCR checker untouched, Judge C should reject the contract proposal; if that demonstrates the registered iteration itself cannot be checked or continued without changing locked mechanism/protocol, autonomously issue `ABORT_ITERATION` with the required evidence rather than ask for input. `USER_INPUT_REQUIRED=NO`.